"""
Chain LLM pour classification et extraction de documents KYC.

Supporte deux backends:
- Vertex AI (google-cloud-aiplatform) pour les modèles Gemini 2.x
- LiteLLM proxy (via langchain-openai) pour les modèles récents (Gemini 3.x)
"""

import base64
import json
import time
from pathlib import Path

from pydantic import ValidationError

from chains.configuration import Configuration
from chains.prompts import (
    PROMPT_CLASSIFICATION,
    PROMPT_EXTRACTION_CNI,
    PROMPT_EXTRACTION_FORMULAIRE,
    PROMPT_EXTRACTION_JUSTIFICATIF,
    PROMPT_EXTRACTION_PASSEPORT,
    PROMPT_EXTRACTION_PERMIS,
    PROMPT_EXTRACTION_RIB,
)
from chains.schemas import (
    RIB,
    CarteIdentite,
    ClassificationDocument,
    FormulaireInscription,
    JustificatifDomicile,
    Passeport,
    PermisConduire,
    ResultatExtractionKYC,
    TypeDocument,
)


class KYCDocumentChain:
    """
    Chain pour classification et extraction de documents KYC.

    Pipeline:
    1. Classification du type de document
    2. Extraction structurée selon le schéma correspondant
    3. Validation des règles métier
    """

    def __init__(self, config: Configuration | None = None):
        """
        Initialise la chain.

        Args:
            config: Configuration (si None, charge depuis config.json)
        """
        self.config = config or Configuration()

        if self.config.use_litellm:
            self._init_litellm()
        else:
            self._init_vertex()

    def _init_vertex(self):
        """Initialise le backend Vertex AI (modèles 2.x)."""
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        self._backend = "vertex"
        vertexai.init(project=self.config.project_id, location=self.config.location)

        self._vertex_model_rad = GenerativeModel(self.config.model_rad)
        self._vertex_model_lad = GenerativeModel(self.config.model_lad)
        self._vertex_gen_config = GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            response_mime_type="application/json",
        )
        print(f"🧠 Vertex AI — RAD: {self.config.model_rad} | LAD: {self.config.model_lad}")

    def _init_litellm(self):
        """Initialise le backend LiteLLM proxy (modèles 3.x)."""
        import httpx
        from langchain_openai import ChatOpenAI

        self._backend = "litellm"

        http_kwargs = {}
        if self.config.litellm_cert_path:
            http_kwargs["http_client"] = httpx.Client(verify=self.config.litellm_cert_path)
            http_kwargs["http_async_client"] = httpx.AsyncClient(verify=self.config.litellm_cert_path)

        self._litellm_rad = ChatOpenAI(
            openai_api_base=self.config.litellm_api_base,
            openai_api_key=self.config.litellm_api_key,
            model=self.config.litellm_model_rad,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            **http_kwargs,
        )
        self._litellm_lad = ChatOpenAI(
            openai_api_base=self.config.litellm_api_base,
            openai_api_key=self.config.litellm_api_key,
            model=self.config.litellm_model_lad,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            **http_kwargs,
        )
        print(
            f"🧠 LiteLLM — RAD: {self.config.litellm_model_rad} | LAD: {self.config.litellm_model_lad}"
        )

    # -------------------------------------------------------------------------
    # Méthodes utilitaires
    # -------------------------------------------------------------------------

    def _load_image_bytes(self, image_path: str | Path) -> tuple[bytes, str]:
        """Charge une image et retourne (bytes, mime_type)."""
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".pdf": "application/pdf",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        return image_bytes, mime_type

    def _generate(self, role: str, prompt: str, image_path: str | Path) -> tuple[str, dict | None]:
        """
        Appel unifié: envoie un prompt + image au modèle approprié.

        Args:
            role: "rad" ou "lad"
            prompt: Le prompt texte
            image_path: Chemin vers l'image

        Returns:
            Tuple (texte réponse, token_usage dict ou None)
        """
        image_bytes, mime_type = self._load_image_bytes(image_path)

        if self._backend == "litellm":
            return self._generate_litellm(role, prompt, image_bytes, mime_type)
        else:
            return self._generate_vertex(role, prompt, image_bytes, mime_type)

    def _generate_vertex(
        self, role: str, prompt: str, image_bytes: bytes, mime_type: str
    ) -> tuple[str, dict | None]:
        """Génère via Vertex AI."""
        from vertexai.generative_models import Part

        image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
        model = self._vertex_model_rad if role == "rad" else self._vertex_model_lad

        response = model.generate_content(
            [prompt, image_part],
            generation_config=self._vertex_gen_config,
        )

        token_usage = self._extract_vertex_tokens(response)
        return response.text, token_usage

    def _generate_litellm(
        self, role: str, prompt: str, image_bytes: bytes, mime_type: str
    ) -> tuple[str, dict | None]:
        """Génère via LiteLLM proxy (OpenAI-compatible)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            SystemMessage(content="Réponds UNIQUEMENT en JSON structuré."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                ]
            ),
        ]

        llm = self._litellm_rad if role == "rad" else self._litellm_lad
        response = llm.invoke(messages)

        token_usage = self._extract_litellm_tokens(response)
        return response.content, token_usage

    # -------------------------------------------------------------------------
    # Token usage
    # -------------------------------------------------------------------------

    def _extract_vertex_tokens(self, response) -> dict[str, int] | None:
        """Extrait les tokens depuis une réponse Vertex AI."""
        try:
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                input_tok = usage.prompt_token_count if hasattr(usage, "prompt_token_count") else 0
                output_tok = (
                    usage.candidates_token_count if hasattr(usage, "candidates_token_count") else 0
                )
                total_tok = (
                    usage.total_token_count
                    if hasattr(usage, "total_token_count")
                    else input_tok + output_tok
                )
                calculated_total = input_tok + output_tok
                overhead = total_tok - calculated_total if total_tok > calculated_total else 0
                return {
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "total_tokens": total_tok,
                    "overhead_tokens": overhead,
                }
        except Exception as e:
            print(f"   ⚠️  Impossible d'extraire les statistiques de tokens: {e}")
        return None

    def _extract_litellm_tokens(self, response) -> dict[str, int] | None:
        """Extrait les tokens depuis une réponse LangChain/LiteLLM."""
        try:
            usage = response.usage_metadata
            if usage:
                input_tok = usage.get("input_tokens", 0)
                output_tok = usage.get("output_tokens", 0)
                total_tok = usage.get("total_tokens", input_tok + output_tok)
                calculated_total = input_tok + output_tok
                overhead = total_tok - calculated_total if total_tok > calculated_total else 0
                return {
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "total_tokens": total_tok,
                    "overhead_tokens": overhead,
                }
        except Exception as e:
            print(f"   ⚠️  Impossible d'extraire les statistiques de tokens: {e}")
        return None

    def _log_token_usage(self, document_type: str, token_usage: dict):
        """Log les statistiques de tokens pour un document."""
        if not token_usage:
            return

        input_tok = token_usage.get("input_tokens", 0)
        output_tok = token_usage.get("output_tokens", 0)
        total_tok = token_usage.get("total_tokens", 0)
        overhead_tok = token_usage.get("overhead_tokens", 0)

        input_cost = (input_tok / 1_000_000) * self.config.INPUT_TOKEN_PRICE_PER_MILLION
        output_cost = (output_tok / 1_000_000) * self.config.OUTPUT_TOKEN_PRICE_PER_MILLION
        total_cost = input_cost + output_cost

        if overhead_tok > 0:
            print(
                f"💰 Tokens {document_type}: "
                f"input={input_tok}, output={output_tok}, total={total_tok} "
                f"(overhead: {overhead_tok}) | Coût: ${total_cost:.6f}"
            )
        else:
            print(
                f"💰 Tokens {document_type}: "
                f"input={input_tok}, output={output_tok}, total={total_tok} | Coût: ${total_cost:.6f}"
            )

    # -------------------------------------------------------------------------
    # Classification et extraction
    # -------------------------------------------------------------------------

    def _parse_json(self, text: str) -> dict:
        """Parse la réponse JSON, gère les listes et les blocs markdown."""
        # Nettoyer les blocs ```json ... ```
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Retirer la première ligne (```json) et la dernière (```)
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        result = json.loads(cleaned)
        if isinstance(result, list):
            result = result[0]
        return result

    def classify_document(
        self, image_path: str | Path
    ) -> tuple[ClassificationDocument, dict | None]:
        """Classifie le type de document."""
        text, token_usage = self._generate("rad", PROMPT_CLASSIFICATION, image_path)

        if token_usage:
            self._log_token_usage("Classification", token_usage)

        result_json = self._parse_json(text)
        return ClassificationDocument(**result_json), token_usage

    def extract_cni(self, image_path: str | Path) -> tuple[CarteIdentite, dict | None]:
        """Extrait les données d'une Carte Nationale d'Identité."""
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_CNI, image_path)

        if token_usage:
            self._log_token_usage("Extraction CNI", token_usage)

        result_json = self._parse_json(text)
        return CarteIdentite(**result_json), token_usage

    def extract_passeport(self, image_path: str | Path) -> tuple[Passeport, dict | None]:
        """Extrait les données d'un passeport."""
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_PASSEPORT, image_path)

        if token_usage:
            self._log_token_usage("Extraction Passeport", token_usage)

        result_json = self._parse_json(text)
        return Passeport(**result_json), token_usage

    def extract_permis(self, image_path: str | Path) -> tuple[PermisConduire, dict | None]:
        """Extrait les données d'un permis de conduire."""
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_PERMIS, image_path)

        if token_usage:
            self._log_token_usage("Extraction Permis", token_usage)

        result_json = self._parse_json(text)
        return PermisConduire(**result_json), token_usage

    def extract_justificatif(
        self, image_path: str | Path
    ) -> tuple[JustificatifDomicile, dict | None]:
        """Extrait les données d'un justificatif de domicile."""
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_JUSTIFICATIF, image_path)

        if token_usage:
            self._log_token_usage("Extraction Justificatif", token_usage)

        result_json = self._parse_json(text)
        return JustificatifDomicile(**result_json), token_usage

    def extract_rib(self, image_path: str | Path) -> tuple[RIB, dict | None]:
        """Extrait les données d'un RIB."""
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_RIB, image_path)

        if token_usage:
            self._log_token_usage("Extraction RIB", token_usage)

        result_json = self._parse_json(text)
        return RIB(**result_json), token_usage

    def extract_formulaire(
        self, image_path: str | Path
    ) -> tuple[FormulaireInscription, dict | None]:
        """
        Extrait les données d'un formulaire d'inscription.

        C'est LE CLIMAX DE LA DÉMO: ~120 cases à cocher détectées visuellement!
        """
        text, token_usage = self._generate("lad", PROMPT_EXTRACTION_FORMULAIRE, image_path)

        if token_usage:
            self._log_token_usage("Extraction Formulaire", token_usage)

        result_json = self._parse_json(text)

        print("\n✅ EXTRACTION RÉUSSIE")

        # Dump JSON brut pour debug
        debug_path = Path("debug_last_response.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"\n📝 JSON brut sauvegardé dans: {debug_path}")
        print("\n📋 Réponse JSON brute du modèle:\n")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))

        try:
            return FormulaireInscription(**result_json), token_usage
        except ValidationError as e:
            e.token_usage = token_usage  # type: ignore[attr-defined]
            raise

    # -------------------------------------------------------------------------
    # Pipeline principal
    # -------------------------------------------------------------------------

    def process_document(self, image_path: str | Path) -> ResultatExtractionKYC:
        """
        Pipeline complet: classification + extraction + validation.

        C'est LA MÉTHODE PRINCIPALE du démonstrateur!
        """
        erreurs = []
        avertissements = []
        classification = None
        total_tokens_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "overhead_tokens": 0,
        }

        try:
            # 1. Classification (RAD - Reconnaissance Automatique de Documents)
            print(f"\n🔍 Classification du document (RAD) : {image_path}")
            t0_rad = time.time()
            classification, token_usage_classification = self.classify_document(image_path)
            latence_rad = time.time() - t0_rad
            confiance_str = (
                f" (confiance: {classification.confiance:.2%})"
                if classification.confiance is not None
                else ""
            )
            print(f"✅ CLASSIFICATION RÉUSSIE : {classification.type_detecte.value}{confiance_str}")

            if token_usage_classification:
                for key in total_tokens_usage:
                    total_tokens_usage[key] += token_usage_classification.get(key, 0)
                rad_cost = (
                    token_usage_classification.get("input_tokens", 0) / 1_000_000
                ) * self.config.INPUT_TOKEN_PRICE_PER_MILLION + (
                    token_usage_classification.get("output_tokens", 0) / 1_000_000
                ) * self.config.OUTPUT_TOKEN_PRICE_PER_MILLION
                print(
                    f"📊 Coût RAD: ${rad_cost:.6f} | Latence: {latence_rad / 2:.1f}s"
                )

            # 2. Extraction selon le type (LAD - Lecture Automatique de Documents)
            print(f"\n📄 Extraction des données (LAD) :")
            extraction_result = None
            token_usage_extraction = None

            extractors = {
                TypeDocument.CARTE_IDENTITE: self.extract_cni,
                TypeDocument.PASSEPORT: self.extract_passeport,
                TypeDocument.PERMIS_CONDUIRE: self.extract_permis,
                TypeDocument.JUSTIFICATIF_DOMICILE: self.extract_justificatif,
                TypeDocument.RIB: self.extract_rib,
                TypeDocument.FORMULAIRE_INSCRIPTION: self.extract_formulaire,
            }

            extractor = extractors.get(classification.type_detecte)
            t0_lad = time.time()
            if extractor:
                extraction_result, token_usage_extraction = extractor(image_path)
            latence_lad = time.time() - t0_lad

            if token_usage_extraction:
                for key in total_tokens_usage:
                    total_tokens_usage[key] += token_usage_extraction.get(key, 0)
                lad_cost = (
                    token_usage_extraction.get("input_tokens", 0) / 1_000_000
                ) * self.config.INPUT_TOKEN_PRICE_PER_MILLION + (
                    token_usage_extraction.get("output_tokens", 0) / 1_000_000
                ) * self.config.OUTPUT_TOKEN_PRICE_PER_MILLION
                print(
                    f"📊 Coût LAD: ${lad_cost:.6f} | Latence: {latence_lad / 2:.1f}s"
                )

            if classification.type_detecte != TypeDocument.FORMULAIRE_INSCRIPTION:
                print("\n✅ EXTRACTION RÉUSSIE")

            latence_totale = latence_rad + latence_lad
            if total_tokens_usage["total_tokens"] > 0:
                total_cost = (
                    total_tokens_usage["input_tokens"] / 1_000_000
                ) * self.config.INPUT_TOKEN_PRICE_PER_MILLION + (
                    total_tokens_usage["output_tokens"] / 1_000_000
                ) * self.config.OUTPUT_TOKEN_PRICE_PER_MILLION
                print(
                    f"\n📊 Coût total (RAD + LAD): ${total_cost:.6f} | Latence totale: {latence_totale / 2:.1f}s"
                )

            # 3. Construction du résultat
            result = ResultatExtractionKYC(
                classification=classification,
                extraction_reussie=True,
                regles_metier_validees=True,
                erreurs=erreurs,
                avertissements=avertissements,
            )

            # Assigner l'extraction au bon champ
            field_map = {
                TypeDocument.CARTE_IDENTITE: "carte_identite",
                TypeDocument.PASSEPORT: "passeport",
                TypeDocument.PERMIS_CONDUIRE: "permis_conduire",
                TypeDocument.JUSTIFICATIF_DOMICILE: "justificatif_domicile",
                TypeDocument.RIB: "rib",
                TypeDocument.FORMULAIRE_INSCRIPTION: "formulaire_inscription",
            }
            field = field_map.get(classification.type_detecte)
            if field:
                setattr(result, field, extraction_result)

            return result

        except ValidationError as e:
            latence_lad = time.time() - t0_lad
            token_usage_from_error = getattr(e, "token_usage", None)
            if token_usage_from_error:
                for key in total_tokens_usage:
                    total_tokens_usage[key] += token_usage_from_error.get(key, 0)

            latence_totale = latence_rad + latence_lad
            if total_tokens_usage["total_tokens"] > 0:
                total_cost = (
                    total_tokens_usage["input_tokens"] / 1_000_000
                ) * self.config.INPUT_TOKEN_PRICE_PER_MILLION + (
                    total_tokens_usage["output_tokens"] / 1_000_000
                ) * self.config.OUTPUT_TOKEN_PRICE_PER_MILLION
                print(
                    f"\n📊 Coût total (RAD + LAD): ${total_cost:.6f} | Latence totale: {latence_totale / 2:.1f}s"
                )

            print("\n" + "=" * 70)
            print("⚠️  RÈGLE MÉTIER NON RESPECTÉE")
            print("=" * 70)
            for err in e.errors():
                field = " → ".join(str(loc) for loc in err["loc"])
                model_name = e.title
                # Extraire le message sans le préfixe "Value error, "
                msg = err["msg"]
                if msg.startswith("Value error, "):
                    msg = msg[len("Value error, "):]
                input_data = err.get("input", {})
                print(f"\n📍 Champ : {model_name}.{field}")
                print(f"💬 Règle : {msg}")
                if isinstance(input_data, dict):
                    print("📦 Valeurs reçues :")
                    for k, v in input_data.items():
                        print(f"   {k}: {v}")
                print()
            print("=" * 70)
            erreurs.append(str(e))
            return ResultatExtractionKYC(
                classification=classification,
                extraction_reussie=True,
                regles_metier_validees=False,
                erreurs=erreurs,
                avertissements=avertissements,
            )

        except Exception as e:
            error_msg = str(e)
            if "<html>" in error_msg.lower():
                error_msg = "Service temporairement indisponible (502 Bad Gateway)"
            print(f"\n❌ Erreur lors du traitement: {error_msg}\n")
            erreurs.append(str(e))
            return ResultatExtractionKYC(
                classification=classification,
                extraction_reussie=False,
                regles_metier_validees=False,
                erreurs=erreurs,
                avertissements=avertissements,
            )
