"""
Configuration pour les chains LLM.

Basé sur Vertex AI (Google Cloud).
"""

import json
import os
from pathlib import Path
from typing import Any


class Configuration:
    """Classe de configuration pour les chains LLM."""

    def __init__(self, config_path: str | None = None):
        """
        Initialise la configuration.

        Args:
            config_path: Chemin vers le fichier config.json
        """
        if config_path is None:
            # Par défaut: config/config.json à la racine du projet
            config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
        else:
            config_path = Path(config_path)

        with open(config_path) as f:
            self._config = json.load(f)

    @property
    def project_id(self) -> str:
        """ID du projet Google Cloud."""
        value = os.getenv("VAR_LLM_PROJECT_ID", os.getenv("GCP_PROJECT_ID", ""))
        if not value:
            raise ValueError(
                "Variable d'environnement manquante : VAR_LLM_PROJECT_ID ou GCP_PROJECT_ID"
            )
        return value

    @property
    def location(self) -> str:
        """Région Google Cloud."""
        value = os.getenv("VAR_LLM_REGION", os.getenv("GCP_LOCATION", ""))
        if not value:
            raise ValueError("Variable d'environnement manquante : VAR_LLM_REGION ou GCP_LOCATION")
        return value

    @property
    def model_rad(self) -> str:
        """Modèle Vertex AI pour la classification (RAD)."""
        return os.getenv(
            "VAR_LLM_MODELE_RAD",
            os.getenv("MODEL_NAME_RAD", "gemini-2.5-flash"),
        )

    @property
    def model_lad(self) -> str:
        """Modèle Vertex AI pour l'extraction (LAD)."""
        return os.getenv(
            "VAR_LLM_MODELE_LAD",
            os.getenv("MODEL_NAME_LAD", "gemini-2.5-pro"),
        )

    @property
    def temperature(self) -> float:
        """Température pour la génération."""
        return float(os.getenv("VAR_LLM_TEMPERATURE", os.getenv("TEMPERATURE", "0.2")))

    @property
    def max_output_tokens(self) -> int:
        """Nombre maximum de tokens en sortie."""
        return int(os.getenv("VAR_LLM_MAX_OUTPUT_TOKEN", "4096"))

    # Token pricing (USD per 1M tokens)
    # RAD: gemini-3.1-flash-lite
    RAD_INPUT_TOKEN_PRICE_PER_MILLION: float = 0.30
    RAD_OUTPUT_TOKEN_PRICE_PER_MILLION: float = 2.50
    # LAD: gemini-3.1-pro-preview (≤200K context)
    LAD_INPUT_TOKEN_PRICE_PER_MILLION: float = 0.30
    LAD_OUTPUT_TOKEN_PRICE_PER_MILLION: float = 2.50

    # --- LiteLLM (proxy pour modèles récents) ---

    @property
    def litellm_api_base(self) -> str | None:
        """URL du proxy LiteLLM. Si défini, active le mode LiteLLM."""
        return os.getenv("LITELLM_API_BASE")

    @property
    def litellm_api_key(self) -> str:
        """Clé API LiteLLM."""
        return os.getenv("LITELLM_API_KEY", "")

    @property
    def litellm_model_rad(self) -> str:
        """Modèle LiteLLM pour la classification (RAD)."""
        return os.getenv("LITELLM_MODEL_RAD", "gemini-3.1-flash-preview")

    @property
    def litellm_model_lad(self) -> str:
        """Modèle LiteLLM pour l'extraction (LAD)."""
        return os.getenv("LITELLM_MODEL_LAD", "gemini-3.1-pro-preview")

    @property
    def litellm_cert_path(self) -> str | None:
        """Chemin vers le certificat TLS pour LiteLLM."""
        path = os.getenv("LITELLM_CERT_PATH", "")
        return path if path else None

    @property
    def use_litellm(self) -> bool:
        """True si LiteLLM est configuré avec une URL et une clé valide."""
        return bool(self.litellm_api_base and self.litellm_api_key and self.litellm_api_key != "CHANGE_ME")

    @property
    def show_raw_rad(self) -> bool:
        return os.getenv("VAR_LLM_SHOW_RAW_RAD", "false").lower() == "true"

    @property
    def show_raw_lad(self) -> bool:
        return os.getenv("VAR_LLM_SHOW_RAW_LAD", "false").lower() == "true"

    @property
    def show_cost(self) -> bool:
        return os.getenv("VAR_LLM_SHOW_COST", "true").lower() == "true"

    @property
    def show_latency(self) -> bool:
        return os.getenv("VAR_LLM_SHOW_LATENCY", "true").lower() == "true"

    @property
    def show_retry_warnings(self) -> bool:
        """Affiche les tentatives de retry sur erreurs transitoires (à couper pour une démo)."""
        return os.getenv("VAR_LLM_SHOW_RETRY_WARNINGS", "true").lower() == "true"

    @property
    def document_types(self) -> list[str]:
        """Types de documents supportés."""
        return self._config["document_types"]

    @property
    def business_rules(self) -> dict[str, Any]:
        """Règles métier."""
        return self._config["business_rules"]

    def get_rule(self, rule_name: str) -> Any:
        """
        Récupère une règle métier spécifique.

        Args:
            rule_name: Nom de la règle

        Returns:
            Valeur de la règle
        """
        return self.business_rules.get(rule_name)
