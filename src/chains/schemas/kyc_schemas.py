"""Schémas KYC pour l'extraction de documents d'identité, justificatifs et IBAN."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TypeDocument(str, Enum):
    """Types de documents d'identité acceptés."""

    CARTE_IDENTITE = "carte_identite"
    PASSEPORT = "passeport"
    PERMIS_CONDUIRE = "permis_conduire"
    JUSTIFICATIF_DOMICILE = "justificatif_domicile"
    RIB = "rib"


class Sexe(str, Enum):
    """Genre."""

    M = "M"
    F = "F"
    X = "X"


class CarteIdentite(BaseModel):
    """Schéma pour l'extraction de documents d'identité."""

    type_document: Optional[TypeDocument] = Field(
        None,
        description="Type de document : passeport, carte d'identité nationale, ou permis de conduire",
    )

    numero_document: str = Field(description="Numéro du document d'identité")

    nom: str = Field(description="Nom de famille (en majuscules)")

    prenom: str = Field(description="Prénom(s)")

    date_naissance: date = Field(description="Date de naissance au format YYYY-MM-DD")

    lieu_naissance: Optional[str] = Field(None, description="Lieu de naissance (ville et pays)")

    nationalite: str = Field(description="Nationalité")

    sexe: Optional[Sexe] = Field(None, description="Genre : M, F, ou X")

    date_emission: date = Field(description="Date de délivrance du document au format YYYY-MM-DD")

    date_expiration: date = Field(description="Date d'expiration du document au format YYYY-MM-DD")

    autorite_emission: Optional[str] = Field(None, description="Autorité émettrice du document")

    adresse: Optional[str] = Field(
        None, description="Adresse complète (si présente sur le document)"
    )

    mrz_ligne1: Optional[str] = Field(
        None,
        description="Première ligne de la zone de lecture automatique (MRZ) si présente",
    )

    mrz_ligne2: Optional[str] = Field(
        None,
        description="Deuxième ligne de la zone de lecture automatique (MRZ) si présente",
    )

    @field_validator("nom")
    @classmethod
    def uppercase_nom(cls, v: str) -> str:
        """Nom de famille en majuscules."""
        return v.upper()

    @property
    def est_valide(self) -> bool:
        """Vérifie si le document est encore valide."""
        return self.date_expiration >= date.today()


class TypeJustificatifDomicile(str, Enum):
    """Types de justificatifs de domicile acceptés."""

    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    TAX_NOTICE = "tax_notice"
    RENTAL_AGREEMENT = "rental_agreement"
    RESIDENCE_CERTIFICATE = "residence_certificate"


class JustificatifDomicile(BaseModel):
    """Schéma pour l'extraction de justificatifs de domicile."""

    type_document: TypeJustificatifDomicile = Field(
        description="Type de justificatif : facture, relevé bancaire, avis d'imposition, etc."
    )

    date_document: date = Field(description="Date du document au format YYYY-MM-DD")

    nom_complet: str = Field(description="Nom complet du titulaire (prénom et nom)")

    adresse_ligne1: str = Field(description="Première ligne de l'adresse (numéro et rue)")

    adresse_ligne2: Optional[str] = Field(
        None, description="Complément d'adresse (bâtiment, appartement, etc.)"
    )

    code_postal: str = Field(description="Code postal")

    ville: str = Field(description="Ville")

    pays: str = Field("France", description="Pays")

    emetteur: Optional[str] = Field(None, description="Émetteur du document (EDF, banque, etc.)")

    est_recent: bool = Field(False, description="Le document a-t-il moins de 3 mois ?")

    @model_validator(mode="after")
    def check_recency(self):
        """Vérifie si le document a moins de 3 mois."""
        from datetime import timedelta

        three_months_ago = date.today() - timedelta(days=90)
        self.est_recent = self.date_document >= three_months_ago
        return self


class RIB(BaseModel):
    """Schéma pour l'extraction de RIB/IBAN."""

    nom_titulaire: str = Field(description="Nom du titulaire du compte")

    iban: str = Field(None, description="Numéro IBAN complet (format international)")

    bic: str = Field(None, description="Code BIC/SWIFT de la banque")

    nom_banque: Optional[str] = Field(None, description="Nom de la banque")

    adresse_banque: Optional[str] = Field(None, description="Adresse de l'agence bancaire")

    numero_compte: Optional[str] = Field(
        None, description="Numéro de compte national (si différent de l'IBAN)"
    )

    code_guichet: Optional[str] = Field(None, description="Code guichet ou code banque")

    iban_valide: bool = Field(False, description="L'IBAN respecte-t-il le format et le checksum ?")

    @field_validator("iban")
    @classmethod
    def clean_iban(cls, v: Optional[str]) -> Optional[str]:
        """Nettoie l'IBAN en supprimant les espaces."""
        if v is None:
            return None
        return v.replace(" ", "").upper()

    @model_validator(mode="after")
    def validate_iban_checksum(self):
        """Valide le checksum de l'IBAN."""
        if not self.iban:
            self.iban_valide = False
            return self

        iban = self.iban.replace(" ", "")
        if len(iban) < 15:
            self.iban_valide = False
            return self

        rearranged = iban[4:] + iban[:4]
        numeric = ""
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord("A") + 10)

        self.iban_valide = int(numeric) % 97 == 1
        return self


class DossierKYC(BaseModel):
    """Dossier KYC complet avec cohérence entre documents."""

    document_identite: CarteIdentite | Passeport = Field(
        description="Document d'identité du client (CNI ou Passeport)"
    )

    justificatif_domicile: JustificatifDomicile = Field(description="Justificatif de domicile")

    rib: RIB = Field(description="RIB/IBAN du client")

    permis_conduire: Optional["PermisConduire"] = Field(
        None, description="Permis de conduire (optionnel)"
    )

    noms_concordent: bool = Field(
        False, description="Les noms correspondent-ils entre les documents ?"
    )

    adresses_concordent: bool = Field(
        False,
        description="Les adresses correspondent-elles entre pièce d'identité et justificatif ?",
    )

    tous_documents_valides: bool = Field(
        False, description="Tous les documents sont-ils valides et récents ?"
    )

    statut_kyc: str = Field(
        "PENDING", description="Statut global du KYC : APPROVED, REJECTED, ou PENDING"
    )

    raisons_rejet: list[str] = Field(
        default_factory=list, description="Liste des raisons de rejet si applicable"
    )

    dossier_complet: bool = Field(
        False, description="Tous les documents requis sont-ils présents ?"
    )

    coherence_identite: bool = Field(
        False, description="Les noms sont-ils cohérents entre les documents ?"
    )

    erreurs_validation: list[str] = Field(
        default_factory=list, description="Erreurs de validation détectées"
    )

    def valider_coherence(self) -> bool:
        """
        Valide la cohérence entre les documents du dossier.

        Vérifie:
        - Cohérence des noms entre les documents
        - Récence du justificatif de domicile
        - Validité du checksum IBAN
        - Validité de la pièce d'identité

        Returns:
            True si le dossier est valide, False sinon
        """
        erreurs = []

        # Vérifier la cohérence des noms (chaque mot du nom d'identité doit apparaître dans le justificatif)
        nom_identite = self.document_identite.nom.upper()
        nom_justif = self.justificatif_domicile.nom_complet.upper()
        mots_identite = nom_identite.split()
        if mots_identite and all(mot in nom_justif.split() for mot in mots_identite):
            self.noms_concordent = True
        else:
            self.noms_concordent = False
            erreurs.append(
                f"Incohérence de nom: '{nom_identite}' (identité) vs '{nom_justif}' (justificatif)"
            )

        # Vérifier la récence du justificatif
        if not self.justificatif_domicile.est_recent:
            erreurs.append("Justificatif de domicile trop ancien (> 3 mois)")

        # Vérifier la validité de l'IBAN
        if not self.rib.iban_valide:
            erreurs.append(f"Checksum IBAN invalide: {self.rib.iban}")

        # Vérifier la validité de la pièce d'identité
        if not self.document_identite.est_valide:
            erreurs.append("Pièce d'identité expirée")

        self.coherence_identite = self.noms_concordent
        self.dossier_complet = True
        self.erreurs_validation = erreurs
        self.raisons_rejet = erreurs
        self.tous_documents_valides = len(erreurs) == 0

        if len(erreurs) == 0:
            self.statut_kyc = "APPROVED"
            return True
        else:
            self.statut_kyc = "REJECTED"
            return False


class Passeport(BaseModel):
    """Schéma pour l'extraction de passeports."""

    type_document: Optional[TypeDocument] = Field(
        None,
        description="Type de document : passeport",
    )

    numero_passeport: str = Field(description="Numéro du passeport")
    nom: str = Field(description="Nom de famille (en majuscules)")
    prenom: str = Field(description="Prénom(s)")
    date_naissance: date = Field(description="Date de naissance au format YYYY-MM-DD")
    lieu_naissance: Optional[str] = Field(None, description="Lieu de naissance (ville et pays)")
    nationalite: str = Field(description="Nationalité")
    sexe: Optional[Sexe] = Field(None, description="Genre : M, F, ou X")
    statut_marital: Optional[str] = Field(
        None,
        description="Statut marital (ex: célibataire, marié, divorcé), si présent sur le passeport",
    )
    date_emission: date = Field(description="Date de délivrance du document au format YYYY-MM-DD")
    date_expiration: date = Field(description="Date d'expiration du document au format YYYY-MM-DD")
    autorite_emission: Optional[str] = Field(None, description="Autorité émettrice du document")
    lieu_delivrance: Optional[str] = Field(
        None, description="Lieu de délivrance du passeport, si différent de l'autorité"
    )
    adresse: Optional[str] = Field(
        None, description="Adresse complète du titulaire, si présente sur le passeport"
    )
    mrz_ligne1: Optional[str] = Field(
        None,
        description="Première ligne de la zone de lecture automatique (MRZ) si présente",
    )
    mrz_ligne2: Optional[str] = Field(
        None,
        description="Deuxième ligne de la zone de lecture automatique (MRZ) si présente",
    )

    @field_validator("nom")
    @classmethod
    def uppercase_nom(cls, v: str) -> str:
        """Nom de famille en majuscules."""
        return v.upper()

    @property
    def est_valide(self) -> bool:
        """Vérifie si le document est encore valide."""
        return self.date_expiration >= date.today()


class PermisConduire(BaseModel):
    """Schéma pour l'extraction de permis de conduire."""

    type_document: Optional[TypeDocument] = Field(
        None,
        description="Type de document : permis de conduire",
    )

    numero_permis: str = Field(description="Numéro du permis de conduire")

    nom: str = Field(description="Nom de famille (en majuscules)")

    prenom: str = Field(description="Prénom(s)")

    date_naissance: date = Field(description="Date de naissance au format YYYY-MM-DD")

    lieu_naissance: Optional[str] = Field(None, description="Lieu de naissance (ville et pays)")

    sexe: Optional[Sexe] = Field(None, description="Genre : M, F, ou X")

    date_emission: date = Field(description="Date de délivrance du document au format YYYY-MM-DD")

    date_expiration: date = Field(
        description="Date de validité administrative au format YYYY-MM-DD"
    )

    categories: list[str] = Field(
        default_factory=list,
        description="Liste des catégories de permis (A, B, C, etc.)",
    )

    @field_validator("nom")
    @classmethod
    def uppercase_nom(cls, v: str) -> str:
        """Nom de famille en majuscules."""
        return v.upper()

    @property
    def est_valide(self) -> bool:
        """Vérifie si le document est encore valide."""
        return self.date_expiration >= date.today()


# Schéma de classification
class ClassificationDocument(BaseModel):
    """Résultat de la classification d'un document."""

    type_detecte: TypeDocument = Field(description="Type de document détecté")
    confiance: Optional[float] = Field(None, description="Score de confiance (0-1)")


# Résultat d'extraction
class ResultatExtractionKYC(BaseModel):
    """Résultat de l'extraction d'un document KYC."""

    classification: Optional[ClassificationDocument] = Field(
        None, description="Résultat de la classification"
    )
    extraction_reussie: bool = Field(False, description="L'extraction a-t-elle réussi ?")
    regles_metier_validees: bool = Field(
        False, description="Les règles métier sont-elles validées ?"
    )
    carte_identite: Optional[CarteIdentite] = Field(None, description="Données de carte d'identité")
    passeport: Optional[Passeport] = Field(None, description="Données de passeport")
    permis_conduire: Optional[PermisConduire] = Field(
        None, description="Données de permis de conduire"
    )
    justificatif_domicile: Optional[JustificatifDomicile] = Field(
        None, description="Données de justificatif de domicile"
    )
    rib: Optional[RIB] = Field(None, description="Données de RIB/IBAN")
    erreurs: list[str] = Field(default_factory=list, description="Messages d'erreur")
    avertissements: list[str] = Field(default_factory=list, description="Messages d'avertissement")
