"""__init__.py pour le module schemas."""

from chains.schemas.formulaire_schemas import (
    FormulaireInscription,
)
from chains.schemas.kyc_schemas import (
    RIB,
    CarteIdentite,
    ClassificationDocument,
    DossierKYC,
    JustificatifDomicile,
    Passeport,
    PermisConduire,
    ResultatExtractionKYC,
    Sexe,
    TypeDocument,
    TypeJustificatifDomicile,
)

__all__ = [
    "CarteIdentite",
    "ClassificationDocument",
    "DossierKYC",
    "FormulaireInscription",
    "JustificatifDomicile",
    "Passeport",
    "PermisConduire",
    "ResultatExtractionKYC",
    "RIB",
    "Sexe",
    "TypeDocument",
    "TypeJustificatifDomicile",
]
