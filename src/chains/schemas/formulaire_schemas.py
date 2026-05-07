"""Schémas Pydantic pour l'extraction du formulaire d'inscription complexe.

Ce formulaire est le "climax" de la démo : ~120 cases à cocher détectées visuellement par le LLM !
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Enums pour les choix du formulaire
# =============================================================================


class Civilite(str, Enum):
    """Civilité du demandeur."""

    MONSIEUR = "M."
    MADAME = "Mme"
    NON_PRECISE = "Ne souhaite pas répondre"


class Nationalite(str, Enum):
    """Nationalité du demandeur."""

    FRANCAISE = "Française"
    UNION_EUROPEENNE = "Union Européenne"
    HORS_UE = "Hors UE"


class ModeContact(str, Enum):
    """Mode de contact préféré."""

    COURRIER = "Courrier postal"
    EMAIL = "Email"
    SMS = "SMS"
    TELEPHONE = "Téléphone"


class StatutProfessionnel(str, Enum):
    """Statut professionnel actuel."""

    SALARIE_CDI = "Salarié(e) CDI"
    SALARIE_CDD = "Salarié(e) CDD"
    INTERIMAIRE = "Intérimaire"
    FONCTIONNAIRE = "Fonctionnaire"
    INDEPENDANT = "Indépendant / Auto-entrepreneur"
    CHEF_ENTREPRISE = "Chef d'entreprise"
    DEMANDEUR_EMPLOI = "Demandeur d'emploi"
    ETUDIANT = "Étudiant(e)"
    RETRAITE = "Retraité(e)"
    SANS_ACTIVITE = "Sans activité"


class CategorieSocioPro(str, Enum):
    """Catégorie socio-professionnelle."""

    CADRE_SUPERIEUR = "Cadre supérieur"
    CADRE = "Cadre"
    TECHNICIEN = "Technicien / Agent de maîtrise"
    EMPLOYE = "Employé"
    OUVRIER = "Ouvrier"


class TrancheExperience(str, Enum):
    """Tranche d'années d'expérience."""

    ZERO_DEUX = "0-2 ans"
    TROIS_CINQ = "3-5 ans"
    SIX_DIX = "6-10 ans"
    PLUS_DIX = "10+ ans"


class NiveauDiplome(str, Enum):
    """Niveau de diplôme."""

    SANS_DIPLOME = "Sans diplôme"
    CAP_BEP = "CAP / BEP"
    BAC = "Baccalauréat"
    BAC_PLUS_2 = "Bac +2 (BTS, DUT)"
    BAC_PLUS_3 = "Bac +3 (Licence)"
    BAC_PLUS_4 = "Bac +4 (Maîtrise)"
    BAC_PLUS_5 = "Bac +5 (Master, Ingénieur)"
    DOCTORAT = "Doctorat"


class DomaineFormation(str, Enum):
    """Domaines de formation possibles."""

    IA_ML = "Intelligence Artificielle / Machine Learning"
    DEV_WEB_MOBILE = "Développement Web / Mobile"
    CLOUD = "Cloud Computing (AWS, GCP, Azure)"
    DATA = "Data Science / Data Engineering"
    CYBERSECURITE = "Cybersécurité"
    DEVOPS = "DevOps / SRE"
    GESTION_PROJET = "Gestion de projet / Agilité"
    ARCHITECTURE = "Architecture logicielle"
    AUTRE = "Autre"


class NiveauVise(str, Enum):
    """Niveau visé pour la formation."""

    INITIATION = "Initiation"
    INTERMEDIAIRE = "Intermédiaire"
    AVANCE = "Avancé"
    EXPERT = "Expert"


class ModaliteFormation(str, Enum):
    """Modalité de formation."""

    PRESENTIEL = "Présentiel"
    DISTANCIEL = "Distanciel synchrone"
    ELEARNING = "E-learning (asynchrone)"
    HYBRIDE = "Hybride"


class Disponibilite(str, Enum):
    """Disponibilité pour la formation."""

    TEMPS_PLEIN = "Temps plein"
    TEMPS_PARTIEL = "Temps partiel (soirs)"
    WEEKENDS = "Week-ends uniquement"
    A_DEFINIR = "À définir selon planning"


class DateDebut(str, Enum):
    """Date de début souhaitée."""

    DES_QUE_POSSIBLE = "Dès que possible"
    SEPTEMBRE_2026 = "Septembre 2026"
    JANVIER_2027 = "Janvier 2027"


class TypeFinancement(str, Enum):
    """Types de financement."""

    CPF = "CPF (Compte Personnel de Formation)"
    PLAN_ENTREPRISE = "Plan de développement des compétences (employeur)"
    POLE_EMPLOI = "Pôle Emploi / France Travail (AIF)"
    OPCO = "OPCO de branche"
    TRANSITION_PRO = "Transition professionnelle (PTP/CPF-TP)"
    AUTOFINANCEMENT = "Autofinancement personnel"
    REGION = "Région / Collectivité territoriale"
    AUTRE = "Autre"


class StatutFinancement(str, Enum):
    """Statut de l'accord de financement."""

    ACCORD_OBTENU = "Oui, accord obtenu"
    EN_COURS = "En cours de demande"
    A_INITIER = "Non, à initier"


class ObjectifFormation(str, Enum):
    """Objectifs possibles de la formation."""

    RECONVERSION = "Reconversion professionnelle"
    MONTEE_COMPETENCES = "Montée en compétences sur poste actuel"
    MANAGEMENT = "Évolution vers un poste de management"
    SPECIALISATION = "Spécialisation technique"
    CERTIFICATION = "Obtention d'une certification"
    CREATION_ENTREPRISE = "Création d'entreprise"
    RETOUR_EMPLOI = "Retour à l'emploi"


class SituationParticuliere(str, Enum):
    """Situations particulières possibles."""

    RQTH = "RQTH (Reconnaissance Qualité Travailleur Handicapé)"
    OBLIGATION_EMPLOI = "Bénéficiaire de l'obligation d'emploi"
    DEMANDEUR_LONGUE_DUREE = "Demandeur d'emploi longue durée (> 12 mois)"
    RSA_ASS = "Bénéficiaire RSA / ASS"
    PARENT_ISOLE = "Parent isolé"
    AUCUNE = "Aucune situation particulière"


class NiveauCompetence(str, Enum):
    """Niveau d'auto-évaluation (1-5 ou N/A)."""

    NIVEAU_1 = "1"
    NIVEAU_2 = "2"
    NIVEAU_3 = "3"
    NIVEAU_4 = "4"
    NIVEAU_5 = "5"
    NON_APPLICABLE = "N/A"


# =============================================================================
# Sous-modèles pour structurer le formulaire
# =============================================================================


class EtatCivil(BaseModel):
    """Section 1 : État civil du demandeur."""

    civilite: Optional[Civilite] = Field(None, description="Civilité cochée (M., Mme, Ne souhaite pas répondre)")
    nom_naissance: Optional[str] = Field(None, description="Nom de naissance")
    nom_usage: Optional[str] = Field(None, description="Nom d'usage si différent")
    prenoms: Optional[str] = Field(None, description="Prénom(s)")
    date_naissance: Optional[date] = Field(
        None, description="Date de naissance au format YYYY-MM-DD"
    )
    lieu_naissance: Optional[str] = Field(None, description="Lieu de naissance")
    nationalite: Optional[Nationalite] = Field(None, description="Nationalité cochée")
    nationalite_autre: Optional[str] = Field(None, description="Précision si nationalité Hors UE")

    @model_validator(mode="after")
    def validate_nationalite_coherence(self) -> "EtatCivil":
        """Valide la cohérence : si nationalité Hors UE → précision requise."""
        if self.nationalite == Nationalite.HORS_UE:
            if not self.nationalite_autre:
                raise ValueError(
                    "Si la nationalité est 'Hors UE', vous devez préciser laquelle"
                )
        return self


class Coordonnees(BaseModel):
    """Section 2 : Coordonnées."""

    adresse: Optional[str] = Field(None, description="Adresse (numéro et rue)")
    complement_adresse: Optional[str] = Field(None, description="Complément d'adresse")
    code_postal: Optional[str] = Field(None, description="Code postal (5 chiffres)")
    ville: Optional[str] = Field(None, description="Ville")
    telephone_fixe: Optional[str] = Field(None, description="Téléphone fixe")
    telephone_mobile: Optional[str] = Field(None, description="Téléphone mobile")
    email: Optional[str] = Field(None, description="Adresse email")
    mode_contact_prefere: Optional[ModeContact] = Field(
        None, description="Mode de contact préféré coché"
    )

    @field_validator("code_postal")
    @classmethod
    def validate_code_postal(cls, v: Optional[str]) -> Optional[str]:
        """Valide que le code postal français a 5 chiffres."""
        if v is None:
            return v
        if len(v) != 5 or not v.isdigit():
            raise ValueError("Code postal doit contenir exactement 5 chiffres")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Valide le format basique de l'email."""
        if v is None:
            return v
        if "@" not in v or "." not in v:
            raise ValueError("Email invalide (doit contenir @ et .)")
        return v

    @field_validator("telephone_mobile")
    @classmethod
    def validate_telephone_mobile(cls, v: Optional[str]) -> Optional[str]:
        """Valide que le téléphone mobile français a 10 chiffres."""
        if v is None:
            return v
        digits_only = "".join(c for c in v if c.isdigit())
        if len(digits_only) != 10:
            raise ValueError("Téléphone mobile doit contenir 10 chiffres")
        return v


class SituationProfessionnelle(BaseModel):
    """Section 3 : Situation professionnelle actuelle."""

    statut: Optional[StatutProfessionnel] = Field(None, description="Statut professionnel coché")
    employeur: Optional[str] = Field(None, description="Nom de l'employeur actuel")
    fonction: Optional[str] = Field(None, description="Fonction / Poste occupé")
    secteur_activite: Optional[str] = Field(None, description="Secteur d'activité")
    categorie_sociopro: Optional[CategorieSocioPro] = Field(
        None, description="Catégorie socio-professionnelle cochée"
    )
    experience: Optional[TrancheExperience] = Field(
        None, description="Tranche d'années d'expérience cochée"
    )


class NiveauEtudes(BaseModel):
    """Section 4 : Niveau d'études et diplômes."""

    dernier_diplome: Optional[NiveauDiplome] = Field(
        None, description="Dernier diplôme obtenu coché"
    )
    intitule_diplome: Optional[str] = Field(None, description="Intitulé exact du diplôme")
    annee_obtention: Optional[int] = Field(None, description="Année d'obtention")
    etablissement: Optional[str] = Field(None, description="Établissement de formation")


class FormationSouhaitee(BaseModel):
    """Section 5 : Formation souhaitée."""

    domaines: list[DomaineFormation] = Field(
        default_factory=list,
        description="Domaines de formation cochés (peut être multiple)",
    )
    domaine_autre_precision: Optional[str] = Field(
        None, description="Précision si 'Autre' est coché"
    )
    niveau_vise: Optional[NiveauVise] = Field(None, description="Niveau visé coché")
    modalite: Optional[ModaliteFormation] = Field(None, description="Modalité souhaitée cochée")
    disponibilite: Optional[Disponibilite] = Field(None, description="Disponibilité cochée")
    date_debut: Optional[DateDebut] = Field(None, description="Date de début souhaitée cochée")


class Financement(BaseModel):
    """Section 6 : Mode de financement."""

    types_financement: list[TypeFinancement] = Field(
        default_factory=list,
        description="Types de financement cochés (peut être multiple)",
    )
    financement_autre_precision: Optional[str] = Field(
        None, description="Précision si 'Autre' est coché"
    )
    statut_accord: Optional[StatutFinancement] = Field(
        None, description="Statut de l'accord de financement coché"
    )

    @model_validator(mode="after")
    def validate_financement_coherence(self) -> "Financement":
        """Valide la cohérence : si types de financement remplis, statut_accord doit l'être aussi."""
        if len(self.types_financement) > 0 and self.statut_accord is None:
            raise ValueError(
                "Si types de financement sont cochés, le statut de l'accord doit être renseigné"
            )
        return self


class AutoEvaluationCompetences(BaseModel):
    """Section 7 : Auto-évaluation des compétences (tableau 7×6).

    C'est LE point fort de la démo : le LLM lit visuellement un tableau
    avec des cases à cocher et en extrait les valeurs !
    """

    programmation: Optional[NiveauCompetence] = Field(
        None, description="Niveau en Programmation (Python, Java, etc.)"
    )
    bases_de_donnees: Optional[NiveauCompetence] = Field(
        None, description="Niveau en Bases de données (SQL, NoSQL)"
    )
    cloud: Optional[NiveauCompetence] = Field(None, description="Niveau en Cloud (AWS, GCP, Azure)")
    machine_learning: Optional[NiveauCompetence] = Field(
        None, description="Niveau en Machine Learning / IA"
    )
    devops: Optional[NiveauCompetence] = Field(None, description="Niveau en DevOps / CI-CD")
    gestion_projet: Optional[NiveauCompetence] = Field(
        None, description="Niveau en Gestion de projet Agile"
    )
    anglais_technique: Optional[NiveauCompetence] = Field(
        None, description="Niveau en Anglais technique"
    )


class Motivations(BaseModel):
    """Section 8 : Motivations et projet professionnel."""

    objectifs: list[ObjectifFormation] = Field(
        default_factory=list,
        description="Objectifs de formation cochés (peut être multiple)",
    )
    projet_professionnel: Optional[str] = Field(
        None, description="Description du projet professionnel (zone de texte libre)"
    )


class SituationParticuliereSection(BaseModel):
    """Section 9 : Situation particulière."""

    situations: list[SituationParticuliere] = Field(
        default_factory=list,
        description="Situations particulières cochées",
    )
    besoin_amenagements: Optional[bool] = Field(
        None, description="Besoin d'aménagements spécifiques (Oui/Non)"
    )
    amenagements_details: Optional[str] = Field(
        None, description="Détails des aménagements si besoin"
    )


class Engagements(BaseModel):
    """Section 10 : Engagements et signature."""

    exactitude_informations: bool = Field(
        False, description="Case 'exactitude des informations' cochée"
    )
    reglement_interieur: bool = Field(
        False, description="Case 'pris connaissance du règlement' cochée"
    )
    conditions_generales: bool = Field(
        False, description="Case 'accepte les CGV et programme' cochée"
    )
    autorisation_contact: bool = Field(False, description="Case 'autorisation de contact' cochée")
    accepte_autres_formations: bool = Field(
        False, description="Case 'accepte infos autres formations' cochée (facultatif)"
    )
    lieu_signature: Optional[str] = Field(None, description="Lieu de signature (Fait à)")
    date_signature: Optional[date] = Field(
        None, description="Date de signature au format YYYY-MM-DD"
    )
    signature_presente: bool = Field(False, description="Une signature manuscrite est présente")


# =============================================================================
# Modèle principal du formulaire
# =============================================================================


class FormulaireInscription(BaseModel):
    """Formulaire d'inscription complet - ~120 cases à cocher.

    Ce schéma représente le formulaire DEMO-2026-05 avec 10 sections.
    C'est le document "climax" de la démo : beaucoup de cases à cocher
    que le LLM détecte visuellement sans aucun entraînement !
    """

    # Métadonnées du formulaire
    numero_formulaire: str = Field(
        default="DEMO-2026-05",
        description="Numéro du formulaire (en haut à droite)",
    )

    # Les 10 sections
    etat_civil: EtatCivil = Field(default_factory=EtatCivil, description="Section 1 : État civil")
    coordonnees: Coordonnees = Field(
        default_factory=Coordonnees, description="Section 2 : Coordonnées"
    )
    situation_professionnelle: SituationProfessionnelle = Field(
        default_factory=SituationProfessionnelle,
        description="Section 3 : Situation professionnelle",
    )
    niveau_etudes: NiveauEtudes = Field(
        default_factory=NiveauEtudes, description="Section 4 : Niveau d'études"
    )
    formation_souhaitee: FormationSouhaitee = Field(
        default_factory=FormationSouhaitee, description="Section 5 : Formation souhaitée"
    )
    financement: Financement = Field(
        default_factory=Financement, description="Section 6 : Financement"
    )
    competences: AutoEvaluationCompetences = Field(
        default_factory=AutoEvaluationCompetences,
        description="Section 7 : Auto-évaluation des compétences",
    )
    motivations: Motivations = Field(
        default_factory=Motivations, description="Section 8 : Motivations"
    )
    situation_particuliere: SituationParticuliereSection = Field(
        default_factory=SituationParticuliereSection,
        description="Section 9 : Situation particulière",
    )
    engagements: Engagements = Field(
        default_factory=Engagements, description="Section 10 : Engagements et signature"
    )

    # Statistiques pour la démo
    @property
    def nombre_cases_cochees(self) -> int:
        """Compte le nombre total de cases cochées dans le formulaire.

        Utile pour la démo : "Le LLM a détecté X cases cochées sur ~120 possibles"
        """
        count = 0

        # Section 1 : État civil
        if self.etat_civil.civilite:
            count += 1
        if self.etat_civil.nationalite:
            count += 1

        # Section 2 : Coordonnées
        if self.coordonnees.mode_contact_prefere:
            count += 1

        # Section 3 : Situation pro
        if self.situation_professionnelle.statut:
            count += 1
        if self.situation_professionnelle.categorie_sociopro:
            count += 1
        if self.situation_professionnelle.experience:
            count += 1

        # Section 4 : Niveau études
        if self.niveau_etudes.dernier_diplome:
            count += 1

        # Section 5 : Formation
        count += len(self.formation_souhaitee.domaines)
        if self.formation_souhaitee.niveau_vise:
            count += 1
        if self.formation_souhaitee.modalite:
            count += 1
        if self.formation_souhaitee.disponibilite:
            count += 1
        if self.formation_souhaitee.date_debut:
            count += 1

        # Section 6 : Financement
        count += len(self.financement.types_financement)
        if self.financement.statut_accord:
            count += 1

        # Section 7 : Compétences (7 lignes)
        competences = self.competences
        for field in [
            competences.programmation,
            competences.bases_de_donnees,
            competences.cloud,
            competences.machine_learning,
            competences.devops,
            competences.gestion_projet,
            competences.anglais_technique,
        ]:
            if field:
                count += 1

        # Section 8 : Motivations
        count += len(self.motivations.objectifs)

        # Section 9 : Situation particulière
        count += len(self.situation_particuliere.situations)
        if self.situation_particuliere.besoin_amenagements is not None:
            count += 1

        # Section 10 : Engagements (5 cases)
        engagements = self.engagements
        if engagements.exactitude_informations:
            count += 1
        if engagements.reglement_interieur:
            count += 1
        if engagements.conditions_generales:
            count += 1
        if engagements.autorisation_contact:
            count += 1
        if engagements.accepte_autres_formations:
            count += 1

        return count

    @property
    def est_complet(self) -> bool:
        """Vérifie si les champs obligatoires sont remplis."""
        # Champs obligatoires marqués d'un * dans le formulaire
        required_checks = [
            self.etat_civil.civilite is not None,
            self.etat_civil.nom_naissance is not None,
            self.etat_civil.prenoms is not None,
            self.etat_civil.date_naissance is not None,
            self.etat_civil.lieu_naissance is not None,
            self.etat_civil.nationalite is not None,
            self.coordonnees.adresse is not None,
            self.coordonnees.code_postal is not None,
            self.coordonnees.ville is not None,
            self.coordonnees.telephone_mobile is not None,
            self.coordonnees.email is not None,
            self.situation_professionnelle.statut is not None,
            self.niveau_etudes.dernier_diplome is not None,
            len(self.formation_souhaitee.domaines) > 0,
            self.formation_souhaitee.niveau_vise is not None,
            self.formation_souhaitee.modalite is not None,
            self.formation_souhaitee.disponibilite is not None,
            self.engagements.exactitude_informations,
            self.engagements.reglement_interieur,
            self.engagements.conditions_generales,
        ]
        return all(required_checks)
