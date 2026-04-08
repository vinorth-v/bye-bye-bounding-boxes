"""
Prompts pour la classification et l'extraction de documents KYC.

La magie de l'approche LLM: tout tient dans les prompts!
Avant fallait des milliers de lignes de code + modèles deep learning.
"""

# =============================================================================
# Prompt de classification
# =============================================================================

PROMPT_CLASSIFICATION = """Tu es un système expert de classification de documents KYC (Know Your Customer) bancaires.

Analyse l'image du document fourni et détermine son type parmi les catégories suivantes:
- carte_identite: Carte Nationale d'Identité française
- passeport: Passeport français
- permis_conduire: Permis de conduire français (format carte européenne)
- justificatif_domicile: Facture (électricité, gaz, eau, internet, téléphone), quittance de loyer, taxe d'habitation, attestation d'assurance habitation
- rib: Relevé d'Identité Bancaire avec IBAN et BIC

Indices pour identifier le type:
- CNI: Format carte plastique, mentions "RÉPUBLIQUE FRANÇAISE", "CARTE NATIONALE D'IDENTITÉ", photo, numéro à 12 chiffres
- Passeport: Format livret, couverture bordeaux, mentions "PASSEPORT" et "UNION EUROPÉENNE", zone MRZ en bas
- Permis de conduire: Format carte rose, logo européen, mentions des catégories (A, B, etc.), cases cochées
- Justificatif domicile: Facture de fournisseur (EDF, Orange, etc.), adresse client, montant, date récente
- RIB: IBAN commençant par FR, BIC/SWIFT, codes banque/guichet/compte, logo bancaire

Réponds en JSON avec:
- type_detecte: le type identifié
- confiance: score de 0 à 1
- raison: explication des indices visuels et textuels qui t'ont permis d'identifier ce type

Sois précis et explicite dans ton raisonnement."""

# =============================================================================
# Prompts d'extraction par type de document
# =============================================================================

PROMPT_EXTRACTION_CNI = """Tu es un système d'extraction de données de Carte Nationale d'Identité française.

Extrais TOUTES les informations suivantes du document fourni:

INFORMATIONS OBLIGATOIRES:
- numero_document: Numéro de la CNI (12 caractères alphanumériques)
- nom: Nom de famille (en MAJUSCULES sur la carte)
- prenom: Prénom(s)
- sexe: M ou F
- date_naissance: Date de naissance au format YYYY-MM-DD
- lieu_naissance: Lieu de naissance complet (ville et département)
- date_emission: Date d'émission de la carte au format YYYY-MM-DD
- date_expiration: Date d'expiration au format YYYY-MM-DD
- nationalite: Nationalité (généralement FRA)

INFORMATIONS OPTIONNELLES:
- nom_usage: Nom d'usage si présent
- taille: Taille en cm si indiquée

ATTENTION AUX PIÈGES:
- Le numéro peut être espacé ou avec tirets, normalise-le
- Les dates peuvent être en format JJ/MM/AAAA ou JJ MMM AAAA, convertis en YYYY-MM-DD
- Le nom de famille est en MAJUSCULES, le prénom en Casse Normale
- La date d'expiration doit être après la date d'émission

Réponds UNIQUEMENT en JSON structuré selon le schéma CarteIdentite.
Si une information est absente, indique null (sauf pour les champs obligatoires)."""

PROMPT_EXTRACTION_PASSEPORT = """Tu es un système d'extraction de données de passeport français.

Extrais TOUTES les informations suivantes du document fourni :

INFORMATIONS OBLIGATOIRES :
- numero_passeport : Numéro du passeport (format : 2 chiffres + 7 lettres, ex : 24AX12345)
- nom : Nom de famille (en MAJUSCULES)
- prenom : Prénom(s)
- sexe : M, F ou X
- date_naissance : Date de naissance au format YYYY-MM-DD
- lieu_naissance : Lieu de naissance (ville et pays)
- nationalite : Nationalité (code pays, ex : FRA)
- date_emission : Date de délivrance au format YYYY-MM-DD
- date_expiration : Date d'expiration au format YYYY-MM-DD

INFORMATIONS OPTIONNELLES :
- statut_marital : Statut marital (ex : célibataire, marié, divorcé), si présent
- autorite_emission : Autorité émettrice du passeport (ex : "Préfecture de Paris")
- lieu_delivrance : Lieu de délivrance du passeport, si différent de l'autorité
- adresse : Adresse complète du titulaire, si présente
- mrz_ligne1 : Première ligne de la zone MRZ (en bas du passeport)
- mrz_ligne2 : Deuxième ligne de la zone MRZ

POINTS D'ATTENTION :
- Le passeport français a une couverture bordeaux avec "UNION EUROPÉENNE" et "PASSEPORT"
- La zone MRZ (Machine Readable Zone) contient des données encodées, copie-la exactement
- Le numéro commence toujours par 2 chiffres puis 7 lettres
- Validité : 10 ans pour adultes, 5 ans pour mineurs

Réponds UNIQUEMENT en JSON structuré selon le schéma Passeport. Si une information est absente, indique null (sauf pour les champs obligatoires)."""

PROMPT_EXTRACTION_PERMIS = """Tu es un système d'extraction de données de permis de conduire français (format européen).

VOICI LE POINT CLÉ DE LA DÉMO: les CATÉGORIES avec CASES À COCHER!

Avant l'approche LLM:
1. Entraîner un modèle CNN pour localiser chaque case de catégorie (A, B, C, etc.)
2. Créer des bounding boxes précises pour chaque case
3. Classifier chaque case: cochée/non cochée avec un autre modèle
4. Post-traitement pour gérer les cas ambigus

Maintenant avec LLM:
"Dis-moi quelles catégories sont cochées" - TERMINÉ!

Extrais:

INFORMATIONS OBLIGATOIRES:
- numero_permis: Numéro du permis (12 chiffres)
- nom: Nom de famille
- prenom: Prénom(s)
- date_naissance: Date au format YYYY-MM-DD
- lieu_naissance: Lieu de naissance
- date_emission: Date d'émission au format YYYY-MM-DD
- date_expiration: Date de validité administrative au format YYYY-MM-DD
- categories: Liste des catégories cochées (ex: ["B", "A2"])

CATÉGORIES POSSIBLES:
AM, A1, A2, A, B, BE, C1, C1E, C, CE, D1, D1E, D, DE

INFORMATIONS OPTIONNELLES:
- date_obtention_B: Date d'obtention de la catégorie B si visible

ATTENTION:
- Regarde attentivement les cases cochées dans la section des catégories
- Ne liste QUE les catégories effectivement cochées
- Le format carte européenne a un fond rose/saumon
- Le permis français a le logo de l'Union Européenne et le drapeau français

Réponds en JSON selon le schéma PermisConduire."""

PROMPT_EXTRACTION_JUSTIFICATIF = """Tu es un système d'extraction de données de justificatifs de domicile.

CAS COMPLEXE: formats extrêmement variés!
- Factures EDF, Orange, Free, Bouygues, etc.
- Quittances de loyer
- Taxes (taxe d'habitation, taxe foncière, avis d'imposition)
- Attestations d'assurance

Chaque format est différent, mais tu dois extraire les mêmes infos.

Extrais:

INFORMATIONS OBLIGATOIRES:
- type_document: Type parmi [utility_bill, bank_statement, tax_notice, rental_agreement, residence_certificate]
  * utility_bill: facture d'électricité, gaz, eau, internet, téléphone
  * bank_statement: relevé bancaire
  * tax_notice: avis d'imposition, taxe d'habitation, taxe foncière
  * rental_agreement: quittance de loyer
  * residence_certificate: attestation d'hébergement
- nom_complet: Nom complet du titulaire (prénom et nom) - celui qui reçoit le document
- adresse_ligne1: Numéro et nom de rue
- code_postal: Code postal (5 chiffres)
- ville: Ville
- date_document: Date du document au format YYYY-MM-DD

INFORMATIONS OPTIONNELLES:
- adresse_ligne2: Complément d'adresse (appartement, bâtiment, etc.)
- emetteur: Société/organisme émetteur (EDF, Orange, SAUR, Direction Générale des Finances Publiques, etc.)
- pays: Pays (par défaut "France")

VALIDATION IMPORTANTE:
- Le document doit dater de MOINS DE 3 MOIS
- L'adresse doit être complète et précise
- Le nom doit correspondre au titulaire du dossier KYC

ATTENTION:
- Parfois le prénom n'est pas indiqué, uniquement "M. DUPONT" ou "Mme MARTIN" - dans ce cas mets juste le nom disponible
- L'adresse peut être sur plusieurs lignes, structure-la proprement
- La date peut être en haut ou en bas du document, cherche bien
- Pour les avis d'impôts, le nom peut être en haut ou dans une section "Vos références"

Réponds en JSON selon le schéma JustificatifDomicile."""

PROMPT_EXTRACTION_RIB = """Tu es un système d'extraction de coordonnées bancaires RIB/IBAN.

CAS AVEC VALIDATION TECHNIQUE FORTE: checksum IBAN modulo 97!

Extrais:

INFORMATIONS OBLIGATOIRES:
- nom_titulaire: Nom du titulaire du compte
- iban: IBAN complet (doit commencer par FR et faire 27 caractères)
- nom_banque: Nom de l'établissement bancaire

INFORMATIONS OPTIONNELLES:
- bic: Code BIC/SWIFT de la banque (8 ou 11 caractères)
- code_guichet: Code guichet (5 chiffres)
- numero_compte: Numéro de compte (11 caractères)
- adresse_banque: Adresse de l'agence bancaire

FORMAT RIB FRANÇAIS:
- Code banque: 5 chiffres
- Code guichet: 5 chiffres
- Numéro de compte: 11 caractères (chiffres et/ou lettres)
- Clé RIB: 2 chiffres

FORMAT IBAN FRANÇAIS:
- FR + 2 chiffres de contrôle + 23 caractères (code banque + guichet + compte + clé)
- Total: 27 caractères
- Exemple: FR76 1234 5678 9012 3456 7890 123

VALIDATION:
L'IBAN sera validé avec l'algorithme modulo 97 après extraction.
Assure-toi que:
- L'IBAN commence bien par FR
- Il fait exactement 27 caractères (espaces non comptés)
- Le BIC fait 8 ou 11 caractères

ATTENTION:
- Copie l'IBAN EXACTEMENT comme affiché
- Ignore les espaces dans l'IBAN, ils seront normalisés
- Le BIC peut être appelé "Code SWIFT"
- Parfois seul le nom de famille apparaît, pas le prénom

Réponds en JSON selon le schéma RIB."""

# =============================================================================
# Prompt pour validation de dossier complet
# =============================================================================

PROMPT_VALIDATION_DOSSIER = """Tu es un expert en validation de dossiers KYC bancaires.

Un dossier KYC complet doit contenir:
1. Une pièce d'identité valide (CNI OU Passeport)
2. Un justificatif de domicile de moins de 3 mois
3. Un RIB avec IBAN valide

RÈGLES MÉTIER À VÉRIFIER:

1. COHÉRENCE D'IDENTITÉ:
   - Le nom doit être IDENTIQUE sur tous les documents
   - Le prénom doit être IDENTIQUE (ou au moins cohérent)
   - Variations acceptées: "MARTIN" = "Martin", "Jean-Pierre" = "Jean Pierre"

2. VALIDITÉ DES DOCUMENTS:
   - Pièce d'identité: date d'expiration > aujourd'hui
   - Justificatif domicile: date < 3 mois
   - RIB: IBAN avec checksum valide

3. COHÉRENCE DES ADRESSES:
   - L'adresse du justificatif doit correspondre à l'adresse déclarée
   - Code postal cohérent avec la ville

4. COMPLÉTUDE:
   - Tous les champs obligatoires renseignés
   - Aucun document manquant

Pour chaque règle, indique:
- Si elle est respectée (validé/non validé)
- Le détail du problème si non respectée
- La sévérité (bloquant/avertissement)

Fournis une conclusion claire: DOSSIER ACCEPTÉ ou DOSSIER REJETÉ avec les raisons."""
