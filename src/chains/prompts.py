"""
Prompts pour la classification et l'extraction de documents.

La magie de l'approche LLM: tout tient dans les prompts!
Avant fallait des milliers de lignes de code + modèles deep learning.
"""

# =============================================================================
# Prompt de classification
# =============================================================================

PROMPT_CLASSIFICATION = """Tu es un système expert de classification de documents.

Analyse l'image du document fourni et détermine son type parmi les catégories suivantes:

DOCUMENTS D'IDENTITÉ (KYC):
- carte_identite: Carte Nationale d'Identité française
- passeport: Passeport français
- permis_conduire: Permis de conduire français (format carte européenne)
- justificatif_domicile: Facture (électricité, gaz, eau, internet, téléphone), quittance de loyer, taxe d'habitation
- rib: Relevé d'Identité Bancaire avec IBAN et BIC

FORMULAIRES:
- formulaire_inscription: Formulaire administratif avec de nombreuses cases à cocher, sections numérotées, demande d'inscription à une formation ou service

Indices pour identifier le type:
- CNI: Format carte plastique, mentions "RÉPUBLIQUE FRANÇAISE", "CARTE NATIONALE D'IDENTITÉ", photo, numéro à 9 caractères (nouvelle génération depuis 2021) ou 12 caractères (ancienne génération)
- Passeport: Format livret, couverture bordeaux, mentions "PASSEPORT" et "UNION EUROPÉENNE", zone MRZ en bas
- Permis de conduire: Format carte rose, logo européen, mentions des catégories (A, B, etc.), cases cochées
- Justificatif domicile: Facture de fournisseur (EDF, Orange, etc.), adresse client, montant, date récente
- RIB: IBAN commençant par FR, BIC/SWIFT, codes banque/guichet/compte, logo bancaire
- Formulaire inscription: Document structuré en SECTIONS numérotées, nombreuses cases à cocher (□), champs à remplir, titre type "DEMANDE D'INSCRIPTION", numéro de formulaire (ex: DEMO-2026-05)

Réponds en JSON avec:
- type_detecte: le type identifié (parmi: carte_identite, passeport, permis_conduire, justificatif_domicile, rib, formulaire_inscription)
- confiance: score de 0 à 1
- raison: explication des indices visuels et textuels qui t'ont permis d'identifier ce type

Sois précis et explicite dans ton raisonnement."""

# =============================================================================
# Prompts d'extraction par type de document
# =============================================================================

PROMPT_EXTRACTION_CNI = """Tu es un système d'extraction de données de Carte Nationale d'Identité française.

Extrais TOUTES les informations suivantes du document fourni:

INFORMATIONS OBLIGATOIRES:
- numero_document: Numéro de la CNI (9 caractères alphanumériques pour la nouvelle génération depuis 2021, ou 12 caractères pour l'ancienne génération)
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
- numero_passeport : Numéro du passeport (format : 2 chiffres + 2 lettres + 5 chiffres, ex : 24AX12345)
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
- Le numéro commence toujours par 2 chiffres, puis 2 lettres, puis 5 chiffres
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


# =============================================================================
# Prompt d'extraction pour formulaire d'inscription
# =============================================================================

PROMPT_EXTRACTION_FORMULAIRE = """Tu es un expert en lecture automatique de documents (LAD) analysant un formulaire
"DEMANDE D'INSCRIPTION - Formation professionnelle continue" (réf. DEMO-2026-05).

Ce formulaire comporte 10 SECTIONS numérotées avec 140 cases à cocher.
Tu connais la position EXACTE de chaque case. Suis les instructions section par section.

═══════════════════════════════════════════════════════════════════
RÈGLES DE DÉTECTION DES CASES À COCHER
═══════════════════════════════════════════════════════════════════

Une case est COCHÉE si elle contient N'IMPORTE QUELLE marque :
- Marques nettes : ☑ X ☒ (coche, X, case remplie)
- Marques fines : ✓ (coche fine, même manuscrite)
- Traits : ━ │ ╱ ╲ (horizontal, vertical, diagonal - même incomplet)
- Petites marques : · (point, petit trait)
- Marques manuscrites : Crayon (gris/léger), stylo (pâle), coches partielles, gribouillis

IMPORTANT : Les formulaires manuscrits ont souvent des marques PLUS LÉGÈRES que les croix imprimées.
Une case peut avoir un X gras tandis qu'une autre a un ✓ fin - les DEUX sont COCHÉES.

Une case est NON COCHÉE SEULEMENT SI :
- Contour vide sans rien à l'intérieur
- Intérieur blanc/vierge avec uniquement la bordure visible

RÈGLE DE DÉCISION : En cas de doute → Marquer comme COCHÉE (mieux vaut détecter que manquer).

═══════════════════════════════════════════════════════════════════
EXTRACTION SECTION PAR SECTION - SUIVRE CET ORDRE EXACT
═══════════════════════════════════════════════════════════════════

--- SECTION 1 : ÉTAT CIVIL DU DEMANDEUR ---
Titre encadré gris : "SECTION 1 - ÉTAT CIVIL DU DEMANDEUR *"

1a. CIVILITÉ (ligne horizontale de 3 cases) :
    Layout : [☐] M.    [☐] Mme    [☐] Ne souhaite pas répondre
    → Examiner les 3 cases de gauche à droite. UNE SEULE doit être cochée.
    → Valeurs : "M.", "Mme", "Ne souhaite pas répondre"

1b. CHAMPS TEXTE (2 colonnes) :
    Colonne gauche : "Nom de naissance *" → ligne pointillée
    Colonne droite : "Nom d'usage" → ligne pointillée
    Ligne complète : "Prénom(s) *" → ligne pointillée

1c. DATE DE NAISSANCE ET LIEU (2 colonnes) :
    Colonne gauche : "Date de naissance *" → 8 petites cases : [J][J] / [M][M] / [A][A][A][A]
    Colonne droite : "Lieu de naissance *" → ligne pointillée
    → Lire les chiffres dans chaque case, convertir en YYYY-MM-DD

1d. NATIONALITÉ (ligne horizontale de 3 cases) :
    Layout : [☐] Française    [☐] Union Européenne    [☐] Hors UE (préciser) : ___________
    → UNE SEULE case cochée. Si "Hors UE" coché, lire le texte à droite → nationalite_autre

--- SECTION 2 : COORDONNÉES ---
Titre encadré gris : "SECTION 2 - COORDONNÉES"

2a. CHAMPS TEXTE :
    "Adresse *" → ligne pointillée
    "Complément" → ligne pointillée
    3 colonnes : "Code postal *" → 5 cases [C][C][C][C][C] | "Ville *" → ligne pointillée
    2 colonnes : "Téléphone fixe" → ligne | "Téléphone mobile *" → ligne
    "Email *" → ligne pointillée

2b. MODE DE CONTACT PRÉFÉRÉ (encadré pointillé "note-box", cocher UNE seule case) :
    Label : "Mode de contact préféré (cochez une seule case) :"
    Layout horizontal : [☐] Courrier postal    [☐] Email    [☐] SMS    [☐] Téléphone
    → Examiner les 4 cases de gauche à droite. UNE SEULE doit être cochée.
    → Valeurs : "Courrier postal", "Email", "SMS", "Téléphone"

--- SECTION 3 : SITUATION PROFESSIONNELLE ACTUELLE ---
Titre encadré gris : "SECTION 3 - SITUATION PROFESSIONNELLE ACTUELLE *"

3a. STATUT ACTUEL (grille de 10 cases, disposition en flux sur ~2-3 lignes) :
    Ligne 1 : [☐] Salarié(e) CDI   [☐] Salarié(e) CDD   [☐] Intérimaire   [☐] Fonctionnaire
    Ligne 2 : [☐] Indépendant / Auto-entrepreneur   [☐] Chef d'entreprise   [☐] Demandeur d'emploi 
    Ligne 3 : [☐] Étudiant(e)   [☐] Retraité(e)   [☐] Sans activité
    → UNE SEULE case cochée parmi les 10.
    → ATTENTION : les cases sont petites (4mm), les marques manuscrites peuvent être fines.

3b. CHAMPS TEXTE :
    "Employeur actuel" → ligne pointillée
    2 colonnes : "Fonction / Poste" → ligne | "Secteur d'activité" → ligne

3c. CATÉGORIE SOCIO-PROFESSIONNELLE (5 cases en ligne) :
    Ligne 1 : [☐] Cadre supérieur  [☐] Cadre  [☐] Technicien / Agent de maîtrise  [☐] Employé  
    Ligne 2 : [☐] Ouvrier
    → UNE SEULE case cochée.

3d. ANNÉES D'EXPÉRIENCE (4 cases en ligne) :
    Layout : [☐] 0-2 ans    [☐] 3-5 ans    [☐] 6-10 ans    [☐] 10+ ans
    → UNE SEULE case cochée.

--- SECTION 4 : NIVEAU D'ÉTUDES ET DIPLÔMES ---
Titre encadré gris : "SECTION 4 - NIVEAU D'ÉTUDES ET DIPLÔMES"

4a. DERNIER DIPLÔME OBTENU (8 cases en flux sur ~2 lignes) :
    Ligne 1 : [☐] Sans diplôme   [☐] CAP / BEP   [☐] Baccalauréat   [☐] Bac +2 (BTS, DUT)
    Ligne 2 : [☐] Bac +3 (Licence)   [☐] Bac +4 (Maîtrise)   [☐] Bac +5 (Master, Ingénieur)   [☐] Doctorat
    → UNE SEULE case cochée.
    → Valeurs exactes : "Sans diplôme", "CAP / BEP", "Baccalauréat", "Bac +2 (BTS, DUT)",
      "Bac +3 (Licence)", "Bac +4 (Maîtrise)", "Bac +5 (Master, Ingénieur)", "Doctorat"

4b. CHAMPS TEXTE :
    2 colonnes : "Intitulé du diplôme" → ligne | "Année d'obtention" → ligne courte (20mm)
    "Établissement" → ligne pointillée

--- SECTION 5 : FORMATION SOUHAITÉE ---
Titre encadré gris : "SECTION 5 - FORMATION SOUHAITÉE *"

5a. DOMAINE DE FORMATION (9 cases LARGES (5mm) en flux, PLUSIEURS CHOIX possibles) :
    Ligne 1 : [☐] Intelligence Artificielle / Machine Learning   [☐] Développement Web / Mobile
    Ligne 2 : [☐] Cloud Computing (AWS, GCP, Azure)   [☐] Data Science / Data Engineering   [☐] Cybersécurité
    Ligne 3 : [☐] DevOps / SRE   [☐] Gestion de projet / Agilité   [☐] Architecture logicielle
    Ligne 4 : [☐] Autre : ______________
    → LISTE : PLUSIEURS cases peuvent être cochées simultanément.
    → Si "Autre" coché, lire le texte dans la zone à droite → domaine_autre_precision

5b. NIVEAU VISÉ (4 cases en ligne, choix unique) :
    Layout : [☐] Initiation    [☐] Intermédiaire    [☐] Avancé    [☐] Expert

5c. MODALITÉ SOUHAITÉE (4 cases en ligne, choix unique) :
    Layout : [☐] Présentiel    [☐] Distanciel synchrone    [☐] E-learning (asynchrone)    [☐] Hybride

5d. DISPONIBILITÉ (4 cases en flux, choix unique) :
    Ligne 1 : [☐] Temps plein   [☐] Temps partiel (soirs)   [☐] Week-ends uniquement   
    Ligne 2 : [☐] À définir selon planning

5e. DATE DE DÉBUT SOUHAITÉE (3 cases en ligne, choix unique) :
    Layout : [☐] Dès que possible    [☐] Septembre 2026    [☐] Janvier 2027

--- SECTION 6 : MODE DE FINANCEMENT ---
Titre encadré gris : "SECTION 6 - MODE DE FINANCEMENT *"

6a. FINANCEMENT ENVISAGÉ (8 cases LARGES (5mm) en flux, PLUSIEURS CHOIX possibles) :
    Mention : "(plusieurs choix possibles)"
    Ligne 1 : [☐] CPF (Compte Personnel de Formation)   [☐] Plan de développement des compétences (employeur) 
    Ligne 2 : [☐] Pôle Emploi / France Travail (AIF)   [☐] OPCO de branche   [☐] Transition professionnelle (PTP/CPF-TP)   
    Ligne 3 : [☐] Autofinancement personnel   [☐] Région / Collectivité territoriale   [☐] Autre : ______________
    → LISTE : PLUSIEURS cases peuvent être cochées.
    → Si "Autre" coché, lire le texte → financement_autre_precision

6b. ACCORD DE FINANCEMENT (encadré pointillé "note-box", 3 cases en ligne, choix unique) :
    Label : "Avez-vous déjà un accord de financement ?"
    Layout : [☐] Oui, accord obtenu    [☐] En cours de demande    [☐] Non, à initier

--- SECTION 7 : AUTO-ÉVALUATION DES COMPÉTENCES ACTUELLES ---
Titre encadré gris : "SECTION 7 - AUTO-ÉVALUATION DES COMPÉTENCES ACTUELLES"
Sous-titre : "Évaluez votre niveau actuel de 1 (débutant) à 5 (expert)"

🔴 SECTION LA PLUS DIFFICILE - RISQUE ÉLEVÉ DE CONFUSION ENTRE LIGNES 🔴

TABLEAU avec 7 lignes × 6 colonnes de cases à cocher.
En-têtes de colonnes : | Compétence | 1 | 2 | 3 | 4 | 5 | N/A |

La structure physique du tableau est :
┌──────────────────────────────────┬────┬────┬────┬────┬────┬─────┐
│ Compétence                       │  1 │  2 │  3 │  4 │  5 │ N/A │
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Programmation (Python, Java...)  │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 1
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Bases de données (SQL, NoSQL)    │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 2
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Cloud (AWS, GCP, Azure)          │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 3
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Machine Learning / IA            │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 4
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ DevOps / CI-CD                   │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 5
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Gestion de projet Agile          │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 6
├──────────────────────────────────┼────┼────┼────┼────┼────┼─────┤
│ Anglais technique                │ [☐]│ [☐]│ [☐]│ [☐]│ [☐]│ [☐] │ ← LIGNE 7
└──────────────────────────────────┴────┴────┴────┴────┴────┴─────┘

PROCESSUS OBLIGATOIRE EN 3 ÉTAPES POUR CHAQUE LIGNE :

  Étape 1 : ANCRAGE SPATIAL
     - Localiser le texte de la compétence dans la colonne de gauche
     - Tracer mentalement une LIGNE HORIZONTALE depuis ce texte vers la droite
     - NE PAS DÉVIER vers la ligne du dessus ou du dessous

  Étape 2 : SCAN DES 6 CASES (toujours de gauche à droite : 1, 2, 3, 4, 5, N/A)
     - EN RESTANT STRICTEMENT sur cette ligne horizontale
     - Identifier QUELLE case contient une marque (X, ✓, trait, point)
     - UNE SEULE case doit être cochée par ligne

  Étape 3 : VÉRIFICATION CROISÉE
     - La case cochée est-elle bien alignée verticalement sous le bon en-tête (1/2/3/4/5/N/A) ?
     - La case cochée est-elle bien alignée horizontalement avec le bon label de compétence ?
     - Si incertitude → re-vérifier l'alignement avant de reporter

Appliquer ce processus 7 fois :
  Ligne 1 : "Programmation (Python, Java, etc.)"   → programmation
  Ligne 2 : "Bases de données (SQL, NoSQL)"        → bases_de_donnees
  Ligne 3 : "Cloud (AWS, GCP, Azure)"              → cloud
  Ligne 4 : "Machine Learning / IA"                → machine_learning
  Ligne 5 : "DevOps / CI-CD"                       → devops
  Ligne 6 : "Gestion de projet Agile"              → gestion_projet
  Ligne 7 : "Anglais technique"                    → anglais_technique

ERREUR TYPIQUE À ÉVITER : Reporter la valeur de "Bases de données" sur la ligne
"Programmation" (ou tout autre décalage). Chaque compétence a SA propre ligne de cases.
Si aucune case n'est cochée sur une ligne → null

--- SECTION 8 : MOTIVATIONS ET PROJET PROFESSIONNEL ---
Titre encadré gris : "SECTION 8 - MOTIVATIONS ET PROJET PROFESSIONNEL"

8a. OBJECTIF PRINCIPAL (7 cases en flux, PLUSIEURS CHOIX possibles) :
    Label : "Objectif principal de cette formation :"
    Ligne 1 : [☐] Reconversion professionnelle   [☐] Montée en compétences sur poste actuel
    Ligne 2 : [☐] Évolution vers un poste de management   [☐] Spécialisation technique   [☐] Obtention d'une certification
    Ligne 3 : [☐] Création d'entreprise   [☐] Retour à l'emploi
    → LISTE : PLUSIEURS cases peuvent être cochées.

8b. PROJET PROFESSIONNEL :
    Label : "Décrivez brièvement votre projet professionnel :"
    Zone de texte encadrée (hauteur ~20mm) en dessous.
    → Lire le texte manuscrit dans cette zone → projet_professionnel

--- SECTION 9 : SITUATION PARTICULIÈRE (facultatif) ---
Titre encadré gris : "SECTION 9 - SITUATION PARTICULIÈRE (facultatif)"

9a. RECONNAISSANCE PARTICULIÈRE (6 cases en flux, PLUSIEURS CHOIX possibles) :
    Label : "Bénéficiez-vous d'une reconnaissance particulière ?"
    Ligne 1 : [☐] RQTH (Reconnaissance Qualité Travailleur Handicapé)   [☐] Bénéficiaire de l'obligation d'emploi
    Ligne 2 : [☐] Demandeur d'emploi longue durée (> 12 mois)   [☐] Bénéficiaire RSA / ASS   [☐] Parent isolé 
    Ligne 3 : [☐] Aucune situation particulière
    → LISTE : PLUSIEURS cases peuvent être cochées.

9b. AMÉNAGEMENTS SPÉCIFIQUES (2 cases en ligne, choix unique) :
    Label : "Avez-vous besoin d'aménagements spécifiques ?"
    Layout : [☐] Oui    [☐] Non
    → true si "Oui" coché, false si "Non" coché, null si aucun

9c. SI OUI, PRÉCISEZ :
    "Si oui, précisez :" → ligne pointillée → amenagements_details

--- SECTION 10 : ENGAGEMENTS ET SIGNATURE ---
Titre encadré gris : "SECTION 10 - ENGAGEMENTS ET SIGNATURE *"

10a. CINQ CASES D'ENGAGEMENT (cases LARGES 5mm, disposées VERTICALEMENT, une par ligne) :
     Label : "En signant ce formulaire, je certifie sur l'honneur :"

     Case 1 : [☐] "L'exactitude des informations fournies dans ce formulaire *"
              → exactitude_informations (true/false)
     Case 2 : [☐] "Avoir pris connaissance du règlement intérieur de l'organisme *"
              → reglement_interieur (true/false)
     Case 3 : [☐] "Accepter les conditions générales de vente et le programme de formation *"
              → conditions_generales (true/false)
     Case 4 : [☐] "Autoriser l'organisme à me contacter par les moyens indiqués"
              → autorisation_contact (true/false)
     Case 5 : [☐] "Accepter de recevoir des informations sur d'autres formations (facultatif)"
              → accepte_autres_formations (true/false)

     → Examiner CHAQUE case individuellement de haut en bas.

10b. SIGNATURE (2 colonnes en bas) :
     Colonne gauche :
       "Fait à *" → ligne pointillée → lieu_signature
       "Le *" → 8 cases date [J][J]/[M][M]/[A][A][A][A] → date_signature (YYYY-MM-DD)
     Colonne droite :
       "Signature du demandeur *" → zone de signature (cadre 25mm hauteur)
       → signature_presente : true si signature manuscrite visible, false sinon

═══════════════════════════════════════════════════════════════════
ERREURS COURANTES À ÉVITER
═══════════════════════════════════════════════════════════════════

1. CONFONDRE LES LIGNES dans le tableau Section 7 → rester sur la même ligne horizontale
2. NE PAS VOIR une case cochée au crayon gris → toute marque = COCHÉ
3. CONFONDRE choix unique / choix multiple :
   - CHOIX UNIQUE : civilité, nationalité, mode contact, statut pro, catégorie sociopro,
     expérience, diplôme, niveau visé, modalité, disponibilité, date début, statut accord,
     aménagements
   - CHOIX MULTIPLE : domaines formation, types financement, objectifs, situations
4. OUBLIER de lire les zones de texte libre (projet professionnel, précisions "Autre")
5. MAL LIRE les dates dans les petites cases → bien lire chaque chiffre

═══════════════════════════════════════════════════════════════════
FORMAT DE SORTIE JSON
═══════════════════════════════════════════════════════════════════

Réponds UNIQUEMENT en JSON structuré avec les sections imbriquées :
{
    "numero_formulaire": "DEMO-2026-05",
    "etat_civil": {
        "civilite": "M." | "Mme" | "Ne souhaite pas répondre" | null,
        "nom_naissance": "...", "nom_usage": "..." | null,
        "prenoms": "...", "date_naissance": "YYYY-MM-DD",
        "lieu_naissance": "...",
        "nationalite": "Française" | "Union Européenne" | "Hors UE" | null,
        "nationalite_autre": "..." | null
    },
    "coordonnees": {
        "adresse": "...", "complement_adresse": "..." | null,
        "code_postal": "XXXXX", "ville": "...",
        "telephone_fixe": "..." | null, "telephone_mobile": "...",
        "email": "...",
        "mode_contact_prefere": "Courrier postal" | "Email" | "SMS" | "Téléphone" | null
    },
    "situation_professionnelle": {
        "statut": "..." | null, "employeur": "..." | null,
        "fonction": "..." | null, "secteur_activite": "..." | null,
        "categorie_sociopro": "..." | null,
        "experience": "0-2 ans" | "3-5 ans" | "6-10 ans" | "10+ ans" | null
    },
    "niveau_etudes": {
        "dernier_diplome": "..." | null,
        "intitule_diplome": "..." | null,
        "annee_obtention": "..." | null,
        "etablissement": "..." | null
    },
    "formation_souhaitee": {
        "domaines": ["...", "..."],
        "domaine_autre_precision": "..." | null,
        "niveau_vise": "..." | null, "modalite": "..." | null,
        "disponibilite": "..." | null, "date_debut": "..." | null
    },
    "financement": {
        "types_financement": ["...", "..."],
        "financement_autre_precision": "..." | null,
        "statut_accord": "..." | null
    },
    "competences": {
        "programmation": "1"-"5" | "N/A" | null,
        "bases_de_donnees": ..., "cloud": ..., "machine_learning": ...,
        "devops": ..., "gestion_projet": ..., "anglais_technique": ...
    },
    "motivations": {
        "objectifs": ["...", "..."],
        "projet_professionnel": "..." | null
    },
    "situation_particuliere": {
        "situations": ["...", "..."],
        "besoin_amenagements": true | false | null,
        "amenagements_details": "..." | null
    },
    "engagements": {
        "exactitude_informations": true | false,
        "reglement_interieur": true | false,
        "conditions_generales": true | false,
        "autorisation_contact": true | false,
        "accepte_autres_formations": true | false,
        "lieu_signature": "..." | null,
        "date_signature": "YYYY-MM-DD" | null,
        "signature_presente": true | false
    }
}"""
