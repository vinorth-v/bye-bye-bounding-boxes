# Bye Bye Bounding Boxes 👋🏾

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Démonstrateur de traitement documentaire avec les LLM multimodaux. Cas d'usage : KYC bancaire (CNI, passeport, permis, RIB, justificatifs).
Le projet montre qu'on peut faire en 2 jours ce qui prenait 6 mois en deep learning classique : sans dataset annoté, sans entraînement, sans bounding boxes.
D'où le nom du repo.

---

📖 **Article associé** : [De 6 mois à 2 jours : la révolution LLM pour le traitement documentaire](https://blog.octo.com/de-6-mois-a-2-jours--la-revolution-llm-pour-le-traitement-documentaire) sur le blog OCTO  
🎤 **Talk** : présenté à La Grosse Conf 2026, Caen.Tech 2026 et au Comptoir OCTO ([replay YouTube](https://youtu.be/EiCVoxKuQ0E))

---

## Pourquoi ce projet ?

### Le problème avec le deep learning classique

Pour créer un système de classification et extraction de documents KYC avec deep learning classique :

- **Plusieurs mois de développement** : collecte de données, annotation, entraînement de modèles CNN, post-processing OCR
- **Coûts élevés** : infrastructure GPU, équipe ML, data labelers
- **Résultats limités** : ne fonctionne que sur les formats appris, nécessite ré-entraînement pour chaque variation

### La solution avec les LLM multimodaux

- **Quelques heures de développement** : prompts + schémas Pydantic + règles métier
- **Coûts minimes** : API calls, zéro infrastructure d'entraînement
- **Résultats supérieurs** : fonctionne out-of-the-box sur nouveaux formats, robuste aux variations

## Fonctionnalités

### Documents traités

1. **Pièces d'identité**
   - Carte Nationale d'Identité (CNI)
   - Passeport
   - Permis de conduire (avec détection des catégories cochées)

2. **Justificatifs de domicile**
   - Factures (électricité, gaz, eau, internet, téléphone)
   - Quittances de loyer
   - Taxes
   - Attestations d'assurance

3. **Coordonnées bancaires**
   - RIB/IBAN avec validation checksum modulo 97

### Règles métier

- Validation de cohérence entre documents (nom, prénom)
- Vérification des dates d'expiration
- Contrôle de l'ancienneté (justificatif < 3 mois)
- Validation technique IBAN (checksum)
- Détection visuelle des cases cochées (permis de conduire)

## Installation

### Prérequis

- Python 3.10+
- Accès Google Cloud avec Vertex AI activé
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages)
- [just](https://github.com/casey/just) (task runner, optionnel)

### Setup

```bash
# Cloner le projet
git clone https://github.com/vinorth-v/bye-bye-bounding-boxes.git
cd bye-bye-bounding-boxes

# Installer les dépendances
uv sync

# Configuration
cp .env.example .env
# Éditer .env avec vos credentials

# Exporter les credentials Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Utilisation

### Document individuel

```bash
uv run python src/main.py path/to/document.pdf
```

### Dossier complet

```bash
uv run python src/main.py --folder path/to/folder/
```

### En Python

```python
from chains.llm_chain import KYCDocumentChain
from pipeline import KYCPipeline

# Un document
chain = KYCDocumentChain()
result = chain.process_document("document.jpg")

# Dossier complet
pipeline = KYCPipeline()
dossier = pipeline.process_folder("dossier_client/")
```

### Commandes just (optionnel)

```bash
just --list  # Affiche toutes les commandes disponibles
just check   # Formate, lint et tests
just test    # Lance les tests
```

## Architecture

```
bye-bye-bounding-boxes/
├── src/
│   ├── chains/
│   │   ├── schemas/
│   │   │   └── kyc_schemas.py      # Schémas Pydantic pour chaque doc
│   │   ├── configuration.py         # Config Google Cloud / Vertex AI
│   │   ├── llm_chain.py            # Chain LLM principale
│   │   └── prompts.py              # Prompts pour classification/extraction
│   ├── utils/
│   │   └── config.py               # Utilitaires de configuration
│   ├── pipeline.py                 # Pipeline multi-documents
│   └── main.py                     # Point d'entrée
├── tests/
│   └── test_schemas.py             # Tests unitaires
└── config/
    └── config.json                 # Configuration du projet
```

## Exemples de code

### Classification automatique

```python
chain = KYCDocumentChain()
result = chain.classify_document("document.jpg")
# -> "carte_identite" avec 98% de confiance
```

### Extraction structurée

```python
cni = chain.extract_cni("cni.jpg")
print(f"{cni.prenom} {cni.nom}")
print(f"Valide: {cni.est_valide}")
```

### Détection des cases cochées (permis)

```python
permis = chain.extract_permis("permis.jpg")
print(permis.categories)
# -> ["B", "A2"]
```

### Validation IBAN

```python
rib = chain.extract_rib("rib.jpg")
print(f"IBAN: {rib.iban}")
print(f"Checksum valide: {rib.iban_valide}")
```

## Tests

```bash
uv run pytest tests/ -v
```

## Contribuer

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les détails.

## Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## Ressources

- [Vertex AI Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview)
- [Pydantic](https://docs.pydantic.dev/)
- [Google Document AI](https://cloud.google.com/document-ai)
