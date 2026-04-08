# Justfile pour le démonstrateur KYC

set dotenv-load := true
set shell := ["zsh", "-uc"]

# 📋 Affiche la liste des recettes disponibles
default:
    just --list

# 🌍 Affiche les variables d'environnement utilisées
[group('debug')]
env:
    @echo "GCP_PROJECT_ID:          {{env('GCP_PROJECT_ID', '')}}"
    @echo "GCP_LOCATION:            {{env('GCP_LOCATION', '')}}"
    @echo "MODEL_NAME:              {{env('MODEL_NAME', '')}}"
    @echo "VAR_LLM_MODELE:          {{env('VAR_LLM_MODELE', '')}}"
    @echo "VAR_LLM_TEMPERATURE:     {{env('VAR_LLM_TEMPERATURE', '')}}"

# 🧹 Nettoie les fichiers temporaires
[group('debug')]
clean:
    rm -rf __pycache__
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".DS_Store" -delete

# 📦 Installe les dépendances
[group('uv')]
sync:
    uv sync

# ✨ Formatte le code avec ruff
[group('uv')]
format:
    uv run ruff format .

# 🔍 Vérifie le code avec ruff
[group('uv')]
lint:
    uv run ruff check .

# 🔧 Fix les problèmes de lint
[group('uv')]
lint-fix:
    uv run ruff check --fix .

# 🧹 Formatte + fix lint en une commande
[group('uv')]
ruff:
    uv run ruff format .
    uv run ruff check --fix .

# 🧪 Exécute les tests
[group('uv')]
test:
    PYTHONPATH=src uv run pytest tests/ -v

# ✅ Formatte, fix lint et lance les tests
[group('validation')]
check: ruff test
    @echo "✅ Toutes les validations sont passées !"

# 🪪 Traite une carte nationale d'identité
[group('demo')]
cni:
    PYTHONPATH=src uv run python src/main.py examples/cni.webp

# 🛂 Traite un passeport
[group('demo')]
passeport:
    PYTHONPATH=src uv run python src/main.py examples/passeport.webp

# 🚗 Traite le permis recto/verso 
[group('demo')]
permis:
    PYTHONPATH=src uv run python src/main.py examples/permis_recto_verso.pdf

# 🏦 Traite un RIB
[group('demo')]
rib:
    PYTHONPATH=src uv run python src/main.py examples/rib.png

# ⚡ Traite une facture EDF
[group('demo')]
edf:
    PYTHONPATH=src uv run python src/main.py examples/jdom_edf.pdf

# 🏛️ Traite un avis d'impôts
[group('demo')]
impots:
    PYTHONPATH=src uv run python src/main.py examples/jdom_impots.pdf

# 📂 Traite tous les documents du dossier examples/
[group('demo')]
dossier:
    PYTHONPATH=src uv run python src/main.py --folder examples/

# 🎯 Traite un document personnalisé
[group('demo')]
run path:
    PYTHONPATH=src uv run python src/main.py {{path}}