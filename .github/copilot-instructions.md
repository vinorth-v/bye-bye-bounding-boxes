# Pratiques de Code - KYC Document Processing

## Raison d'être

Ce fichier recense les pratiques de code que nous suivons pour ce projet. Il vise à orienter développeurs comme Agents IA.

Les pratiques peuvent être :
- **[obligatoire]** : elles doivent être suivies
- **[fortement-conseillée]** : elles doivent être suivies sauf exception
- **[recommandée]** : il est mieux de les suivre

---

## Catégorie : Validation de code

**[obligatoire]** : La code base ne doit pas produire d'erreur lors de l'exécution de `uv run ruff check src/`

**[obligatoire]** : La code base ne doit pas produire d'erreur lors de l'exécution de `uv run ruff format --check src/`

**[obligatoire]** : La code base ne doit pas produire d'erreur lors de l'exécution de `uv run pytest tests/`

**[obligatoire]** : La longueur maximale d'une ligne est de 120 caractères

**[obligatoire]** : Exécuter `uv run ruff check --fix src/` avant de commiter pour corriger automatiquement les problèmes

**[obligatoire]** : Exécuter `uv run ruff format src/` avant de commiter pour formatter le code

**[fortement-conseillée]** : La couverture de test doit être d'au moins 80%

**[obligatoire]** : Les tests doivent s'exécuter en moins de 30 secondes pour garantir un feedback rapide

---

## Catégorie : Principes généraux

**[obligatoire]** : Les traitements doivent être idempotents (rejouer deux fois ne doit pas poser problème)

**[obligatoire]** : Les traitements doivent être tolérants à la panne (capable de rejouer un fichier spécifique)

**[obligatoire]** : Minimiser les dépendances externes

**[fortement-conseillée]** : Privilégier le profilage et l'optimisation du code plutôt que l'ajout de ressources (CPU/RAM)

---

## Catégorie : Architecture

**[obligatoire]** : Le projet suit une architecture modulaire avec la structure suivante :
- `src/chains/` : Contient les chaînes LLM et la logique de traitement
- `src/chains/schemas/` : Contient les schémas Pydantic (spécifications des données)
- `src/utils/` : Contient les utilitaires de configuration
- `config/` : Contient les fichiers de configuration
- `templates/` : Contient les templates de prompts
- `examples/` : Contient les exemples d'utilisation

**[fortement-conseillée]** : Les schémas Pydantic dans `schemas/` servent de spécifications et doivent être définis avant l'implémentation

**[obligatoire]** : Utiliser des imports absolus depuis la racine du projet (ex: `from chains.schemas.schema import SchemaRAD`)

**[obligatoire]** : Ne jamais utiliser d'imports relatifs (ex: `from ..schemas import ...`)

**[obligatoire]** : Grouper les imports en 3 sections séparées par une ligne vide : (1) stdlib, (2) third-party, (3) modules internes

**[obligatoire]** : Les imports entre modules doivent respecter les règles suivantes :
- `schemas/` ne doit pas importer d'autres modules métier
- `chains/` peut importer `schemas/`, `utils/`, `config/`
- `utils/` peut importer `config/` uniquement
- `pipeline.py` et `main.py` peuvent importer tous les modules

---

## Catégorie : Nommage

**[obligatoire]** : Les noms de fichiers doivent être en snake_case

**[obligatoire]** : Les schémas doivent être préfixés par `Schema` (ex: `SchemaCNI`, `SchemaPasseport`)

**[fortement-conseillée]** : Les fonctions de transformation doivent avoir des noms descriptifs (ex: `extract_taxpayer_info`, `validate_form_fields`)

**[obligatoire]** : Les constantes de configuration doivent être en UPPER_SNAKE_CASE

---

## Catégorie : Types de données

**[obligatoire]** : Les signatures de fonctions doivent avoir des type-hints (types de paramètres et type de retour)

**[fortement-conseillée]** : Utiliser les types built-in (`list`, `dict`, `set`, `tuple`) au lieu de `typing.List`, etc.

**[fortement-conseillée]** : Utiliser l'opérateur union `|` (ex: `str | None`) au lieu de `Optional[str]`

**[obligatoire]** : Utiliser Pydantic pour définir les structures de données complexes

**[fortement-conseillée]** : Les schémas Pydantic doivent inclure des descriptions via `Field(description=...)`

**[obligatoire]** : Les énumérations doivent hériter de `str, Enum` pour la sérialisation JSON

**[recommandée]** : Utiliser des alias de types pour améliorer la clarté (ex: `UserId = str`)

---

## Catégorie : Gestion des schémas (Spec-Driven Development)

**[obligatoire]** : Les schémas Pydantic dans `chains/schemas/` sont la source de vérité pour les structures de données

**[obligatoire]** : Chaque type de document KYC doit avoir son propre schéma (CNI, Passeport, Permis, etc.)

**[fortement-conseillée]** : Les schémas doivent être validés avec des exemples de données réelles

**[obligatoire]** : Le dictionnaire `LAD_SCHEMAS` doit mapper chaque `DocumentClass` à son schéma correspondant

**[fortement-conseillée]** : Les modifications de schémas doivent être accompagnées de tests de validation

---

## Catégorie : LLM et Prompts

**[obligatoire]** : Les prompts doivent être définis dans `chains/prompts.py`

**[fortement-conseillée]** : Les prompts doivent inclure des exemples de sortie attendue

**[obligatoire]** : Utiliser le retry parser pour gérer les erreurs de parsing LLM

**[fortement-conseillée]** : Les configurations LLM (température, modèle) doivent être dans `chains/configuration.py`

---

## Catégorie : Gestion des erreurs

**[obligatoire]** : Ne pas produire de code défensif. Se protéger uniquement contre les problèmes déjà rencontrés

**[obligatoire]** : Ne pas utiliser `.get()` sur les dictionnaires dont on est certain de la structure - utiliser `[]` pour faire échouer rapidement en cas de problème

**[fortement-conseillée]** : Utiliser `raise ValueError` pour les erreurs de validation de paramètres ou de données

**[fortement-conseillée]** : Ne pas attraper d'exceptions génériques (`except Exception`), préférer des exceptions spécifiques

**[recommandée]** : Laisser les exceptions remonter naturellement plutôt que de retourner `None` en cas d'erreur

**[recommandée]** : Implémenter des retry avec backoff exponentiel pour les erreurs transitoires (appels LLM, etc.)

---

## Catégorie : Style de code

**[obligatoire]** : Utiliser des f-strings pour le formatage de chaînes

**[fortement-conseillée]** : Les fonctions de traitement doivent être pures et ne pas avoir d'effets de bord quand c'est possible

**[recommandée]** : Ne pas produire de docstrings sauf si la fonction est complexe et nécessite une explication détaillée

**[fortement-conseillée]** : Éviter les commentaires redondants - si le nom de la fonction/variable est explicite, ne pas ajouter de commentaire

**[fortement-conseillée]** : Préférer du code auto-documenté (noms explicites) plutôt que des commentaires explicatifs

**[recommandée]** : Préférer des fonctions courtes et focalisées sur une seule responsabilité

**[recommandée]** : Favoriser les list / dict comprehensions

**[obligatoire]** : Utiliser async/await pour les opérations I/O (lecture de fichiers, appels API)

---

## Catégorie : Tests

**[obligatoire]** : Utiliser pytest pour écrire les tests

**[obligatoire]** : Structurer les tests avec Given/When/Then dans les commentaires

**[obligatoire]** : Les schémas Pydantic doivent être testés avec des exemples de données valides et invalides

**[fortement-conseillée]** : Nommer les tests selon le pattern `test_<nom_fonction>_<cas_teste>`

**[recommandée]** : Utiliser des fixtures pytest et un fichier `conftest.py` à la racine des tests

**[recommandée]** : Tester les transformations de schémas (résolution de `$ref`, génération JSON Schema)

**[fortement-conseillée]** : Les tests doivent valider que les schémas correspondent aux vrais documents KYC

---

## Catégorie : Configuration

**[obligatoire]** : Les secrets ne doivent JAMAIS être en dur dans le code

**[obligatoire]** : Les secrets doivent être récupérés via variables d'environnement

**[fortement-conseillée]** : Les configurations doivent être dans `config/config.json` ou gérées par `utils/config.py`

**[obligatoire]** : Les configurations sensibles (API keys, credentials) doivent utiliser les variables d'environnement

---

## Catégorie : Documentation

**[recommandée]** : Utiliser le style Google pour les docstrings (https://google.github.io/styleguide/pyguide.html)

**[obligatoire]** : Chaque schéma doit avoir une docstring décrivant son objectif

**[fortement-conseillée]** : Les champs Pydantic doivent avoir des descriptions via `Field(description=...)`

**[recommandée]** : Documenter les mappings entre les champs du formulaire et les champs du schéma

**[fortement-conseillée]** : Maintenir des exemples de données dans `examples/`

**[obligatoire]** : Le README doit expliquer comment utiliser les schémas et les chaînes LLM

**[recommandée]** : Documenter les Args, Returns et Raises dans les docstrings de fonctions complexes

---

## Catégorie : Pipeline et Processing

**[obligatoire]** : Les pipelines doivent être définis dans `pipeline.py`

**[fortement-conseillée]** : Chaque étape du pipeline doit être une fonction testable indépendamment

**[obligatoire]** : Les erreurs de parsing doivent être loggées avec le contexte du document

**[fortement-conseillée]** : Utiliser des types structurés (Pydantic) en sortie de chaque étape

---

## Catégorie : Vertex AI et Google Cloud

**[obligatoire]** : Les configurations Vertex AI doivent être dans `chains/configuration.py`

**[fortement-conseillée]** : Utiliser les modèles Document AI pour l'extraction de formulaires quand disponible

**[obligatoire]** : Les credentials Google Cloud doivent être gérées via les variables d'environnement standards

---

## Catégorie : CI/CD

**[obligatoire]** : La CI doit exécuter `uv run ruff check src/`

**[obligatoire]** : La CI doit exécuter `uv run ruff format --check src/`

**[obligatoire]** : La CI doit exécuter `uv run pytest tests/`

**[fortement-conseillée]** : Utiliser `uv` pour la gestion des dépendances dans la CI

**[obligatoire]** : Les jobs CI doivent échouer si les validations ne passent pas

---

## Catégorie : Logging et Observabilité

**[obligatoire]** : Utiliser le module `logging` standard de Python, pas `print()`

**[fortement-conseillée]** : Logger les étapes importantes du pipeline avec le niveau approprié (INFO, WARNING, ERROR)

**[obligatoire]** : Logger les erreurs LLM avec le contexte complet (document ID, tentative, erreur)

**[recommandée]** : Utiliser des structured logs (JSON) pour faciliter l'analyse

**[fortement-conseillée]** : Logger les métriques de performance (temps d'exécution, tokens utilisés)

---

## Catégorie : Sécurité

**[obligatoire]** : Ne jamais logger de données sensibles (contenu de documents, PII)

**[obligatoire]** : Valider et sanitizer les entrées utilisateur avant traitement

**[fortement-conseillée]** : Utiliser des secrets managers pour les credentials (pas de fichiers .env en production)

**[recommandée]** : Implémenter des limites de rate limiting pour les appels LLM

---

## Catégorie : Performance

**[fortement-conseillée]** : Utiliser le streaming pour traiter les gros fichiers (éviter de tout charger en mémoire)

**[recommandée]** : Mettre en cache les résultats LLM coûteux quand approprié

**[obligatoire]** : Éviter les boucles imbriquées avec opérations coûteuses (préférer vectorisation ou batch)

**[fortement-conseillée]** : Profiler le code avant d'optimiser (utiliser `cProfile` ou `py-spy`)

**[recommandée]** : Utiliser `asyncio.gather()` pour paralléliser les appels LLM indépendants

---

## Catégorie : Gestion des dépendances

**[obligatoire]** : Utiliser `uv` pour la gestion des packages

**[obligatoire]** : Ne jamais exécuter `uv add <pkg>` sans réfléchir - ajouter uniquement les dépendances nécessaires

**[obligatoire]** : Fixer les versions des dépendances critiques dans `pyproject.toml`

**[fortement-conseillée]** : Utiliser `uv lock` pour générer un fichier de lock reproductible

**[obligatoire]** : Après ajout/modification de dépendances, exécuter `uv sync` pour mettre à jour l'environnement

**[obligatoire]** : Commiter `pyproject.toml` et `uv.lock` ensemble après modification des dépendances

**[recommandée]** : Documenter pourquoi chaque dépendance est nécessaire

**[obligatoire]** : Supprimer les dépendances inutilisées dès qu'elles sont identifiées

**[obligatoire]** : Auditer régulièrement les dépendances pour les vulnérabilités de sécurité

---

## Catégorie : Qualité des données

**[obligatoire]** : Valider les schémas de sortie avec Pydantic (pas de validation manuelle)

**[fortement-conseillée]** : Définir des contraintes de validation dans les schémas (`Field(gt=0)`, `constr(pattern=...)`)

**[recommandée]** : Créer des fixtures de test avec des cas limites réalistes

**[obligatoire]** : Rejeter rapidement les données invalides (fail-fast) plutôt que d'essayer de les réparer

---

## Catégorie : Code Pythonique

**[recommandée]** : Utiliser les context managers (`with`) pour la gestion de ressources

**[recommandée]** : Préférer `pathlib.Path` à `os.path` pour la manipulation de chemins

**[fortement-conseillée]** : Utiliser les dataclasses ou Pydantic pour les structures de données, pas les dicts

**[recommandée]** : Utiliser `enumerate()` plutôt que `range(len())`

**[recommandée]** : Utiliser des generators pour les grandes séquences de données

**[fortement-conseillée]** : Éviter les variables globales mutables

---

## Catégorie : Versioning et Git

**[obligatoire]** : Les messages de commit doivent suivre le format Conventional Commits (`feat:`, `fix:`, `docs:`, `refacto:`, `chore:`, `test:`, `perf:`, `ci:` etc.)

**[obligatoire]** : Format des commits : `<type>(<scope>): <description>` (ex: `feat(rad): add RAD classification schema`)

**[obligatoire]** : Utiliser des noms de branches conventionnels : `feat/<scope>/<description>`, `fix/<scope>/<description>`, `refactor/<scope>/<description>`

**[fortement-conseillée]** : Chaque commit doit passer les validations (ruff, pytest)

**[recommandée]** : Garder les commits atomiques et focalisés sur un seul changement

**[obligatoire]** : Ne jamais commiter de secrets ou credentials

**[recommandée]** : Utiliser des branches de feature pour les développements (`feat/`, `fix/`)

**[obligatoire]** : Supprimer le code mort ou commenté au lieu de le garder

**[recommandée]** : Les merge requests doivent être de taille raisonnable avec un objectif clair

---

## Catégorie : Revue de code

**[fortement-conseillée]** : Le code doit être review avant merge (pull request obligatoire)

**[recommandée]** : Les reviews doivent vérifier la conformité aux pratiques de code

**[obligatoire]** : Les tests doivent passer en CI avant merge

**[fortement-conseillée]** : Documenter les décisions architecturales importantes (ADR)
