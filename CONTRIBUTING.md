# Contribuer au projet

Merci de votre intérêt pour ce projet ! Voici comment contribuer.

## Signaler un bug

1. Vérifiez que le bug n'a pas déjà été signalé dans les [Issues](../../issues)
2. Créez une nouvelle issue avec :
   - Description claire du problème
   - Étapes pour reproduire
   - Comportement attendu vs observé
   - Version de Python et des dépendances

## Proposer une amélioration

1. Ouvrez une issue pour discuter de votre idée
2. Attendez la validation avant de commencer le développement

## Soumettre du code

### Prérequis

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages)
- [just](https://github.com/casey/just) (task runner, optionnel)

### Setup

```bash
# Cloner votre fork
git clone https://github.com/VOTRE_USERNAME/kyc-document-processing.git
cd kyc-document-processing

# Installer les dépendances
uv sync

# Vérifier que tout fonctionne
just check
```

### Workflow

1. Créez une branche depuis `main` :
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. Faites vos modifications en suivant les conventions de code

3. Vérifiez votre code :
   ```bash
   just check  # Formate, lint et tests
   ```

4. Committez avec des messages clairs :
   ```bash
   git commit -m "feat: ajouter extraction de nouveau document"
   ```

5. Poussez et créez une Pull Request

### Conventions de code

- **Formatage** : Black (ligne max 100 caractères)
- **Linting** : Ruff
- **Types** : Annotations de type obligatoires pour les fonctions publiques
- **Tests** : Couverture minimale de 80%
- **Docstrings** : Format Google pour les fonctions publiques

### Structure des commits

Utilisez les [Conventional Commits](https://www.conventionalcommits.org/) :

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation
- `refactor:` refactoring sans changement de comportement
- `test:` ajout ou modification de tests
- `chore:` maintenance (dépendances, CI, etc.)

## Questions ?

Ouvrez une issue avec le label `question`.
