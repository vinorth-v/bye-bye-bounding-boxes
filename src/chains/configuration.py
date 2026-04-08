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
            raise ValueError(
                "Variable d'environnement manquante : VAR_LLM_REGION ou GCP_LOCATION"
            )
        return value

    @property
    def model(self) -> str:
        """Modèle Vertex AI à utiliser."""
        return os.getenv("VAR_LLM_MODELE", os.getenv("MODEL_NAME", "gemini-1.5-flash-002"))

    @property
    def temperature(self) -> float:
        """Température pour la génération."""
        return float(os.getenv("VAR_LLM_TEMPERATURE", os.getenv("TEMPERATURE", "0.2")))

    @property
    def max_output_tokens(self) -> int:
        """Nombre maximum de tokens en sortie."""
        return int(os.getenv("VAR_LLM_MAX_OUTPUT_TOKEN", "4096"))

    # Token pricing (USD per 1M tokens) - Gemini 2.5 Flash
    INPUT_TOKEN_PRICE_PER_MILLION: float = 0.15
    OUTPUT_TOKEN_PRICE_PER_MILLION: float = 0.60

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
