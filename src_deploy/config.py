import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class Config:
    """Configuration for the moderation server"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration settings
        Args:
            config_path: Optional path to config YAML file
        """
        self.config_values: Dict[str, Any] = {}
        self.config_path = config_path

        # Set default values
        self.project_root = Path(__file__).parent.parent
        self.data_root = self.project_root / "data"
        self.gcp_credentials_path = None
        self.embeddings_file = "rag/gcp-embeddings.jsonl"
        self.prompt_path = self.project_root / "prompts" / "moderation_prompts.yml"

        # BigQuery settings
        self.bq_project = "stage-test-tables"
        self.bq_dataset = "stage_test_tables"
        self.bq_table = "test_comment_mod_embeddings"
        self.bq_top_k = 5
        self.bq_distance_type = "COSINE"
        self.bq_options = '{"fraction_lists_to_search": 0.1, "use_brute_force": false}'

        # Load config file if provided
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file
        Args:
            config_path: Path to config YAML file
        """
        try:
            with open(config_path, "r") as f:
                self.config_values = yaml.safe_load(f)

            # Update values from config
            if "local" in self.config_values:
                local_config = self.config_values["local"]
                if "PROJECT_ROOT" in local_config:
                    self.project_root = Path(local_config["PROJECT_ROOT"])
                if "DATA_ROOT" in local_config:
                    self.data_root = Path(local_config["DATA_ROOT"])

            # Load GCP credentials path
            if (
                "secrets" in self.config_values
                and "GCP_CREDENTIALS_PATH" in self.config_values["secrets"]
            ):
                self.gcp_credentials_path = Path(
                    self.config_values["secrets"]["GCP_CREDENTIALS_PATH"]
                )

        except Exception as e:
            print(f"Error loading config: {e}")


# Create default config instance
config = Config()


def init_config(config_path: Optional[str] = None) -> Config:
    """
    Initialize configuration with custom path
    Args:
        config_path: Path to config YAML file
    Returns:
        Config instance
    """
    global config
    config = Config(config_path)
    return config
