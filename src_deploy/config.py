"""
Configuration for the moderation server using environment variables
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration for the moderation server using environment variables"""

    def __init__(self):
        """
        Initialize configuration settings from environment variables
        """
        # Set default values
        self.data_root = Path("/app/data")  # Docker default path

        # Load environment variables or use defaults

        # Path settings
        self.data_root = Path(os.environ.get("DATA_ROOT", str(self.data_root)))

        # GCP credentials (as a JSON string)
        self.gcp_credentials = os.environ.get("GCP_CREDENTIALS")

        # GCS settings for embeddings
        self.gcs_bucket = os.environ.get("GCS_BUCKET")
        self.gcs_embeddings_path = os.environ.get(
            "GCS_EMBEDDINGS_PATH", "rag/gcp-embeddings.jsonl"
        )

        # GCS prompt path
        self.gcs_prompt_path = os.environ.get(
            "GCS_PROMPT_PATH", "rag/moderation_prompts.yml"
        )

        # Path for prompts (local fallback)
        self.prompt_path = Path(
            os.environ.get(
                "PROMPT_PATH",
                str(self.data_root / "prompts" / "moderation_prompts.yml"),
            )
        )

        # Server settings
        self.host = os.environ.get("SERVER_HOST", "0.0.0.0")
        self.port = int(os.environ.get("SERVER_PORT", "8080"))
        self.debug = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")
        self.reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")

        # BigQuery settings
        self.bq_project = os.environ.get("BQ_PROJECT", "stage-test-tables")
        self.bq_dataset = os.environ.get("BQ_DATASET", "stage_test_tables")
        self.bq_table = os.environ.get("BQ_TABLE", "test_comment_mod_embeddings")
        self.bq_top_k = int(os.environ.get("BQ_TOP_K", "5"))
        self.bq_distance_type = os.environ.get("BQ_DISTANCE_TYPE", "COSINE")
        self.bq_options = os.environ.get(
            "BQ_OPTIONS", '{"fraction_lists_to_search": 0.1, "use_brute_force": false}'
        )

        # LLM settings (for future use)
        self.llm_url = os.environ.get("LLM_URL")
        self.embedding_url = os.environ.get("EMBEDDING_URL")
        self.api_key = os.environ.get("API_KEY")

        # Application settings
        self.max_input_length = int(os.environ.get("MAX_INPUT_LENGTH", "2000"))
        self.max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for debugging
        Returns:
            Dictionary with configuration values
        """
        return {
            "data_root": str(self.data_root),
            "gcp_credentials": (
                "[CREDENTIALS AVAILABLE]" if self.gcp_credentials else None
            ),
            "prompt_path": str(self.prompt_path),
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "reload": self.reload,
            "bq_project": self.bq_project,
            "bq_dataset": self.bq_dataset,
            "bq_table": self.bq_table,
            "bq_top_k": self.bq_top_k,
            "bq_distance_type": self.bq_distance_type,
            "gcs_bucket": self.gcs_bucket,
            "gcs_embeddings_path": self.gcs_embeddings_path,
            "gcs_prompt_path": self.gcs_prompt_path,
            "max_input_length": self.max_input_length,
            "max_new_tokens": self.max_new_tokens,
        }


# Create default config instance
config = Config()


def reload_config() -> Config:
    """
    Reload configuration from environment variables
    Returns:
        Updated Config instance
    """
    global config
    config = Config()
    return config
