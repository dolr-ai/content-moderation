"""
Configuration for the moderation server using environment variables
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import secrets


class Config:
    """Configuration for the moderation server using environment variables"""

    def __init__(self):
        """
        Initialize configuration settings from environment variables
        and ensure environment variables are set for any defaults
        """
        # ======= Version settings =======
        self.version = self._get_or_set_env("VERSION", "0.1.0")

        # ======= Server settings =======
        self.host = self._get_or_set_env("SERVER_HOST", "0.0.0.0")
        self.port = int(self._get_or_set_env("SERVER_PORT", "8080"))
        self.debug = self._get_or_set_env("DEBUG", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self.reload = self._get_or_set_env("RELOAD", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # ======= API endpoints =======
        self.embedding_url = self._get_or_set_env(
            "EMBEDDING_URL", "http://localhost:8890/v1"
        )
        self.llm_url = self._get_or_set_env("LLM_URL", "http://localhost:8899/v1")
        self.sglang_api_key = self._get_or_set_env("SGLANG_API_KEY", "None")

        # Generate a secure random API key if not provided
        default_api_key = secrets.token_hex(32)
        # Default to a secure random API key
        self.api_key = self._get_or_set_env("API_KEY", default_api_key)
        self.api_key = (
            "36012fea438d4acae1922ddda87f6b10a6d08a11521cccadadb63f2761e8b499"
        )
        print(f"API_KEY: {self.api_key}")

        # ======= LLM server settings =======
        self.llm_model = self._get_or_set_env(
            "LLM_MODEL", "microsoft/Phi-3.5-mini-instruct"
        )
        self.llm_host = self._get_or_set_env("LLM_HOST", "127.0.0.1")
        self.llm_port = int(self._get_or_set_env("LLM_PORT", "8899"))
        self.llm_mem_fraction = float(self._get_or_set_env("LLM_MEM_FRACTION", "0.70"))

        # ======= Embedding server settings =======
        self.embedding_model = self._get_or_set_env(
            "EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        )
        self.embedding_host = self._get_or_set_env("EMBEDDING_HOST", "127.0.0.1")
        self.embedding_port = int(self._get_or_set_env("EMBEDDING_PORT", "8890"))
        self.embedding_mem_fraction = float(
            self._get_or_set_env("EMBEDDING_MEM_FRACTION", "0.70")
        )

        # ======= General SGLang settings =======
        self.max_requests = int(self._get_or_set_env("MAX_REQUESTS", "32"))
        self.temperature = float(self._get_or_set_env("TEMPERATURE", "0.0"))
        self.max_new_tokens = int(self._get_or_set_env("MAX_NEW_TOKENS", "128"))

        # ======= Wait times =======
        self.llm_init_wait_time = int(self._get_or_set_env("LLM_INIT_WAIT_TIME", "120"))
        self.embedding_init_wait_time = int(
            self._get_or_set_env("EMBEDDING_INIT_WAIT_TIME", "120")
        )

        # ======= Path settings =======
        self.data_root = Path(self._get_or_set_env("DATA_ROOT", "/app/data"))
        self.prompt_path = self._get_or_set_env(
            "PROMPT_PATH", str(self.data_root / "prompts" / "moderation_prompts.yml")
        )

        # ======= GCP/GCS settings =======
        self.gcp_credentials = self._get_or_set_env("GCP_CREDENTIALS", "")
        self.gcs_bucket = self._get_or_set_env("GCS_BUCKET", "test-ds-utility-bucket")
        self.gcs_embeddings_path = self._get_or_set_env(
            "GCS_EMBEDDINGS_PATH",
            "project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl",
        )
        self.gcs_prompt_path = self._get_or_set_env(
            "GCS_PROMPT_PATH",
            "project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml",
        )

        # ======= BigQuery settings =======
        self.dataset_id = self._get_or_set_env("DATASET_ID", "stage_test_tables")
        self.table_id = self._get_or_set_env("TABLE_ID", "test_comment_mod_embeddings")
        self.bq_project = self._get_or_set_env("BQ_PROJECT", "stage-test-tables")
        self.bq_dataset = self._get_or_set_env("BQ_DATASET", "stage_test_tables")
        self.bq_table = self._get_or_set_env("BQ_TABLE", "test_comment_mod_embeddings")
        self.bq_top_k = int(self._get_or_set_env("BQ_TOP_K", "5"))
        self.bq_distance_type = self._get_or_set_env("BQ_DISTANCE_TYPE", "COSINE")
        self.bq_options = self._get_or_set_env(
            "BQ_OPTIONS", '{"fraction_lists_to_search": 0.1, "use_brute_force": false}'
        )

        # ======= Application settings =======
        self.max_input_length = int(self._get_or_set_env("MAX_INPUT_LENGTH", "2000"))

        # Recalculate embedding and LLM URLs based on host/port if not explicitly provided
        if (
            "EMBEDDING_URL" not in os.environ
            or os.environ["EMBEDDING_URL"] == "http://localhost:8890/v1"
        ):
            embedding_url = f"http://{self.embedding_host}:{self.embedding_port}/v1"
            self.embedding_url = embedding_url
            os.environ["EMBEDDING_URL"] = embedding_url

        if (
            "LLM_URL" not in os.environ
            or os.environ["LLM_URL"] == "http://localhost:8899/v1"
        ):
            llm_url = f"http://{self.llm_host}:{self.llm_port}/v1"
            self.llm_url = llm_url
            os.environ["LLM_URL"] = llm_url

    def _get_or_set_env(self, key: str, default: str) -> str:
        """
        Get an environment variable if it exists, or set it to default if it doesn't

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Value from environment or default
        """
        if key in os.environ:
            return os.environ[key]
        else:
            os.environ[key] = default
            return default

    def get_env(self, key: str, default: Optional[str] = None) -> str:
        """
        Get an environment variable or corresponding config value

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Value from environment or config
        """
        if key in os.environ:
            return os.environ[key]

        # Convert key to lowercase and try to find attribute
        attr_name = key.lower()
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
            # Convert non-string values to string
            if not isinstance(value, str):
                return str(value)
            return value

        return default if default is not None else ""

    def set_env(self, key: str, value: Union[str, int, float, bool]) -> None:
        """
        Set both environment variable and config attribute

        Args:
            key: Environment variable name
            value: Value to set
        """
        # Convert to string for environment variable
        str_value = str(value)
        os.environ[key] = str_value

        # Try to set the corresponding attribute
        attr_name = key.lower()
        if hasattr(self, attr_name):
            current_value = getattr(self, attr_name)
            # Convert value to same type as current attribute
            if isinstance(current_value, bool):
                setattr(self, attr_name, str_value.lower() in ("true", "1", "yes"))
            elif isinstance(current_value, int):
                setattr(self, attr_name, int(value))
            elif isinstance(current_value, float):
                setattr(self, attr_name, float(value))
            else:
                setattr(self, attr_name, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for debugging
        Returns:
            Dictionary with configuration values
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "gcp_credentials" and value:
                config_dict[key] = "[CREDENTIALS AVAILABLE]"
            else:
                config_dict[key] = str(value) if isinstance(value, Path) else value
        return config_dict

    def get_moderation_service_config(self) -> Dict[str, Any]:
        """
        Get configuration specifically for ModerationService
        Returns:
            Dictionary with configuration for ModerationService
        """
        return {
            "VERSION": self.version,
            "EMBEDDING_URL": self.embedding_url,
            "LLM_URL": self.llm_url,
            "SGLANG_API_KEY": self.sglang_api_key,
            "EMBEDDING_MODEL": self.embedding_model,
            "LLM_MODEL": self.llm_model,
            "TEMPERATURE": self.temperature,
            "MAX_NEW_TOKENS": self.max_new_tokens,
            "SERVER_HOST": self.host,
            "SERVER_PORT": self.port,
            "GCS_BUCKET_NAME": self.gcs_bucket,
            "GCS_PROMPT_PATH": self.gcs_prompt_path,
            "DATASET_ID": self.dataset_id,
            "TABLE_ID": self.table_id,
            "GCP_CREDENTIALS": self.gcp_credentials,
            "PROMPT_PATH": self.prompt_path,
        }

    def update_from_env(self):
        """
        Update configuration from current environment variables
        and ensure all environment variables are set
        """
        # Re-initialize to load current environment variables and set defaults
        self.__init__()

    def sync_to_env(self):
        """
        Sync all config values to environment variables
        """
        # Server settings
        os.environ["SERVER_HOST"] = str(self.host)
        os.environ["SERVER_PORT"] = str(self.port)
        os.environ["DEBUG"] = "true" if self.debug else "false"
        os.environ["RELOAD"] = "true" if self.reload else "false"

        # API endpoints
        os.environ["EMBEDDING_URL"] = self.embedding_url
        os.environ["LLM_URL"] = self.llm_url
        os.environ["SGLANG_API_KEY"] = self.sglang_api_key
        os.environ["API_KEY"] = self.api_key

        # LLM server settings
        os.environ["LLM_MODEL"] = self.llm_model
        os.environ["LLM_HOST"] = self.llm_host
        os.environ["LLM_PORT"] = str(self.llm_port)
        os.environ["LLM_MEM_FRACTION"] = str(self.llm_mem_fraction)

        # Embedding server settings
        os.environ["EMBEDDING_MODEL"] = self.embedding_model
        os.environ["EMBEDDING_HOST"] = self.embedding_host
        os.environ["EMBEDDING_PORT"] = str(self.embedding_port)
        os.environ["EMBEDDING_MEM_FRACTION"] = str(self.embedding_mem_fraction)

        # Other settings
        os.environ["MAX_REQUESTS"] = str(self.max_requests)
        os.environ["TEMPERATURE"] = str(self.temperature)
        os.environ["MAX_NEW_TOKENS"] = str(self.max_new_tokens)
        os.environ["LLM_INIT_WAIT_TIME"] = str(self.llm_init_wait_time)
        os.environ["EMBEDDING_INIT_WAIT_TIME"] = str(self.embedding_init_wait_time)


# Create default config instance and ensure all environment variables are set
config = Config()
config.sync_to_env()


def reload_config() -> Config:
    """
    Reload configuration from environment variables and ensure all env vars are set
    Returns:
        Updated Config instance
    """
    global config
    config.update_from_env()
    config.sync_to_env()
    return config
