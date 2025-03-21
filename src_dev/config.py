
"""
Configuration manager for content moderation system
"""
import os
import yaml
from pathlib import Path
from huggingface_hub import login as hf_login


class Config:
    """Configuration manager for content moderation system"""

    def __init__(self, config_path=None):
        """
        Initialize configuration

        Args:
            config_path: Path to configuration file (default: look for dev_config.yml)
        """
        if config_path is None:
            # Try to find config in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "dev_config.yml"

        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                print("config loaded from ", config_path)
                # Login to Hugging Face
                hf_login(config["tokens"]["HF_TOKEN"])
                print("logged in to Hugging Face")
                return config
        except Exception as e:
            print(f"Could not load config file: {e}")
            return {}

    def get(self, *keys, default=None):
        """
        Get configuration value

        Args:
            *keys: Key path to configuration value
            default: Default value if key path doesn't exist

        Returns:
            Configuration value
        """
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def project_root(self):
        """Get project root path"""
        return Path(self.get("local", "PROJECT_ROOT", default="."))

    @property
    def data_root(self):
        """Get data root path"""
        return Path(self.get("local", "DATA_ROOT", default="./data"))

    @property
    def hf_token(self):
        """Get Hugging Face token"""
        return self.get("tokens", "HF_TOKEN")

    @property
    def openai_api_key(self):
        """Get OpenAI API key"""
        return self.get("tokens", "OPENAI_API_KEY")


# Initialize global configuration
config = Config()
