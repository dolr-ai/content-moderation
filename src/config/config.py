#!/usr/bin/env python3
"""
Configuration module for content moderation system.
Handles loading configuration from YAML files and environment variables.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the content moderation system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.
                         If None, tries to find it in standard locations.
        """
        self.config_data = {}

        # Try to find config file if not specified
        if config_path is None:
            possible_paths = [
                os.environ.get("CONFIG_PATH"),
                "./dev_config.yml",
                "../dev_config.yml",
                str(Path.home() / "work/yral/content-moderation/dev_config.yml"),
            ]

            for path in possible_paths:
                if path and os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            print(
                f"Warning: No configuration file found. Using environment variables only."
            )

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.
        """
        try:
            with open(config_path, "r") as f:
                self.config_data = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section (e.g., 'local', 'tokens')
            key: Configuration key
            default: Default value if not found

        Returns:
            The configuration value or default if not found
        """
        # First check environment variables (they take precedence)
        env_var = f"{section.upper()}_{key.upper()}"
        if env_var in os.environ:
            return os.environ[env_var]

        # Then check the loaded config
        if section in self.config_data and key in self.config_data[section]:
            return self.config_data[section][key]

        return default

    def get_hf_token(self) -> Optional[str]:
        """Get the Hugging Face token."""
        return self.get("tokens", "HF_TOKEN")

    def get_openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key."""
        return self.get("tokens", "OPENAI_API_KEY")

    def get_project_root(self) -> str:
        """Get the project root directory."""
        return self.get("local", "PROJECT_ROOT", os.getcwd())

    def get_data_root(self) -> str:
        """Get the data root directory."""
        return self.get(
            "local", "DATA_ROOT", os.path.join(self.get_project_root(), "data")
        )


# Create a singleton instance
config = Config()
