#!/usr/bin/env python3
"""
Run script for the moderation server
"""

import os
import sys
from pathlib import Path
from src_deploy.main import run_server

if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    # Default configurations
    default_config = str(project_root / "dev_config.yml")
    default_prompt = str(project_root / "prompts" / "moderation_prompts.yml")
    default_embeddings = str(project_root / "data" / "rag" / "gcp-embeddings.jsonl")

    # Run the server with default configuration
    run_server(
        host="0.0.0.0",
        port=8080,
        config_path=os.environ.get("CONFIG_PATH", default_config),
        reload=True,  # Enable reload for development
        prompt_path=os.environ.get("PROMPT_PATH", default_prompt),
        embeddings_file=os.environ.get("EMBEDDINGS_FILE", default_embeddings),
    )
