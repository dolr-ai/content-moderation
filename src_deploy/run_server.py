#!/usr/bin/env python3
"""
Run script for the moderation server
"""

import os
import sys
from pathlib import Path
import argparse

# Try to load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load from .env file if it exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    print("python-dotenv not installed. Environment variables must be set manually.")

# Add src_deploy to path
sys.path.append(str(Path(__file__).parent.parent))

from src_deploy.main import run_server
from src_deploy.config import config, reload_config

if __name__ == "__main__":
    # Parse command line arguments (these will override environment variables)
    parser = argparse.ArgumentParser(description="Start the moderation server")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--gcp-credentials-file",
        help="Path to GCP credentials JSON file (will be read and set as GCP_CREDENTIALS)",
    )
    parser.add_argument("--prompt", help="Path to prompts file (local file)")
    parser.add_argument("--bucket", help="GCS bucket name")
    parser.add_argument(
        "--gcs-embeddings-path", help="Path to embeddings in GCS bucket"
    )
    parser.add_argument("--gcs-prompt-path", help="Path to prompts file in GCS bucket")
    parser.add_argument("--env-file", help="Path to .env file")

    args = parser.parse_args()

    # Load from custom .env file if specified
    if args.env_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=args.env_file)
            print(f"Loaded environment from {args.env_file}")
            # Reload config after loading .env
            reload_config()
        except ImportError:
            print("python-dotenv not installed. Cannot load custom .env file.")

    # Load GCP credentials from file if specified
    if args.gcp_credentials_file:
        try:
            with open(args.gcp_credentials_file, "r") as f:
                gcp_credentials = f.read().strip()
            os.environ["GCP_CREDENTIALS"] = gcp_credentials
            print(f"Loaded GCP credentials from {args.gcp_credentials_file}")
        except Exception as e:
            print(f"Error loading GCP credentials from file: {e}")

    # Override environment variables with command line arguments
    if args.host:
        os.environ["SERVER_HOST"] = args.host
    if args.port:
        os.environ["SERVER_PORT"] = str(args.port)
    if args.reload:
        os.environ["RELOAD"] = "true"
    if args.debug:
        os.environ["DEBUG"] = "true"
    if args.prompt:
        os.environ["PROMPT_PATH"] = args.prompt
    if args.bucket:
        os.environ["GCS_BUCKET"] = args.bucket
    if args.gcs_embeddings_path:
        os.environ["GCS_EMBEDDINGS_PATH"] = args.gcs_embeddings_path
    if args.gcs_prompt_path:
        os.environ["GCS_PROMPT_PATH"] = args.gcs_prompt_path

    # Reload config after applying command line arguments
    if any(
        [
            args.host,
            args.port,
            args.reload,
            args.debug,
            args.gcp_credentials_file,
            args.prompt,
            args.bucket,
            args.gcs_embeddings_path,
            args.gcs_prompt_path,
        ]
    ):
        reload_config()

    # Print configuration for debugging
    if config.debug:
        print("Configuration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

    # Run the server with the configuration
    run_server(
        host=config.host,
        port=config.port,
        reload=config.reload,
        debug=config.debug,
    )
