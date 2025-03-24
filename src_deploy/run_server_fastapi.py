#!/usr/bin/env python3
"""
Run script for the moderation server
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add src_deploy to path
sys.path.append(str(Path(__file__).parent.parent))

from main import run_server
from config import config, reload_config

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

    # Add arguments for LLM and embedding services
    parser.add_argument("--llm-url", help="URL for the LLM service")
    parser.add_argument("--embedding-url", help="URL for the embedding service")
    parser.add_argument("--api-key", help="API key for the services")
    parser.add_argument(
        "--skip-sglang", action="store_true", help="Skip starting SGLang servers"
    )

    args = parser.parse_args()

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
    if args.llm_url:
        os.environ["LLM_URL"] = args.llm_url
    if args.embedding_url:
        os.environ["EMBEDDING_URL"] = args.embedding_url
    if args.api_key:
        os.environ["API_KEY"] = args.api_key

    # Reload config after applying command line arguments
    reload_config()

    # Start SGLang servers first if not skipped
    if not args.skip_sglang:
        # Start both LLM and embedding servers
        from sglang_servers import start_sglang_servers

        # Start servers and wait for them to be ready
        llm_process, embedding_process = start_sglang_servers()

        # Wait for servers to be ready
        print("Waiting for SGLang servers to be ready...")
        time.sleep(5)  # Give some time for the servers to start

        # Check if servers are still running
        if llm_process and embedding_process:
            if llm_process.poll() is not None:
                print(f"LLM server process exited with code {llm_process.returncode}")
                sys.exit(1)
            if embedding_process.poll() is not None:
                print(
                    f"Embedding server process exited with code {embedding_process.returncode}"
                )
                sys.exit(1)
            print("SGLang servers started successfully")
        else:
            print("Failed to start SGLang servers")
            sys.exit(1)

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
