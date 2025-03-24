#!/usr/bin/env python3
"""
Combined script to start SGLang servers and FastAPI server for content moderation.
This script is a convenience for starting all components in one command.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src_deploy to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config, reload_config
from sglang_servers import start_sglang_servers
from main import run_server

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Start content moderation system with all components"
    )

    # Server settings
    parser.add_argument("--host", help="Host for the FastAPI server")
    parser.add_argument("--port", type=int, help="Port for the FastAPI server")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for FastAPI"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # GCP settings
    parser.add_argument(
        "--gcp-credentials-file",
        help="Path to GCP credentials JSON file (will be read and set as GCP_CREDENTIALS)",
    )
    parser.add_argument("--bucket", help="GCS bucket name")
    parser.add_argument(
        "--gcs-embeddings-path", help="Path to embeddings in GCS bucket"
    )
    parser.add_argument("--gcs-prompt-path", help="Path to prompts file in GCS bucket")
    parser.add_argument("--prompt", help="Path to local prompts file")

    # LLM server settings
    parser.add_argument("--llm-model", help="Model to use for LLM")
    parser.add_argument("--llm-host", help="Host for the LLM server")
    parser.add_argument("--llm-port", type=int, help="Port for the LLM server")

    # Embedding server settings
    parser.add_argument("--embedding-model", help="Model to use for embeddings")
    parser.add_argument("--embedding-host", help="Host for the embedding server")
    parser.add_argument(
        "--embedding-port", type=int, help="Port for the embedding server"
    )

    # General SGLang settings
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--mem-fraction", help="Memory fraction for SGLang servers")
    parser.add_argument(
        "--max-requests", help="Max running requests for SGLang servers"
    )

    # Control flags
    parser.add_argument(
        "--skip-sglang", action="store_true", help="Skip starting SGLang servers"
    )
    parser.add_argument(
        "--llm-only", action="store_true", help="Start only the LLM server"
    )
    parser.add_argument(
        "--embedding-only", action="store_true", help="Start only the embedding server"
    )

    args = parser.parse_args()

    # Process arguments and set environment variables

    # Server settings
    if args.host:
        os.environ["SERVER_HOST"] = args.host
    if args.port:
        os.environ["SERVER_PORT"] = str(args.port)
    if args.reload:
        os.environ["RELOAD"] = "true"
    if args.debug:
        os.environ["DEBUG"] = "true"

    # GCP settings
    if args.gcp_credentials_file:
        try:
            with open(args.gcp_credentials_file, "r") as f:
                gcp_credentials = f.read().strip()
            os.environ["GCP_CREDENTIALS"] = gcp_credentials
            print(f"Loaded GCP credentials from {args.gcp_credentials_file}")
        except Exception as e:
            print(f"Error loading GCP credentials from file: {e}")

    if args.bucket:
        os.environ["GCS_BUCKET"] = args.bucket
    if args.gcs_embeddings_path:
        os.environ["GCS_EMBEDDINGS_PATH"] = args.gcs_embeddings_path
    if args.gcs_prompt_path:
        os.environ["GCS_PROMPT_PATH"] = args.gcs_prompt_path
    if args.prompt:
        os.environ["PROMPT_PATH"] = args.prompt

    # LLM server settings
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_host:
        os.environ["LLM_HOST"] = args.llm_host
    if args.llm_port:
        os.environ["LLM_PORT"] = str(args.llm_port)

    # Embedding server settings
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    if args.embedding_host:
        os.environ["EMBEDDING_HOST"] = args.embedding_host
    if args.embedding_port:
        os.environ["EMBEDDING_PORT"] = str(args.embedding_port)

    # General SGLang settings
    if args.api_key:
        os.environ["SGLANG_API_KEY"] = args.api_key
        os.environ["API_KEY"] = args.api_key
    if args.mem_fraction:
        os.environ["MEM_FRACTION"] = args.mem_fraction
    if args.max_requests:
        os.environ["MAX_REQUESTS"] = args.max_requests

    # Reload config after applying command line arguments
    reload_config()

    # Print startup banner
    print("=" * 80)
    print("CONTENT MODERATION SYSTEM STARTUP")
    print("=" * 80)

    # Start SGLang servers if not skipped
    llm_process = None
    embedding_process = None

    if not args.skip_sglang:
        print("\n[STARTUP] Starting SGLang servers...")

        # Determine which servers to start
        start_llm = not args.embedding_only
        start_embedding = not args.llm_only

        if start_llm and start_embedding:
            # Start both servers
            llm_process, embedding_process = start_sglang_servers()
        elif start_llm:
            # Start only LLM server
            from sglang_servers import start_llm_server

            llm_process = start_llm_server()
            print("[STARTUP] Started LLM server only (embedding server skipped)")
        elif start_embedding:
            # Start only embedding server
            from sglang_servers import start_embedding_server

            embedding_process = start_embedding_server()
            print("[STARTUP] Started embedding server only (LLM server skipped)")

        # Wait for servers to be ready
        print("[STARTUP] Waiting for SGLang servers to be ready...")
        time.sleep(5)  # Give some time for the servers to start

        # Check if servers are still running
        if start_llm and llm_process and llm_process.poll() is not None:
            print(
                f"[ERROR] LLM server process exited with code {llm_process.returncode}"
            )
            sys.exit(1)
        if (
            start_embedding
            and embedding_process
            and embedding_process.poll() is not None
        ):
            print(
                f"[ERROR] Embedding server process exited with code {embedding_process.returncode}"
            )
            sys.exit(1)

        print("[STARTUP] SGLang servers started successfully")
    else:
        print("[STARTUP] Skipping SGLang servers (--skip-sglang flag set)")

    # Print configuration for debugging
    if config.debug:
        print("\n[CONFIG] Current configuration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

    # Start the FastAPI server
    print("\n[STARTUP] Starting FastAPI server...")
    run_server(
        host=config.host,
        port=config.port,
        reload=config.reload,
        debug=config.debug,
    )
