#!/usr/bin/env python3
"""
Combined script to start SGLang servers and FastAPI server for content moderation.
This script is a convenience for starting all components in one command.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

# Add src_deploy to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config, reload_config
from sglang_servers import start_llm_server, start_embedding_server
from main import run_server

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--llm-mem-fraction", type=float, help="Memory fraction for LLM server"
    )
    parser.add_argument(
        "--embedding-mem-fraction",
        type=float,
        help="Memory fraction for embedding server",
    )
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
    if args.llm_mem_fraction:
        os.environ["LLM_MEM_FRACTION"] = str(args.llm_mem_fraction)
    if args.embedding_mem_fraction:
        os.environ["EMBEDDING_MEM_FRACTION"] = str(args.embedding_mem_fraction)
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

        # Set default memory fractions if not already set
        if "LLM_MEM_FRACTION" not in os.environ:
            os.environ["LLM_MEM_FRACTION"] = "0.70"
        if "EMBEDDING_MEM_FRACTION" not in os.environ:
            os.environ["EMBEDDING_MEM_FRACTION"] = "0.40"

        llm_mem_fraction = os.environ["LLM_MEM_FRACTION"]
        embedding_mem_fraction = os.environ["EMBEDDING_MEM_FRACTION"]

        # Determine which servers to start
        start_llm = not args.embedding_only
        start_embedding = not args.llm_only

        # Update environment variables for the FastAPI server to use the servers
        if not os.environ.get("LLM_URL") and start_llm:
            llm_host = os.environ.get("LLM_HOST", "0.0.0.0")
            llm_port = os.environ.get("LLM_PORT", "8899")
            os.environ["LLM_URL"] = f"http://{llm_host}:{llm_port}/v1"
            logger.info(f"Setting LLM_URL to {os.environ['LLM_URL']}")

        if not os.environ.get("EMBEDDING_URL") and start_embedding:
            embedding_host = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
            embedding_port = os.environ.get("EMBEDDING_PORT", "8890")
            os.environ["EMBEDDING_URL"] = f"http://{embedding_host}:{embedding_port}/v1"
            logger.info(f"Setting EMBEDDING_URL to {os.environ['EMBEDDING_URL']}")

        # Start LLM server first
        if start_llm:
            llm_model = os.environ.get("LLM_MODEL", "microsoft/Phi-3.5-mini-instruct")
            llm_port = int(os.environ.get("LLM_PORT", "8899"))
            api_key = os.environ.get("SGLANG_API_KEY", "None")

            print(
                f"[STARTUP] Starting LLM server with memory fraction {llm_mem_fraction}..."
            )
            llm_process = start_llm_server(
                llm_model, llm_port, api_key, llm_mem_fraction
            )

            # Wait for LLM server to start and check if it's running
            print("[STARTUP] Waiting for LLM server to initialize...")
            wait_time = 30  # seconds - increased wait time for model loading
            time.sleep(wait_time)

            if llm_process and llm_process.poll() is not None:
                print(
                    f"[ERROR] LLM server process exited with code {llm_process.returncode}"
                )
                sys.exit(1)

            print("[STARTUP] LLM server process started")

        # Now start embedding server
        if start_embedding:
            embedding_model = os.environ.get(
                "EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            )
            embedding_port = int(os.environ.get("EMBEDDING_PORT", "8890"))
            api_key = os.environ.get("SGLANG_API_KEY", "None")

            print(
                f"[STARTUP] Starting embedding server with memory fraction {embedding_mem_fraction}..."
            )
            embedding_process = start_embedding_server(
                embedding_model, embedding_port, api_key, embedding_mem_fraction
            )

            # Wait for embedding server to start and check if it's running
            print("[STARTUP] Waiting for embedding server to initialize...")
            wait_time = 180  # seconds - increased wait time for model loading
            time.sleep(wait_time)

            if embedding_process and embedding_process.poll() is not None:
                print(
                    f"[ERROR] Embedding server process exited with code {embedding_process.returncode}"
                )
                sys.exit(1)

            print("[STARTUP] Embedding server process started")

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
