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

from config import config, reload_config
from servers.server_sglang import start_llm_server, start_embedding_server
from utils.check_gpu import do_all_gpu_checks
from servers.server_fastapi import run_server

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start content moderation system with all components"
    )

    # No-wait flag
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start SGLang servers without waiting indefinitely"
    )

    # Server settings
    parser.add_argument(
        "--host",
        help=f"Host for the FastAPI server (default: {config.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port for the FastAPI server (default: {config.port})",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for FastAPI"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # GCP settings
    parser.add_argument(
        "--gcp-credentials-file",
        help="Path to GCP credentials JSON file (will be read and set as GCP_CREDENTIALS)",
    )
    parser.add_argument(
        "--bucket", help=f"GCS bucket name (default: {config.gcs_bucket})"
    )
    parser.add_argument(
        "--gcs-embeddings-path",
        help=f"Path to embeddings in GCS bucket (default: {config.gcs_embeddings_path})",
    )
    parser.add_argument(
        "--gcs-prompt-path",
        help=f"Path to prompts file in GCS bucket (default: {config.gcs_prompt_path})",
    )
    parser.add_argument(
        "--prompt", help=f"Path to local prompts file (default: {config.prompt_path})"
    )

    # LLM server settings
    parser.add_argument(
        "--llm-model",
        help=f"Model to use for LLM (default: {config.llm_model})",
    )
    parser.add_argument(
        "--llm-host",
        help=f"Host for the LLM server (default: {config.llm_host})",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        help=f"Port for the LLM server (default: {config.llm_port})",
    )

    # Embedding server settings
    parser.add_argument(
        "--embedding-model",
        help=f"Model to use for embeddings (default: {config.embedding_model})",
    )
    parser.add_argument(
        "--embedding-host",
        help=f"Host for the embedding server (default: {config.embedding_host})",
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        help=f"Port for the embedding server (default: {config.embedding_port})",
    )

    # General SGLang settings
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--llm-mem-fraction",
        type=float,
        help=f"Memory fraction for LLM server (default: {config.llm_mem_fraction})",
    )
    parser.add_argument(
        "--embedding-mem-fraction",
        type=float,
        help=f"Memory fraction for embedding server (default: {config.embedding_mem_fraction})",
    )
    parser.add_argument(
        "--max-requests",
        help=f"Max running requests for SGLang servers (default: {config.max_requests})",
    )

    # Control flags
    parser.add_argument(
        "--llm-only", action="store_true", help="Start only the LLM server"
    )
    parser.add_argument(
        "--embedding-only", action="store_true", help="Start only the embedding server"
    )

    return parser.parse_args()


def setup_environment(args):
    """
    Set up environment variables based on command line arguments
    This will override config values for the current run
    """
    # Server settings
    if args.host:
        os.environ["SERVER_HOST"] = args.host
        config.host = args.host
    if args.port:
        os.environ["SERVER_PORT"] = str(args.port)
        config.port = args.port
    if args.reload:
        os.environ["RELOAD"] = "true"
        config.reload = True
    if args.debug:
        os.environ["DEBUG"] = "true"
        config.debug = True

    # GCP settings
    if args.gcp_credentials_file:
        try:
            with open(args.gcp_credentials_file, "r") as f:
                gcp_credentials = f.read().strip()
            os.environ["GCP_CREDENTIALS"] = gcp_credentials
            config.gcp_credentials = gcp_credentials
            logger.info(f"Loaded GCP credentials from {args.gcp_credentials_file}")
        except Exception as e:
            logger.error(f"Error loading GCP credentials from file: {e}")

    if args.bucket:
        os.environ["GCS_BUCKET"] = args.bucket
        config.gcs_bucket = args.bucket
    if args.gcs_embeddings_path:
        os.environ["GCS_EMBEDDINGS_PATH"] = args.gcs_embeddings_path
        config.gcs_embeddings_path = args.gcs_embeddings_path
    if args.gcs_prompt_path:
        os.environ["GCS_PROMPT_PATH"] = args.gcs_prompt_path
        config.gcs_prompt_path = args.gcs_prompt_path
    if args.prompt:
        os.environ["PROMPT_PATH"] = args.prompt
        config.prompt_path = args.prompt

    # LLM server settings
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
        config.llm_model = args.llm_model
    if args.llm_host:
        os.environ["LLM_HOST"] = args.llm_host
        config.llm_host = args.llm_host
    if args.llm_port:
        os.environ["LLM_PORT"] = str(args.llm_port)
        config.llm_port = args.llm_port

    # Embedding server settings
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
        config.embedding_model = args.embedding_model
    if args.embedding_host:
        os.environ["EMBEDDING_HOST"] = args.embedding_host
        config.embedding_host = args.embedding_host
    if args.embedding_port:
        os.environ["EMBEDDING_PORT"] = str(args.embedding_port)
        config.embedding_port = args.embedding_port

    # General SGLang settings
    if args.api_key:
        os.environ["SGLANG_API_KEY"] = args.api_key
        os.environ["API_KEY"] = args.api_key
        config.sglang_api_key = args.api_key
        config.api_key = args.api_key
    if args.llm_mem_fraction:
        os.environ["LLM_MEM_FRACTION"] = str(args.llm_mem_fraction)
        config.llm_mem_fraction = args.llm_mem_fraction
    if args.embedding_mem_fraction:
        os.environ["EMBEDDING_MEM_FRACTION"] = str(args.embedding_mem_fraction)
        config.embedding_mem_fraction = args.embedding_mem_fraction
    if args.max_requests:
        os.environ["MAX_REQUESTS"] = args.max_requests
        config.max_requests = int(args.max_requests)

    # Update URL environment variables based on host/port settings
    embedding_host = config.embedding_host
    embedding_port = config.embedding_port
    llm_host = config.llm_host
    llm_port = config.llm_port

    # These are the URLs that the FastAPI service will use to connect to the SGLang servers
    config.embedding_url = f"http://{embedding_host}:{embedding_port}/v1"
    config.llm_url = f"http://{llm_host}:{llm_port}/v1"

    os.environ["EMBEDDING_URL"] = config.embedding_url
    os.environ["LLM_URL"] = config.llm_url

    logger.info(f"EMBEDDING_URL: {config.embedding_url}")
    logger.info(f"LLM_URL: {config.llm_url}")


def start_servers(args):
    """Start the SGLang servers (LLM and embedding)"""
    llm_process = None
    embedding_process = None

    logger.info("\n[STARTUP] Starting SGLang servers...")

    # Use values from centralized config
    llm_mem_fraction = config.llm_mem_fraction
    embedding_mem_fraction = config.embedding_mem_fraction

    # Determine which servers to start
    start_llm = not args.embedding_only
    start_embedding = not args.llm_only

    # Start LLM server first
    if start_llm:
        llm_model = config.llm_model
        llm_port = config.llm_port
        api_key = config.sglang_api_key

        logger.info(
            f"[STARTUP] Starting LLM server with memory fraction {llm_mem_fraction}..."
        )
        llm_process = start_llm_server(llm_model, llm_port, api_key, llm_mem_fraction)

        # Wait for LLM server to start and check if it's running
        logger.info("[STARTUP] Waiting for LLM server to initialize...")
        wait_time = config.llm_init_wait_time
        time.sleep(wait_time)

        if llm_process and llm_process.poll() is not None:
            logger.error(
                f"[ERROR] LLM server process exited with code {llm_process.returncode}"
            )
            sys.exit(1)

        logger.info("[STARTUP] LLM server process started")

    # Now start embedding server
    if start_embedding:
        embedding_model = config.embedding_model
        embedding_port = config.embedding_port
        api_key = config.sglang_api_key

        logger.info(
            f"[STARTUP] Starting embedding server with memory fraction {embedding_mem_fraction}..."
        )
        embedding_process = start_embedding_server(
            embedding_model, embedding_port, api_key, embedding_mem_fraction
        )

        # Wait for embedding server to start and check if it's running
        logger.info("[STARTUP] Waiting for embedding server to initialize...")
        wait_time = config.embedding_init_wait_time
        time.sleep(wait_time)

        if embedding_process and embedding_process.poll() is not None:
            logger.error(
                f"[ERROR] Embedding server process exited with code {embedding_process.returncode}"
            )
            sys.exit(1)

        logger.info("[STARTUP] Embedding server process started")

    # After starting servers, verify that URLs are set correctly
    logger.info("[STARTUP] Verifying server URLs...")
    if start_llm and not config.llm_url:
        logger.error("[ERROR] LLM_URL not set properly!")
        sys.exit(1)

    if start_embedding and not config.embedding_url:
        logger.error("[ERROR] EMBEDDING_URL not set properly!")
        sys.exit(1)

    logger.info("[STARTUP] SGLang servers started successfully")
    logger.info(f"LLM URL: {config.llm_url}")
    logger.info(f"EMBEDDING URL: {config.embedding_url}")

    return llm_process, embedding_process


def check_gpu():
    """Run GPU checks and log results"""
    result = do_all_gpu_checks()
    logger.info("--------------------------------")
    logger.info(f"GPU checks result: {result}")
    logger.info("--------------------------------")
    return result


def start_fastapi_server(config):
    """Start the FastAPI server"""
    logger.info("\n[STARTUP] Starting FastAPI server...")

    # Make sure GCP credentials are available to the FastAPI server
    if "GCP_CREDENTIALS" in os.environ and os.environ["GCP_CREDENTIALS"]:
        logger.info("GCP credentials will be used by FastAPI server")
        # Ensure the credentials are properly set in config
        config.gcp_credentials = os.environ["GCP_CREDENTIALS"]
    else:
        logger.warning("GCP_CREDENTIALS not found in environment variables")

    # Print configuration for debugging
    if config.debug:
        logger.info("\n[CONFIG] Current configuration:")
        for key, value in config.to_dict().items():
            # Skip large credentials value
            if key == "gcp_credentials" and value:
                logger.info(f"  {key}: [CREDENTIALS AVAILABLE]")
            else:
                logger.info(f"  {key}: {value}")

    # Reload config to ensure latest environment variables are used
    reload_config()

    run_server(
        host=config.host,
        port=config.port,
        reload=config.reload,
        debug=config.debug,
    )


def main():
    """Main function to orchestrate server startup"""
    # Check GPU status
    check_gpu()

    # Parse command line arguments
    args = parse_arguments()

    # Set up environment
    setup_environment(args)

    # Print startup banner
    logger.info("=" * 80)
    logger.info("CONTENT MODERATION SYSTEM STARTUP")
    logger.info("=" * 80)

    # Start SGLang servers
    llm_process, embedding_process = start_servers(args)

    # Check if we should wait for SGLang servers indefinitely
    # This allows the startup script to continue to the FastAPI server
    if args.no_wait:
        logger.info("Started SGLang servers, exiting without waiting (--no-wait flag)")
        return

    # Otherwise stay running to keep the SGLang servers alive
    logger.info("Started SGLang servers, waiting indefinitely...")
    while True:
        # Check if processes are still running
        if llm_process and llm_process.poll() is not None:
            logger.error(
                f"LLM server process exited with code {llm_process.returncode}"
            )
            break
        if embedding_process and embedding_process.poll() is not None:
            logger.error(
                f"Embedding server process exited with code {embedding_process.returncode}"
            )
            break
        time.sleep(10)  # Check every 10 seconds

    # Start FastAPI server
    # start_fastapi_server()


if __name__ == "__main__":
    main()
