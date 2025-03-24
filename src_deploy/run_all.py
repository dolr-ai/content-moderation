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
from utils.check_gpu import do_all_gpu_checks
from main import run_server

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # Server settings
    "SERVER_HOST": "0.0.0.0",
    "SERVER_PORT": "8080",
    "RELOAD": "false",
    "DEBUG": "false",
    # LLM server settings
    "LLM_MODEL": "microsoft/Phi-3.5-mini-instruct",
    "LLM_HOST": "0.0.0.0",
    "LLM_PORT": "8899",
    "LLM_MEM_FRACTION": "0.70",
    # Embedding server settings
    "EMBEDDING_MODEL": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "EMBEDDING_HOST": "0.0.0.0",
    "EMBEDDING_PORT": "8890",
    "EMBEDDING_MEM_FRACTION": "0.70",
    # General SGLang settings
    "SGLANG_API_KEY": "None",
    "API_KEY": "None",
    "MAX_REQUESTS": "32",
    # Wait times
    "LLM_INIT_WAIT_TIME": "180",
    "EMBEDDING_INIT_WAIT_TIME": "180",
}


def setup_default_env():
    """Set default environment variables if not already set"""
    for key, value in DEFAULT_CONFIG.items():
        if key not in os.environ:
            os.environ[key] = value


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start content moderation system with all components"
    )

    # Server settings
    parser.add_argument(
        "--host",
        help=f"Host for the FastAPI server (default: {DEFAULT_CONFIG['SERVER_HOST']})",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port for the FastAPI server (default: {DEFAULT_CONFIG['SERVER_PORT']})",
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
    parser.add_argument("--bucket", help="GCS bucket name")
    parser.add_argument(
        "--gcs-embeddings-path", help="Path to embeddings in GCS bucket"
    )
    parser.add_argument("--gcs-prompt-path", help="Path to prompts file in GCS bucket")
    parser.add_argument("--prompt", help="Path to local prompts file")

    # LLM server settings
    parser.add_argument(
        "--llm-model",
        help=f"Model to use for LLM (default: {DEFAULT_CONFIG['LLM_MODEL']})",
    )
    parser.add_argument(
        "--llm-host",
        help=f"Host for the LLM server (default: {DEFAULT_CONFIG['LLM_HOST']})",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        help=f"Port for the LLM server (default: {DEFAULT_CONFIG['LLM_PORT']})",
    )

    # Embedding server settings
    parser.add_argument(
        "--embedding-model",
        help=f"Model to use for embeddings (default: {DEFAULT_CONFIG['EMBEDDING_MODEL']})",
    )
    parser.add_argument(
        "--embedding-host",
        help=f"Host for the embedding server (default: {DEFAULT_CONFIG['EMBEDDING_HOST']})",
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        help=f"Port for the embedding server (default: {DEFAULT_CONFIG['EMBEDDING_PORT']})",
    )

    # General SGLang settings
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--llm-mem-fraction",
        type=float,
        help=f"Memory fraction for LLM server (default: {DEFAULT_CONFIG['LLM_MEM_FRACTION']})",
    )
    parser.add_argument(
        "--embedding-mem-fraction",
        type=float,
        help=f"Memory fraction for embedding server (default: {DEFAULT_CONFIG['EMBEDDING_MEM_FRACTION']})",
    )
    parser.add_argument(
        "--max-requests",
        help=f"Max running requests for SGLang servers (default: {DEFAULT_CONFIG['MAX_REQUESTS']})",
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
    """Set up environment variables based on command line arguments"""
    # Set default environment variables first
    setup_default_env()

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
            logger.info(f"Loaded GCP credentials from {args.gcp_credentials_file}")
        except Exception as e:
            logger.error(f"Error loading GCP credentials from file: {e}")

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


def start_servers(args):
    """Start the SGLang servers (LLM and embedding)"""
    llm_process = None
    embedding_process = None

    logger.info("\n[STARTUP] Starting SGLang servers...")

    # Use environment variables set earlier
    llm_mem_fraction = float(os.environ["LLM_MEM_FRACTION"])
    embedding_mem_fraction = float(os.environ["EMBEDDING_MEM_FRACTION"])

    # Determine which servers to start
    start_llm = not args.embedding_only
    start_embedding = not args.llm_only

    # Update environment variables for the FastAPI server to use the servers
    if not os.environ.get("LLM_URL") and start_llm:
        llm_host = os.environ["LLM_HOST"]
        llm_port = os.environ["LLM_PORT"]
        os.environ["LLM_URL"] = f"http://{llm_host}:{llm_port}/v1"
        logger.info(f"Setting LLM_URL to {os.environ['LLM_URL']}")

    if not os.environ.get("EMBEDDING_URL") and start_embedding:
        embedding_host = os.environ["EMBEDDING_HOST"]
        embedding_port = os.environ["EMBEDDING_PORT"]
        os.environ["EMBEDDING_URL"] = f"http://{embedding_host}:{embedding_port}/v1"
        logger.info(f"Setting EMBEDDING_URL to {os.environ['EMBEDDING_URL']}")

    # Start LLM server first
    if start_llm:
        llm_model = os.environ["LLM_MODEL"]
        llm_port = int(os.environ["LLM_PORT"])
        api_key = os.environ["SGLANG_API_KEY"]

        logger.info(
            f"[STARTUP] Starting LLM server with memory fraction {llm_mem_fraction}..."
        )
        llm_process = start_llm_server(llm_model, llm_port, api_key, llm_mem_fraction)

        # Wait for LLM server to start and check if it's running
        logger.info("[STARTUP] Waiting for LLM server to initialize...")
        wait_time = int(
            os.environ.get("LLM_INIT_WAIT_TIME", DEFAULT_CONFIG["LLM_INIT_WAIT_TIME"])
        )
        time.sleep(wait_time)

        if llm_process and llm_process.poll() is not None:
            logger.error(
                f"[ERROR] LLM server process exited with code {llm_process.returncode}"
            )
            sys.exit(1)

        logger.info("[STARTUP] LLM server process started")

    # Now start embedding server
    if start_embedding:
        embedding_model = os.environ["EMBEDDING_MODEL"]
        embedding_port = int(os.environ["EMBEDDING_PORT"])
        api_key = os.environ["SGLANG_API_KEY"]

        logger.info(
            f"[STARTUP] Starting embedding server with memory fraction {embedding_mem_fraction}..."
        )
        embedding_process = start_embedding_server(
            embedding_model, embedding_port, api_key, embedding_mem_fraction
        )

        # Wait for embedding server to start and check if it's running
        logger.info("[STARTUP] Waiting for embedding server to initialize...")
        wait_time = int(
            os.environ.get(
                "EMBEDDING_INIT_WAIT_TIME", DEFAULT_CONFIG["EMBEDDING_INIT_WAIT_TIME"]
            )
        )
        time.sleep(wait_time)

        if embedding_process and embedding_process.poll() is not None:
            logger.error(
                f"[ERROR] Embedding server process exited with code {embedding_process.returncode}"
            )
            sys.exit(1)

        logger.info("[STARTUP] Embedding server process started")

    logger.info("[STARTUP] SGLang servers started successfully")

    return llm_process, embedding_process


def check_gpu():
    """Run GPU checks and log results"""
    result = do_all_gpu_checks()
    logger.info("--------------------------------")
    logger.info(f"GPU checks result: {result}")
    logger.info("--------------------------------")
    return result


def start_fastapi_server():
    """Start the FastAPI server"""
    logger.info("\n[STARTUP] Starting FastAPI server...")

    # Print configuration for debugging
    if config.debug:
        logger.info("\n[CONFIG] Current configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")

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
    start_servers(args)

    # Start FastAPI server
    start_fastapi_server()


if __name__ == "__main__":
    main()
