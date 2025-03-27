#!/usr/bin/env python3
"""
Entrypoint script for content moderation system.
Handles starting and coordinating all components of the system.
"""

import os
import sys
import argparse
import time
import logging
import subprocess
from pathlib import Path

# Import from the modules
from config import config, reload_config
from utils.check_gpu import do_all_gpu_checks

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Run GPU checks and log results"""
    result = do_all_gpu_checks()
    logger.info("--------------------------------")
    logger.info(f"GPU checks result: {result}")
    logger.info("--------------------------------")
    logger.info(f"API_KEY: {config.api_key}")
    return result


def parse_arguments():
    """Parse command line arguments for the entrypoint"""
    parser = argparse.ArgumentParser(description="Content Moderation System Entrypoint")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["all", "sglang", "fastapi", "test"],
        default="all",
        help="Mode to run: all (default), sglang (SGLang servers only), fastapi (FastAPI only), or test",
    )

    # No-wait flag for sglang servers
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start SGLang servers without waiting indefinitely",
    )

    # Add the missing arguments that are needed in start_sglang_servers.py
    parser.add_argument("--host", help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gcp-credentials-file", help="Path to GCP credentials file")
    parser.add_argument("--bucket", help="GCS bucket name")
    parser.add_argument("--gcs-embeddings-path", help="Path to embeddings in GCS")
    parser.add_argument("--gcs-prompt-path", help="Path to prompts in GCS")
    parser.add_argument("--prompt", help="Path to prompt file")
    parser.add_argument("--llm-model", help="LLM model to use")
    parser.add_argument("--llm-host", help="LLM host")
    parser.add_argument("--llm-port", type=int, help="LLM port")
    parser.add_argument("--embedding-model", help="Embedding model to use")
    parser.add_argument("--embedding-host", help="Embedding host")
    parser.add_argument("--embedding-port", type=int, help="Embedding port")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--llm-mem-fraction", type=float, help="LLM memory fraction")
    parser.add_argument(
        "--embedding-mem-fraction", type=float, help="Embedding memory fraction"
    )
    parser.add_argument("--max-requests", type=int, help="Maximum concurrent requests")
    parser.add_argument("--llm-only", action="store_true", help="Start only LLM server")
    parser.add_argument(
        "--embedding-only", action="store_true", help="Start only embedding server"
    )

    return parser.parse_args()


def start_sglang_servers(no_wait=False):
    """Start the SGLang LLM and embedding servers using start_sglang_servers.py"""
    logger.info("\n[STARTUP] Starting SGLang servers...")

    # Import here to avoid circular imports
    from servers.start_sglang_servers import (
        main as sglang_main,
        parse_arguments as sglang_parse_args,
    )

    # Prepare arguments to pass to start_sglang_servers.py
    orig_argv = sys.argv

    # Build new arguments list for start_sglang_servers.py
    new_argv = [sys.argv[0]]

    # Add the no-wait flag if specified
    if no_wait:
        new_argv.append("--no-wait")

    # Copy relevant arguments from original argv
    for arg in orig_argv[1:]:
        # Skip --mode argument as that's specific to entrypoint.py
        if not arg.startswith("--mode"):
            new_argv.append(arg)

    # Replace sys.argv temporarily
    sys.argv = new_argv

    try:
        # Run the sglang_main function
        logger.info("[STARTUP] Delegating to start_sglang_servers.py")
        result = sglang_main()
        logger.info("[STARTUP] start_sglang_servers.py completed")
        return result
    finally:
        # Restore original sys.argv
        sys.argv = orig_argv


def start_fastapi_server():
    """Start the FastAPI server"""
    logger.info("\n[STARTUP] Starting FastAPI server...")

    # Import here to avoid circular imports
    from servers.server_fastapi import run_server

    # Run the FastAPI server with settings from config
    run_server(
        host=config.host,
        port=config.port,
        reload=config.reload,
        debug=config.debug,
    )


def run_tests():
    """Run system tests"""
    logger.info("\n[TESTS] Running system tests...")

    # Import test module
    from tests.test_services import main as run_test_services

    # Run tests
    run_test_services()


def main():
    """Main function to orchestrate system startup"""
    # Print startup banner
    logger.info("=" * 80)
    logger.info("CONTENT MODERATION SYSTEM STARTUP")
    logger.info("=" * 80)

    # Check GPU status
    check_gpu()

    # Parse command line arguments
    args = parse_arguments()

    # Reload config to ensure all environment variables are set
    reload_config()

    # Run in the specified mode
    if args.mode == "all":
        # Start both SGLang servers and FastAPI
        if args.no_wait:
            # Start SGLang servers without waiting
            start_sglang_servers(no_wait=True)
            # Then start FastAPI server
            start_fastapi_server()
        else:
            # Start SGLang servers and wait indefinitely
            start_sglang_servers()
            # The code below won't execute unless the SGLang servers exit
            start_fastapi_server()
    elif args.mode == "sglang":
        # Start only SGLang servers
        start_sglang_servers(no_wait=args.no_wait)
    elif args.mode == "fastapi":
        # Start only FastAPI server
        start_fastapi_server()
    elif args.mode == "test":
        # Run tests
        run_tests()


if __name__ == "__main__":
    main()
