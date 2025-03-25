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
