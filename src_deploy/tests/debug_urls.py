#!/usr/bin/env python3
"""
Debug script to check API URL connectivity
"""

import os
import sys
import requests
import logging
from argparse import ArgumentParser

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_url(url, description):
    """Check if a URL is reachable"""
    logger.info(f"Checking {description} URL: {url}")

    try:
        # For embedding and LLM URLs, we need to test endpoints that support GET
        # or add authentication headers for endpoints that might require them
        if "embedding" in description.lower():
            # For embedding services, we'll use a POST request with auth headers
            response = requests.post(
                f"{url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer None",
                },
                json={"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "input": "Test"},
                timeout=5,
            )
        elif "llm" in description.lower():
            # For LLM services, we'll use a POST request with auth headers
            response = requests.post(
                f"{url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer None",
                },
                json={
                    "model": "microsoft/Phi-3.5-mini-instruct",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
                timeout=5,
            )
        else:
            # For other services (like FastAPI health endpoint), use GET
            response = requests.get(f"{url}", timeout=5)

        if response.status_code < 400:
            logger.info(
                f"✓ {description} URL is reachable! Status: {response.status_code}"
            )
            return True
        else:
            logger.error(
                f"✗ {description} URL returned error status: {response.status_code}"
            )
            logger.error(f"Response: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(
            f"✗ {description} URL connection error! The service may not be running."
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(
            f"✗ {description} URL timeout! The service is slow or unresponsive."
        )
        return False
    except Exception as e:
        logger.error(f"✗ {description} URL error: {e}")
        return False


def print_env_vars():
    """Print environment variables related to the URLs"""
    logger.info("Environment Variables:")

    # LLM variables
    llm_host = os.environ.get("LLM_HOST", "127.0.0.1")
    llm_port = os.environ.get("LLM_PORT", "8899")
    llm_url = os.environ.get("LLM_URL", f"http://{llm_host}:{llm_port}/v1")

    # Embedding variables
    emb_host = os.environ.get("EMBEDDING_HOST", "127.0.0.1")
    emb_port = os.environ.get("EMBEDDING_PORT", "8890")
    emb_url = os.environ.get("EMBEDDING_URL", f"http://{emb_host}:{emb_port}/v1")

    # Server variables
    server_host = os.environ.get("SERVER_HOST", "0.0.0.0")
    server_port = os.environ.get("SERVER_PORT", "8080")

    logger.info(f"LLM_HOST: {llm_host}")
    logger.info(f"LLM_PORT: {llm_port}")
    logger.info(f"LLM_URL: {llm_url}")
    logger.info(f"EMBEDDING_HOST: {emb_host}")
    logger.info(f"EMBEDDING_PORT: {emb_port}")
    logger.info(f"EMBEDDING_URL: {emb_url}")
    logger.info(f"SERVER_HOST: {server_host}")
    logger.info(f"SERVER_PORT: {server_port}")

    return {
        "llm_url": llm_url,
        "emb_url": emb_url,
        "server_url": f"http://{server_host if server_host != '0.0.0.0' else '127.0.0.1'}:{server_port}",
    }


def check_connectivity():
    """Check connectivity to all services"""
    urls = print_env_vars()

    # Check each URL
    llm_ok = check_url(urls["llm_url"], "LLM")
    embedding_ok = check_url(urls["emb_url"], "Embedding")
    api_ok = check_url(f"{urls['server_url']}/health", "FastAPI")

    # Print summary
    logger.info("\n=== Connectivity Summary ===")
    logger.info(f"LLM Service: {'✓ OK' if llm_ok else '✗ FAIL'}")
    logger.info(f"Embedding Service: {'✓ OK' if embedding_ok else '✗ FAIL'}")
    logger.info(f"FastAPI Service: {'✓ OK' if api_ok else '✗ FAIL'}")

    # Suggest fixes
    if not (llm_ok and embedding_ok and api_ok):
        logger.info("\n=== Troubleshooting ===")

        if not llm_ok:
            logger.info("LLM Service Fix:")
            logger.info("  1. Check if the LLM server is running")
            logger.info("  2. Verify the LLM_HOST and LLM_PORT settings")
            logger.info("  3. Run: python start_sglang_servers.py --llm-only")

        if not embedding_ok:
            logger.info("Embedding Service Fix:")
            logger.info("  1. Check if the embedding server is running")
            logger.info("  2. Verify the EMBEDDING_HOST and EMBEDDING_PORT settings")
            logger.info("  3. Run: python start_sglang_servers.py --embedding-only")

        if not api_ok:
            logger.info("FastAPI Service Fix:")
            logger.info("  1. Check if the FastAPI server is running")
            logger.info("  2. Verify the SERVER_HOST and SERVER_PORT settings")
            logger.info("  3. Check FastAPI logs for errors")

        return False

    return True


def main():
    """Main function"""
    parser = ArgumentParser(description="Debug API URL connectivity")
    parser.add_argument("--llm-url", help="Override LLM URL")
    parser.add_argument("--embedding-url", help="Override embedding URL")
    parser.add_argument("--server-url", help="Override server URL")

    args = parser.parse_args()

    # Override environment variables if provided
    if args.llm_url:
        os.environ["LLM_URL"] = args.llm_url
    if args.embedding_url:
        os.environ["EMBEDDING_URL"] = args.embedding_url
    if args.server_url:
        # Parse server URL into host and port
        parts = args.server_url.split(":")
        if len(parts) == 3:  # http://host:port
            os.environ["SERVER_HOST"] = parts[1].replace("//", "")
            os.environ["SERVER_PORT"] = parts[2]

    # Check connectivity
    success = check_connectivity()

    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
