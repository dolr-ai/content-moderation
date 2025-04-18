#!/usr/bin/env python3
"""
Test script for content moderation services
This script tests the SGLang embedding and LLM servers directly, and then tests the moderation API.
"""

import os
import sys
import argparse
import json
import time
import logging
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_embedding_service(url="http://localhost:8890/v1"):
    """Test the embedding service directly"""
    logger.info(f"Testing embedding service at {url}...")

    try:
        # Test the embedding server
        response = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer None",
            },
            json={
                "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                "input": "This is a test sentence for embedding.",
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            embedding_dim = len(data["data"][0]["embedding"])
            logger.info(
                f"Embedding service working! Generated embedding with {embedding_dim} dimensions."
            )
            return True
        else:
            logger.error(
                f"Embedding service request failed with status {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error testing embedding service: {e}")
        return False


def test_llm_service(url="http://localhost:8899/v1"):
    """Test the LLM service directly"""
    logger.info(f"Testing LLM service at {url}...")

    try:
        # Test the LLM server
        response = requests.post(
            f"{url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer None",
            },
            json={
                "model": "microsoft/Phi-3.5-mini-instruct",
                "messages": [{"role": "user", "content": "Who are you?"}],
                "max_tokens": 128,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
            logger.info(f"LLM service working! Response: {response_text[:50]}...")
            return True
        else:
            logger.error(
                f"LLM service request failed with status {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error testing LLM service: {e}")
        return False


def test_moderation_api(url="http://localhost:8080"):
    """Test the moderation API"""
    logger.info(f"Testing moderation API at {url}...")

    try:
        # Test the health endpoint first
        response = requests.get(f"{url}/health", timeout=10)

        if response.status_code == 200:
            logger.info(f"Health check successful: {response.json()}")
        else:
            logger.error(f"Health check failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

        # Now test the moderation endpoint
        response = requests.post(
            f"{url}/moderate",
            headers={"Content-Type": "application/json"},
            json={
                "text": "This is a test sentence for moderation.",
                "num_examples": 3,
                "max_input_length": 2000,
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            logger.info(f"Moderation API working! Category: {data.get('category')}")
            return True
        else:
            logger.error(
                f"Moderation API request failed with status {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error testing moderation API: {e}")
        return False


def diagnose_services():
    """Run diagnostics on all services"""
    failures = 0

    # Test embedding service
    embedding_url = os.environ.get("EMBEDDING_URL", "http://localhost:8890/v1")
    if not test_embedding_service(embedding_url):
        logger.error("Embedding service test failed!")
        failures += 1
    else:
        logger.info("Embedding service test passed!")

    # Test LLM service
    llm_url = os.environ.get("LLM_URL", "http://localhost:8899/v1")
    if not test_llm_service(llm_url):
        logger.error("LLM service test failed!")
        failures += 1
    else:
        logger.info("LLM service test passed!")

    # Test moderation API
    api_url = f"http://{os.environ.get('SERVER_HOST', 'localhost')}:{os.environ.get('SERVER_PORT', '8080')}"
    if not test_moderation_api(api_url):
        logger.error("Moderation API test failed!")
        failures += 1
    else:
        logger.info("Moderation API test passed!")

    # Final status
    if failures == 0:
        logger.info("ALL TESTS PASSED! Services are working properly.")
        return True
    else:
        logger.error(f"{failures} tests failed. Please check the logs for details.")
        return False


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test content moderation services")
    parser.add_argument("--embedding-url", help="URL for the embedding service")
    parser.add_argument("--llm-url", help="URL for the LLM service")
    parser.add_argument("--api-url", help="URL for the moderation API")
    parser.add_argument("--all", action="store_true", help="Test all services")
    parser.add_argument(
        "--embedding", action="store_true", help="Test only the embedding service"
    )
    parser.add_argument("--llm", action="store_true", help="Test only the LLM service")
    parser.add_argument(
        "--api", action="store_true", help="Test only the moderation API"
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Set URLs from args if provided
    if args.embedding_url:
        os.environ["EMBEDDING_URL"] = args.embedding_url
    if args.llm_url:
        os.environ["LLM_URL"] = args.llm_url
    if args.api_url:
        parts = args.api_url.split(":")
        if len(parts) == 3:  # http://host:port
            host = parts[1].replace("//", "")
            port = parts[2]
            os.environ["SERVER_HOST"] = host
            os.environ["SERVER_PORT"] = port

    # Determine which tests to run
    if args.all or (not args.embedding and not args.llm and not args.api):
        # Run all tests by default
        return diagnose_services()
    else:
        # Run specific tests
        failures = 0

        if args.embedding:
            embedding_url = os.environ.get("EMBEDDING_URL", "http://localhost:8890/v1")
            if not test_embedding_service(embedding_url):
                failures += 1

        if args.llm:
            llm_url = os.environ.get("LLM_URL", "http://localhost:8899/v1")
            if not test_llm_service(llm_url):
                failures += 1

        if args.api:
            api_url = f"http://{os.environ.get('SERVER_HOST', 'localhost')}:{os.environ.get('SERVER_PORT', '8080')}"
            if not test_moderation_api(api_url):
                failures += 1

        # Final status
        if failures == 0:
            logger.info("ALL REQUESTED TESTS PASSED!")
            return True
        else:
            logger.error(f"{failures} tests failed. Please check the logs for details.")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
