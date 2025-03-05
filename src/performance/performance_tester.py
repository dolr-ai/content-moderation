"""
Performance Testing Module for Content Moderation System

This module provides functionality to test the performance of the moderation server,
including latency and throughput analysis.
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class PerformanceTester:
    """Performance tester for content moderation system"""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        input_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        num_examples: int = 3,
        num_samples: Optional[int] = None,
        max_characters: int = 2000,
    ):
        """
        Initialize the performance tester

        Args:
            server_url: URL of the moderation server
            input_file: Path to input JSONL file with texts to test
            output_dir: Directory to save test results
            num_examples: Number of similar examples to use in moderation
            num_samples: Number of samples to test (None for all)
        """
        self.server_url = server_url
        self.input_file = input_file
        self.output_dir = output_dir or Path("performance_results")
        self.num_examples = num_examples
        self.num_samples = num_samples
        self.results = []

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load test data if provided
        self.test_data = []
        if input_file:
            self.load_test_data(input_file, num_samples=num_samples)

    def load_test_data(
        self, input_file: Union[str, Path], num_samples: Optional[int] = None
    ) -> None:
        """
        Load test data from JSONL file

        Args:
            input_file: Path to input JSONL file
            num_samples: Number of samples to test (None for all)
        """
        try:
            df = pd.read_json(input_file, lines=True)
            if num_samples is not None:
                df = df.sample(n=min(num_samples, len(df)))

            required_columns = ["text"]

            # Check if required columns exist
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing required columns in input file: {missing}")
                return

            # Extract test data
            self.test_data = df.to_dict(orient="records")
            logger.info(f"Loaded {len(self.test_data)} test samples from {input_file}")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise

    def test_single_request(
        self, text: str, max_input_length: int = 2000
    ) -> Tuple[Dict[str, Any], float]:
        """
        Test a single moderation request and measure latency

        Args:
            text: Text to moderate

        Returns:
            Tuple of (response_data, latency_in_seconds)
        """
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.server_url}/moderate",
                json={
                    "text": text[:max_input_length],
                    "num_examples": self.num_examples,
                },
                timeout=60,
            )

            end_time = time.time()
            latency = end_time - start_time

            if response.status_code == 200:
                return response.json(), latency
            else:
                logger.error(
                    f"Error in request: {response.status_code} - {response.text}"
                )
                return {
                    "error": response.text,
                    "status_code": response.status_code,
                }, latency

        except requests.exceptions.ConnectionError as e:
            end_time = time.time()
            latency = end_time - start_time
            error_msg = f"Connection error to {self.server_url}: {str(e) or 'Connection refused'}"
            logger.error(f"Connection error: {error_msg}")
            return {"error": error_msg}, latency

        except requests.exceptions.Timeout:
            end_time = time.time()
            latency = end_time - start_time
            error_msg = f"Request to {self.server_url} timed out after 60 seconds"
            logger.error(f"Timeout error: {error_msg}")
            return {"error": error_msg}, latency

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(
                f"Exception in request: {type(e).__name__} - {str(e) or 'No details available'}"
            )
            return {"error": str(e) or f"Empty {type(e).__name__} exception"}, latency

    def run_sequential_test(
        self, num_samples: Optional[int] = None, max_input_length: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Run sequential test on the moderation server

        Args:
            num_samples: Number of samples to test (None for all)

        Returns:
            List of test results
        """
        if not self.test_data:
            logger.error("No test data available")
            return []

        # Limit number of samples if specified
        test_samples = self.test_data
        if num_samples is not None:
            test_samples = test_samples[:num_samples]

        results = []
        start_time = time.time()

        for i, sample in enumerate(tqdm(test_samples, desc="Testing")):
            text = sample["text"]
            response, latency = self.test_single_request(
                text, max_input_length=max_input_length
            )

            result = {
                "sample_id": i,
                "text": text,
                "latency": latency,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }

            if "moderation_category" in sample:
                result["expected_category"] = sample["moderation_category"]

            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate throughput
        throughput = len(test_samples) / total_time if total_time > 0 else 0

        # Add summary
        summary = {
            "total_samples": len(test_samples),
            "total_time_seconds": total_time,
            "throughput_requests_per_second": throughput,
            "avg_latency_seconds": (
                np.mean([r["latency"] for r in results]) if results else 0
            ),
            "median_latency_seconds": (
                np.median([r["latency"] for r in results]) if results else 0
            ),
            "p95_latency_seconds": (
                np.percentile([r["latency"] for r in results], 95) if results else 0
            ),
            "p99_latency_seconds": (
                np.percentile([r["latency"] for r in results], 99) if results else 0
            ),
            "min_latency_seconds": (
                min([r["latency"] for r in results]) if results else 0
            ),
            "max_latency_seconds": (
                max([r["latency"] for r in results]) if results else 0
            ),
        }

        logger.info(f"Sequential test completed: {summary}")

        self.results = results
        self.summary = summary

        return results

    async def test_single_request_async(
        self, text: str, max_input_length: int = 2000
    ) -> Tuple[Dict[str, Any], float]:
        """
        Test a single moderation request and measure latency using asyncio

        Args:
            text: Text to moderate
            max_input_length: Maximum input length to send

        Returns:
            Tuple of (response_data, latency_in_seconds)
        """
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/moderate",
                    json={
                        "text": text[:max_input_length],
                        "num_examples": self.num_examples,
                    },
                    timeout=60,
                ) as response:
                    end_time = time.time()
                    latency = end_time - start_time

                    if response.status == 200:
                        response_data = await response.json()
                        return response_data, latency
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Error in request: {response.status} - {error_text}"
                        )
                        return {
                            "error": error_text,
                            "status_code": response.status,
                        }, latency

        except aiohttp.ClientConnectorError as e:
            end_time = time.time()
            latency = end_time - start_time
            error_msg = f"Connection error to {self.server_url}: {str(e) or 'Connection refused'}"
            logger.error(f"Connection error: {error_msg}")
            return {"error": error_msg}, latency

        except asyncio.TimeoutError:
            end_time = time.time()
            latency = end_time - start_time
            error_msg = f"Request to {self.server_url} timed out after 60 seconds"
            logger.error(f"Timeout error: {error_msg}")
            return {"error": error_msg}, latency

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            print(
                f"Exception type: {type(e).__name__}, details: {str(e) or 'No details available'}"
            )
            logger.error(
                f"Exception in request: {type(e).__name__} - {str(e) or 'No details available'}"
            )
            return {"error": str(e) or f"Empty {type(e).__name__} exception"}, latency

    async def run_concurrent_test_async(
        self,
        num_samples: Optional[int] = None,
        concurrency: int = 10,
        max_input_length: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Run concurrent test on the moderation server using asyncio

        Args:
            num_samples: Number of samples to test (None for all)
            concurrency: Maximum number of concurrent requests
            max_input_length: Maximum input length to send

        Returns:
            List of test results
        """
        if not self.test_data:
            logger.error("No test data available")
            return []

        # Limit number of samples if specified
        test_samples = self.test_data
        if num_samples is not None:
            test_samples = test_samples[:num_samples]

        results = []
        start_time = time.time()

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def process_sample(i, sample):
            async with semaphore:
                text = sample["text"]
                response, latency = await self.test_single_request_async(
                    text, max_input_length=max_input_length
                )

                result = {
                    "sample_id": i,
                    "text": text,
                    "latency": latency,
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                }

                if "moderation_category" in sample:
                    result["expected_category"] = sample["moderation_category"]

                return result

        # Create tasks for all samples
        tasks = [process_sample(i, sample) for i, sample in enumerate(test_samples)]

        # Run tasks with progress bar
        completed_results = await tqdm_asyncio.gather(
            *tasks, desc=f"Testing with {concurrency} concurrent requests"
        )

        # Add all results
        results.extend(completed_results)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate throughput
        throughput = len(test_samples) / total_time if total_time > 0 else 0

        # Add summary
        summary = {
            "total_samples": len(test_samples),
            "total_time_seconds": total_time,
            "throughput_requests_per_second": throughput,
            "concurrency": concurrency,
            "avg_latency_seconds": (
                np.mean([r["latency"] for r in results]) if results else 0
            ),
            "median_latency_seconds": (
                np.median([r["latency"] for r in results]) if results else 0
            ),
            "p95_latency_seconds": (
                np.percentile([r["latency"] for r in results], 95) if results else 0
            ),
            "p99_latency_seconds": (
                np.percentile([r["latency"] for r in results], 99) if results else 0
            ),
            "min_latency_seconds": (
                min([r["latency"] for r in results]) if results else 0
            ),
            "max_latency_seconds": (
                max([r["latency"] for r in results]) if results else 0
            ),
        }

        logger.info(f"Concurrent test completed: {summary}")

        self.results = results
        self.summary = summary

        return results

    def run_concurrent_test(
        self,
        num_samples: Optional[int] = None,
        concurrency: int = 10,
        max_input_length: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Run concurrent test on the moderation server (asyncio wrapper)

        Args:
            num_samples: Number of samples to test (None for all)
            concurrency: Number of concurrent requests
            max_input_length: Maximum input length to send

        Returns:
            List of test results
        """
        return asyncio.run(
            self.run_concurrent_test_async(
                num_samples=num_samples,
                concurrency=concurrency,
                max_input_length=max_input_length,
            )
        )

    async def run_concurrency_scaling_test_async(
        self,
        num_samples: int = 100,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64],
    ) -> Dict[str, Any]:
        """
        Run tests with different concurrency levels to analyze scaling using asyncio

        Args:
            num_samples: Number of samples to test per concurrency level
            concurrency_levels: List of concurrency levels to test

        Returns:
            Dictionary with test results for each concurrency level
        """
        if not self.test_data:
            logger.error("No test data available")
            return {}

        # Ensure we have enough test data
        if len(self.test_data) < num_samples:
            logger.warning(
                f"Not enough test data ({len(self.test_data)} < {num_samples}). "
                f"Using all available data."
            )
            num_samples = len(self.test_data)

        scaling_results = {}

        for concurrency in concurrency_levels:
            logger.info(f"Testing with concurrency level: {concurrency}")

            # Run test with current concurrency level
            await self.run_concurrent_test_async(
                num_samples=num_samples, concurrency=concurrency
            )

            # Store summary results
            scaling_results[concurrency] = self.summary

        # Create scaling report
        throughput_values = [
            scaling_results[c]["throughput_requests_per_second"]
            for c in concurrency_levels
        ]

        latency_values = [
            scaling_results[c]["avg_latency_seconds"] for c in concurrency_levels
        ]

        scaling_report = {
            "concurrency_levels": concurrency_levels,
            "throughput_values": throughput_values,
            "latency_values": latency_values,
            "details": scaling_results,
        }

        self.scaling_report = scaling_report

        return scaling_report

    def run_concurrency_scaling_test(
        self,
        num_samples: int = 100,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64],
    ) -> Dict[str, Any]:
        """
        Run tests with different concurrency levels to analyze scaling (asyncio wrapper)

        Args:
            num_samples: Number of samples to test per concurrency level
            concurrency_levels: List of concurrency levels to test

        Returns:
            Dictionary with test results for each concurrency level
        """
        return asyncio.run(
            self.run_concurrency_scaling_test_async(
                num_samples=num_samples, concurrency_levels=concurrency_levels
            )
        )

    def save_results(self, filename: str = "performance_results.json") -> str:
        """
        Save test results to file

        Args:
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        if not self.results:
            logger.error("No results to save")
            return ""

        output_path = Path(self.output_dir) / filename

        try:
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "summary": self.summary,
                        "results": self.results,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Results saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""

    def save_scaling_report(self, filename: str = "scaling_report.json") -> str:
        """
        Save scaling test results to file

        Args:
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        if not hasattr(self, "scaling_report"):
            logger.error("No scaling report to save")
            return ""

        output_path = Path(self.output_dir) / filename

        try:
            with open(output_path, "w") as f:
                json.dump(self.scaling_report, f, indent=2)

            logger.info(f"Scaling report saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving scaling report: {e}")
            return ""


async def check_server_health_async(server_url: str, timeout: int = 5) -> bool:
    """
    Check if the server is available and healthy using asyncio

    Args:
        server_url: URL of the server
        timeout: Timeout in seconds

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health", timeout=timeout) as response:
                return response.status == 200
    except Exception as e:
        logger.error(
            f"Server not available: {type(e).__name__} - {str(e) or 'No details available'}"
        )
        return False


def check_server_health(server_url: str, timeout: int = 5) -> bool:
    """
    Check if the server is available and healthy (asyncio wrapper)

    Args:
        server_url: URL of the server
        timeout: Timeout in seconds

    Returns:
        True if server is healthy, False otherwise
    """
    return asyncio.run(check_server_health_async(server_url, timeout))


def parse_concurrency_levels(
    concurrency_levels_str: Optional[str],
) -> Optional[List[int]]:
    """
    Parse comma-separated concurrency levels string into a list of integers

    Args:
        concurrency_levels_str: Comma-separated concurrency levels

    Returns:
        List of concurrency levels as integers
    """
    if not concurrency_levels_str:
        return None

    try:
        return [int(level.strip()) for level in concurrency_levels_str.split(",")]
    except ValueError as e:
        logger.error(f"Error parsing concurrency levels: {e}")
        return None


def run_performance_test(
    input_file: Union[str, Path],
    server_url: str = "http://localhost:8000",
    output_dir: Optional[Union[str, Path]] = None,
    num_examples: int = 3,
    test_type: str = "sequential",
    num_samples: Optional[int] = None,
    concurrency: int = 10,
    run_scaling_test: bool = False,
    concurrency_levels: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run performance test on the moderation server

    Args:
        input_file: Path to input JSONL file with texts to test
        server_url: URL of the moderation server
        output_dir: Directory to save test results
        num_examples: Number of similar examples to use in moderation
        test_type: Type of test to run ("sequential" or "concurrent")
        num_samples: Number of samples to test (None for all)
        concurrency: Number of concurrent requests for concurrent test
        run_scaling_test: Whether to run concurrency scaling test
        concurrency_levels: Comma-separated string of concurrency levels for scaling test

    Returns:
        Dictionary with test results
    """
    # Check if server is available
    if not check_server_health(server_url):
        return {"error": "Server health check failed"}

    # Parse concurrency levels if provided
    parsed_concurrency_levels = parse_concurrency_levels(concurrency_levels)

    # Initialize tester
    try:
        tester = PerformanceTester(
            server_url=server_url,
            input_file=input_file,
            output_dir=output_dir,
            num_examples=num_examples,
            num_samples=num_samples,
        )
    except Exception as e:
        logger.error(f"Failed to initialize performance tester: {e}")
        return {"error": f"Failed to initialize performance tester: {e}"}

    # Run appropriate test
    try:
        if run_scaling_test:
            # Use default or parsed concurrency levels
            if parsed_concurrency_levels is None:
                parsed_concurrency_levels = [1, 2, 4, 8, 16, 32]

            # Use a default number of samples if not specified
            scaling_samples = num_samples if num_samples is not None else 100

            # Run scaling test
            scaling_report = tester.run_concurrency_scaling_test(
                num_samples=scaling_samples,
                concurrency_levels=parsed_concurrency_levels,
            )

            # Save results
            report_path = tester.save_scaling_report()

            return {
                "scaling_report": scaling_report,
                "report_path": report_path,
            }
        else:
            # Run regular test
            if test_type == "concurrent":
                results = tester.run_concurrent_test(
                    num_samples=num_samples,
                    concurrency=concurrency,
                )
            else:  # sequential
                results = tester.run_sequential_test(num_samples=num_samples)

            # Save results
            results_path = tester.save_results()

            return {
                "summary": tester.summary,
                "results_path": results_path,
            }
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        return {"error": f"Error running performance test: {e}"}
