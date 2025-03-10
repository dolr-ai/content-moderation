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
            input_file: Path to input file (JSONL)
            output_dir: Directory to save results
            num_examples: Number of examples to use
            num_samples: Number of samples to test
            max_characters: Maximum number of characters to test
        """
        self.server_url = server_url
        self.output_dir = (
            Path(output_dir) if output_dir else Path.cwd() / "performance_results"
        )
        self.num_examples = num_examples
        self.max_characters = max_characters
        self.test_data = []
        self.results = []
        self.summary = {}
        self.scaling_results = {}

        # Shared HTTP session
        self.http_session = None
        self.http_session_lock = asyncio.Lock()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load test data if provided
        if input_file:
            self.load_test_data(input_file, num_samples)

    async def get_http_session(self):
        """
        Get or create a shared HTTP session

        Returns:
            aiohttp.ClientSession instance
        """
        async with self.http_session_lock:
            if self.http_session is None or self.http_session.closed:
                # Configure optimized session
                connector = aiohttp.TCPConnector(
                    limit=100,  # Connection pool size
                    limit_per_host=50,  # Connections per host
                    keepalive_timeout=60,  # Keep connections alive for 60 seconds
                )
                timeout = aiohttp.ClientTimeout(total=90)  # 90 seconds timeout
                self.http_session = aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                )
            return self.http_session

    async def close(self):
        """
        Close HTTP session
        """
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None

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
        throughput = self.calculate_throughput(len(test_samples), total_time)

        # Add summary
        summary = {
            "total_samples": len(test_samples),
            "total_time_seconds": total_time,
            "throughput": throughput,
            "avg_latency_seconds": np.mean([r["latency"] for r in results]) if results else 0,
            "median_latency_seconds": np.median([r["latency"] for r in results]) if results else 0,
            "p95_latency_seconds": np.percentile([r["latency"] for r in results], 95) if results else 0,
            "p99_latency_seconds": np.percentile([r["latency"] for r in results], 99) if results else 0,
            "min_latency_seconds": min([r["latency"] for r in results]) if results else 0,
            "max_latency_seconds": max([r["latency"] for r in results]) if results else 0,
        }

        logger.info(f"Sequential test completed: {summary}")

        self.results = results
        self.summary = summary

        return results

    async def test_single_request_async(
        self, text: str, max_input_length: int = 2000
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Test a single moderation request and measure latency using asyncio

        Args:
            text: Text to moderate
            max_input_length: Maximum input length to send

        Returns:
            Tuple of (response_data, latency_in_seconds, is_timeout)
        """
        start_time = time.time()
        is_timeout = False

        try:
            # Get shared HTTP session
            session = await self.get_http_session()

            async with session.post(
                f"{self.server_url}/moderate",
                json={
                    "text": text[:max_input_length],
                    "num_examples": self.num_examples,
                },
                timeout=90,
            ) as response:
                end_time = time.time()
                latency = end_time - start_time

                if response.status == 200:
                    response_data = await response.json()
                    return response_data, latency, is_timeout
                else:
                    error_text = await response.text()
                    logger.error(f"Error in request: {response.status} - {error_text}")
                    return (
                        {
                            "error": error_text,
                            "status_code": response.status,
                        },
                        latency,
                        is_timeout,
                    )

        except aiohttp.ClientConnectorError as e:
            end_time = time.time()
            latency = end_time - start_time
            error_msg = f"Connection error to {self.server_url}: {str(e) or 'Connection refused'}"
            logger.error(f"Connection error: {error_msg}")
            return {"error": error_msg}, latency, is_timeout

        except asyncio.TimeoutError:
            end_time = time.time()
            latency = end_time - start_time
            is_timeout = True
            error_msg = f"Request to {self.server_url} timed out after 90 seconds"
            logger.error(f"Timeout error: {error_msg}")
            return {"error": error_msg, "timeout": True}, latency, is_timeout

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            print(
                f"Exception type: {type(e).__name__}, details: {str(e) or 'No details available'}"
            )
            logger.error(
                f"Exception in request: {type(e).__name__} - {str(e) or 'No details available'}"
            )
            return (
                {"error": str(e) or f"Empty {type(e).__name__} exception"},
                latency,
                is_timeout,
            )

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
                response, latency, is_timeout = await self.test_single_request_async(
                    text, max_input_length=max_input_length
                )

                result = {
                    "sample_id": i,
                    "text": text,
                    "latency": latency,
                    "response": response,
                    "is_timeout": is_timeout,
                    "timestamp": datetime.now().isoformat(),
                }

                if "moderation_category" in sample:
                    result["expected_category"] = sample["moderation_category"]

                return result

        try:
            # Create tasks for all samples
            tasks = [process_sample(i, sample) for i, sample in enumerate(test_samples)]

            # Execute tasks with progress bar
            results = await tqdm_asyncio.gather(*tasks, desc="Processing samples")

            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time
            successful_requests = sum(
                1 for r in results if "error" not in r["response"]
            )
            timeouts = sum(1 for r in results if r["is_timeout"])
            latencies = [r["latency"] for r in results if not r["is_timeout"]]

            # Prepare summary
            if latencies:
                summary = {
                    "total_time": total_time,
                    "throughput": self.calculate_throughput(len(results), total_time),
                    "requests": len(results),
                    "successful_requests": successful_requests,
                    "failed_requests": len(results) - successful_requests,
                    "timeouts": timeouts,
                    "avg_latency": np.mean(latencies) if latencies else 0,
                    "median_latency": np.median(latencies) if latencies else 0,
                    "p90_latency": np.percentile(latencies, 90) if latencies else 0,
                    "p95_latency": np.percentile(latencies, 95) if latencies else 0,
                    "p99_latency": np.percentile(latencies, 99) if latencies else 0,
                    "min_latency": min(latencies) if latencies else 0,
                    "max_latency": max(latencies) if latencies else 0,
                    "concurrency": concurrency,
                }
            else:
                summary = {
                    "total_time": total_time,
                    "requests": len(results),
                    "successful_requests": 0,
                    "failed_requests": len(results),
                    "timeouts": timeouts,
                    "concurrency": concurrency,
                }

            logger.info(
                f"Concurrent test completed with {concurrency} workers: {summary}"
            )

            self.results = results
            self.summary = summary

            return results
        finally:
            # Ensure resources are cleaned up
            await self.close()

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
            scaling_results[c]["throughput"] for c in concurrency_levels
        ]

        latency_values = [
            scaling_results[c]["avg_latency"] for c in concurrency_levels
        ]

        timeout_counts = [
            scaling_results[c]["timeouts"] for c in concurrency_levels
        ]

        # Fix: Calculate timeout rates using requests count from scaling results
        timeout_rates = [
            scaling_results[c]["timeouts"] / scaling_results[c]["requests"]
            for c in concurrency_levels
        ]

        scaling_report = {
            "concurrency_levels": concurrency_levels,
            "throughput_values": throughput_values,
            "latency_values": latency_values,
            "timeout_counts": timeout_counts,
            "timeout_rates": timeout_rates,
            "details": scaling_results,
        }

        self.scaling_results = scaling_report

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
        if not hasattr(self, "scaling_results"):
            logger.error("No scaling report to save")
            return ""

        output_path = Path(self.output_dir) / filename

        try:
            with open(output_path, "w") as f:
                json.dump(self.scaling_results, f, indent=2)

            logger.info(f"Scaling report saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving scaling report: {e}")
            return ""

    def calculate_throughput(self, num_requests: int, total_time: float) -> float:
        """
        Calculate throughput safely with error handling

        Args:
            num_requests: Number of requests processed
            total_time: Total time taken in seconds

        Returns:
            Throughput in requests per second
        """
        try:
            if total_time <= 0:
                logger.warning("Total time is zero or negative, returning 0 throughput")
                return 0
            return num_requests / total_time
        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
            return 0


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
    Run performance test and save results

    Args:
        input_file: Path to input file
        server_url: URL of the moderation server
        output_dir: Directory to save results
        num_examples: Number of examples to use
        test_type: Type of test (sequential or concurrent)
        num_samples: Number of samples to test (None for all)
        concurrency: Maximum number of concurrent requests
        run_scaling_test: Whether to run scaling test
        concurrency_levels: Comma-separated list of concurrency levels

    Returns:
        Dictionary with test results
    """
    # Check if server is up before starting
    if not check_server_health(server_url):
        logger.error(f"Server at {server_url} is not responding")
        return {"error": f"Server at {server_url} is not responding"}

    # Initialize performance tester
    tester = PerformanceTester(
        server_url=server_url,
        input_file=input_file,
        output_dir=output_dir,
        num_examples=num_examples,
    )

    # Parse concurrency levels
    parsed_concurrency_levels = parse_concurrency_levels(concurrency_levels)

    try:
        # Run test based on type
        if run_scaling_test:
            if test_type == "concurrent":
                logger.info(
                    f"Running concurrency scaling test with levels: {parsed_concurrency_levels or '[1, 2, 4, 8, 16, 32, 64]'}"
                )
                results = asyncio.run(
                    tester.run_concurrency_scaling_test_async(
                        num_samples=num_samples or 100,
                        concurrency_levels=parsed_concurrency_levels,
                    )
                )
                # Save scaling report
                tester.save_scaling_report()
                return results
            else:
                logger.error("Scaling test requires concurrent test type")
                return {"error": "Scaling test requires concurrent test type"}

        if test_type == "sequential":
            logger.info(f"Running sequential test with {num_samples or 'all'} samples")
            results = tester.run_sequential_test(num_samples=num_samples)
        elif test_type == "concurrent":
            logger.info(
                f"Running concurrent test with {num_samples or 'all'} samples and concurrency {concurrency}"
            )
            results = asyncio.run(
                tester.run_concurrent_test_async(
                    num_samples=num_samples, concurrency=concurrency
                )
            )
        else:
            logger.error(f"Invalid test type: {test_type}")
            return {"error": f"Invalid test type: {test_type}"}

        # Save results
        output_file = tester.save_results()
        logger.info(f"Results saved to {output_file}")

        return {"summary": tester.summary, "output_file": output_file}

    except Exception as e:
        logger.error(f"Error in performance test: {type(e).__name__}: {str(e)}")
        return {"error": f"Performance test error: {str(e)}"}
    finally:
        # Ensure any remaining resources are cleaned up
        if hasattr(tester, "http_session") and tester.http_session is not None:
            asyncio.run(tester.close())
