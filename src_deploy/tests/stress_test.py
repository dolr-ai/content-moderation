#!/usr/bin/env python3
"""
Stress Test for Content Moderation Service

This script performs stress testing on the moderation service to evaluate:
1. Latency under different concurrency levels
2. Throughput and performance degradation
3. Error rates and stability under load
4. Resource utilization

Usage:


python ./tests/stress_test.py \
  --server-url server_url_fly_or_local \
  --input-file /root/content-moderation/data/benchmark_v1.jsonl \
  --scaling-test \
  --concurrency-levels 4,8 \
  --duration-per-level 10 \
  --output-dir results_perf_test \
  --api-key your-api-key-here
"""

import os
import sys
import time
import json
import logging
import asyncio
import argparse
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import concurrent.futures

import pandas as pd
import numpy as np
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StressTester:
    """Stress tester for content moderation service"""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        input_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        num_examples: int = 3,
        max_input_length: int = 2000,
        request_timeout: int = 90,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the stress tester

        Args:
            server_url: URL of the moderation server
            input_file: Path to input file (JSONL)
            output_dir: Directory to save results
            num_examples: Number of examples to use in RAG
            max_input_length: Maximum text length to process
            request_timeout: Timeout for requests in seconds
            api_key: API key for authorization
        """
        self.server_url = server_url
        self.output_dir = (
            Path(output_dir) if output_dir else Path("stress_test_results")
        )
        self.num_examples = num_examples
        self.max_input_length = max_input_length
        self.request_timeout = request_timeout
        self.api_key = api_key

        # Test data and results
        self.test_data = []
        self.results = []
        self.error_count = 0
        self.success_count = 0

        # Performance metrics
        self.latencies = {"total": [], "embedding": [], "llm": [], "bigquery": []}

        # Shared HTTP session
        self.http_session = None
        self.http_session_lock = asyncio.Lock()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load test data if provided
        if input_file:
            self.load_test_data(input_file)
        else:
            logger.error(
                "Input file is required. Please provide a JSONL file with test data."
            )
            raise ValueError("Input file is required")

    async def get_http_session(self):
        """
        Get or create a shared HTTP session for connection pooling

        Returns:
            aiohttp.ClientSession instance
        """
        async with self.http_session_lock:
            if self.http_session is None or self.http_session.closed:
                connector = aiohttp.TCPConnector(
                    limit=200,  # Higher connection pool for stress testing
                    limit_per_host=100,  # Higher connections per host
                    keepalive_timeout=30,  # Shorter keepalive for high throughput
                    force_close=False,  # Let the pool manage connections
                    enable_cleanup_closed=True,  # Clean up closed connections
                )
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                self.http_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    raise_for_status=False,  # Don't raise exceptions for non-200 responses
                )
            return self.http_session

    async def close(self):
        """Close HTTP session"""
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
            num_samples: Number of samples to use (None for all)
        """
        try:
            df = pd.read_json(input_file, lines=True)
            required_columns = ["text"]

            # Check if required columns exist
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing required columns in input file: {missing}")
                return

            # Extract test data (all samples)
            self.test_data = df.to_dict(orient="records")
            logger.info(f"Loaded {len(self.test_data)} test samples from {input_file}")

            # Check if expected categories are available
            self.has_expected_categories = "moderation_category" in df.columns
            if self.has_expected_categories:
                # Show category distribution
                category_counts = df["moderation_category"].value_counts()
                logger.info(f"Category distribution: {category_counts.to_dict()}")
            else:
                logger.warning(
                    "No 'moderation_category' column found in data. Accuracy metrics will not be available."
                )

            # Store original data size for reference
            self.original_data_size = len(self.test_data)

            # If num_samples is provided, apply sampling
            if num_samples is not None:
                self.sample_test_data(
                    num_samples, stratified=self.has_expected_categories
                )

        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise

    def sample_test_data(self, num_samples: int, stratified: bool = False) -> None:
        """
        Sample test data, either randomly or with stratification by moderation_category

        Args:
            num_samples: Number of samples to select
            stratified: Whether to use stratified sampling by moderation_category
        """
        if not self.test_data:
            logger.warning("No test data available to sample from")
            return

        # If we want fewer samples than we have, perform sampling
        if num_samples < len(self.test_data):
            if stratified and "moderation_category" in self.test_data[0]:
                # Convert back to DataFrame for stratified sampling
                df = pd.DataFrame(self.test_data)

                # Group by moderation_category
                grouped = df.groupby("moderation_category")

                # Calculate samples per category, ensuring at least 1 sample per category if possible
                categories = list(grouped.groups.keys())
                samples_per_category = max(1, num_samples // len(categories))
                remainder = num_samples - (samples_per_category * len(categories))

                # Sample from each category
                sampled_data = []
                for category, group in grouped:
                    # Sample min(samples_per_category, group size) from each category
                    category_samples = min(samples_per_category, len(group))
                    sampled_data.append(group.sample(n=category_samples))

                # If we have remainder, sample additional from the entire dataset
                if remainder > 0 and len(df) > sum(len(g) for g in sampled_data):
                    # Get indices already sampled
                    sampled_indices = set()
                    for sample_df in sampled_data:
                        sampled_indices.update(sample_df.index)

                    # Sample from remaining data
                    remaining_df = df.loc[~df.index.isin(sampled_indices)]
                    if len(remaining_df) > 0:
                        additional = min(remainder, len(remaining_df))
                        sampled_data.append(remaining_df.sample(n=additional))

                # Combine all samples
                sampled_df = pd.concat(sampled_data)
                self.test_data = sampled_df.to_dict(orient="records")
                logger.info(
                    f"Performed stratified sampling: {len(self.test_data)} samples from {self.original_data_size} total"
                )
            else:
                # Simple random sampling
                random.shuffle(self.test_data)
                self.test_data = self.test_data[:num_samples]
                logger.info(
                    f"Performed random sampling: {len(self.test_data)} samples from {self.original_data_size} total"
                )

    async def test_single_request(
        self, text: str, expected_category: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Test a single moderation request

        Args:
            text: Text to moderate
            expected_category: Expected moderation category if available

        Returns:
            Tuple of (response data, success boolean)
        """
        session = await self.get_http_session()

        # Prepare request data
        request_data = {
            "text": text,
            "num_examples": self.num_examples,
            "max_input_length": self.max_input_length,
        }

        # Prepare headers with API key if provided
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            start_time = time.time()

            async with session.post(
                f"{self.server_url}/moderate",
                json=request_data,
                headers=headers,
            ) as response:
                response_data = await response.json()
                elapsed_time = time.time() - start_time

                if response.status != 200:
                    logger.warning(
                        f"Request failed with status {response.status}: {response_data}"
                    )
                    return {
                        "status_code": response.status,
                        "error": response_data,
                        "latency": elapsed_time * 1000,  # Convert to ms
                    }, False

                # Extract timing metrics
                timing = response_data.get("timing", {})
                predicted_category = response_data.get("category", "unknown")

                result = {
                    "status_code": response.status,
                    "category": predicted_category,
                    "latency": elapsed_time * 1000,  # Convert to ms
                    "embedding_time": timing.get("embedding_time_ms", 0),
                    "llm_time": timing.get("llm_time_ms", 0),
                    "bigquery_time": timing.get("bigquery_time_ms", 0),
                    "total_time": timing.get("total_time_ms", 0),
                    "text_length": len(text),
                }

                # Add accuracy metrics if expected category is available
                if expected_category:
                    result["expected_category"] = expected_category
                    result["is_correct"] = predicted_category == expected_category

                return result, True

        except Exception as e:
            logger.error(f"Error in test_single_request: {e}")
            return {"error": str(e), "latency": time.time() - start_time}, False

    async def run_concurrent_test(
        self,
        concurrency: int = 10,
        duration: int = 60,
        ramp_up: int = 5,
        constant_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run a concurrent load test for specified duration

        Args:
            concurrency: Maximum number of concurrent requests
            duration: Test duration in seconds
            ramp_up: Time to ramp up to full concurrency in seconds
            constant_rate: If set, maintain constant request rate instead of concurrency

        Returns:
            Dictionary with test results
        """
        if not self.test_data:
            logger.error("No test data available. Cannot proceed with test.")
            return {}

        # Reset counters and metrics
        self.results = []
        self.error_count = 0
        self.success_count = 0
        self.latencies = {"total": [], "embedding": [], "llm": [], "bigquery": []}

        # Initialize accuracy tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        self.per_category_accuracy = {}

        logger.info(
            f"Starting concurrent test with concurrency={concurrency}, duration={duration}s"
        )

        start_time = time.time()
        end_time = start_time + duration

        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency)

        # Task queue and completion tracking
        tasks = set()
        request_count = 0

        # Progress bar
        progress = tqdm(total=duration, desc="Stress Test Progress", unit="s")
        last_update = start_time

        async def worker():
            nonlocal request_count

            while time.time() < end_time:
                # Calculate current concurrency based on ramp-up
                current_time = time.time()
                elapsed = current_time - start_time

                if elapsed < ramp_up:
                    # During ramp-up, gradually increase concurrency
                    current_max_concurrency = max(
                        1, int(concurrency * (elapsed / ramp_up))
                    )
                    if semaphore._value < concurrency - current_max_concurrency:
                        # Need to wait until semaphore value increases
                        await asyncio.sleep(0.1)
                        continue

                async with semaphore:
                    # Select a random sample from test data
                    sample = random.choice(self.test_data)
                    text = sample.get("text", "")
                    expected_category = sample.get("moderation_category")  # May be None

                    # Run the test request
                    result, success = await self.test_single_request(
                        text, expected_category
                    )
                    request_count += 1

                    if success:
                        self.success_count += 1
                        self.results.append(result)

                        # Track latencies
                        self.latencies["total"].append(result["latency"])
                        self.latencies["embedding"].append(
                            result.get("embedding_time", 0)
                        )
                        self.latencies["llm"].append(result.get("llm_time", 0))
                        self.latencies["bigquery"].append(
                            result.get("bigquery_time", 0)
                        )

                        # Track accuracy if expected category is available
                        if expected_category:
                            self.total_predictions += 1
                            predicted = result.get("category", "unknown")
                            is_correct = predicted == expected_category

                            if is_correct:
                                self.correct_predictions += 1

                            # Per-category metrics
                            if expected_category not in self.per_category_accuracy:
                                self.per_category_accuracy[expected_category] = {
                                    "correct": 0,
                                    "total": 0,
                                    "accuracy": 0.0,
                                }

                            self.per_category_accuracy[expected_category]["total"] += 1
                            if is_correct:
                                self.per_category_accuracy[expected_category][
                                    "correct"
                                ] += 1
                    else:
                        self.error_count += 1

                # For constant rate, calculate sleep time
                if constant_rate is not None:
                    sleep_time = 1.0 / constant_rate
                    await asyncio.sleep(sleep_time)

        # Start workers
        num_workers = (
            concurrency * 2
        )  # Create more workers than concurrency to ensure consistent load
        for _ in range(num_workers):
            task = asyncio.create_task(worker())
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        # Update progress bar
        while time.time() < end_time:
            current = time.time()
            progress.update(min(current - last_update, end_time - last_update))
            last_update = current
            await asyncio.sleep(0.1)

        progress.close()

        # Wait for remaining tasks to complete
        # Cancel any tasks that haven't started yet
        remaining_tasks = list(tasks)
        for task in remaining_tasks:
            task.cancel()

        if remaining_tasks:
            await asyncio.gather(*remaining_tasks, return_exceptions=True)

        # Calculate metrics
        total_time = time.time() - start_time

        # Calculate per-category accuracy percentages
        for category in self.per_category_accuracy:
            cat_data = self.per_category_accuracy[category]
            if cat_data["total"] > 0:
                cat_data["accuracy"] = cat_data["correct"] / cat_data["total"] * 100

        metrics = self._calculate_metrics(total_time, request_count)
        metrics["concurrency"] = concurrency
        metrics["duration"] = duration
        metrics["ramp_up"] = ramp_up

        # Add accuracy metrics
        if self.total_predictions > 0:
            metrics["accuracy"] = (
                self.correct_predictions / self.total_predictions * 100
            )
            metrics["per_category_accuracy"] = self.per_category_accuracy

        logger.info(
            f"Test completed: {request_count} total requests, {self.success_count} successful, {self.error_count} errors"
        )
        logger.info(
            f"Average latency: {metrics['average_latency']:.2f}ms, Throughput: {metrics['requests_per_second']:.2f} req/s"
        )

        if self.total_predictions > 0:
            logger.info(f"Overall accuracy: {metrics['accuracy']:.2f}%")

        return metrics

    def _calculate_metrics(
        self, total_time: float, request_count: int
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics

        Args:
            total_time: Total test time in seconds
            request_count: Number of requests completed

        Returns:
            Dictionary with calculated metrics
        """
        if not self.latencies["total"]:
            return {
                "requests_per_second": 0,
                "average_latency": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0,
                "error_rate": 1.0 if request_count > 0 else 0,
            }

        total_latencies = sorted(self.latencies["total"])
        embedding_latencies = sorted(self.latencies["embedding"])
        llm_latencies = sorted(self.latencies["llm"])
        bigquery_latencies = sorted(self.latencies["bigquery"])

        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            return data[int(len(data) * p / 100)]

        metrics = {
            # Throughput
            "requests_per_second": request_count / total_time,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "total_requests": request_count,
            # Overall latency
            "average_latency": (
                statistics.mean(total_latencies) if total_latencies else 0
            ),
            "p50_latency": percentile(total_latencies, 50),
            "p95_latency": percentile(total_latencies, 95),
            "p99_latency": percentile(total_latencies, 99),
            # Component latencies
            "avg_embedding_time": (
                statistics.mean(embedding_latencies) if embedding_latencies else 0
            ),
            "avg_llm_time": statistics.mean(llm_latencies) if llm_latencies else 0,
            "avg_bigquery_time": (
                statistics.mean(bigquery_latencies) if bigquery_latencies else 0
            ),
            # Error rate
            "error_rate": self.error_count / request_count if request_count > 0 else 0,
        }

        # Add accuracy metrics if we have expected categories
        if hasattr(self, "total_predictions") and self.total_predictions > 0:
            metrics["accuracy"] = (
                self.correct_predictions / self.total_predictions * 100
            )
            metrics["per_category_accuracy"] = self.per_category_accuracy

        return metrics

    async def run_scaling_test(
        self,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64],
        duration_per_level: int = 30,
        cooldown: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run tests at different concurrency levels to measure scaling

        Args:
            concurrency_levels: List of concurrency levels to test
            duration_per_level: Duration to test each level in seconds
            cooldown: Cooldown time between tests in seconds

        Returns:
            Dictionary with test results for each concurrency level
        """
        if not self.test_data:
            logger.error("No test data available. Please provide a JSONL file.")
            return {"scaling_results": []}

        scaling_results = []

        for concurrency in concurrency_levels:
            logger.info(f"=== Testing concurrency level: {concurrency} ===")

            # Run test at this concurrency level
            metrics = await self.run_concurrent_test(
                concurrency=concurrency, duration=duration_per_level
            )

            # Store results
            scaling_results.append(metrics)

            # Save intermediate results
            self._save_scaling_results(scaling_results)

            # Cooldown between tests
            if concurrency != concurrency_levels[-1]:
                logger.info(f"Cooling down for {cooldown}s before next test...")
                await asyncio.sleep(cooldown)

        # Generate summary visualizations
        self._generate_scaling_visualizations(scaling_results)

        return {"scaling_results": scaling_results}

    def _save_scaling_results(self, scaling_results: List[Dict[str, Any]]) -> str:
        """
        Save scaling test results

        Args:
            scaling_results: List of metrics for each concurrency level

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scaling_test_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(scaling_results, f, indent=2)

        logger.info(f"Scaling results saved to {filepath}")
        return str(filepath)

    def _save_results(self, metrics: Dict[str, Any], prefix: str = "") -> str:
        """
        Save test results to disk

        Args:
            metrics: Dictionary with test metrics
            prefix: Optional prefix for filename

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}stress_test_{timestamp}.json"
        filepath = self.output_dir / filename

        # Add raw data for detailed analysis
        result_data = {
            "metrics": metrics,
            "raw_results": self.results[
                :1000
            ],  # Limit raw results to prevent huge files
            "test_config": {
                "server_url": self.server_url,
                "num_examples": self.num_examples,
                "max_input_length": self.max_input_length,
                "timestamp": timestamp,
            },
        }

        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    def _generate_accuracy_visualizations(
        self, scaling_results: List[Dict[str, Any]], output_dir: Optional[Path] = None
    ) -> None:
        """
        Generate visualizations for accuracy metrics

        Args:
            scaling_results: List of metrics for each concurrency level
            output_dir: Directory to save visualizations
        """
        try:
            # Extract accuracy data
            concurrency_levels = [r["concurrency"] for r in scaling_results]
            overall_accuracy = [r.get("accuracy", 0) for r in scaling_results]

            # Collect all unique categories
            all_categories = set()
            for result in scaling_results:
                if "per_category_accuracy" in result:
                    for category in result["per_category_accuracy"]:
                        all_categories.add(category)

            # Extract per-category accuracy
            category_data = {}
            for category in all_categories:
                category_data[category] = []
                for result in scaling_results:
                    if (
                        "per_category_accuracy" in result
                        and category in result["per_category_accuracy"]
                    ):
                        category_data[category].append(
                            result["per_category_accuracy"][category].get("accuracy", 0)
                        )
                    else:
                        category_data[category].append(0)

            # Set Seaborn style
            sns.set(style="whitegrid")

            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Overall accuracy vs Concurrency
            sns.lineplot(
                x=concurrency_levels,
                y=overall_accuracy,
                marker="o",
                ax=axes[0],
                label="Overall",
            )
            axes[0].set_title("Overall Accuracy vs Concurrency")
            axes[0].set_xlabel("Concurrency Level")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].set_ylim(0, 100)

            # Per-category accuracy vs Concurrency
            for category, data in category_data.items():
                if len(data) == len(concurrency_levels):
                    sns.lineplot(
                        x=concurrency_levels,
                        y=data,
                        marker="o",
                        label=category,
                        ax=axes[1],
                    )

            axes[1].set_title("Category Accuracy vs Concurrency")
            axes[1].set_xlabel("Concurrency Level")
            axes[1].set_ylabel("Accuracy (%)")
            axes[1].set_ylim(0, 100)
            axes[1].legend(loc="lower left", bbox_to_anchor=(1, 0.5))

            # Adjust layout and save
            plt.tight_layout()

            if output_dir is None:
                output_dir = self.output_dir

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"accuracy_visualization_{timestamp}.png"
            plt.savefig(filepath)
            logger.info(f"Accuracy visualization saved to {filepath}")

            plt.close(fig)

        except Exception as e:
            logger.error(f"Error generating accuracy visualizations: {e}")

    def _generate_scaling_visualizations(
        self, scaling_results: List[Dict[str, Any]]
    ) -> None:
        """
        Generate visualizations for scaling test results

        Args:
            scaling_results: List of metrics for each concurrency level
        """
        try:
            # Extract data for plotting
            concurrency_levels = [r["concurrency"] for r in scaling_results]
            throughput = [r["requests_per_second"] for r in scaling_results]
            avg_latency = [r["average_latency"] for r in scaling_results]
            p95_latency = [r["p95_latency"] for r in scaling_results]
            error_rates = [
                r["error_rate"] * 100 for r in scaling_results
            ]  # Convert to percentage

            # Set Seaborn style
            sns.set(style="whitegrid")

            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Throughput vs Concurrency
            sns.lineplot(x=concurrency_levels, y=throughput, marker="o", ax=axes[0, 0])
            axes[0, 0].set_title("Throughput vs Concurrency")
            axes[0, 0].set_xlabel("Concurrency Level")
            axes[0, 0].set_ylabel("Requests per Second")

            # Latency vs Concurrency
            sns.lineplot(
                x=concurrency_levels,
                y=avg_latency,
                marker="o",
                label="Average",
                ax=axes[0, 1],
            )
            sns.lineplot(
                x=concurrency_levels,
                y=p95_latency,
                marker="s",
                label="95th Percentile",
                ax=axes[0, 1],
            )
            axes[0, 1].set_title("Latency vs Concurrency")
            axes[0, 1].set_xlabel("Concurrency Level")
            axes[0, 1].set_ylabel("Latency (ms)")
            axes[0, 1].legend()

            # Error Rate vs Concurrency
            sns.lineplot(
                x=concurrency_levels,
                y=error_rates,
                marker="o",
                color="red",
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("Error Rate vs Concurrency")
            axes[1, 0].set_xlabel("Concurrency Level")
            axes[1, 0].set_ylabel("Error Rate (%)")

            # Component latencies
            components = ["avg_embedding_time", "avg_llm_time", "avg_bigquery_time"]
            component_labels = ["Embedding", "LLM", "BigQuery"]
            component_data = {
                label: [r.get(comp, 0) for r in scaling_results]
                for comp, label in zip(components, component_labels)
            }

            for label, data in component_data.items():
                sns.lineplot(
                    x=concurrency_levels, y=data, marker="o", label=label, ax=axes[1, 1]
                )

            axes[1, 1].set_title("Component Latencies vs Concurrency")
            axes[1, 1].set_xlabel("Concurrency Level")
            axes[1, 1].set_ylabel("Latency (ms)")
            axes[1, 1].legend()

            # Adjust layout and save
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"scaling_visualization_{timestamp}.png"
            plt.savefig(filepath)
            logger.info(f"Scaling visualization saved to {filepath}")

            plt.close(fig)

            # Generate accuracy visualizations if we have accuracy data
            if any("accuracy" in r for r in scaling_results):
                self._generate_accuracy_visualizations(scaling_results)

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")


async def check_server_health(server_url: str, api_key: Optional[str] = None, timeout: int = 5) -> bool:
    """
    Check if server is healthy

    Args:
        server_url: URL of the server
        api_key: API key for authorization
        timeout: Timeout for request in seconds

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    logger.error(f"Health check failed with status {response.status}")
                    return False

                data = await response.json()
                status = data.get("status", "")

                if status != "healthy":
                    logger.error(f"Server status is {status}, not healthy")
                    return False

                return True
    except Exception as e:
        logger.error(f"Error checking server health: {e}")
        return False


async def main():
    """Main entry point for stress test"""
    parser = argparse.ArgumentParser(
        description="Stress test for content moderation service"
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8000", help="URL of moderation server"
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to input data file (JSONL)"
    )
    parser.add_argument(
        "--output-dir", default="stress_test_results", help="Directory to save results"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--scaling-test",
        action="store_true",
        help="Run scaling test with multiple concurrency levels",
    )
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of concurrency levels",
    )
    parser.add_argument(
        "--duration-per-level",
        type=int,
        default=30,
        help="Duration for each concurrency level",
    )
    parser.add_argument(
        "--ramp-up", type=int, default=5, help="Ramp-up time in seconds"
    )
    parser.add_argument(
        "--cooldown", type=int, default=10, help="Cooldown between tests in seconds"
    )
    parser.add_argument(
        "--num-examples", type=int, default=3, help="Number of examples to use"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=2000, help="Maximum input length"
    )
    parser.add_argument(
        "--request-timeout", type=int, default=90, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--num-samples", type=int, help="Number of samples to use from the input file"
    )
    parser.add_argument(
        "--api-key", help="API key for authorization"
    )

    args = parser.parse_args()

    # Check if server is healthy
    logger.info(f"Checking if server at {args.server_url} is healthy...")
    server_healthy = await check_server_health(args.server_url, args.api_key)

    if not server_healthy:
        logger.error("Server is not healthy. Aborting stress test.")
        return 1

    # Initialize stress tester
    stress_tester = StressTester(
        server_url=args.server_url,
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        max_input_length=args.max_input_length,
        request_timeout=args.request_timeout,
        api_key=args.api_key,
    )

    # Apply sampling if specified
    # startified is hardcoded to true
    if args.input_file and args.num_samples:
        stress_tester.sample_test_data(args.num_samples, stratified=True)

    try:
        if args.scaling_test:
            # Parse concurrency levels
            concurrency_levels = [
                int(level) for level in args.concurrency_levels.split(",")
            ]

            # Run scaling test
            scaling_results = await stress_tester.run_scaling_test(
                concurrency_levels=concurrency_levels,
                duration_per_level=args.duration_per_level,
                cooldown=args.cooldown,
            )

            # Save final results
            stress_tester._save_results(scaling_results, prefix="scaling_")

        else:
            # Run single concurrent test
            metrics = await stress_tester.run_concurrent_test(
                concurrency=args.concurrency,
                duration=args.duration,
                ramp_up=args.ramp_up,
            )

            # Save results
            stress_tester._save_results(metrics)

    finally:
        # Close HTTP session
        await stress_tester.close()

    return 0


if __name__ == "__main__":
    asyncio.run(main())
