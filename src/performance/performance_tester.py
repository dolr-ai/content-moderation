#!/usr/bin/env python3
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
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    ):
        """
        Initialize the performance tester

        Args:
            server_url: URL of the moderation server
            input_file: Path to input JSONL file with texts to test
            output_dir: Directory to save test results
            num_examples: Number of similar examples to use in moderation
        """
        self.server_url = server_url
        self.input_file = input_file
        self.output_dir = output_dir or Path("performance_results")
        self.num_examples = num_examples
        self.results = []

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load test data if provided
        self.test_data = []
        if input_file:
            self.load_test_data(input_file)

    def load_test_data(self, input_file: Union[str, Path]) -> None:
        """
        Load test data from JSONL file

        Args:
            input_file: Path to input JSONL file
        """
        try:
            df = pd.read_json(input_file, lines=True)
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

    def test_single_request(self, text: str) -> Tuple[Dict[str, Any], float]:
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
                json={"text": text, "num_examples": self.num_examples},
                timeout=30,
            )

            end_time = time.time()
            latency = end_time - start_time

            if response.status_code == 200:
                return response.json(), latency
            else:
                logger.error(f"Error in request: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}, latency

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(f"Exception in request: {e}")
            return {"error": str(e)}, latency

    def run_sequential_test(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
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
            response, latency = self.test_single_request(text)

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
        throughput = len(test_samples) / total_time

        # Add summary
        summary = {
            "total_samples": len(test_samples),
            "total_time_seconds": total_time,
            "throughput_requests_per_second": throughput,
            "avg_latency_seconds": np.mean([r["latency"] for r in results]),
            "median_latency_seconds": np.median([r["latency"] for r in results]),
            "p95_latency_seconds": np.percentile([r["latency"] for r in results], 95),
            "p99_latency_seconds": np.percentile([r["latency"] for r in results], 99),
            "min_latency_seconds": min([r["latency"] for r in results]),
            "max_latency_seconds": max([r["latency"] for r in results]),
        }

        logger.info(f"Sequential test completed: {summary}")

        self.results = results
        self.summary = summary

        return results

    def run_concurrent_test(
        self,
        num_samples: Optional[int] = None,
        concurrency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run concurrent test on the moderation server

        Args:
            num_samples: Number of samples to test (None for all)
            concurrency: Number of concurrent requests

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Create a list of futures
            futures = []
            for i, sample in enumerate(test_samples):
                text = sample["text"]
                future = executor.submit(self.test_single_request, text)
                futures.append((i, sample, future))

            # Process results as they complete
            for i, sample, future in tqdm(
                concurrent.futures.as_completed(
                    [f[2] for f in futures]
                ),
                total=len(futures),
                desc=f"Testing with {concurrency} concurrent requests"
            ):
                response, latency = future.result()

                result = {
                    "sample_id": i,
                    "text": sample["text"],
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
        throughput = len(test_samples) / total_time

        # Add summary
        summary = {
            "total_samples": len(test_samples),
            "total_time_seconds": total_time,
            "throughput_requests_per_second": throughput,
            "concurrency": concurrency,
            "avg_latency_seconds": np.mean([r["latency"] for r in results]),
            "median_latency_seconds": np.median([r["latency"] for r in results]),
            "p95_latency_seconds": np.percentile([r["latency"] for r in results], 95),
            "p99_latency_seconds": np.percentile([r["latency"] for r in results], 99),
            "min_latency_seconds": min([r["latency"] for r in results]),
            "max_latency_seconds": max([r["latency"] for r in results]),
        }

        logger.info(f"Concurrent test completed: {summary}")

        self.results = results
        self.summary = summary

        return results

    def run_concurrency_scaling_test(
        self,
        num_samples: int = 100,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64]
    ) -> Dict[str, Any]:
        """
        Run tests with different concurrency levels to analyze scaling

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
            self.run_concurrent_test(num_samples=num_samples, concurrency=concurrency)

            # Store summary results
            scaling_results[concurrency] = self.summary

        # Create scaling report
        throughput_values = [
            scaling_results[c]["throughput_requests_per_second"]
            for c in concurrency_levels
        ]

        latency_values = [
            scaling_results[c]["avg_latency_seconds"]
            for c in concurrency_levels
        ]

        scaling_report = {
            "concurrency_levels": concurrency_levels,
            "throughput_values": throughput_values,
            "latency_values": latency_values,
            "details": scaling_results,
        }

        self.scaling_report = scaling_report

        return scaling_report

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
                json.dump({
                    "summary": self.summary,
                    "results": self.results,
                }, f, indent=2)

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

    def generate_plots(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Generate performance plots

        Args:
            output_dir: Directory to save plots (defaults to self.output_dir)
        """
        if not self.results:
            logger.error("No results to plot")
            return

        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set plot style
        plt.style.use("ggplot")
        sns.set(style="whitegrid")

        # Extract latency data
        latencies = [r["latency"] for r in self.results]

        # Plot 1: Latency histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(latencies, kde=True)
        plt.title("Latency Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(Path(output_dir) / "latency_histogram.png")
        plt.close()

        # Plot 2: Latency CDF
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(latencies)
        plt.title("Latency Cumulative Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.savefig(Path(output_dir) / "latency_cdf.png")
        plt.close()

        # If we have scaling results, plot those too
        if hasattr(self, "scaling_report"):
            # Plot 3: Throughput vs Concurrency
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.scaling_report["concurrency_levels"],
                self.scaling_report["throughput_values"],
                marker="o",
                linestyle="-",
                linewidth=2,
            )
            plt.title("Throughput vs Concurrency")
            plt.xlabel("Concurrency Level")
            plt.ylabel("Throughput (requests/second)")
            plt.grid(True)
            plt.savefig(Path(output_dir) / "throughput_vs_concurrency.png")
            plt.close()

            # Plot 4: Latency vs Concurrency
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.scaling_report["concurrency_levels"],
                self.scaling_report["latency_values"],
                marker="o",
                linestyle="-",
                linewidth=2,
                color="red",
            )
            plt.title("Average Latency vs Concurrency")
            plt.xlabel("Concurrency Level")
            plt.ylabel("Average Latency (seconds)")
            plt.grid(True)
            plt.savefig(Path(output_dir) / "latency_vs_concurrency.png")
            plt.close()

        logger.info(f"Performance plots saved to {output_dir}")

    def generate_report(self, output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive performance report in Markdown format

        Args:
            output_file: Path to output file (defaults to performance_report.md in output_dir)

        Returns:
            Path to the generated report
        """
        if not self.results:
            logger.error("No results to generate report")
            return ""

        output_file = output_file or Path(self.output_dir) / "performance_report.md"

        # Generate plots first
        self.generate_plots()

        # Create report content
        report = [
            "# Content Moderation System Performance Report",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Test Configuration",
            f"- Server URL: {self.server_url}",
            f"- Number of test samples: {self.summary['total_samples']}",
            f"- Number of examples used in moderation: {self.num_examples}",
            "",
            "## Performance Summary",
            f"- Total test time: {self.summary['total_time_seconds']:.2f} seconds",
            f"- Throughput: **{self.summary['throughput_requests_per_second']:.2f} requests/second**",
            f"- Average latency: {self.summary['avg_latency_seconds']*1000:.2f} ms",
            f"- Median latency: {self.summary['median_latency_seconds']*1000:.2f} ms",
            f"- 95th percentile latency: {self.summary['p95_latency_seconds']*1000:.2f} ms",
            f"- 99th percentile latency: {self.summary['p99_latency_seconds']*1000:.2f} ms",
            f"- Min latency: {self.summary['min_latency_seconds']*1000:.2f} ms",
            f"- Max latency: {self.summary['max_latency_seconds']*1000:.2f} ms",
            "",
            "## Latency Distribution",
            "![Latency Histogram](latency_histogram.png)",
            "",
            "![Latency CDF](latency_cdf.png)",
            "",
        ]

        # Add scaling results if available
        if hasattr(self, "scaling_report"):
            report.extend([
                "## Concurrency Scaling Analysis",
                "### Throughput vs Concurrency",
                "![Throughput vs Concurrency](throughput_vs_concurrency.png)",
                "",
                "### Latency vs Concurrency",
                "![Latency vs Concurrency](latency_vs_concurrency.png)",
                "",
                "### Scaling Data",
                "| Concurrency | Throughput (req/s) | Avg Latency (ms) |",
                "|-------------|-------------------|-----------------|",
            ])

            for i, concurrency in enumerate(self.scaling_report["concurrency_levels"]):
                throughput = self.scaling_report["throughput_values"][i]
                latency = self.scaling_report["latency_values"][i] * 1000  # Convert to ms
                report.append(f"| {concurrency} | {throughput:.2f} | {latency:.2f} |")

            report.append("")

        # Add system recommendations
        report.extend([
            "## System Recommendations",
            "",
            "Based on the performance test results, here are some recommendations:",
            "",
        ])

        # Add specific recommendations based on results
        if hasattr(self, "scaling_report"):
            # Find optimal concurrency (highest throughput)
            optimal_concurrency_index = np.argmax(self.scaling_report["throughput_values"])
            optimal_concurrency = self.scaling_report["concurrency_levels"][optimal_concurrency_index]
            max_throughput = self.scaling_report["throughput_values"][optimal_concurrency_index]

            report.extend([
                f"1. **Optimal Concurrency**: The system performs best with a concurrency level of {optimal_concurrency}, "
                f"achieving a throughput of {max_throughput:.2f} requests/second.",
                "",
                f"2. **Estimated Capacity**: Based on the maximum throughput, a single server instance can handle "
                f"approximately {int(max_throughput * 3600)} requests per hour.",
                "",
            ])

            # Check if throughput plateaus or decreases with higher concurrency
            if optimal_concurrency_index < len(self.scaling_report["concurrency_levels"]) - 1:
                report.append(
                    "3. **Scaling Limitation**: The throughput appears to plateau or decrease with higher concurrency levels, "
                    "suggesting resource contention. Consider optimizing the server or adding more resources."
                )
            else:
                report.append(
                    "3. **Good Scaling**: The system scales well with increased concurrency. For higher throughput, "
                    "consider deploying multiple server instances."
                )
        else:
            report.extend([
                f"1. **Current Throughput**: The system can handle {self.summary['throughput_requests_per_second']:.2f} "
                f"requests/second, or approximately {int(self.summary['throughput_requests_per_second'] * 3600)} requests per hour.",
                "",
                "2. **Concurrency Testing**: Consider running concurrency scaling tests to determine the optimal "
                "number of concurrent requests the system can handle.",
            ])

        # Write report to file
        try:
            with open(output_file, "w") as f:
                f.write("\n".join(report))

            logger.info(f"Performance report saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return ""


def run_performance_test(
    input_file: Union[str, Path],
    server_url: str = "http://localhost:8000",
    output_dir: Optional[Union[str, Path]] = None,
    num_examples: int = 3,
    test_type: str = "sequential",
    num_samples: Optional[int] = None,
    concurrency: int = 10,
    run_scaling_test: bool = False,
    concurrency_levels: Optional[List[int]] = None,
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
        concurrency_levels: List of concurrency levels for scaling test

    Returns:
        Dictionary with test results
    """
    # Initialize tester
    tester = PerformanceTester(
        server_url=server_url,
        input_file=input_file,
        output_dir=output_dir,
        num_examples=num_examples,
    )

    # Check if server is available
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"Server health check failed: {response.status_code} - {response.text}")
            return {"error": "Server health check failed"}
    except Exception as e:
        logger.error(f"Server not available: {e}")
        return {"error": f"Server not available: {e}"}

    # Run appropriate test
    if run_scaling_test:
        # Use default concurrency levels if not specified
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8, 16, 32]

        # Run scaling test
        scaling_report = tester.run_concurrency_scaling_test(
            num_samples=num_samples or 100,
            concurrency_levels=concurrency_levels,
        )

        # Save results
        tester.save_scaling_report()

        # Generate report
        report_path = tester.generate_report()

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

        # Generate report
        report_path = tester.generate_report()

        return {
            "summary": tester.summary,
            "results_path": results_path,
            "report_path": report_path,
        }