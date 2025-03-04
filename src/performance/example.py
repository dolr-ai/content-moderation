#!/usr/bin/env python3
"""
Example script for using the performance testing module

This script demonstrates how to use the performance testing module
to analyze the throughput and latency of the moderation server.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.performance.performance_tester import PerformanceTester, run_performance_test
from src.performance.generate_test_data import generate_test_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def example_generate_test_data():
    """Example of generating test data"""
    # Generate test data
    output_file = "data/test_data.jsonl"
    generate_test_data(
        num_samples=1000,
        output_file=output_file,
        category_distribution={
            "clean": 0.4,
            "offensive_language": 0.2,
            "hate_or_discrimination": 0.1,
            "spam_or_scams": 0.1,
            "random": 0.2,
        },
    )
    logger.info(f"Generated test data: {output_file}")
    return output_file


def example_sequential_test(input_file):
    """Example of running a sequential test"""
    # Initialize tester
    tester = PerformanceTester(
        server_url="http://localhost:8000",
        input_file=input_file,
        output_dir="performance_results/sequential",
        num_examples=3,
    )

    # Run sequential test
    logger.info("Running sequential test...")
    tester.run_sequential_test(num_samples=100)

    # Save results
    results_path = tester.save_results()
    logger.info(f"Results saved to: {results_path}")

    # Generate report
    report_path = tester.generate_report()
    logger.info(f"Report saved to: {report_path}")


def example_concurrent_test(input_file):
    """Example of running a concurrent test"""
    # Initialize tester
    tester = PerformanceTester(
        server_url="http://localhost:8000",
        input_file=input_file,
        output_dir="performance_results/concurrent",
        num_examples=3,
    )

    # Run concurrent test
    logger.info("Running concurrent test...")
    tester.run_concurrent_test(num_samples=100, concurrency=10)

    # Save results
    results_path = tester.save_results()
    logger.info(f"Results saved to: {results_path}")

    # Generate report
    report_path = tester.generate_report()
    logger.info(f"Report saved to: {report_path}")


def example_scaling_test(input_file):
    """Example of running a scaling test"""
    # Initialize tester
    tester = PerformanceTester(
        server_url="http://localhost:8000",
        input_file=input_file,
        output_dir="performance_results/scaling",
        num_examples=3,
    )

    # Run scaling test
    logger.info("Running scaling test...")
    scaling_report = tester.run_concurrency_scaling_test(
        num_samples=100,
        concurrency_levels=[1, 2, 4, 8, 16],
    )

    # Save results
    report_path = tester.save_scaling_report()
    logger.info(f"Scaling report saved to: {report_path}")

    # Generate report
    report_path = tester.generate_report()
    logger.info(f"Report saved to: {report_path}")


def example_using_helper_function(input_file):
    """Example of using the helper function"""
    # Run performance test using the helper function
    logger.info("Running performance test using helper function...")
    results = run_performance_test(
        input_file=input_file,
        server_url="http://localhost:8000",
        output_dir="performance_results/helper",
        num_examples=3,
        test_type="concurrent",
        num_samples=100,
        concurrency=10,
    )

    logger.info(f"Results: {results}")


def main():
    """Main function"""
    # Check if the moderation server is running
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("Moderation server is not running or not healthy")
            logger.error(f"Response: {response.status_code} - {response.text}")
            return
    except Exception as e:
        logger.error(f"Error connecting to moderation server: {e}")
        logger.error("Please make sure the moderation server is running")
        return

    # Generate test data if it doesn't exist
    input_file = "data/test_data.jsonl"
    if not os.path.exists(input_file):
        input_file = example_generate_test_data()

    # Run examples
    example_sequential_test(input_file)
    example_concurrent_test(input_file)
    example_scaling_test(input_file)
    example_using_helper_function(input_file)


if __name__ == "__main__":
    main()