"""
Performance Visualization Module for Content Moderation System

This module provides visualization functionality for performance test results
from the content moderation system.
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class PerformanceVisualizer:
    """Visualizer for performance test results"""

    def __init__(
        self,
        results_file: Optional[Union[str, Path]] = None,
        scaling_report_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the performance visualizer

        Args:
            results_file: Path to performance results JSON file
            scaling_report_file: Path to scaling report JSON file
            output_dir: Directory to save visualizations
        """
        self.results_file = results_file
        self.scaling_report_file = scaling_report_file
        self.output_dir = output_dir or Path("performance_visualizations")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize data structures
        self.results = None
        self.summary = None
        self.scaling_report = None

        # Load data from files if provided
        if results_file:
            self.load_results(results_file)

        if scaling_report_file:
            self.load_scaling_report(scaling_report_file)

    def load_results(self, results_file: Union[str, Path]) -> None:
        """
        Load performance test results from JSON file

        Args:
            results_file: Path to results JSON file
        """
        try:
            with open(results_file, "r") as f:
                data = json.load(f)

            self.results = data.get("results", [])
            self.summary = data.get("summary", {})

            logger.info(f"Loaded performance results from {results_file}")
        except Exception as e:
            logger.error(f"Error loading performance results: {e}")
            raise

    def load_scaling_report(self, scaling_report_file: Union[str, Path]) -> None:
        """
        Load scaling test report from JSON file

        Args:
            scaling_report_file: Path to scaling report JSON file
        """
        try:
            with open(scaling_report_file, "r") as f:
                self.scaling_report = json.load(f)

            logger.info(f"Loaded scaling report from {scaling_report_file}")
        except Exception as e:
            logger.error(f"Error loading scaling report: {e}")
            raise

    def generate_latency_histogram(
        self, filename: str = "latency_histogram.png"
    ) -> str:
        """
        Generate histogram of request latencies

        Args:
            filename: Name of the output file

        Returns:
            Path to the generated plot
        """
        if not self.results:
            logger.error("No results available for generating latency histogram")
            return ""

        # Extract latency data
        latencies = [r["latency"] for r in self.results]

        if not latencies:
            logger.error("No latency data available")
            return ""

        # Set plot style
        plt.style.use("ggplot")
        sns.set(style="whitegrid")

        # Create figure
        plt.figure(figsize=(10, 6))
        sns.histplot(latencies, kde=True)
        plt.title("Latency Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency histogram saved to {output_path}")
        return str(output_path)

    def generate_latency_cdf(self, filename: str = "latency_cdf.png") -> str:
        """
        Generate cumulative distribution function (CDF) of request latencies

        Args:
            filename: Name of the output file

        Returns:
            Path to the generated plot
        """
        if not self.results:
            logger.error("No results available for generating latency CDF")
            return ""

        # Extract latency data
        latencies = [r["latency"] for r in self.results]

        if not latencies:
            logger.error("No latency data available")
            return ""

        # Set plot style
        plt.style.use("ggplot")
        sns.set(style="whitegrid")

        # Create figure
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(latencies)
        plt.title("Latency Cumulative Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency CDF saved to {output_path}")
        return str(output_path)

    def generate_throughput_vs_concurrency(
        self, filename: str = "throughput_vs_concurrency.png"
    ) -> str:
        """
        Generate plot of throughput vs concurrency

        Args:
            filename: Name of the output file

        Returns:
            Path to the generated plot
        """
        if not self.scaling_report:
            logger.error(
                "No scaling report available for generating throughput vs concurrency plot"
            )
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        throughput_values = self.scaling_report.get("throughput_values", [])

        if not concurrency_levels or not throughput_values:
            logger.error("No concurrency or throughput data available")
            return ""

        # Set plot style
        plt.style.use("ggplot")

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(
            concurrency_levels,
            throughput_values,
            marker="o",
            linestyle="-",
            linewidth=2,
        )
        plt.title("Throughput vs Concurrency")
        plt.xlabel("Concurrency Level")
        plt.ylabel("Throughput (requests/second)")
        plt.grid(True)

        # Add annotations for max throughput
        max_idx = np.argmax(throughput_values)
        max_concurrency = concurrency_levels[max_idx]
        max_throughput = throughput_values[max_idx]

        plt.annotate(
            f"Max: {max_throughput:.2f} req/s at concurrency {max_concurrency}",
            xy=(max_concurrency, max_throughput),
            xytext=(max_concurrency + 1, max_throughput + 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
        )

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Throughput vs concurrency plot saved to {output_path}")
        return str(output_path)

    def generate_latency_vs_concurrency(
        self, filename: str = "latency_vs_concurrency.png"
    ) -> str:
        """
        Generate plot of latency vs concurrency

        Args:
            filename: Name of the output file

        Returns:
            Path to the generated plot
        """
        if not self.scaling_report:
            logger.error(
                "No scaling report available for generating latency vs concurrency plot"
            )
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        latency_values = self.scaling_report.get("latency_values", [])

        if not concurrency_levels or not latency_values:
            logger.error("No concurrency or latency data available")
            return ""

        # Set plot style
        plt.style.use("ggplot")

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(
            concurrency_levels,
            latency_values,
            marker="o",
            linestyle="-",
            linewidth=2,
            color="red",
        )
        plt.title("Average Latency vs Concurrency")
        plt.xlabel("Concurrency Level")
        plt.ylabel("Average Latency (seconds)")
        plt.grid(True)

        # Add annotations for min latency
        min_idx = np.argmin(latency_values)
        min_concurrency = concurrency_levels[min_idx]
        min_latency = latency_values[min_idx]

        plt.annotate(
            f"Min: {min_latency:.3f}s at concurrency {min_concurrency}",
            xy=(min_concurrency, min_latency),
            xytext=(min_concurrency + 1, min_latency + 0.05),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3),
        )

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency vs concurrency plot saved to {output_path}")
        return str(output_path)

    def generate_all_plots(self) -> Dict[str, str]:
        """
        Generate all available plots

        Returns:
            Dictionary mapping plot types to file paths
        """
        plots = {}

        # Generate plots from performance test results
        if self.results:
            plots["latency_histogram"] = self.generate_latency_histogram()
            plots["latency_cdf"] = self.generate_latency_cdf()

        # Generate plots from scaling test results
        if self.scaling_report:
            plots["throughput_vs_concurrency"] = (
                self.generate_throughput_vs_concurrency()
            )
            plots["latency_vs_concurrency"] = self.generate_latency_vs_concurrency()

        return plots

    def generate_markdown_report(
        self, output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a comprehensive performance report in Markdown format

        Args:
            output_file: Path to output file (defaults to performance_report.md in output_dir)

        Returns:
            Path to the generated report
        """
        if not self.results and not self.scaling_report:
            logger.error("No results available to generate report")
            return ""

        output_file = output_file or Path(self.output_dir) / "performance_report.md"

        # Generate plots first
        plots = self.generate_all_plots()

        # Create report content
        report = [
            "# Content Moderation System Performance Report",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]

        # Add performance test results if available
        if self.summary:
            report.extend(
                [
                    "## Test Configuration",
                    f"- Number of test samples: {self.summary.get('total_samples', 'N/A')}",
                    f"- Concurrency: {self.summary.get('concurrency', 1)}",
                    "",
                    "## Performance Summary",
                    f"- Total test time: {self.summary.get('total_time_seconds', 0):.2f} seconds",
                    f"- Throughput: **{self.summary.get('throughput_requests_per_second', 0):.2f} requests/second**",
                    f"- Average latency: {self.summary.get('avg_latency_seconds', 0)*1000:.2f} ms",
                    f"- Median latency: {self.summary.get('median_latency_seconds', 0)*1000:.2f} ms",
                    f"- 95th percentile latency: {self.summary.get('p95_latency_seconds', 0)*1000:.2f} ms",
                    f"- 99th percentile latency: {self.summary.get('p99_latency_seconds', 0)*1000:.2f} ms",
                    f"- Min latency: {self.summary.get('min_latency_seconds', 0)*1000:.2f} ms",
                    f"- Max latency: {self.summary.get('max_latency_seconds', 0)*1000:.2f} ms",
                    "",
                ]
            )

            # Add latency distribution plots if available
            if "latency_histogram" in plots and "latency_cdf" in plots:
                report.extend(
                    [
                        "## Latency Distribution",
                        "![Latency Histogram](latency_histogram.png)",
                        "",
                        "![Latency CDF](latency_cdf.png)",
                        "",
                    ]
                )

        # Add scaling results if available
        if self.scaling_report:
            report.extend(
                [
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
                ]
            )

            # Add scaling data table
            concurrency_levels = self.scaling_report.get("concurrency_levels", [])
            throughput_values = self.scaling_report.get("throughput_values", [])
            latency_values = self.scaling_report.get("latency_values", [])

            for i, concurrency in enumerate(concurrency_levels):
                throughput = throughput_values[i]
                latency = latency_values[i] * 1000  # Convert to ms
                report.append(f"| {concurrency} | {throughput:.2f} | {latency:.2f} |")

            report.append("")

        # Add performance recommendations
        report.extend(
            [
                "## System Recommendations",
                "",
            ]
        )

        # Add specific recommendations based on results
        if self.scaling_report:
            # Find optimal concurrency (highest throughput)
            concurrency_levels = self.scaling_report.get("concurrency_levels", [])
            throughput_values = self.scaling_report.get("throughput_values", [])

            if concurrency_levels and throughput_values:
                optimal_concurrency_index = np.argmax(throughput_values)
                optimal_concurrency = concurrency_levels[optimal_concurrency_index]
                max_throughput = throughput_values[optimal_concurrency_index]

                report.extend(
                    [
                        f"1. **Optimal Concurrency**: The system performs best with a concurrency level of {optimal_concurrency}, "
                        f"achieving a throughput of {max_throughput:.2f} requests/second.",
                        "",
                        f"2. **Estimated Capacity**: Based on the maximum throughput, a single server instance can handle "
                        f"approximately {int(max_throughput * 3600)} requests per hour.",
                        "",
                    ]
                )

                # Check if throughput plateaus or decreases with higher concurrency
                if optimal_concurrency_index < len(concurrency_levels) - 1:
                    report.append(
                        "3. **Scaling Limitation**: The throughput appears to plateau or decrease with higher concurrency levels, "
                        "suggesting resource contention. Consider optimizing the server or adding more resources."
                    )
                else:
                    report.append(
                        "3. **Good Scaling**: The system scales well with increased concurrency. For higher throughput, "
                        "consider deploying multiple server instances."
                    )
        elif self.summary:
            report.extend(
                [
                    f"1. **Current Throughput**: The system can handle {self.summary.get('throughput_requests_per_second', 0):.2f} "
                    f"requests/second, or approximately {int(self.summary.get('throughput_requests_per_second', 0) * 3600)} requests per hour.",
                    "",
                    "2. **Concurrency Testing**: Consider running concurrency scaling tests to determine the optimal "
                    "number of concurrent requests the system can handle.",
                ]
            )

        # Write report to file
        try:
            with open(output_file, "w") as f:
                f.write("\n".join(report))

            logger.info(f"Performance report saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return ""


def visualize_performance_results(
    results_file: Optional[Union[str, Path]] = None,
    scaling_report_file: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    generate_report: bool = True,
) -> Dict[str, Any]:
    """
    Visualize performance test results

    Args:
        results_file: Path to performance results JSON file
        scaling_report_file: Path to scaling report JSON file
        output_dir: Directory to save visualizations
        generate_report: Whether to generate a comprehensive report

    Returns:
        Dictionary with paths to generated files
    """
    # Validate inputs
    if not results_file and not scaling_report_file:
        return {"error": "Either results_file or scaling_report_file must be provided"}

    try:
        # Initialize visualizer
        visualizer = PerformanceVisualizer(
            results_file=results_file,
            scaling_report_file=scaling_report_file,
            output_dir=output_dir,
        )

        # Generate plots
        plots = visualizer.generate_all_plots()

        # Generate report if requested
        report_path = None
        if generate_report:
            report_path = visualizer.generate_markdown_report()

        return {
            "plots": plots,
            "report": report_path,
        }
    except Exception as e:
        logger.error(f"Error visualizing performance results: {e}")
        return {"error": f"Error visualizing performance results: {e}"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize performance test results")
    parser.add_argument(
        "--results-file", type=str, help="Path to performance results JSON file"
    )
    parser.add_argument(
        "--scaling-report-file", type=str, help="Path to scaling report JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="performance_visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Do not generate comprehensive report"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.results_file and not args.scaling_report_file:
        parser.error("Either --results-file or --scaling-report-file must be provided")

    # Run visualization
    result = visualize_performance_results(
        results_file=args.results_file,
        scaling_report_file=args.scaling_report_file,
        output_dir=args.output_dir,
        generate_report=not args.no_report,
    )

    # Print result
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print("Visualization completed successfully")
        if "plots" in result:
            print(f"Generated plots: {', '.join(result['plots'].keys())}")
        if "report" in result:
            print(f"Generated report: {result['report']}")
