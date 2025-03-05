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
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class PerformanceVisualizer:
    """Visualizer for performance test results with enhanced graphics"""

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

        # Set up matplotlib style for high quality output
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Set up the plot style for high quality visualizations using Seaborn"""
        # Use Seaborn's built-in styling
        sns.set_theme(style="whitegrid", context="talk", palette="deep")

        # Just a few customizations on top of Seaborn's defaults
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.pad_inches"] = 0.2

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
        """Generate histogram of request latencies using Seaborn"""
        if not self.results:
            logger.error("No results available for generating latency histogram")
            return ""

        # Extract latency data
        latencies = [r["latency"] for r in self.results]
        if not latencies:
            logger.error("No latency data available")
            return ""

        # Create figure with Seaborn
        plt.figure()

        # Use Seaborn's distplot with KDE
        ax = sns.histplot(latencies, kde=True, color="royalblue", stat="density")

        # Add percentile lines
        percentiles = {"median": 50, "p95": 95, "p99": 99}
        colors = {"median": "green", "p95": "orange", "p99": "red"}
        percentile_values = {}

        for name, p in percentiles.items():
            value = np.percentile(latencies, p)
            percentile_values[name] = value
            plt.axvline(
                x=value,
                color=colors[name],
                linestyle="--",
                label=f"{name}: {value:.3f}s",
            )

        # Add statistics text box
        mean = np.mean(latencies)
        std = np.std(latencies)
        stats_text = (
            f"Mean: {mean:.3f}s\n"
            f"Std Dev: {std:.3f}s\n"
            f"Median: {percentile_values['median']:.3f}s\n"
            f"95th: {percentile_values['p95']:.3f}s\n"
            f"99th: {percentile_values['p99']:.3f}s"
        )

        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

        # Set titles and labels
        plt.title("Request Latency Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Density")
        plt.legend()

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency histogram saved to {output_path}")
        return str(output_path)

    def generate_latency_cdf(self, filename: str = "latency_cdf.png") -> str:
        """Generate CDF of request latencies using Seaborn"""
        if not self.results:
            logger.error("No results available for generating latency CDF")
            return ""

        # Extract latency data
        latencies = [r["latency"] for r in self.results]
        if not latencies:
            logger.error("No latency data available")
            return ""

        # Create figure with Seaborn
        plt.figure()

        # Plot CDF using Seaborn
        ax = sns.ecdfplot(latencies, complementary=False)

        # Add percentile lines
        percentiles = {"median": 50, "p95": 95, "p99": 99}
        colors = {"median": "green", "p95": "orange", "p99": "red"}

        for name, p in percentiles.items():
            value = np.percentile(latencies, p)
            plt.axvline(x=value, color=colors[name], linestyle="--")
            plt.axhline(y=p / 100, color=colors[name], linestyle=":")

            # Add annotation
            plt.annotate(
                f"{name}: {value:.3f}s",
                xy=(value, p / 100),
                xytext=(10, 0),
                textcoords="offset points",
                color=colors[name],
                arrowprops=dict(arrowstyle="->", color=colors[name]),
            )

        # Set titles and labels
        plt.title("Latency Cumulative Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Cumulative Probability")

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency CDF saved to {output_path}")
        return str(output_path)

    def generate_throughput_vs_concurrency(
        self, filename: str = "throughput_vs_concurrency.png"
    ) -> str:
        """Generate plot of throughput vs concurrency using Seaborn"""
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

        # Create figure with Seaborn
        plt.figure()

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {"Concurrency": concurrency_levels, "Throughput": throughput_values}
        )

        # Plot with Seaborn
        ax = sns.lineplot(x="Concurrency", y="Throughput", data=df, marker="o")

        # Fill area under the curve
        plt.fill_between(concurrency_levels, throughput_values, alpha=0.2)

        # Add annotation for max throughput
        max_idx = np.argmax(throughput_values)
        max_concurrency = concurrency_levels[max_idx]
        max_throughput = throughput_values[max_idx]

        plt.annotate(
            f"Max: {max_throughput:.2f} req/s\nat concurrency {max_concurrency}",
            xy=(max_concurrency, max_throughput),
            xytext=(0, 25),
            textcoords="offset points",
            ha="center",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->"),
        )

        # Set titles and labels
        plt.title("System Throughput vs Concurrency")
        plt.xlabel("Concurrency Level (Number of Concurrent Requests)")
        plt.ylabel("Throughput (requests/second)")

        # Format x-axis to show only integer values
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Throughput vs concurrency plot saved to {output_path}")
        return str(output_path)

    def generate_latency_vs_concurrency(
        self, filename: str = "latency_vs_concurrency.png"
    ) -> str:
        """Generate plot of latency vs concurrency using Seaborn"""
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

        # Create figure with Seaborn
        plt.figure()

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {"Concurrency": concurrency_levels, "Latency": latency_values}
        )

        # Plot with Seaborn
        ax = sns.lineplot(
            x="Concurrency", y="Latency", data=df, marker="o", color="crimson"
        )

        # Fill area under the curve
        plt.fill_between(concurrency_levels, latency_values, alpha=0.2, color="crimson")

        # Add annotation for min latency
        min_idx = np.argmin(latency_values)
        min_concurrency = concurrency_levels[min_idx]
        min_latency = latency_values[min_idx]

        plt.annotate(
            f"Min: {min_latency:.3f}s\nat concurrency {min_concurrency}",
            xy=(min_concurrency, min_latency),
            xytext=(0, 25),
            textcoords="offset points",
            ha="center",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->"),
        )

        # Set titles and labels
        plt.title("Average Request Latency vs Concurrency")
        plt.xlabel("Concurrency Level (Number of Concurrent Requests)")
        plt.ylabel("Average Latency (seconds)")

        # Format x-axis to show only integer values
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Latency vs concurrency plot saved to {output_path}")
        return str(output_path)

    def generate_combined_scaling_plot(
        self, filename: str = "combined_scaling_metrics.png"
    ) -> str:
        """Generate combined plot with both throughput and latency vs concurrency using Seaborn"""
        if not self.scaling_report:
            logger.error("No scaling report available for generating combined plot")
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        throughput_values = self.scaling_report.get("throughput_values", [])
        latency_values = self.scaling_report.get("latency_values", [])

        if not concurrency_levels or not throughput_values or not latency_values:
            logger.error("No concurrency, throughput or latency data available")
            return ""

        # Create figure with two y-axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {
                "Concurrency": concurrency_levels,
                "Throughput": throughput_values,
                "Latency": latency_values,
            }
        )

        # Plot throughput on first axis
        sns.lineplot(
            x="Concurrency",
            y="Throughput",
            data=df,
            marker="o",
            ax=ax1,
            color="royalblue",
            label="Throughput",
        )

        # Plot latency on second axis
        sns.lineplot(
            x="Concurrency",
            y="Latency",
            data=df,
            marker="s",
            ax=ax2,
            color="crimson",
            label="Latency",
        )

        # Fill areas under curves
        ax1.fill_between(
            concurrency_levels, throughput_values, alpha=0.1, color="royalblue"
        )
        ax2.fill_between(concurrency_levels, latency_values, alpha=0.1, color="crimson")

        # Find optimal concurrency (highest throughput)
        max_idx = np.argmax(throughput_values)
        max_concurrency = concurrency_levels[max_idx]
        max_throughput = throughput_values[max_idx]
        optimal_latency = latency_values[max_idx]

        # Mark optimal concurrency
        ax1.axvline(
            x=max_concurrency,
            color="darkslategray",
            linestyle="--",
            label=f"Optimal Concurrency: {max_concurrency}",
        )

        # Add annotation for optimal point
        ax1.annotate(
            f"Optimal Point\nConcurrency: {max_concurrency}\nThroughput: {max_throughput:.2f} req/s\nLatency: {optimal_latency:.3f}s",
            xy=(max_concurrency, max_throughput),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->"),
        )

        # Set titles and labels
        ax1.set_title("System Performance Scaling Analysis")
        ax1.set_xlabel("Concurrency Level (Number of Concurrent Requests)")
        ax1.set_ylabel("Throughput (requests/second)", color="royalblue")
        ax2.set_ylabel("Average Latency (seconds)", color="crimson")

        # Set tick colors
        ax1.tick_params(axis="y", colors="royalblue")
        ax2.tick_params(axis="y", colors="crimson")

        # Format x-axis to show only integer values
        ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Save figure
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Combined scaling metrics plot saved to {output_path}")
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
            plots["combined_scaling_metrics"] = self.generate_combined_scaling_plot()

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

        # Create report content with enhanced styling
        report = [
            "# Content Moderation System Performance Report",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Executive Summary",
            "",
        ]

        # Add executive summary based on available data
        if self.summary:
            report.extend(
                [
                    f"The content moderation system processed **{self.summary.get('total_samples', 'N/A')}** requests ",
                    f"with an average latency of **{self.summary.get('avg_latency_seconds', 0)*1000:.2f} ms** ",
                    f"and achieved a throughput of **{self.summary.get('throughput_requests_per_second', 0):.2f} requests/second**.",
                    "",
                ]
            )

            # Add 95th percentile info if available
            if "p95_latency_seconds" in self.summary:
                report.append(
                    f"95% of requests were processed in under **{self.summary.get('p95_latency_seconds', 0)*1000:.2f} ms**."
                )
                report.append("")

        # Add scaling summary if available
        if self.scaling_report:
            concurrency_levels = self.scaling_report.get("concurrency_levels", [])
            throughput_values = self.scaling_report.get("throughput_values", [])

            if concurrency_levels and throughput_values:
                max_idx = np.argmax(throughput_values)
                max_concurrency = concurrency_levels[max_idx]
                max_throughput = throughput_values[max_idx]

                report.extend(
                    [
                        f"The system's performance peaks at a concurrency level of **{max_concurrency}**, ",
                        f"achieving a maximum throughput of **{max_throughput:.2f} requests/second**.",
                        "",
                    ]
                )

        # Add performance test results if available
        if self.summary:
            report.extend(
                [
                    "## Test Configuration",
                    f"- **Number of test samples:** {self.summary.get('total_samples', 'N/A')}",
                    f"- **Concurrency:** {self.summary.get('concurrency', 1)}",
                    f"- **Test duration:** {self.summary.get('total_time_seconds', 0):.2f} seconds",
                    "",
                    "## Performance Summary",
                    "",
                    "### Key Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Throughput | **{self.summary.get('throughput_requests_per_second', 0):.2f}** requests/second |",
                    f"| Average latency | {self.summary.get('avg_latency_seconds', 0)*1000:.2f} ms |",
                    f"| Median latency | {self.summary.get('median_latency_seconds', 0)*1000:.2f} ms |",
                    f"| 95th percentile latency | {self.summary.get('p95_latency_seconds', 0)*1000:.2f} ms |",
                    f"| 99th percentile latency | {self.summary.get('p99_latency_seconds', 0)*1000:.2f} ms |",
                    f"| Min latency | {self.summary.get('min_latency_seconds', 0)*1000:.2f} ms |",
                    f"| Max latency | {self.summary.get('max_latency_seconds', 0)*1000:.2f} ms |",
                    "",
                ]
            )

            # Add latency distribution plots if available
            if "latency_histogram" in plots and "latency_cdf" in plots:
                report.extend(
                    [
                        "### Latency Distribution",
                        "",
                        "The histogram below shows the distribution of request latencies across all test samples:",
                        "",
                        f"![Latency Histogram]({Path(plots['latency_histogram']).name})",
                        "",
                        "The cumulative distribution function (CDF) shows the percentage of requests completed within specific latency thresholds:",
                        "",
                        f"![Latency CDF]({Path(plots['latency_cdf']).name})",
                        "",
                        "**Key observations:**",
                        f"- **50% of requests** complete in under {self.summary.get('median_latency_seconds', 0)*1000:.2f} ms",
                        f"- **95% of requests** complete in under {self.summary.get('p95_latency_seconds', 0)*1000:.2f} ms",
                        f"- **99% of requests** complete in under {self.summary.get('p99_latency_seconds', 0)*1000:.2f} ms",
                        "",
                    ]
                )

        # Add scaling results if available
        if self.scaling_report:
            report.extend(
                [
                    "## Concurrency Scaling Analysis",
                    "",
                    "This section examines how system performance scales with increasing levels of concurrent requests.",
                    "",
                ]
            )

            # Add combined plot if available
            if "combined_scaling_metrics" in plots:
                report.extend(
                    [
                        "### Combined Performance Metrics",
                        "",
                        "The following graph shows throughput and latency trends as concurrency increases:",
                        "",
                        f"![Combined Scaling Metrics]({Path(plots['combined_scaling_metrics']).name})",
                        "",
                    ]
                )

            # Add individual plots if available
            if "throughput_vs_concurrency" in plots:
                report.extend(
                    [
                        "### Throughput Scaling",
                        "",
                        "This graph shows how system throughput changes with increasing concurrency:",
                        "",
                        f"![Throughput vs Concurrency]({Path(plots['throughput_vs_concurrency']).name})",
                        "",
                    ]
                )

            if "latency_vs_concurrency" in plots:
                report.extend(
                    [
                        "### Latency Impact",
                        "",
                        "This graph shows how request latency is affected by increasing concurrency:",
                        "",
                        f"![Latency vs Concurrency]({Path(plots['latency_vs_concurrency']).name})",
                        "",
                    ]
                )

            # Add scaling data table
            report.extend(
                [
                    "### Detailed Scaling Data",
                    "",
                    "| Concurrency | Throughput (req/s) | Avg Latency (ms) |",
                    "|-------------|-------------------|-----------------|",
                ]
            )

            concurrency_levels = self.scaling_report.get("concurrency_levels", [])
            throughput_values = self.scaling_report.get("throughput_values", [])
            latency_values = self.scaling_report.get("latency_values", [])

            for i, concurrency in enumerate(concurrency_levels):
                throughput = throughput_values[i]
                latency = latency_values[i] * 1000  # Convert to ms

                # Highlight optimal concurrency
                if throughput == max(throughput_values):
                    report.append(
                        f"| **{concurrency}** | **{throughput:.2f}** | {latency:.2f} |"
                    )
                else:
                    report.append(
                        f"| {concurrency} | {throughput:.2f} | {latency:.2f} |"
                    )

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
            latency_values = self.scaling_report.get("latency_values", [])

            if concurrency_levels and throughput_values:
                optimal_concurrency_index = np.argmax(throughput_values)
                optimal_concurrency = concurrency_levels[optimal_concurrency_index]
                max_throughput = throughput_values[optimal_concurrency_index]
                optimal_latency = latency_values[optimal_concurrency_index] * 1000  # ms

                report.extend(
                    [
                        f"### 1. Optimal Operating Point",
                        "",
                        f"- **Recommended concurrency level:** {optimal_concurrency}",
                        f"- **Expected throughput:** {max_throughput:.2f} requests/second",
                        f"- **Expected latency:** {optimal_latency:.2f} ms",
                        "",
                        f"The system performs best with **{optimal_concurrency}** concurrent requests, achieving a throughput of **{max_throughput:.2f} requests/second**.",
                        "",
                        f"### 2. Capacity Planning",
                        "",
                        f"- **Hourly capacity:** ~{int(max_throughput * 3600)} requests/hour",
                        f"- **Daily capacity:** ~{int(max_throughput * 3600 * 24)} requests/day",
                        "",
                    ]
                )

                # Check if throughput plateaus or decreases with higher concurrency
                if optimal_concurrency_index < len(concurrency_levels) - 1:
                    # Calculate how much throughput drops from peak to highest concurrency
                    throughput_drop_pct = (
                        (max_throughput - throughput_values[-1]) / max_throughput
                    ) * 100
                    latency_increase_pct = (
                        (latency_values[-1] - latency_values[optimal_concurrency_index])
                        / latency_values[optimal_concurrency_index]
                    ) * 100

                    if throughput_drop_pct > 10:
                        report.extend(
                            [
                                f"### 3. Scaling Limitations",
                                "",
                                f"- Throughput **decreases by {throughput_drop_pct:.1f}%** from peak when concurrency exceeds {optimal_concurrency}",
                                f"- Latency **increases by {latency_increase_pct:.1f}%** from optimal when reaching maximum tested concurrency",
                                "",
                                "**Recommendation:** The performance degradation at higher concurrency levels indicates resource contention. Consider:",
                                "",
                                "- Optimizing the server code to reduce CPU or memory bottlenecks",
                                "- Increasing available system resources (CPU, memory, or I/O capacity)",
                                "- Implementing a load balancer with multiple server instances to distribute traffic",
                                "- Adding a rate limiter to prevent exceeding optimal concurrency",
                                "",
                            ]
                        )
                    else:
                        report.extend(
                            [
                                f"### 3. Scaling Characteristics",
                                "",
                                "The system handles increasing concurrency well, with minimal performance degradation beyond the optimal point.",
                                "",
                                "**Recommendation:** For higher load requirements, consider horizontal scaling by deploying multiple server instances behind a load balancer.",
                                "",
                            ]
                        )
                else:
                    report.extend(
                        [
                            f"### 3. Further Testing Recommended",
                            "",
                            "The highest throughput was observed at the maximum tested concurrency level. To find the true performance limits:",
                            "",
                            "- Test with higher concurrency levels to find the peak throughput and saturation point",
                            "- Monitor system resources (CPU, memory, network) during high concurrency tests",
                            "- Consider conducting longer duration tests to evaluate stability under sustained load",
                            "",
                        ]
                    )
        elif self.summary:
            report.extend(
                [
                    f"### Current Performance",
                    "",
                    f"- The system currently handles **{self.summary.get('throughput_requests_per_second', 0):.2f} requests/second** under test conditions",
                    f"- This translates to approximately **{int(self.summary.get('throughput_requests_per_second', 0) * 3600)} requests per hour**",
                    "",
                    "### Recommendations",
                    "",
                    "1. **Concurrency Testing:** Conduct additional tests with varying concurrency levels to find the optimal operating point",
                    "2. **Resource Monitoring:** Add resource utilization monitoring during tests (CPU, memory, I/O) to identify bottlenecks",
                    "3. **Extended Testing:** Perform longer duration tests to evaluate system stability under sustained load",
                    "",
                ]
            )

        # Add conclusion
        report.extend(
            [
                "## Conclusion",
                "",
                "This performance analysis provides a baseline for understanding the content moderation system's capabilities and limitations.",
                "By operating the system at the recommended concurrency level and implementing the suggested optimizations,",
                "you can ensure optimal performance and reliability under production loads.",
                "",
                "---",
                f"*Report generated by Performance Visualization Module, {datetime.now().strftime('%Y-%m-%d')}*",
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
