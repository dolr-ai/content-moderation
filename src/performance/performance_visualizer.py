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
        """Generate plot of throughput vs concurrency level using Seaborn"""
        if not self.scaling_report:
            logger.error("No scaling report available for generating throughput plot")
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        throughput_values = self.scaling_report.get("throughput_values", [])

        if not concurrency_levels or not throughput_values:
            logger.error("No concurrency or throughput data available")
            return ""

        # Create figure with larger size
        plt.figure(figsize=(12, 8))

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {
                "Concurrency": concurrency_levels,
                "Throughput": throughput_values,
            }
        )

        # Create plot
        ax = sns.lineplot(
            x="Concurrency",
            y="Throughput",
            data=df,
            marker="o",
            color="royalblue",
            linewidth=2.5,
            markersize=10,
        )

        # Fill area under curve
        plt.fill_between(
            concurrency_levels, throughput_values, alpha=0.1, color="royalblue"
        )

        # Find optimal concurrency (highest throughput)
        max_idx = np.argmax(throughput_values)
        max_concurrency = concurrency_levels[max_idx]
        max_throughput = throughput_values[max_idx]

        # Mark optimal concurrency
        plt.axvline(
            x=max_concurrency,
            color="darkslategray",
            linestyle="--",
            label=f"Optimal Concurrency: {max_concurrency}",
            linewidth=2,
        )

        # Add annotation for optimal point
        plt.annotate(
            f"Optimal Point\nConcurrency: {max_concurrency}\nThroughput: {max_throughput:.2f} req/s",
            xy=(max_concurrency, max_throughput),
            xytext=(30, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, pad=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=12,
        )

        # Set titles and labels with larger font sizes
        plt.title("Throughput vs Concurrency Level", fontsize=16, pad=20)
        plt.xlabel(
            "Concurrency Level (Number of Concurrent Requests)",
            fontsize=14,
            labelpad=10,
        )
        plt.ylabel("Throughput (requests/second)", fontsize=14, labelpad=10)

        # Set tick sizes
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add grid for better readability
        plt.grid(True, linestyle="--", alpha=0.6)

        # Format x-axis to show only integer values
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Add legend with larger font
        plt.legend(fontsize=12, framealpha=0.9)

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save figure with higher DPI
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Throughput vs concurrency plot saved to {output_path}")
        return str(output_path)

    def generate_latency_vs_concurrency(
        self, filename: str = "latency_vs_concurrency.png"
    ) -> str:
        """Generate plot of latency vs concurrency level using Seaborn"""
        if not self.scaling_report:
            logger.error("No scaling report available for generating latency plot")
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        latency_values = self.scaling_report.get("latency_values", [])

        if not concurrency_levels or not latency_values:
            logger.error("No concurrency or latency data available")
            return ""

        # Create figure with larger size
        plt.figure(figsize=(12, 8))

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {
                "Concurrency": concurrency_levels,
                "Latency": latency_values,
            }
        )

        # Create plot
        ax = sns.lineplot(
            x="Concurrency",
            y="Latency",
            data=df,
            marker="s",
            color="crimson",
            linewidth=2.5,
            markersize=10,
        )

        # Fill area under curve
        plt.fill_between(concurrency_levels, latency_values, alpha=0.1, color="crimson")

        # Find minimum latency point
        min_idx = np.argmin(latency_values)
        min_concurrency = concurrency_levels[min_idx]
        min_latency = latency_values[min_idx]

        # Mark minimum latency point
        plt.axvline(
            x=min_concurrency,
            color="darkslategray",
            linestyle="--",
            label=f"Min Latency Concurrency: {min_concurrency}",
            linewidth=2,
        )

        # Add annotation for minimum latency point
        plt.annotate(
            f"Minimum Latency Point\nConcurrency: {min_concurrency}\nLatency: {min_latency:.3f}s",
            xy=(min_concurrency, min_latency),
            xytext=(30, 30),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, pad=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=12,
        )

        # Set titles and labels with larger font sizes
        plt.title("Latency vs Concurrency Level", fontsize=16, pad=20)
        plt.xlabel(
            "Concurrency Level (Number of Concurrent Requests)",
            fontsize=14,
            labelpad=10,
        )
        plt.ylabel("Average Latency (seconds)", fontsize=14, labelpad=10)

        # Set tick sizes
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add grid for better readability
        plt.grid(True, linestyle="--", alpha=0.6)

        # Format x-axis to show only integer values
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Add legend with larger font
        plt.legend(fontsize=12, framealpha=0.9)

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save figure with higher DPI
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=150)
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

        # Ensure all data is numeric types
        concurrency_levels = [int(c) for c in concurrency_levels]
        throughput_values = [float(t) for t in throughput_values]
        latency_values = [float(l) for l in latency_values]

        # Create figure with two y-axes - increase figure size and set better aspect ratio
        fig, ax1 = plt.subplots(figsize=(16, 9))
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
        throughput_line = sns.lineplot(
            x="Concurrency",
            y="Throughput",
            data=df,
            marker="o",
            ax=ax1,
            color="royalblue",
            label="Throughput",
            linewidth=3,
            markersize=12,
        )

        # Plot latency on second axis
        latency_line = sns.lineplot(
            x="Concurrency",
            y="Latency",
            data=df,
            marker="s",
            ax=ax2,
            color="crimson",
            label="Latency",
            linewidth=3,
            markersize=12,
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

        # Mark optimal concurrency with a vertical line
        optimal_line = ax1.axvline(
            x=max_concurrency,
            color="darkslategray",
            linestyle="--",
            linewidth=2,
        )

        # Add annotation for optimal point in a better position
        # Move annotation away from data points and lines
        annotation_x_offset = (concurrency_levels[-1] - concurrency_levels[0]) * 0.1
        annotation_y_offset = max_throughput * 0.2

        ax1.annotate(
            f"Optimal Point\nConcurrency: {max_concurrency}\nThroughput: {max_throughput:.2f} req/s\nLatency: {optimal_latency:.3f}s",
            xy=(max_concurrency, max_throughput),
            xytext=(
                max_concurrency - annotation_x_offset,
                max_throughput + annotation_y_offset,
            ),
            bbox=dict(boxstyle="round,pad=0.8", fc="white", alpha=0.9, ec="gray"),
            arrowprops=dict(
                arrowstyle="simple", connectionstyle="arc3,rad=0.2", fc="gray"
            ),
            fontsize=12,
            ha="center",
        )

        # Set titles and labels with larger font sizes
        ax1.set_title("System Performance Scaling Analysis", fontsize=18, pad=20)
        ax1.set_xlabel(
            "Concurrency Level (Number of Concurrent Requests)",
            fontsize=14,
            labelpad=10,
        )
        ax1.set_ylabel(
            "Throughput (requests/second)", color="royalblue", fontsize=14, labelpad=10
        )
        ax2.set_ylabel(
            "Average Latency (seconds)", color="crimson", fontsize=14, labelpad=10
        )

        # Set tick colors and sizes
        ax1.tick_params(axis="y", colors="royalblue", labelsize=12)
        ax2.tick_params(axis="y", colors="crimson", labelsize=12)
        ax1.tick_params(axis="x", labelsize=12)

        # Format x-axis to show only integer values
        ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Expand x-axis limits to provide more space
        current_xlim = ax1.get_xlim()
        padding = (current_xlim[1] - current_xlim[0]) * 0.1
        ax1.set_xlim(current_xlim[0] - padding, current_xlim[1] + padding)

        # Add grid for better readability
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Create custom legend with all elements but prevent duplicates
        legend_elements = [
            mpl.lines.Line2D(
                [0],
                [0],
                color="royalblue",
                lw=3,
                marker="o",
                markersize=8,
                label="Throughput",
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                color="crimson",
                lw=3,
                marker="s",
                markersize=8,
                label="Latency",
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                color="darkslategray",
                lw=2,
                linestyle="--",
                label=f"Optimal Concurrency: {max_concurrency}",
            ),
        ]

        # Add a single, clean legend to the plot
        ax1.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=12,
            framealpha=0.9,
            bbox_to_anchor=(0.01, 0.99),
        )

        # Remove the auto-generated legends
        if ax2.get_legend():
            ax2.get_legend().remove()

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save figure with higher DPI for better quality
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Combined scaling metrics plot saved to {output_path}")
        return str(output_path)

    def generate_timeouts_vs_concurrency(
        self, filename: str = "timeouts_vs_concurrency.png"
    ) -> str:
        """Generate plot of timeout counts vs concurrency level"""
        if not self.scaling_report:
            logger.error("No scaling report available for generating timeouts plot")
            return ""

        # Get data from scaling report
        concurrency_levels = self.scaling_report.get("concurrency_levels", [])
        timeout_counts = self.scaling_report.get("timeout_counts", [])
        timeout_rates = self.scaling_report.get("timeout_rates", [])

        if not concurrency_levels or not isinstance(timeout_counts, list):
            logger.error("No concurrency or timeout data available")
            return ""

        # Ensure all values are proper numeric types
        try:
            concurrency_levels = [int(c) for c in concurrency_levels]
            timeout_counts = [int(c) if c is not None else 0 for c in timeout_counts]
            timeout_rates = [float(r) if r is not None else 0.0 for r in timeout_rates]
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting data types for timeout plot: {e}")
            return ""

        # Check if there are any timeouts to display
        if sum(timeout_counts) == 0:
            logger.info("No timeouts to display in the visualization")
            # Create a simple plot showing no timeouts
            plt.figure(figsize=(16, 9))
            plt.bar(
                concurrency_levels, [0] * len(concurrency_levels), color="lightgray"
            )
            plt.title("Request Timeouts by Concurrency Level", fontsize=18, pad=20)
            plt.xlabel("Concurrency Level (Number of Concurrent Requests)", fontsize=14)
            plt.ylabel("Number of Timeouts", fontsize=14)
            plt.text(
                sum(concurrency_levels) / len(concurrency_levels),
                0.5,
                "No timeouts were observed during testing",
                ha="center",
                va="center",
                fontsize=16,
            )
            plt.tight_layout()
            output_path = Path(self.output_dir) / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(output_path)

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax2 = ax1.twinx()

        # Create DataFrame for Seaborn
        df = pd.DataFrame(
            {
                "Concurrency": concurrency_levels,
                "Timeout Count": timeout_counts,
                "Timeout Rate": [
                    rate * 100 for rate in timeout_rates
                ],  # Convert to percentage
            }
        )

        # Plot timeout counts on first axis
        sns.barplot(
            x="Concurrency",
            y="Timeout Count",
            data=df,
            ax=ax1,
            color="firebrick",
            alpha=0.7,
        )

        # Plot timeout rate on second axis
        sns.lineplot(
            x="Concurrency",
            y="Timeout Rate",
            data=df,
            marker="o",
            ax=ax2,
            color="darkred",
            label="Timeout Rate (%)",
            linewidth=3,
            markersize=12,
        )

        # Set titles and labels with larger font sizes
        ax1.set_title("Request Timeouts by Concurrency Level", fontsize=18, pad=20)
        ax1.set_xlabel(
            "Concurrency Level (Number of Concurrent Requests)",
            fontsize=14,
            labelpad=10,
        )
        ax1.set_ylabel(
            "Number of Timeouts", color="firebrick", fontsize=14, labelpad=10
        )
        ax2.set_ylabel("Timeout Rate (%)", color="darkred", fontsize=14, labelpad=10)

        # Set tick colors and sizes
        ax1.tick_params(axis="y", colors="firebrick", labelsize=12)
        ax2.tick_params(axis="y", colors="darkred", labelsize=12)
        ax1.tick_params(axis="x", labelsize=12)

        # Format x-axis to show only integer values
        ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Add legend for the line plot
        lines, labels = ax2.get_legend_handles_labels()
        ax2.legend(lines, labels, loc="upper right", fontsize=12, framealpha=0.9)

        # Add grid for better readability
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        # Annotate any points with high timeout rates
        if timeout_rates and any(
            rate > 0.05 for rate in timeout_rates
        ):  # More than 5% timeouts
            max_idx = np.argmax(timeout_rates)
            max_concurrency = concurrency_levels[max_idx]
            max_rate = timeout_rates[max_idx] * 100

            # Add annotation for highest timeout rate point
            ax2.annotate(
                f"High Timeout Rate: {max_rate:.1f}%",
                xy=(max_concurrency, max_rate),
                xytext=(10, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="darkred"),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                fontsize=12,
            )

        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save figure with higher DPI
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Timeouts vs concurrency plot saved to {output_path}")
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
            try:
                plots["latency_histogram"] = self.generate_latency_histogram()
                logger.info(f"Generated latency histogram")
            except Exception as e:
                logger.error(f"Error generating latency histogram: {e}")

            try:
                plots["latency_cdf"] = self.generate_latency_cdf()
                logger.info(f"Generated latency CDF")
            except Exception as e:
                logger.error(f"Error generating latency CDF: {e}")

        # Generate plots from scaling test results
        if self.scaling_report:
            try:
                plots["throughput_vs_concurrency"] = (
                    self.generate_throughput_vs_concurrency()
                )
                logger.info(f"Generated throughput vs concurrency plot")
            except Exception as e:
                logger.error(f"Error generating throughput vs concurrency plot: {e}")

            try:
                plots["latency_vs_concurrency"] = self.generate_latency_vs_concurrency()
                logger.info(f"Generated latency vs concurrency plot")
            except Exception as e:
                logger.error(f"Error generating latency vs concurrency plot: {e}")

            try:
                plots["combined_scaling_metrics"] = (
                    self.generate_combined_scaling_plot()
                )
                logger.info(f"Generated combined scaling metrics plot")
            except Exception as e:
                logger.error(f"Error generating combined scaling metrics plot: {e}")

            try:
                plots["timeouts_vs_concurrency"] = (
                    self.generate_timeouts_vs_concurrency()
                )
                logger.info(f"Generated timeouts vs concurrency plot")
            except Exception as e:
                logger.error(f"Error generating timeouts vs concurrency plot: {e}")

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
        try:
            if not self.results and not self.scaling_report:
                logger.error("No results available to generate report")
                return ""

            output_file = output_file or Path(self.output_dir) / "performance_report.md"

            # Generate plots first
            plots = self.generate_all_plots()
            if plots is None:
                plots = {}
                logger.warning("No plots were generated for the report")

            # Create a single string for the report content
            report_parts = []

            # Header and title
            report_parts.append("# Content Moderation System Performance Report\n")
            report_parts.append(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
            )

            # Executive Summary section
            report_parts.append("## Executive Summary\n")

            if self.summary:
                # Basic performance stats
                report_parts.append(
                    f"The content moderation system processed **{self.summary.get('total_samples', 'N/A')}** requests "
                    f"with an average latency of **{self.summary.get('avg_latency_seconds', 0)*1000:.2f} ms** "
                    f"and achieved a throughput of **{self.summary.get('throughput_requests_per_second', 0):.2f} requests/second**.\n"
                )

                # Timeout information if available
                if (
                    "timeout_count" in self.summary
                    and self.summary.get("timeout_count", 0) > 0
                ):
                    report_parts.append(
                        f"During testing, **{self.summary.get('timeout_count', 0)}** requests timed out, "
                        f"representing a timeout rate of **{self.summary.get('timeout_rate', 0)*100:.2f}%**.\n"
                    )

                # Percentile latency info
                if "p95_latency_seconds" in self.summary:
                    report_parts.append(
                        f"95% of requests were processed in under **{self.summary.get('p95_latency_seconds', 0)*1000:.2f} ms**.\n"
                    )

            # Add scaling summary if available
            if self.scaling_report:
                concurrency_levels = self.scaling_report.get("concurrency_levels", [])
                throughput_values = self.scaling_report.get("throughput_values", [])
                timeout_counts = self.scaling_report.get("timeout_counts", [])

                if (
                    concurrency_levels
                    and throughput_values
                    and len(concurrency_levels) > 0
                    and len(throughput_values) > 0
                ):
                    # Safely get max throughput index
                    if len(throughput_values) > 0:
                        max_idx = np.argmax(throughput_values)
                        if max_idx < len(concurrency_levels):
                            max_concurrency = concurrency_levels[max_idx]
                            max_throughput = throughput_values[max_idx]

                            report_parts.append(
                                f"The system's performance peaks at a concurrency level of **{max_concurrency}**, "
                                f"achieving a maximum throughput of **{max_throughput:.2f} requests/second**.\n"
                            )

                    # Add timeout information to executive summary if present
                    if timeout_counts:
                        total_timeouts = sum(timeout_counts)
                        if total_timeouts > 0:
                            report_parts.append(
                                f"**Important:** A total of **{total_timeouts}** timeouts were observed during the scaling tests. "
                                f"This indicates potential system limitations at higher concurrency levels.\n"
                            )

            # Join all parts to create the final report
            report_content = "\n".join(report_parts)

            # Write report to file
            with open(output_file, "w") as f:
                f.write(report_content)

            logger.info(f"Performance report saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            return ""
