# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import yaml
import os

# Configuration
CONFIDENCE_THRESHOLD = 0.50

# Define the primary category mapping
PRIMARY_CATEGORY_MAP = {
    "clean": 0,
    "hate_or_discrimination": 1,
    "violence_or_threats": 2,
    "offensive_language": 3,
    "nsfw_content": 4,
    "spam_or_scams": 5,
}

# Load configuration
DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])

# Load Phi-3.5 results
llm_results_dir = DATA_ROOT / "benchmark_results" / "llm"
phi_results_path = next(llm_results_dir.glob("*phi*_benchmark_results_*.jsonl"))
results_df = pd.read_json(phi_results_path, lines=True)


def process_model_results(df):
    """Process the raw results dataframe to extract scores and predictions."""
    df = df.copy()

    for category in PRIMARY_CATEGORY_MAP.keys():
        df[f"score_{category}"] = df["processed_response"].apply(
            lambda x: (
                x["scores"][category] if isinstance(x, dict) and "scores" in x else 0
            )
        )

    df["predicted_category"] = df["processed_response"].apply(
        lambda x: (
            x["predicted_category"]
            if isinstance(x, dict)
            and x["scores"][x["predicted_category"]] >= CONFIDENCE_THRESHOLD
            else "clean"
        )
    )

    return df


def calculate_evaluation_metrics(results_df):
    """Calculate comprehensive evaluation metrics for moderation results."""
    y_true = results_df["actual_category"].map(PRIMARY_CATEGORY_MAP)
    y_pred = results_df["predicted_category"].map(PRIMARY_CATEGORY_MAP)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(PRIMARY_CATEGORY_MAP.values()), zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    overall_accuracy = accuracy_score(y_true, y_pred)
    avg_latency = results_df["processing_time"].mean()
    p95_latency = results_df["processing_time"].quantile(0.95)
    error_rate = len(
        results_df[results_df["raw_llm_response"].apply(lambda x: "error" in str(x))]
    ) / len(results_df)

    metrics = {
        "overall": {
            "accuracy": overall_accuracy,
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": f1.mean(),
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "error_rate": error_rate,
        },
        "per_category": {
            category: {
                "precision": p,
                "recall": r,
                "f1": f,
                "support": s,
                "accuracy": acc,
            }
            for category, p, r, f, s, acc in zip(
                PRIMARY_CATEGORY_MAP.keys(),
                precision,
                recall,
                f1,
                support,
                per_class_accuracy,
            )
        },
        "confusion_matrix": conf_matrix,
    }

    return metrics


def generate_markdown_report(df, metrics_dict):
    """Generate a markdown report for metrics and error analysis"""
    report = []

    # Header
    report.append("# Model Performance Report: Phi-3.5\n")

    # Metrics Section
    report.append("## Performance Metrics\n")
    report.append("### Overall Metrics")
    report.append("| Metric | Value |")
    report.append("|--------|--------|")
    for metric, value in metrics_dict["overall"].items():
        report.append(f"| {metric} | {value:.3f} |")

    report.append("\n### Per-Category Metrics")
    report.append("| Category | Precision | Recall | F1 | Support | Accuracy |")
    report.append("|----------|-----------|---------|-----|----------|-----------|")
    for category, metrics in metrics_dict["per_category"].items():
        report.append(
            f"| {category} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} | {metrics['accuracy']:.3f} |"
        )

    # Error Analysis
    report.append("\n## Error Analysis\n")

    total_samples = len(df)
    misclassified = df[df["actual_category"] != df["predicted_category"]]
    total_misclassified = len(misclassified)

    report.append(f"Total samples: {total_samples}")
    report.append(
        f"Misclassified samples: {total_misclassified} ({total_misclassified/total_samples*100:.2f}%)\n"
    )

    # Analyze each category
    for category in PRIMARY_CATEGORY_MAP.keys():
        if category == "clean":
            continue

        report.append(f"### Analysis for {category.upper()}")

        false_negatives = df[
            (df["actual_category"] == category) & (df["predicted_category"] == "clean")
        ]
        if len(false_negatives) > 0:
            report.append(
                f"\n#### False Negatives (actual={category}, predicted=clean): {len(false_negatives)}"
            )
            report.append("```")
            examples = false_negatives.sample(min(5, len(false_negatives)))
            for _, row in examples.iterrows():
                report.append(f"Text: {row['text'][:750].strip()}")
                if "score_clean" in row:
                    report.append(f"Confidence Score: {row['score_clean']:.3f}")
                report.append("-" * 80)
            report.append("```\n")

    return "\n".join(report)


def generate_category_error_report(df, category):
    """Generate detailed error analysis report for a specific category"""
    report = []

    report.append(f"# Error Analysis for {category.upper()}\n")

    # False Negatives (actual=category, predicted=clean)
    if category != "clean":
        false_negatives = df[
            (df["actual_category"] == category) & (df["predicted_category"] == "clean")
        ]
        if len(false_negatives) > 0:
            report.append(
                f"## False Negatives (actual={category}, predicted=clean): {len(false_negatives)}\n"
            )
            report.append("```")
            # Sample up to 5 false negatives
            for _, row in false_negatives.sample(
                min(5, len(false_negatives))
            ).iterrows():
                report.append(f"Text: {row['text'][:750].strip()}")
                if "score_clean" in row:
                    report.append(f"Confidence Score: {row['score_clean']:.3f}")
                report.append("-" * 80)
            report.append("```\n")

    # False Positives (actual=clean/other, predicted=category)
    false_positives = df[
        (df["actual_category"] != category) & (df["predicted_category"] == category)
    ]
    if len(false_positives) > 0:
        report.append(
            f"## False Positives (predicted={category}): {len(false_positives)}\n"
        )
        report.append("### Breakdown by actual category:")
        for actual_cat in false_positives["actual_category"].unique():
            specific_fps = false_positives[
                false_positives["actual_category"] == actual_cat
            ]
            report.append(f"\n#### Actual: {actual_cat} (Count: {len(specific_fps)})")
            report.append("```")
            # Sample up to 5 false positives for each actual category
            for _, row in specific_fps.sample(min(5, len(specific_fps))).iterrows():
                report.append(f"Text: {row['text'][:750].strip()}")
                if f"score_{category}" in row:
                    report.append(f"Confidence Score: {row[f'score_{category}']:.3f}")
                report.append("-" * 80)
            report.append("```\n")

    return "\n".join(report)


# Process results and generate reports
processed_df = process_model_results(results_df)
metrics = calculate_evaluation_metrics(processed_df)

# Generate and save main summary report
report = generate_markdown_report(processed_df, metrics)
output_dir = PROJECT_ROOT / "docs/llm-misclassifications"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "phi35_report.md"

with open(output_path, "w") as f:
    f.write(report)

# Create phi-specific directory and generate detailed category reports
phi_output_dir = output_dir / "phi35"
phi_output_dir.mkdir(parents=True, exist_ok=True)

# Generate individual category reports
for category in PRIMARY_CATEGORY_MAP.keys():
    category_report = generate_category_error_report(processed_df, category)
    category_file = phi_output_dir / f"{category}.md"

    with open(category_file, "w") as f:
        f.write(category_report)

print(f"Main report saved to: {output_path}")
print(f"Detailed category reports saved in: {phi_output_dir}")
