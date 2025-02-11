# %%[markdown]
# # Detailed Metrics Analysis for LLM Moderation Models
# This notebook analyzes the detailed metrics from various LLM moderation model benchmark results.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import yaml
import glob
import json
import os

# %%[markdown]
# ## Configuration Parameters

# %%
# Confidence threshold for classification
# Any prediction with confidence score below this threshold will be classified as 'clean'
CONFIDENCE_THRESHOLD = 0.50

# %%
# Load Configuration and Data

# %%
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

# Get all LLM benchmark results
llm_results_dir = DATA_ROOT / "benchmark_results" / "llm"
model_files = {
    path.stem.split("_benchmark_results_")[0]: path
    for path in llm_results_dir.glob("*_benchmark_results_*.jsonl")
}
print(model_files)

# Create a dictionary to store dataframes for each model
results_dfs = {}

# Load data for each model
for model_name, file_path in model_files.items():
    print(f"Loading results for {model_name} from: {file_path}")
    results_dfs[model_name] = pd.read_json(file_path, lines=True)

# %%[markdown]
# ## Process Results


# %%
def process_model_results(df, model_name):
    """
    Process the raw results dataframe to extract scores and predictions.
    Handles different model response formats including LlamaGuard.

    Args:
        df (pd.DataFrame): Raw results dataframe
        model_name (str): Name of the model (to handle different formats)

    Returns:
        pd.DataFrame: Processed dataframe with extracted scores and predictions
    """
    df = df.copy()

    if "llamaguard" in model_name.lower():
        # Process LlamaGuard format
        for category in PRIMARY_CATEGORY_MAP.keys():
            df[f"score_{category}"] = df["processed_response"].apply(
                lambda x: (
                    (x.get("confidence", 0) if x.get("category") == category else 0)
                    if isinstance(x, dict)
                    else 0
                )
            )

        # Set predicted category based on LlamaGuard response
        df["predicted_category"] = df["processed_response"].apply(
            lambda x: (
                x.get("category", "clean")
                if isinstance(x, dict)
                and x.get("confidence", 0) >= CONFIDENCE_THRESHOLD
                else "clean"
            )
        )

        # Add any additional LlamaGuard-specific processing
        df["llamaguard_categories"] = df["processed_response"].apply(
            lambda x: (
                x.get("violated_llamaguard_categories", [])
                if isinstance(x, dict)
                else []
            )
        )

    else:
        # Process standard format (existing logic)
        for category in PRIMARY_CATEGORY_MAP.keys():
            df[f"score_{category}"] = df["processed_response"].apply(
                lambda x: (
                    x["scores"][category]
                    if isinstance(x, dict) and "scores" in x
                    else 0
                )
            )

        # Apply confidence threshold to predictions
        df["predicted_category"] = df["processed_response"].apply(
            lambda x: (
                x["predicted_category"]
                if isinstance(x, dict)
                and x["scores"][x["predicted_category"]] >= CONFIDENCE_THRESHOLD
                else "clean"
            )
        )

    return df


# Process results for each model
for model_name in results_dfs:
    print(f"Processing results for {model_name}")
    results_dfs[model_name] = process_model_results(results_dfs[model_name], model_name)

# %%
results_dfs["llama32-3B"].head()

# %%[markdown]
# ## Calculate Evaluation Metrics


# %%
def calculate_evaluation_metrics(results_df):
    """
    Calculate comprehensive evaluation metrics for moderation results.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Extract actual and predicted categories
    y_true = results_df["actual_category"].map(PRIMARY_CATEGORY_MAP)
    y_pred = results_df["predicted_category"].map(PRIMARY_CATEGORY_MAP)

    # Calculate metrics for each category
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(PRIMARY_CATEGORY_MAP.values()), zero_division=0
    )

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Calculate average latency
    avg_latency = results_df["processing_time"].mean()
    p95_latency = results_df["processing_time"].quantile(0.95)

    # Calculate error rate
    error_rate = len(
        results_df[results_df["raw_llm_response"].apply(lambda x: "error" in str(x))]
    ) / len(results_df)

    # Prepare metrics dictionary
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


# Calculate metrics for each model
model_metrics = {
    model_name: calculate_evaluation_metrics(df)
    for model_name, df in results_dfs.items()
}

# %%[markdown]
# ## Visualization Functions


# %%
def plot_confusion_matrix(conf_matrix, categories, title):
    """
    Plot confusion matrix heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"{title} - Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()


def plot_model_comparison(model_metrics, metric_name):
    """
    Plot comparison of a specific metric across models.
    """
    models = list(model_metrics.keys())
    values = [metrics["overall"][metric_name] for metrics in model_metrics.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values)
    plt.title(f"Model Comparison - {metric_name.replace('_', ' ').title()}")
    plt.xticks(rotation=45)
    plt.ylabel(metric_name.replace("_", " ").title())

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()


def plot_precision_recall_matrices(metrics, categories, model_name):
    """
    Plot separate precision and recall matrices.

    Args:
        metrics (dict): Dictionary containing metrics data
        categories (list): List of category names
        model_name (str): Name of the model being analyzed
    """
    n_categories = len(categories)

    # Create matrices for precision and recall
    precision_matrix = np.zeros((n_categories, n_categories))
    recall_matrix = np.zeros((n_categories, n_categories))

    # Fill matrices using confusion matrix data
    conf_matrix = metrics["confusion_matrix"]
    for i in range(n_categories):
        for j in range(n_categories):
            precision_matrix[i, j] = (
                conf_matrix[i, j] / conf_matrix[:, j].sum()
                if conf_matrix[:, j].sum() != 0
                else 0
            )
            recall_matrix[i, j] = (
                conf_matrix[i, j] / conf_matrix[i, :].sum()
                if conf_matrix[i, :].sum() != 0
                else 0
            )

    # Plot precision matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        precision_matrix,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Precision (%)"},
    )
    plt.title(f"Model: {model_name} - Precision Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tick_params(axis="x", rotation=45, labelsize=10)
    plt.tick_params(axis="y", rotation=45, labelsize=10)
    plt.tight_layout()
    plt.show()

    # Plot recall matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Recall (%)"},
    )
    plt.title(f"Model: {model_name} - Recall Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tick_params(axis="x", rotation=45, labelsize=10)
    plt.tick_params(axis="y", rotation=45, labelsize=10)
    plt.tight_layout()
    plt.show()


# %%[markdown]
# ## Analysis and Visualization

# %%
# Plot confusion matrices for each model
for model_name, metrics in model_metrics.items():
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        list(PRIMARY_CATEGORY_MAP.keys()),
        f"Model: {model_name}",
    )
    plt.show()

# Plot model comparisons for key metrics
key_metrics = ["accuracy", "macro_f1", "avg_latency", "p95_latency"]
for metric in key_metrics:
    plot_model_comparison(model_metrics, metric)
    plt.show()

# Plot precision and recall matrices for each model
for model_name, metrics in model_metrics.items():
    plot_precision_recall_matrices(
        metrics, list(PRIMARY_CATEGORY_MAP.keys()), model_name
    )

# %%[markdown]
# ## Print Summary Metrics

# %%
# Print detailed metrics for each model
for model_name, metrics in model_metrics.items():
    print(f"\n=== Model: {model_name} ===")
    print("\nOverall Metrics:")
    for metric, value in metrics["overall"].items():
        print(f"{metric}: {value:.4f}")

    print("\nPer-Category Metrics:")
    for category, category_metrics in metrics["per_category"].items():
        print(f"\n{category}:")
        for metric, value in category_metrics.items():
            print(f"  {metric}: {value:.4f}")

# %%[markdown]
# ## Error Analysis


# %%
def analyze_errors(df, model_name):
    """
    Analyze and print details about misclassifications, including examples.
    """
    print(f"\n=== Error Analysis for {model_name} ===")

    def print_examples(samples_df, category_from, category_to, n=5):
        """Helper function to print misclassification examples"""
        print(f"\n{category_from} → {category_to} Examples:")
        print("-" * 80)
        examples = samples_df.sample(min(n, len(samples_df)))
        for _, row in examples.iterrows():
            print(f"Text: {row['text']}")
            if "score_" + category_to in row:
                print(f"Confidence Score: {row['score_' + category_to]:.3f}")
            print("-" * 80)

    # Get misclassified samples
    misclassified = df[df["actual_category"] != df["predicted_category"]]

    total_samples = len(df)
    total_misclassified = len(misclassified)

    print(f"\nTotal samples: {total_samples}")
    print(
        f"Misclassified samples: {total_misclassified} ({total_misclassified/total_samples*100:.2f}%)"
    )

    # Analyze misclassifications by category
    for category in PRIMARY_CATEGORY_MAP.keys():
        if category == "clean":
            continue  # We'll handle clean category separately for better organization

        print(f"\n{'='*40}")
        print(f"Analysis for {category.upper()}")
        print(f"{'='*40}")

        # False Negatives (actual=category, predicted=clean)
        false_negatives = df[
            (df["actual_category"] == category) & (df["predicted_category"] == "clean")
        ]
        fn_count = len(false_negatives)

        if fn_count > 0:
            print(f"\nFalse Negatives (actual={category}, predicted=clean): {fn_count}")
            print_examples(false_negatives, category, "clean")

        # Other misclassifications
        other_misclassifications = df[
            (df["actual_category"] == category)
            & (df["predicted_category"] != category)
            & (df["predicted_category"] != "clean")
        ]

        if len(other_misclassifications) > 0:
            for pred_cat in other_misclassifications["predicted_category"].unique():
                specific_misclass = other_misclassifications[
                    other_misclassifications["predicted_category"] == pred_cat
                ]
                if len(specific_misclass) > 0:
                    print(f"\nMisclassified as {pred_cat}: {len(specific_misclass)}")
                    print_examples(specific_misclass, category, pred_cat)

    # Handle clean category separately
    print(f"\n{'='*40}")
    print("Analysis for CLEAN")
    print(f"{'='*40}")

    clean_misclassified = df[
        (df["actual_category"] == "clean") & (df["predicted_category"] != "clean")
    ]

    if len(clean_misclassified) > 0:
        for pred_cat in clean_misclassified["predicted_category"].unique():
            specific_misclass = clean_misclassified[
                clean_misclassified["predicted_category"] == pred_cat
            ]
            if len(specific_misclass) > 0:
                print(
                    f"\nFalse Positives (actual=clean, predicted={pred_cat}): {len(specific_misclass)}"
                )
                print_examples(specific_misclass, "clean", pred_cat)


# Run error analysis for each model
for model_name, df in results_dfs.items():
    analyze_errors(df, model_name)

# %% [markdown]
# ## Generate Markdown Report


def generate_markdown_report(df, model_name, metrics_dict):
    """
    Generate a markdown report for metrics and error analysis
    """
    report = []

    # Header
    report.append(f"# Model Performance Report: {model_name}\n")

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

    # Error Analysis Section
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

        # False Negatives
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
                report.append(f"Text: {row['text'][:1000]}")
                if "score_clean" in row:
                    report.append(f"Confidence Score: {row['score_clean']:.3f}")
                report.append("-" * 80)
            report.append("```\n")

        # Other misclassifications
        other_misclassifications = df[
            (df["actual_category"] == category)
            & (df["predicted_category"] != category)
            & (df["predicted_category"] != "clean")
        ]

        if len(other_misclassifications) > 0:
            for pred_cat in other_misclassifications["predicted_category"].unique():
                specific_misclass = other_misclassifications[
                    other_misclassifications["predicted_category"] == pred_cat
                ]
                if len(specific_misclass) > 0:
                    report.append(
                        f"\n#### Misclassified as {pred_cat}: {len(specific_misclass)}"
                    )
                    report.append("```")
                    examples = specific_misclass.sample(min(5, len(specific_misclass)))
                    for _, row in examples.iterrows():
                        report.append(f"Text: {row['text'][:1000]}")
                        if f"score_{pred_cat}" in row:
                            report.append(
                                f"Confidence Score: {row[f'score_{pred_cat}']:.3f}"
                            )
                        report.append("-" * 80)
                    report.append("```\n")

    # Clean category analysis
    report.append("### Analysis for CLEAN")
    clean_misclassified = df[
        (df["actual_category"] == "clean") & (df["predicted_category"] != "clean")
    ]

    if len(clean_misclassified) > 0:
        for pred_cat in clean_misclassified["predicted_category"].unique():
            specific_misclass = clean_misclassified[
                clean_misclassified["predicted_category"] == pred_cat
            ]
            if len(specific_misclass) > 0:
                report.append(
                    f"\n#### False Positives (actual=clean, predicted={pred_cat}): {len(specific_misclass)}"
                )
                report.append("```")
                examples = specific_misclass.sample(min(5, len(specific_misclass)))
                for _, row in examples.iterrows():
                    report.append(f"Text: {row['text'][:1000]}")
                    if f"score_{pred_cat}" in row:
                        report.append(
                            f"Confidence Score: {row[f'score_{pred_cat}']:.3f}"
                        )
                    report.append("-" * 80)
                report.append("```\n")

    return "\n".join(report)


def save_and_display_report(df, model_name, metrics_dict, project_root):
    """
    Generate, save, and display the markdown report
    """
    # Generate the report
    report = generate_markdown_report(df, model_name, metrics_dict)

    # Save to file
    output_dir = os.path.join(project_root, "docs/llm-misclassifications")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{model_name.lower().replace(' ', '_')}_report.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(report)

    print(f"Report saved to: {filepath}")

    # Display in notebook
    from IPython.display import display, Markdown

    display(Markdown(report))


# Update the main analysis function to use the new reporting
def analyze_results(df, model_name):
    """
    Analyze results and generate reports
    """
    metrics_dict = calculate_evaluation_metrics(df)
    save_and_display_report(df, model_name, metrics_dict, PROJECT_ROOT)


# %%[markdown]
# ## Generate and Save Markdown Reports

# %%
# Generate reports for each model
for model_name, df in results_dfs.items():
    print(f"\nGenerating report for {model_name}...")
    metrics = model_metrics[model_name]  # Get pre-calculated metrics
    save_and_display_report(df, model_name, metrics, PROJECT_ROOT)
    print(f"Completed report for {model_name}\n")
    print("-" * 80)  # Add separator between models

# Print locations of saved reports
print("\nAll reports have been saved to:")
for model_name in results_dfs.keys():
    filename = f"{model_name.lower().replace(' ', '_')}_report.md"
    filepath = os.path.join(PROJECT_ROOT, "docs/llm-misclassifications", filename)
    print(f"- {filepath}")
