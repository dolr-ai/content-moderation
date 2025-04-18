# %% [markdown]
# # Model Performance Comparison Analysis for RAG + examples system
# This notebook compares the performance of old and new model predictions against actual categories.

# %% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import Markdown, display


# %% Data Loading
def load_data(jsonl_path):
    """Load data from JSONL file"""
    return pd.read_json(jsonl_path, lines=True)


# %% Basic Performance Metrics
def calculate_basic_metrics(df):
    """Calculate basic performance metrics excluding error category"""
    # Print category distributions first
    print("\nCategory Distributions:")
    print("\nActual Categories:")
    print(df["actual_category"].value_counts())
    print("\nOld Model Predictions:")
    print(df["old_prediction"].value_counts())
    print("\nNew Model Predictions:")
    print(df["new_prediction"].value_counts())

    # Filter out error predictions
    df_old = df[df["old_prediction"] != "error"].copy()
    df_new = df[df["new_prediction"] != "error"].copy()

    # Calculate accuracy on filtered data
    metrics = {
        "accuracy": {
            "old": (
                (df_old["old_prediction"] == df_old["actual_category"]).mean()
                if len(df_old) > 0
                else 0
            ),
            "new": (
                (df_new["new_prediction"] == df_new["actual_category"]).mean()
                if len(df_new) > 0
                else 0
            ),
        }
    }

    # Get valid categories (excluding error)
    valid_categories = sorted(set(df["actual_category"].unique()) - {"error"})

    # Classification reports with filtered data and valid categories
    metrics["old_report"] = pd.DataFrame(
        classification_report(
            df_old["actual_category"],
            df_old["old_prediction"],
            output_dict=True,
            zero_division=0,
            labels=valid_categories,
        )
    ).transpose()

    metrics["new_report"] = pd.DataFrame(
        classification_report(
            df_new["actual_category"],
            df_new["new_prediction"],
            output_dict=True,
            zero_division=0,
            labels=valid_categories,
        )
    ).transpose()

    # Calculate improvements
    metrics["improvements"] = (
        metrics["new_report"].loc[:, ["precision", "recall", "f1-score"]]
        - metrics["old_report"].loc[:, ["precision", "recall", "f1-score"]]
    )

    # Print error prediction statistics separately
    error_stats = {
        "old": (df["old_prediction"] == "error").sum(),
        "new": (df["new_prediction"] == "error").sum(),
    }
    if error_stats["old"] > 0 or error_stats["new"] > 0:
        print("\nError Predictions:")
        print(
            f"Old model errors: {error_stats['old']} ({error_stats['old']/len(df)*100:.2f}%)"
        )
        print(
            f"New model errors: {error_stats['new']} ({error_stats['new']/len(df)*100:.2f}%)"
        )

    return metrics


# %% Confusion Matrix Visualization
def plot_confusion_matrices(df):
    """Plot confusion matrices for both models with proper category handling, excluding error category"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Increased figure size

    # Get all possible categories from both predictions and actual, excluding 'error'
    all_categories = sorted(
        set(
            list(df["actual_category"].unique())
            + list(df["old_prediction"].unique())
            + list(df["new_prediction"].unique())
        )
        - {"error"}
    )  # Exclude error category

    # Filter out error predictions for confusion matrix calculation
    df_old = df[df["old_prediction"] != "error"].copy()
    df_new = df[df["new_prediction"] != "error"].copy()

    # Create confusion matrices with all categories except error
    cm_old = confusion_matrix(
        df_old["actual_category"], df_old["old_prediction"], labels=all_categories
    )
    cm_new = confusion_matrix(
        df_new["actual_category"], df_new["new_prediction"], labels=all_categories
    )

    # Plot with consistent categories
    sns.heatmap(
        cm_old,
        annot=True,
        fmt="d",
        ax=ax1,
        cmap="Blues",
        xticklabels=all_categories,
        yticklabels=all_categories,
    )
    ax1.set_title("Old Model Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)

    sns.heatmap(
        cm_new,
        annot=True,
        fmt="d",
        ax=ax2,
        cmap="Blues",
        xticklabels=all_categories,
        yticklabels=all_categories,
    )
    ax2.set_title("New Model Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

    # Print error category statistics separately
    if (
        "error" in df["old_prediction"].unique()
        or "error" in df["new_prediction"].unique()
    ):
        print("\nError Category Statistics:")
        print(f"Old model error predictions: {(df['old_prediction'] == 'error').sum()}")
        print(f"New model error predictions: {(df['new_prediction'] == 'error').sum()}")


# %% Improvements Visualization
def plot_improvements(improvements):
    """Plot heatmap of metric improvements, excluding error category"""
    # Remove summary metrics and error category
    improvements_clean = improvements.drop(
        ["accuracy", "macro avg", "weighted avg", "error"], errors="ignore"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        improvements_clean,
        annot=True,
        cmap="RdYlGn",
        center=0,
        fmt=".3f",
        cbar_kws={"label": "Improvement (New - Old)"},
    )
    plt.title("Metric Improvements by Category\n(Green = Better, Red = Worse)")
    plt.ylabel("Categories")
    plt.xlabel("Metrics")
    plt.tight_layout()
    plt.show()

    # Print error category improvements separately if it exists
    if "error" in improvements.index:
        print("\nError Category Metrics Change:")
        print(improvements.loc["error"])


# %% Error Analysis
def analyze_prediction_changes(df):
    """Analyze improvements and deteriorations in predictions"""
    # Create boolean masks first
    old_incorrect = df["old_prediction"] != df["actual_category"]
    new_correct = df["new_prediction"] == df["actual_category"]
    old_correct = df["old_prediction"] == df["actual_category"]
    new_incorrect = df["new_prediction"] != df["actual_category"]

    # Apply masks using loc
    improvements = df.loc[old_incorrect & new_correct]
    deteriorations = df.loc[old_correct & new_incorrect]

    return improvements, deteriorations


# %% Category-wise Analysis
def analyze_categories(df):
    """Perform detailed category-wise analysis"""
    # Calculate category distribution
    category_distribution = df["actual_category"].value_counts()

    # Calculate precision and recall per category using classification reports
    old_report = pd.DataFrame(
        classification_report(
            df["actual_category"], df["old_prediction"], output_dict=True
        )
    ).transpose()

    new_report = pd.DataFrame(
        classification_report(
            df["actual_category"], df["new_prediction"], output_dict=True
        )
    ).transpose()

    # Remove summary metrics
    old_report = old_report.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )
    new_report = new_report.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )

    return {
        "category_distribution": category_distribution,
        "old_metrics": old_report,
        "new_metrics": new_report,
        "consistently_wrong": df[
            (df["old_prediction"] != df["actual_category"])
            & (df["new_prediction"] != df["actual_category"])
        ],
        "error_transitions": pd.crosstab(
            df[df["old_prediction"] != df["actual_category"]]["old_prediction"],
            df[df["old_prediction"] != df["actual_category"]]["new_prediction"],
            margins=True,
        ),
    }


# %% Category Visualization
def plot_category_analysis(category_metrics):
    """Plot category-wise analysis visualizations"""
    # Category Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=category_metrics["category_distribution"].index,
        y=category_metrics["category_distribution"].values,
    )
    plt.title("Distribution of Categories in Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Get common categories between old and new metrics
    common_categories = sorted(
        set(category_metrics["old_metrics"].index)
        & set(category_metrics["new_metrics"].index)
    )

    # Remove summary rows if present
    common_categories = [
        cat
        for cat in common_categories
        if cat not in ["accuracy", "macro avg", "weighted avg"]
    ]

    # Precision and Recall Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    width = 0.35

    x = np.arange(len(common_categories))

    # Precision plot
    old_precision = [
        category_metrics["old_metrics"].loc[cat, "precision"]
        for cat in common_categories
    ]
    new_precision = [
        category_metrics["new_metrics"].loc[cat, "precision"]
        for cat in common_categories
    ]

    ax1.bar(x - width / 2, old_precision, width, label="Old Model")
    ax1.bar(x + width / 2, new_precision, width, label="New Model")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision Comparison by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_categories, rotation=45)
    ax1.legend()

    # Recall plot
    old_recall = [
        category_metrics["old_metrics"].loc[cat, "recall"] for cat in common_categories
    ]
    new_recall = [
        category_metrics["new_metrics"].loc[cat, "recall"] for cat in common_categories
    ]

    ax2.bar(x - width / 2, old_recall, width, label="Old Model")
    ax2.bar(x + width / 2, new_recall, width, label="New Model")
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall Comparison by Category")
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_categories, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print categories that differ between models
    old_cats = set(category_metrics["old_metrics"].index) - {
        "accuracy",
        "macro avg",
        "weighted avg",
    }
    new_cats = set(category_metrics["new_metrics"].index) - {
        "accuracy",
        "macro avg",
        "weighted avg",
    }

    if old_cats != new_cats:
        print("\nCategory differences between models:")
        print(f"Categories only in old model: {old_cats - new_cats}")
        print(f"Categories only in new model: {new_cats - old_cats}")


# %% Misclassification Analysis
def analyze_misclassifications(df):
    """Analyze and report misclassifications for both models"""

    # Create detailed misclassification reports
    def get_misclassifications(predictions, actual):
        misclassified = df[
            predictions != actual
        ].copy()  # Create a copy to avoid warnings
        false_positives = {}
        false_negatives = {}

        for category in df["actual_category"].unique():
            # False Positives: Predicted as category X but actually something else
            fps = misclassified.loc[
                (predictions == category) & (actual != category)
            ].copy()
            if len(fps) > 0:
                false_positives[category] = fps

            # False Negatives: Actually category X but predicted as something else
            fns = misclassified.loc[
                (predictions != category) & (actual == category)
            ].copy()
            if len(fns) > 0:
                false_negatives[category] = fns

        return false_positives, false_negatives

    # Analyze both models
    old_fp, old_fn = get_misclassifications(df["old_prediction"], df["actual_category"])
    new_fp, new_fn = get_misclassifications(df["new_prediction"], df["actual_category"])

    return {
        "old_model": {"false_positives": old_fp, "false_negatives": old_fn},
        "new_model": {"false_positives": new_fp, "false_negatives": new_fn},
    }


def print_misclassification_report(misclassification_results):
    """Generate detailed misclassification report in markdown format"""
    markdown_report = []

    # for model_name in ["old_model", "new_model"]:
    for model_name in ["new_model"]:
        markdown_report.append(f"\n# {model_name.upper()} Analysis\n")

        # Get all unique categories
        all_categories = set()
        for fp in misclassification_results[model_name]["false_positives"].keys():
            all_categories.add(fp)
        for fn in misclassification_results[model_name]["false_negatives"].keys():
            all_categories.add(fn)

        # Sort categories for consistent output
        for category in sorted(all_categories):
            markdown_report.append(f"# {category}\n")

            # False Positives Section
            markdown_report.append(
                f"## False Positives (Incorrectly classified as {category} category)\n"
            )
            if category in misclassification_results[model_name]["false_positives"]:
                cases = misclassification_results[model_name]["false_positives"][
                    category
                ]
                actual_counts = cases["actual_category"].value_counts()

                for actual_cat, count in actual_counts.items():
                    markdown_report.append(
                        f"### `{category}` Actually `{actual_cat}` ({count} cases)\n"
                    )
                    examples = cases[cases["actual_category"] == actual_cat].head(5)
                    for idx, example in examples.iterrows():
                        markdown_report.append(f"**Example {idx + 1}:**\n")
                        markdown_report.append(
                            f"```\n{example['text'][:2000]}...\n```\n"
                        )
            else:
                markdown_report.append("*No false positives in this category*\n")

            # False Negatives Section
            markdown_report.append(
                f"## False Negatives (Actually `{actual_cat}` category, but misclassified)\n"
            )
            if category in misclassification_results[model_name]["false_negatives"]:
                cases = misclassification_results[model_name]["false_negatives"][
                    category
                ]
                pred_counts = cases[
                    f"{model_name.split('_')[0]}_prediction"
                ].value_counts()

                for pred_cat, count in pred_counts.items():
                    markdown_report.append(
                        f"### `{category}` Predicted as `{pred_cat}` ({count} cases)\n"
                    )
                    examples = cases[
                        cases[f"{model_name.split('_')[0]}_prediction"] == pred_cat
                    ].head(5)
                    for idx, example in examples.iterrows():
                        markdown_report.append(f"**Example {idx + 1}:**\n")
                        markdown_report.append(
                            f"```\n{example['text'][:2000]}...\n```\n"
                        )
            else:
                markdown_report.append("*No false negatives in this category*\n")

            markdown_report.append("---\n")

    return "\n".join(markdown_report)


# %% Main Analysis
def run_analysis(jsonl_path):
    """Run complete analysis pipeline with enhanced error handling"""
    # Load data
    df = load_data(jsonl_path)

    # Calculate basic metrics
    basic_metrics = calculate_basic_metrics(df)

    # Print accuracy improvements
    print("\nOverall Accuracy (including error predictions):")
    print(f"Old Model: {basic_metrics['accuracy']['old']:.3f}")
    print(f"New Model: {basic_metrics['accuracy']['new']:.3f}")
    print(
        f"Improvement: {basic_metrics['accuracy']['new'] - basic_metrics['accuracy']['old']:.3f}"
    )

    # print("\nAccuracy excluding error predictions:")
    # print(f"Old Model: {basic_metrics['accuracy_no_errors']['old']:.3f}")
    # print(f"New Model: {basic_metrics['accuracy_no_errors']['new']:.3f}")
    # print(f"Improvement: {basic_metrics['accuracy_no_errors']['new'] - basic_metrics['accuracy_no_errors']['old']:.3f}")

    # Plot basic visualizations
    plot_confusion_matrices(df)
    plot_improvements(basic_metrics["improvements"])

    # Analyze prediction changes
    improvements, deteriorations = analyze_prediction_changes(df)
    print(f"\nNumber of improved cases: {len(improvements)}")
    print(f"Number of deteriorated cases: {len(deteriorations)}")

    # Category analysis
    category_metrics = analyze_categories(df)
    plot_category_analysis(category_metrics)

    # Add misclassification analysis
    misclassification_results = analyze_misclassifications(df)
    markdown_report = print_misclassification_report(misclassification_results)
    # display(Markdown("# Misclassification Analysis Report"))
    # display(Markdown(markdown_report))

    return {
        "basic_metrics": basic_metrics,
        "improvements": improvements,
        "deteriorations": deteriorations,
        "category_metrics": category_metrics,
        "misclassification_results": misclassification_results,
    }


# %% Execute Analysis
if __name__ == "__main__":
    jsonl_path = "/root/content-moderation/data/benchmark_results/rag/rag_before_after_20250225_145419.jsonl"
    results = run_analysis(jsonl_path)
