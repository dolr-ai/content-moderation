# %% Imports
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

# %% Constants
CATEGORY_ORDER = [
    "clean",
    "hate_or_discrimination",
    "violence_or_threats",
    "offensive_language",
    "nsfw_content",
    "spam_or_scams",
]

VALID_CATEGORIES = set(CATEGORY_ORDER)


# %% Utility Functions
def load_data(jsonl_path: str) -> pd.DataFrame:
    """Load data from JSONL file and prepare it for analysis"""
    df = pd.read_json(jsonl_path, lines=True)
    # Handle None values in predictions
    df["new_primary"] = df["new_primary"].fillna("none")
    df["new_secondary"] = df["new_secondary"].fillna("none")
    df["old_prediction"] = df["old_prediction"].fillna("none")
    return df


def count_none_predictions(df: pd.DataFrame) -> Dict[str, int]:
    """Count occurrences of None/none in predictions"""
    return {
        "primary_none": sum(df["new_primary"].isin(["none", "None"])),
        "secondary_none": sum(df["new_secondary"].isin(["none", "None"])),
        "old_prediction_none": sum(df["old_prediction"].isin(["none", "None"])),
    }


# %% Evaluation Metrics Functions
def calculate_dual_prediction_metrics(
    df: pd.DataFrame, valid_categories: Set[str]
) -> pd.DataFrame:
    """Calculate metrics for dual prediction system (primary + secondary)"""
    metrics_data = []

    # First create combined predictions like in confusion matrix
    predictions = []
    for _, row in df.iterrows():
        if row["actual_category"] == row["new_primary"]:
            predictions.append(row["new_primary"])
        elif row["actual_category"] == row["new_secondary"]:
            predictions.append(row["new_secondary"])
        else:
            predictions.append(row["new_primary"])
    df["combined_pred"] = predictions

    for category in CATEGORY_ORDER:  # Use CATEGORY_ORDER to maintain consistent order
        actual_positive = df["actual_category"] == category
        predicted_positive = df["combined_pred"] == category

        true_positives = sum(actual_positive & predicted_positive)
        false_positives = sum(~actual_positive & predicted_positive)
        false_negatives = sum(actual_positive & ~predicted_positive)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_data.append(
            {
                "category": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(actual_positive),
            }
        )

    return pd.DataFrame(metrics_data).set_index("category")


def calculate_old_model_metrics(
    df: pd.DataFrame, valid_categories: Set[str]
) -> pd.DataFrame:
    """Calculate metrics for old model predictions"""
    metrics_data = []

    for category in CATEGORY_ORDER:  # Use CATEGORY_ORDER to maintain consistent order
        actual_positive = df["actual_category"] == category
        predicted_positive = df["old_prediction"] == category

        true_positives = sum(actual_positive & predicted_positive)
        false_positives = sum(~actual_positive & predicted_positive)
        false_negatives = sum(actual_positive & ~predicted_positive)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_data.append(
            {
                "category": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(actual_positive),
            }
        )

    return pd.DataFrame(metrics_data).set_index("category")


def create_precision_recall_matrix(
    df: pd.DataFrame, valid_categories: List[str], use_dual_prediction: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create precision and recall matrices, handling dual predictions if specified"""
    if use_dual_prediction:
        predictions = []
        for _, row in df.iterrows():
            if row["actual_category"] == row["new_primary"]:
                predictions.append(row["new_primary"])
            elif row["actual_category"] == row["new_secondary"]:
                predictions.append(row["new_secondary"])
            else:
                predictions.append(row["new_primary"])
        df["combined_pred"] = predictions
        pred_col = "combined_pred"
    else:
        pred_col = "old_prediction"

    conf_matrix = pd.crosstab(df["actual_category"], df[pred_col], normalize=False)

    # Ensure all categories are present and in correct order
    for category in valid_categories:
        if category not in conf_matrix.index:
            conf_matrix.loc[category] = 0
        if category not in conf_matrix.columns:
            conf_matrix[category] = 0

    # Reorder according to CATEGORY_ORDER
    conf_matrix = conf_matrix.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER)

    # Calculate precision and recall matrices
    precision_matrix = conf_matrix / conf_matrix.sum(axis=0)
    recall_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)

    return precision_matrix.fillna(0), recall_matrix.fillna(0)


# %% Visualization Functions
def plot_precision_recall_matrix(
    old_matrix: pd.DataFrame, new_matrix: pd.DataFrame, metric_name: str
):
    """Plot old vs new matrices side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot old model matrix
    sns.heatmap(old_matrix, annot=True, fmt=".2f", ax=ax1, cmap="Blues", vmin=0, vmax=1)
    ax1.set_title(f"Old Model {metric_name} Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Plot new model matrix
    sns.heatmap(new_matrix, annot=True, fmt=".2f", ax=ax2, cmap="Blues", vmin=0, vmax=1)
    ax2.set_title(f"New Model {metric_name} Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


# %% Main Analysis Function
def generate_misclassification_report(df: pd.DataFrame) -> str:
    """Generate a markdown report of misclassification examples"""
    markdown_report = """
# Misclassification Analysis Report

This report shows examples where the new model (primary + secondary predictions) misclassified content.
Each category shows up to 5 examples of misclassifications.

"""
    for category in CATEGORY_ORDER:
        # Get misclassifications for this category
        misclassified = df[
            (df["actual_category"] == category)
            & (df["new_primary"] != category)
            & (
                (df["new_secondary"] != category)
                | (df["new_secondary"].isin(["none", "None"]))
            )
        ]

        markdown_report += f"\n## Misclassifications for {category}\n\n"

        if len(misclassified) == 0:
            markdown_report += "No misclassifications found for this category.\n"
            continue

        for _, row in misclassified.sample(frac=1, random_state=42).head(5).iterrows():
            markdown_report += f"""
### Example (ID: {row['text_id']})
- **Text**: {row['text'][:2000]}{'...' if len(row['text']) > 2000 else ''}
- **Actual Category**: {row['actual_category']}
- **New Model Primary Prediction**: {row['new_primary']}
- **New Model Secondary Prediction**: {row['new_secondary']}
- **Old Model Prediction**: {row['old_prediction']}
"""

    return markdown_report


def save_markdown_report(report: str, filename: str = "misclassification_report.md"):
    """Save the markdown report to a file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)


def calculate_accuracy_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall accuracy for both old and new models"""

    # Old model accuracy
    old_correct = sum(df["actual_category"] == df["old_prediction"])
    old_accuracy = old_correct / len(df)

    # New model accuracy (considering both primary and secondary predictions)
    new_correct = sum(
        (df["actual_category"] == df["new_primary"])
        | (df["actual_category"] == df["new_secondary"])
    )
    new_accuracy = new_correct / len(df)

    # Primary-only accuracy
    primary_correct = sum(df["actual_category"] == df["new_primary"])
    primary_accuracy = primary_correct / len(df)

    return {
        "old_model_accuracy": old_accuracy,
        "new_model_accuracy": new_accuracy,
        "primary_only_accuracy": primary_accuracy,
        "total_samples": len(df),
        "old_correct": old_correct,
        "new_correct": new_correct,
        "primary_correct": primary_correct,
    }


def main(file_path: str):
    # Load and prepare data
    df = load_data(file_path)

    # Count None occurrences
    none_counts = count_none_predictions(df)
    print("\nNone value occurrences:")
    for k, v in none_counts.items():
        print(f"{k}: {v}")

    # Calculate metrics for new dual prediction system
    print("\nNew Model Metrics (Using Primary + Secondary):")
    new_metrics = calculate_dual_prediction_metrics(df, VALID_CATEGORIES)
    print(new_metrics)

    # Calculate metrics for old model
    print("\nOld Model Metrics:")
    old_metrics = calculate_old_model_metrics(df, VALID_CATEGORIES)
    print(old_metrics)

    # Create matrices for both models
    new_precision_matrix, new_recall_matrix = create_precision_recall_matrix(
        df, list(VALID_CATEGORIES), use_dual_prediction=True
    )
    old_precision_matrix, old_recall_matrix = create_precision_recall_matrix(
        df, list(VALID_CATEGORIES), use_dual_prediction=False
    )

    # Plot precision matrices comparison
    print("\nPrecision Matrices Comparison (Old vs New):")
    plot_precision_recall_matrix(
        old_precision_matrix, new_precision_matrix, "Precision"
    )

    # Plot recall matrices comparison
    print("\nRecall Matrices Comparison (Old vs New):")
    plot_precision_recall_matrix(old_recall_matrix, new_recall_matrix, "Recall")

    # Calculate and display accuracy metrics
    print("\nOverall Accuracy Metrics:")
    accuracy_metrics = calculate_accuracy_metrics(df)
    print(
        f"Old Model Accuracy: {accuracy_metrics['old_model_accuracy']:.3f} ({accuracy_metrics['old_correct']}/{accuracy_metrics['total_samples']} samples)"
    )
    print(
        f"New Model Accuracy (Primary + Secondary): {accuracy_metrics['new_model_accuracy']:.3f} ({accuracy_metrics['new_correct']}/{accuracy_metrics['total_samples']} samples)"
    )
    print(
        f"New Model Accuracy (Primary Only): {accuracy_metrics['primary_only_accuracy']:.3f} ({accuracy_metrics['primary_correct']}/{accuracy_metrics['total_samples']} samples)"
    )

    improvement = (
        accuracy_metrics["new_model_accuracy"] - accuracy_metrics["old_model_accuracy"]
    )
    print(
        f"\nAccuracy Improvement: {improvement:.3f} ({'+' if improvement > 0 else ''}{improvement*100:.1f}%)"
    )

    # Generate and save misclassification report
    print("\nGenerating misclassification report...")
    report = generate_misclassification_report(df)
    save_markdown_report(report)
    print(f"Report saved as 'misclassification_report.md'")


# %% Run the analysis
if __name__ == "__main__":
    main(
        file_path="/Users/sagar/work/yral/content-moderation/data/exp-prompt-few-shot/prompt-v10/phi_before_after_20250218_133819.jsonl"
    )
