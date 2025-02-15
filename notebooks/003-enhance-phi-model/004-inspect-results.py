# %% [markdown]
# # Model Performance Comparison Analysis
# This notebook compares the performance of old and new model predictions against actual categories.

# %% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# %% Data Loading
def load_data(jsonl_path):
    """Load data from JSONL file"""
    return pd.read_json(jsonl_path, lines=True)


# %% Basic Performance Metrics
def calculate_basic_metrics(df):
    """Calculate basic performance metrics including accuracy and classification reports"""
    # Overall accuracy
    metrics = {
        "accuracy": {
            "old": (df["old_prediction"] == df["actual_category"]).mean(),
            "new": (df["new_prediction"] == df["actual_category"]).mean(),
        }
    }

    # Classification reports
    metrics["old_report"] = pd.DataFrame(
        classification_report(
            df["actual_category"], df["old_prediction"], output_dict=True
        )
    ).transpose()

    metrics["new_report"] = pd.DataFrame(
        classification_report(
            df["actual_category"], df["new_prediction"], output_dict=True
        )
    ).transpose()

    # Calculate improvements
    metrics["improvements"] = (
        metrics["new_report"].loc[:, ["precision", "recall", "f1-score"]]
        - metrics["old_report"].loc[:, ["precision", "recall", "f1-score"]]
    )

    return metrics


# %% Confusion Matrix Visualization
def plot_confusion_matrices(df):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    categories = sorted(df["actual_category"].unique())

    # Old model matrix
    cm_old = confusion_matrix(df["actual_category"], df["old_prediction"])
    sns.heatmap(
        cm_old,
        annot=True,
        fmt="d",
        ax=ax1,
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
    )
    ax1.set_title("Old Model Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # New model matrix
    cm_new = confusion_matrix(df["actual_category"], df["new_prediction"])
    sns.heatmap(
        cm_new,
        annot=True,
        fmt="d",
        ax=ax2,
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
    )
    ax2.set_title("New Model Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


# %% Improvements Visualization
def plot_improvements(improvements):
    """Plot heatmap of metric improvements"""
    improvements_clean = improvements.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
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


# %% Error Analysis
def analyze_prediction_changes(df):
    """Analyze improvements and deteriorations in predictions"""
    improvements = df[
        (df["old_prediction"] != df["actual_category"])
        & (df["new_prediction"] == df["actual_category"])
    ]

    deteriorations = df[
        (df["old_prediction"] == df["actual_category"])
        & (df["new_prediction"] != df["actual_category"])
    ]

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

    # Precision and Recall Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    width = 0.35

    categories = category_metrics["old_metrics"].index
    x = np.arange(len(categories))

    # Precision plot
    ax1.bar(
        x - width / 2,
        category_metrics["old_metrics"]["precision"],
        width,
        label="Old Model",
    )
    ax1.bar(
        x + width / 2,
        category_metrics["new_metrics"]["precision"],
        width,
        label="New Model",
    )
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision Comparison by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()

    # Recall plot
    ax2.bar(
        x - width / 2,
        category_metrics["old_metrics"]["recall"],
        width,
        label="Old Model",
    )
    ax2.bar(
        x + width / 2,
        category_metrics["new_metrics"]["recall"],
        width,
        label="New Model",
    )
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall Comparison by Category")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# %% Main Analysis
def run_analysis(jsonl_path):
    """Run complete analysis pipeline"""
    # Load data
    df = load_data(jsonl_path)

    # Calculate basic metrics
    basic_metrics = calculate_basic_metrics(df)

    # Print accuracy improvements
    print("\nOverall Accuracy:")
    print(f"Old Model: {basic_metrics['accuracy']['old']:.3f}")
    print(f"New Model: {basic_metrics['accuracy']['new']:.3f}")
    print(
        f"Improvement: {basic_metrics['accuracy']['new'] - basic_metrics['accuracy']['old']:.3f}"
    )

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

    return {
        "basic_metrics": basic_metrics,
        "improvements": improvements,
        "deteriorations": deteriorations,
        "category_metrics": category_metrics,
    }


# %% Execute Analysis
if __name__ == "__main__":
    jsonl_path = "/root/content-moderation/notebooks/003-enhance-phi-model/phi_before_after.jsonl_20250215_011322.jsonl"
    results = run_analysis(jsonl_path)
