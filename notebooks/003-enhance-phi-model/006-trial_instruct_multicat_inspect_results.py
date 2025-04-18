# %% [markdown]
# # Model Performance Comparison Analysis with Dual Categories
# This notebook compares the performance of old and new model predictions against actual categories,
# taking into account both primary and secondary predictions from the new model.

# %% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import Markdown, display

# %% Data Loading
def load_data(jsonl_path):
    """Load data from JSONL file and prepare it for analysis"""
    df = pd.read_json(jsonl_path, lines=True)

    # Create a combined prediction column for new model
    df['new_prediction'] = df.apply(
        lambda x: [x['new_primary'], x['new_secondary']]
        if x['new_secondary'] != 'none'
        else [x['new_primary']],
        axis=1
    )

    return df

# %% Enhanced Performance Metrics
def calculate_enhanced_metrics(df):
    """Calculate enhanced performance metrics including dual category matching"""

    # Function to check if actual category matches either prediction
    def is_match(row):
        return row['actual_category'] in row['new_prediction']

    # Calculate various metrics
    metrics = {
        "accuracy": {
            "old": (df["old_prediction"] == df["actual_category"]).mean(),
            "new_strict": (df["new_primary"] == df["actual_category"]).mean(),
            "new_flexible": df.apply(is_match, axis=1).mean()
        }
    }

    # Calculate per-category metrics for old model
    metrics["old_report"] = pd.DataFrame(
        classification_report(
            df["actual_category"],
            df["old_prediction"],
            output_dict=True
        )
    ).transpose()

    # Calculate per-category metrics for new model (strict - primary only)
    metrics["new_strict_report"] = pd.DataFrame(
        classification_report(
            df["actual_category"],
            df["new_primary"],
            output_dict=True
        )
    ).transpose()

    # Calculate flexible accuracy per category
    category_flexible_metrics = {}
    for category in df['actual_category'].unique():
        category_mask = df['actual_category'] == category
        category_flexible_metrics[category] = {
            'precision': (
                df[df['new_primary'] == category].apply(is_match, axis=1).mean()
            ),
            'recall': (
                df[category_mask].apply(is_match, axis=1).mean()
            )
        }
        # Calculate F1 score
        p = category_flexible_metrics[category]['precision']
        r = category_flexible_metrics[category]['recall']
        category_flexible_metrics[category]['f1-score'] = (
            2 * (p * r) / (p + r) if (p + r) > 0 else 0
        )
        category_flexible_metrics[category]['support'] = category_mask.sum()

    metrics["new_flexible_report"] = pd.DataFrame(category_flexible_metrics).transpose()

    # Calculate improvements
    metrics["improvements"] = {
        "strict": (
            metrics["new_strict_report"].loc[:, ["precision", "recall", "f1-score"]]
            - metrics["old_report"].loc[:, ["precision", "recall", "f1-score"]]
        ),
        "flexible": (
            metrics["new_flexible_report"].loc[:, ["precision", "recall", "f1-score"]]
            - metrics["old_report"].loc[:, ["precision", "recall", "f1-score"]]
        )
    }

    return metrics

# %% Secondary Category Analysis
def analyze_secondary_categories(df):
    """Analyze the impact and distribution of secondary categories"""

    # Calculate how often secondary categories are used
    secondary_usage = {
        'total_predictions': len(df),
        'with_secondary': (df['new_secondary'] != 'none').sum(),
        'secondary_distribution': df['new_secondary'].value_counts()
    }

    # Analyze when secondary categories help
    def secondary_helps(row):
        return (row['new_primary'] != row['actual_category'] and
                row['new_secondary'] == row['actual_category'])

    secondary_usage['secondary_helps'] = df.apply(secondary_helps, axis=1).sum()

    # Create secondary category transition matrix
    secondary_transitions = pd.crosstab(
        df['new_primary'],
        df['new_secondary'],
        margins=True
    )

    return {
        'usage_stats': secondary_usage,
        'transitions': secondary_transitions
    }

# %% Confusion Matrix Visualization
def plot_enhanced_confusion_matrices(df):
    """Plot three sets of confusion matrix comparisons:
    1. Old model vs New model primary predictions (side by side)
    2. Old model vs New model secondary predictions (side by side)
    3. Old model vs Unified predictions (side by side)
    """
    categories = sorted(df["actual_category"].unique())

    # 1. Old vs Primary predictions
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    cm_old = confusion_matrix(df["actual_category"], df["old_prediction"])
    cm_new_primary = confusion_matrix(df["actual_category"], df["new_primary"])

    sns.heatmap(cm_old, annot=True, fmt="d", ax=ax1, cmap="Blues", xticklabels=categories, yticklabels=categories)
    ax1.set_title("Old Model Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    sns.heatmap(cm_new_primary, annot=True, fmt="d", ax=ax2, cmap="Blues", xticklabels=categories, yticklabels=categories)
    ax2.set_title("New Model Primary Predictions")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 2. Old vs Secondary predictions (for cases with secondary predictions)
    secondary_data = df[df['new_secondary'] != 'none'].copy()
    if len(secondary_data) > 0:
        # Replace 'none' with np.nan and dropna
        secondary_data['new_secondary'] = secondary_data['new_secondary'].replace('none', np.nan)
        secondary_data = secondary_data.dropna(subset=['new_secondary'])

        if len(secondary_data) > 0:  # Check again after dropping NaN values
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            cm_old_secondary = confusion_matrix(
                secondary_data["actual_category"],
                secondary_data["old_prediction"]
            )
            cm_new_secondary = confusion_matrix(
                secondary_data["actual_category"],
                secondary_data["new_secondary"]
            )

            sns.heatmap(cm_old_secondary, annot=True, fmt="d", ax=ax1, cmap="Blues", xticklabels=categories, yticklabels=categories)
            ax1.set_title("Old Model (Secondary Cases Only)")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")

            sns.heatmap(cm_new_secondary, annot=True, fmt="d", ax=ax2, cmap="Blues", xticklabels=categories, yticklabels=categories)
            ax2.set_title("New Model Secondary Predictions")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            plt.tight_layout()
            plt.show()

    # 3. Old vs Unified predictions
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create unified predictions by combining primary and secondary
    # Strategy for unified confusion matrix:
    # - When actual category matches either primary or secondary, it's counted as correct
    # - This means reducing the count in the misclassification cells and
    #   increasing the count in the diagonal (correct classification) cell

    # First create basic confusion matrix with primary predictions
    cm_unified = confusion_matrix(df["actual_category"], df["new_primary"])
    total_before = cm_unified.sum()

    # Adjust for secondary predictions
    for idx, row in df.iterrows():
        actual = row['actual_category']
        primary = row['new_primary']
        secondary = row['new_secondary']

        # If primary was wrong but secondary is correct
        if actual != primary and secondary == actual:
            # Find indices in confusion matrix
            actual_idx = categories.index(actual)
            primary_idx = categories.index(primary)

            # Decrease count from primary misclassification
            cm_unified[actual_idx][primary_idx] -= 1
            # Increase count in correct classification
            cm_unified[actual_idx][actual_idx] += 1

    # Validation: total predictions should remain the same
    total_after = cm_unified.sum()
    assert total_before == total_after, f"Total predictions changed from {total_before} to {total_after}"

    cm_old = confusion_matrix(df["actual_category"], df["old_prediction"])

    sns.heatmap(cm_old, annot=True, fmt="d", ax=ax1, cmap="Blues", xticklabels=categories, yticklabels=categories)
    ax1.set_title("Old Model Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    sns.heatmap(cm_unified, annot=True, fmt="d", ax=ax2, cmap="Blues", xticklabels=categories, yticklabels=categories)
    ax2.set_title("New Model Unified Predictions\n(Primary + Secondary)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

# %% Improvements Visualization
def plot_dual_improvements(improvements):
    """Plot heatmap of metric improvements for both strict and flexible matching side by side.
    - Strict: Using only primary predictions
    - Flexible: Using both primary and secondary predictions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for improvements_data, ax, title in [
        (improvements['strict'], ax1, 'Primary Only'),
        (improvements['flexible'], ax2, 'Flexible (Primary + Secondary)')
    ]:
        # Clean data by removing aggregate rows
        improvements_clean = improvements_data.drop(
            ["accuracy", "macro avg", "weighted avg"],
            errors="ignore"
        )

        sns.heatmap(
            improvements_clean,
            annot=True,
            cmap="RdYlGn",
            center=0,
            fmt=".3f",
            ax=ax,
            cbar_kws={"label": "Improvement"}
        )
        ax.set_title(f"Metric Improvements - {title}")
        ax.set_ylabel("Categories")
        ax.set_xlabel("Metrics")

    plt.tight_layout()
    plt.show()

# %% Secondary Category Impact Visualization
def plot_secondary_category_impact(secondary_analysis):
    """Visualize the impact and distribution of secondary categories"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot secondary category usage
    usage_data = pd.Series({
        'With Secondary': secondary_analysis['usage_stats']['with_secondary'],
        'Without Secondary': (
            secondary_analysis['usage_stats']['total_predictions'] -
            secondary_analysis['usage_stats']['with_secondary']
        )
    })

    usage_data.plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax1,
        colors=['lightblue', 'lightgray']
    )
    ax1.set_title('Secondary Category Usage')

    # Plot secondary category distribution (excluding 'none')
    secondary_dist = secondary_analysis['usage_stats']['secondary_distribution']
    secondary_dist = secondary_dist[secondary_dist.index != 'none']

    if len(secondary_dist) > 0:
        secondary_dist.plot(
            kind='bar',
            ax=ax2,
            color='lightblue'
        )
        ax2.set_title('Secondary Category Distribution')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Count')
        plt.xticks(rotation=45)
    else:
        ax2.text(0.5, 0.5, "No secondary predictions", ha='center')
        ax2.set_title('Secondary Category Distribution')

    plt.tight_layout()
    plt.show()

# %% Error Analysis
def analyze_prediction_changes(df):
    """Analyze improvements and deteriorations in predictions"""

    def get_prediction_status(row):
        old_correct = row['old_prediction'] == row['actual_category']
        new_primary_correct = row['new_primary'] == row['actual_category']
        new_secondary_correct = (
            row['new_secondary'] != 'none' and
            row['new_secondary'] == row['actual_category']
        )
        new_any_correct = new_primary_correct or new_secondary_correct

        if old_correct and not new_any_correct:
            return 'deterioration'
        elif not old_correct and new_any_correct:
            return 'improvement'
        elif old_correct and new_any_correct:
            return 'maintained_correct'
        else:
            return 'maintained_incorrect'

    df['change_status'] = df.apply(get_prediction_status, axis=1)

    return {
        'improvements': df[df['change_status'] == 'improvement'],
        'deteriorations': df[df['change_status'] == 'deterioration'],
        'maintained_correct': df[df['change_status'] == 'maintained_correct'],
        'maintained_incorrect': df[df['change_status'] == 'maintained_incorrect']
    }

# %% Precision-Recall Comparison
def plot_precision_recall_comparison(df):
    """Plot precision and recall confusion matrices comparing old model vs new flexible model"""
    categories = sorted(df["actual_category"].unique())
    n_categories = len(categories)

    # Create matrices for precision and recall for both models
    old_precision = np.zeros((n_categories, n_categories))
    old_recall = np.zeros((n_categories, n_categories))
    new_precision = np.zeros((n_categories, n_categories))
    new_recall = np.zeros((n_categories, n_categories))

    # Calculate metrics for each category pair
    for i, actual_cat in enumerate(categories):
        actual_mask = df['actual_category'] == actual_cat
        actual_count = actual_mask.sum()

        for j, pred_cat in enumerate(categories):
            # Old model calculations
            old_pred_mask = df['old_prediction'] == pred_cat
            old_pred_count = old_pred_mask.sum()
            old_correct = (actual_mask & old_pred_mask).sum()

            # Calculate old model metrics
            if old_pred_count > 0:
                old_precision[i, j] = old_correct / old_pred_count
            if actual_count > 0:
                old_recall[i, j] = old_correct / actual_count

            # New model calculations (flexible)
            # Consider a prediction correct if either primary or secondary matches
            # but count each prediction only once
            primary_pred_mask = df['new_primary'] == pred_cat
            secondary_pred_mask = df['new_secondary'] == pred_cat
            new_pred_mask = primary_pred_mask | secondary_pred_mask
            new_pred_count = new_pred_mask.sum()

            # For each actual category, count how many were correctly predicted
            # as this prediction category (either primary or secondary, not both)
            new_correct = (actual_mask & new_pred_mask).sum()

            # Calculate new model metrics
            if new_pred_count > 0:
                new_precision[i, j] = new_correct / new_pred_count
            if actual_count > 0:
                new_recall[i, j] = new_correct / actual_count

    # Plot the matrices
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Precision plots
    sns.heatmap(old_precision, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=categories, yticklabels=categories,
                ax=ax1)
    ax1.set_title("Old Model - Precision")
    ax1.set_xlabel("Predicted Category")
    ax1.set_ylabel("Actual Category")

    sns.heatmap(new_precision, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=categories, yticklabels=categories,
                ax=ax2)
    ax2.set_title("New Model (Flexible) - Precision")
    ax2.set_xlabel("Predicted Category")
    ax2.set_ylabel("Actual Category")

    # Recall plots
    sns.heatmap(old_recall, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=categories, yticklabels=categories,
                ax=ax3)
    ax3.set_title("Old Model - Recall")
    ax3.set_xlabel("Predicted Category")
    ax3.set_ylabel("Actual Category")

    sns.heatmap(new_recall, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=categories, yticklabels=categories,
                ax=ax4)
    ax4.set_title("New Model (Flexible) - Recall")
    ax4.set_xlabel("Predicted Category")
    ax4.set_ylabel("Actual Category")

    plt.tight_layout()
    plt.show()

# %% Run Analysis
def run_analysis(jsonl_path):
    """Run complete analysis pipeline"""
    # Load data
    df = load_data(jsonl_path)

    # Calculate enhanced metrics
    metrics = calculate_enhanced_metrics(df)

    # Print accuracy improvements
    print("\nAccuracy Metrics:")
    print(f"Old Model: {metrics['accuracy']['old']:.3f}")
    print(f"New Model (Strict): {metrics['accuracy']['new_strict']:.3f}")
    print(f"New Model (Flexible): {metrics['accuracy']['new_flexible']:.3f}")
    print(
        f"Strict Improvement: {metrics['accuracy']['new_strict'] - metrics['accuracy']['old']:.3f}"
    )
    print(
        f"Flexible Improvement: {metrics['accuracy']['new_flexible'] - metrics['accuracy']['old']:.3f}"
    )

    # Analyze secondary categories
    secondary_analysis = analyze_secondary_categories(df)
    print("\nSecondary Category Usage:")
    print(
        f"Predictions with secondary category: "
        f"{secondary_analysis['usage_stats']['with_secondary']} "
        f"({secondary_analysis['usage_stats']['with_secondary'] / len(df) * 100:.1f}%)"
    )
    print(
        f"Cases where secondary category helps: "
        f"{secondary_analysis['usage_stats']['secondary_helps']}"
    )

    # Plot visualizations
    plot_enhanced_confusion_matrices(df)
    plot_dual_improvements(metrics['improvements'])
    plot_secondary_category_impact(secondary_analysis)
    plot_precision_recall_comparison(df)

    # Analyze prediction changes
    changes = analyze_prediction_changes(df)
    print("\nPrediction Changes:")
    for status, cases in changes.items():
        print(f"{status.replace('_', ' ').title()}: {len(cases)} cases")

    return {
        'metrics': metrics,
        'secondary_analysis': secondary_analysis,
        'changes': changes
    }

# %% Execute Analysis
if __name__ == "__main__":
    jsonl_path = "/root/content-moderation/notebooks/003-enhance-phi-model/phi_before_after_20250218_081627.jsonl"
    results = run_analysis(jsonl_path)