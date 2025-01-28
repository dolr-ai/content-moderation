#%%[markdown]
# # Detailed Metrics Analysis for GCP Moderation API
# This notebook analyzes the detailed metrics from the GCP moderation API benchmark results.

#%%
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import yaml
import glob

#%%[markdown]
# ## Load Configuration and Data

#%%
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

# Get the latest benchmark results
gcp_results_dir = DATA_ROOT / "benchmark_results" / "gcp"
result_files = list(gcp_results_dir.glob("gcp_benchmark_results_*.jsonl"))
latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)

print(f"Loading results from: {latest_result_file}")
results_df = pd.read_json(latest_result_file, lines=True)

#%%[markdown]
# ## Calculate Evaluation Metrics

#%%
def calculate_evaluation_metrics(results_df):
    """
    Calculate comprehensive evaluation metrics for moderation results.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Extract actual and predicted categories
    y_true = results_df['actual_category'].map(PRIMARY_CATEGORY_MAP)
    y_pred = results_df['processed_response'].apply(
        lambda x: PRIMARY_CATEGORY_MAP[x['predicted_category']] if isinstance(x, dict) else None
    )

    # Calculate metrics for each category with explicit zero_division handling
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(PRIMARY_CATEGORY_MAP.values()),
        zero_division=0  # Set to 0 to handle undefined cases
    )

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Calculate average latency
    avg_latency = results_df['processing_time'].mean()
    p95_latency = results_df['processing_time'].quantile(0.95)

    # Calculate error rate
    error_rate = len(results_df[results_df['raw_api_response'].apply(lambda x: 'error' in x)]) / len(results_df)

    # Prepare metrics dictionary
    metrics = {
        'overall': {
            'accuracy': overall_accuracy,
            'macro_precision': precision.mean(),
            'macro_recall': recall.mean(),
            'macro_f1': f1.mean(),
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'error_rate': error_rate
        },
        'per_category': {
            category: {
                'precision': p,
                'recall': r,
                'f1': f,
                'support': s,
                'accuracy': acc
            }
            for category, p, r, f, s, acc in zip(
                PRIMARY_CATEGORY_MAP.keys(),
                precision,
                recall,
                f1,
                support,
                per_class_accuracy
            )
        },
        'confusion_matrix': conf_matrix
    }

    return metrics

#%%
# Calculate metrics
metrics = calculate_evaluation_metrics(results_df)

#%%[markdown]
# ## Visualize Results

#%%
def plot_confusion_matrix(conf_matrix, categories):
    """
    Plot confusion matrix heatmap.

    Args:
        conf_matrix (np.array): Confusion matrix
        categories (list): List of category names
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title('GCP Moderation API - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

def plot_precision_recall_heatmaps(metrics, categories):
    """
    Plot separate NxN heatmaps for precision and recall across categories.

    Args:
        metrics (dict): Dictionary containing metrics data
        categories (list): List of category names
    """
    n_categories = len(categories)

    # Create NxN matrices for precision and recall
    precision_matrix = np.zeros((n_categories, n_categories))
    recall_matrix = np.zeros((n_categories, n_categories))

    # Fill matrices using confusion matrix data
    conf_matrix = metrics['confusion_matrix']
    for i in range(n_categories):
        for j in range(n_categories):
            # Precision: True Positives / (True Positives + False Positives)
            precision_matrix[i, j] = conf_matrix[i, j] / conf_matrix[:, j].sum() if conf_matrix[:, j].sum() != 0 else 0
            # Recall: True Positives / (True Positives + False Negatives)
            recall_matrix[i, j] = conf_matrix[i, j] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() != 0 else 0

    # Create subplots with higher DPI for better quality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=150)

    # Plot precision heatmap
    sns.heatmap(
        precision_matrix,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=categories,
        yticklabels=categories,
        ax=ax1,
        cbar_kws={'label': 'Precision (%)'}
    )
    ax1.set_title('Precision Matrix', fontsize=20)
    ax1.set_xlabel('Predicted', fontsize=16)
    ax1.set_ylabel('Actual', fontsize=16)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', rotation=45, labelsize=12)

    # Plot recall heatmap
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=categories,
        yticklabels=categories,
        ax=ax2,
        cbar_kws={'label': 'Recall (%)'}
    )
    ax2.set_title('Recall Matrix', fontsize=20)
    ax2.set_xlabel('Predicted', fontsize=16)
    ax2.set_ylabel('Actual', fontsize=16)
    ax2.tick_params(axis='x', rotation=45, labelsize=12)
    ax2.tick_params(axis='y', rotation=45, labelsize=12)

    plt.suptitle('GCP Moderation API - Precision & Recall Matrices', fontsize=24, y=1.05)
    plt.tight_layout()

#%%
# First calculate metrics
metrics = calculate_evaluation_metrics(results_df)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
plot_confusion_matrix(metrics['confusion_matrix'], list(PRIMARY_CATEGORY_MAP.keys()))
plt.show()

# Plot precision-recall heatmaps
plt.figure(figsize=(20, 8))
plot_precision_recall_heatmaps(metrics, list(PRIMARY_CATEGORY_MAP.keys()))
plt.show()

#%%[markdown]
# ## Analyze Confidence Threshold Impact

#%%
def analyze_threshold_impact(results_df, thresholds=np.arange(0.1, 1.0, 0.1)):
    """
    Analyze impact of different confidence thresholds on classification metrics.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results
        thresholds (np.array): Array of thresholds to evaluate

    Returns:
        dict: Dictionary containing metrics at different thresholds
    """
    threshold_metrics = {}

    for threshold in thresholds:
        # Apply threshold to predictions
        y_pred_threshold = results_df['processed_response'].apply(
            lambda x: 'clean' if isinstance(x, dict) and x['predicted_score'] < threshold else x['predicted_category']
        )

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            results_df['actual_category'],
            y_pred_threshold,
            average='macro'
        )

        threshold_metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return threshold_metrics

#%%
# Calculate threshold impact
threshold_metrics = analyze_threshold_impact(results_df)

#%%
def plot_threshold_impact(threshold_metrics):
    """
    Plot impact of different confidence thresholds on metrics.

    Args:
        threshold_metrics (dict): Dictionary containing metrics at different thresholds
    """
    thresholds = list(threshold_metrics.keys())
    metrics_data = {
        'precision': [m['precision'] for m in threshold_metrics.values()],
        'recall': [m['recall'] for m in threshold_metrics.values()],
        'f1': [m['f1'] for m in threshold_metrics.values()]
    }

    plt.figure(figsize=(10, 6))
    for metric, values in metrics_data.items():
        plt.plot(thresholds, values, marker='o', label=metric)

    plt.title('GCP Moderation API - Impact of Confidence Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

#%%
# Plot threshold impact
plot_threshold_impact(threshold_metrics)
plt.show()

#%%[markdown]
# ## Print Summary Metrics

#%%
# Calculate all metrics
metrics = calculate_evaluation_metrics(results_df)

# Calculate threshold impact
threshold_metrics = analyze_threshold_impact(results_df)

# Print summary metrics
print("\nOverall Metrics:")
for metric, value in metrics['overall'].items():
    print(f"{metric}: {value:.4f}")

print("\nPer-Category Metrics:")
for category, category_metrics in metrics['per_category'].items():
    print(f"\n{category}:")
    for metric, value in category_metrics.items():
        print(f"  {metric}: {value:.4f}")

# Plot confusion matrix
plt.figure(figsize=(12, 10))
plot_confusion_matrix(metrics['confusion_matrix'], list(PRIMARY_CATEGORY_MAP.keys()))
plt.show()

# Plot precision-recall heatmaps
plt.figure(figsize=(20, 8))
plot_precision_recall_heatmaps(metrics, list(PRIMARY_CATEGORY_MAP.keys()))
plt.show()

# Plot threshold impact
plt.figure(figsize=(10, 6))
plot_threshold_impact(threshold_metrics)
plt.show()