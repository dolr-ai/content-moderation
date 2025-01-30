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
# ## Configuration Parameters

#%%
# Confidence threshold for classification
# Any prediction with confidence score below this threshold will be classified as 'clean'
# This helps reduce false positives and ensures high-confidence predictions
CONFIDENCE_THRESHOLD = 0.50

#%%
# Load Configuration and Data

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
#%%


# First, extract scores for each category into separate columns
for category in PRIMARY_CATEGORY_MAP.keys():
    results_df[f'score_{category}'] = results_df['processed_response'].apply(
        lambda x: x['scores'][category] if isinstance(x, dict) and 'scores' in x else 0
    )

# Apply confidence threshold to predictions
# If the highest confidence score is below threshold, classify as 'clean'
results_df['predicted_category'] = results_df['processed_response'].apply(
    lambda x: x['predicted_category'] if isinstance(x, dict) and x['scores'][x['predicted_category']] >= CONFIDENCE_THRESHOLD
    else 'clean'  # default to clean if confidence is below threshold
)

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
    y_pred = results_df['predicted_category'].map(PRIMARY_CATEGORY_MAP)

    # Calculate metrics for each category with explicit zero_division handling
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(PRIMARY_CATEGORY_MAP.values()),
        zero_division=0
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
    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Count'}
    )
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12 )
    plt.title('GCP Moderation API - Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.tight_layout()

def plot_precision_recall_matrices(metrics, categories):
    """
    Plot separate precision and recall matrices.

    Args:
        metrics (dict): Dictionary containing metrics data
        categories (list): List of category names
    """
    n_categories = len(categories)

    # Create matrices for precision and recall
    precision_matrix = np.zeros((n_categories, n_categories))
    recall_matrix = np.zeros((n_categories, n_categories))

    # Fill matrices using confusion matrix data
    conf_matrix = metrics['confusion_matrix']
    for i in range(n_categories):
        for j in range(n_categories):
            precision_matrix[i, j] = conf_matrix[i, j] / conf_matrix[:, j].sum() if conf_matrix[:, j].sum() != 0 else 0
            recall_matrix[i, j] = conf_matrix[i, j] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() != 0 else 0

    # Plot precision matrix
    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(
        precision_matrix,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Precision (%)'}
    )
    plt.title('GCP Moderation API - Precision Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.tick_params(axis='x', rotation=45, labelsize=12)
    plt.tick_params(axis='y', rotation=45, labelsize=12)
    plt.tight_layout()
    plt.show()

    # Plot recall matrix
    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Recall (%)'}
    )
    plt.title('GCP Moderation API - Recall Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.tick_params(axis='x', rotation=45, labelsize=12)
    plt.tick_params(axis='y', rotation=45, labelsize=12)
    plt.tight_layout()
    plt.show()

#%%
# First calculate metrics
metrics = calculate_evaluation_metrics(results_df)

#%%[markdown]
# ## Analyze Confidence Threshold Impact

#%%
def analyze_threshold_impact(results_df, thresholds=np.arange(0.1, 1.0, 0.1)):
    """
    Analyze impact of different confidence thresholds on classification metrics.
    """
    threshold_metrics = {}
    score_columns = [f'score_{cat}' for cat in PRIMARY_CATEGORY_MAP.keys()]

    for threshold in thresholds:
        # Apply threshold to get predictions
        predictions = results_df[score_columns].apply(
            lambda row: 'clean' if row.max() < threshold else
                       list(PRIMARY_CATEGORY_MAP.keys())[row.argmax()],
            axis=1
        )

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            results_df['actual_category'],
            predictions,
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

#%%[markdown]
# ## Analyze Score Distributions

#%%
def plot_category_score_distributions(results_df, categories):
    """
    Plot score distributions for each category showing detailed misclassification patterns.
    Only includes predictions where the confidence score is >= CONFIDENCE_THRESHOLD
    """
    for actual_category in categories:
        # Get all predictions for this actual category
        category_mask = results_df['actual_category'] == actual_category
        category_data = results_df[category_mask]

        scores_data = []
        plot_labels = []

        # Add correct predictions - use the score of the correct category
        correct_mask = category_data['predicted_category'] == actual_category

        # NOTE:
        # why are we adding None?:
        # if the confidence score is below the threshold, then we are not able to predict with confidence, ideally we should not label it with that category
        # we can create another category "unresolved" for these cases where we failed to predict a category with confidence
        # for now, we are not adding these cases to the plot
        correct_scores = category_data[correct_mask]['processed_response'].apply(
            lambda x: x['scores'][actual_category] if isinstance(x, dict) and x['scores'][actual_category] >= CONFIDENCE_THRESHOLD else None
        ).dropna().tolist()


        # add correct scores to plot
        if correct_scores:
            scores_data.append(correct_scores)
            plot_labels.append(f'Correct\n(n={len(correct_scores)})')

        # Add misclassifications - use the score of the predicted (wrong) category
        for pred_category in categories:
            if pred_category != actual_category:
                # get misclassification scores
                misclass_mask = category_data['predicted_category'] == pred_category
                misclass_scores = category_data[misclass_mask]['processed_response'].apply(
                    lambda x: x['scores'][pred_category] if isinstance(x, dict) and x['scores'][pred_category] >= CONFIDENCE_THRESHOLD else None
                ).dropna().tolist()

                if misclass_scores:
                    scores_data.append(misclass_scores)
                    plot_labels.append(f'Predicted as {pred_category}\n(n={len(misclass_scores)})')

        # Create visualization only if we have data
        if scores_data:
            plt.figure(figsize=(15, 8))
            bp = plt.boxplot(scores_data, patch_artist=True)

            # Color the boxes
            colors = ['lightgreen'] + ['navajowhite'] * (len(scores_data) - 1)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)


            plt.title(f'Score Distribution for Actual Category: {actual_category}\n(Showing scores of predicted categories)', pad=20)
            plt.ylabel('Confidence Score')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xticks(range(1, len(plot_labels) + 1), plot_labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            # Print statistics
            print(f"\nStatistics for {actual_category}:")
            print(f"Total samples: {len(category_data)}")
            print(f"Correct predictions: {len(correct_scores)}")

            if correct_scores:
                print(f"\nCorrect prediction scores:")
                print(f"Mean: {np.mean(correct_scores):.3f}")
                print(f"Median: {np.median(correct_scores):.3f}")
                print(f"95th percentile: {np.percentile(correct_scores, 95):.3f}")

            for scores, label in zip(scores_data[1:], plot_labels[1:]):
                if scores:
                    print(f"\n{label.split('(')[0].strip()} scores:")
                    print(f"Count: {len(scores)}")
                    print(f"Mean: {np.mean(scores):.3f}")
                    print(f"Median: {np.median(scores):.3f}")
                    print(f"95th percentile: {np.percentile(scores, 95):.3f}")

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

# Plot separate precision and recall matrices
plot_precision_recall_matrices(metrics, list(PRIMARY_CATEGORY_MAP.keys()))

# Plot threshold impact
plt.figure(figsize=(10, 6))
plot_threshold_impact(threshold_metrics)
plt.show()

# %%
# Plot score distributions for all categories
plot_category_score_distributions(results_df, PRIMARY_CATEGORY_MAP.keys())
#%%


#%%


#%%
# check_category = 'nsfw_content'
# df_req = results_df[(results_df['actual_category']==check_category) & (results_df['predicted_category']=='clean')]
# for i in df_req.sample(100)['text'].tolist():
#     print(i)
#     print('\n---\n')
