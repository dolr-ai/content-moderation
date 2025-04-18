#%%[markdown]
# # Detailed Metrics Analysis for OpenAI Moderation API
# This notebook analyzes the detailed metrics from the OpenAI moderation API benchmark results.

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
openai_results_dir = DATA_ROOT / "benchmark_results" / "openai"
result_files = list(openai_results_dir.glob("openai_benchmark_results_*.jsonl"))
latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)

print(f"Loading results from: {latest_result_file}")
results_df = pd.read_json(latest_result_file, lines=True)

# Add confidence score extraction after loading results
for category in PRIMARY_CATEGORY_MAP.keys():
    results_df[f'score_{category}'] = results_df['processed_response'].apply(
        lambda x: x['scores'][category] if isinstance(x, dict) and 'scores' in x else 0
    )

# Add confidence threshold configuration
CONFIDENCE_THRESHOLD = 0.50

# Apply confidence threshold to predictions
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
    y_pred = results_df['processed_response'].apply(
        lambda x: PRIMARY_CATEGORY_MAP[x['predicted_category']] if isinstance(x, dict) and 'error' not in x else None
    )

    # Remove rows with errors or None values
    valid_indices = y_pred.notna()
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    # Calculate metrics for each category
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

    # Calculate latency metrics
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
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Count'}
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title('OpenAI Moderation API - Confusion Matrix', fontsize=20)
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
    plt.title('OpenAI Moderation API - Precision Matrix', fontsize=20)
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
    plt.title('OpenAI Moderation API - Recall Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.tick_params(axis='x', rotation=45, labelsize=12)
    plt.tick_params(axis='y', rotation=45, labelsize=12)
    plt.tight_layout()
    plt.show()

#%%
# Calculate and visualize metrics
metrics = calculate_evaluation_metrics(results_df)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
plot_confusion_matrix(metrics['confusion_matrix'], list(PRIMARY_CATEGORY_MAP.keys()))
plt.show()

# Plot separate precision and recall matrices
plot_precision_recall_matrices(metrics, list(PRIMARY_CATEGORY_MAP.keys()))

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
            lambda x: 'clean' if isinstance(x, dict) and x.get('predicted_score', 0) < threshold
            else x.get('predicted_category', 'clean') if isinstance(x, dict) else 'clean'
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
# Calculate and plot threshold impact
threshold_metrics = analyze_threshold_impact(results_df)

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

    plt.title('OpenAI Moderation API - Impact of Confidence Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

plot_threshold_impact(threshold_metrics)
plt.show()

#%%[markdown]
# ## Print Summary Metrics

#%%
# Print summary metrics
print("\nOverall Metrics:")
for metric, value in metrics['overall'].items():
    print(f"{metric}: {value:.4f}")

print("\nPer-Category Metrics:")
for category, category_metrics in metrics['per_category'].items():
    print(f"\n{category}:")
    for metric, value in category_metrics.items():
        print(f"  {metric}: {value:.4f}")

#%%[markdown]
# ## Analyze Misclassifications

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

        # Add correct predictions
        correct_mask = category_data['predicted_category'] == actual_category
        correct_scores = category_data[correct_mask]['processed_response'].apply(
            lambda x: x['scores'][actual_category] if isinstance(x, dict) and x['scores'][actual_category] >= CONFIDENCE_THRESHOLD else None
        ).dropna().tolist()

        if correct_scores:
            scores_data.append(correct_scores)
            plot_labels.append(f'Correct\n(n={len(correct_scores)})')

        # Add misclassifications
        for pred_category in categories:
            if pred_category != actual_category:
                misclass_mask = category_data['predicted_category'] == pred_category
                misclass_scores = category_data[misclass_mask]['processed_response'].apply(
                    lambda x: x['scores'][pred_category] if isinstance(x, dict) and x['scores'][pred_category] >= CONFIDENCE_THRESHOLD else None
                ).dropna().tolist()

                if misclass_scores:
                    scores_data.append(misclass_scores)
                    plot_labels.append(f'Predicted as {pred_category}\n(n={len(misclass_scores)})')

        if scores_data:
            plt.figure(figsize=(15, 8))
            bp = plt.boxplot(scores_data, patch_artist=True)

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

def print_high_confidence_misclassifications(df, category, is_false_negative=True, n_samples=100):
    """
    Generate markdown content for misclassified examples with high confidence scores.
    """
    output = []

    if is_false_negative:
        mask = (df['actual_category'] == category) & (df['predicted_category'] == 'clean')
        scenario = f"`False Negative` ({category} -> clean)"
    else:
        mask = (df['actual_category'] == 'clean') & (df['predicted_category'] == category)
        scenario = f"`False Positive` (clean -> {category})"

    filtered_df = df[mask].copy()
    score_col = f'score_{category if not is_false_negative else "clean"}'
    filtered_df = filtered_df.sort_values(by=score_col, ascending=False)

    output.append(f"\n## High Confidence Misclassifications: {scenario}")

    samples = filtered_df.sample(min(n_samples, len(filtered_df)), random_state=1343)

    for idx, row in samples.iterrows():
        output.append(f"### Sample {idx}")
        output.append("```")
        output.append(f"Text: {row['text']}")
        output.append(f"Confidence Score: {row[score_col]:.3f}")
        output.append("```")
    output.append("---\n")

    return "\n".join(output)

def analyze_all_categories(results_df, primary_categories, project_root, n_samples=100):
    """
    Analyze misclassifications for all categories and save to markdown files.
    """
    docs_dir = project_root / "docs" / "openai-misclassifications"
    docs_dir.mkdir(exist_ok=True)

    for category in [cat for cat in primary_categories.keys() if cat != 'clean']:
        output = []
        output.append(f"# Analysis for Category: {category}\n")
        output.append(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        output.append(print_high_confidence_misclassifications(
            results_df, category, is_false_negative=True, n_samples=n_samples
        ))

        output.append(print_high_confidence_misclassifications(
            results_df, category, is_false_negative=False, n_samples=n_samples
        ))

        output_file = docs_dir / f"openai-{category}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

        print(f"Saved analysis for {category} to {output_file}")

# Add this at the end of your analysis section
plot_category_score_distributions(results_df, PRIMARY_CATEGORY_MAP.keys())

# Run the misclassification analysis
analyze_all_categories(results_df, PRIMARY_CATEGORY_MAP, PROJECT_ROOT, n_samples=50)
