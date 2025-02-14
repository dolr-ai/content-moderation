#%%
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df = pd.read_json("./benchmark_results/phi35_new_prompt_results_20250214_045248.jsonl", lines=True)

#%%
df.head()

#%%
random_index = 4
print(df['actual_category'].iloc[random_index])
print(df['model_response'].iloc[random_index])

#%%
df_ba = pd.read_json("./phi_before_after.jsonl", lines=True)

#%%
df_ba[(df_ba['actual_category'] != df_ba['old_prediction']) & (df_ba['new_prediction'] == df_ba['actual_category'])]

#%%
df_ba[(df_ba['actual_category'] != df_ba['new_prediction'])]

#%%
# Generate classification reports for both old and new predictions
def generate_classification_report_comparison(df_ba):
    old_report = classification_report(df_ba['actual_category'], df_ba['old_prediction'], output_dict=True)
    new_report = classification_report(df_ba['actual_category'], df_ba['new_prediction'], output_dict=True)

    # Convert to DataFrames for easier comparison
    old_metrics = pd.DataFrame(old_report).transpose()
    new_metrics = pd.DataFrame(new_report).transpose()

    # Add a column to identify the model
    old_metrics['model'] = 'old'
    new_metrics['model'] = 'new'

    # Combine the metrics
    comparison_df = pd.concat([old_metrics, new_metrics])

    # Calculate improvements
    improvements = new_metrics.loc[:, ['precision', 'recall', 'f1-score']] - old_metrics.loc[:, ['precision', 'recall', 'f1-score']]
    improvements['model'] = 'improvement'

    # Add improvements to the comparison
    final_comparison = pd.concat([comparison_df, improvements])

    # Display the results
    print("Metrics Comparison (including improvements):")
    print(final_comparison[['precision', 'recall', 'f1-score', 'model']])

# Call the function with the DataFrame
generate_classification_report_comparison(df_ba)

#%%
def plot_metrics_heatmap(df_ba):
    # Generate classification reports
    old_report = classification_report(df_ba['actual_category'], df_ba['old_prediction'], output_dict=True)
    new_report = classification_report(df_ba['actual_category'], df_ba['new_prediction'], output_dict=True)

    # Create DataFrames
    old_df = pd.DataFrame(old_report).transpose()
    new_df = pd.DataFrame(new_report).transpose()

    # Calculate improvements
    improvements = new_df.loc[:, ['precision', 'recall', 'f1-score']] - old_df.loc[:, ['precision', 'recall', 'f1-score']]

    # Remove 'accuracy', 'macro avg', and 'weighted avg' rows for cleaner visualization
    improvements = improvements.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(improvements,
                annot=True,
                cmap='RdYlGn',  # Red for negative, Green for positive
                center=0,
                fmt='.2f',
                cbar_kws={'label': 'Improvement (New - Old)'})

    plt.title('Improvements in Model Metrics\n(Green = Better, Red = Worse)')
    plt.ylabel('Categories')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.show()

# Call the function
plot_metrics_heatmap(df_ba)

#%%
