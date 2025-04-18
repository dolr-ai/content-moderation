# %% [markdown]
# ## Data sampling
#
# This notebook samples datasets for evaluation:
# - 0: Clean
# - 1: Hate & Discrimination
# - 2: Violence & Threats
# - 3: Offensive Language
# - 4: Sexual Content
# - 5: Spam & Scams

# %%
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json
from huggingface_hub import login as hf_login
import yaml
from IPython.display import display
import matplotlib.pyplot as plt
from pprint import pprint

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

# Load configuration from YAML
DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set up paths and tokens
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]

# Huggingface login
hf_login(HF_TOKEN)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


# Function to load all relabelled datasets
def load_relabelled_datasets():
    """Load and combine all relabelled datasets from the processed directory"""
    processed_dir = DATA_ROOT / "processed"
    datasets = []

    # Sort the relabelled files to ensure consistent order
    relabelled_files = sorted(
        [
            f
            for f in processed_dir.glob("*-relabel*.jsonl")
            if "embeddings" not in f.name
        ]
    )

    for file_path in relabelled_files:
        try:
            df = pd.read_json(file_path, lines=True)
            # Add source column
            df["source"] = file_path.stem.split("-")[0]
            datasets.append(df)
            logger.info(f"Loaded {file_path.name} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    logger.info(f"Total combined dataset size: {len(combined_df)} rows")

    return combined_df


# Load the datasets
df_all_data = load_relabelled_datasets()
df_all_data = df_all_data[df_all_data["moderation_category"] != "neither_hsol"]
# %%
df_all_data.keys()
# %%
df_all_data = df_all_data.dropna(subset=["moderation_category"])
# %%

df_benchmark = pd.read_json(DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", lines=True)
print(df_benchmark.columns)
print(df_benchmark.shape)
# %%

# Perform outer merge to identify non-matching rows
df_comparison = df_all_data[["text", "moderation_category", "source"]].merge(
    df_benchmark[["text", "moderation_category", "source"]],
    on=["text", "moderation_category", "source"],
    how="outer",
    indicator=True,
)

# Filter for rows only in df_all_data (left_only)
df_non_benchmark = df_comparison[df_comparison["_merge"] == "left_only"].drop(
    "_merge", axis=1
)

# Display results
print(f"Total rows in all data: {len(df_all_data)}")
print(f"Total rows in benchmark: {len(df_benchmark)}")
print(f"Rows not in benchmark: {len(df_non_benchmark)}")
display(df_non_benchmark.head())

# %%
df_non_benchmark["moderation_category"].value_counts()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

TARGET_SAMPLES = 5000
df_vec_db = pd.DataFrame()
borrowed_counts = {}

for category in PRIMARY_CATEGORY_MAP.keys():
    # Get all available samples from non-benchmark data
    category_samples = df_non_benchmark[
        df_non_benchmark["moderation_category"] == category
    ].copy()
    available_non_benchmark = len(category_samples)

    # If we have more than target samples in non-benchmark, sample down to target
    if available_non_benchmark >= TARGET_SAMPLES:
        category_samples = category_samples.sample(n=TARGET_SAMPLES, random_state=42)
        borrowed_counts[category] = 0
    else:
        # Need to borrow from benchmark
        samples_needed = TARGET_SAMPLES - available_non_benchmark
        benchmark_samples = df_benchmark[
            df_benchmark["moderation_category"] == category
        ].copy()
        benchmark_available = len(benchmark_samples)
        borrowed_counts[category] = min(samples_needed, benchmark_available)

        if borrowed_counts[category] > 0:
            benchmark_samples = benchmark_samples.sample(
                n=borrowed_counts[category], random_state=42
            )
            category_samples = pd.concat([category_samples, benchmark_samples])

    category_samples["is_benchmark"] = (
        category_samples.index.isin(benchmark_samples.index)
        if "benchmark_samples" in locals()
        else False
    )
    df_vec_db = pd.concat([df_vec_db, category_samples])

# Reset index
df_vec_db = df_vec_db.reset_index(drop=True)

# Create report DataFrame
report_data = []
for category in PRIMARY_CATEGORY_MAP.keys():
    total = len(df_vec_db[df_vec_db["moderation_category"] == category])
    borrowed = borrowed_counts[category]
    original = len(
        df_non_benchmark[df_non_benchmark["moderation_category"] == category]
    )
    report_data.append(
        {
            "Category": category,
            "Total Samples": total,
            "Target Met": total == TARGET_SAMPLES,
            "Available in Non-Benchmark": original,
            "Borrowed from Benchmark": borrowed,
            "Target Samples": TARGET_SAMPLES,
        }
    )

df_report = pd.DataFrame(report_data)
display(df_report)

# Quick verification of no duplicates
duplicates = df_vec_db.duplicated(subset=["text"]).sum()
print(f"\nNumber of duplicate texts: {duplicates}")

# %%
df_vec_db.drop_duplicates(subset=["text"])

# %%
df_vec_db.to_json(
    DATA_ROOT / "rag" / "vector_db_text.jsonl", orient="records", lines=True
)
