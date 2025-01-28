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

    # Find all relabelled jsonl files
    relabelled_files = [
        f for f in processed_dir.glob("*-relabel*.jsonl") if "embeddings" not in f.name
    ]

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
# %%
sample_size = 1000
df_grouped = (
    df_all_data.groupby(["source", "moderation_category"], as_index=False)
    .agg(
        num_total_samples=("text", "size"),
        num_unique_samples=("text", "nunique"),
        all_text=("text", lambda x: list(set(x))),
    )
    .sort_values(
        by=["moderation_category", "source", "num_total_samples"],
        ascending=False,
        ignore_index=True,
    )
)

random.seed(1343)
df_grouped["sampled_text"] = df_grouped.apply(
    lambda x: random.sample(
        x["all_text"],
        min(sample_size, x["num_unique_samples"]),
    ),
    axis=1,
)
# %%
df_grouped["num_samples"] = df_grouped["sampled_text"].apply(len)
# %%
df_grouped["num_samples"].sum()
# %%

df_sampled = df_grouped[
    ["source", "moderation_category", "num_samples", "sampled_text"]
]
#%%
for ix, row in df_sampled.iterrows():
    print("-" * 30)
    print(row["source"], row["moderation_category"])
    print("-" * 30)
    for i, s in enumerate(np.random.choice(row["sampled_text"], 5, replace=False)):
        print(f"{i+1}. {s[:800]}...")
# %%
df_sampled_flat = df_sampled.explode("sampled_text").drop(columns=["num_samples"])
df_sampled_flat = df_sampled_flat.reset_index(drop=True)

#%%
df_sampled_flat = df_sampled_flat.sort_values(
    by=["source", "moderation_category"], ignore_index=True
)
#%%
df_sampled_flat.groupby("moderation_category").agg(
    num_samples=("sampled_text", "count")
)
#%%
os.makedirs(DATA_ROOT / "benchmark", exist_ok=True)
df_sampled_flat.rename(columns={"sampled_text": "text"}).to_json(
    DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", orient="records", lines=True
)
#%%
df_sampled_flat
