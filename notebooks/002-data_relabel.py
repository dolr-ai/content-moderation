# %% [markdown]
# # Import libraries
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np

# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
DATA_ROOT = Path(os.getenv("DATA_ROOT"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


# %%[markdown]
# # functions
def load_saved_samples(input_dir):
    """Load saved datasets from JSON files"""
    input_path = DATA_ROOT / input_dir
    datasets = {}

    for file in input_path.glob("*.json"):
        name = file.stem.replace("-sample", "")
        # Use read_json with convert_dates=False to avoid timezone warnings
        df = pd.read_json(file, lines=True, convert_dates=False)
        # Convert datetime columns after loading if needed
        if "date" in df.columns:
            # Remove timezone information from the date strings
            df["date"] = df["date"].str.replace(r"\s+[A-Z]{3}\s+\d{4}$", "", regex=True)
            # Specify the date format explicitly
            df["date"] = pd.to_datetime(
                df["date"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce"
            )
        if "created_utc" in df.columns:
            # Check if the values are already in datetime format
            if df["created_utc"].dtype == "object":
                df["created_utc"] = pd.to_datetime(
                    df["created_utc"], unit="s", utc=True, errors="coerce"
                )
            else:
                df["created_utc"] = pd.to_datetime(
                    df["created_utc"], utc=True, errors="coerce"
                )
        datasets[name] = df
        logger.info(f"Loaded {name} sample from {file}")

    return datasets


# %% [markdown]
# # Load samples
datasets = load_saved_samples("raw_sampled")

# %%
print(datasets.keys())
# %%
df_hsol = datasets["hate_speech"]
df_hsol.head()

# %%


def relabel_hsol(x):
    d = x.to_dict()
    l = ["hate_speech", "offensive_language", "neither"]
    ix = np.argmax([d[k] for k in l])
    l_ = [
        "hate_speech",
        "offensive_language",
        "neither_hate_speech_nor_offensive_language",
    ]
    return l_[ix]


df_hsol["target"] = df_hsol.apply(lambda x: relabel_hsol(x), axis=1)
# %%
df_hsol["target"].value_counts()

# %%
df_sarcasm = datasets["sarcasm"]
df_sarcasm.head()

# %%
df_jigsaw = datasets["jigsaw"]
print(df_jigsaw.head())
print(df_jigsaw.shape)
# %%
df_jigsaw["toxicity_type"].value_counts()
# %%

df_jigsaw[df_jigsaw["toxicity_type"] == "toxic_obscene"]["comment_text"].sample(
    10
).tolist()
# %%


def create_moderation_labels(row):
    """
    Creates moderation labels from Jigsaw's numeric columns and returns both detailed and unified labels
    Input expects columns: toxic, severe_toxic, obscene, threat, insult, identity_hate
    All values are 0 or 1
    """
    moderation_labels = []

    # Profanity: toxic OR obscene
    if row["toxic"] == 1 or row["obscene"] == 1:
        moderation_labels.append("profanity")

    # Adult Content: obscene
    if row["obscene"] == 1:
        moderation_labels.append("adult_content")

    # Violence: threat OR severe_toxic
    if row["threat"] == 1 or row["severe_toxic"] == 1:
        moderation_labels.append("violence")

    # Harassment: insult OR threat
    if row["insult"] == 1 or row["threat"] == 1:
        moderation_labels.append("harassment")

    # Hate: identity_hate
    if row["identity_hate"] == 1:
        moderation_labels.append("hate")

    # Create unified target category (ordered by severity)
    unified_target = "clean"
    if "hate" in moderation_labels:
        unified_target = "hate"
    elif "violence" in moderation_labels:
        unified_target = "violence"
    elif "harassment" in moderation_labels:
        unified_target = "harassment"
    elif "adult_content" in moderation_labels:
        unified_target = "adult_content"
    elif "profanity" in moderation_labels:
        unified_target = "profanity"

    return {
        "original_labels": {
            "toxic": row["toxic"],
            "severe_toxic": row["severe_toxic"],
            "obscene": row["obscene"],
            "threat": row["threat"],
            "insult": row["insult"],
            "identity_hate": row["identity_hate"],
        },
        "derived_categories": moderation_labels,
        "moderation_required": len(moderation_labels) > 0,
        "unified_target": unified_target,
    }


# Apply the new labeling function
df_jigsaw["moderation_results"] = df_jigsaw.apply(create_moderation_labels, axis=1)

# Extract unified target for analysis
df_jigsaw["target"] = df_jigsaw["moderation_results"].apply(
    lambda x: x["unified_target"]
)

# Show distribution of unified categories
print("\nUnified Category Distribution:")
print(df_jigsaw["target"].value_counts())

# %%
df_jigsaw
# df_jigsaw[df_jigsaw["target"] == "profanity"]["comment_text"].sample(10).tolist()

# %%
# Example analysis
print("\nSample comments by category:")
for category in df_jigsaw["target"].value_counts().index:
    print(f"\n{category.upper()} examples:")
    # Filter for rows containing this category
    mask = df_jigsaw["target"] == category
    print(df_jigsaw[mask]["comment_text"].sample(min(3, mask.sum())).tolist())
