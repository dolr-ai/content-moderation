"""
Goal: Relabel existing sentiment analysis datasets to final moderation labels
0. clean
1. Hate & Discrimination
2. Violence & Threats
3. Offensive Language
4. Sexual Content
5. Spam & Scams
"""

import pandas as pd
import yaml
from pathlib import Path

# Load config
DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])

PRIMARY_CATEGORY_MAP = {
    "clean": 0,
    "hate_or_discrimination": 1,
    "violence_or_threats": 2,
    "offensive_language": 3,
    "sexual_content": 4,
    "spam_or_scams": 5,
}


class DatasetLoader:
    """Base class for loading datasets"""

    def __init__(self):
        pass

    def load_hate_speech(self):
        """Load Hate Speech and Offensive Language dataset"""
        return pd.read_csv(DATA_ROOT / "raw" / "hsol-2017" / "labeled_data.csv")

    def load_twitter_comments(self):
        """Load Twitter Comments dataset"""
        return pd.read_csv(
            DATA_ROOT
            / "raw"
            / "twitter-sentiment140"
            / "training.1600000.processed.noemoticon.csv",
            encoding="latin1",
            names=["target", "ids", "date", "flag", "user", "text"],
        )

    def load_financial_news_sentiment(self, split="train"):
        """Load Financial News Sentiment dataset"""
        splits = {"train": "sent_train.csv", "validation": "sent_valid.csv"}
        return pd.read_csv(
            "hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits[split]
        )

    def load_fingpt_sentiment(self, split="train"):
        """Load FinGPT Sentiment dataset"""
        df = pd.read_parquet(
            "hf://datasets/FinGPT/fingpt-sentiment-train/data/train-00000-of-00001-dabab110260ac909.parquet"
        )
        return df

    def load_scam_data(self):
        """Load Scam Data dataset"""
        df = pd.read_csv("hf://datasets/OtabekRizayev/scam-data/Scam-Data.csv")
        return df

    def load_all_scam_spam(self):
        """Load All Scam Spam dataset"""
        df = pd.read_csv("hf://datasets/FredZhang7/all-scam-spam/junkmail_dataset.csv")
        return df


# Hate Speech dataset relabeling
def relabel_hate_speech(df):
    """Relabel Hate Speech and Offensive Language dataset"""
    hate_speech_map = {
        0: "hate_speech",
        1: "offensive_language",
        2: "neither_hsol",  # neither hate speech nor offensive language
    }

    df["target"] = df["class"].map(hate_speech_map)
    return df


def map_to_primary_category(label: str) -> str:
    """
    Maps HSOL labels to primary moderation categories.

    Args:
        label: String containing the original label (hate_speech, offensive_language, or neither_hsol)

    Returns:
        str: Primary category name
    """
    if label == "hate_speech":
        return "hate_or_discrimination"
    elif label == "offensive_language":
        return "offensive_language"
    else:
        return "neither_hsol"


# %%
dl = DatasetLoader()

# Load and show original distribution
df_hsol = dl.load_hate_speech()
print("Original Label Distribution:")
print("-" * 50)
print(df_hsol["class"].value_counts())
print("\nOriginal Label Mapping:")
print("-" * 50)
print("0: Hate speech")
print("1: Offensive language")
print("2: Neither")
print("\n")

# %%

# Relabel and continue with the process
df_hsol = relabel_hate_speech(df_hsol)
print("Distribution after initial relabeling:")
print("-" * 50)
print(df_hsol["target"].value_counts())
print("\n")

# Apply mapping to create moderation categories
df_hsol["moderation_category"] = df_hsol["target"].apply(map_to_primary_category)

# Convert string labels to numeric using PRIMARY_CATEGORY_MAP
df_hsol["moderation_label"] = df_hsol["moderation_category"].map(
    lambda x: PRIMARY_CATEGORY_MAP.get(x, "unknown")
)
df_hsol = df_hsol[df_hsol["moderation_category"] != "unknown"].reset_index(drop=True)

# Create final dataset with essential columns
df_final = (
    df_hsol[["tweet", "moderation_category", "moderation_label"]]
    .rename(columns={"tweet": "text"})
    .copy()
)

# Display distribution of categories
print("Distribution of Moderation Categories:")
print("-" * 50)
print(df_final["moderation_category"].value_counts())
print("\nDistribution of Numeric Labels:")
print("-" * 50)
print(df_final["moderation_label"].value_counts())

# Save the relabeled dataset
output_path = DATA_ROOT / "processed" / "hsol-relabelled.jsonl"
df_final.to_json(output_path, orient="records", lines=True)
print(f"\nSaved relabeled dataset to: {output_path}")

# Display sample entries
print("\nSample entries from each category:")
print("=" * 80)
for category in df_final["moderation_category"].unique():
    print(f"\n{category.upper()}")
    print("-" * 80)
    samples = df_final[df_final["moderation_category"] == category]["text"].sample(
        n=3, random_state=42
    )
    for idx, text in enumerate(samples, 1):
        print(f"{idx}. {text[:200]}...")
