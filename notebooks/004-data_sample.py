import os
import pandas as pd
from pathlib import Path
import logging

# Import configs
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
DATA_ROOT = Path(os.getenv("DATA_ROOT"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class DatasetSampler:
    """Class for sampling datasets with different strategies"""

    def __init__(self, sample_size=1000):
        self.sample_size = sample_size

    def sample_hate_speech(self, df):
        """Stratified sampling based on hate speech, offensive language, and neither vote ratios"""
        max_size = len(df)
        logger.info(f"Max size of Hate Speech dataset: {max_size}")
        logger.info(f"Sampled size: {self.sample_size}")

        df["total_votes"] = df["count"]
        df["hate_ratio"] = df["hate_speech"] / df["total_votes"]
        df["offensive_ratio"] = df["offensive_language"] / df["total_votes"]
        df["neither_ratio"] = df["neither"] / df["total_votes"]
        df["dominant_category"] = df[
            ["hate_ratio", "offensive_ratio", "neither_ratio"]
        ].idxmax(axis=1)

        logger.info(df["dominant_category"].value_counts())

        if self.sample_size == -1:
            return df

        sample_per_category = self.sample_size // df["dominant_category"].nunique()
        sampled = []
        for category in df["dominant_category"].unique():
            category_df = df[df["dominant_category"] == category]
            sample = category_df.sample(
                min(len(category_df), sample_per_category), random_state=42
            )
            sampled.append(sample)
        return pd.concat(sampled, ignore_index=True)

    def sample_twitter_comments(self, df):
        """Stratified sampling based on sentiment labels"""
        max_size = len(df)
        logger.info(f"Max size of Twitter Comments dataset: {max_size}")
        logger.info(f"Sampled size: {self.sample_size}")

        if self.sample_size == -1:
            return df

        sample_per_class = self.sample_size // df["target"].nunique()
        sampled = []
        for target in df["target"].unique():
            target_df = df[df["target"] == target]
            sample = target_df.sample(
                min(len(target_df), sample_per_class), random_state=42
            )
            sampled.append(sample)
        return pd.concat(sampled, ignore_index=True)

    def sample_jigsaw_toxic(self, df):
        """Stratified sampling based on toxicity labels"""
        max_size = len(df)
        logger.info(f"Max size of Jigsaw Toxic Comment dataset: {max_size}")
        logger.info(f"Sampled size: {self.sample_size}")

        if self.sample_size == -1:
            return df

        toxic_columns = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        df["toxicity_type"] = df[toxic_columns].apply(
            lambda x: "_".join(x.index[x.astype(bool)]) or "clean", axis=1
        )

        unique_types = df["toxicity_type"].value_counts()
        total_toxic_types = len(unique_types)
        sample_per_type = max(self.sample_size // total_toxic_types, 100)

        sampled = []
        for toxicity_type in df["toxicity_type"].unique():
            type_df = df[df["toxicity_type"] == toxicity_type]
            sample = type_df.sample(min(len(type_df), sample_per_type), random_state=42)
            sampled.append(sample)
        return pd.concat(sampled, ignore_index=True)


def sample_datasets(datasets, sample_size=1_000_000):
    """Sample data from all datasets using the DatasetSampler"""
    sampler = DatasetSampler(sample_size=sample_size)

    sampled_datasets = {
        "hate_speech": sampler.sample_hate_speech(datasets["hate_speech"]),
        "sentiment": sampler.sample_twitter_comments(datasets["sentiment"]),
        "jigsaw": sampler.sample_jigsaw_toxic(datasets["jigsaw"]),
    }

    return sampled_datasets


def save_samples(datasets, output_dir):
    """Save sampled datasets to JSON files"""
    output_path = DATA_ROOT / output_dir
    output_path.mkdir(exist_ok=True)

    for name, df in datasets.items():
        if df is not None:
            output_file = output_path / f"{name}-sample.json"
            df.to_json(output_file, orient="records", lines=True)
            logger.info(f"Saved {name} sample to {output_file}")


def load_saved_samples(input_dir):
    """Load saved datasets from JSON files"""
    input_path = DATA_ROOT / input_dir
    datasets = {}

    for file in input_path.glob("*.json"):
        name = file.stem.replace("-sample", "")
        df = pd.read_json(file, lines=True)
        datasets[name] = df
        logger.info(f"Loaded {name} sample from {file}")

    return datasets


# %% [markdown]
# # Save samples
if False:
    # sample_size=-1 for full dataset
    datasets = sample_datasets(sample_size=10_000)
    save_samples(datasets, "raw_sampled")

# %% [markdown]
# # Load saved samples
if False:
    datasets = load_saved_samples("raw_sampled")
    df_hate_speech = datasets["hate_speech"]
    df_hate_speech.head()
    df_sarcasm = datasets["sarcasm"]
    df_sarcasm.head()
