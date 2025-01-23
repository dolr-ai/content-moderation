# %% [markdown]
# # Import libraries
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json

# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
DATA_ROOT = Path(os.getenv("DATA_ROOT"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# %% [markdown]
# # Define data loader


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

    def load_jigsaw_toxic(self):
        """Load Jigsaw Toxic Comment Classification dataset"""
        return pd.read_csv(DATA_ROOT / "raw" / "jigsaw-toxic-comment" / "train.csv")


# %% [markdown]
# # load all datasets
dl = DatasetLoader()
datasets = {
    "hate_speech": dl.load_hate_speech(),
    "twitter_comments": dl.load_twitter_comments(),
    "jigsaw": dl.load_jigsaw_toxic(),
}

# %%
df_hate_speech = datasets["hate_speech"]
display(df_hate_speech.head())
print(df_hate_speech.shape)
print(df_hate_speech["class"].value_counts())

# %%
df_twitter_comments = datasets["twitter_comments"]
display(df_twitter_comments.head())
print(df_twitter_comments.shape)
print(df_twitter_comments["target"].value_counts())

# %%
df_jigsaw = datasets["jigsaw"]
display(df_jigsaw.head())
print(df_jigsaw.shape)

jigsaw_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

df_jigsaw[jigsaw_labels].apply(
    lambda x: (
        "neutral"
        if "-".join([i for i in jigsaw_labels if x[i] == 1]) == ""
        else "-".join([i for i in jigsaw_labels if x[i] == 1])
    ),
    axis=1,
).value_counts().head(20)

# %%
