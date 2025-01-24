# %% [markdown]
# # Import libraries
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import yaml
from IPython.display import display
from pprint import pprint

DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]

# huggingface login
hf_login(HF_TOKEN)

# logging
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


# %% [markdown]
# # load all datasets
dl = DatasetLoader()
datasets = {
    "hate_speech": dl.load_hate_speech(),
    "twitter_comments": dl.load_twitter_comments(),
    "jigsaw": dl.load_jigsaw_toxic(),
    "fi_news": dl.load_financial_news_sentiment(),
    "fingpt": dl.load_fingpt_sentiment(),
    "scam": dl.load_scam_data(),
    "all_scam_spam": dl.load_all_scam_spam(),
}
# %%

# %% [markdown]
# Check datasets
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
        "clean"
        if "-".join([i for i in jigsaw_labels if x[i] == 1]) == ""
        else "-".join([i for i in jigsaw_labels if x[i] == 1])
    ),
    axis=1,
).value_counts().head(20)

# %%

# this is more of sentiment of market / stock
# this is to understand if the model is able to identify finance / stock related data (this is not for training)
# to check if model is able to gauge $SYMBOL_NAME in text

df_fi_news = datasets["fi_news"]
display(df_fi_news.head())
print(df_fi_news.shape)

print(df_fi_news["label"].value_counts())

unique_outputs = df_fi_news["label"].unique()
for output in unique_outputs:
    samples = df_fi_news[df_fi_news["label"] == output]["text"].sample(n=10)
    print(f"Samples for output '{output}':")
    pprint(samples.tolist())
    print("---")


# %%

# this is more of sentiment of market / stock
# this is to understand if the model is able to identify finance / stock related data (this is not for training)
# to check if model is able to gauge $SYMBOL_NAME in text

df_fingpt = datasets["fingpt"]
display(df_fingpt.head())
print(df_fingpt.shape)
unique_outputs = df_fingpt["output"].unique()
print(unique_outputs)

for output in unique_outputs:
    samples = df_fingpt[df_fingpt["output"] == output]["input"].sample(
        n=4, random_state=1
    )
    print(f"Samples for output '{output}':")
    pprint(samples.tolist())
    print("---")


print(df_fingpt["output"].value_counts())

# %%

df_scam = datasets["scam"]
display(df_scam.head())
print(df_scam.shape)
unique_outputs = df_scam["label"].unique()
print(unique_outputs)

for output in unique_outputs:
    samples = df_scam[df_scam["label"] == output]["text"].sample(n=4, random_state=1)
    print(f"Samples for output '{output}':")
    pprint(samples.tolist())
    print("---")


print(df_scam["label"].value_counts())

# %%
df_all_scam_spam = datasets["all_scam_spam"]
display(df_all_scam_spam.head())
print(df_all_scam_spam.shape)
unique_outputs = df_all_scam_spam["is_spam"].unique()
print(unique_outputs)

for output in unique_outputs:
    samples = df_all_scam_spam[df_all_scam_spam["is_spam"] == output]["text"].sample(
        n=4
    )
    print(f"Samples for output '{output}':")
    for i in samples.tolist():
        print(f"\t{i[:100]}")
    print("---")
