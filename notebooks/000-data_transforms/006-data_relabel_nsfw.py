"""
Goal: Relabel existing sentiment analysis datasets to final moderation labels
0. clean
1. Hate & Discrimination
2. Violence & Threats
3. Offensive Language
4. Sexual Content
5. Spam & Scams
"""

# %% [markdown]
# ## Profanity dataset relabeling
#
# This notebook explores the Sentiment140 dataset and relabels it into moderation categories:
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
from dotenv import load_dotenv
import yaml
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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

    def load_nsfw(self):
        """Load NSFW dataset"""
        df = pd.read_parquet(
            "hf://datasets/amaye15/NSFW-descriptions/data/train-00000-of-00001.parquet"
        )
        return df


# %%
dl = DatasetLoader()
df_nsfw = dl.load_nsfw()
# %%
df_nsfw = df_nsfw[df_nsfw["language"] == "English"].reset_index(drop=True)
# %%
df_nsfw.shape[0]
# %%
df_nsfw["moderation_category"] = "nsfw_content"
df_nsfw["moderation_label"] = df_nsfw["moderation_category"].map(
    lambda x: PRIMARY_CATEGORY_MAP.get(x, "unknown")
)
# %%
df_nsfw = df_nsfw[df_nsfw["text"].str.split(" ").apply(len) > 5].reset_index(drop=True)
# %%
df_nsfw.drop(columns=["language"]).to_json(
    DATA_ROOT / "processed" / "nsfw_data-relabeled.jsonl",
    orient="records",
    lines=True,
)
# %%
df_nsfw.shape[0]
# %%
