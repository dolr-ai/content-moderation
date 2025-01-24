"""
Goal: Relabel existing sentiment analysis datasets to final moderation labels
0. Neutral
1. Hate & Discrimination
2. Violence & Threats
3. Offensive Language
4. Sexual Content
5. Spam & Scams

"""

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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PRIMARY_CATEGORY_MAP = {
    "neutral": 0,
    "hate_or_discrimination": 1,
    "violence_or_threats": 2,
    "offensive_language": 3,
    "sexual_content": 4,
    "spam_or_scams": 5,
}

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
# # Load all datasets
dl = DatasetLoader()
datasets = {
    # "hate_speech": dl.load_hate_speech(),
    "jigsaw": dl.load_jigsaw_toxic(),
    # "twitter_comments": dl.load_twitter_comments(),
    # "fi_news": dl.load_financial_news_sentiment(),
    # "fingpt": dl.load_fingpt_sentiment(),
    # "scam": dl.load_scam_data(),
    # "all_scam_spam": dl.load_all_scam_spam(),
}
# %%[markdown]
# ## Relabel Hate Speech and Offensive Language dataset

df_hate_speech = datasets["hate_speech"]
df_hate_speech.head()

hate_speech_map = {
    0: "hate_speech",
    1: "offensive_language",
    # label 2: neither hate speech nor offensive language saw few examples with sexual content, terrorism, threats, etc.
    2: "neither_hsol",
}

df_hate_speech["target"] = df_hate_speech["class"].map(hate_speech_map)
df_hate_speech.head()


# check sample of neither hate speech nor offensive language
df_hate_speech[df_hate_speech["target"] == "neither_hsol"].sample(10)["tweet"].tolist()

# todo: get only hate speech and offensive language subsets from this dataset

# %% [markdown]
# ## Relabel Jigsaw Toxic Comment Classification dataset


def relabel_jigsaw(row, PRIMARY_CATEGORY_MAP):
    """
    Creates moderation labels aligned with primary content moderation categories

    Assumptions:
    1. We assume "toxic" alone indicates offensive language rather than hate
    2. We assume "obscene" could indicate either sexual content or offensive language
       - When combined with identity_hate, we prioritize hate category
    3. We assume severe_toxic combined with identity_hate indicates higher intensity hate speech
    4. We prioritize hate & discrimination over other categories when multiple flags exist

    Input columns needed:
    - toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    labels = []

    # Hate & Discrimination
    # Clear cases: identity_hate
    # Complex cases: severe_toxic + identity markers
    if row["identity_hate"] == 1:
        labels.append("hate_discrimination")

    # Violence & Threats
    # Clear cases: direct threats
    # Complex cases: severe_toxic + threat implications
    if row["threat"] == 1:
        labels.append("violence_threats")

    # Offensive Language
    # Cases: toxic, obscene (without sexual context), insults
    if (row["toxic"] == 1 or row["obscene"] == 1 or row["insult"] == 1) and not row[
        "identity_hate"
    ]:
        labels.append("offensive_language")

    # Sexual Content
    # Cases: obscene with sexual context
    # Note: This is a weak classifier as we can't definitively determine sexual content
    # from these labels alone
    if row["obscene"] == 1 and not row["identity_hate"]:
        labels.append("sexual_content")

    # Determine primary category (ordered by severity)
    primary_category = "clean"
    if "hate_or_discrimination" in labels:
        primary_category = "hate_or_discrimination"
    elif "violence_or_threats" in labels:
        primary_category = "violence_or_threats"
    elif "sexual_content" in labels:
        primary_category = "sexual_content"
    elif "offensive_language" in labels:
        primary_category = "offensive_language"

    return primary_category


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

df_jigsaw["joint_label"] = df_jigsaw[jigsaw_labels].apply(
    lambda x: (
        "neutral"
        if "-".join([i for i in jigsaw_labels if x[i] == 1]) == ""
        else "-".join([i for i in jigsaw_labels if x[i] == 1])
    ),
    axis=1,
)

df_jigsaw["joint_label"].value_counts().head(20)

# %%
df_jigsaw_joint_label_counts = df_jigsaw["joint_label"].value_counts().reset_index()

df_jigsaw_joint_label_samples = df_jigsaw.groupby("joint_label", as_index=False).agg(
    sample=("comment_text", lambda x: x.sample(min(100, len(x))).tolist())
)

df_jigsaw_samples_grp = df_jigsaw_joint_label_samples.merge(
    df_jigsaw_joint_label_counts, on="joint_label", how="left"
).sort_values(by="count", ascending=False)

# df_jigsaw_samples_grp.head(20).to_dict(orient="records")
# %%
# df_jigsaw_samples_grp[df_jigsaw_samples_grp["joint_label"] == "obscene"][
#     "sample"
# ].tolist()


print(np.hstack(df_jigsaw_joint_label_samples["sample"].tolist()).shape)

# %%

df_flat_jigsaw_samples = df_jigsaw_joint_label_samples.explode("sample")
df_flat_jigsaw_samples.head()

# %%


def get_batch_embeddings(
    df,
    text_column="sample",
    batch_size=8,
    model_path="Alibaba-NLP/gte-modernbert-base",
):
    """
    Get embeddings for text data in batches

    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of column containing text data
        batch_size (int): Batch size for processing
        model_path (str): HuggingFace model path

    Returns:
        np.ndarray: Array of embeddings
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode

    all_embeddings = []

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i : i + batch_size].tolist()

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            max_length=512,  # Adjust based on your needs
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Get embeddings
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token embedding
            embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize

        all_embeddings.append(embeddings.cpu().numpy())

        if (i + batch_size) % 50 == 0:
            logger.info(f"Processed {i + batch_size} samples")

    # Concatenate all batches
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings


if False:
    # Example usage:
    embeddings = get_batch_embeddings(df_flat_jigsaw_samples)

    print(embeddings.shape)

    df_flat_jigsaw_samples["embedding"] = embeddings.tolist()

    df_flat_jigsaw_samples.to_json(
        DATA_ROOT / "processed" / "jigsaw_sample_embeddings.jsonl",
        orient="records",
        lines=True,
    )

# %%


def perform_clustering(embeddings, n_clusters=3, random_state=42):
    """
    Perform KMeans clustering on embeddings.

    Parameters:
    -----------
    embeddings : array-like
        List or array of embedding vectors
    n_clusters : int, default=3
        Number of clusters to form
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    labels : array
        Cluster labels for each embedding
    cluster_centers : array
        Coordinates of cluster centers
    """
    X = np.asarray(embeddings)
    print(X.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    return labels, kmeans.cluster_centers_


def reduce_dimensions_tsne(embeddings, perplexity=30, n_components=2, random_state=42):
    """
    Reduce dimensionality of embeddings using t-SNE.

    Parameters:
    -----------
    embeddings : array-like
        List or array of embedding vectors
    perplexity : float, default=30
        The perplexity parameter for t-SNE
    n_components : int, default=2
        Number of components to reduce to
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    reduced_data : array
        Reduced dimensionality data
    """
    X = np.array(embeddings)
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    reduced_data = tsne.fit_transform(X)

    return reduced_data


def plot_clusters(reduced_data, labels, reduced_centers=None):
    """
    Create scatter plot of clustered data with numbered centers.

    Parameters:
    -----------
    reduced_data : array
        2D array of reduced dimensionality data
    labels : array
        Cluster labels for each point
    reduced_centers : array, optional
        Reduced dimensionality cluster centers

    Returns:
    --------
    fig : matplotlib figure
        The generated plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    scatter = ax.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.6
    )

    # Calculate and plot cluster centers in reduced space
    n_clusters = len(np.unique(labels))
    for i in range(n_clusters):
        # Get points belonging to cluster i
        cluster_points = reduced_data[labels == i]
        # Calculate center as mean of points
        center = np.mean(cluster_points, axis=0)

        # Plot cluster number
        ax.text(
            center[0],
            center[1],
            str(i + 1),  # Using 1-based numbering
            fontsize=15,
            fontweight="bold",
            bbox=dict(
                facecolor="white", edgecolor="black", boxstyle="circle", alpha=0.8
            ),
            horizontalalignment="center",
            verticalalignment="center",
            zorder=5,
        )  # Ensure numbers are on top

    # Add labels and colorbar
    ax.set_title("Embedding Clusters (t-SNE visualization)")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    plt.colorbar(scatter, label="Cluster Label")

    return fig


df_flat_jigsaw_samples = pd.read_json(
    DATA_ROOT / "processed" / "jigsaw_sample_embeddings.jsonl", lines=True
)

labels, kmeans_centers = perform_clustering(
    df_flat_jigsaw_samples["embedding"].tolist(),
    # n_clusters=len(df_flat_jigsaw_samples["joint_label"].unique()),
    n_clusters=len(PRIMARY_CATEGORY_MAP),
)

reduced_data = reduce_dimensions_tsne(df_flat_jigsaw_samples["embedding"].tolist())

plot_clusters(reduced_data, labels, reduced_centers=kmeans_centers)


# %%


def plot_clusters_grid(reduced_data, labels, df):
    """
    Create a 2x3 grid of scatter plots for individual clusters.

    Parameters:
    -----------
    reduced_data : array
        2D array of reduced dimensionality data
    labels : array
        Cluster labels for each point
    df : pandas DataFrame
        DataFrame containing the text samples
    """
    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    # Color map for consistency
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        # Plot points for current cluster
        mask = labels == i
        axes[i].scatter(
            reduced_data[~mask, 0],
            reduced_data[~mask, 1],
            c="lightgray",
            alpha=0.1,
            label="Other clusters",
            s=30,
        )
        axes[i].scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            label=f"Cluster {i}",
            s=50,
        )

        # Calculate and plot cluster center
        center = np.mean(reduced_data[mask], axis=0)
        axes[i].text(
            center[0],
            center[1],
            f"Cluster {i}",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
            horizontalalignment="center",
            verticalalignment="center",
        )

        axes[i].set_title(f"Cluster {i}", fontsize=14, pad=10)
        axes[i].legend(fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Create the visualization
fig = plot_clusters_grid(reduced_data, labels, df_flat_jigsaw_samples)
plt.show()

# Print sample texts and statistics for each cluster
print("\nCluster Analysis")
print("=" * 80)

for i in range(len(np.unique(labels))):
    cluster_mask = labels == i
    cluster_df = df_flat_jigsaw_samples[cluster_mask]

    print(f"\nCluster {i}")
    print("-" * 80)
    print(f"Number of samples: {len(cluster_df)}")
    print("\nMost common joint labels:")
    print(cluster_df["joint_label"].value_counts().head())
    print("\nSample texts:")
    for idx, text in enumerate(cluster_df["sample"].sample(n=5, random_state=42), 1):
        print(f"{idx}. {text[:150]}...")
    print("\n")

# %%


def map_to_primary_category(joint_label: str) -> str:
    """
    Maps the joint labels to primary moderation categories.
    Takes a joint_label string (hyphenated or single) and returns primary category.

    Args:
        joint_label: String containing one or more labels separated by hyphens

    Returns:
        str: Primary category name
    """
    # Convert to set of labels for easier checking
    labels = set(joint_label.split("-"))

    # Priority order of checks
    if "identity_hate" in labels:
        return "hate_discrimination"
    elif "threat" in labels:
        return "violence_threats"
    elif any(
        label in labels for label in ["toxic", "obscene", "insult", "severe_toxic"]
    ):
        return "offensive_language"
    else:
        return "clean"


df_flat_jigsaw_samples["refined_moderation_label"] = df_flat_jigsaw_samples[
    "joint_label"
].apply(map_to_primary_category)

# %%
df_flat_jigsaw_samples
# %%


def plot_tsne_moderation_labels(df, reduced_data):
    """
    Create t-SNE visualizations for moderation labels.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the 'refined_moderation_label' column
    reduced_data : numpy array
        2D array of reduced dimensionality data from t-SNE

    Returns:
    --------
    None. Displays two plots:
    1. Combined plot with all labels
    2. Individual plots for each label
    """
    # Create combined plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a color map for the unique labels
    unique_labels = df["refined_moderation_label"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    # Plot points colored by refined moderation label
    for label in unique_labels:
        mask = df["refined_moderation_label"] == label
        ax.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[label_color_map[label]],
            label=label,
            alpha=0.6,
        )

        # Calculate and plot center for each label
        center = np.mean(reduced_data[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            label,
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.set_title("Combined t-SNE Plot by Refined Moderation Labels")
    ax.legend()
    plt.show()

    # Create individual plots
    n_labels = len(unique_labels)
    n_cols = 2
    n_rows = (n_labels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.ravel()

    for idx, label in enumerate(unique_labels):
        mask = df["refined_moderation_label"] == label

        # Plot background points
        axes[idx].scatter(
            reduced_data[~mask, 0],
            reduced_data[~mask, 1],
            c="lightgray",
            alpha=0.1,
            label="Other labels",
        )

        # Plot points for current label
        axes[idx].scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[label_color_map[label]],
            alpha=0.6,
            label=label,
        )

        # Calculate and plot center
        center = np.mean(reduced_data[mask], axis=0)
        axes[idx].text(
            center[0],
            center[1],
            label,
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
            horizontalalignment="center",
            verticalalignment="center",
        )

        axes[idx].set_title(f"{label} Distribution")
        axes[idx].legend()

    # Remove empty subplots if any
    for idx in range(len(unique_labels), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

    # Print statistics for each label
    print("\nLabel Distribution:")
    print(df["refined_moderation_label"].value_counts())


# Example usage:
plot_tsne_moderation_labels(df_flat_jigsaw_samples, reduced_data)

# %%

df_flat_jigsaw_samples.shape

datasets["jigsaw"].shape
