# %% [markdown]
# # Content Moderation Dataset Analysis
#
# This notebook explores sentiment analysis datasets and relabels them into moderation categories:
# - 0: Clean
# - 1: Hate & Discrimination
# - 2: Violence & Threats
# - 3: Offensive Language
# - 4: Sexual Content
# - 5: Spam & Scams

# %% [markdown]
# ## Import Required Libraries

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

# %% [markdown]
# ## Configuration Setup
# Define the primary category mapping and configurations

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

# %% [markdown]
# ## Dataset Loading Functionality
# Define the DatasetLoader class to handle various dataset sources


# %%
class DatasetLoader:
    """Base class for loading various sentiment and content moderation datasets"""

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
# ## Load Datasets
# For this analysis, we'll focus on the Jigsaw dataset while keeping the infrastructure for other datasets

# %%
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

# %% [markdown]
# ## Analyze Jigsaw Dataset
# Explore the structure and create joint labels

# %%
df_jigsaw = datasets["jigsaw"]

# Display sample data and shape
display(df_jigsaw.head())
print(f"Dataset shape: {df_jigsaw.shape}")

# Define Jigsaw label columns
jigsaw_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Create joint labels
df_jigsaw["joint_label"] = df_jigsaw[jigsaw_labels].apply(
    lambda x: (
        "clean"
        if "-".join([i for i in jigsaw_labels if x[i] == 1]) == ""
        else "-".join([i for i in jigsaw_labels if x[i] == 1])
    ),
    axis=1,
)

# Display label distribution
print("\nJoint Label Distribution (Top 20):")
print(df_jigsaw["joint_label"].value_counts().head(20))

# %% [markdown]
# ## Sample Creation and Analysis

# %%
# Create sample groups
df_jigsaw_joint_label_counts = df_jigsaw["joint_label"].value_counts().reset_index()

df_jigsaw_joint_label_samples = df_jigsaw.groupby("joint_label", as_index=False).agg(
    sample=("comment_text", lambda x: x.sample(min(100, len(x))).tolist())
)

df_jigsaw_samples_grp = df_jigsaw_joint_label_samples.merge(
    df_jigsaw_joint_label_counts, on="joint_label", how="left"
).sort_values(by="count", ascending=False)

# Create flat samples dataframe
df_flat_jigsaw_samples = df_jigsaw_joint_label_samples.explode("sample")
print("\nFlat samples shape:", df_flat_jigsaw_samples.shape)

# %% [markdown]
# ## Text Embedding Generation
# Define function to generate embeddings using the Alibaba-NLP/gte-modernbert-base model


# %%
def get_batch_embeddings(
    df,
    text_column="sample",
    batch_size=8,
    model_path="Alibaba-NLP/gte-modernbert-base",
):
    """
    Generate embeddings for text data in batches
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    all_embeddings = []

    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i : i + batch_size].tolist()

        batch_dict = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

        if (i + batch_size) % 50 == 0:
            logger.info(f"Processed {i + batch_size} samples")

    return np.vstack(all_embeddings)


# %% [markdown]
# ## Generate or Load Embeddings
# Handle both first-time generation and loading from existing file

# %%
embedding_file = DATA_ROOT / "processed" / "jigsaw_sample_embeddings.jsonl"

if embedding_file.exists():
    # Load existing embeddings
    logger.info("Loading existing embeddings from file...")
    df_flat_jigsaw_samples = pd.read_json(embedding_file, lines=True)
    embeddings = np.array(df_flat_jigsaw_samples["embedding"].tolist())
else:
    # Generate new embeddings
    logger.info("Generating new embeddings...")
    embeddings = get_batch_embeddings(df_flat_jigsaw_samples)
    df_flat_jigsaw_samples["embedding"] = embeddings.tolist()

    # Save embeddings
    df_flat_jigsaw_samples.to_json(
        embedding_file,
        orient="records",
        lines=True,
    )

print("Embeddings shape:", embeddings.shape)

# %% [markdown]
# ## Clustering Analysis Functions


# %%
def perform_clustering(embeddings, n_clusters=3, random_state=42):
    """Perform KMeans clustering on embeddings"""
    X = np.asarray(embeddings)
    print(f"Clustering data shape: {X.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    return labels, kmeans.cluster_centers_


def reduce_dimensions_tsne(embeddings, perplexity=30, n_components=2, random_state=42):
    """Reduce dimensionality using t-SNE"""
    X = np.array(embeddings)
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    return tsne.fit_transform(X)


def plot_clusters(reduced_data, labels, reduced_centers=None):
    """Create scatter plot of clustered data"""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.6
    )

    n_clusters = len(np.unique(labels))
    for i in range(n_clusters):
        cluster_points = reduced_data[labels == i]
        center = np.mean(cluster_points, axis=0)

        ax.text(
            center[0],
            center[1],
            str(i + 1),
            fontsize=15,
            fontweight="bold",
            bbox=dict(
                facecolor="white", edgecolor="black", boxstyle="circle", alpha=0.8
            ),
            horizontalalignment="center",
            verticalalignment="center",
            zorder=5,
        )

    ax.set_title("Embedding Clusters (t-SNE visualization)")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    plt.colorbar(scatter, label="Cluster Label")

    return fig


# %% [markdown]
# ## Perform Clustering Analysis

# %%
# Generate clusters
labels, kmeans_centers = perform_clustering(
    df_flat_jigsaw_samples["embedding"].tolist(), n_clusters=len(PRIMARY_CATEGORY_MAP)
)

# Reduce dimensions for visualization
reduced_data = reduce_dimensions_tsne(df_flat_jigsaw_samples["embedding"].tolist())

# Plot clusters
plt.figure(figsize=(12, 8))
plot_clusters(reduced_data, labels, reduced_centers=kmeans_centers)
plt.show()

# %% [markdown]
# ## Detailed Cluster Analysis


# %%
def plot_clusters_grid(reduced_data, labels, df):
    """Create a grid of scatter plots for individual clusters"""
    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        mask = labels == i

        # Plot background points
        axes[i].scatter(
            reduced_data[~mask, 0],
            reduced_data[~mask, 1],
            c="lightgray",
            alpha=0.1,
            label="Other clusters",
            s=30,
        )

        # Plot cluster points
        axes[i].scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            label=f"Cluster {i}",
            s=50,
        )

        # Plot cluster center
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


# Create visualization
fig = plot_clusters_grid(reduced_data, labels, df_flat_jigsaw_samples)
plt.show()

# Print cluster analysis
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

# %% [markdown]
# ## Primary Category Mapping and Visualization


# %%
def map_to_primary_category(joint_label: str) -> str:
    """
    Maps joint labels to primary moderation categories.

    Args:
        joint_label: String containing one or more labels separated by hyphens

    Returns:
        str: Primary category name
    """
    labels = set(joint_label.split("-"))

    # Priority-based mapping
    if "identity_hate" in labels:
        return "hate_or_discrimination"
    elif "threat" in labels:
        return "violence_or_threats"
    elif any(
        label in labels for label in ["toxic", "obscene", "insult", "severe_toxic"]
    ):
        return "offensive_language"
    else:
        return "clean"


# Apply mapping to samples
df_flat_jigsaw_samples["refined_moderation_label"] = df_flat_jigsaw_samples[
    "joint_label"
].apply(map_to_primary_category)

# %% [markdown]
# ## Visualize Moderation Labels Distribution


# %%
def plot_tsne_moderation_labels(df, reduced_data):
    """Create t-SNE visualizations for moderation labels"""
    # Combined plot
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = df["refined_moderation_label"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    # Plot points by moderation label
    for label in unique_labels:
        mask = df["refined_moderation_label"] == label
        ax.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[label_color_map[label]],
            label=label,
            alpha=0.6,
        )

        # Plot label centers
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

    # Individual plots
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

        # Plot label points
        axes[idx].scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[label_color_map[label]],
            alpha=0.6,
            label=label,
        )

        # Plot label center
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

    # Remove empty subplots
    for idx in range(len(unique_labels), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

    # Print label distribution statistics
    print("\nLabel Distribution:")
    print(df_flat_jigsaw_samples["refined_moderation_label"].value_counts())


# %% [markdown]
# ## Generate Final Visualizations and Statistics

# %%
# Create moderation label visualizations
plot_tsne_moderation_labels(df_flat_jigsaw_samples, reduced_data)

# Print final dataset statistics
print("\nDataset Statistics:")
print("-" * 50)
print(f"Total samples in flat dataset: {df_flat_jigsaw_samples.shape[0]}")
print(f"Original Jigsaw dataset size: {datasets['jigsaw'].shape[0]}")

# Print samples for each moderation label
print("\nSample texts for each moderation label:")
print("=" * 80)
for label in df_flat_jigsaw_samples["refined_moderation_label"].unique():
    print(f"\n{label.upper()}")
    print("-" * 80)
    samples = df_flat_jigsaw_samples[
        df_flat_jigsaw_samples["refined_moderation_label"] == label
    ]["sample"].sample(n=10, random_state=42)
    for idx, text in enumerate(samples, 1):
        print(f"{idx}. {text[:200]}...")
    print()

# %% [markdown]
# # Finalize dataset based on relablelling logic

# %% [markdown]
# ## Relabel Original Jigsaw Dataset


# save the relabeled dataset
datasets = {
    "jigsaw": dl.load_jigsaw_toxic(),
}

df_jigsaw = datasets["jigsaw"]

# Define Jigsaw label columns
jigsaw_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Create joint labels for original dataset
df_jigsaw["joint_label"] = df_jigsaw[jigsaw_labels].apply(
    lambda x: (
        "clean"
        if "-".join([i for i in jigsaw_labels if x[i] == 1]) == ""
        else "-".join([i for i in jigsaw_labels if x[i] == 1])
    ),
    axis=1,
)

# Apply mapping to create moderation categories
df_jigsaw["moderation_category"] = df_jigsaw["joint_label"].apply(
    map_to_primary_category
)

# Convert string labels to numeric using PRIMARY_CATEGORY_MAP
df_jigsaw["moderation_label"] = df_jigsaw["moderation_category"].map(
    PRIMARY_CATEGORY_MAP
)

# Create final dataset with essential columns
df_final = (
    df_jigsaw[["comment_text", "moderation_category", "moderation_label"]]
    .rename(columns={"comment_text": "text"})
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
output_path = DATA_ROOT / "processed" / "jigsaw-relabelled.jsonl"
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
