"""
Goal: Relabel existing sentiment analysis datasets to final moderation labels
0. Neutral
1. Hate & Discrimination
2. Violence & Threats
3. Offensive Language
4. Sexual Content
5. Spam & Scams
"""

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
    "sexual_content": 4,
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


# %%
dl = DatasetLoader()
df_sent140 = dl.load_twitter_comments()

# %%
df_sent140.head()

# %%

df_neg = df_sent140[df_sent140["target"] == 0].sample(n=1000, random_state=42)
df_neg["label"] = "negative"

df_pos = df_sent140[df_sent140["target"] == 4].sample(n=1000, random_state=42)
df_pos["label"] = "positive"

df_sent_sample = pd.concat([df_neg, df_pos], axis=0, ignore_index=True)
df_sent_sample.head()


# %% [markdown]
# ## Text Embedding Generation


# %%
def get_batch_embeddings(
    df,
    text_column="text",
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

# %%
embedding_file = DATA_ROOT / "processed" / "sent140_sample_embeddings.jsonl"

if embedding_file.exists():
    # Load existing embeddings
    logger.info("Loading existing embeddings from file...")
    df_sent_sample = pd.read_json(embedding_file, lines=True)
    embeddings = np.array(df_sent_sample["embedding"].tolist())
else:
    # Generate new embeddings
    logger.info("Generating new embeddings...")
    embeddings = get_batch_embeddings(df_sent_sample)
    df_sent_sample["embedding"] = embeddings.tolist()

    # Save embeddings
    df_sent_sample.to_json(
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

    ax.set_title("Sentiment140 Sample Embedding Clusters (t-SNE visualization)")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    plt.colorbar(scatter, label="Cluster Label")

    return fig


def plot_clusters_grid(reduced_data, labels, df):
    """Create a grid of scatter plots for individual clusters"""
    n_clusters = len(np.unique(labels))

    # Calculate required grid dimensions
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))
    if n_clusters == 1:
        axes = np.array([axes])
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

    # Remove empty subplots if any
    for idx in range(n_clusters, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


# %% [markdown]
# ## Perform Clustering Analysis

# %%
# Generate clusters
n_clusters = 6  # You can adjust this number
labels, kmeans_centers = perform_clustering(embeddings, n_clusters=n_clusters)

# Reduce dimensions for visualization
reduced_data = reduce_dimensions_tsne(embeddings)

# Plot clusters
plt.figure(figsize=(12, 8))
plot_clusters(reduced_data, labels, reduced_centers=kmeans_centers)
plt.show()

# Create detailed visualization
fig = plot_clusters_grid(reduced_data, labels, df_sent_sample)
plt.show()
# %%
# Print cluster analysis
print("\nCluster Analysis")
print("=" * 80)

for i in range(n_clusters):
    cluster_mask = labels == i
    cluster_df = df_sent_sample[cluster_mask]

    print(f"\nCluster {i}")
    print("-" * 80)
    print(f"Number of samples: {len(cluster_df)}")

    # Show label counts for the current cluster
    label_counts = cluster_df["label"].value_counts()
    print("Label counts:")
    print(label_counts)

    print("\nSample texts:")
    for idx, text in enumerate(cluster_df["text"].sample(n=15, random_state=42), 1):
        print(f"{idx}. {text[:600]}...")
    print("\n")

# %%

# %% [markdown]
# ## Load Jigsaw Embeddings and Labels

# %%
# Load Jigsaw embeddings and labels
jigsaw_embedding_file = DATA_ROOT / "processed" / "jigsaw_sample_embeddings.jsonl"

# Load Jigsaw embeddings
df_jigsaw_embeddings = pd.read_json(jigsaw_embedding_file, lines=True)
jigsaw_embeddings = np.array(df_jigsaw_embeddings["embedding"].tolist())

df_jigsaw_req = df_jigsaw_embeddings

# %% [markdown]
# ## Compute Cosine Similarities


# %%
def compute_batch_cosine_similarities(embeddings1, embeddings2, batch_size=100):
    """
    Compute cosine similarities between two sets of embeddings in batches
    to manage memory usage
    """
    n_samples1 = len(embeddings1)
    n_samples2 = len(embeddings2)
    similarities = np.zeros((n_samples1, n_samples2))

    for i in range(0, n_samples1, batch_size):
        batch_end = min(i + batch_size, n_samples1)
        batch1 = embeddings1[i:batch_end]

        # Normalize batch1
        batch1_normalized = batch1 / np.linalg.norm(batch1, axis=1)[:, np.newaxis]

        for j in range(0, n_samples2, batch_size):
            j_end = min(j + batch_size, n_samples2)
            batch2 = embeddings2[j:j_end]

            # Normalize batch2
            batch2_normalized = batch2 / np.linalg.norm(batch2, axis=1)[:, np.newaxis]

            # Compute similarities for this batch pair
            similarities[i:batch_end, j:j_end] = np.dot(
                batch1_normalized, batch2_normalized.T
            )

        if (i + batch_size) % 50 == 0:
            logger.info(f"Processed {i + batch_size}/{n_samples1} samples")

    return similarities


# Compute similarities
logger.info("Computing cosine similarities...")
similarities = compute_batch_cosine_similarities(embeddings, jigsaw_embeddings)
# %%
similarities

# %% [markdown]
# ## Analyze Similarity Distribution

# %%
# Define configurable thresholds
SIMILARITY_THRESHOLD = 0.55  # Similarity threshold for considering a match
PERCENTAGE_THRESHOLD = 70  # Percentage threshold for clean classification

# Calculate percentage of low similarities for each Sent140 sentence
# For each row (Sent140 sentence), count how many Jigsaw sentences have similarity < threshold
low_sim_counts = (similarities < SIMILARITY_THRESHOLD).sum(axis=1)
# Convert to percentage (divide by total number of Jigsaw sentences)
low_sim_percentage = (low_sim_counts / similarities.shape[1]) * 100

print(f"Shape of similarities matrix: {similarities.shape}")
print(f"Number of Jigsaw sentences: {similarities.shape[1]}")
print(f"Number of Sent140 sentences: {similarities.shape[0]}")

# Plot distribution of percentage of low similarities
plt.figure(figsize=(10, 6))
plt.hist(low_sim_percentage, bins=50)
plt.title(f"Distribution of Percentage of Low Similarities (<{SIMILARITY_THRESHOLD})")
plt.xlabel(f"Percentage of Jigsaw Sentences with Similarity <{SIMILARITY_THRESHOLD}")
plt.ylabel("Count")
plt.axvline(
    x=PERCENTAGE_THRESHOLD,
    color="r",
    linestyle="--",
    label=f"{PERCENTAGE_THRESHOLD}% threshold",
)
plt.legend()
plt.show()

# Print statistics
print("\nLow Similarity Percentage Statistics:")
print("-" * 50)
print(f"Mean percentage: {np.mean(low_sim_percentage):.1f}%")
print(f"Median percentage: {np.median(low_sim_percentage):.1f}%")
print(f"Std percentage: {np.std(low_sim_percentage):.1f}%")
print(f"Min percentage: {np.min(low_sim_percentage):.1f}%")
print(f"Max percentage: {np.max(low_sim_percentage):.1f}%")

# Identify clean sentences (>threshold% low similarities)
clean_mask = low_sim_percentage > PERCENTAGE_THRESHOLD
clean_samples = df_sent_sample[clean_mask].copy()

# Map to primary category (0 for clean)
clean_samples["moderation_label"] = clean_samples["label"].map(
    {
        "positive": PRIMARY_CATEGORY_MAP["clean"],
        "negative": PRIMARY_CATEGORY_MAP["clean"],
    }
)

# Save the clean samples
output_path = DATA_ROOT / "processed" / "sent140-relabelled.jsonl"
clean_samples[["text", "moderation_label"]].to_json(
    output_path, orient="records", lines=True
)

# Print summary and examples
print(
    f"\nFound {len(clean_samples)} clean samples ({(len(clean_samples)/len(df_sent_sample)*100):.1f}% of total)"
)
print(f"Saved clean samples to: {output_path}")

print("\nExample clean samples:")
for _, row in clean_samples[["text", "moderation_label"]].head(5).iterrows():
    print(f"\nText: {row['text']}")
    print(f"Label: {row['moderation_label']}")
    print(f"Percentage of low similarities: {low_sim_percentage[_]:.1f}%")
