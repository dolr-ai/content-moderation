# %% [markdown]
# # Vector Database Creation
#
# This notebook creates a vector database from the sampled text data:
# 1. Load the sampled data
# 2. Create embeddings using a sentence transformer
# 3. Build and save the FAISS vector database
# 4. Test basic similarity search

# %%
import os
import pandas as pd
from pathlib import Path
import logging
import yaml
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import login as hf_login
from tqdm.auto import tqdm

# %% [markdown]
# ## Setup and Configuration

# %%
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

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# %% [markdown]
# ## Load Data


# %%
def load_vector_db_data(file_path):
    """Load the sampled data for vector database creation"""

    df = pd.read_json(file_path, lines=True)
    logger.info(f"Loaded {len(df)} samples for vector database")
    return df


file_path = DATA_ROOT / "rag" / "vector_db_text.jsonl"
df = load_vector_db_data(file_path)
df = df.head(1000)

# %% [markdown]
# ## Create Embeddings


# %%
def create_embeddings(texts, model_name, batch_size=32):
    """
    Create embeddings for the input texts using a sentence transformer

    Args:
        texts (list): List of text strings to embed
        model_name (str): Name of the sentence transformer model
        batch_size (int): Batch size for embedding creation

    Returns:
        numpy.ndarray: Matrix of embeddings
    """
    model = SentenceTransformer(model_name, device=DEVICE, trust_remote_code=True)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, normalize_embeddings=True, device=DEVICE)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# Create embeddings
model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
embeddings = create_embeddings(df["text"].tolist(), model_name)
logger.info(f"Created embeddings with shape: {embeddings.shape}")

# %% [markdown]
# ## Build FAISS Index


# %%
def build_faiss_index(embeddings, index_type="L2"):
    """
    Build a FAISS index for the embeddings

    Args:
        embeddings (numpy.ndarray): Matrix of embeddings
        index_type (str): Type of FAISS index to create

    Returns:
        faiss.Index: FAISS index
    """
    dimension = embeddings.shape[1]

    if index_type == "L2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IP":  # Inner Product
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(embeddings.astype("float32"))
    return index


# Build index
index = build_faiss_index(embeddings)
logger.info(f"Built FAISS index with {index.ntotal} vectors")

# %% [markdown]
# ## Save Database


# %%
def save_vector_database(index, df, embeddings, save_dir):
    """
    Save the vector database components

    Args:
        index: FAISS index
        df: DataFrame with text and metadata
        embeddings: numpy array of embeddings
        save_dir: Directory to save the database
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(save_dir / "vector_db_test.faiss"))

    # Save metadata
    df.to_json(save_dir / "metadata.jsonl", orient="records", lines=True)

    # Save embeddings
    np.save(save_dir / "embeddings.npy", embeddings)

    logger.info(f"Saved vector database to {save_dir}")


# Save the database
save_vector_database(index, df, embeddings, DATA_ROOT / "rag" / "vector_db")

# %% [markdown]
# ## Test Search


# %%
def test_search(query_text, model_name, k=5):
    """
    Test similarity search with a query

    Args:
        query_text (str): Text to search for
        k (int): Number of results to return
    """
    # Create embedding for query
    query_embedding = create_embeddings([query_text], model_name)

    # Search
    D, I = index.search(query_embedding.astype("float32"), k)

    # Display results
    print(f"\nQuery: {query_text}\n")
    print("Top matches:")
    for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        text = df.iloc[idx]["text"]
        category = df.iloc[idx]["moderation_category"]
        print(f"\n{i}. Distance: {dist:.2f}")
        print(f"Category: {category}")
        print(f"Text: {text[:2000]}...")


# Test with a sample query
test_search("Yo nigga you need to get lost from this place", model_name, k=3)
