#%% [markdown]
# Experimenting with FAISS Vector Database for Content Moderation
#
# This notebook demonstrates how to use the FAISS vector database for content similarity search
# using the EmbeddingClient from our previous implementation.

#%% [markdown]
# ## 1. Setup and Imports
# First, let's import necessary libraries and set up our client

#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from openai import OpenAI
import faiss
from tqdm.auto import tqdm
import logging
from typing import List, Optional, Dict, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


#%% [markdown]
# ## 2. Embedding Client Implementation
# We'll use the same EmbeddingClient implementation as before


class EmbeddingClient:
    """Client for interacting with the embedding model and vector database"""

    def __init__(
        self,
        base_url: str = "http://localhost:8890/v1",
        api_key: str = "None",
        model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        vector_db_path: Optional[str] = None
    ):
        """
        Initialize the embedding client

        Args:
            base_url: The base URL for the embedding server
            api_key: API key for authentication (if needed)
            model_name: Name of the embedding model to use
            vector_db_path: Path to the vector database directory (if searching)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # Load vector database if path is provided
        self.index = None
        self.metadata_df = None

        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def load_vector_database(self, db_path: Union[str, Path]):
        """
        Load a FAISS vector database and its metadata

        Args:
            db_path: Path to the vector database directory
        """
        db_path = Path(db_path)

        # Load FAISS index
        index_path = db_path / "vector_db_text.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            logger.warning(f"FAISS index not found at {index_path}")
            return False

        # Load metadata
        metadata_path = db_path / "metadata.jsonl"
        if metadata_path.exists():
            self.metadata_df = pd.read_json(metadata_path, lines=True)
            logger.info(f"Loaded metadata with {len(self.metadata_df)} records")
        else:
            logger.warning(f"Metadata not found at {metadata_path}")
            return False

        return True

    def create_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for text input

        Args:
            text: Single text string or list of text strings

        Returns:
            numpy.ndarray: Generated embeddings
        """
        if isinstance(text, str):
            text = [text]

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query text

        Args:
            query: Query text to search for
            k: Number of results to return
            return_scores: Whether to include similarity scores in results

        Returns:
            List of dictionaries containing search results with metadata
        """
        if self.index is None or self.metadata_df is None:
            raise ValueError("Vector database not loaded. Call load_vector_database first.")

        # Create embedding for query
        query_embedding = self.create_embedding(query)

        # Search
        D, I = self.index.search(query_embedding.astype("float32"), k)

        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            result = {
                "index": int(idx),
                "text": self.metadata_df.iloc[idx]["text"],
                "category": self.metadata_df.iloc[idx].get("moderation_category", None)
            }

            if return_scores:
                result["distance"] = float(dist)

            results.append(result)

        return results

    def batch_create_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for a large batch of texts

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding creation
            show_progress: Whether to show a progress bar

        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Creating embeddings")

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.create_embedding(batch)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def build_vector_database(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        index_type: str = "L2",
        save_dir: Optional[Union[str, Path]] = None
    ):
        """
        Build a complete vector database from texts

        Args:
            texts: List of text strings to include
            metadata: Optional list of metadata dictionaries for each text
            batch_size: Batch size for embedding creation
            index_type: Type of FAISS index ("L2" or "IP")
            save_dir: Optional directory to save the database

        Returns:
            Tuple of (index, metadata_df, embeddings)
        """
        # Create embeddings
        embeddings = self.batch_create_embeddings(texts, batch_size)

        # Build FAISS index
        dimension = embeddings.shape[1]

        if index_type == "L2":
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":  # Inner Product
            index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        index.add(embeddings.astype("float32"))

        # Create metadata DataFrame
        if metadata is None:
            metadata = [{"text": text} for text in texts]

        metadata_df = pd.DataFrame(metadata)

        # Save if requested
        if save_dir:
            self._save_vector_database(index, metadata_df, embeddings, save_dir)

        # Update client state
        self.index = index
        self.metadata_df = metadata_df

        return index, metadata_df, embeddings

    def _save_vector_database(self, index, df, embeddings, save_dir):
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
        faiss.write_index(index, str(save_dir / "vector_db_text.faiss"))

        # Save metadata
        df.to_json(save_dir / "metadata.jsonl", orient="records", lines=True)

        # Save embeddings
        np.save(save_dir / "embeddings.npy", embeddings)

        logger.info(f"Saved vector database to {save_dir}")
        logger.info(f"- FAISS index: {save_dir}/vector_db_text.faiss")
        logger.info(f"- Metadata: {save_dir}/metadata.jsonl")
        logger.info(f"- Embeddings: {save_dir}/embeddings.npy")


# Initialize the client
client = EmbeddingClient(
    base_url="http://localhost:8890/v1",
    api_key="None",
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
)

#%% [markdown]
# ## 2. Basic Embedding Generation
# Let's first test creating embeddings for some sample texts

#%%
# Sample texts for testing
test_texts = [
    "This is a friendly greeting message",
    "You are going to regret this, I will find you",
    "Have a nice day everyone!",
    "I'm going to hurt you badly",
    "Let's discuss this peacefully"
]

# Generate embeddings
embeddings = client.create_embedding(test_texts)
print(f"Generated {len(embeddings)} embeddings with dimension {embeddings[0].shape[0]}")

# Show sample embedding
print("\nFirst embedding (first 10 dimensions):")
print(embeddings[0][:10])

#%% [markdown]
# ## 3. Loading Existing Vector Database
# Now let's load the vector database we created earlier

#%%
# Path to your vector database
DB_PATH = "/root/content-moderation/data/rag/faiss_vector_db"

# Load the database
success = client.load_vector_database(DB_PATH)
if success:
    print(f"Successfully loaded vector database with {client.index.ntotal} vectors")
    print(f"Metadata contains {len(client.metadata_df)} records")
else:
    print("Failed to load vector database")

#%% [markdown]
# ## 4. Similarity Search Examples
# Let's try some example searches with different types of content

#%%
def display_search_results(query: str, k: int = 5):
    """Helper function to display search results nicely"""
    print(f"\nQuery: {query}")
    print("-" * 80)

    results = client.similarity_search(query, k=k)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"Category: {result.get('category', 'N/A')}")

        # Truncate long texts
        text = result["text"]
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"Text: {text}")

#%%
# Test different types of queries
queries = [
    "Hello, how are you doing today?",
    "I will find you and make you pay for this",
    "Let's have a peaceful discussion about this topic",
    "You better watch your back",
    "Thank you for your help"
]

for query in queries:
    display_search_results(query, k=3)

#%% [markdown]
# ## 5. Analyzing Embedding Similarity
# Let's analyze similarity between different types of content

#%%
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    emb1 = client.create_embedding(text1)
    emb2 = client.create_embedding(text2)

    # Normalize vectors
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    # Calculate cosine similarity
    similarity = np.dot(emb1_norm[0], emb2_norm[0])
    return similarity

#%%
# Test pairs of texts
text_pairs = [
    ("Hello, how are you?", "Hi, how's it going?"),
    ("I will hurt you", "I'm going to cause you pain"),
    ("Let's talk peacefully", "We should discuss this calmly"),
    ("You're dead meat", "I'm coming for you"),
    ("Have a great day", "Enjoy your day")
]

print("Similarity Analysis:")
print("-" * 80)
for text1, text2 in text_pairs:
    similarity = calculate_similarity(text1, text2)
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")

#%% [markdown]
# ## 6. Batch Processing Example
# Demonstrate how to process multiple texts efficiently

#%%
# Create a list of test texts
batch_texts = [
    f"Sample text {i} for batch processing" for i in range(10)
]

# Process in batch
batch_embeddings = client.batch_create_embeddings(
    batch_texts,
    batch_size=4,
    show_progress=True
)

print(f"\nProcessed {len(batch_embeddings)} texts")
print(f"Embedding shape: {batch_embeddings.shape}")

#%% [markdown]
# ## 7. Advanced Search with Metadata Filtering
# Demonstrate how to combine vector search with metadata filtering

#%%
def search_with_category_filter(query: str, category: str, k: int = 5):
    """Search with additional category filtering"""
    # First get more results than needed since we'll filter
    results = client.similarity_search(query, k=k*2)

    # Filter by category
    filtered_results = [
        r for r in results
        if r.get('category') == category
    ]

    return filtered_results[:k]

#%%
# Test category-filtered search
test_queries = [
    ("I will find you", "threat"),
    ("Have a nice day", "neutral"),
    ("You're going to regret this", "threat")
]

for query, category in test_queries:
    print(f"\nSearching for '{query}' in category '{category}':")
    print("-" * 80)

    results = search_with_category_filter(query, category, k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Text: {result['text'][:200]}...")
