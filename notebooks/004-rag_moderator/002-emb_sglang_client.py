#!/usr/bin/env python3
"""
Embedding Client for Content Moderation Vector Database

This script provides a client interface to create embeddings and search
through the content moderation vector database using the SGLang server.
It uses the OpenAI client for API compatibility.

# create the vector database
python ./notebooks/004-rag_moderator/002-emb_sglang_client.py \
  --input-jsonl /root/content-moderation/data/rag/vector_db_text.jsonl \
  --save-dir /root/content-moderation/data/rag/faiss_vector_db
  --sample 1000 # exclude this parameter for full load

# create and search the vector database
python ./notebooks/004-rag_moderator/002-emb_sglang_client.py \
  --input-jsonl /root/content-moderation/data/rag/vector_db_text.jsonl \
  --save-dir /root/content-moderation/data/rag/faiss_vector_db \
  --db-path /root/content-moderation/data/rag/faiss_vector_db \
  --query "You wait right there i am coming for you"

# search existing vector database
python ./notebooks/004-rag_moderator/002-emb_sglang_client.py \
  --db-path /root/content-moderation/data/rag/faiss_vector_db \
  --query "You wait right there i am coming for you"
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm.auto import tqdm
from openai import OpenAI
import logging
from typing import List, Optional, Dict, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Client for embedding and searching content moderation database"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8890/v1",
        help="Base URL for the embedding server (default: http://localhost:8890/v1)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="None",
        help="API key for authentication (default: None)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Name of the embedding model (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query text (for search mode)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to vector database directory (for search mode)"
    )
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)"
    )

    return parser.parse_args()


def demo_embedding(client):
    """Demo embedding functionality"""
    print("\n=== Embedding Demo ===")

    # Create embeddings for sample texts
    texts = [
        "This is a test sentence for embedding.",
        "Here's another example of text to embed.",
        "Embeddings are useful for semantic search and similarity."
    ]

    embeddings = client.create_embedding(texts)

    # Display sample
    print(f"Created {len(embeddings)} embeddings with dimension {embeddings[0].shape[0]}")
    print(f"\nSample embedding (first 10 values):\n{embeddings[0][:10]}")

    # Calculate similarity between first two embeddings
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"\nCosine similarity between first two texts: {similarity:.4f}")


def demo_search(client, query, k=5):
    """Demo search functionality"""
    print(f"\n=== Search Demo ===")
    print(f"Query: {query}")

    results = client.similarity_search(query, k=k)

    print(f"\nTop {len(results)} matches:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"Category: {result.get('category', 'N/A')}")

        # Truncate long texts
        text = result["text"]
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"Text: {text}")


def demo_create_vector_db(client, texts, metadata=None, save_dir=None):
    """Demo vector database creation functionality"""
    print("\n=== Vector Database Creation Demo ===")
    print(f"Creating vector database with {len(texts)} texts")

    index, metadata_df, embeddings = client.build_vector_database(
        texts=texts,
        metadata=metadata,
        batch_size=32,
        index_type="L2",
        save_dir=save_dir
    )

    print(f"\nCreated vector database:")
    print(f"- Number of vectors: {index.ntotal}")
    print(f"- Embedding dimension: {embeddings.shape[1]}")
    print(f"- Metadata fields: {', '.join(metadata_df.columns)}")

    if save_dir:
        print(f"\nDatabase saved to: {save_dir}")

    return index, metadata_df, embeddings


def create_vector_db_from_jsonl(client, input_jsonl, save_dir, text_field="text", batch_size=32, sample_size=None):
    """
    Create a vector database from a JSONL file containing texts and metadata

    Args:
        client: EmbeddingClient instance
        input_jsonl: Path to input JSONL file
        save_dir: Directory to save the vector database
        text_field: Name of the field containing text in the JSONL
        batch_size: Batch size for embedding creation
        sample_size: Number of records to sample (None for all records)
    """
    # Load data
    df = pd.read_json(input_jsonl, lines=True)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)))
    logger.info(f"Loaded {len(df)} records from {input_jsonl}")

    # Extract texts and metadata
    texts = df[text_field].tolist()
    metadata = df.to_dict('records')

    # Create vector database
    return demo_create_vector_db(client, texts, metadata, save_dir)


def main():
    """Main function for the embedding client demo and operations.

    Usage Examples:
        1. Create vector database:
           python script.py --input-jsonl data.jsonl --save-dir /path/to/save/db --sample 1000

        2. Search existing database:
           python script.py --db-path /path/to/db --query 'your search query'

        3. Create and search in one command:
           python script.py --input-jsonl data.jsonl --save-dir /path/to/db --query 'your query' --db-path /path/to/db --sample 1000
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Client for embedding and searching content moderation database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8890/v1",
        help="Base URL for the embedding server (default: http://localhost:8890/v1)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="None",
        help="API key for authentication (default: None)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Name of the embedding model (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query text (for search mode)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to vector database directory (for search mode)"
    )
    parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        help="Input JSONL file for creating vector database"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save the vector database"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of records to sample from input JSONL (default: use all records)"
    )

    args = parser.parse_args()

    # Create client
    client = EmbeddingClient(
        base_url=args.server,
        api_key=args.api_key,
        model_name=args.model
    )

    # First handle vector database creation if requested
    if args.input_jsonl and args.save_dir:
        logger.info(f"Creating vector database from {args.input_jsonl}")
        logger.info(f"Saving to directory: {args.save_dir}")
        if args.sample:
            logger.info(f"Using sample size: {args.sample}")

        # Ensure save directory exists
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create vector database
        index, metadata_df, embeddings = create_vector_db_from_jsonl(
            client=client,
            input_jsonl=args.input_jsonl,
            save_dir=str(save_dir),
            text_field="text",
            batch_size=32,
            sample_size=args.sample
        )
        logger.info("Vector database creation completed successfully")

    # Then handle search if requested
    if args.query and args.db_path:
        # Load the vector database
        if not client.load_vector_database(args.db_path):
            logger.error("Failed to load vector database. Please ensure it exists and is properly formatted.")
            return

        # Perform search
        demo_search(client, args.query, args.results)

    # If no specific operation requested, show demo
    if not (args.input_jsonl or args.query):
        demo_embedding(client)
        print("\nUsage examples:")
        print(main.__doc__)


if __name__ == "__main__":
    main()