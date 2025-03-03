#!/usr/bin/env python3
"""
Vector Database Management for Content Moderation

This module manages the vector database for content moderation,
including creation and searching.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm.auto import tqdm
from openai import OpenAI
import logging
from typing import List, Optional, Dict, Any, Union, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class VectorDB:
    """Manager for vector database"""

    def __init__(
        self,
        base_url: str = "http://localhost:8890/v1",
        api_key: str = "None",
        model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        vector_db_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize vector database manager

        Args:
            base_url: Base URL for embedding server
            api_key: API key for authentication
            model_name: Embedding model name
            vector_db_path: Path to vector database directory
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # Vector DB state
        self.index = None
        self.metadata_df = None

        # Load vector database if path provided
        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def load_vector_database(self, db_path: Union[str, Path]) -> bool:
        """
        Load a FAISS vector database and its metadata

        Args:
            db_path: Path to vector database directory

        Returns:
            True if successful, False otherwise
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
            response = self.client.embeddings.create(model=self.model_name, input=text)

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def batch_create_embeddings(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
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
            batch = texts[i : i + batch_size]
            batch_embeddings = self.create_embedding(batch)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def build_vector_database(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        index_type: str = "L2",
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
        """
        Build a vector database from texts

        Args:
            texts: List of text strings to include
            metadata: Optional list of metadata dictionaries
            batch_size: Batch size for embedding creation
            index_type: Type of FAISS index ("L2" or "IP")
            save_dir: Directory to save the database

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

        # Update instance state
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

    def similarity_search(
        self, query: str, k: int = 5, return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query text

        Args:
            query: Query text to search for
            k: Number of results to return
            return_scores: Whether to include similarity scores

        Returns:
            List of dictionaries containing search results
        """
        if self.index is None or self.metadata_df is None:
            raise ValueError(
                "Vector database not loaded. Call load_vector_database first."
            )

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
                "category": self.metadata_df.iloc[idx].get("moderation_category", None),
            }

            if return_scores:
                result["distance"] = float(dist)

            results.append(result)

        return results

    def create_vector_db_from_jsonl(
        self,
        input_jsonl: Union[str, Path],
        save_dir: Union[str, Path],
        text_field: str = "text",
        batch_size: int = 32,
        sample_size: Optional[int] = None,
        prune_text_to_max_chars: Optional[int] = None,
    ) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
        """
        Create a vector database from a JSONL file

        Args:
            input_jsonl: Path to input JSONL file
            save_dir: Directory to save the vector database
            text_field: Name of the field containing text
            batch_size: Batch size for embedding creation
            sample_size: Number of records to sample

        Returns:
            Tuple of (index, metadata_df, embeddings)
        """
        try:
            # Load data
            df = pd.read_json(input_jsonl, lines=True)
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)))
            if prune_text_to_max_chars:
                df[text_field] = df[text_field].str[:prune_text_to_max_chars]

            logger.info(f"Loaded {len(df)} records from {input_jsonl}")

            # Extract texts as strings
            # Make sure we're extracting just the text field string values, not the entire record
            texts = []
            metadata = []

            for _, row in df.iterrows():
                # Get the text value or skip if not present
                if text_field in row and isinstance(row[text_field], str):
                    texts.append(row[text_field])
                    metadata.append(row.to_dict())
                else:
                    # If the record is already a simple string
                    if isinstance(row, str):
                        texts.append(row)
                        metadata.append({text_field: row})
                    else:
                        logger.warning(
                            f"Skipping record: missing or invalid '{text_field}' field"
                        )

            if not texts:
                raise ValueError(
                    f"No valid text entries found with field '{text_field}'"
                )

            logger.info(f"Extracted {len(texts)} valid text entries for embedding")

            # Create vector database
            return self.build_vector_database(
                texts=texts, metadata=metadata, batch_size=batch_size, save_dir=save_dir
            )
        except Exception as e:
            logger.error(f"Error in create_vector_db_from_jsonl: {e}")
            raise
