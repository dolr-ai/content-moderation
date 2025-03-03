#!/usr/bin/env python3
"""
Vector Database module for content moderation system.

This module provides functionality to create, load, and query a vector database
for similarity search in the content moderation system.
"""
import os
import sys
import json
import numpy as np
import faiss
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass, asdict

# Add path handling for imports
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Use relative or absolute imports based on how the script is being run
if __name__ == "__main__" or "src" not in __name__:
    # Running as script or from outside the package
    from src.config.config import config
else:
    # Running from within the package
    from ..config.config import config


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RAGExample:
    """
    RAG Example class to store document with its metadata.

    Attributes:
        text: The text content of the example
        category: The moderation category of the example
        distance: The distance from the query (lower is more similar)
        metadata: Additional metadata for the example
    """

    text: str
    category: str
    distance: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class VectorDatabase:
    """Vector database for similarity search in content moderation."""

    def __init__(self, dimension: int = 1024):
        """
        Initialize the vector database.

        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.categories = []
        self.metadata = []

    def create_index(self) -> None:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created new FAISS index with dimension {self.dimension}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        categories: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector database.

        Args:
            texts: List of document texts
            embeddings: Matrix of document embeddings
            categories: List of document categories
            metadata: List of document metadata (optional)
        """
        if self.index is None:
            self.create_index()

        # Ensure embeddings are in the right format
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))

        # Store document texts and categories
        self.documents.extend(texts)
        self.categories.extend(categories)

        # Store metadata if provided
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

        logger.info(f"Added {len(texts)} documents to the vector database")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RAGExample]:
        """
        Search for similar documents.

        Args:
            query_embedding: Embedding of the query
            k: Number of results to return

        Returns:
            List of RAGExample objects
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector database is empty, cannot search")
            return []

        # Ensure query embedding is in the right format
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search the index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), min(k, len(self.documents))
        )

        # Create result objects
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                result = RAGExample(
                    text=self.documents[idx],
                    category=self.categories[idx],
                    distance=float(distances[0][i]),
                    metadata=self.metadata[idx],
                )
                results.append(result)

        return results

    def save(self, path: Union[str, Path]) -> bool:
        """
        Save the vector database to disk.

        Args:
            path: Directory path to save the database

        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.warning("No index to save")
            return False

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        try:
            # Save the FAISS index
            faiss.write_index(self.index, str(path / "index.faiss"))

            # Save the documents and metadata
            with open(path / "documents.json", "w") as f:
                json.dump(
                    {
                        "dimension": self.dimension,
                        "documents": self.documents,
                        "categories": self.categories,
                        "metadata": self.metadata,
                    },
                    f,
                )

            logger.info(f"Saved vector database to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            return False

    def load(self, path: Union[str, Path]) -> bool:
        """
        Load the vector database from disk.

        Args:
            path: Directory path to load the database from

        Returns:
            True if successful, False otherwise
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"Database path does not exist: {path}")
            return False

        try:
            # Load the documents and metadata
            with open(path / "documents.json", "r") as f:
                data = json.load(f)
                self.dimension = data["dimension"]
                self.documents = data["documents"]
                self.categories = data["categories"]
                self.metadata = data["metadata"]

            # Load the FAISS index
            self.index = faiss.read_index(str(path / "index.faiss"))

            logger.info(
                f"Loaded vector database from {path} "
                f"({len(self.documents)} documents)"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False


def create_vector_database(
    texts: List[str],
    embeddings: np.ndarray,
    categories: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> VectorDatabase:
    """
    Create a new vector database from documents and embeddings.

    Args:
        texts: List of document texts
        embeddings: Matrix of document embeddings
        categories: List of document categories
        metadata: List of document metadata (optional)
        output_path: Path to save the database (optional)

    Returns:
        The created VectorDatabase object
    """
    # Determine the dimension from the embeddings
    if len(embeddings.shape) == 1:
        dimension = embeddings.shape[0]
    else:
        dimension = embeddings.shape[1]

    # Create the database
    db = VectorDatabase(dimension=dimension)
    db.add_documents(texts, embeddings, categories, metadata)

    # Save if path is provided
    if output_path:
        db.save(output_path)

    return db


def load_vector_database(path: Union[str, Path]) -> Optional[VectorDatabase]:
    """
    Load a vector database from disk.

    Args:
        path: Path to the database directory

    Returns:
        The loaded VectorDatabase object, or None if loading failed
    """
    db = VectorDatabase()
    if db.load(path):
        return db
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector database management for content moderation"
    )
    parser.add_argument("--load", type=str, help="Path to load the database from")
    parser.add_argument(
        "--info", action="store_true", help="Print database information"
    )

    args = parser.parse_args()

    if args.load:
        db = load_vector_database(args.load)
        if db and args.info:
            print(f"Database dimension: {db.dimension}")
            print(f"Number of documents: {len(db.documents)}")
            print(f"Categories: {set(db.categories)}")
    else:
        print("No action specified. Use --load to load a database.")
