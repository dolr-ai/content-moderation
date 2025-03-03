#!/usr/bin/env python3
"""
Setup script for creating and populating the vector database.

This script loads training data, generates embeddings, and creates a vector database
for similarity search in the content moderation system.
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import aiohttp
import asyncio
from tqdm import tqdm

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config.config import config
from src.vector_db.vector_database import create_vector_database, VectorDatabase


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_embeddings_async(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    texts: List[str],
) -> List[List[float]]:
    """
    Create embeddings for texts using the embedding server.

    Args:
        session: aiohttp client session
        base_url: Base URL of the embedding server
        api_key: API key for the server
        model: Model name
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    payload = {
        "model": model,
        "input": texts,
    }

    async with session.post(
        f"{base_url}/embeddings", headers=headers, json=payload
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(
                f"Error from embedding server: {response.status} - {error_text}"
            )

        result = await response.json()
        return [item["embedding"] for item in result["data"]]


async def batch_create_embeddings_async(
    texts: List[str],
    embedding_url: str,
    api_key: Optional[str] = None,
    model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Create embeddings for a list of texts in batches.

    Args:
        texts: List of texts to embed
        embedding_url: URL of the embedding server
        api_key: API key for the server
        model: Model name
        batch_size: Batch size for embedding creation

    Returns:
        Array of embeddings
    """
    all_embeddings = []

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i : i + batch_size]

            try:
                batch_embeddings = await create_embeddings_async(
                    session, embedding_url, api_key, model, batch_texts
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i}: {e}")
                # Fill with zeros for failed batches
                for _ in range(len(batch_texts)):
                    all_embeddings.append([0.0] * 1024)  # Default dimension

    return np.array(all_embeddings)


def load_training_data(
    file_path: Union[str, Path], format: str = "csv"
) -> pd.DataFrame:
    """
    Load training data from a file.

    Args:
        file_path: Path to the training data file
        format: File format (csv, json, etc.)

    Returns:
        DataFrame containing the training data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")

    if format.lower() == "csv":
        return pd.read_csv(file_path)
    elif format.lower() == "json":
        return pd.read_json(file_path)
    elif format.lower() == "jsonl":
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {format}")


async def setup_vector_database(
    training_data_path: Union[str, Path],
    output_path: Union[str, Path],
    embedding_url: str = "http://localhost:8890/v1",
    api_key: Optional[str] = None,
    text_column: str = "text",
    category_column: str = "category",
    batch_size: int = 32,
    file_format: str = "csv",
) -> VectorDatabase:
    """
    Set up the vector database from training data.

    Args:
        training_data_path: Path to the training data file
        output_path: Path to save the vector database
        embedding_url: URL of the embedding server
        api_key: API key for the server
        text_column: Name of the column containing text data
        category_column: Name of the column containing category data
        batch_size: Batch size for embedding creation
        file_format: Format of the training data file

    Returns:
        The created vector database
    """
    # Load training data
    logger.info(f"Loading training data from {training_data_path}")
    df = load_training_data(training_data_path, format=file_format)

    # Extract texts and categories
    texts = df[text_column].tolist()
    categories = df[category_column].tolist()

    # Create metadata from other columns
    metadata = []
    for _, row in df.iterrows():
        meta = {
            col: row[col]
            for col in df.columns
            if col not in [text_column, category_column]
        }
        metadata.append(meta)

    # Create embeddings
    logger.info(f"Creating embeddings for {len(texts)} documents")
    embeddings = await batch_create_embeddings_async(
        texts, embedding_url, api_key, batch_size=batch_size
    )

    # Create and save the vector database
    logger.info(f"Creating vector database at {output_path}")
    db = create_vector_database(texts, embeddings, categories, metadata, output_path)

    logger.info(f"Vector database setup complete: {len(texts)} documents indexed")
    return db


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up the vector database for content moderation"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to the training data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the vector database",
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8890/v1",
        help="URL of the embedding server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the server",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing text data",
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="category",
        help="Name of the column containing category data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding creation",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        default="csv",
        choices=["csv", "json", "jsonl"],
        help="Format of the training data file",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_path is None:
        data_root = config.get_data_root()
        args.output_path = os.path.join(data_root, "vector_db")

    # Use API key from config if not provided
    if args.api_key is None:
        args.api_key = config.get_hf_token()

    # Set up the vector database
    await setup_vector_database(
        args.training_data,
        args.output_path,
        args.embedding_url,
        args.api_key,
        args.text_column,
        args.category_column,
        args.batch_size,
        args.file_format,
    )


if __name__ == "__main__":
    asyncio.run(main())
