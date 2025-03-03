#!/usr/bin/env python3
"""
Content Moderation System for text classification.

This module provides functionality to classify text content into moderation categories
using LLM inference with RAG enhancement.
"""
import os
import sys
import re
import json
import logging
import numpy as np
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
import time
from datetime import datetime

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
    from src.prompts.templates import prompt_manager
    from src.vector_db.vector_database import load_vector_database, RAGExample
else:
    # Running from within the package
    from ..config.config import config
    from ..prompts.templates import prompt_manager
    from ..vector_db.vector_database import load_vector_database, RAGExample


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Valid categories for content moderation
VALID_CATEGORIES = {
    "hate_or_discrimination",
    "violence_or_threats",
    "offensive_language",
    "nsfw_content",
    "spam_or_scams",
    "clean",
}


def extract_category(model_response: str) -> Dict[str, Any]:
    """
    Extract the category, confidence, and explanation from the model response.

    Args:
        model_response: Raw response from the model

    Returns:
        Dictionary with category, confidence, and explanation
    """
    # Default values
    result = {
        "category": "unknown",
        "confidence": "LOW",
        "explanation": "Failed to parse model response",
    }

    # Extract category
    category_match = re.search(
        r"Category:\s*(\w+(?:_\w+)*)", model_response, re.IGNORECASE
    )
    if category_match:
        category = category_match.group(1).lower()
        if category in VALID_CATEGORIES:
            result["category"] = category

    # Extract confidence
    confidence_match = re.search(
        r"Confidence:\s*(HIGH|MEDIUM|LOW)", model_response, re.IGNORECASE
    )
    if confidence_match:
        result["confidence"] = confidence_match.group(1).upper()

    # Extract explanation
    explanation_match = re.search(
        r"Explanation:\s*(.+?)(?:\n|$)", model_response, re.IGNORECASE
    )
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    return result


class ContentModerator:
    """Content moderation system using LLM with RAG enhancement."""

    def __init__(
        self,
        embedding_url: str = "http://localhost:8890/v1",
        llm_url: str = "http://localhost:8899/v1",
        api_key: Optional[str] = None,
        embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        llm_model: str = "microsoft/Phi-3.5-mini-instruct",
        vector_db_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ):
        """
        Initialize the content moderation system.

        Args:
            embedding_url: URL of the embedding server
            llm_url: URL of the LLM server
            api_key: API key for the servers
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model
            vector_db_path: Path to the vector database
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.embedding_url = embedding_url
        self.llm_url = llm_url
        self.api_key = api_key or "None"
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load vector database if path is provided
        self.vector_db = None
        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def load_vector_database(self, db_path: Union[str, Path]) -> bool:
        """
        Load the vector database.

        Args:
            db_path: Path to the vector database

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_db = load_vector_database(db_path)
            if self.vector_db:
                logger.info(
                    f"Loaded vector database with {len(self.vector_db.documents)} documents"
                )
                return True
            else:
                logger.error(f"Failed to load vector database from {db_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    async def create_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for text using the embedding server.

        Args:
            text: Text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.embedding_url}/embeddings", headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Error from embedding server: {response.status} - {error_text}"
                    )

                result = await response.json()
                embeddings = [item["embedding"] for item in result["data"]]

                return np.array(embeddings)

    async def similarity_search(self, query: str, k: int = 5) -> List[RAGExample]:
        """
        Search for similar examples in the vector database.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of similar examples
        """
        if not self.vector_db:
            logger.warning(
                "Vector database not loaded, cannot perform similarity search"
            )
            return []

        # Create embedding for the query
        query_embedding = await self.create_embedding(query)

        # Search the vector database
        results = self.vector_db.search(query_embedding, k=k)

        return results

    def create_prompt_with_examples(
        self, query: str, examples: List[RAGExample], max_examples: int = 3
    ) -> str:
        """
        Create a prompt with RAG examples.

        Args:
            query: Query text
            examples: List of similar examples
            max_examples: Maximum number of examples to include

        Returns:
            Complete prompt with examples
        """
        # Limit the number of examples
        examples = examples[:max_examples]

        # Create the prompt using the template manager
        prompt = prompt_manager.get_rag_prompt(query, examples)

        return prompt

    async def classify_text(
        self, query: str, num_examples: int = 3, use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Classify text content into moderation categories.

        Args:
            query: Text to classify
            num_examples: Number of examples to include in the prompt
            use_rag: Whether to use RAG enhancement

        Returns:
            Classification result with category, confidence, and explanation
        """
        start_time = time.time()
        result = {
            "text": query,
            "category": "unknown",
            "confidence": "LOW",
            "explanation": "Failed to classify",
            "processing_time": 0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get similar examples if using RAG
            examples = []
            if use_rag and self.vector_db:
                examples = await self.similarity_search(query, k=num_examples)
                logger.debug(f"Found {len(examples)} similar examples")

            # Create the prompt
            if use_rag and examples:
                prompt = self.create_prompt_with_examples(query, examples, num_examples)
            else:
                prompt = prompt_manager.get_moderation_prompt(query)

            # Call the LLM
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )

            payload = {
                "model": self.llm_model,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.llm_url}/chat/completions", headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Error from LLM server: {response.status} - {error_text}"
                        )

                    llm_result = await response.json()
                    model_response = llm_result["choices"][0]["message"]["content"]

            # Extract the category, confidence, and explanation
            classification = extract_category(model_response)

            # Update the result
            result.update(classification)

            # Add RAG information if used
            if use_rag and examples:
                result["rag_examples"] = [ex.to_dict() for ex in examples]

            # Add processing time
            result["processing_time"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            result["explanation"] = f"Error: {str(e)}"
            result["processing_time"] = time.time() - start_time
            return result

    async def classify_batch(
        self, texts: List[str], num_examples: int = 3, use_rag: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of texts.

        Args:
            texts: List of texts to classify
            num_examples: Number of examples to include in the prompt
            use_rag: Whether to use RAG enhancement

        Returns:
            List of classification results
        """
        tasks = [self.classify_text(text, num_examples, use_rag) for text in texts]
        return await asyncio.gather(*tasks)


async def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Content moderation system for text classification"
    )
    parser.add_argument(
        "text", nargs="?", help="Text to classify (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default=None,
        help="Path to the vector database",
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8890/v1",
        help="URL of the embedding server",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8899/v1",
        help="URL of the LLM server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the servers",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG enhancement",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to include in the prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for the result (JSON format)",
    )

    args = parser.parse_args()

    # Get text from stdin if not provided as argument
    if args.text is None:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        args.text = sys.stdin.read().strip()

    # Use API key from config if not provided
    if args.api_key is None:
        args.api_key = config.get_hf_token()

    # Set default vector database path if not provided
    if args.vector_db is None:
        data_root = config.get_data_root()
        default_db_path = os.path.join(data_root, "vector_db")
        if os.path.exists(default_db_path):
            args.vector_db = default_db_path

    # Create the moderator
    moderator = ContentModerator(
        embedding_url=args.embedding_url,
        llm_url=args.llm_url,
        api_key=args.api_key,
        vector_db_path=args.vector_db,
    )

    # Classify the text
    result = await moderator.classify_text(
        args.text, args.num_examples, not args.no_rag
    )

    # Output the result
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
