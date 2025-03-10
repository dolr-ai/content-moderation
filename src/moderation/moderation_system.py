"""
Content Moderation System

This module provides a content moderation system that uses RAG (Retrieval Augmented Generation)
to classify text content.
"""

import os
import yaml
import jinja2
import numpy as np
import pandas as pd
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from openai import OpenAI
import aiohttp
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Define valid categories
VALID_CATEGORIES = {
    "hate_or_discrimination",
    "violence_or_threats",
    "offensive_language",
    "nsfw_content",
    "spam_or_scams",
    "clean",
}


@dataclass
class RAGEx:
    """RAGEx: RAG+Examples
    Class to store RAG search results with their metadata"""

    text: str
    category: str
    distance: float


class ModerationSystem:
    """Content moderation system using RAG and LLM"""

    def __init__(
        self,
        embedding_url: str = "http://localhost:8890/v1",
        llm_url: str = "http://localhost:8899/v1",
        api_key: str = "None",
        embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        llm_model: str = "microsoft/Phi-3.5-mini-instruct",
        vector_db_path: Optional[Union[str, Path]] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ):
        """
        Initialize the content moderation system

        Args:
            embedding_url: URL for embedding server
            llm_url: URL for LLM server
            api_key: API key for authentication
            embedding_model: Model for embeddings
            llm_model: Model for LLM
            vector_db_path: Path to vector database
            prompt_path: Path to prompt file
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens for LLM response
        """
        # Store URLs as instance attributes
        self.embedding_url = embedding_url
        self.llm_url = llm_url
        self.api_key = api_key

        # Initialize clients
        self.embedding_client = OpenAI(base_url=embedding_url, api_key=api_key)
        self.llm_client = OpenAI(base_url=llm_url, api_key=api_key)

        # Create shared HTTP client session for better connection pooling
        self.http_session = None
        self.http_session_lock = asyncio.Lock()

        # Model settings
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Vector DB state
        self.index = None
        self.metadata_df = None

        # Load prompts
        self.prompt_path = (
            prompt_path
            or Path(__file__).parent.parent.parent
            / "prompts"
            / "moderation_prompts.yml"
        )
        self.prompts = self._load_prompt(self.prompt_path)
        logger.info(f"Loaded prompt from: {self.prompt_path}")

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment()

        # Load vector database if path provided
        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def _load_prompt(self, prompt_path: Union[str, Path]) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        try:
            with open(prompt_path, "r") as f:
                prompts = yaml.safe_load(f)
                return prompts
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            # Fallback to default prompts
            return {
                "system_prompt": (
                    "You are a content moderation expert. Your task is to analyze content "
                    "and categorize it into one of the following categories:\n\n"
                    "1. hate_or_discrimination: Content targeting protected characteristics with negative intent/prejudice\n"
                    "2. violence_or_threats: Content that threatens, depicts, or promotes violence\n"
                    "3. offensive_language: Hostile or inappropriate content WITHOUT targeting protected characteristics\n"
                    "4. nsfw_content: Explicit sexual content or material intended to arouse\n"
                    "5. spam_or_scams: Deceptive or unsolicited content designed to mislead\n"
                    "6. clean: Content that is allowed and doesn't fall into above categories\n\n"
                    "Please format your response exactly as:\n"
                    "Category: [exact category_name]\n"
                    "Confidence: [HIGH/MEDIUM/LOW]\n"
                    "Explanation: [short 1/2 line explanation]"
                ),
                "rag_prompt": (
                    "Here are some example classifications:\n\n"
                    "{% for example in examples %}"
                    "Text: {{ example.text }}\n"
                    "Category: {{ example.category }}\n\n"
                    "{% endfor %}"
                    "Now, please classify this text:\n"
                    "{{ query }}"
                ),
            }

    def load_vector_database(self, db_path: Union[str, Path]) -> bool:
        """
        Load the FAISS vector database and metadata

        Args:
            db_path: Path to vector database directory

        Returns:
            True if successful, False otherwise
        """
        db_path = Path(db_path)

        try:
            # Import faiss here to avoid dependency issues
            import faiss

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
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    def create_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for input text

        Args:
            text: Single text string or list of text strings

        Returns:
            numpy.ndarray: Generated embeddings
        """
        if isinstance(text, str):
            text = [text]

        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model, input=text
            )

            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[RAGEx]:
        """
        Perform similarity search and return structured results

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of RAGEx objects with similar texts and metadata
        """
        if self.index is None or self.metadata_df is None:
            raise ValueError(
                "Vector database not loaded. Call load_vector_database first."
            )

        # Create query embedding and search
        query_embedding = self.create_embedding(query)
        D, I = self.index.search(query_embedding.astype("float32"), k)

        # Format results
        results = []
        for dist, idx in zip(D[0], I[0]):
            example = RAGEx(
                text=self.metadata_df.iloc[idx]["text"],
                category=self.metadata_df.iloc[idx].get(
                    "moderation_category", "unknown"
                ),
                distance=float(dist),
            )
            results.append(example)

        return results

    def create_prompt_with_examples(
        self,
        query: str,
        examples: List[RAGEx],
        num_examples: int = 3,
        max_text_length: int = 2000,
    ) -> str:
        """
        Create a prompt with similar examples for few-shot learning

        Args:
            query: Text to classify
            examples: List of RAGEx objects with similar texts
            num_examples: Maximum number of examples to include
            max_text_length: Maximum length for text examples

        Returns:
            Formatted prompt with examples
        """
        # Sort examples by distance and take top k
        sorted_examples = sorted(examples, key=lambda x: x.distance)[:num_examples]

        # Truncate text to max_text_length
        truncated_examples = []
        for ex in sorted_examples:
            truncated_examples.append(
                RAGEx(
                    text=ex.text[:max_text_length],
                    category=ex.category,
                    distance=ex.distance,
                )
            )

        # Render prompt template
        template = self.jinja_env.from_string(self.prompts.get("rag_prompt", ""))
        prompt = template.render(
            examples=truncated_examples, query=query[:max_text_length]
        )

        return prompt

    def extract_category(self, model_response: str) -> str:
        """
        Parse the model response to extract category

        Args:
            model_response: Raw response from LLM

        Returns:
            Extracted category
        """
        try:
            category_match = re.search(
                r"Category:\s*(\w+(?:_?\w+)*)", model_response, re.IGNORECASE
            )
            if category_match:
                category = category_match.group(1).lower()
                return category if category in VALID_CATEGORIES else "clean"
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return "error_parsing"
        return "no_category_found"

    def classify_text(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Classify text using RAG-enhanced LLM

        Args:
            query: Text to classify
            num_examples: Number of similar examples to use

        Returns:
            Dictionary with classification results
        """
        # Get similar examples using RAG
        similar_examples = self.similarity_search(
            query[:max_input_length], k=num_examples
        )

        # Create prompt with examples
        user_prompt = self.create_prompt_with_examples(
            query[:max_input_length], similar_examples
        )

        try:
            # Use llm_client for chat completions with system prompt
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompts.get("system_prompt", ""),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract and validate category from response
            raw_response = response.choices[0].message.content.strip()
            category = self.extract_category(raw_response)

            return {
                "query": query,
                "category": category,
                "raw_response": raw_response,
                "similar_examples": [
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in similar_examples
                ],
                "prompt": user_prompt,
            }

        except Exception as e:
            logger.error(f"Error in LLM inference: {str(e)}")
            return {
                "query": query,
                "category": "error",
                "error_message": str(e),
                "similar_examples": [
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in similar_examples
                ],
                "prompt": user_prompt,
            }

    async def similarity_search_async(self, query: str, k: int = 5) -> List[RAGEx]:
        """
        Async version of similarity search using vector database

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of RAGEx objects with similar examples
        """
        try:
            # Get shared HTTP session
            session = await self.get_http_session()

            # Create embedding for query
            async with session.post(
                f"{self.embedding_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.embedding_model,
                    "input": query,
                },
                timeout=aiohttp.ClientTimeout(total=30),  # Add timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in embedding API: {error_text}")
                    return []

                data = await response.json()
                query_embedding = np.array(data["data"][0]["embedding"])

            # Perform vector search asynchronously
            if not self.index:
                logger.error("Vector database not loaded")
                return []

            # Run vector search in a separate thread to avoid blocking
            results = await asyncio.to_thread(
                self.index.search, query_embedding.astype("float32").reshape(1, -1), k
            )

            # Process results
            search_results = []

            # Check if results is already a tuple of (D, I)
            if isinstance(results, tuple) and len(results) == 2:
                D, I = results
            else:
                # Log the actual type and shape of results for debugging
                logger.warning(f"Unexpected results format: {type(results)}")
                if hasattr(results, 'shape'):
                    logger.warning(f"Results shape: {results.shape}")

                # If it's a single array, we need to create indices
                if isinstance(results, np.ndarray):
                    D = results
                    # Create sequential indices
                    I = np.array([[i for i in range(min(k, D.shape[1]))]])
                else:
                    # If we can't determine the format, return empty results
                    logger.error("Cannot process search results in unknown format")
                    return []

            # Ensure we have valid arrays to work with
            if len(D) > 0 and hasattr(D, 'shape') and D.shape[0] > 0 and D.shape[1] > 0:
                for i in range(min(k, D.shape[1])):
                    try:
                        dist = float(D[0][i])
                        idx = int(I[0][i])
                        if 0 <= idx < len(self.metadata_df):
                            item = RAGEx(
                                text=self.metadata_df.iloc[idx]["text"],
                                category=self.metadata_df.iloc[idx].get("moderation_category", "unknown"),
                                distance=dist,
                            )
                            search_results.append(item)
                        else:
                            logger.warning(f"Index {idx} out of bounds for metadata_df with length {len(self.metadata_df)}")
                    except Exception as e:
                        logger.warning(f"Error processing result at index {i}: {e}")

            return search_results

        except Exception as e:
            logger.error(
                f"Error in async similarity search: {type(e).__name__}: {str(e)}"
            )
            return []

    async def classify_text_async(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Async version of classify_text using RAG-enhanced LLM

        Args:
            query: Text to classify
            num_examples: Number of similar examples to use
            max_input_length: Maximum input length

        Returns:
            Dictionary with classification results
        """
        try:
            # Get similar examples using RAG
            similar_examples = await self.similarity_search_async(
                query[:max_input_length], k=num_examples
            )

            if not similar_examples:
                logger.warning(f"No similar examples found for query: {query[:100]}...")

            # Create prompt with examples
            user_prompt = self.create_prompt_with_examples(
                query[:max_input_length], similar_examples
            )

            # Get shared HTTP session
            session = await self.get_http_session()

            # Use async HTTP client for LLM API with timeout
            try:
                logger.info(f"Sending request to LLM API at {self.llm_url}")
                async with session.post(
                    f"{self.llm_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.prompts.get("system_prompt", ""),
                            },
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),  # 60 seconds timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Error in LLM API (status {response.status}): {error_text}"
                        )
                        raise Exception(
                            f"LLM API error (status {response.status}): {error_text}"
                        )

                    data = await response.json()
                    if "choices" not in data or not data["choices"]:
                        logger.error(f"Invalid response from LLM API: {data}")
                        raise Exception(
                            "Invalid response from LLM API (no choices in response)"
                        )

                    raw_response = data["choices"][0]["message"]["content"].strip()
                    logger.debug(f"Raw LLM response: {raw_response}")

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error to LLM API at {self.llm_url}: {str(e)}")
                raise Exception(
                    f"Could not connect to LLM API at {self.llm_url}: {str(e)}"
                )
            except aiohttp.ClientError as e:
                logger.error(
                    f"Client error in LLM API request: {type(e).__name__}: {str(e)}"
                )
                raise Exception(f"LLM API client error: {type(e).__name__}: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout error in LLM API request after 60 seconds")
                raise Exception(f"LLM API request timed out after 60 seconds")
            except Exception as e:
                logger.error(
                    f"Unexpected error in LLM API request: {type(e).__name__}: {str(e)}"
                )
                raise Exception(
                    f"Unexpected error in LLM API request: {type(e).__name__}: {str(e)}"
                )

            # Extract and validate category from response
            try:
                category = self.extract_category(raw_response)
                if not category:
                    logger.warning(
                        f"Could not extract category from response: {raw_response}"
                    )
                    category = "clean"  # Default to clean if extraction fails
            except Exception as e:
                logger.error(f"Error extracting category: {type(e).__name__}: {str(e)}")
                category = "clean"  # Default to clean if extraction fails

            return {
                "query": query,
                "category": category,
                "raw_response": raw_response,
                "similar_examples": [
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in similar_examples
                ],
                "prompt": user_prompt,
            }
        except Exception as e:
            logger.error(f"Error in classification: {type(e).__name__}: {str(e)}")
            raise Exception(f"Classification error: {type(e).__name__}: {str(e)}")

    async def get_http_session(self):
        """
        Get or create a shared HTTP session for better connection pooling

        Returns:
            aiohttp.ClientSession instance
        """
        async with self.http_session_lock:
            if self.http_session is None or self.http_session.closed:
                # Configure session with proper connection limits to handle concurrent requests
                connector = aiohttp.TCPConnector(
                    limit=100,  # Connection pool size
                    limit_per_host=20,  # Connections per host
                    keepalive_timeout=60,  # Keep connections alive for 60 seconds
                )
                timeout = aiohttp.ClientTimeout(total=90)  # 90 seconds timeout
                self.http_session = aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                )
            return self.http_session

    async def close(self):
        """
        Clean up resources when shutting down
        """
        if self.http_session is not None and not self.http_session.closed:
            logger.info("Closing HTTP session")
            await self.http_session.close()
