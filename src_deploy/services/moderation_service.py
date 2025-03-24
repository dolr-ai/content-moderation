"""
Moderation service for content classification
"""

import logging
import yaml
import jinja2
import re
import pandas as pd
import io
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import aiohttp
import json
import numpy as np
from openai import OpenAI
import asyncio

from utils.gcp_utils import GCPUtils
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


class ModerationService:
    """Content moderation service using BigQuery for RAG search"""

    def __init__(
        self,
        gcp_credentials: Optional[str] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        bucket_name: Optional[str] = None,
        gcs_prompt_path: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
    ):
        """
        Initialize the content moderation service

        Args:
            gcp_credentials: GCP credentials JSON as a string
            prompt_path: Path to prompts file (local file path)
            bucket_name: GCS bucket name for prompts file
            gcs_prompt_path: Path to prompts file in GCS
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        # Initialize GCP utils
        self.gcp_utils = GCPUtils(
            gcp_credentials=gcp_credentials,
            bucket_name=bucket_name,
            dataset_id=dataset_id,
            table_id=table_id,
        )

        # Store GCS paths
        self.bucket_name = bucket_name
        self.gcs_prompt_path = gcs_prompt_path or "rag/moderation_prompts.yml"

        # Load prompts
        self.prompt_path = prompt_path or config.prompt_path
        self.prompts = self._load_prompts()
        logger.info(f"Loaded prompts successfully")

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment()

        # Initialize clients
        self.embedding_url = config.embedding_url
        self.llm_url = config.llm_url
        self.api_key = config.api_key

        # Configure model names
        self.embedding_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        self.llm_model = "microsoft/Phi-3.5-mini-instruct"

        # Initialize OpenAI clients
        self.embedding_client = None
        self.llm_client = None

        # Initialize if URLs are configured
        if self.embedding_url:
            self.embedding_client = OpenAI(
                base_url=self.embedding_url, api_key=self.api_key or "None"
            )
            logger.info(f"Initialized embedding client with URL {self.embedding_url}")

        if self.llm_url:
            self.llm_client = OpenAI(
                base_url=self.llm_url, api_key=self.api_key or "None"
            )
            logger.info(f"Initialized LLM client with URL {self.llm_url}")

        # Create HTTP session for async calls
        self.http_session = None
        self.http_session_lock = asyncio.Lock()

    def _load_prompts(self) -> Dict[str, Any]:
        """
        Load prompts from YAML file (local or GCS)
        Returns:
            Dictionary with prompt templates
        """
        try:
            # Try loading from GCS first if bucket is configured
            if self.bucket_name and self.gcp_utils.storage_client:
                logger.info(
                    f"Loading prompts from GCS: gs://{self.bucket_name}/{self.gcs_prompt_path}"
                )
                try:
                    yaml_content = self.gcp_utils.download_file_from_gcs(
                        gcs_path=self.gcs_prompt_path, as_string=True
                    )
                    prompts = yaml.safe_load(yaml_content)
                    logger.info(f"Loaded prompts from GCS")
                    return prompts
                except Exception as gcs_error:
                    logger.warning(f"Failed to load prompts from GCS: {gcs_error}")
                    logger.info(f"Falling back to local prompt file")

            # Try loading from local file
            with open(self.prompt_path, "r") as f:
                prompts = yaml.safe_load(f)
                logger.info(f"Loaded prompts from local file: {self.prompt_path}")
                return prompts

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")

        # Fallback to default prompts
        logger.warning("Using default hardcoded prompts")
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

    def similarity_search(self, embedding: List[float], top_k: int = 5) -> List[RAGEx]:
        """
        Perform similarity search using BigQuery
        Args:
            embedding: Embedding vector to search
            top_k: Number of results to return
        Returns:
            List of RAGEx objects with similar examples
        """
        try:
            results_df = self.gcp_utils.bigquery_vector_search(
                embedding=embedding, top_k=top_k
            )

            # Convert to RAGEx objects
            results = []
            for _, row in results_df.iterrows():
                example = RAGEx(
                    text=row["text"],
                    category=row.get("moderation_category", "unknown"),
                    distance=float(row["distance"]),
                )
                results.append(example)
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

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

    def create_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for input text using the configured embedding API

        Args:
            text: Single text string or list of text strings

        Returns:
            numpy.ndarray: Generated embeddings
        """
        if isinstance(text, str):
            text = [text]

        if not self.embedding_client:
            logger.error(
                "Embedding client not initialized. Check EMBEDDING_URL environment variable."
            )
            raise ValueError("Embedding client not initialized")

        try:
            logger.info(f"Creating embedding for text of length {len(text[0][:50])}...")
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model, input=text
            )

            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def call_llm(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 128
    ) -> str:
        """
        Call the LLM API to classify text

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with examples and query
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response
        """
        if not self.llm_client:
            logger.error(
                "LLM client not initialized. Check LLM_URL environment variable."
            )
            raise ValueError("LLM client not initialized")

        try:
            logger.info(f"Calling LLM with prompt of length {len(user_prompt)}")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Got LLM response of length {len(raw_response)}")
            return raw_response

        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            raise

    def classify_text(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Classify text using RAG examples

        Args:
            query: Text to classify
            num_examples: Number of similar examples to use
            max_input_length: Maximum input length

        Returns:
            Dictionary with classification results
        """
        try:
            # Create embedding for the query
            if not self.embedding_client:
                logger.error("Embedding client required for text classification")
                raise ValueError("Embedding client not initialized")

            # Create embedding for the query
            embedding = self.create_embedding(query[:max_input_length])[0].tolist()

            # Get similar examples using BigQuery
            similar_examples = self.similarity_search(
                embedding=embedding, top_k=num_examples
            )

            # Create prompt with examples
            user_prompt = self.create_prompt_with_examples(
                query[:max_input_length], similar_examples
            )

            # Use LLM if available, otherwise use mock response
            llm_used = False
            raw_response = ""
            category = "clean"  # Default

            if self.llm_client:
                try:
                    # Call LLM for classification
                    system_prompt = self.prompts.get("system_prompt", "")
                    raw_response = self.call_llm(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=config.max_new_tokens,
                    )
                    category = self.extract_category(raw_response)
                    llm_used = True
                    logger.info(f"LLM classification result: {category}")
                except Exception as e:
                    logger.error(f"Error calling LLM, using default response: {e}")
                    raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: Error occurred during LLM classification."""
            else:
                # Mock LLM response if no LLM client available
                logger.info("No LLM client available, using mock response")
                raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: This is a placeholder response. No LLM service available."""

            return {
                "query": query,
                "category": category,
                "raw_response": raw_response,
                "similar_examples": [
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in similar_examples
                ],
                "prompt": user_prompt,
                # Add metadata for debugging
                "embedding_used": "actual",
                "llm_used": llm_used,
            }

        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            raise

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

    async def create_embedding_async(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Async version of create_embedding

        Args:
            text: Single text string or list of text strings

        Returns:
            numpy.ndarray: Generated embeddings
        """
        if isinstance(text, str):
            text = [text]

        try:
            # Get shared HTTP session
            session = await self.get_http_session()

            # Make async request to embedding API
            async with session.post(
                f"{self.embedding_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key or 'None'}"},
                json={
                    "model": self.embedding_model,
                    "input": text,
                },
                timeout=aiohttp.ClientTimeout(total=30),  # Add timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in embedding API: {error_text}")
                    raise Exception(f"Embedding API error: {error_text}")

                data = await response.json()
                embeddings = np.array([data["data"][0]["embedding"]])
                return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings asynchronously: {e}")
            raise

    async def call_llm_async(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 128
    ) -> str:
        """
        Async version of call_llm

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with examples and query
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response
        """
        try:
            # Get shared HTTP session
            session = await self.get_http_session()

            # Make async request to LLM API
            async with session.post(
                f"{self.llm_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key or 'None'}"},
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=60),  # 60 seconds timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in LLM API: {error_text}")
                    raise Exception(f"LLM API error: {error_text}")

                data = await response.json()
                if "choices" not in data or not data["choices"]:
                    logger.error(f"Invalid response from LLM API: {data}")
                    raise Exception("Invalid response from LLM API")

                raw_response = data["choices"][0]["message"]["content"].strip()
                return raw_response

        except Exception as e:
            logger.error(f"Error calling LLM asynchronously: {e}")
            raise

    async def classify_text_async(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Async version of classify_text

        Args:
            query: Text to classify
            num_examples: Number of similar examples to use
            max_input_length: Maximum input length

        Returns:
            Dictionary with classification results
        """
        try:
            # Create embedding for the query
            if not self.embedding_url:
                logger.error("Embedding URL required for async text classification")
                raise ValueError("Embedding URL not configured")

            # Create embedding for the query
            embedding = await self.create_embedding_async(query[:max_input_length])
            embedding = embedding[0].tolist()

            # Get similar examples using BigQuery
            similar_examples = self.similarity_search(
                embedding=embedding, top_k=num_examples
            )

            # Create prompt with examples
            user_prompt = self.create_prompt_with_examples(
                query[:max_input_length], similar_examples
            )

            # Use LLM if available, otherwise use mock response
            llm_used = False
            raw_response = ""
            category = "clean"  # Default

            if self.llm_url:
                try:
                    # Call LLM for classification
                    system_prompt = self.prompts.get("system_prompt", "")
                    raw_response = await self.call_llm_async(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=config.max_new_tokens,
                    )
                    category = self.extract_category(raw_response)
                    llm_used = True
                    logger.info(f"LLM classification result: {category}")
                except Exception as e:
                    logger.error(f"Error calling LLM, using default response: {e}")
                    raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: Error occurred during LLM classification."""
            else:
                # Mock LLM response if no LLM client available
                logger.info("No LLM URL available, using mock response")
                raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: This is a placeholder response. No LLM service available."""

            return {
                "query": query,
                "category": category,
                "raw_response": raw_response,
                "similar_examples": [
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in similar_examples
                ],
                "prompt": user_prompt,
                # Add metadata for debugging
                "embedding_used": "actual",
                "llm_used": llm_used,
            }

        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            raise
