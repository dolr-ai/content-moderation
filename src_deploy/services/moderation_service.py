"""
Moderation Service

This module provides a service layer for content moderation, using BigQuery for RAG (Retrieval Augmented Generation).
"""

import os
import logging
import asyncio
import yaml
import jinja2
import re
import numpy as np
import time
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
import aiohttp
from openai import OpenAI

from models.api_models import ModerationRequest, ModerationResponse, TimingMetrics
from utils.gcp_utils import GCPUtils

# Configure logging
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
    """Service for content moderation using BigQuery for RAG"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the moderation service

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ready = False
        self.version = config.version

        # Initialize GCP utils
        self.gcp_utils = None

        # Set up prompts
        self.prompt_path = self.config.get("PROMPT_PATH")
        self.bucket_name = self.config.get("GCS_BUCKET_NAME")
        self.gcs_prompt_path = self.config.get("GCS_PROMPT_PATH")
        self.prompts = None

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment()

        # Store service endpoints
        self.embedding_url = self.config.get("EMBEDDING_URL")
        self.llm_url = self.config.get("LLM_URL")
        self.api_key = self.config.get("SGLANG_API_KEY")

        # Model settings
        self.embedding_model = self.config.get("EMBEDDING_MODEL")
        self.llm_model = self.config.get("LLM_MODEL")

        # BigQuery settings
        self.dataset_id = self.config.get("DATASET_ID")
        self.table_id = self.config.get("TABLE_ID")

        # Initialize HTTP session for async calls
        self.http_session = None
        self.http_session_lock = asyncio.Lock()

        # OpenAI clients
        self.embedding_client = None
        self.llm_client = None

    async def initialize(self) -> bool:
        """
        Initialize the moderation system asynchronously

        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing moderation service...")

            # Load prompts
            self.prompts = await asyncio.to_thread(self._load_prompts)

            # Initialize GCP utils
            gcp_credentials = self.config.get("GCP_CREDENTIALS")

            # Calculate optimal BigQuery pool size based on system resources and config
            # Default pool size is now calculated based on expected concurrency
            cpu_count = os.cpu_count() or 4
            default_pool_size = min(cpu_count * 5, 40)  # Scale with CPU but cap at 40
            bq_pool_size = int(self.config.get("BQ_POOL_SIZE", default_pool_size))

            logger.info(
                f"Initializing GCP utils with BigQuery pool size: {bq_pool_size} (CPU cores: {cpu_count})"
            )

            self.gcp_utils = GCPUtils(
                gcp_credentials=gcp_credentials,
                bucket_name=self.bucket_name,
                dataset_id=self.dataset_id,
                table_id=self.table_id,
                bq_pool_size=bq_pool_size,
            )

            # Initialize OpenAI clients
            if self.embedding_url:
                self.embedding_client = OpenAI(
                    base_url=self.embedding_url, api_key=self.api_key
                )
                logger.info(
                    f"Initialized embedding client with URL {self.embedding_url}"
                )

            if self.llm_url:
                self.llm_client = OpenAI(base_url=self.llm_url, api_key=self.api_key)
                logger.info(f"Initialized LLM client with URL {self.llm_url}")

            # Mark as ready
            self.ready = True
            logger.info("Moderation service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize moderation service: {e}")
            self.ready = False
            return False

    def _load_prompts(self) -> Dict[str, Any]:
        """
        Load prompts from YAML file (local or GCS)
        Returns:
            Dictionary with prompt templates
        """
        try:
            # Try loading from GCS first if bucket is configured
            if (
                self.bucket_name
                and self.gcp_utils
                and self.gcp_utils.storage_client
                and self.gcs_prompt_path
            ):
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

            # Try loading from local file if path provided
            if self.prompt_path and Path(self.prompt_path).exists():
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
                    limit=192,  # Connection pool size
                    limit_per_host=192,  # Connections per host
                    keepalive_timeout=60,  # Keep connections alive for 60 seconds
                )
                timeout = aiohttp.ClientTimeout(total=90)  # 90 seconds timeout
                self.http_session = aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                )
            return self.http_session

    async def create_embedding_async(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Async version of create_embedding - transforms text into embeddings through API call

        This method:
        1. Validates and normalizes the input text
        2. Makes an asynchronous request to the embedding API
        3. Handles various error conditions and timeouts
        4. Returns parsed embedding as numpy array

        Args:
            text: Single text string or list of text strings to embed
                  Can handle a single string or a list of strings

        Returns:
            numpy.ndarray: Generated embeddings as a numpy array

        Raises:
            ValueError: If embedding URL is not configured
            Exception: For API errors or other failures
        """
        start_time = time.time()
        try:
            # Get shared HTTP session - reuses connections for better performance
            session = await self.get_http_session()

            if not self.embedding_url:
                raise ValueError("No embedding URL configured")

            # Ensure input is properly formatted (text must be a string, not None or other types)
            # This prevents errors in the embedding API's text tokenization process
            if isinstance(text, str):
                input_text = text if text else ""  # Convert empty string if None/empty
            elif isinstance(text, list):
                # Ensure all items in the list are strings and not None
                input_text = [item if item else "" for item in text]
            else:
                # Convert to string as fallback for unexpected input types
                input_text = str(text) if text is not None else ""

            # Log request start time
            request_start = time.time()
            # Make async request to embedding API
            # Using timeout to prevent requests from hanging indefinitely
            async with session.post(
                f"{self.embedding_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.embedding_model,
                    "input": input_text,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30
                ),  # 30 second timeout for embedding generation
            ) as response:
                # Handle non-200 responses with detailed error logging
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in embedding API: {error_text}")
                    raise Exception(f"Embedding API error: {error_text}")

                # Parse response JSON and extract embedding
                data = await response.json()

                # Log request duration
                request_duration = time.time() - request_start
                logger.info(f"Embedding API request took {request_duration*1000:.2f}ms")

                # Convert to numpy array for vector operations
                # Expecting structure: {"data": [{"embedding": [0.1, 0.2, ...]}]}
                embeddings = np.array([data["data"][0]["embedding"]])

                # Log total time including processing
                total_duration = time.time() - start_time
                logger.info(
                    f"Total embedding generation took {total_duration*1000:.2f}ms"
                )

                return embeddings

        except Exception as e:
            # Log the error with full traceback and re-raise to allow caller to handle
            duration = time.time() - start_time
            logger.error(
                f"Error creating embeddings asynchronously after {duration*1000:.2f}ms: {e}"
            )
            raise

    async def call_llm_async(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 128
    ) -> str:
        """
        Async version of call_llm - sends prompts to LLM and gets classification response

        This method:
        1. Constructs a chat completion request with system and user prompts
        2. Makes an asynchronous request to the LLM API
        3. Handles timeouts and error conditions
        4. Parses and returns the text response

        Args:
            system_prompt: System prompt to define LLM behavior and instructions
            user_prompt: User prompt with examples and query to classify
            max_tokens: Maximum tokens to generate in the response (limits response length)

        Returns:
            str: Raw text response from the LLM

        Raises:
            ValueError: If LLM URL is not configured
            Exception: For API errors or other failures
        """
        start_time = time.time()
        try:
            # Get shared HTTP session for connection pooling and reuse
            session = await self.get_http_session()

            if not self.llm_url:
                raise ValueError("No LLM URL configured")

            # Log request start time
            request_start = time.time()
            # Make async request to LLM API
            # Using a longer timeout for LLM as it typically takes more time than embeddings
            async with session.post(
                f"{self.llm_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,  # Use 0 temperature for deterministic/consistent results
                    "max_tokens": max_tokens,  # Control response length
                },
                timeout=aiohttp.ClientTimeout(
                    total=60
                ),  # 60 seconds timeout for LLM generation
            ) as response:
                # Handle non-200 responses with detailed error logging
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in LLM API: {error_text}")
                    raise Exception(f"LLM API error: {error_text}")

                # Parse response and validate structure
                data = await response.json()

                # Log request duration
                request_duration = time.time() - request_start
                logger.info(f"LLM API request took {request_duration*1000:.2f}ms")

                if "choices" not in data or not data["choices"]:
                    logger.error(f"Invalid response from LLM API: {data}")
                    raise Exception("Invalid response from LLM API")

                # Extract the content from the first choice in the response
                # Expected structure: {"choices": [{"message": {"content": "..."}}]}
                raw_response = data["choices"][0]["message"]["content"].strip()

                # Log total time including processing
                total_duration = time.time() - start_time
                logger.info(f"Total LLM call took {total_duration*1000:.2f}ms")

                return raw_response

        except Exception as e:
            # Log the error and re-raise to allow caller to handle
            duration = time.time() - start_time
            logger.error(
                f"Error calling LLM asynchronously after {duration*1000:.2f}ms: {e}"
            )
            raise

    async def moderate_content(self, request: ModerationRequest) -> ModerationResponse:
        """
        Moderate content using async methods for BigQuery RAG and LLM classification

        This method orchestrates the full moderation workflow:
        1. Validates service configuration and readiness
        2. Generates embeddings for the input text
        3. Uses BigQuery vector search to find similar examples
        4. Creates a few-shot prompt with similar examples
        5. Calls the LLM to classify the content
        6. Handles errors at each step with appropriate fallbacks
        7. Returns a structured response with results and metadata

        Args:
            request: Moderation request with text to moderate and parameters

        Returns:
            ModerationResponse with moderation results and metadata

        Raises:
            RuntimeError: If service is not initialized
            ValueError: If required configuration is missing
        """
        if not self.ready:
            raise RuntimeError("Moderation service not initialized")

        # Initialize timing metrics
        embedding_time_ms = 0
        llm_time_ms = 0
        bigquery_time_ms = 0
        start_time = time.time()

        try:
            # Validate essential configuration before making API calls
            if not self.embedding_url:
                raise ValueError("Embedding URL is not configured")
            if not self.embedding_model:
                raise ValueError("Embedding model is not configured")
            if not self.dataset_id or not self.table_id:
                raise ValueError("BigQuery dataset or table ID is not configured")

            # 1. Create embedding for the query - truncate input if needed
            embedding_start = time.time()
            embedding = await self.create_embedding_async(
                request.text[: request.max_input_length]
            )
            embedding_time_ms = (time.time() - embedding_start) * 1000
            embedding_list = embedding[
                0
            ].tolist()  # Convert numpy array to list for JSON serialization

            # 2. Get similar examples using BigQuery vector search
            # Use the new async BigQuery implementation directly
            bigquery_start = time.time()

            # Optimize vector search options based on concurrency
            # Adjust search parameters for better performance under load
            vector_search_options = {
                # Increase search fraction for better recall at high concurrency
                "fraction_lists_to_search": 0.15,
                # Don't use brute force by default for better scalability
                "use_brute_force": False,
            }

            similar_examples = await self.gcp_utils.bigquery_vector_search_async(
                embedding=embedding_list,
                top_k=request.num_examples,
                options=json.dumps(vector_search_options),
            )
            bigquery_time_ms = (time.time() - bigquery_start) * 1000

            # Convert DataFrame results to RAGEx objects for easier handling
            rag_examples = []
            for _, row in similar_examples.iterrows():
                example = RAGEx(
                    text=row["text"],
                    category=row.get("moderation_category", "unknown"),
                    distance=float(row["distance"]),
                )
                rag_examples.append(example)

            # 3. Create prompt with examples for few-shot learning
            user_prompt = self.create_prompt_with_examples(
                request.text[: request.max_input_length],
                rag_examples,
                num_examples=request.num_examples,
            )

            # 4. Use LLM for classification with proper error handling
            llm_used = False
            raw_response = ""
            category = "clean"  # Default fallback category

            if self.llm_url and self.llm_model:
                try:
                    # Get system prompt from config
                    system_prompt = self.prompts.get("system_prompt", "")

                    # Call LLM for classification asynchronously
                    llm_start = time.time()
                    max_tokens = int(self.config.get("MAX_NEW_TOKENS", 128))
                    raw_response = await self.call_llm_async(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )
                    llm_time_ms = (time.time() - llm_start) * 1000

                    # Extract category from structured LLM response
                    category = self.extract_category(raw_response)
                    llm_used = True
                    logger.info(f"LLM classification result: {category}")

                except Exception as e:
                    # Gracefully handle LLM errors with a default response
                    logger.error(f"Error calling LLM, using default response: {e}")
                    raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: Error occurred during LLM classification."""
            else:
                # Mock LLM response if no LLM client available
                logger.info("No LLM configuration available, using mock response")
                raw_response = f"""Category: clean
Confidence: MEDIUM
Explanation: This is a placeholder response. No LLM service available."""

            # Calculate total processing time
            total_time_ms = (time.time() - start_time) * 1000

            # Create timing metrics
            timing = TimingMetrics(
                embedding_time_ms=embedding_time_ms,
                llm_time_ms=llm_time_ms,
                bigquery_time_ms=bigquery_time_ms,
                total_time_ms=total_time_ms,
            )

            # Build and return structured response with all relevant data
            return ModerationResponse(
                query=request.text,
                category=category,
                raw_response=raw_response,
                similar_examples=[
                    {"text": ex.text, "category": ex.category, "distance": ex.distance}
                    for ex in rag_examples
                ],
                prompt=user_prompt,
                embedding_used=self.embedding_model or "unknown",
                llm_used=llm_used,
                timing=timing,
            )

        except Exception as e:
            # Log and return error response with minimal information
            logger.error(f"Error in content moderation: {str(e)}")

            # Calculate total processing time even for errors
            total_time_ms = (time.time() - start_time) * 1000

            # Create timing metrics with whatever we have
            timing = TimingMetrics(
                embedding_time_ms=embedding_time_ms,
                llm_time_ms=llm_time_ms,
                bigquery_time_ms=bigquery_time_ms,
                total_time_ms=total_time_ms,
            )

            return ModerationResponse(
                query=request.text,
                category="error",
                raw_response=f"Error: {str(e)}",
                similar_examples=[],
                prompt="",
                embedding_used="none",
                llm_used=False,
                timing=timing,
            )

    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of the service

        Returns:
            Dictionary with health status including:
            - Service initialization status
            - GCP credentials and connection status
            - Dependency service connectivity (embedding and LLM)
            - Configuration details
        """
        health_status = {
            "status": "healthy" if self.ready else "initializing",
            "version": self.version,
        }

        # Check GCP credentials and connectivity
        gcp_status = {
            "credentials_configured": False,
            "bq_client_initialized": False,
            "storage_client_initialized": False,
            "dataset_id": self.dataset_id or "not configured",
            "table_id": self.table_id or "not configured",
        }

        if self.gcp_utils:
            gcp_status["credentials_configured"] = (
                self.gcp_utils.credentials is not None
            )
            gcp_status["bq_client_initialized"] = (
                self.gcp_utils.bq_client is not None
                and self.gcp_utils.bq_pool_initialized
            )
            gcp_status["storage_client_initialized"] = (
                self.gcp_utils.storage_client is not None
            )
            gcp_status["bq_pool_size"] = self.gcp_utils.bq_pool_size

        health_status["gcp"] = gcp_status

        # Check dependent services connectivity
        services_status = {
            "embedding": {
                "url": self.embedding_url or "not configured",
                "model": self.embedding_model or "not configured",
                "available": False,
            },
            "llm": {
                "url": self.llm_url or "not configured",
                "model": self.llm_model or "not configured",
                "available": False,
            },
        }

        # Test embedding service connectivity
        if self.embedding_url:
            try:
                session = await self.get_http_session()
                async with session.post(
                    f"{self.embedding_url}/embeddings",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer None",
                    },
                    json={
                        "model": self.embedding_model,
                        "input": "Test",
                    },
                    timeout=2,  # Quick timeout for health check
                ) as response:
                    services_status["embedding"]["available"] = response.status < 400
            except Exception as e:
                logger.warning(f"Failed to connect to embedding service: {str(e)}")
                services_status["embedding"]["available"] = False

        # Test LLM service connectivity
        if self.llm_url:
            try:
                session = await self.get_http_session()
                async with session.post(
                    f"{self.llm_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer None",
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                    timeout=2,  # Quick timeout for health check
                ) as response:
                    services_status["llm"]["available"] = response.status < 400
            except Exception as e:
                logger.warning(f"Failed to connect to LLM service: {str(e)}")
                services_status["llm"]["available"] = False

        health_status["services"] = services_status

        # Include config info
        health_status["config"] = {
            "embedding_model": self.embedding_model or "not configured",
            "llm_model": self.llm_model or "not configured",
            "dataset_id": self.dataset_id or "not configured",
            "table_id": self.table_id or "not configured",
        }

        # Set overall health based on all components
        if not self.ready:
            health_status["status"] = "initializing"
        elif not gcp_status["bq_client_initialized"]:
            health_status["status"] = "degraded - GCP not connected"
        elif not (
            services_status["embedding"]["available"]
            and services_status["llm"]["available"]
        ):
            health_status["status"] = "degraded - dependent services unavailable"

        return health_status

    async def shutdown(self) -> None:
        """Clean up resources when shutting down"""
        if self.http_session is not None and not self.http_session.closed:
            logger.info("Closing HTTP session")
            await self.http_session.close()

        # Also close BigQuery resources
        if self.gcp_utils:
            await self.gcp_utils.close()
