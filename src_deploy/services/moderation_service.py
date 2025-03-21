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

from src_deploy.utils.gcp_utils import GCPUtils
from src_deploy.config import config

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
        gcs_embeddings_path: Optional[str] = None,
        gcs_prompt_path: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
    ):
        """
        Initialize the content moderation service

        Args:
            gcp_credentials: GCP credentials JSON as a string
            prompt_path: Path to prompts file (local file path)
            bucket_name: GCS bucket name for embeddings file
            gcs_embeddings_path: Path to embeddings file in GCS
            gcs_prompt_path: Path to prompts file in GCS
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        # Initialize GCP utils
        self.gcp_utils = GCPUtils(
            gcp_credentials=gcp_credentials,
            bucket_name=bucket_name,
            gcs_embeddings_path=gcs_embeddings_path or "rag/gcp-embeddings.jsonl",
            dataset_id=dataset_id,
            table_id=table_id,
        )

        # Store GCS paths
        self.bucket_name = bucket_name
        self.gcs_embeddings_path = gcs_embeddings_path or "rag/gcp-embeddings.jsonl"
        self.gcs_prompt_path = gcs_prompt_path or "rag/moderation_prompts.yml"

        # Load prompts
        self.prompt_path = prompt_path or config.prompt_path
        self.prompts = self._load_prompts()
        logger.info(f"Loaded prompts successfully")

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment()

        # Embeddings data
        self.embeddings_df = None

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

    def load_embeddings(self) -> None:
        """
        Load embeddings from GCS
        """
        if not self.bucket_name:
            logger.error("No GCS bucket name provided for embeddings")
            return

        try:
            # Download embeddings file content from GCS
            logger.info(
                f"Loading embeddings from gs://{self.bucket_name}/{self.gcs_embeddings_path}"
            )
            content = self.gcp_utils.download_file_from_gcs(
                gcs_path=self.gcs_embeddings_path
            )

            # Parse the JSONL content
            self.embeddings_df = self.gcp_utils.load_embeddings_from_jsonl(content)
            logger.info(f"Loaded {len(self.embeddings_df)} embeddings from GCS")
        except Exception as e:
            logger.error(f"Failed to load embeddings from GCS: {e}")
            raise

    def get_random_embedding(self) -> List[float]:
        """
        Get a random embedding from the loaded embeddings
        Returns:
            Random embedding vector
        """
        if self.embeddings_df is None or len(self.embeddings_df) == 0:
            # If embeddings aren't loaded, try to load them
            try:
                self.load_embeddings()
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                raise ValueError("Embeddings could not be loaded")

        return self.gcp_utils.get_random_embedding(self.embeddings_df)

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

    def classify_text(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Classify text using RAG examples (without calling LLM)
        Args:
            query: Text to classify
            num_examples: Number of similar examples to use
            max_input_length: Maximum input length
        Returns:
            Dictionary with classification results
        """
        # Get random embedding for similarity search
        random_embedding = self.get_random_embedding()

        # Get similar examples using BigQuery
        similar_examples = self.similarity_search(
            embedding=random_embedding, top_k=num_examples
        )

        # Create prompt with examples
        user_prompt = self.create_prompt_with_examples(
            query[:max_input_length], similar_examples
        )

        # Mock LLM response for demonstration purposes
        # In production, this would call a real LLM
        default_category = "clean"
        mock_response = f"""Category: {default_category}
Confidence: MEDIUM
Explanation: This is a placeholder response. In a production environment, this would be processed by an LLM."""

        return {
            "query": query,
            "category": default_category,
            "raw_response": mock_response,
            "similar_examples": [
                {"text": ex.text, "category": ex.category, "distance": ex.distance}
                for ex in similar_examples
            ],
            "prompt": user_prompt,
            # Add metadata for debugging
            "embedding_used": "random",
            "llm_used": False,
        }

    # Placeholder method for future LLM integration
    def call_llm(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 128
    ) -> str:
        """
        Placeholder for LLM API call
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with examples and query
            max_tokens: Maximum tokens to generate
        Returns:
            LLM response
        """
        # This would be implemented with a real LLM API in production
        logger.info("LLM API call is not implemented. Using mock response.")
        return """Category: clean
Confidence: MEDIUM
Explanation: This is a placeholder response."""

    # For future implementation with real LLM
    def classify_text_with_llm(
        self,
        query: str,
        num_examples: int = 3,
        max_input_length: int = 2000,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Future implementation of text classification with LLM
        """
        # For now, use mock implementation
        return self.classify_text(query, num_examples, max_input_length)
