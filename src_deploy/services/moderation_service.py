"""
Moderation service for content classification
"""

import logging
import yaml
import jinja2
import re
import pandas as pd
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
        gcp_credentials_path: Optional[Union[str, Path]] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        embeddings_file: Optional[Union[str, Path]] = None,
        bucket_name: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
    ):
        """
        Initialize the content moderation service

        Args:
            gcp_credentials_path: Path to GCP credentials JSON file
            prompt_path: Path to prompts file
            embeddings_file: Path to embeddings file in GCS or local
            bucket_name: GCS bucket name for embeddings file
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        # Initialize GCP utils
        self.gcp_utils = GCPUtils(
            credentials_path=(
                Path(gcp_credentials_path) if gcp_credentials_path else None
            ),
            bucket_name=bucket_name,
            embeddings_file=(
                str(embeddings_file) if embeddings_file else "rag/gcp-embeddings.jsonl"
            ),
            dataset_id=dataset_id,
            table_id=table_id,
        )

        # Load prompts
        self.prompt_path = prompt_path or config.prompt_path
        self.prompts = self._load_prompt(self.prompt_path)
        logger.info(f"Loaded prompt from: {self.prompt_path}")

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment()

        # Embeddings data
        self.embeddings_df = None
        self.embeddings_file_path = None

        # Initialize the embeddings cache from local file if available
        if embeddings_file and Path(embeddings_file).exists():
            self.embeddings_file_path = Path(embeddings_file)
            self.load_embeddings()

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

    def load_embeddings(self, file_path: Optional[Path] = None) -> None:
        """
        Load embeddings from file
        Args:
            file_path: Path to embeddings file
        """
        try:
            file_path = file_path or self.embeddings_file_path
            if file_path is None:
                logger.error("No embeddings file path provided")
                return

            self.embeddings_df = self.gcp_utils.read_embeddings_file(file_path)
            self.embeddings_file_path = file_path
            logger.info(f"Loaded {len(self.embeddings_df)} embeddings from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def download_embeddings(
        self,
        bucket_name: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        local_path: Optional[Path] = None,
    ) -> Path:
        """
        Download embeddings from GCS and load them
        Args:
            bucket_name: GCS bucket name
            embeddings_path: Path to embeddings in GCS
            local_path: Local path to save embeddings
        Returns:
            Path to downloaded file
        """
        try:
            file_path = self.gcp_utils.download_embeddings_from_gcs(
                bucket_name=bucket_name,
                embeddings_path=embeddings_path,
                local_path=local_path,
            )
            self.load_embeddings(file_path)
            return file_path
        except Exception as e:
            logger.error(f"Failed to download and load embeddings: {e}")
            raise

    def get_random_embedding(self) -> List[float]:
        """
        Get a random embedding from the loaded embeddings
        Returns:
            Random embedding vector
        """
        if self.embeddings_df is None:
            if self.embeddings_file_path and Path(self.embeddings_file_path).exists():
                self.load_embeddings()
            else:
                raise ValueError("Embeddings not loaded. Call load_embeddings first.")

        return self.gcp_utils.get_random_embedding(df=self.embeddings_df)

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
            logger.info(f"Similarity search results: {results}")
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
        # todo: get query embedding from embedding server
        # Get random embedding for similarity search
        random_embedding = self.get_random_embedding()

        # Get similar examples using BigQuery
        similar_examples = self.similarity_search(
            embedding=random_embedding, top_k=num_examples
        )

        # todo: one drawback of query[:max_input_length] is that it will truncate the query if it's longer than max_input_length
        # todo: if the slur/ insult is truncated, the similarity search will not find relevant examples and category might be wrong
        # todo: batch the query in chunks and process each chunk separately and then aggregate the results
        # Create prompt with examples
        user_prompt = self.create_prompt_with_examples(
            query[:max_input_length], similar_examples
        )

        # todo: call the LLM with the prompt and get category and store response in raw_response
        # todo: extract relevant fields from the response and store in the response dictionary

        default_category = "clean"
        # Mock LLM response for demonstration purposes
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

        """
        # Future implementation with real LLM
        response = self.call_llm(
            system_prompt=self.prompts.get("system_prompt", ""),
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )

        category = self.extract_category(response)
        """

        # For now, use mock implementation
        return self.classify_text(query, num_examples, max_input_length)
