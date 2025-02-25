# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from openai import OpenAI
import faiss
import logging
import json
from dataclasses import dataclass
from typing import List, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Add these constants at the top level
SYSTEM_PROMPT = """You are a content moderation expert. Your task is to analyze content and categorize it into one of the following categories:

1. hate_or_discrimination: Content targeting protected characteristics with negative intent/prejudice
2. violence_or_threats: Content that threatens, depicts, or promotes violence
3. offensive_language: Hostile or inappropriate content WITHOUT targeting protected characteristics
4. nsfw_content: Explicit sexual content or material intended to arouse
5. spam_or_scams: Deceptive or unsolicited content designed to mislead
6. clean: Content that is allowed and doesn't fall into above categories

Please format your response exactly as:
Category: [exact category_name]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [short 1/2 line explanation]"""

VALID_CATEGORIES = {
    "hate_or_discrimination",
    "violence_or_threats",
    "offensive_language",
    "nsfw_content",
    "spam_or_scams",
    "clean"
}

def extract_category(model_response: str) -> str:
    """Parse the model response to extract category."""
    try:
        category_match = re.search(
            r"Category:\s*(\w+(?:_?\w+)*)",
            model_response,
            re.IGNORECASE
        )
        if category_match:
            category = category_match.group(1).lower()
            return category if category in VALID_CATEGORIES else "clean"
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return "error_parsing"
    return "no_category_found"

@dataclass
class RAGEx:
    """RAGEx: RAG+Examples
    Class to store RAG search results with their metadata"""
    text: str
    category: str
    distance: float

class ContentModerationSystem:
    def __init__(
        self,
        embedding_url: str = "http://localhost:8890/v1",  # Embedding server port
        llm_url: str = "http://localhost:8899/v1",        # LLM server port
        api_key: str = "None",
        embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        llm_model: str = "microsoft/Phi-3.5-mini-instruct",
        vector_db_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 100
    ):
        """
        Initialize the content moderation system with both embedding and LLM capabilities

        Args:
            embedding_url: URL for the embedding server
            llm_url: URL for the LLM server
            api_key: API key (if needed)
            embedding_model: Model to use for embeddings
            llm_model: Model to use for LLM inference
            vector_db_path: Path to the vector database
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens for LLM response
        """
        # Initialize separate clients for embedding and LLM
        self.embedding_client = OpenAI(base_url=embedding_url, api_key=api_key)
        self.llm_client = OpenAI(base_url=llm_url, api_key=api_key)

        # Model settings
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Vector DB state
        self.index = None
        self.metadata_df = None

        # Load vector database if path provided
        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def load_vector_database(self, db_path: Union[str, Path]) -> bool:
        """Load the FAISS vector database and metadata"""
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
        """Create embeddings for input text"""
        if isinstance(text, str):
            text = [text]

        try:
            response = self.embedding_client.embeddings.create(  # Use embedding_client
                model=self.embedding_model,
                input=text
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[RAGEx]:
        """
        Perform similarity search and return structured results

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of RAGEx objects containing similar texts and their metadata
        """
        if self.index is None or self.metadata_df is None:
            raise ValueError("Vector database not loaded. Call load_vector_database first.")

        # Create query embedding and search
        query_embedding = self.create_embedding(query)
        D, I = self.index.search(query_embedding.astype("float32"), k)

        # Format results
        results = []
        for dist, idx in zip(D[0], I[0]):
            example = RAGEx(
                text=self.metadata_df.iloc[idx]["text"],
                category=self.metadata_df.iloc[idx].get("moderation_category", "unknown"),
                distance=float(dist)
            )
            results.append(example)

        return results

    def create_prompt_with_examples(
        self,
        query: str,
        examples: List[RAGEx],
        max_examples: int = 3
    ) -> str:
        """
        Create a prompt that includes similar examples for few-shot learning
        """
        # Sort examples by distance and take top k
        sorted_examples = sorted(examples, key=lambda x: x.distance)[:max_examples]

        # Create the prompt with examples
        prompt = "Here are some example classifications:\n\n"
        for i, example in enumerate(sorted_examples, 1):
            prompt += f"Text: {example.text}\n"
            prompt += f"Category: {example.category}\n\n"

        # Add the query
        prompt += f"Now, please classify this text:\n{query}"
        return prompt

    def classify_text(
        self,
        query: str,
        num_examples: int = 3
    ) -> Dict[str, Any]:
        """
        Classify text using RAG-enhanced LLM
        """
        # Get similar examples using RAG
        similar_examples = self.similarity_search(query, k=num_examples)

        # Create prompt with examples
        user_prompt = self.create_prompt_with_examples(query, similar_examples)

        try:
            # Use llm_client for chat completions with system prompt
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract and validate the category from response
            raw_response = response.choices[0].message.content.strip()
            category = extract_category(raw_response)

            return {
                "query": query,
                "category": category,
                "raw_response": raw_response,
                "similar_examples": [
                    {
                        "text": ex.text,
                        "category": ex.category,
                        "distance": ex.distance
                    }
                    for ex in similar_examples
                ],
                "prompt": user_prompt
            }

        except Exception as e:
            logger.error(f"Error in LLM inference: {str(e)}")
            return {
                "query": query,
                "category": "error",
                "error_message": str(e),
                "similar_examples": [
                    {
                        "text": ex.text,
                        "category": ex.category,
                        "distance": ex.distance
                    }
                    for ex in similar_examples
                ],
                "prompt": user_prompt
            }

# %%
def main():
    """Demo usage of the content moderation system"""
    # Initialize system with separate URLs for embedding and LLM
    system = ContentModerationSystem(
        embedding_url="http://localhost:8890/v1",  # Embedding server
        llm_url="http://localhost:8899/v1",        # LLM server
        vector_db_path="/root/content-moderation/data/rag/faiss_vector_db"
    )

    # Example queries
    queries = [
        "You wait right there i am coming for you",
        "Hello, how are you doing today?",
        "I will find you and make you pay for what you did"
    ]

    # Process each query
    for query in queries:
        print(f"\nProcessing query: {query}")
        result = system.classify_text(query, num_examples=10)

        if "error_message" in result:
            print(f"Error occurred: {result['error_message']}")
            print("\nSimilar examples found (RAG still working):")
        else:
            print(f"Category: {result['category']}")
            print(f"Raw response:\n{result['raw_response']}")
            print("\nSimilar examples used:")

        for ex in result['similar_examples']:
            print(f"- {ex['category']}: {ex['text'][:100]}...")

if __name__ == "__main__":
    main()
