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
import asyncio
import aiohttp
import time
from datetime import datetime
from tqdm import tqdm

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
async def moderate_content_async(session, system, query: str, num_examples: int = 3):
    """Async version of content moderation"""
    # Get similar examples using RAG
    similar_examples = system.similarity_search(query, k=num_examples)
    user_prompt = system.create_prompt_with_examples(query, similar_examples)

    try:
        # Fix URL construction to avoid double slashes
        base_url = str(system.llm_client.base_url).rstrip('/')  # Remove trailing slash if present
        endpoint = f"{base_url}/chat/completions"  # Now we'll have correct URL
        logger.debug(f"Making request to endpoint: {endpoint}")
        logger.debug(f"Model being used: {system.llm_model}")

        request_payload = {
            "model": system.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": system.temperature,
            "max_tokens": system.max_tokens
        }
        logger.debug(f"Request payload: {json.dumps(request_payload, indent=2)}")

        async with session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {system.llm_client.api_key}",
                "Content-Type": "application/json"
            },
            json=request_payload
        ) as response:
            logger.debug(f"Response status: {response.status}")
            response_text = await response.text()
            logger.debug(f"Raw response: {response_text}")

            if response.status != 200:
                raise Exception(f"API call failed with status {response.status}: {response_text}")

            result = json.loads(response_text)

            # Add error checking for response structure
            if not result.get("choices") or not result["choices"]:
                raise Exception("No choices in response")

            if not result["choices"][0].get("message") or not result["choices"][0]["message"].get("content"):
                raise Exception("No message content in response")

            raw_response = result["choices"][0]["message"]["content"].strip()
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
        logger.error(f"Error in async moderation: {str(e)}")
        logger.error(f"Base URL used: {base_url}")
        logger.error(f"Full endpoint: {endpoint}")
        return {
            "query": query,
            "category": "error",
            "raw_response": f"Error: {str(e)}",
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

async def process_batch_async(batch: List[Dict], session, system):
    """Process a batch of items asynchronously"""
    tasks = []
    for item in batch:
        task = asyncio.create_task(
            moderate_content_async(session, system, item["text"], num_examples=7)
        )
        tasks.append((item, task))

    results = []
    for item, task in tasks:
        try:
            response = await task
            results.append({
                "text_id": item["text_id"],
                "text": item["text"],
                "actual_category": item["moderation_category"],
                "model_response": response["raw_response"],
                "predicted_category": response.get("category", "error"),  # Add fallback
                "raw_response": response,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Error processing text_id {item.get('text_id', 'unknown')}: {str(e)}")
            # Add error result to maintain batch size
            results.append({
                "text_id": item["text_id"],
                "text": item["text"],
                "actual_category": item["moderation_category"],
                "model_response": f"Error: {str(e)}",
                "predicted_category": "error",
                "raw_response": {"error": str(e)},
                "timestamp": datetime.now().isoformat(),
            })

    return results

async def run_rag_benchmark_async(
    system: ContentModerationSystem,
    benchmark_data: List[Dict],
    batch_size: int = 16,
    output_dir: str = "benchmark_results",
    model_name: str = "rag_moderation",
    concurrent_batches: int = 4,
    timestamp: str = None,  # Add timestamp parameter
):
    """Run benchmark using async processing"""
    os.makedirs(output_dir, exist_ok=True)
    # Use provided timestamp or generate new one
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_results_{timestamp}.jsonl")

    total_time = 0
    all_results = []
    chunk_size = batch_size * concurrent_batches

    async with aiohttp.ClientSession() as session:
        for chunk_start in tqdm(range(0, len(benchmark_data), chunk_size)):
            chunk = benchmark_data[chunk_start:chunk_start + chunk_size]
            chunk_start_time = time.time()

            batches = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
            tasks = [process_batch_async(batch, session, system) for batch in batches]
            batch_results = await asyncio.gather(*tasks)

            with open(output_file, "a") as f:
                for results in batch_results:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                        all_results.append(result)

            chunk_time = time.time() - chunk_start_time
            total_time += chunk_time
            await asyncio.sleep(0.1)

    logger.info(f"\nBenchmark completed!")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {total_time/len(benchmark_data):.2f} seconds")
    logger.info(f"Results saved to: {output_file}")

    return all_results, timestamp  # Return timestamp along with results

def create_rag_comparison_file(
    previous_results: List[Dict],
    new_results: List[Dict],
    output_file: str = "rag_before_after",
    timestamp: str = None,  # Add timestamp parameter
):
    """Create a comparison file between old and new RAG model results"""
    df_new = pd.DataFrame(new_results)
    df_old = pd.DataFrame(previous_results)

    # Extract categories from new results
    df_new["new_prediction"] = df_new["predicted_category"]

    # Extract categories from old results
    def extract_old_prediction(row):
        try:
            if "processed_response" in row and isinstance(row["processed_response"], dict):
                return row["processed_response"].get("predicted_category", "unknown")
            elif "predicted_category" in row:
                return row["predicted_category"]
            elif "category" in row:
                return row["category"]
            return "unknown"
        except Exception as e:
            logger.error(f"Error extracting old prediction: {e}")
            return "unknown"

    df_old["old_prediction"] = df_old.apply(extract_old_prediction, axis=1)

    # Create simplified comparison dataframe with only requested columns
    df_comparison = df_new[["text_id", "text", "actual_category", "new_prediction"]].merge(
        df_old[["text_id", "old_prediction"]],
        on="text_id",
        how="left"
    )

    # Use provided timestamp or generate new one
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_file}_{timestamp}.jsonl"
    df_comparison.to_json(output_file, orient="records", lines=True)
    logger.info(f"Saved comparison file to {output_file}")

def main():
    """Demo usage of the content moderation system with comparison"""
    # Initialize system
    system = ContentModerationSystem(
        embedding_url="http://localhost:8890/v1",
        llm_url="http://localhost:8899/v1",
        vector_db_path="/root/content-moderation/data/rag/faiss_vector_db"
    )

    # Generate timestamp once
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load previous results
    previous_results = pd.read_json(
        "/root/content-moderation/data/benchmark_results/llm/phi35_benchmark_results_20250210_010356.jsonl",
        lines=True
    ).head(32).to_dict("records")

    # Create benchmark data from previous results
    benchmark_data = [
        {
            "text_id": item["text_id"],
            "text": item["text"],
            "moderation_category": item["actual_category"],
        }
        for item in previous_results
    ]

    # Run benchmark with new RAG system using the same timestamp
    new_results, _ = asyncio.run(
        run_rag_benchmark_async(
            system=system,
            benchmark_data=benchmark_data,
            batch_size=8,
            concurrent_batches=4,
            output_dir="/root/content-moderation/data/benchmark_results/rag",
            model_name="rag_moderation",
            timestamp=timestamp,  # Pass the timestamp
        )
    )

    # Create comparison file with the same timestamp
    create_rag_comparison_file(
        previous_results,
        new_results,
        timestamp=timestamp  # Pass the same timestamp
    )

if __name__ == "__main__":
    main()
