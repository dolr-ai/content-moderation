#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from openai import OpenAI
import faiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

class EmbeddingClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8890/v1",
        api_key: str = "None",
        model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        vector_db_path: Optional[str] = None
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.index = None
        self.metadata_df = None

        if vector_db_path:
            self.load_vector_database(vector_db_path)

    def load_vector_database(self, db_path: Union[str, Path]):
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
        if isinstance(text, str):
            text = [text]

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
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
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.metadata_df is None:
            raise ValueError("Vector database not loaded. Call load_vector_database first.")

        query_embedding = self.create_embedding(query)
        D, I = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            result = {
                "index": int(idx),
                "text": self.metadata_df.iloc[idx]["text"],
                "category": self.metadata_df.iloc[idx].get("moderation_category", None)
            }
            if return_scores:
                result["distance"] = float(dist)
            results.append(result)

        return results

#%%
# Initialize client
client = EmbeddingClient(
    base_url="http://localhost:8890/v1",
    api_key="None",
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
)

#%%
# Load the database
DB_PATH = "/root/content-moderation/data/rag/faiss_vector_db"
client.load_vector_database(DB_PATH)

# Example search
query = "Hello, how are you"
results = client.similarity_search(query, k=10)

# Display results
for i, result in enumerate(results, 1):
    print(f"\n{i}. Distance: {result.get('distance', 'N/A'):.4f}")
    print(f"Category: {result.get('category', 'N/A')}")
    print(f"Text: {result['text'][:2000]}...")