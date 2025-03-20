"""
Google Cloud Platform utilities for the moderation server
"""

import logging
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from google.cloud import bigquery, storage
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCPUtils:
    """Google Cloud Platform utilities for the moderation server"""

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        bucket_name: Optional[str] = None,
        embeddings_file: str = "rag/gcp-embeddings.jsonl",
        project_id: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
    ):
        """
        Initialize GCP utilities
        Args:
            credentials_path: Path to GCP credentials JSON file
            bucket_name: GCS bucket name
            embeddings_file: Path to embeddings file in GCS or local
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.embeddings_file = embeddings_file
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

        self.credentials = None
        self.bq_client = None
        self.storage_client = None

        # Initialize credentials if path provided
        if credentials_path and credentials_path.exists():
            self.initialize_credentials(credentials_path)

    def initialize_credentials(self, credentials_path: Path) -> None:
        """
        Initialize GCP credentials
        Args:
            credentials_path: Path to GCP credentials JSON file
        """
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.project_id = self.project_id or self.credentials.project_id

            # Initialize clients
            self.bq_client = bigquery.Client(
                credentials=self.credentials, project=self.project_id
            )
            self.storage_client = storage.Client(
                credentials=self.credentials, project=self.project_id
            )

            logger.info(f"Initialized GCP credentials from {credentials_path}")
        except Exception as e:
            logger.error(f"Failed to initialize GCP credentials: {e}")
            raise

    def download_embeddings_from_gcs(
        self,
        bucket_name: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        local_path: Optional[Path] = None,
    ) -> Path:
        """
        Download embeddings file from Google Cloud Storage
        Args:
            bucket_name: GCS bucket name
            embeddings_path: Path to embeddings file in GCS
            local_path: Local path to save the file
        Returns:
            Path to downloaded file
        """
        if not self.storage_client:
            raise ValueError(
                "Storage client not initialized. Call initialize_credentials first."
            )

        bucket_name = bucket_name or self.bucket_name
        if not bucket_name:
            raise ValueError("Bucket name not provided")

        embeddings_path = embeddings_path or self.embeddings_file

        if not local_path:
            local_path = Path("./") / Path(embeddings_path).name

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(embeddings_path)
            blob.download_to_filename(str(local_path))
            logger.info(
                f"Downloaded embeddings from gs://{bucket_name}/{embeddings_path} to {local_path}"
            )
            return local_path
        except Exception as e:
            logger.error(f"Failed to download embeddings from GCS: {e}")
            raise

    def read_embeddings_file(self, file_path: Path) -> pd.DataFrame:
        """
        Read embeddings from JSONL file
        Args:
            file_path: Path to embeddings file
        Returns:
            DataFrame with embeddings
        """
        try:
            df = pd.read_json(file_path, lines=True)
            logger.info(f"Read {len(df)} embeddings from {file_path}")

            # Convert embeddings to Python lists for JSON serialization
            if "embedding" in df.columns:
                df["embedding"] = df["embedding"].apply(lambda x: list(np.float64(x)))

            return df
        except Exception as e:
            logger.error(f"Failed to read embeddings file: {e}")
            raise

    def get_random_embedding(
        self, df: Optional[pd.DataFrame] = None, file_path: Optional[Path] = None
    ) -> List[float]:
        """
        Get a random embedding from the dataframe or file
        Args:
            df: DataFrame with embeddings
            file_path: Path to embeddings file
        Returns:
            Random embedding vector
        """
        if df is None and file_path is not None:
            df = self.read_embeddings_file(file_path)

        if df is None:
            raise ValueError("Either dataframe or file_path must be provided")

        random_embedding = df.sample(1)["embedding"].iloc[0]
        return random_embedding

    def bigquery_vector_search(
        self,
        embedding: List[float],
        top_k: int = 5,
        distance_type: str = "COSINE",
        options: str = '{"fraction_lists_to_search": 0.1, "use_brute_force": false}',
    ) -> pd.DataFrame:
        """
        Perform vector search in BigQuery
        Args:
            embedding: Embedding vector to search
            top_k: Number of results to return
            distance_type: Distance metric type (COSINE, EUCLIDEAN, etc.)
            options: JSON string with search options
        Returns:
            DataFrame with search results
        """
        if not self.bq_client:
            raise ValueError(
                "BigQuery client not initialized. Call initialize_credentials first."
            )

        # Convert embedding to string for SQL query
        embedding_str = "[" + ", ".join(str(x) for x in embedding) + "]"

        # Construct query
        query = f"""
        SELECT
          base.text,
          base.moderation_category,
          distance
        FROM VECTOR_SEARCH(
          TABLE `{self.dataset_id}.{self.table_id}`,
          'embedding',
          (SELECT ARRAY<FLOAT64>{embedding_str}),
          top_k => {top_k},
          distance_type => '{distance_type}',
          options => '{options}'
        )
        ORDER BY distance
        LIMIT {top_k};
        """

        logger.info("Executing BigQuery vector search query")

        try:
            results = self.bq_client.query(query).to_dataframe()
            logger.info(f"BigQuery vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"BigQuery vector search failed: {e}")
            raise
