"""
Google Cloud Platform utilities for the moderation server
"""

import logging
import pandas as pd
import numpy as np
import json
import io
from typing import List, Dict, Any, Optional, Union
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
        gcp_credentials: Optional[str] = None,
        bucket_name: Optional[str] = None,
        gcs_embeddings_path: str = "rag/gcp-embeddings.jsonl",
        project_id: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
    ):
        """
        Initialize GCP utilities
        Args:
            gcp_credentials: GCP credentials JSON as a string
            bucket_name: GCS bucket name
            gcs_embeddings_path: Path to embeddings file in GCS
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        self.gcp_credentials = gcp_credentials
        self.bucket_name = bucket_name
        self.gcs_embeddings_path = gcs_embeddings_path
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

        self.credentials = None
        self.bq_client = None
        self.storage_client = None

        # Initialize credentials if provided
        if gcp_credentials:
            self.initialize_credentials_from_string(gcp_credentials)

    def initialize_credentials_from_string(self, credentials_json: str) -> None:
        """
        Initialize GCP credentials from a JSON string
        Args:
            credentials_json: GCP credentials JSON as a string
        """
        try:
            # Parse credentials JSON
            info = json.loads(credentials_json)

            # Create credentials object
            self.credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.project_id = self.project_id or self.credentials.project_id

            # Initialize clients
            self.bq_client = bigquery.Client(
                credentials=self.credentials, project=self.project_id
            )
            self.storage_client = storage.Client(
                credentials=self.credentials, project=self.project_id
            )

            logger.info(f"Initialized GCP credentials from JSON string")
        except Exception as e:
            logger.error(f"Failed to initialize GCP credentials from string: {e}")
            raise

    def download_file_from_gcs(
        self,
        gcs_path: str,
        bucket_name: Optional[str] = None,
        as_string: bool = False,
    ) -> Union[bytes, str]:
        """
        Download file from Google Cloud Storage to memory
        Args:
            gcs_path: Path to file in GCS
            bucket_name: GCS bucket name (overrides the default)
            as_string: Whether to return as string (UTF-8 decoded)
        Returns:
            File content as bytes or string
        """
        if not self.storage_client:
            raise ValueError("Storage client not initialized")

        bucket_name = bucket_name or self.bucket_name
        if not bucket_name:
            raise ValueError("Bucket name not provided")

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            # Download to memory
            content = blob.download_as_bytes()
            logger.info(f"Downloaded gs://{bucket_name}/{gcs_path} to memory")

            if as_string:
                return content.decode("utf-8")
            return content
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            raise

    def load_embeddings_from_jsonl(
        self, content: Union[bytes, str, Path]
    ) -> pd.DataFrame:
        """
        Read embeddings from JSONL content (file or memory)
        Args:
            content: JSONL content as bytes, string, or file path
        Returns:
            DataFrame with embeddings
        """
        try:
            # Handle different content types
            if isinstance(content, bytes):
                # Content is already in memory as bytes
                df = pd.read_json(io.BytesIO(content), lines=True)
            elif isinstance(content, str) and (
                Path(content).exists() or content.startswith("{")
            ):
                # Content is either a file path or a JSON string
                if Path(content).exists():
                    df = pd.read_json(content, lines=True)
                else:
                    # Assuming it's a single JSON object as string
                    df = pd.read_json(io.StringIO(f"[{content}]"))
            elif isinstance(content, Path):
                # Content is a Path object
                df = pd.read_json(content, lines=True)
            else:
                raise ValueError("Unsupported content type")

            # Convert embeddings to Python lists for JSON serialization if needed
            if "embedding" in df.columns:
                df["embedding"] = df["embedding"].apply(
                    lambda x: list(np.float64(x)) if not isinstance(x, list) else x
                )

            logger.info(f"Loaded {len(df)} embeddings")
            return df
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def get_random_embedding(self, df: pd.DataFrame) -> List[float]:
        """
        Get a random embedding from the dataframe
        Args:
            df: DataFrame with embeddings
        Returns:
            Random embedding vector
        """
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

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
            raise ValueError("BigQuery client not initialized")

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

        try:
            results = self.bq_client.query(query).to_dataframe()
            logger.info(f"BigQuery vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"BigQuery vector search failed: {e}")
            raise
