"""
Google Cloud Platform utilities for the moderation server
"""

import logging
import pandas as pd
import numpy as np
import json
import io
import time
import asyncio
import concurrent.futures
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
        project_id: Optional[str] = None,
        dataset_id: str = "stage_test_tables",
        table_id: str = "test_comment_mod_embeddings",
        bq_pool_size: int = 20,
    ):
        """
        Initialize GCP utilities
        Args:
            gcp_credentials: GCP credentials JSON as a string
            bucket_name: GCS bucket name
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            bq_pool_size: Size of the BigQuery client pool
        """
        self.gcp_credentials = gcp_credentials
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.bq_pool_size = bq_pool_size

        self.credentials = None
        self.bq_client = None
        self.storage_client = None

        # Client pools for scaling
        self.bq_client_pool = []
        self.bq_pool_lock = asyncio.Lock()
        self.bq_pool_initialized = False

        # Thread pool for executing BigQuery operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.bq_pool_size, thread_name_prefix="bq_worker"
        )

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

            # Initialize a single client for backward compatibility
            self.bq_client = bigquery.Client(
                credentials=self.credentials, project=self.project_id
            )
            self.storage_client = storage.Client(
                credentials=self.credentials, project=self.project_id
            )

            # Initialize the BigQuery client pool
            self._initialize_bq_client_pool()

            logger.info(f"Initialized GCP credentials from JSON string")
        except Exception as e:
            logger.error(f"Failed to initialize GCP credentials from string: {e}")
            raise e

    def _initialize_bq_client_pool(self) -> None:
        """Initialize a pool of BigQuery clients for better scaling"""
        try:
            if not self.credentials:
                logger.error("Cannot initialize BigQuery client pool: No credentials")
                return

            # Create multiple BigQuery clients
            for i in range(self.bq_pool_size):
                client = bigquery.Client(
                    credentials=self.credentials,
                    project=self.project_id,
                    # Configure BigQuery client for better performance
                    # These settings help manage resource usage under load
                )
                self.bq_client_pool.append(client)

            logger.info(
                f"Initialized BigQuery client pool with {self.bq_pool_size} clients"
            )
            self.bq_pool_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client pool: {e}")
            self.bq_pool_initialized = False

    async def get_bq_client(self):
        """
        Get a BigQuery client from the pool
        Returns:
            A BigQuery client from the pool
        """
        async with self.bq_pool_lock:
            if not self.bq_pool_initialized:
                # Fall back to the single client if pool isn't initialized
                return self.bq_client

            if not self.bq_client_pool:
                logger.error("BigQuery client pool is empty")
                return self.bq_client

            # Simple round-robin selection from the pool
            client = self.bq_client_pool.pop(0)
            self.bq_client_pool.append(client)
            return client

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

    def _execute_bigquery_search(
        self,
        client,
        query: str,
        job_config=None,
    ) -> pd.DataFrame:
        """
        Execute a BigQuery query with a specific client
        Args:
            client: BigQuery client to use
            query: Query to execute
            job_config: Optional job configuration
        Returns:
            DataFrame with query results
        """
        start_time = time.time()
        retry_count = 0
        max_retries = 3
        retry_delay = 0.5  # Start with 0.5 second delay

        while retry_count <= max_retries:
            try:
                # Execute query with timeout and retry settings
                query_job = client.query(query, job_config=job_config)

                # Set a timeout for the query execution to prevent hanging
                timeout = 25  # seconds
                start_wait = time.time()

                # Wait for the job to complete with timeout
                while not query_job.done() and (time.time() - start_wait) < timeout:
                    time.sleep(0.1)

                if not query_job.done():
                    raise TimeoutError(f"Query execution timed out after {timeout}s")

                # Check for errors
                if query_job.errors:
                    raise Exception(f"Query failed with errors: {query_job.errors}")

                # Convert to DataFrame
                results = query_job.to_dataframe()

                duration = time.time() - start_time
                logger.info(
                    f"BigQuery query execution took {duration*1000:.2f}ms after {retry_count} retries"
                )
                return results

            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    duration = time.time() - start_time
                    logger.error(
                        f"BigQuery query failed after {duration*1000:.2f}ms and {retry_count-1} retries: {e}"
                    )
                    raise

                # Implement exponential backoff
                sleep_time = retry_delay * (
                    2 ** (retry_count - 1)
                )  # Exponential backoff
                logger.warning(
                    f"BigQuery query attempt {retry_count} failed: {e}. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

    async def bigquery_vector_search_async(
        self,
        embedding: List[float],
        top_k: int = 5,
        distance_type: str = "COSINE",
        options: str = '{"fraction_lists_to_search": 0.1, "use_brute_force": false}',
    ) -> pd.DataFrame:
        """
        Perform vector search in BigQuery asynchronously
        Args:
            embedding: Embedding vector to search
            top_k: Number of results to return
            distance_type: Distance metric type (COSINE, EUCLIDEAN, etc.)
            options: JSON string with search options
        Returns:
            DataFrame with search results
        """
        start_time = time.time()

        # Get client from pool
        client = await self.get_bq_client()

        # Convert embedding to string for SQL query
        embedding_str = "[" + ", ".join(str(x) for x in embedding) + "]"

        # Construct query with timeout settings
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
            # Create job config with advanced settings for high concurrency
            job_config = bigquery.QueryJobConfig(
                priority=bigquery.QueryPriority.INTERACTIVE,
                use_query_cache=True,
                maximum_bytes_billed=100_000_000,  # Limit bytes billed to control costs
                labels={
                    "service": "moderation",
                    "component": "vector_search",
                },
            )

            # Execute in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._execute_bigquery_search,
                client,
                query,
                job_config,
            )

            # Calculate total time including query construction
            total_duration = time.time() - start_time
            logger.info(
                f"Total BigQuery vector search took {total_duration*1000:.2f}ms"
            )
            logger.info(f"BigQuery vector search returned {len(results)} results")

            return results
        except Exception as e:
            # Log error with timing information
            duration = time.time() - start_time
            logger.error(
                f"BigQuery vector search failed after {duration*1000:.2f}ms: {e}"
            )
            raise

    def bigquery_vector_search(
        self,
        embedding: List[float],
        top_k: int = 5,
        distance_type: str = "COSINE",
        options: str = '{"fraction_lists_to_search": 0.1, "use_brute_force": false}',
    ) -> pd.DataFrame:
        """
        Perform vector search in BigQuery (synchronous wrapper for the async version)
        Args:
            embedding: Embedding vector to search
            top_k: Number of results to return
            distance_type: Distance metric type (COSINE, EUCLIDEAN, etc.)
            options: JSON string with search options
        Returns:
            DataFrame with search results
        """
        start_time = time.time()

        if not self.bq_client:
            raise ValueError("BigQuery client not initialized")

        # If we're in synchronous context, use the default client directly
        # Convert embedding to string for SQL query
        embedding_str = "[" + ", ".join(str(x) for x in embedding) + "]"

        # Construct query with the same settings as the async version
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
            # Create job config with advanced settings for high concurrency
            job_config = bigquery.QueryJobConfig(
                priority=bigquery.QueryPriority.INTERACTIVE,
                use_query_cache=True,
                maximum_bytes_billed=100_000_000,  # Limit bytes billed to control costs
                labels={
                    "service": "moderation",
                    "component": "vector_search",
                },
            )

            # Log query execution start
            query_start = time.time()

            # Use the same execute function with retry logic
            results = self._execute_bigquery_search(self.bq_client, query, job_config)

            # Calculate total time including query construction
            total_duration = time.time() - start_time
            logger.info(
                f"Total BigQuery vector search took {total_duration*1000:.2f}ms"
            )
            logger.info(f"BigQuery vector search returned {len(results)} results")

            return results
        except Exception as e:
            # Log error with timing information
            duration = time.time() - start_time
            logger.error(
                f"BigQuery vector search failed after {duration*1000:.2f}ms: {e}"
            )
            raise

    async def close(self):
        """Clean up resources when shutting down"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
            logger.info("Shut down BigQuery thread pool")
