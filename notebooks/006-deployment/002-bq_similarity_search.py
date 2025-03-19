# %%
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from dataclasses import dataclass
from typing import List
import time
from functools import wraps


def time_execution(func):
    """Decorator to measure execution time of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


# %%
class GCPClient:
    """Class to handle Google Cloud Platform operations"""

    def __init__(self, gcp_key_path):
        """Initialize GCP client with credentials

        Args:
            gcp_key_path (str): Path to the GCP service account JSON key file
        """
        self.credentials = service_account.Credentials.from_service_account_file(
            gcp_key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self.client = bigquery.Client(credentials=self.credentials)

    def execute_query(self, query):
        """Execute a BigQuery SQL query and return results as a DataFrame

        Args:
            query (str): The SQL query to execute

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame
        """
        query_job = self.client.query(query)
        return query_job.to_dataframe()


@dataclass
class RAGEx:
    """RAGEx: RAG+Examples
    Class to store RAG search results with their metadata"""

    text: str
    category: str
    distance: float


@time_execution
def get_similar_texts(
    gcp_client,
    query_embedding,
    dataset_name="org_dataset",
    table_name="comment_moderation_embeddings",
    top_k=5,
    distance_type="COSINE",
    use_brute_force=False,
    fraction_lists_to_search=0.1,
    filter_conditions=None,
) -> List[RAGEx]:
    """Get the top-k most similar texts to the query embedding, optimized for performance

    Args:
        gcp_client (GCPClient): Initialized GCP client
        query_embedding (list): The embedding vector to search for similar texts
        dataset_name (str): BigQuery dataset name
        table_name (str): BigQuery table name
        top_k (int): Number of similar texts to retrieve
        distance_type (str): The distance metric to use ('COSINE', 'EUCLIDEAN', or 'DOT_PRODUCT')
        use_brute_force (bool): Whether to use brute force search for highest accuracy
        fraction_lists_to_search (float): Fraction of index lists to search
        filter_conditions (str, optional): Additional WHERE conditions for filtering the data

    Returns:
        List[RAGEx]: List of RAGEx objects containing similar texts with their categories and distances
    """
    # Convert embedding to a proper SQL array format - back to the original approach
    # This is more reliable than using parameter binding for array types
    embedding_array = ", ".join(str(float(val)) for val in query_embedding)

    # Add filter conditions if provided
    filter_clause = ""
    if filter_conditions:
        filter_clause = f"WHERE {filter_conditions}"

    # Set options based on parameters
    options = {}
    if use_brute_force:
        options = '"use_brute_force": true'
    else:
        # Only add fraction_lists_to_search if not using brute force
        options = f'"fraction_lists_to_search": {fraction_lists_to_search}'

    # Construct the optimized query
    query = f"""
    SELECT
      base.text,
      base.moderation_category as category,
      distance
    FROM
      VECTOR_SEARCH(
        (
          SELECT * FROM `{dataset_name}.{table_name}`
          {filter_clause}
        ),
        'embedding',
        (SELECT ARRAY<FLOAT64>[{embedding_array}]),
        top_k => {top_k},
        distance_type => '{distance_type}',
        options => '{{{options}}}'
      )
      ORDER BY distance;
    """

    # Execute the query and get the results as a DataFrame
    df_results = gcp_client.execute_query(query)

    # Convert DataFrame to list of RAGEx objects with optimized processing
    results = []
    for _, row in df_results.iterrows():
        result = RAGEx(
            text=row["text"],
            category=row["category"],
            distance=round(float(row["distance"]), 6),
        )
        results.append(result)

    return results


@time_execution
def run_random_similarity_search(
    gcp_key_path,
    dataset_name="org_dataset",
    table_name="comment_moderation_embeddings",
    top_k=5,
    distance_type="COSINE",
    use_brute_force=False,
    fraction_lists_to_search=0.1,
) -> List[RAGEx]:
    """Perform similarity search using a random embedding from the table
    This is to test the connection to the GCP client and functioning of the query

    Args:
        gcp_key_path (str): Path to the GCP service account JSON key file
        dataset_name (str): BigQuery dataset name
        table_name (str): BigQuery table name
        top_k (int): Number of similar texts to retrieve

    Returns:
        List[RAGEx]: List of RAGEx objects containing similar texts with their categories and distances
    """
    # Initialize GCP client
    gcp_client = GCPClient(gcp_key_path)

    # Query to get a random embedding
    random_query = f"""
    SELECT embedding
    FROM {dataset_name}.{table_name}
    ORDER BY RAND()
    LIMIT 1
    """

    # Get a random embedding
    random_embedding_df = gcp_client.execute_query(random_query)
    random_embedding = random_embedding_df["embedding"].iloc[0]

    # Use the random embedding to find similar texts
    return get_similar_texts(
        gcp_client=gcp_client,
        query_embedding=random_embedding,
        dataset_name=dataset_name,
        table_name=table_name,
        top_k=top_k,
        distance_type=distance_type,
        use_brute_force=use_brute_force,
        fraction_lists_to_search=fraction_lists_to_search,
    )


# %%
if __name__ == "__main__":
    # Configuration
    gcp_key_path = "/Users/sagar/Downloads/vectordb-bq-0907c2e2227f.json"

    # Example 1: Use a random embedding from the table
    print("Running similarity search with a random embedding...")
    similar_texts = run_random_similarity_search(
        gcp_key_path,
        dataset_name="org_dataset",
        table_name="comment_moderation_embeddings",
        top_k=5,
        distance_type="COSINE",
        use_brute_force=False,
        fraction_lists_to_search=0.1,
    )
    print("\nTop 5 similar texts:")

    # Print formatted results
    for i, result in enumerate(similar_texts, 1):
        print(
            f"\n{i}. Text: {result.text[:100]}..."
            if len(result.text) > 100
            else f"\n{i}. Text: {result.text}"
        )
        print(f"   Category: {result.category}")
        print(f"   Distance: {result.distance:.6f}")

    # Example 2: Use a provided embedding
    # print("\nRunning similarity search with a provided embedding...")
    # # This would be your actual embedding from your model
    # sample_embedding = [0.1, 0.2, 0.3, 0.4]  # Replace with actual embedding dimensions
    #
    # gcp_client = GCPClient(gcp_key_path)
    # similar_texts_with_custom_embedding = get_similar_texts(
    #     gcp_client, sample_embedding
    # )
    #
    # print("\nTop 5 similar texts for custom embedding:")
    # for i, result in enumerate(similar_texts_with_custom_embedding, 1):
    #     print(f"\n{i}. Text: {result.text[:100]}..." if len(result.text) > 100 else f"\n{i}. Text: {result.text}")
    #     print(f"   Category: {result.category}")
    #     print(f"   Distance: {result.distance:.6f}")
