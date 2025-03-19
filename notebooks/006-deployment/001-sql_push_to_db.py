import json
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
# Path configurations
gcp_key_path = "path/to/your/gcp_key.json"
jsonl_path = "path/to/your/embeddings.jsonl"

# %%
# Setup GCP credentials and client
credentials = service_account.Credentials.from_service_account_file(
    gcp_key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(credentials=credentials)


# %%
# Function to read data using pandas
def read_jsonl_pandas(file_path):
    """Read JSONL file using pandas"""
    df = pd.read_json(file_path, lines=True)

    # Convert embeddings to lists instead of numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: [float(val) for val in x])

    return df


# %%
def insert_batch_to_bigquery(batch_df):
    """Insert a batch of records into BigQuery using the streaming API"""
    table_id = "org_dataset.comment_moderation_embeddings"

    # Convert DataFrame batch to list of dictionaries
    rows_to_insert = batch_df.to_dict("records")

    # Insert data
    errors = client.insert_rows_json(table_id, rows_to_insert)

    if errors:
        print(f"Encountered errors while inserting batch: {errors}")
        return False
    return True


# %%
def process_and_insert_batches(df, batch_size=64, max_workers=4):
    """Process and insert data in batches using parallel execution"""
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    print(f"Total records: {total_rows}")
    print(f"Number of batches: {num_batches}")
    print(f"Using {max_workers} workers for parallel processing")

    successful_batches = 0
    failed_batches = 0
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            future = executor.submit(insert_batch_to_bigquery, batch_df)
            futures.append(future)

        # Process results as they complete
        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing batches")
        ):
            if future.result():
                successful_batches += 1
            else:
                failed_batches += 1

            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"\nProgress: {i+1}/{num_batches} batches processed")
                print(f"Successful: {successful_batches}, Failed: {failed_batches}")

    print("\n\nFinal Results:")
    print(f"Total batches processed: {num_batches}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")


# %%
# Main execution
if __name__ == "__main__":
    print("Reading data from file using pandas...")
    df = read_jsonl_pandas(jsonl_path)

    print("\nDataset Info:")
    print(df.info())

    print("\nSample record:")
    print(df.iloc[0])

    print("\nStarting parallel batch insertion process...")
    process_and_insert_batches(df, batch_size=64, max_workers=8)
