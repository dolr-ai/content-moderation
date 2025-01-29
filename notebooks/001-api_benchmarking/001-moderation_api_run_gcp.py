#%%[markdown]
# # Google Cloud Platform Moderation API Benchmarking
# This notebook runs benchmarking tests for the Google Cloud Platform's moderation API across our benchmark dataset.

#%%
# Import necessary libraries and boilerplate code
from tqdm.notebook import tqdm
import time
from datetime import datetime
import os
import pandas as pd
from pathlib import Path
from huggingface_hub import login as hf_login
import logging
import json
import yaml
from google.oauth2 import service_account
from google.cloud import language_v1
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#%% [markdown]
# # Configs
# Define the primary category mapping
PRIMARY_CATEGORY_MAP = {
    "clean": 0,
    "hate_or_discrimination": 1,
    "violence_or_threats": 2,
    "offensive_language": 3,
    "nsfw_content": 4,
    "spam_or_scams": 5,
}

# Load configuration from YAML
DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set up paths and tokens
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]

# Huggingface login
hf_login(HF_TOKEN)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

#%% [markdown]
# # Authenticate Service Account
# This function authenticates using a service account file path

#%%
def authenticate_service_account(service_account_file):
    """
    Authenticate using a service account file.

    Args:
        service_account_file (str): Path to the service account JSON file.

    Returns:
        Credentials: Authenticated credentials for Google Cloud services.
    """
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return credentials

#%% [markdown]
# # Analyze Text
# This function uses the Google Cloud Natural Language API to analyze text.

#%%
def analyze_text(text, credentials_path):
    """
    Analyze text using Google Cloud Natural Language API

    Args:
        text (str): Text to analyze
        credentials_path (str): Path to Google Cloud credentials file

    Returns:
        dict: Raw API response with sentiment and categories
    """
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = language_v1.LanguageServiceClient(credentials=credentials)

    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="en"
    )

    try:
        response = client.moderate_text(
            request={"document": document}
        )

        return {
            "moderation_categories": [
                {
                    "name": category.name,
                    "confidence": category.confidence
                }
                for category in response.moderation_categories
            ]
        }

    except Exception as e:
        return {"error": str(e)}

#%% [markdown]
# # Process Google Cloud Moderation Response
# This function processes the raw response from the Google Cloud moderation API and maps it to primary categories.

#%%
def process_google_moderation_response(response):
    """
    Process Google Cloud moderation response and map to primary categories.

    Args:
        response (dict): Raw response from Google Cloud moderation API

    Returns:
        dict: Processed results with primary category scores
    """
    processed_results = {
        "clean": 1.0,
        "hate_or_discrimination": 0.0,
        "violence_or_threats": 0.0,
        "offensive_language": 0.0,
        "nsfw_content": 0.0,
        "spam_or_scams": 0.0,
    }

    if "error" in response:
        return {"error": response["error"]}

    category_mapping = {
        # Hate or discrimination
        "Toxic": "hate_or_discrimination",
        "Derogatory": "hate_or_discrimination",
        "Insult": "hate_or_discrimination",

        # Violence or threats
        "Violent": "violence_or_threats",
        "Firearms & Weapons": "violence_or_threats",
        "Death, Harm & Tragedy": "violence_or_threats",
        "War & Conflict": "violence_or_threats",

        # Offensive language
        "Profanity": "offensive_language",

        # NSFW content
        "Sexual": "nsfw_content",

        # Potential spam indicators
        "Finance": "spam_or_scams",
        "Legal": "spam_or_scams",
    }

    for category in response.get("moderation_categories", []):
        mapped_category = category_mapping.get(category["name"])
        if mapped_category:
            current_score = processed_results[mapped_category]
            processed_results[mapped_category] = max(current_score, category["confidence"])

    harmful_score = max(
        processed_results["hate_or_discrimination"],
        processed_results["violence_or_threats"],
        processed_results["offensive_language"],
        processed_results["nsfw_content"],
        processed_results["spam_or_scams"]
    )

    # Calculate the clean score (reduce the score by the maximum harmful score)
    processed_results["clean"] = 1.0 - harmful_score

    # Get the predicted category (highest scoring category)
    predicted_category = max(processed_results.items(), key=lambda x: x[1])[0]

    return {
        "scores": processed_results,
        "predicted_category": predicted_category, # this is primary predicted category
        "predicted_score": processed_results[predicted_category]
    }

#%%[markdown]
# ## Setup Logging and Configuration

# Authentication
credentials = authenticate_service_account(config["secrets"]["GCP_CREDENTIALS_PATH"])

#%%[markdown]
# ## Run Benchmarking
# We'll process each text through the API and store both raw and processed results.

#%%
def process_text(args):
    """
    Process a single text entry with the API.

    Args:
        args (tuple): (text_id, text, actual_category, batch_num)

    Returns:
        dict: Result dictionary with processing details
    """
    text_id, text, actual_category, batch_num = args
    start_time = time.time()

    try:
        raw_response = analyze_text(text, config["secrets"]["GCP_CREDENTIALS_PATH"])
        processed_response = process_google_moderation_response(raw_response)

        processing_time = time.time() - start_time
        return {
            'text_id': text_id,
            'text': text,
            'actual_category': actual_category,
            'raw_api_response': raw_response,
            'processed_response': processed_response,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'batch_num': batch_num
        }
    except Exception as e:
        logger.error(f"Error processing text_id {text_id}: {str(e)}")
        return None

def run_benchmark(df, output_dir, batch_size=100, max_workers=5):
    """
    Run benchmarking on the dataset in parallel with progress tracking and error handling.

    Args:
        df (pd.DataFrame): Input DataFrame with texts to analyze
        output_dir (Path): Directory to save final results
        batch_size (int): Number of samples to process before saving intermediate results
        max_workers (int): Maximum number of concurrent workers

    Returns:
        pd.DataFrame: DataFrame with benchmark results
    """
    # Create cache directory within output directory
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = time.time()
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))

        batch_data = [
            (idx, row['text'], row['moderation_category'], batch_num)
            for idx, row in df.iloc[start_idx:end_idx].iterrows()
        ]

        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(process_text, args): args
                for args in batch_data
            }

            for future in tqdm(
                as_completed(future_to_text),
                total=len(batch_data),
                desc=f"Batch {batch_num + 1}/{total_batches}"
            ):
                result = future.result()
                if result is not None:
                    batch_results.append(result)

        batch_processing_time = time.time() - batch_start

        # Add batch timing information
        for result in batch_results:
            result['batch_processing_time'] = batch_processing_time

        results.extend(batch_results)

        # Save intermediate results for this batch
        if batch_results:
            intermediate_df = pd.DataFrame(batch_results)
            intermediate_df.to_json(
                cache_dir / f"gcp_results_batch_{batch_num + 1}_of_{total_batches}.jsonl",
                orient='records',
                lines=True
            )

        logger.info(f"Batch {batch_num + 1}/{total_batches} completed in {batch_processing_time:.2f} seconds")

        # Add a small delay between batches to prevent API rate limiting
        time.sleep(0.5)

    return pd.DataFrame(results)

#%%[markdown]
# ## Execute Benchmarking

#%%
# Run the benchmark
gcp_results_dir = DATA_ROOT / "benchmark_results" / "gcp"
os.makedirs(gcp_results_dir, exist_ok=True)

df_benchmark = pd.read_json(DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", lines=True)
logger.info("Starting benchmarking run...")
# df_benchmark = df_benchmark.sample(1000) # sample for testing
df_results = run_benchmark(df_benchmark, output_dir=gcp_results_dir)

#%%[markdown]
# ## Save Results

#%%
# Save the complete results


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = gcp_results_dir / f"gcp_benchmark_results_{timestamp}.jsonl"

df_results.to_json(
    output_path,
    orient='records',
    lines=True
)

logger.info(f"Benchmarking complete. Results saved to {output_path}")

#%%[markdown]
# ## Basic Results Analysis

#%%
# Calculate some basic metrics
metrics = {
    'total_processed': len(df_results),
    'successful_calls': len(df_results[df_results['raw_api_response'].apply(lambda x: 'error' not in x)]),
    'failed_calls': len(df_results[df_results['raw_api_response'].apply(lambda x: 'error' in x)]),
}

print("\nBenchmarking Summary:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")
