#%%[markdown]
# # OpenAI Moderation API Benchmarking
# This notebook runs benchmarking tests for OpenAI's moderation API across our benchmark dataset.

#%%
# Import necessary libraries and boilerplate code
from tqdm.notebook import tqdm
import time
from datetime import datetime
import os
import pandas as pd
from pathlib import Path
import logging
import json
import yaml
from openai import OpenAI
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

# Set up paths
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

#%% [markdown]
# # Analyze Text Function
# This function uses the OpenAI Moderation API to analyze text.

#%%
def analyze_text(text, api_key):
    """
    Analyze text using OpenAI Moderation API

    Args:
        text (str): Text to analyze
        api_key (str): OpenAI API key

    Returns:
        dict: Raw API response with moderation categories
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.moderations.create(
            model="text-moderation-latest",
            input=text
        )

        # Extract the categories and scores from the first result
        categories = response.results[0].categories
        category_scores = response.results[0].category_scores

        return {
            "categories": {k: v for k, v in categories.model_dump().items()},
            "category_scores": {k: float(v) if v is not None else 0.0 for k, v in category_scores.model_dump().items()}
        }
    except Exception as e:
        return {"error": str(e)}

#%% [markdown]
# # Process OpenAI Moderation Response
# This function processes the raw response from the OpenAI moderation API and maps it to primary categories.

#%%
def process_openai_moderation_response(response):
    """
    Process OpenAI moderation response and map to primary categories.

    Args:
        response (dict): Raw response from OpenAI moderation API

    Returns:
        dict: Processed results with primary category scores
    """
    if "error" in response:
        return {"error": response["error"]}

    # Initialize scores for each primary category
    processed_results = {
        "clean": 1.0,
        "hate_or_discrimination": 0.0,
        "violence_or_threats": 0.0,
        "offensive_language": 0.0,
        "nsfw_content": 0.0,
        "spam_or_scams": 0.0,
    }

    # Category mapping from OpenAI to our primary categories
    category_mapping = {
        # Hate or discrimination
        "hate": "hate_or_discrimination",
        "hate/threatening": "hate_or_discrimination",
        "harassment": "hate_or_discrimination",
        "harassment/threatening": "hate_or_discrimination",

        # Violence or threats
        "violence": "violence_or_threats",
        "violence/graphic": "violence_or_threats",
        "illicit/violent": "violence_or_threats",
        "self-harm": "violence_or_threats",
        "self-harm/intent": "violence_or_threats",
        "self-harm/instructions": "violence_or_threats",

        # NSFW content
        "sexual": "nsfw_content",
        "sexual/minors": "nsfw_content",

        # todo: there is no category for offensive_language in openai fix metrics for that
        # todo: illicit does not mean scam or spam openai does not have a category for it fix metrics for this
        # Other categories that might indicate spam/scams
        "illicit": "spam_or_scams"
    }

    try:
        scores = response["category_scores"]

        # Map OpenAI categories to our primary categories
        for openai_category, score in scores.items():
            mapped_category = category_mapping.get(openai_category)
            if mapped_category:
                current_score = processed_results[mapped_category]
                # Ensure score is a float and handle None values
                score_value = float(score) if score is not None else 0.0
                processed_results[mapped_category] = max(current_score, score_value)

        # If any harmful category has a significant score, reduce the clean score
        harmful_score = max(
            processed_results["hate_or_discrimination"],
            processed_results["violence_or_threats"],
            processed_results["offensive_language"],
            processed_results["nsfw_content"],
            processed_results["spam_or_scams"]
        )
        processed_results["clean"] = 1.0 - harmful_score

        # Get the predicted category (highest scoring category)
        predicted_category = max(processed_results.items(), key=lambda x: x[1])[0]

        return {
            "scores": processed_results,
            "predicted_category": predicted_category,
            "predicted_score": processed_results[predicted_category]
        }
    except Exception as e:
        return {"error": str(e)}

#%%[markdown]
# # Run Benchmarking Functions

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
        raw_response = analyze_text(text, config["tokens"]["OPENAI_API_KEY"])
        processed_response = process_openai_moderation_response(raw_response)

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
                cache_dir / f"openai_results_batch_{batch_num + 1}_of_{total_batches}.jsonl",
                orient='records',
                lines=True
            )

        logger.info(f"Batch {batch_num + 1}/{total_batches} completed in {batch_processing_time:.2f} seconds")

        # Add a small delay between batches to prevent API rate limiting
        time.sleep(2)  # Increased delay for OpenAI API rate limits

    return pd.DataFrame(results)

#%%[markdown]
# # Execute Benchmarking

#%%
# Run the benchmark
openai_results_dir = DATA_ROOT / "benchmark_results" / "openai"
os.makedirs(openai_results_dir, exist_ok=True)

df_benchmark = pd.read_json(DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", lines=True)
logger.info("Starting benchmarking run...")
# df_benchmark = df_benchmark.sample(1000)  # sample for testing
df_results = run_benchmark(df_benchmark, output_dir=openai_results_dir, batch_size=100)

#%%[markdown]
# # Save Results

#%%
# Save the complete results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = openai_results_dir / f"openai_benchmark_results_{timestamp}.jsonl"

df_results.to_json(
    output_path,
    orient='records',
    lines=True
)

logger.info(f"Benchmarking complete. Results saved to {output_path}")

#%%[markdown]
# # Basic Results Analysis

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
