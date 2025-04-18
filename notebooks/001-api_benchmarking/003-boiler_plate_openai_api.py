# %%[markdown]
# # OpenAI Moderation API Boilerplate
# This notebook contains basic boiler plate code for OpenAI API benchmarking.
# %%
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json
import yaml
from IPython.display import display
from pprint import pprint
from openai import OpenAI

# %%
# Define the primary category mapping (same as GCP version)
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

# %%
# Load benchmark data
df_benchmark = pd.read_json(DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", lines=True)

# %%
def analyze_text(text, api_key):
    """
    Simple function to analyze text using OpenAI Moderation API

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
            # Handle potential None values in scores
            "category_scores": {k: float(v) if v is not None else 0.0 for k, v in category_scores.model_dump().items()}
        }
    except Exception as e:
        return {"error": str(e)}

# %%
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

    # Updated category mapping from OpenAI to our primary categories
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

# %%
# Example usage
OPENAI_API_KEY = config["tokens"].get("OPENAI_API_KEY")

# Test with a random example
ix = random.randint(0, len(df_benchmark))
text = df_benchmark['text'].iloc[ix]
actual_category = df_benchmark['moderation_category'].iloc[ix]

raw_result = analyze_text(text, OPENAI_API_KEY)
processed_result = process_openai_moderation_response(raw_result)

print("\nText:", text)
print("\nActual category:", actual_category)
print("\nProcessed results:")
pprint(processed_result)

# %%
# Calculate total characters in dataset
df_benchmark['num_chars'] = df_benchmark['text'].apply(lambda x: len(x))
df_benchmark['num_chars'].sum()
