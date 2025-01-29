# %%[markdown]
# # Boiler Plate API
# This notebook contains basic boiler plate code for API benchmarking.
# %%
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json
from huggingface_hub import login as hf_login
import yaml
from IPython.display import display
from pprint import pprint
from google.oauth2 import service_account


# %%
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
#%%
df_benchmark = pd.read_json(DATA_ROOT / "benchmark" / "benchmark_v1.jsonl", lines=True)
#%%
df_benchmark
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


credentials = authenticate_service_account(
    config["secrets"]["GCP_CREDENTIALS_PATH"]
)



#%%
from google.oauth2 import service_account
from google.cloud import language_v1

def analyze_text(text, credentials_path):
    """
    Simple function to analyze text using Google Cloud Natural Language API

    Args:
        text (str): Text to analyze
        credentials_path (str): Path to Google Cloud credentials file

    Returns:
        dict: Raw API response with sentiment and categories
    """
    # Authenticate
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Initialize client
    client = language_v1.LanguageServiceClient(credentials=credentials)

    # Prepare document
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="en"
    )

    try:
        # Call the moderateText endpoint
        response = client.moderate_text(
            request={"document": document}
        )

        # Format the response
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

# Example usage

# Replace with your credentials file path
CREDENTIALS_PATH = config["secrets"]["GCP_CREDENTIALS_PATH"]
print(CREDENTIALS_PATH)

ix = random.randint(0, len(df_benchmark))
text = df_benchmark['text'].iloc[ix]
actual_category = df_benchmark['moderation_category'].iloc[ix]
result = analyze_text(text, CREDENTIALS_PATH)
print(result)

#%%

df_benchmark['num_chars'] = df_benchmark['text'].apply(lambda x: len(x))
df_benchmark['num_chars'].sum()

2_838_888

#%%

def process_google_moderation_response(response):
    """
    Process Google Cloud moderation response and map to primary categories.

    Args:
        response (dict): Raw response from Google Cloud moderation API

    Returns:
        dict: Processed results with primary category scores
    """
    # Initialize scores for each primary category
    processed_results = {
        "clean": 1.0,  # Start with assumption of clean
        "hate_or_discrimination": 0.0,
        "violence_or_threats": 0.0,
        "offensive_language": 0.0,
        "nsfw_content": 0.0,
        "spam_or_scams": 0.0,
    }

    if "error" in response:
        return {"error": response["error"]}

    # Category mapping
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

    # Process each category from the response
    for category in response.get("moderation_categories", []):
        mapped_category = category_mapping.get(category["name"])
        if mapped_category:
            current_score = processed_results[mapped_category]
            # Take the maximum confidence score for each primary category
            processed_results[mapped_category] = max(current_score, category["confidence"])

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

# Example usage
# ix = random.randint(0, len(df_benchmark))
ix =5000
text = df_benchmark['text'].iloc[ix]
actual_category = df_benchmark['moderation_category'].iloc[ix]
raw_result = analyze_text(text, CREDENTIALS_PATH)
processed_result = process_google_moderation_response(raw_result)

print("\nText:", text)
print("\nActual category:", actual_category)
print("\nProcessed results:")
pprint(processed_result)


#%%
