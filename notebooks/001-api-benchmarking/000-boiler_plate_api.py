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
        # Get sentiment analysis
        sentiment = client.analyze_sentiment(
            request={"document": document}
        ).document_sentiment

        # Get content classification
        categories = client.classify_text(
            request={"document": document}
        ).categories

        # Return raw API response
        return {
            "sentiment": {
                "score": sentiment.score,
                "magnitude": sentiment.magnitude
            },
            "categories": [
                {
                    "name": category.name,
                    "confidence": category.confidence
                }
                for category in categories
            ]
        }

    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Replace with your credentials file path
    CREDENTIALS_PATH = config["secrets"]["GCP_CREDENTIALS_PATH"]
    print(CREDENTIALS_PATH)

    text = "I love this wonderful product! The customer service was excellent."
    result = analyze_text(text, CREDENTIALS_PATH)
    print(result)