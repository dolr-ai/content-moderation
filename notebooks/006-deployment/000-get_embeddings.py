# %% [markdown]
# # Import libraries
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
import logging
import json
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import yaml
from IPython.display import display
from pprint import pprint

# Import OpenAI client and create embedding client
from openai import OpenAI
from tqdm.auto import tqdm

DEV_CONFIG_PATH = "/root/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(config["remote"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["remote"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]

# huggingface login
hf_login(HF_TOKEN)

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# %% [markdown]
# # Load data


df = pd.read_json(f"{DATA_ROOT}/rag/vector_db_text.jsonl", lines=True)

# %% [markdown]
# # Get embeddings function


# Initialize embedding client
client = OpenAI(
    base_url="http://localhost:8890/v1",
    api_key="None"
)

def batch_create_embeddings(texts, batch_size=32, model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"):
    """
    Create embeddings for a batch of texts
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error creating embeddings for batch {i}: {e}")
            raise

    return embeddings
# %% [markdown]
# # Get embeddings

df_req = df[['text', 'moderation_category']].dropna().copy()
df_req['text'] = df_req['text'].str.strip().str[:2000]


# Get embeddings for all texts
texts = df_req['text'].tolist()
embeddings = batch_create_embeddings(texts)

# Add embeddings to dataframe
df_req['embedding'] = embeddings

# Verify embedding dimensions and do a quick validation
sample_idx = random.randint(0, len(embeddings)-1)
logger.info(f"Sample embedding dimension: {len(embeddings[sample_idx])}")
logger.info(f"Sample embedding first 5 values: {embeddings[sample_idx][:5]}")

# Save embeddings to JSONL
output_path = DATA_ROOT / "rag" / "embeddings.jsonl"
df_req.to_json(output_path, orient='records', lines=True, double_precision=15)

# Verify the saved data
df_verify = pd.read_json(output_path, lines=True)
logger.info("\nVerifying saved data:")
logger.info(f"Original shape: {df_req.shape}, Loaded shape: {df_verify.shape}")
logger.info(f"Sample embedding matches: {np.allclose(df_req.iloc[sample_idx]['embedding'], df_verify.iloc[sample_idx]['embedding'])}")

#%%

assert ((df_verify['text'] == df_req['text']).sum() == len(df_verify))
assert ((df_verify['moderation_category'] == df_req['moderation_category']).sum() == len(df_verify))

#%%
df_all = pd.concat(
    [
        df_verify.rename(columns={
            'text': 'verify_text',
            'moderation_category': 'verify_moderation_category',
            'embedding': 'verify_embedding'
        }),
        df_req.rename(columns={
            'text': 'req_text',
            'moderation_category': 'req_moderation_category',
            'embedding': 'req_embedding'
        })
    ],
    axis=1
)
#%%
# important to cast to float32 to avoid precision issues
assert df_all.apply(lambda x: np.array_equal(np.array(x['req_embedding'], dtype=np.float32), np.array(x['verify_embedding'], dtype=np.float32)), axis=1).sum() == len(df_all)
