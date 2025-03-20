# %%
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
import yaml
from pathlib import Path

# %%

DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
GCP_CREDENTIALS_PATH = Path(config["secrets"]["GCP_CREDENTIALS_PATH"])

# %%
# Read embeddings data
df = pd.read_json(
    DATA_ROOT / "rag" / "gcp-embeddings.jsonl",
    lines=True,
)
print(df.columns)
df["embedding"] = df["embedding"].apply(lambda x: list(np.float64(x)))

# %%
# Get a random embedding
random_embedding = df.sample(1)["embedding"].iloc[0]
embedding_str = "[" + ", ".join(str(x) for x in random_embedding) + "]"

# Initialize GCP client
credentials = service_account.Credentials.from_service_account_file(
    GCP_CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(credentials=credentials)

# Construct and execute query
query = f"""
SELECT
  base.text,
  base.moderation_category,
  distance
FROM VECTOR_SEARCH(
  TABLE stage_test_tables.test_comment_mod_embeddings,
  'embedding',
  (SELECT ARRAY<FLOAT64>{embedding_str}),
  top_k => 5,
  distance_type => 'COSINE',
  options => '{{"fraction_lists_to_search": 0.1, "use_brute_force": false}}'
)
ORDER BY distance
LIMIT 5;
"""

print(query)

# %%
start_time = datetime.now()
results = client.query(query).to_dataframe()
print(
    f"Query execution time: {(datetime.now() - start_time).total_seconds():.2f} seconds"
)

print("\nSimilar texts:")
for _, row in results.iterrows():
    print(f"\nText: {row['text'][:100]}...")
    print(f"Category: {row['moderation_category']}")
    print(f"Distance: {row['distance']:.6f}")
