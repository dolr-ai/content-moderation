# %%
import os
import requests
import openai
from sglang.utils import print_highlight
import json
import subprocess

# API key which is used in the setup_sglang.py file
API_KEY = "<add-your-api-key>"

client = openai.Client(
    base_url="http://<add-your-internal-ip>:8890/v1",
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model="microsoft/Phi-3.5-mini-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")


# %%

API_KEY = "7VltkUwKFrnyMPeC4bgyLxQAIfXhN6bV"
INTERNAL_IP = "192.168.8.210"  # Your internal IP
EXTERNAL_IP = "35.225.29.184"  # Your external IP
PORT = 8890


def test_connection(base_url):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.get(f"{base_url}/v1/models", headers=headers)
        print(f"Testing {base_url}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error connecting to {base_url}: {e}")


# Test different URLs
urls = [
    f"http://{INTERNAL_IP}:{PORT}",
    f"http://{EXTERNAL_IP}:{PORT}",
]

for url in urls:
    test_connection(url)

# %%
