# %% [markdown]
# # this is not working for SGLang skip quantization in bnb

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

DEV_CONFIG_PATH = "/root/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# %% [markdown]
# # Set configs
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]
HF_USERNAME = config["tokens"]["HF_USERNAME"]

# huggingface login
hf_login(HF_TOKEN)

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# %% [markdown]
# # Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %% [markdown]
# ## Load the base model tokenizer
# %%
def load_base_model_tokenizer(model_id="microsoft/Phi-3.5-mini-instruct"):
    """
    Load the base model tokenizer
    """
    logger.info(f"Loading base model tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Base model loaded successfully")
    return tokenizer

# Load the base model
tokenizer = load_base_model_tokenizer()

# %% [markdown]
# ## Quantize model to 4-bit

# %%
def quantize_model_4bit(model_id="microsoft/Phi-3.5-mini-instruct"):
    """
    Quantize the model to 4-bit using bitsandbytes
    """
    logger.info(f"Quantizing model to 4-bit: {model_id}")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load model with 4-bit quantization
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    logger.info(f"4-bit quantized model loaded successfully")
    return model_4bit

# %%
# Quantize model to 4-bit
model_4bit = quantize_model_4bit()

# %% [markdown]
# ## Quantize model to 8-bit

# %%
def quantize_model_8bit(model_id="microsoft/Phi-3.5-mini-instruct"):
    """
    Quantize the model to 8-bit using bitsandbytes
    """
    logger.info(f"Quantizing model to 8-bit: {model_id}")

    # Configure 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    # Load model with 8-bit quantization
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    logger.info(f"8-bit quantized model loaded successfully")
    return model_8bit

# %%
# Quantize model to 8-bit
model_8bit = quantize_model_8bit()

# %% [markdown]
# # Prepare models for SGLang compatibility

# %%
def save_model_for_sglang(model, tokenizer, output_dir, bits="4bit"):
    """
    Save the model in a format compatible with SGLang
    """
    output_path = Path(output_dir) / f"phi-3.5-mini-instruct-{bits}"
    logger.info(f"Saving {bits} model for SGLang at: {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Model saved successfully for SGLang compatibility")
    return output_path

# %%
# Save models for SGLang
PROJECT_ROOT = Path("/root/content-moderation")
output_dir = PROJECT_ROOT / "models" / "quantized"
model_4bit_path = save_model_for_sglang(model_4bit, tokenizer, output_dir, bits="4bit")
model_8bit_path = save_model_for_sglang(model_8bit, tokenizer, output_dir, bits="8bit")

# %% [markdown]
# # Push quantized models to Hugging Face Hub

# %%
def push_to_huggingface(model_path, model_type="4bit"):
    """
    Push the quantized model to Hugging Face Hub
    """
    logger.info(f"Pushing {model_type} model to Hugging Face Hub")

    # Create repository name with proper format
    repo_name = f"phi-3.5-mini-instruct-{model_type}-bnb"

    # Push to Hugging Face Hub
    model_path.push_to_hub(
        repo_id=f"{HF_USERNAME}/{repo_name}",
        private=True,
        commit_message=f"Upload {model_type} quantized Phi-3.5-mini-instruct model"
    )

    # Push tokenizer to the same repo
    tokenizer.push_to_hub(
        repo_id=f"{HF_USERNAME}/{repo_name}",
        private=True
    )

    logger.info(f"Successfully pushed {model_type} model to {HF_USERNAME}/{repo_name}")
    return f"{HF_USERNAME}/{repo_name}"

# %%
# Push 4-bit model to Hugging Face
hf_repo_4bit = push_to_huggingface(model_4bit, "4bit")

# %%
# Push 8-bit model to Hugging Face
hf_repo_8bit = push_to_huggingface(model_8bit, "8bit")

# %%
# Display repository URLs
print("Hugging Face Repositories:")
print(f"- 4-bit Model: https://huggingface.co/{hf_repo_4bit}")
print(f"- 8-bit Model: https://huggingface.co/{hf_repo_8bit}")

