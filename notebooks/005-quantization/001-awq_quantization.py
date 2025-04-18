# %%
import os
import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login as hf_login
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Load config
DEV_CONFIG_PATH = "/root/content-moderation/dev_config.yml"
with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract config values
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]
HF_USERNAME = config["tokens"]["HF_USERNAME"]

# Login to Hugging Face
hf_login(HF_TOKEN)

from awq import AutoAWQForCausalLM
# %%
def quantize_model_awq(model_id="microsoft/Phi-3.5-mini-instruct", output_dir=None):
    """
    Quantize the model using AWQ
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / "quantized" / "phi-3.5-mini-instruct-awq"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading model for AWQ quantization: {model_id}")

    # Load model and prepare it for quantization
    awq_model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    logger.info(f"Starting AWQ quantization...")

    # Get a sample text for calibration
    sample_texts = [
        "Artificial intelligence has transformed the way we",
        "The future of sustainable energy depends on",
        "Machine learning algorithms can analyze data to",
        "Climate change poses significant challenges to",
        "Modern healthcare systems utilize technology to"
    ]

    # Quantize the model - we'll use 4 bits quantization with a group size of 128
    awq_model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,     # Use zero-point quantization
            "q_group_size": 128,    # Group size
            "w_bit": 4,             # Quantization bits (4 for int4)
            "version": "GEMM"       # Version parameter
        }
    )

    # Save the quantized model and tokenizer
    logger.info(f"Saving AWQ quantized model to: {output_dir}")
    awq_model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Also save AWQ config file that SGLang needs
    with open(os.path.join(output_dir, "quant_config.json"), "w") as f:
        import json
        json.dump({
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }, f)

    logger.info(f"AWQ quantized model saved successfully")
    return output_dir

def push_to_huggingface(model_path, model_name="phi-3.5-mini-instruct-4bit-awq"):
    """
    Push the quantized model to Hugging Face Hub
    """
    logger.info(f"Pushing AWQ model to Hugging Face Hub")

    # Create repository name with proper format
    repo_name = model_name
    repo_id = f"{HF_USERNAME}/{repo_name}"

    # Push using the built-in methods from transformers
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True)

    # Upload all files in the directory
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=f"Upload AWQ quantized Phi-3.5-mini-instruct model"
    )

    logger.info(f"Successfully pushed AWQ model to {repo_id}")
    return repo_id

if __name__ == "__main__":
    # Quantize the model
    model_path = quantize_model_awq()

    # Push to HuggingFace
    hf_repo = push_to_huggingface(model_path)

    # Print repository URL
    print(f"Hugging Face Repository: https://huggingface.co/{hf_repo}")

    # Print SGLang launch command
    print("\nTo launch with SGLang, use:")
    print(f"python -m sglang.launch_server --model-path {hf_repo} --quantization awq --dtype bfloat16 --attention-backend flashinfer")