#!/usr/bin/env python3
"""
SGLang Server for Qwen2-1.5B-Instruct

This script launches an optimized SGLang server for the Alibaba Qwen2 1.5B Instruct model
on T4 GPUs. It's configured for generation tasks with varied sentence lengths.

Usage:
    python sglang_qwen2_server.py [--port PORT] [--host HOST] [--api-key API_KEY]



python3 './notebooks/004-rag_moderator/001-emb_sglang_server.py' --model "Alibaba-NLP/gte-Qwen2-1.5B-instruct" --port 8890 --host 0.0.0.0 --mem-fraction 0.75 --max-requests 32 --is-embedding
"""
import yaml
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional
import logging
from huggingface_hub import login as hf_login

# Load configuration from YAML
DEV_CONFIG_PATH = "/root/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set up paths and tokens
PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])
HF_TOKEN = config["tokens"]["HF_TOKEN"]


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch SGLang server for Qwen2-1.5B-Instruct"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Model to use (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8898,
        help="Port to run the server on (default: 8898)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="None",
        help="API key for authentication (default: None)",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.75,
        help="Fraction of GPU memory to use for static allocation (default: 0.75)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=32,
        help="Maximum number of concurrent requests (default: 32)",
    )
    parser.add_argument(
        "--is-embedding",
        type=bool,
        default=False,
        help="Whether to run as an embedding server (default: False)",
    )
    return parser.parse_args()


def setup_environment():
    """Configure environment variables for better performance on T4 GPU"""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set PyTorch to more efficiently manage memory on T4
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Set a specific CUDA device if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def launch_server(
    model: str,
    port: int, host: str, api_key: str, mem_fraction: float, max_requests: int
) -> Optional[subprocess.Popen]:
    """Launch the SGLang server with T4-optimized parameters for Qwen2-1.5B-Instruct"""

    # Ensure HuggingFace login before loading model
    try:
        hf_login(HF_TOKEN)
        logger.info("Successfully logged into Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
        return None

    print(f"Starting SGLang server with model: {model}")
    print(f"Server will be available at http://{host}:{port}")

    # Configure server command optimized for T4 GPU with Qwen2-1.5B
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--api-key",
        api_key,
        # Memory and performance settings optimized for T4
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_requests),
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",  # Helps with stability on T4
        "--dtype",
        "float16",  # Use fp16 for better memory efficiency
        # For varied sentence lengths, use FCFS instead of LPM
        "--schedule-policy",
        "fcfs",  # First-come-first-served scheduling
        "--schedule-conservativeness",
        "0.9",  # Be more conservative with scheduling
        # Prefill and chunk settings
        "--chunked-prefill-size",
        "1024",  # More memory-efficient chunk size
        # Logging and metrics
        "--enable-metrics",
        "--show-time-cost",
        "--enable-cache-report",
        "--log-level",
        "info",
    ]

    # Launch the server
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Set up signal handlers for graceful shutdown
        def handle_signal(sig, frame):
            print("\nShutting down SGLang server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            print("Server shutdown complete")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Monitor server startup
        print("Waiting for server to initialize...")
        server_started = False
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            if "Uvicorn running on" in line or "Application startup complete" in line:
                server_started = True
                break

        if server_started:
            print("\nServer started successfully!")
            return process
        else:
            print("Server failed to start properly")
            process.terminate()
            return None

    except Exception as e:
        print(f"Error launching server: {e}")
        return None


def monitor_server(process):
    """Keep the server running and monitor its output"""
    try:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
    except KeyboardInterrupt:
        print("\nShutting down SGLang server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server shutdown complete")
        sys.exit(0)


def check_gpu_info():
    """Check GPU information and print helpful details"""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA device(s)")

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.2f} GB")

            # Print CUDA version
            print(f"CUDA Version: {torch.version.cuda}")

            # Check current memory usage
            if hasattr(torch.cuda, "memory_allocated"):
                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"Memory Allocated: {mem_allocated:.2f} GB")
                print(f"Memory Reserved: {mem_reserved:.2f} GB")

            return True
        else:
            print("No CUDA devices available!")
            return False
    except Exception as e:
        print(f"Error checking GPU info: {e}")
        return False


def main():
    """Main function to set up and run the server"""
    args = parse_args()
    setup_environment()

    # Print system information
    print("=== System Information ===")
    check_gpu_info()
    print("=========================")

    # Launch server with optimized settings
    process = launch_server(
        model=args.model,
        port=args.port,
        host=args.host,
        api_key=args.api_key,
        mem_fraction=args.mem_fraction,
        max_requests=args.max_requests,
    )

    if process:
        print("\nServer is running. Press Ctrl+C to stop.")
        monitor_server(process)
    else:
        print("Failed to start the server. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
