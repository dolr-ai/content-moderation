#!/usr/bin/env python3
"""
SGLang Server Launcher for T4 GPUs

This script launches either an LLM server or an embedding server (or both)
using SGLang, optimized for T4 GPUs.

Usage:
    python sglang_servers.py [--llm] [--embedding] [--llm-port PORT] [--emb-port PORT] [--llm-model MODEL] [--emb-model MODEL]

Examples:
    # Launch both servers with default settings
    python sglang_servers.py --llm --embedding

    # Launch only the LLM server
    python sglang_servers.py --llm

    # Launch only the embedding server
    python sglang_servers.py --embedding

    # Launch both with custom ports
    python sglang_servers.py --llm --embedding --llm-port 8899 --emb-port 8890
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import threading
from typing import Optional, List, Dict
import logging
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Try to load configuration from YAML if available
DEV_CONFIG_PATH = "/root/content-moderation/dev_config.yml"
HF_TOKEN = None

try:
    with open(DEV_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        HF_TOKEN = config.get("tokens", {}).get("HF_TOKEN")
        if HF_TOKEN:
            logger.info("Loaded HF_TOKEN from config file")
except Exception as e:
    logger.warning(f"Could not load config file: {e}")
    logger.warning("Continuing without config file")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch SGLang servers for LLM and/or embedding on T4 GPU"
    )

    # Server selection
    parser.add_argument("--llm", action="store_true", help="Launch LLM server")
    parser.add_argument(
        "--embedding", action="store_true", help="Launch embedding server"
    )

    # LLM server settings
    parser.add_argument(
        "--llm-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="LLM model to use (default: microsoft/Phi-3.5-mini-instruct)",
    )
    parser.add_argument(
        "--llm-port", type=int, default=8899, help="Port for LLM server (default: 8899)"
    )

    # Embedding server settings
    parser.add_argument(
        "--emb-model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Embedding model to use (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)",
    )
    parser.add_argument(
        "--emb-port",
        type=int,
        default=8890,
        help="Port for embedding server (default: 8890)",
    )

    # Common settings
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
        "--hf-token",
        type=str,
        default=HF_TOKEN,
        help="HuggingFace token for downloading models",
    )

    # Advanced settings
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

    args = parser.parse_args()

    # If neither server is specified, show help and exit
    if not args.llm and not args.embedding:
        parser.print_help()
        sys.exit(1)

    return args


def setup_environment():
    """Configure environment variables for better performance on T4 GPU"""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set PyTorch to more efficiently manage memory on T4
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Set a specific CUDA device if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set HuggingFace token if available
    if HF_TOKEN:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


def check_gpu_info():
    """Check GPU information and print helpful details"""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA device(s)")

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.2f} GB")

            # Print CUDA version
            logger.info(f"CUDA Version: {torch.version.cuda}")

            # Check current memory usage
            if hasattr(torch.cuda, "memory_allocated"):
                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"Memory Allocated: {mem_allocated:.2f} GB")
                logger.info(f"Memory Reserved: {mem_reserved:.2f} GB")

            return True
        else:
            logger.warning("No CUDA devices available!")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU info: {e}")
        return False


def build_server_command(
    model: str,
    port: int,
    host: str,
    api_key: str,
    mem_fraction: float,
    max_requests: int,
    is_embedding: bool = False,
) -> List[str]:
    """Build the command to launch a SGLang server"""

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
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_requests),
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        "256",
        "--enable-metrics",
        "--show-time-cost",
        "--enable-cache-report",
        "--log-level",
        "info",
    ]

    # Add embedding-specific settings
    if is_embedding:
        cmd.extend(
            [
                "--is-embedding",
            ]
        )
    else:
        # LLM-specific settings
        cmd.extend(
            [
                "--schedule-policy",
                "lpm",
                "--schedule-conservativeness",
                "0.8",
            ]
        )

    return cmd


def launch_server(cmd: List[str], server_type: str) -> Optional[subprocess.Popen]:
    """Launch a server with the given command"""
    logger.info(f"Starting {server_type} server with command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Monitor server startup
        logger.info(f"Waiting for {server_type} server to initialize...")
        server_started = False

        # Increase timeout to 120 seconds (from 30)
        timeout = 90 if server_type == "Embedding" else 120
        for _ in range(timeout):
            if process.poll() is not None:
                # Process exited unexpectedly
                output, _ = process.communicate()
                logger.error(
                    f"{server_type} server process exited with code {process.returncode}"
                )
                logger.error(f"Server output: {output}")
                return None

            time.sleep(1)

            # Check if server is running by checking if the port is in use
            try:
                port = int(cmd[cmd.index("--port") + 1])
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("localhost", port))
                    if result == 0:
                        server_started = True
                        break
            except Exception as e:
                logger.warning(f"Error checking server status: {e}")

        if server_started:
            logger.info(f"{server_type} server started successfully!")
            return process
        else:
            logger.error(f"{server_type} server failed to start within timeout period")
            output, _ = process.communicate()
            logger.error(f"Server output: {output}")
            process.terminate()
            return None

    except Exception as e:
        logger.error(f"Error launching {server_type} server: {e}")
        return None


def monitor_server_output(process: subprocess.Popen, server_type: str):
    """Monitor and log server output in a separate thread"""

    def _monitor():
        for line in iter(process.stdout.readline, ""):
            logger.info(f"[{server_type}] {line.strip()}")

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return thread


def main():
    """Main function to set up and run the servers"""
    args = parse_args()
    setup_environment()

    # Print system information
    logger.info("=== System Information ===")
    check_gpu_info()
    logger.info("=========================")

    # Dictionary to store running processes
    processes = {}
    monitor_threads = {}

    # Adjust memory fractions for each server type
    if args.llm and args.embedding:
        # Give much more memory to LLM and minimal to embedding
        llm_fraction = 0.75
        emb_fraction = 0.25
        logger.info(
            f"Running both servers with LLM mem fraction: {llm_fraction}, Embedding mem fraction: {emb_fraction}"
        )
    else:
        # Single server gets most of the memory
        llm_fraction = emb_fraction = 0.75

    # Launch embedding server first (it's smaller)
    if args.embedding:
        emb_cmd = build_server_command(
            model=args.emb_model,
            port=args.emb_port,
            host=args.host,
            api_key=args.api_key,
            mem_fraction=emb_fraction,
            max_requests=16,  # Reduced from 32
            is_embedding=True,
        )

        emb_process = launch_server(emb_cmd, "Embedding")
        if emb_process:
            processes["embedding"] = emb_process
            monitor_threads["embedding"] = monitor_server_output(
                emb_process, "Embedding"
            )
            logger.info(
                f"Embedding server running at http://{args.host}:{args.emb_port}"
            )
            # Increased wait time to ensure embedding server is fully stabilized
            time.sleep(20)

    # Launch LLM server if requested
    if args.llm:
        llm_cmd = build_server_command(
            model=args.llm_model,
            port=args.llm_port,
            host=args.host,
            api_key=args.api_key,
            mem_fraction=llm_fraction,
            max_requests=16,
            is_embedding=False,
        )

        llm_process = launch_server(llm_cmd, "LLM")
        if llm_process:
            processes["llm"] = llm_process
            monitor_threads["llm"] = monitor_server_output(llm_process, "LLM")
            logger.info(f"LLM server running at http://{args.host}:{args.llm_port}")

    # Check if any servers were started successfully
    if not processes:
        logger.error("No servers were started successfully. Exiting.")
        sys.exit(1)

    # Set up signal handlers for graceful shutdown
    def handle_signal(sig, frame):
        logger.info("\nShutting down servers...")
        for server_type, process in processes.items():
            logger.info(f"Terminating {server_type} server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"{server_type} server did not terminate gracefully, killing..."
                )
                process.kill()
        logger.info("All servers shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Keep the main thread running
    logger.info("\nServers are running. Press Ctrl+C to stop.")
    try:
        while all(p.poll() is None for p in processes.values()):
            time.sleep(1)

        # If we get here, at least one process has exited
        for server_type, process in processes.items():
            if process.poll() is not None:
                logger.error(
                    f"{server_type} server exited unexpectedly with code {process.returncode}"
                )

        # Terminate remaining processes
        for server_type, process in processes.items():
            if process.poll() is None:
                logger.info(f"Terminating {server_type} server...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
    except KeyboardInterrupt:
        handle_signal(None, None)


if __name__ == "__main__":
    main()
