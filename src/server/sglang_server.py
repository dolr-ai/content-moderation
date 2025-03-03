#!/usr/bin/env python3
"""
SGLang Server Launcher for LLM and Embedding models.

This module provides functionality to launch and manage SGLang servers
for both LLM inference and embedding generation.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import threading
from typing import Optional, List, Dict, Union
import logging
from pathlib import Path

# Import the configuration
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Use relative or absolute imports based on how the script is being run
if __name__ == "__main__" or "src" not in __name__:
    # Running as script or from outside the package
    from src.config.config import config
else:
    # Running from within the package
    from ..config.config import config


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch SGLang servers for LLM and/or embedding models."
    )

    # Server type flags
    parser.add_argument("--llm", action="store_true", help="Launch the LLM server")
    parser.add_argument(
        "--embedding", action="store_true", help="Launch the embedding server"
    )

    # Port configuration
    parser.add_argument(
        "--llm-port", type=int, default=8899, help="Port for the LLM server"
    )
    parser.add_argument(
        "--emb-port", type=int, default=8890, help="Port for the embedding server"
    )

    # Model configuration
    parser.add_argument(
        "--llm-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Model name for the LLM server",
    )
    parser.add_argument(
        "--emb-model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Model name for the embedding server",
    )

    # Hardware configuration
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=100,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to",
    )

    # API key
    parser.add_argument(
        "--api-key",
        type=str,
        default="None",
        help="API key for the server (optional)",
    )

    return parser.parse_args()


def setup_environment():
    """Set up the environment variables for SGLang."""
    # Get HF token from config
    hf_token = config.get_hf_token()

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        logger.info("Set HF_TOKEN from configuration")
    else:
        logger.warning("HF_TOKEN not found in configuration")

    # Set CUDA_VISIBLE_DEVICES if needed
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("Set CUDA_VISIBLE_DEVICES=0")


def check_gpu_info():
    """Check GPU information and availability."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA device(s)")

            for i in range(gpu_count):
                device = torch.cuda.get_device_properties(i)
                logger.info(
                    f"GPU {i}: {device.name}, "
                    f"Memory: {device.total_memory / 1e9:.2f} GB"
                )

            return True
        else:
            logger.warning("CUDA is not available. Running in CPU mode.")
            return False

    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU info.")
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
    """
    Build the command to launch a SGLang server.

    Args:
        model: Model name or path
        port: Port number
        host: Host address
        api_key: API key (optional)
        mem_fraction: Fraction of GPU memory to use
        max_requests: Maximum number of concurrent requests
        is_embedding: Whether this is an embedding server

    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--port",
        str(port),
        "--host",
        host,
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_requests),
        "--api-key",
        api_key,
    ]

    # Add embedding flag if needed
    if is_embedding:
        cmd.append("--is-embedding")

    return cmd


def launch_server(cmd: List[str], server_type: str) -> Optional[subprocess.Popen]:
    """
    Launch a server process.

    Args:
        cmd: Command to execute
        server_type: Type of server (for logging)

    Returns:
        Process object if successful, None otherwise
    """
    try:
        logger.info(f"Launching {server_type} server with command: {' '.join(cmd)}")

        # Create a process with pipe for stdout and stderr
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start monitoring the output in a separate thread
        monitor_server_output(process, server_type)

        # Wait a bit to see if the process crashes immediately
        time.sleep(2)

        if process.poll() is not None:
            # Process has already terminated
            returncode = process.poll()
            logger.error(
                f"{server_type} server failed to start (exit code {returncode})"
            )
            return None

        logger.info(f"{server_type} server started successfully (PID: {process.pid})")
        return process

    except Exception as e:
        logger.error(f"Error launching {server_type} server: {e}")
        return None


def monitor_server_output(process: subprocess.Popen, server_type: str):
    """
    Monitor the output of a server process in a separate thread.

    Args:
        process: Process to monitor
        server_type: Type of server (for logging)
    """

    def _monitor():
        """Thread function to monitor process output."""
        for line in process.stdout:
            logger.info(f"{server_type} server: {line.strip()}")

        for line in process.stderr:
            logger.error(f"{server_type} server error: {line.strip()}")

    # Start the monitoring thread
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()


def main():
    """Main function to launch the servers."""
    args = parse_args()

    # Check if at least one server type is specified
    if not (args.llm or args.embedding):
        logger.error("Please specify at least one server type (--llm or --embedding)")
        sys.exit(1)

    # Set up environment
    setup_environment()

    # Check GPU info
    check_gpu_info()

    # Get API key from args or config
    api_key = args.api_key or config.get_hf_token()

    # Dictionary to store processes
    processes = {}

    # Launch LLM server if requested
    if args.llm:
        llm_cmd = build_server_command(
            model=args.llm_model,
            port=args.llm_port,
            host=args.host,
            api_key=api_key,
            mem_fraction=args.mem_fraction,
            max_requests=args.max_requests,
            is_embedding=False,
        )

        llm_process = launch_server(llm_cmd, "LLM")
        if llm_process:
            processes["llm"] = llm_process
            logger.info(f"LLM server running at http://{args.host}:{args.llm_port}/v1")

    # Launch embedding server if requested
    if args.embedding:
        emb_cmd = build_server_command(
            model=args.emb_model,
            port=args.emb_port,
            host=args.host,
            api_key=api_key,
            mem_fraction=args.mem_fraction,
            max_requests=args.max_requests,
            is_embedding=True,
        )

        emb_process = launch_server(emb_cmd, "Embedding")
        if emb_process:
            processes["embedding"] = emb_process
            logger.info(
                f"Embedding server running at http://{args.host}:{args.emb_port}/v1"
            )

    if not processes:
        logger.error("Failed to start any servers")
        sys.exit(1)

    # Set up signal handler for graceful shutdown
    def handle_signal(sig, frame):
        logger.info("Received signal to shut down servers")

        for server_type, process in processes.items():
            if process.poll() is None:  # If process is still running
                logger.info(f"Terminating {server_type} server (PID: {process.pid})")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"{server_type} server did not terminate gracefully, killing..."
                    )
                    process.kill()

        logger.info("All servers shut down")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Keep the main thread alive
    try:
        while True:
            # Check if any process has terminated
            for server_type, process in list(processes.items()):
                if process.poll() is not None:
                    returncode = process.poll()
                    logger.error(
                        f"{server_type} server terminated unexpectedly "
                        f"(exit code {returncode})"
                    )
                    del processes[server_type]

            # Exit if all processes have terminated
            if not processes:
                logger.error("All servers have terminated, exiting")
                sys.exit(1)

            time.sleep(1)

    except KeyboardInterrupt:
        # This should be caught by the signal handler
        pass


if __name__ == "__main__":
    main()
