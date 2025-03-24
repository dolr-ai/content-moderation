#!/usr/bin/env python3
"""
Module to manage SGL.ai server instances for LLM and embedding services
"""

import os
import sys
import subprocess
import time
import signal
import atexit
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from config import config, reload_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables to track server processes
llm_server_process = None
embedding_server_process = None


def get_default_sglang_config() -> Dict[str, Any]:
    """
    Get configuration for SGLang servers from environment variables

    Returns:
        Dictionary with configuration values from environment
    """
    # All defaults are set in run_all.py's DEFAULT_CONFIG
    return {
        "llm_model": os.environ["LLM_MODEL"],
        "embedding_model": os.environ["EMBEDDING_MODEL"],
        "llm_host": os.environ["LLM_HOST"],
        "embedding_host": os.environ["EMBEDDING_HOST"],
        "llm_port": int(os.environ["LLM_PORT"]),
        "embedding_port": int(os.environ["EMBEDDING_PORT"]),
        "api_key": os.environ["SGLANG_API_KEY"],
        "llm_mem_fraction": float(os.environ["LLM_MEM_FRACTION"]),
        "embedding_mem_fraction": float(os.environ["EMBEDDING_MEM_FRACTION"]),
        "max_requests": os.environ["MAX_REQUESTS"],
        "attention_backend": os.environ.get("ATTENTION_BACKEND", "triton"),
        "dtype": os.environ.get("DTYPE", "float16"),
        "chunked_prefill_size": os.environ.get("CHUNKED_PREFILL_SIZE", "512"),
        "log_level": os.environ.get("LOG_LEVEL", "info"),
        "watchdog_timeout": os.environ.get("WATCHDOG_TIMEOUT", "60"),
        "schedule_policy": os.environ.get("SCHEDULE_POLICY", "lpm"),
        "schedule_conservativeness": os.environ.get("SCHEDULE_CONSERVATIVENESS", "0.8"),
    }


def build_command(
    model_path: str,
    host: str,
    port: int,
    is_embedding: bool = False,
    config: Dict[str, Any] = None,
) -> List[str]:
    """
    Build command to launch SGLang server

    Args:
        model_path: Path to the model
        host: Host to bind the server to
        port: Port to run the server on
        is_embedding: Whether this is an embedding server
        config: Additional configuration options

    Returns:
        List of command arguments
    """
    if config is None:
        config = get_default_sglang_config()

    cmd = [sys.executable, "-m", "sglang.launch_server"]

    # Add main arguments
    cmd.extend(["--model-path", model_path])
    cmd.extend(["--host", host])
    cmd.extend(["--port", str(port)])
    cmd.extend(["--api-key", config["api_key"]])

    # Use the appropriate memory fraction based on server type
    if is_embedding:
        cmd.extend(["--mem-fraction-static", str(config["embedding_mem_fraction"])])
    else:
        cmd.extend(["--mem-fraction-static", str(config["llm_mem_fraction"])])

    cmd.extend(["--max-running-requests", config["max_requests"]])
    cmd.extend(["--attention-backend", config["attention_backend"]])
    cmd.extend(["--dtype", config["dtype"]])
    cmd.extend(["--chunked-prefill-size", config["chunked_prefill_size"]])
    cmd.extend(["--log-level", config["log_level"]])
    cmd.extend(["--watchdog-timeout", config["watchdog_timeout"]])
    cmd.extend(["--schedule-policy", config["schedule_policy"]])
    cmd.extend(["--schedule-conservativeness", config["schedule_conservativeness"]])

    # Add boolean flags
    cmd.append("--disable-cuda-graph")
    cmd.append("--enable-metrics")
    cmd.append("--show-time-cost")
    cmd.append("--enable-cache-report")

    # Add embedding flag if this is an embedding server
    if is_embedding:
        cmd.append("--is-embedding")

    return cmd


def start_llm_server(model_path, port, api_key=None, mem_fraction=0.70):
    """Start the LLM server with the specified model and settings.

    Args:
        model_path: Path to the model to load
        port: Port to run the server on
        api_key: API key for the model, if any
        mem_fraction: Memory fraction to allocate to the model (default: 0.70)
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        os.environ["LLM_HOST"],
        "--port",
        str(port),
        "--api-key",
        str(api_key),
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        os.environ["MAX_REQUESTS"],
        "--attention-backend",
        "triton",
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        "512",
        "--log-level",
        "info",
        "--watchdog-timeout",
        "60",
        "--schedule-policy",
        "lpm",
        "--schedule-conservativeness",
        "0.8",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--show-time-cost",
        "--enable-cache-report",
    ]

    global llm_server_process

    try:
        # Log the command
        logger.info(f"Starting LLM server with command: {' '.join(command)}")

        # Start the LLM server process
        llm_server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy(),
        )

        # Start threads to handle server output
        log_process_output(llm_server_process, "LLM")

        logger.info(f"LLM server started with PID {llm_server_process.pid}")
        return llm_server_process

    except Exception as e:
        logger.error(f"Failed to start LLM server: {e}")
        return None


def start_embedding_server(model_path, port, api_key=None, mem_fraction=0.30):
    """Start the embedding server with the specified model and settings.

    Args:
        model_path: Path to the model to load
        port: Port to run the server on
        api_key: API key for the model, if any
        mem_fraction: Memory fraction to allocate to the model (default: 0.30)
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        os.environ["EMBEDDING_HOST"],
        "--port",
        str(port),
        "--api-key",
        str(api_key),
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        os.environ["MAX_REQUESTS"],
        "--attention-backend",
        "triton",
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        "512",
        "--log-level",
        "info",
        "--watchdog-timeout",
        "60",
        "--schedule-policy",
        "lpm",
        "--schedule-conservativeness",
        "0.8",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--show-time-cost",
        "--enable-cache-report",
        "--is-embedding",
    ]

    global embedding_server_process

    try:
        # Log the command
        logger.info(f"Starting embedding server with command: {' '.join(command)}")

        # Start the embedding server process
        embedding_server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy(),
        )

        # Start threads to handle server output
        log_process_output(embedding_server_process, "EMBEDDING")

        logger.info(f"Embedding server started with PID {embedding_server_process.pid}")
        return embedding_server_process

    except Exception as e:
        logger.error(f"Failed to start embedding server: {e}")
        return None


def log_process_output(process: subprocess.Popen, prefix: str) -> None:
    """
    Log process output with a prefix to distinguish between servers

    Args:
        process: Process to log output from
        prefix: Prefix to add to log messages
    """
    import threading
    import re

    def log_output(stream, default_log_func, prefix):
        # Regular expression to identify log levels
        log_level_regex = re.compile(r"\[(.*?)\]|\b(INFO|WARNING|ERROR|DEBUG)\b")

        for line in stream:
            line_str = line.strip()

            # Determine appropriate log level
            log_func = default_log_func
            match = log_level_regex.search(line_str)
            if match:
                level = match.group(1) or match.group(2)
                if level:
                    level = level.upper()
                    if "ERROR" in level or "EXCEPTION" in level or "FATAL" in level:
                        log_func = logger.error
                    elif "WARN" in level:
                        log_func = logger.warning
                    elif "INFO" in level:
                        log_func = logger.info
                    elif "DEBUG" in level:
                        log_func = logger.debug

            log_func(f"[{prefix}] {line_str}")

    # Start threads for stdout and stderr
    stdout_thread = threading.Thread(
        target=log_output, args=(process.stdout, logger.info, prefix), daemon=True
    )
    stderr_thread = threading.Thread(
        target=log_output, args=(process.stderr, logger.warning, prefix), daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()


def start_sglang_servers() -> (
    Tuple[Optional[subprocess.Popen], Optional[subprocess.Popen]]
):
    """
    Start both LLM and embedding servers

    Returns:
        Tuple of (LLM server process, embedding server process)
    """
    # Get configuration
    config = get_default_sglang_config()

    # Update environment variables for the FastAPI server to use the servers
    if not os.environ.get("LLM_URL"):
        os.environ["LLM_URL"] = f"http://{config['llm_host']}:{config['llm_port']}/v1"
        logger.info(f"Setting LLM_URL to {os.environ['LLM_URL']}")

    if not os.environ.get("EMBEDDING_URL"):
        os.environ["EMBEDDING_URL"] = (
            f"http://{config['embedding_host']}:{config['embedding_port']}/v1"
        )
        logger.info(f"Setting EMBEDDING_URL to {os.environ['EMBEDDING_URL']}")

    # Update config with new values
    reload_config()

    # Start LLM server first
    llm_process = start_llm_server(
        config["llm_model"],
        config["llm_port"],
        config["api_key"],
        config["llm_mem_fraction"],
    )

    # Give the LLM server time to start
    logger.info("Waiting for LLM server to initialize (process)...")
    wait_time = int(os.environ.get("LLM_INIT_WAIT_TIME", "180"))
    time.sleep(wait_time)

    # Check if LLM server process is still running
    if llm_process and llm_process.poll() is not None:
        logger.error(f"LLM server process exited with code {llm_process.returncode}")
        raise RuntimeError(
            f"LLM server failed to start with code {llm_process.returncode}"
        )

    # Start embedding server
    embedding_process = start_embedding_server(
        config["embedding_model"],
        config["embedding_port"],
        config["api_key"],
        config["embedding_mem_fraction"],
    )

    # Give embedding server time to start
    logger.info("Waiting for embedding server to initialize (process)...")
    wait_time = int(os.environ.get("EMBEDDING_INIT_WAIT_TIME", "180"))
    time.sleep(wait_time)

    # Check if embedding server process is still running
    if embedding_process and embedding_process.poll() is not None:
        logger.error(
            f"Embedding server process exited with code {embedding_process.returncode}"
        )
        raise RuntimeError(
            f"Embedding server failed to start with code {embedding_process.returncode}"
        )

    logger.info("Both servers started successfully")

    # Register cleanup function
    atexit.register(cleanup_servers)

    return llm_process, embedding_process


def cleanup_servers() -> None:
    """
    Clean up server processes on exit
    """
    global llm_server_process, embedding_server_process

    if llm_server_process:
        logger.info(f"Terminating LLM server (PID {llm_server_process.pid})")
        try:
            llm_server_process.terminate()
            llm_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"LLM server did not terminate gracefully, killing...")
            llm_server_process.kill()
        except Exception as e:
            logger.error(f"Error terminating LLM server: {e}")

    if embedding_server_process:
        logger.info(
            f"Terminating embedding server (PID {embedding_server_process.pid})"
        )
        try:
            embedding_server_process.terminate()
            embedding_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Embedding server did not terminate gracefully, killing...")
            embedding_server_process.kill()
        except Exception as e:
            logger.error(f"Error terminating embedding server: {e}")


# Handle signals to make sure cleanup happens
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_servers()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # This script is deprecated for direct use
    logger.warning("=" * 80)
    logger.warning("DEPRECATION WARNING")
    logger.warning("Direct execution of sglang_servers.py is deprecated.")
    logger.warning("Please use run_all.py instead for a more comprehensive setup.")
    logger.warning("=" * 80)

    # Still allow running it for testing/debugging
    user_input = input("Do you want to continue anyway? (y/n): ")
    if user_input.lower() != "y":
        logger.info("Exiting as requested.")
        sys.exit(0)

    # Import and use the DEFAULT_CONFIG from run_all to ensure consistent settings
    try:
        sys.path.append(str(Path(__file__).parent))
        from run_all import DEFAULT_CONFIG, setup_default_env

        setup_default_env()
        logger.info("Loaded default configuration from run_all.py")
    except ImportError:
        logger.warning(
            "Could not import DEFAULT_CONFIG from run_all.py, using environment variables"
        )
        # Set some basic defaults if run standalone
        for key, value in {
            "LLM_MODEL": "microsoft/Phi-3.5-mini-instruct",
            "EMBEDDING_MODEL": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "LLM_HOST": "0.0.0.0",
            "EMBEDDING_HOST": "0.0.0.0",
            "LLM_PORT": "8899",
            "EMBEDDING_PORT": "8890",
            "LLM_MEM_FRACTION": "0.70",
            "EMBEDDING_MEM_FRACTION": "0.70",
            "SGLANG_API_KEY": "None",
            "MAX_REQUESTS": "32",
        }.items():
            if key not in os.environ:
                os.environ[key] = value

    logger.info("Starting servers in standalone mode...")
    llm_proc, emb_proc = start_sglang_servers()

    # Keep the script running to keep the servers alive
    try:
        while True:
            time.sleep(1)

            # Check if processes are still running
            if llm_proc and llm_proc.poll() is not None:
                logger.error(f"LLM server exited with code {llm_proc.returncode}")
                break

            if emb_proc and emb_proc.poll() is not None:
                logger.error(f"Embedding server exited with code {emb_proc.returncode}")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        cleanup_servers()
