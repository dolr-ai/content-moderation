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
    Get default configuration for SGLang servers

    Returns:
        Dictionary with default configuration values
    """
    return {
        "llm_model": os.environ.get("LLM_MODEL", "microsoft/Phi-3.5-mini-instruct"),
        "embedding_model": os.environ.get(
            "EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        ),
        "llm_host": os.environ.get("LLM_HOST", "0.0.0.0"),
        "embedding_host": os.environ.get("EMBEDDING_HOST", "0.0.0.0"),
        "llm_port": int(os.environ.get("LLM_PORT", "8899")),
        "embedding_port": int(os.environ.get("EMBEDDING_PORT", "8890")),
        "api_key": os.environ.get("SGLANG_API_KEY", "None"),
        "mem_fraction": os.environ.get("MEM_FRACTION", "0.80"),
        "max_requests": os.environ.get("MAX_REQUESTS", "32"),
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
    cmd.extend(["--mem-fraction-static", config["mem_fraction"]])
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


def start_llm_server(config: Dict[str, Any] = None) -> Optional[subprocess.Popen]:
    """
    Start the LLM server

    Args:
        config: Configuration for the server

    Returns:
        Server process or None if failed
    """
    global llm_server_process

    if config is None:
        config = get_default_sglang_config()

    try:
        # Build command for LLM server
        llm_cmd = build_command(
            model_path=config["llm_model"],
            host=config["llm_host"],
            port=config["llm_port"],
            is_embedding=False,
            config=config,
        )

        # Log the command
        logger.info(f"Starting LLM server with command: {' '.join(llm_cmd)}")

        # Start the LLM server process
        llm_server_process = subprocess.Popen(
            llm_cmd,
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


def start_embedding_server(config: Dict[str, Any] = None) -> Optional[subprocess.Popen]:
    """
    Start the embedding server

    Args:
        config: Configuration for the server

    Returns:
        Server process or None if failed
    """
    global embedding_server_process

    if config is None:
        config = get_default_sglang_config()

    try:
        # Build command for embedding server
        embedding_cmd = build_command(
            model_path=config["embedding_model"],
            host=config["embedding_host"],
            port=config["embedding_port"],
            is_embedding=True,
            config=config,
        )

        # Log the command
        logger.info(
            f"Starting embedding server with command: {' '.join(embedding_cmd)}"
        )

        # Start the embedding server process
        embedding_server_process = subprocess.Popen(
            embedding_cmd,
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

    def log_output(stream, log_func, prefix):
        for line in stream:
            log_func(f"[{prefix}] {line.strip()}")

    # Start threads for stdout and stderr
    stdout_thread = threading.Thread(
        target=log_output, args=(process.stdout, logger.info, prefix), daemon=True
    )
    stderr_thread = threading.Thread(
        target=log_output, args=(process.stderr, logger.error, prefix), daemon=True
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
    # Get default configuration
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

    # Start both servers
    llm_process = start_llm_server(config)

    # Give the LLM server a moment to start
    time.sleep(2)

    embedding_process = start_embedding_server(config)

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
    # This allows running just the server setup separately
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
