"""
SGLang Server Manager for LLM and Embedding servers

This module manages the SGLang servers for LLM and embedding,
optimized for T4 GPUs.
"""

import os
import signal
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class ServerManager:
    """Manager for SGLang servers (LLM and embedding)"""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        llm_model: str = "microsoft/Phi-3.5-mini-instruct",
        llm_port: int = 8899,
        emb_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        emb_port: int = 8890,
        host: str = "0.0.0.0",
        api_key: str = "None",
        mem_fraction_llm: float = 0.80,
        mem_fraction_emb: float = 0.25,
        max_requests: int = 32,
        llm_gpu_id: int = 0,
        emb_gpu_id: int = 0,
    ):
        """
        Initialize server manager

        Args:
            hf_token: Hugging Face token for downloading models
            llm_model: LLM model to use
            llm_port: Port for LLM server
            emb_model: Embedding model to use
            emb_port: Port for embedding server
            host: Host to bind to
            api_key: API key for authentication
            mem_fraction_llm: Fraction of GPU memory to use for LLM
            mem_fraction_emb: Fraction of GPU memory to use for embedding
            max_requests: Maximum number of concurrent requests
            llm_gpu_id: GPU ID to use for LLM server
            emb_gpu_id: GPU ID to use for embedding server
        """
        self.hf_token = hf_token
        self.llm_model = llm_model
        self.llm_port = llm_port
        self.emb_model = emb_model
        self.emb_port = emb_port
        self.host = host
        self.api_key = api_key
        self.mem_fraction_llm = mem_fraction_llm
        self.mem_fraction_emb = mem_fraction_emb
        self.max_requests = max_requests
        self.llm_gpu_id = llm_gpu_id
        self.emb_gpu_id = emb_gpu_id

        # Store processes
        self.processes = {}
        self.monitor_threads = {}

    def setup_environment(self):
        """Configure environment variables for better performance on GPU"""
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # Set HuggingFace token if available
        if self.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

    def build_server_command(
        self,
        model: str,
        port: int,
        host: str,
        api_key: str,
        mem_fraction: float,
        max_requests: int,
        is_embedding: bool = False,
        timeout_seconds: int = 60,
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
            "512",
            "--enable-metrics",
            "--show-time-cost",
            "--enable-cache-report",
            "--log-level",
            "info",
            # "--quantization fp8",

            #  will kill overly long batch generations
            "--watchdog-timeout",
            str(timeout_seconds),
        ]

        # Add embedding-specific settings
        if is_embedding:
            cmd.extend(["--is-embedding"])
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

    def launch_server(
        self, cmd: List[str], server_type: str
    ) -> Optional[subprocess.Popen]:
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

            timeout = 120
            if server_type == "Embedding":
                timeout = 120
            if server_type == "LLM":
                timeout = 180
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
                logger.error(
                    f"{server_type} server failed to start within timeout period"
                )
                output, _ = process.communicate()
                logger.error(f"Server output: {output}")
                process.terminate()
                return None

        except Exception as e:
            logger.error(f"Error launching {server_type} server: {e}")
            return None

    def monitor_server_output(self, process: subprocess.Popen, server_type: str):
        """Monitor and log server output in a separate thread"""

        def _monitor():
            for line in iter(process.stdout.readline, ""):
                logger.info(f"[{server_type}] {line.strip()}")

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        return thread

    def start_embedding_server(self, timeout_seconds: int = 60):
        """Start the embedding server"""
        # Set GPU for embedding server
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.emb_gpu_id)

        emb_cmd = self.build_server_command(
            model=self.emb_model,
            port=self.emb_port,
            host=self.host,
            api_key=self.api_key,
            mem_fraction=self.mem_fraction_emb,
            max_requests=self.max_requests,
            is_embedding=True,
            timeout_seconds=timeout_seconds
        )

        emb_process = self.launch_server(emb_cmd, "Embedding")
        if emb_process:
            self.processes["embedding"] = emb_process
            self.monitor_threads["embedding"] = self.monitor_server_output(
                emb_process, "Embedding"
            )
            logger.info(
                f"Embedding server running at http://{self.host}:{self.emb_port}"
            )
            # Wait to ensure embedding server is fully stabilized
            time.sleep(20)
            return True
        return False

    def start_llm_server(self, timeout_seconds: int = 120):
        """Start the LLM server"""
        # Set GPU for LLM server
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.llm_gpu_id)

        llm_cmd = self.build_server_command(
            model=self.llm_model,
            port=self.llm_port,
            host=self.host,
            api_key=self.api_key,
            mem_fraction=self.mem_fraction_llm,
            max_requests=self.max_requests,
            is_embedding=False,
            timeout_seconds=timeout_seconds
        )

        llm_process = self.launch_server(llm_cmd, "LLM")
        if llm_process:
            self.processes["llm"] = llm_process
            self.monitor_threads["llm"] = self.monitor_server_output(llm_process, "LLM")
            logger.info(f"LLM server running at http://{self.host}:{self.llm_port}")
            return True
        return False

    def start_servers(self, start_embedding=True, start_llm=True, emb_timeout=60, llm_timeout=120):
        """Start the requested servers with timeouts"""
        self.setup_environment()

        success = True

        # Launch embedding server first (it's smaller)
        if start_embedding:
            success = success and self.start_embedding_server(timeout_seconds=emb_timeout)

        # Launch LLM server
        if start_llm:
            success = success and self.start_llm_server(timeout_seconds=llm_timeout)

        # Check if any servers were started successfully
        if not self.processes:
            logger.error("No servers were started successfully.")
            return False

        return success

    def stop_servers(self):
        """Stop all running servers"""
        logger.info("Shutting down servers...")
        for server_type, process in self.processes.items():
            logger.info(f"Terminating {server_type} server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"{server_type} server did not terminate gracefully, killing..."
                )
                process.kill()

        self.processes = {}
        self.monitor_threads = {}
        logger.info("All servers shutdown complete")
        return True

    def run_servers(self, start_embedding=True, start_llm=True, emb_timeout=60, llm_timeout=120):
        """Run servers until interrupted"""
        if not self.start_servers(start_embedding, start_llm, emb_timeout, llm_timeout):
            return False

        # Set up signal handlers for graceful shutdown
        def handle_signal(sig, frame):
            self.stop_servers()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Keep the main thread running
        logger.info("\nServers are running. Press Ctrl+C to stop.")
        try:
            while all(p.poll() is None for p in self.processes.values()):
                time.sleep(1)

            # If we get here, at least one process has exited
            for server_type, process in self.processes.items():
                if process.poll() is not None:
                    logger.error(
                        f"{server_type} server exited unexpectedly with code {process.returncode}"
                    )

            # Terminate remaining processes
            self.stop_servers()

        except KeyboardInterrupt:
            self.stop_servers()

        return True
