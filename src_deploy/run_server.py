#!/usr/bin/env python3
"""
Script to launch the sglang server with configured parameters.
"""
import argparse
import subprocess
import os
import sys

import sys
import subprocess


def check_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        print("nvidia-smi output:")
        print(output.decode("utf-8"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "nvidia-smi failed or not found - but we'll continue if PyTorch detects GPU"
        )
        return False


def check_cuda_libraries():
    try:
        output = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        print("Checking for CUDA libraries:")

        cuda_libs = ["libcuda.so", "libcudart.so", "libnvidia-ml.so"]
        for lib in cuda_libs:
            if lib in output:
                print(f"✅ {lib} found")
            else:
                print(f"❌ {lib} NOT found")

        # We'll just report, not fail if libs are missing
        return True
    except subprocess.CalledProcessError:
        print(
            "Failed to check CUDA libraries - but we'll continue if PyTorch detects GPU"
        )
        return True


def check_torch_cuda():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                )
        return cuda_available
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")
        return False


def setup_huggingface_auth():
    """Set up authentication with Hugging Face if token is available"""
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print(
            f"HF_TOKEN is set. Value: {hf_token[:4]}{'*' * (len(hf_token) - 8)}{hf_token[-4:]}"
        )
        try:
            from huggingface_hub import login

            login(token=hf_token)
            print("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Error logging in to Hugging Face Hub: {e}", file=sys.stderr)
    else:
        print(
            "HF_TOKEN environment variable not set. Skipping Hugging Face authentication."
        )


def main():
    # Set up Hugging Face authentication
    setup_huggingface_auth()

    # Default configuration
    default_config = {
        "model_path": "microsoft/Phi-3.5-mini-instruct",
        "host": "0.0.0.0",
        "port": "8899",
        "api_key": "None",
        "mem_fraction_static": "0.9",
        "max_running_requests": "1024",
        "attention_backend": "triton",
        "disable_cuda_graph": True,
        "dtype": "float16",
        "chunked_prefill_size": "512",
        "enable_metrics": True,
        "show_time_cost": True,
        "enable_cache_report": True,
        "log_level": "info",
        "watchdog_timeout": "120",
        "schedule_policy": "lpm",
        "schedule_conservativeness": "0.8",
    }

    # Parse command-line arguments to override defaults
    parser = argparse.ArgumentParser(
        description="Launch sglang server with configurable parameters"
    )
    parser.add_argument(
        "--model-path",
        default=default_config["model_path"],
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--host", default=default_config["host"], help="Host IP to bind the server to"
    )
    parser.add_argument(
        "--port", default=default_config["port"], help="Port to run the server on"
    )
    parser.add_argument(
        "--api-key",
        default=default_config["api_key"],
        help="API key for authentication",
    )
    parser.add_argument(
        "--mem-fraction-static",
        default=default_config["mem_fraction_static"],
        help="Static memory fraction",
    )
    parser.add_argument(
        "--max-running-requests",
        default=default_config["max_running_requests"],
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--attention-backend",
        default=default_config["attention_backend"],
        help="Backend for attention operations",
    )
    parser.add_argument(
        "--dtype",
        default=default_config["dtype"],
        help="Data type to use for inference",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        default=default_config["chunked_prefill_size"],
        help="Size for chunked prefill",
    )
    parser.add_argument(
        "--log-level", default=default_config["log_level"], help="Logging level"
    )
    parser.add_argument(
        "--watchdog-timeout",
        default=default_config["watchdog_timeout"],
        help="Watchdog timeout in seconds",
    )
    parser.add_argument(
        "--schedule-policy",
        default=default_config["schedule_policy"],
        help="Scheduling policy",
    )
    parser.add_argument(
        "--schedule-conservativeness",
        default=default_config["schedule_conservativeness"],
        help="Schedule conservativeness value",
    )

    # Add boolean flags
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        default=default_config["disable_cuda_graph"],
        help="Disable CUDA graph",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=default_config["enable_metrics"],
        help="Enable metrics collection",
    )
    parser.add_argument(
        "--show-time-cost",
        action="store_true",
        default=default_config["show_time_cost"],
        help="Show time cost measurements",
    )
    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=default_config["enable_cache_report"],
        help="Enable cache reporting",
    )

    args = parser.parse_args()

    # Build command to launch sglang server
    cmd = [sys.executable, "-m", "sglang.launch_server"]

    # Add all arguments
    cmd.extend(["--model-path", args.model_path])
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", args.port])
    cmd.extend(["--api-key", args.api_key])
    cmd.extend(["--mem-fraction-static", args.mem_fraction_static])
    cmd.extend(["--max-running-requests", args.max_running_requests])
    cmd.extend(["--attention-backend", args.attention_backend])
    cmd.extend(["--dtype", args.dtype])
    cmd.extend(["--chunked-prefill-size", args.chunked_prefill_size])
    cmd.extend(["--log-level", args.log_level])
    cmd.extend(["--watchdog-timeout", args.watchdog_timeout])
    cmd.extend(["--schedule-policy", args.schedule_policy])
    cmd.extend(["--schedule-conservativeness", args.schedule_conservativeness])

    # Add boolean flags
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if args.enable_metrics:
        cmd.append("--enable-metrics")
    if args.show_time_cost:
        cmd.append("--show-time-cost")
    if args.enable_cache_report:
        cmd.append("--enable-cache-report")

    print(f"Launching sglang server with command: {' '.join(cmd)}")

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running sglang server: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    nvidia_smi_ok = check_nvidia_smi()
    cuda_libs_ok = check_cuda_libraries()
    torch_cuda_ok = check_torch_cuda()

    # Only fail if PyTorch can't see CUDA
    if torch_cuda_ok:
        print("✅ PyTorch can access GPU - proceeding with server startup!")
    else:
        print("❌ PyTorch cannot access GPU - server will fail without GPU!")
        sys.exit(1)

    main()
