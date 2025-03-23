#!/bin/bash

# Exit on any error
set -e

# Activate virtual environment
source $HOME/.venv/bin/activate

# Function to check GPU status
check_gpu() {
  echo "Checking GPU status..."
  if ! nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA GPU not detected. Make sure to run with '--gpus all' flag."
    exit 1
  fi

  # Print GPU info
  nvidia-smi

  # Verify CUDA with PyTorch
  echo "Verifying CUDA with PyTorch..."
  if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ CUDA is available: {torch.cuda.is_available()}'); print(f'✓ CUDA version: {torch.version.cuda}'); print(f'✓ GPU device(s): {torch.cuda.device_count()}'); print(f'✓ GPU name: {torch.cuda.get_device_name(0)}')"; then
    echo "ERROR: CUDA verification failed"
    exit 1
  fi
}

# Function to verify sglang installation
verify_sglang() {
  echo "Verifying sglang installation..."
  if ! python3 -c "import sglang; print(f'✓ SGLang version: {sglang.__version__}')"; then
    echo "ERROR: sglang installation verification failed"
    exit 1
  fi
}

# Main execution
echo "Starting sglang server setup..."
check_gpu
verify_sglang

echo "Starting sglang server with Phi-3.5-mini-instruct model..."
echo "Server will be available on port 8899"

# Start the sglang server
exec python3 -m sglang.launch_server --model-path microsoft/Phi-3.5-mini-instruct \
  --host 0.0.0.0 \
  --port 8899 \
  --api-key None \
  --mem-fraction-static 0.9 \
  --max-running-requests 1024 \
  --attention-backend triton \
  --disable-cuda-graph \
  --dtype float16 \
  --chunked-prefill-size 512 \
  --enable-metrics \
  --show-time-cost \
  --enable-cache-report \
  --log-level info \
  --watchdog-timeout 120 \
  --schedule-policy lpm \
  --schedule-conservativeness 0.8