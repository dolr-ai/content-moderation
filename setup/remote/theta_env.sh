#!/bin/bash

# Exit on any error
set -e

# Function to check last command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Block 1: GPU Check
echo "Checking GPU..."
nvidia-smi
check_status "GPU check"

# Block 2: UV Installation and Environment Setup
echo "Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
check_status "UV installation"

# Add UV to PATH and make it persistent (add both potential UV installation locations)
export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
# Also add to .profile to ensure it's available in all shells
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.profile

# Source both files to ensure immediate availability
source ~/.bashrc
source ~/.profile

# Verify UV installation more thoroughly
which uv || { echo "UV is not in PATH. Installation failed." >&2; exit 1; }
uv --version || { echo "UV installation is broken." >&2; exit 1; }

# Create and activate virtual environment
echo "Setting up virtual environment..."
uv venv .venv --python=3.10
check_status "Virtual environment creation"
source .venv/bin/activate
check_status "Virtual environment activation"

# Block 3: Python Dependencies
echo "Installing Python packages..."
uv pip install --upgrade pip
check_status "Pip upgrade"

# Install transformers first with a compatible version
# do NOT change transformers version else sglang will not work
uv pip install "transformers==4.48.3"
check_status "Transformers installation"

uv pip install sgl-kernel --force-reinstall --no-deps
check_status "SGL kernel installation"

uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
check_status "SGLang installation"
uv pip install jupyter pandas tqdm nvitop scikit-learn seaborn matplotlib faiss-gpu faiss-cpu bitsandbytes
uv pip install google-cloud-bigquery google-api-python-client google-cloud-storage google-cloud-bigquery-storage-api --upgrade
uv pip install db-dtypes==1.4.2
if [[ "$@" == *"--a100"* ]]; then
    uv pip install flash-attn --no-build-isolation
    uv pip install autoawq --no-build-isolation
    uv pip install accelerate
fi
check_status "Python packages installation"

# Block 4: CUDA Check
echo "Checking CUDA..."
python3 -c "import torch; print(torch.cuda.is_available())"
check_status "CUDA check"

# Block 5: System Dependencies
echo "Installing system dependencies..."
apt-get update
check_status "apt-get update"
apt-get install -y --no-install-recommends build-essential
check_status "build-essential installation"
export CC=gcc
apt-get install -y --no-install-recommends ninja-build python3.10-dev
check_status "ninja-build and python-dev installation"

# Block 6: NVIDIA CUDA Setup
echo "Setting up NVIDIA CUDA..."
apt-get update && apt-get install -y --no-install-recommends wget
check_status "wget installation"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
check_status "CUDA keyring download"
dpkg -i cuda-keyring_1.1-1_all.deb
check_status "CUDA keyring installation"
apt-get update
check_status "apt-get update after CUDA keyring"

# Install CUDA development tools
apt-get install -y --no-install-recommends cuda-nvcc-12-4 cuda-cudart-dev-12-4
check_status "CUDA tools installation"

# Block 7: Environment Variables
echo "Setting up environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Block 8: Final Checks
rm cuda-keyring_1.1-1_all.deb
echo "Performing final checks..."
echo "Internal IP: $(hostname -I)"
echo "External IP: $(curl -s ifconfig.me)"


mkdir -p ./data