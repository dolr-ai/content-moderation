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

# Function to verify installation
verify_installation() {
    local package=$1
    python3 -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: $package installation verification failed"
        exit 1
    else
        echo "✓ $package successfully installed"
    fi
}

echo "Starting setup for sglang server on A10 GPU..."

# Block 1: GPU Check
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    check_status "GPU check"
else
    echo "Note: nvidia-smi not available. GPU check will be performed when container runs with appropriate nvidia runtime"
fi

# Block 2: UV Installation and Environment Setup
echo "Installing UV package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
check_status "UV installation"

# Add UV to PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify UV installation
which uv
check_status "UV path verification"
uv --version
check_status "UV version check"

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
uv venv .venv --python=3.10
check_status "Virtual environment creation"
source .venv/bin/activate
check_status "Virtual environment activation"

# Add venv activation to .bashrc
echo 'source $HOME/.venv/bin/activate' >> ~/.bashrc

# Block 3: Python Dependencies
echo "Installing Python packages..."
uv pip install --upgrade pip
check_status "Pip upgrade"

# Install transformers first with a compatible version
echo "Installing transformers..."
uv pip install "transformers==4.48.3"
check_status "Transformers installation"
verify_installation "transformers"

# Install sglang and dependencies
echo "Installing sglang and dependencies..."
uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
check_status "SGLang installation"
verify_installation "sglang"

# Install additional required packages
echo "Installing additional packages..."
uv pip install accelerate bitsandbytes triton
check_status "Additional packages installation"
verify_installation "accelerate"
verify_installation "bitsandbytes"
verify_installation "triton"

# Block 4: CUDA Check
echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"
check_status "CUDA check"

# Create model directory
mkdir -p ~/models
check_status "Model directory creation"

echo "Setup completed successfully!"
echo "The next step will download the model when the server starts."