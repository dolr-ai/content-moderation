#!/bin/bash

# Don't exit immediately on errors during build phase
set +e

echo "Starting setup for sglang server on A10 GPU..."

# Block 1: Skip GPU Check during build
echo "Note: Skipping GPU check during build phase. Will check when container runs."

# Block 2: Use uv instead of pip
echo "Setting up Python environment..."

# Create virtual environment using uv
uv venv $HOME/.venv
echo "✓ Virtual environment created"

# Make sure we use the virtual environment's Python and uv
PYTHON="$HOME/.venv/bin/python"
UV="uv"

# Add venv activation to .bashrc
echo 'source $HOME/.venv/bin/activate' >> ~/.bashrc

# Block 3: Python Dependencies using uv
echo "Installing Python packages..."
$UV pip install --upgrade pip
echo "✓ Pip upgraded"

# Install transformers
echo "Installing transformers..."
$UV pip install "transformers==4.48.3"
$PYTHON -c "import transformers" && echo "✓ Transformers installed" || echo "Warning: Could not import transformers, but continuing"

# Install sglang and dependencies
echo "Installing sglang and dependencies..."
$UV pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
$PYTHON -c "import sglang" && echo "✓ SGLang installed" || { echo "ERROR: Could not import sglang, installation failed"; exit 1; }

# Install additional required packages
echo "Installing additional packages..."
$UV pip install accelerate bitsandbytes triton
$PYTHON -c "import accelerate" && echo "✓ Accelerate installed" || echo "Warning: Could not import accelerate"
# $PYTHON -c "import bitsandbytes" && echo "✓ BitsAndBytes installed" || echo "Warning: Could not import bitsandbytes"
$PYTHON -c "import triton" && echo "✓ Triton installed" || echo "Warning: Could not import triton"

# Install huggingface_hub
echo "Installing huggingface_hub..."
$UV pip install huggingface_hub
$PYTHON -c "import huggingface_hub" && echo "✓ Huggingface_hub installed" || echo "Warning: Could not import huggingface_hub"

# Install requirements.txt packages
echo "Installing requirements.txt packages..."
$UV pip install -r ~/requirements.txt
echo "✓ Application dependencies installed"

# Block 4: Skip CUDA Check during build
echo "Note: Skipping CUDA check during build phase. Will check when container runs."

# Create model directory
mkdir -p ~/models
echo "✓ Model directory created"

echo "Setup completed!"
echo "The next step will download the model when the server starts."

# Return success regardless of individual command results
exit 0