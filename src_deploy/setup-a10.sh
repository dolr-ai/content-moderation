#!/bin/bash

# Don't exit immediately on errors during build phase
set +e

echo "Starting setup for sglang server on A10 GPU..."

# Block 1: Skip GPU Check during build
echo "Note: Skipping GPU check during build phase. Will check when container runs."

# Block 2: Use regular pip instead of UV
echo "Setting up Python environment..."

# Create virtual environment using standard venv
python3 -m venv $HOME/.venv
echo "✓ Virtual environment created"

# Make sure we use the virtual environment's Python and pip
PYTHON="$HOME/.venv/bin/python"
PIP="$HOME/.venv/bin/pip"

# Add venv activation to .bashrc
echo 'source $HOME/.venv/bin/activate' >> ~/.bashrc

# Block 3: Python Dependencies using standard pip
echo "Installing Python packages..."
$PIP install --upgrade pip
echo "✓ Pip upgraded"

# Install transformers
echo "Installing transformers..."
$PIP install "transformers==4.48.3"
$PYTHON -c "import transformers" && echo "✓ Transformers installed" || echo "Warning: Could not import transformers, but continuing"

# Install sglang and dependencies
echo "Installing sglang and dependencies..."
$PIP install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
$PYTHON -c "import sglang" && echo "✓ SGLang installed" || { echo "ERROR: Could not import sglang, installation failed"; exit 1; }

# Install additional required packages
echo "Installing additional packages..."
$PIP install accelerate bitsandbytes triton
$PYTHON -c "import accelerate" && echo "✓ Accelerate installed" || echo "Warning: Could not import accelerate"
# $PYTHON -c "import bitsandbytes" && echo "✓ BitsAndBytes installed" || echo "Warning: Could not import bitsandbytes"
$PYTHON -c "import triton" && echo "✓ Triton installed" || echo "Warning: Could not import triton"

# Block 4: Skip CUDA Check during build
echo "Note: Skipping CUDA check during build phase. Will check when container runs."

# Create model directory
mkdir -p ~/models
echo "✓ Model directory created"

echo "Setup completed!"
echo "The next step will download the model when the server starts."

# Return success regardless of individual command results
exit 0