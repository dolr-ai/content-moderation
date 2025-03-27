#!/bin/bash

# Don't exit immediately on errors during build phase
set +e

echo "Starting setup for sglang server on A10 GPU..."

# Block 1: Skip GPU Check during build
echo "########################################################"
echo "Note: Skipping GPU checks during build phase. Will check when container runs."
echo "You might see a warning about GPU like: "
echo "Can't initialize NVML. OR No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda' "
echo "Please ignore these warnings. They are expected."
echo "########################################################"

# Block 2: Python environment is already created by Dockerfile
echo "Using uv for package installation..."

# Make sure we're using the venv from Dockerfile
if [ -d "$HOME/.venv" ]; then
    echo "Using existing virtual environment"
    source $HOME/.venv/bin/activate
    PYTHON="python"
else
    echo "Warning: Virtual environment not found, creating one now"
    uv venv $HOME/.venv
    source $HOME/.venv/bin/activate
    PYTHON="python"
fi

# Block 3: Python Dependencies using uv
echo "Installing Python packages..."

# First install setuptools - critical for triton
echo "Installing setuptools and wheel first..."
uv pip install -U setuptools wheel

# Install packages in parallel with efficient dependency resolution
echo "Installing core packages..."
uv pip install -U "transformers==4.48.3" triton "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

echo "Installing additional packages..."
uv pip install -U accelerate bitsandbytes huggingface_hub

# Validate installations
$PYTHON -c "import setuptools" && echo "✓ Setuptools installed" || echo "ERROR: Could not import setuptools"
$PYTHON -c "import transformers" && echo "✓ Transformers installed" || echo "Warning: Could not import transformers, but continuing"
$PYTHON -c "import triton" && echo "✓ Triton installed" || echo "Warning: Could not import triton"
$PYTHON -c "import sglang" && echo "✓ SGLang installed" || { echo "ERROR: Could not import sglang, installation failed"; exit 1; }
$PYTHON -c "import accelerate" && echo "✓ Accelerate installed" || echo "Warning: Could not import accelerate"
$PYTHON -c "import huggingface_hub" && echo "✓ Huggingface_hub installed" || echo "Warning: Could not import huggingface_hub"

# Install requirements.txt packages efficiently
if [ -f ~/requirements.txt ]; then
    echo "Installing requirements.txt packages..."
    uv pip install -r ~/requirements.txt
    echo "✓ Application dependencies installed"
else
    echo "No requirements.txt found, skipping"
fi

# Block 4: Skip CUDA Check during build
echo "Note: Skipping CUDA check during build phase. Will check when container runs."

echo "Setup completed!"
echo "The next step will download the model when the server starts."

# Return success regardless of individual command results
exit 0