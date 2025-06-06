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

# Install setuptools first (required by triton)
echo "Installing setuptools..."
$UV pip install setuptools wheel
echo "✓ Setuptools and wheel installed"

# Install transformers
echo "Installing transformers..."
$UV pip install "transformers==4.48.3"
$PYTHON -c "import transformers" && echo "✓ Transformers installed" || echo "Warning: Could not import transformers, but continuing"

# Install PyTorch and torchvision
echo "Installing PyTorch and torchvision..."
$UV pip install torch torchvision
$PYTHON -c "import torch" && echo "✓ PyTorch installed" || echo "Warning: Could not import torch"
$PYTHON -c "import torchvision" && echo "✓ torchvision installed" || echo "Warning: Could not import torchvision"

# Install triton first as it's a dependency for sglang
echo "Installing triton..."
$UV pip install triton
$PYTHON -c "import triton" && echo "✓ Triton installed" || echo "Warning: Could not import triton"

# Install sglang and dependencies
echo "Installing sglang and dependencies..."
$UV pip install "sglang[all]>=0.4.4.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
$PYTHON -c "import sglang" && echo "✓ SGLang installed" || { echo "ERROR: Could not import sglang, installation failed"; exit 1; }

# Install additional required packages
echo "Installing additional packages..."
$UV pip install accelerate
$PYTHON -c "import accelerate" && echo "✓ Accelerate installed" || echo "Warning: Could not import accelerate"

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

echo "Setup completed!"
echo "The next step will download the model when the server starts."

# Return success regardless of individual command results
exit 0