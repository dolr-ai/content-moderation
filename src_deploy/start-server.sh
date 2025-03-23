#!/bin/bash

# Activate the virtual environment
source $HOME/.venv/bin/activate

# Check GPU at runtime
echo "Checking GPU at runtime..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    if [ $? -ne 0 ]; then
        echo "Error: GPU not available"
        # Continue anyway, but warn
        echo "Warning: Continuing without GPU"
    else
        echo "✓ GPU available"

        # Verify CUDA is working with PyTorch
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"
    fi
else
    echo "Warning: nvidia-smi not available. Continuing without GPU verification."
fi

# Start the server (add your actual server start command)
echo "Starting sglang server..."

# Example: If your server is started with a Python script
python3 /home/ubuntu/run_server.py

# Keep container running if script exits
exec tail -f /dev/null