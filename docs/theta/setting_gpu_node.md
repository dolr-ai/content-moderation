```bash
# gpu check
nvidia-smi

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv .venv --python=3.10
source .venv/bin/activate

# python
uv pip install --upgrade pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
uv pip install jupyter

# cuda check
python3 -c "import torch; print(torch.cuda.is_available())"

# system
apt-get update
apt-get install build-essential
export CC=gcc
apt-get install ninja-build
apt-get install python3.10-dev


# Add NVIDIA repository
apt-get update && apt-get install -y --no-install-recommends wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Install just the required CUDA development tools
apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# sglang
python -m sglang.launch_server --model-path microsoft/Phi-3.5-mini-instruct --port 30000

# what worked in T4
python -m sglang.launch_server --model-path microsoft/Phi-3.5-mini-instruct --port 30000 --attention-backend triton --disable-cuda-graph --mem-fraction-static 0.7

# check processes
fuser -v /dev/nvidia*


# check internal ip
hostname -I

# check external ip
curl ifconfig.me
```
