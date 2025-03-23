FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies - removed CUDA packages that aren't in standard repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app
RUN mkdir -p ./data

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "transformers==4.48.3" && \
    pip install --no-cache-dir "sglang[all]>=0.4.2.post4" && \
    pip install --no-cache-dir jupyter pandas tqdm nvitop scikit-learn seaborn matplotlib faiss-gpu faiss-cpu bitsandbytes && \
    pip install --no-cache-dir accelerate

# Copy source code
COPY . .

# Copy GPU check script
COPY ./src_deploy/check_gpu.py /app/check_gpu.py

# Setup proper NVIDIA configurations
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create symbolic links for CUDA libraries if they don't exist
# This helps applications find the libraries more reliably
RUN mkdir -p /usr/local/cuda/lib64 && \
    if [ ! -e /usr/local/cuda/lib64/libcuda.so.1 ]; then \
        ln -s /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 || true; \
    fi && \
    ldconfig

# Start the server (continue even if GPU check fails)
CMD ["sh", "-c", "python /app/check_gpu.py || echo 'WARNING: GPU check failed but continuing' && python -m sglang.launch_server --model-path microsoft/Phi-3.5-mini-instruct --host 0.0.0.0 --port 8899 --api-key None --mem-fraction-static 0.9 --max-running-requests 1024 --attention-backend triton --disable-cuda-graph --dtype float16 --chunked-prefill-size 512 --enable-metrics --show-time-cost --enable-cache-report --log-level info --watchdog-timeout 120 --schedule-policy lpm --schedule-conservativeness 0.8"]