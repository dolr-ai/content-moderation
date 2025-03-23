FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
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
    pip install --no-cache-dir flash-attn --no-build-isolation && \
    # pip install --no-cache-dir autoawq --no-build-isolation && \
    pip install --no-cache-dir accelerate

# Install NVIDIA driver runtime libraries (important for libcuda.so)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-utils-525 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Add GPU check script
COPY ./src_deploy/check_gpu.py /app/check_gpu.py

# Run ldconfig to update library cache
RUN ldconfig

CMD ["sh", "-c", "python /app/check_gpu.py && python -m sglang.launch_server --model-path microsoft/Phi-3.5-mini-instruct --host 0.0.0.0 --port 8899 --api-key None --mem-fraction-static 0.9 --max-running-requests 1024 --attention-backend triton --disable-cuda-graph --dtype float16 --chunked-prefill-size 512 --enable-metrics --show-time-cost --enable-cache-report --log-level info --watchdog-timeout 120 --schedule-policy lpm --schedule-conservativeness 0.8"]

