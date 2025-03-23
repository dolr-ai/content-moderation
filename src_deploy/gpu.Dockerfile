FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

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
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app
RUN mkdir -p ./data

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"

# Create and activate virtual environment
RUN uv venv .venv --python=3.10
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install Python packages
RUN uv pip install --upgrade pip && \
    uv pip install "transformers==4.48.3" && \
    uv pip install sgl-kernel --force-reinstall --no-deps && \
    uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/ && \
    uv pip install jupyter pandas tqdm nvitop scikit-learn seaborn matplotlib faiss-gpu faiss-cpu bitsandbytes && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install autoawq --no-build-isolation && \
    uv pip install accelerate


# Copy source code
COPY . .

CMD ["python", "-m", "sglang.launch_server", "--model-path", "microsoft/Phi-3.5-mini-instruct", "--host", "0.0.0.0", "--port", "8899", "--api-key", "None", "--mem-fraction-static", "0.9", "--max-running-requests", "1024", "--attention-backend", "triton", "--disable-cuda-graph", "--dtype", "float16", "--chunked-prefill-size", "512", "--enable-metrics", "--show-time-cost", "--enable-cache-report", "--log-level", "info", "--watchdog-timeout", "120", "--schedule-policy", "lpm", "--schedule-conservativeness", "0.8"]

