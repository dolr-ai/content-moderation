FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NB_USER=ubuntu
ENV NB_UID=1000
ENV HOME=/home/ubuntu
ENV SHELL=/bin/bash
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set shell
SHELL ["/bin/bash", "-c"]

# Create ubuntu user with UID 1000
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && useradd -m -s /bin/bash -N -u $NB_UID $NB_USER \
    && mkdir -p /home/$NB_USER \
    && chown -R $NB_USER:users /home/$NB_USER \
    && echo "$NB_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install newer GCC/G++ for updated libstdc++ (needed for bitsandbytes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    git \
    gnupg \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set up NVIDIA CUDA
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb

# Switch to ubuntu user for the Python environment setup
USER $NB_USER
WORKDIR /home/$NB_USER

# Copy setup script
COPY --chown=$NB_USER:users ./src_deploy/setup-a10.sh /home/$NB_USER/setup-a10.sh
COPY --chown=$NB_USER:users ./src_deploy/start-server.sh /home/$NB_USER/start-server.sh

# Also copy any server scripts needed (later)
# COPY --chown=$NB_USER:users ./src_deploy/run_server.py /home/$NB_USER/run_server.py

# Make scripts executable
USER root
RUN chmod +x /home/$NB_USER/setup-a10.sh /home/$NB_USER/start-server.sh
USER $NB_USER

# Run GPU setup script - don't fail if GPU checks fail during build
RUN /home/$NB_USER/setup-a10.sh || echo "Setup script had issues but we're continuing the build"

# Create model directory
RUN mkdir -p /home/$NB_USER/models

# Expose sglang server port
EXPOSE 8899

# Set entrypoint to start the sglang server
# ENTRYPOINT ["/home/ubuntu/start-server.sh"]
CMD ["python3", "-m", "sglang.launch_server", "--model-path", "microsoft/Phi-3.5-mini-instruct", \
"--host", "0.0.0.0", \
"--port", "8899", \
"--api-key", "None", \
"--mem-fraction-static", "0.9", \
"--max-running-requests", "1024", \
"--attention-backend", "triton", \
"--disable-cuda-graph", \
"--dtype", "float16", \
"--chunked-prefill-size", "512", \
"--enable-metrics", \
"--show-time-cost", \
"--enable-cache-report", \
"--log-level", "info", \
"--watchdog-timeout", "120", \
"--schedule-policy", "lpm", \
"--schedule-conservativeness", "0.8"]