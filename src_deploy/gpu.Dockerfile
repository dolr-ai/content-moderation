FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    NB_USER=ubuntu \
    NB_UID=1000 \
    HOME=/home/ubuntu \
    SHELL=/bin/bash \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA=cuda>=12.1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=/home/ubuntu:$PYTHONPATH

# Set shell
SHELL ["/bin/bash", "-c"]

# Create ubuntu user with UID 1000, install system dependencies, and set up NVIDIA CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    locales \
    curl \
    wget \
    ca-certificates \
    git \
    gnupg \
    build-essential \
    g++ \
    gcc \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-setuptools \
    ninja-build \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && useradd -m -s /bin/bash -N -u $NB_UID $NB_USER \
    && mkdir -p /home/$NB_USER \
    && chown -R $NB_USER:users /home/$NB_USER \
    && echo "$NB_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    nvidia-utils-535 \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm cuda-keyring_1.1-1_all.deb \
    && ldconfig \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true

# Switch to ubuntu user for the Python environment setup
USER $NB_USER
WORKDIR /home/$NB_USER

# Expose sglang server port
EXPOSE 8080

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/

# Test that uv works
RUN uv --version

# Copy the entire src_deploy directory structure
COPY --chown=$NB_USER:users ./src_deploy/ /home/$NB_USER/

# Make scripts executable and set up directories
USER $NB_USER
RUN uv venv $HOME/.venv \
    && . $HOME/.venv/bin/activate \
    && uv pip install -r requirements_master.txt

USER $NB_USER
RUN chmod +x /home/$NB_USER/startup.sh

# Set entrypoint to our startup script
CMD ["/home/ubuntu/startup.sh"]