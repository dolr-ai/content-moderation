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
ENV PYTHONPATH=/home/ubuntu:$PYTHONPATH

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

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set up NVIDIA CUDA
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb \
    && ldconfig

# Create symbolic link for libcuda.so if missing
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true

# Switch to ubuntu user for the Python environment setup
USER $NB_USER
WORKDIR /home/$NB_USER

# Expose sglang server port
EXPOSE 8080

# Copy the entire src_deploy directory structure
COPY --chown=$NB_USER:users ./src_deploy/ /home/$NB_USER/

# Make scripts executable
USER root
RUN chmod +x /home/$NB_USER/setup.sh /home/$NB_USER/entrypoint.py /home/$NB_USER/startup.sh \
    && chmod +x /home/$NB_USER/servers/*.py /home/$NB_USER/tests/*.py
USER $NB_USER

# Run GPU setup script - don't fail if GPU checks fail during build
RUN /home/$NB_USER/setup.sh || echo "Setup script had issues but we're continuing the build"

# Create necessary directories for logs
RUN mkdir -p /home/$NB_USER/logs

# Set entrypoint to our startup script
CMD ["/home/ubuntu/startup.sh"]