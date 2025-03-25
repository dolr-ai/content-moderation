FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

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
    PYTHONPATH=/home/ubuntu:$PYTHONPATH

# Set shell
SHELL ["/bin/bash", "-c"]

# Create ubuntu user with UID 1000 and install system dependencies
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
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Switch to ubuntu user for the Python environment setup
USER $NB_USER
WORKDIR /home/$NB_USER

# Expose sglang server port
EXPOSE 8080

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/

# Copy the entire src_deploy directory structure
COPY --chown=$NB_USER:users ./src_deploy/ /home/$NB_USER/

# Make scripts executable and set up directories
USER root
RUN chmod +x /home/$NB_USER/setup.sh /home/$NB_USER/entrypoint.py /home/$NB_USER/startup.sh \
    && chmod +x /home/$NB_USER/servers/*.py /home/$NB_USER/tests/*.py

USER $NB_USER

# Run GPU setup script and create logs directory
RUN /home/$NB_USER/setup.sh || echo "Setup script had issues but we're continuing the build" \
    && mkdir -p /home/$NB_USER/logs

# Set entrypoint to our startup script
CMD ["/home/ubuntu/startup.sh"]