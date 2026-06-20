FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv

RUN uv python install 3.12 \
    && uv venv /opt/venv --python 3.12

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install \
    hydra-core \
    line-profiler \
    numpy \
    omegaconf \
    open3d \
    pandas \
    pytest \
    scipy \
    smplx \
    tqdm \
    trimesh \
    wandb

CMD ["/bin/bash"]
