# =========================================================
# Base (shared configuration)
# =========================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =========================================================
# Minimal (CPU-only) — includes warp + examples
# =========================================================
FROM base AS minimal

ENV OMP_NUM_THREADS=2

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir "gradio>=4" && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 && \
    pip install --no-cache-dir "scipy" && \
    pip install --no-cache-dir "anny[warp,examples]@git+https://github.com/naver/anny.git"

EXPOSE 7860
CMD ["python", "-m", "anny.examples.interactive_demo"]

# =========================================================
# Full (GPU + Warp) — lightweight CUDA runtime base
# =========================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS full

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git wget ca-certificates build-essential \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 && \
    python3 -m pip install --no-cache-dir "gradio>=4" && \
    python3 -m pip install --no-cache-dir "anny[warp,examples]@git+https://github.com/naver/anny.git"

EXPOSE 7860
CMD ["python3", "-m", "anny.examples.interactive_demo"]

