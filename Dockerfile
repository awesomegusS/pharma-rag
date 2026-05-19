# ── Pharma RAG — Dockerfile ───────────────────────────────────────────────────
#
# Builds a local-only container with:
#   - CUDA 12.1 base (GPU support for Mistral)
#   - Tesseract OCR (for scanned PDF pages)
#   - All Python dependencies
#   - The app mounted at /app
#
# Build:
#   docker build -t pharma-rag .
#
# Run (GPU):
#   docker run --gpus all -p 7860:7860 --env-file .env \
#     -v $(pwd)/models:/app/models pharma-rag
#
# Run (CPU only — slow):
#   docker run -p 7860:7860 --env-file .env \
#     -e MISTRAL_N_GPU_LAYERS=0 \
#     -v $(pwd)/models:/app/models pharma-rag

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    build-essential \
    cmake \
    tesseract-ocr \
    tesseract-ocr-eng \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Alias python3.12 → python3 → python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# NOTE: This image assumes the host machine has an NVIDIA GPU with CUDA 12.x.
# If running on CPU only, override at build time:
#   docker build --build-arg CMAKE_ARGS="" -t pharma-rag .
# If running on macOS with Metal, build natively instead (see README).
# ── Install llama-cpp-python with CUDA support first (long build) ─────────────
ARG CMAKE_ARGS="-DGGML_CUDA=on"
ENV CMAKE_ARGS=${CMAKE_ARGS}
RUN pip install --upgrade pip && \
    pip install llama-cpp-python==0.3.23 --no-cache-dir

# ── Install remaining Python dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/      ./app/
COPY main.py   .

# ── Create model directory (model file mounted at runtime via -v) ─────────────
RUN mkdir -p /app/models

# ── Expose Gradio port ────────────────────────────────────────────────────────
EXPOSE 7860

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "main.py"]
