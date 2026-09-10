# Production Dockerfile for Upframe Pipeline
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /opt/venv

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . /workspace

# Install upframe in editable mode
RUN pip install -e .

ENTRYPOINT ["upframe"]
CMD ["--help"]
