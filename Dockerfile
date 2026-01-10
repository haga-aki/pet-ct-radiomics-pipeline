# PET-CT Radiomics Pipeline - Docker Image
# =========================================
# Multi-stage build for optimized image size
# Supports both CPU and GPU (NVIDIA CUDA) execution

# =========================================
# Stage 1: Base image with CUDA support
# =========================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools for pyradiomics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# =========================================
# Stage 2: Python dependencies
# =========================================
FROM base AS dependencies

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python packages (numpy first for pyradiomics build)
# IMPORTANT: Version constraints are critical for compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy>=1.26,<2.0 && \
    pip install --no-cache-dir \
    pandas>=2.0 \
    tqdm>=4.65 \
    pyyaml>=6.0 \
    scikit-learn>=1.3 \
    scipy>=1.10 \
    SimpleITK>=2.3 \
    nibabel>=5.0 \
    pydicom>=2.4 \
    dicom2nifti>=2.4 \
    matplotlib>=3.7 \
    seaborn>=0.12 \
    "TotalSegmentator>=2.0,<3.0"

# Install pyradiomics with build dependencies
RUN pip install --no-cache-dir setuptools wheel cython versioneer six pykwalify && \
    pip install --no-cache-dir --no-build-isolation "pyradiomics>=3.0,<4.0"

# =========================================
# Stage 3: Final image
# =========================================
FROM dependencies AS final

# Copy application code
COPY run_pipeline.py .
COPY run_full_analysis.py .
COPY suv_converter.py .
COPY create_final_suv.py .
COPY gui_launcher.py .
COPY visualize_mask_verification.py .
COPY params.yaml .
COPY config.yaml.example .
COPY CHANGELOG.md .

# Copy documentation (optional, for reference)
COPY docs/ ./docs/

# Create necessary directories
RUN mkdir -p /data/input /data/output /app/nifti_images /app/segmentations

# Set environment variables
ENV PET_PIPELINE_ROOT=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run_pipeline.py", "--help"]

# =========================================
# Usage:
# =========================================
# Build:
#   docker build -t pet-ct-radiomics .
#
# Run (CPU):
#   docker run -v /path/to/dicom:/data/input -v /path/to/output:/data/output \
#       pet-ct-radiomics python run_pipeline.py --input /data/input --output /data/output
#
# Run (GPU):
#   docker run --gpus all -v /path/to/dicom:/data/input -v /path/to/output:/data/output \
#       pet-ct-radiomics python run_pipeline.py --input /data/input --output /data/output
# =========================================
