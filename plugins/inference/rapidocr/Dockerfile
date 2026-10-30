# CUDA 12.8 runtime with a pinned cuDNN 9.10.2 override for PP-OCRv6.
# Build context: output of build-inputs.py.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates \
        libglib2.0-0 libsm6 libxext6 libxrender1 gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Hash-locked upstream artifacts; the runtime export intentionally substitutes
# headless OpenCV for the GUI distribution required by RapidOCR metadata.
ARG PIP_FIND_LINKS
ARG PIP_TRUSTED_HOST
COPY requirements-build.txt requirements-runtime.txt /build/
RUN python3 -m pip install --require-hashes --no-deps -r /build/requirements-build.txt \
    && python3 -m pip install --require-hashes --no-deps --no-build-isolation -r /build/requirements-runtime.txt

COPY wheels/ /build/wheels/
RUN python3 -m pip install --no-deps \
        /build/wheels/aap_task_contract-*.whl \
        /build/wheels/aap_protocol_v2-*.whl \
        /build/wheels/aap_lifecycle_contract-*.whl \
        /build/wheels/aap_backend_runtime-*.whl \
        /build/wheels/rapidocr_backend-*.whl \
    && rm -rf /build
COPY validate_managed_lifecycle.py /app/scripts/

# Base supplies CUDA libraries; the pinned wheel supplies cuDNN 9.10.2.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
    RAPIDOCR_MODEL_DIR=/app/models \
    RAPIDOCR_DEVICE=gpu
EXPOSE 8005
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]
