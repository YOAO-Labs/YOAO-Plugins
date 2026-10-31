# Build context: YOAO-Plugins root with the pinned submodule initialized.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel@sha256:a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17

ARG TORCH_ARCH="7.0;7.5;8.0;8.6"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    BUILD_WITH_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    CUDA_HOME=/usr/local/cuda-12.1/ CUDA_VERSION=12.1 CC=gcc-10 CXX=g++-10
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ninja-build gcc-10 g++-10 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY plugins/inference/grounded-sam2/build-requirements.lock plugins/inference/grounded-sam2/requirements.lock plugins/inference/grounded-sam2/first-party-requirements.lock /tmp/
RUN python -m pip install --no-deps --require-hashes -r /tmp/build-requirements.lock \
    && python -m pip install --no-deps --require-hashes --no-build-isolation -r /tmp/requirements.lock \
    && python -m pip install --no-deps --require-hashes -r /tmp/first-party-requirements.lock
COPY plugins/inference/grounded-sam2/vendor/grounded-sam-2/ /app/vendor/grounded-sam-2/
RUN test -f vendor/grounded-sam-2/setup.py && test -f vendor/grounded-sam-2/grounding_dino/setup.py \
    && python -m pip install --no-deps --no-build-isolation /app/vendor/grounded-sam-2 \
    && python -m pip install --no-deps --no-build-isolation /app/vendor/grounded-sam-2/grounding_dino
COPY plugins/inference/grounded-sam2/ /tmp/grounded-sam2/
RUN python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/grounded-sam2 \
    && python -m pip install --no-deps /tmp/wheels/grounded_sam2_backend-*.whl \
    && python -c "import torch, torchvision; assert torch.__version__.split('+')[0] == '2.3.1'; assert torchvision.__version__.split('+')[0] == '0.18.1'" \
    && rm -rf /tmp/grounded-sam2 /tmp/wheels && mkdir -p /app/checkpoints /app/.cache/huggingface
EXPOSE 8001
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
