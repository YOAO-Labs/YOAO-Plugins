# syntax=docker/dockerfile:1
# CUDA 12.8 + cuDNN 9 runtime, no Torch dependency. Build context: repository root.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The Git source is fixed to an audited commit; all registry artifacts are
# hash-locked. GUI OpenCV requirements are intentionally replaced by contrib
# headless in the runtime export (see README).
ARG PIP_FIND_LINKS
ARG PIP_TRUSTED_HOST
COPY plugins/inference/onnxtools/requirements-build.txt plugins/inference/onnxtools/requirements-runtime.txt \
     plugins/inference/onnxtools/requirements-vcs.txt plugins/inference/onnxtools/requirements-firstparty.txt /build/
RUN python3 -m pip install --require-hashes --no-deps -r /build/requirements-build.txt \
    && python3 -m pip install --require-hashes --no-deps --no-build-isolation -r /build/requirements-runtime.txt \
    && python3 -m pip install --no-deps --no-build-isolation -r /build/requirements-vcs.txt

# The manifest identity is supplied by the builder; only the Python 3.10
# compatible contracts used by this plugin are selected.
ARG CONTRACT_MANIFEST
ARG CONTRACT_MANIFEST_SHA256
ARG FIRSTPARTY_FIND_LINKS
COPY scripts/fetch-contracts.py /build/fetch-contracts.py
RUN --mount=type=bind,source=.,target=/source,readonly \
    python3 /build/fetch-contracts.py \
        --manifest "$CONTRACT_MANIFEST" --sha256 "$CONTRACT_MANIFEST_SHA256" \
        --output /build/contracts \
        --package yoao-task-contract --package yoao-protocol-v2 \
        --package yoao-lifecycle-contract \
    && python3 -m pip install --no-deps \
        /build/contracts/yoao_task_contract-0.1.0-py3-none-any.whl \
        /build/contracts/yoao_protocol_v2-0.2.0-py3-none-any.whl \
        /build/contracts/yoao_lifecycle_contract-0.1.0-py3-none-any.whl \
    && python3 -m pip install --no-deps --require-hashes \
        --find-links "$FIRSTPARTY_FIND_LINKS" -r /build/requirements-firstparty.txt
COPY plugins/inference/onnxtools/pyproject.toml plugins/inference/onnxtools/LICENSE plugins/inference/onnxtools/main.py \
     plugins/inference/onnxtools/predictor.py plugins/inference/onnxtools/schemas.py plugins/inference/onnxtools/attribute_schema.py \
     plugins/inference/onnxtools/handle_pool.py plugins/inference/onnxtools/gpu_lifecycle.py /build/yoao-onnxtools-plugin/
COPY plugins/inference/onnxtools/pretrained-model-assets.json /build/yoao-onnxtools-plugin/
COPY plugins/inference/onnxtools/scripts/create_plugin_manifest.py /build/yoao-onnxtools-plugin/scripts/
COPY plugins/inference/onnxtools/scripts/validate_managed_lifecycle.py /build/yoao-onnxtools-plugin/scripts/
RUN python3 -m pip install --no-build-isolation --no-deps /build/yoao-onnxtools-plugin \
    && rm -rf /build
COPY plugins/inference/onnxtools/scripts/validate_managed_lifecycle.py /app/scripts/
RUN mkdir -p /app/models
EXPOSE 8004
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
