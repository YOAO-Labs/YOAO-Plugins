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
COPY plugins/onnxtools/requirements-build.txt plugins/onnxtools/requirements-runtime.txt \
     plugins/onnxtools/requirements-vcs.txt plugins/onnxtools/requirements-firstparty.txt /build/
RUN python3 -m pip install --require-hashes --no-deps -r /build/requirements-build.txt \
    && python3 -m pip install --require-hashes --no-deps --no-build-isolation -r /build/requirements-runtime.txt \
    && python3 -m pip install --no-deps --no-build-isolation -r /build/requirements-vcs.txt

# Fetch the manifest and wheels with the shared trusted hash verifier; never
# build contracts from a neighbouring checkout.
COPY scripts/fetch-contracts.py /build/fetch-contracts.py
RUN python3 /build/fetch-contracts.py /build/contracts \
    && python3 -m pip install --no-deps \
        /build/contracts/aap_task_contract-0.1.0-py3-none-any.whl \
        /build/contracts/aap_protocol_v2-0.2.0-py3-none-any.whl \
        /build/contracts/aap_lifecycle_contract-0.1.0-py3-none-any.whl \
    && python3 -m pip install --no-deps --require-hashes -r /build/requirements-firstparty.txt
COPY plugins/onnxtools/pyproject.toml plugins/onnxtools/LICENSE plugins/onnxtools/main.py \
     plugins/onnxtools/predictor.py plugins/onnxtools/schemas.py plugins/onnxtools/attribute_schema.py \
     plugins/onnxtools/handle_pool.py plugins/onnxtools/gpu_lifecycle.py /build/onnxtools-backend/
COPY plugins/onnxtools/scripts/validate_managed_lifecycle.py /build/onnxtools-backend/scripts/
RUN python3 -m pip install --no-build-isolation --no-deps /build/onnxtools-backend \
    && rm -rf /build
COPY plugins/onnxtools/scripts/validate_managed_lifecycle.py /app/scripts/
RUN mkdir -p /app/models
EXPOSE 8004
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
