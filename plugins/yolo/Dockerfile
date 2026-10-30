# Build context: the YOAO-Plugins repository root, not the platform checkout.
FROM mirror.gcr.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.config/Ultralytics YOLO_AUTOINSTALL=false \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    POLARS_MAX_THREADS=4 NUMEXPR_MAX_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libgl1 libglib2.0-0 fonts-dejavu-core python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/yolo-venv
ENV VIRTUAL_ENV=/opt/yolo-venv PATH=/opt/yolo-venv/bin:$PATH
WORKDIR /app
COPY plugins/yolo/requirements.lock plugins/yolo/first-party-requirements.lock /tmp/
RUN python -m pip install --require-hashes --no-deps -r /tmp/requirements.lock \
    && python -m pip install --require-hashes --no-deps -r /tmp/first-party-requirements.lock
RUN python -m pip install --no-deps --no-build-isolation \
    "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"
COPY scripts/fetch-contracts.py /tmp/fetch-contracts.py
RUN python /tmp/fetch-contracts.py /tmp/contracts \
    && python -m pip install --no-deps /tmp/contracts/*.whl
COPY plugins/yolo/ /tmp/yolo/
RUN python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/yolo \
    && python -m pip install --no-deps /tmp/wheels/yolo_backend-*.whl \
    && python -c "import torch,torchvision,ultralytics,pynvml; assert torch.__version__ == '2.14.1+cu132'; assert torchvision.__version__ == '0.29.1+cu132'; assert ultralytics.__version__ == '8.4.183'" \
    && rm -rf /tmp/yolo /tmp/wheels /tmp/contracts \
    && mkdir -p /app/checkpoints /app/.config/Ultralytics \
    && ln -sfn /app/checkpoints/weights /app/weights
EXPOSE 8003
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
