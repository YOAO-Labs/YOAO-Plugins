# Build context: the YOAO-Plugins repository root.
FROM mirror.gcr.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722

LABEL io.yoao.plugin.manifest='{"manifest_schema":"yoao-extension-manifest/v1","extension_point":"trainer","plugin_id":"yoao.trainer.ultralytics","version":"1.0.0","display_name":"Ultralytics YOLO Trainer","description":"Official Ultralytics YOLO 8.4.183 detection and instance segmentation trainer plugin for YOAO.","homepage":"https://github.com/YOAO-Labs/YOAO-Plugins","publisher":"YOAO Official","license":"AGPL-3.0-only","protocol_version":"yoao-trainer/v1","entrypoint":["yoao-trainer-ultralytics"],"capabilities":["train","evaluate","preview","export_weights"],"resource":{"memory_mb":8192},"extension":{"framework_id":"ultralytics","framework_version":"8.4.183","requires_gpu":true}}'

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_AUTOINSTALL=false YOLO_OFFLINE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    POLARS_MAX_THREADS=4 NUMEXPR_MAX_THREADS=4 \
    CUDA_VERSION=13.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates fonts-dejavu-core fonts-noto-cjk libgl1 libglib2.0-0 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/trainer-venv
ENV VIRTUAL_ENV=/opt/trainer-venv PATH=/opt/trainer-venv/bin:$PATH

WORKDIR /app
COPY plugins/trainer/ultralytics/requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes --no-deps -r /tmp/requirements.lock

ARG PLUGIN_SDK_WHEEL=/first-party-inputs/yoao_plugin_sdk-0.1.0-py3-none-any.whl
ARG PLUGIN_SDK_SHA256=efb6936e3338313598c39e648cc573c81486485204d6fcee113711b7ceb12282
RUN --mount=type=bind,from=first-party-inputs,target=/first-party-inputs \
    test -n "$PLUGIN_SDK_WHEEL" && test -n "$PLUGIN_SDK_SHA256" \
    && printf '%s  %s\n' "$PLUGIN_SDK_SHA256" "$PLUGIN_SDK_WHEEL" | sha256sum -c - \
    && python -m pip install --no-deps "$PLUGIN_SDK_WHEEL"

COPY plugins/trainer/ultralytics/ /tmp/ultralytics-trainer/
RUN python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/ultralytics-trainer \
    && python -m pip install --no-deps /tmp/wheels/yoao_trainer_ultralytics-*.whl \
    && cp /tmp/ultralytics-trainer/manifest.json /app/manifest.json \
    && python -c "import torch, torchvision, ultralytics, yoao_plugin_sdk, yoao_trainer_ultralytics; assert torch.__version__ == '2.14.1+cu132'; assert torchvision.__version__ == '0.29.1+cu132'; assert ultralytics.__version__ == '8.4.183'" \
    && rm -rf /tmp/ultralytics-trainer /tmp/wheels /tmp/requirements.lock \
    && groupadd --gid 10001 yoao \
    && useradd --uid 10001 --gid 10001 --home-dir /tmp --no-create-home --shell /usr/sbin/nologin yoao \
    && chmod -R a+rX /opt/trainer-venv /app

USER 10001:10001

ENTRYPOINT ["yoao-trainer-ultralytics"]
