# Build context: the YOAO-Plugins repository root.
FROM mirror.gcr.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722

LABEL io.yoao.plugin.manifest='{"manifest_schema":"yoao-extension-manifest/v1","extension_point":"trainer","plugin_id":"yoao.trainer.libreyolo","version":"1.0.0","display_name":"LibreYOLO Trainer","description":"Official MIT-licensed LibreYOLO9 object detection trainer plugin for YOAO.","homepage":"https://github.com/YOAO-Labs/YOAO-Plugins","publisher":"YOAO Official","license":"MIT","protocol_version":"yoao-trainer/v1","entrypoint":["yoao-trainer-libreyolo"],"capabilities":["train","evaluate","export_weights"],"resource":{"memory_mb":8192},"extension":{"framework_id":"libreyolo","framework_version":"1.6.0","requires_gpu":true}}'

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    LIBREYOLO_FASTER_COCO_EVAL=0 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/libreyolo-trainer-venv
ENV VIRTUAL_ENV=/opt/libreyolo-trainer-venv \
    PATH=/opt/libreyolo-trainer-venv/bin:$PATH

WORKDIR /app

COPY plugins/trainer/libreyolo/requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes --no-deps -r /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock

ARG PLUGIN_SDK_WHEEL=/first-party-inputs/yoao_plugin_sdk-0.1.0-py3-none-any.whl
ARG PLUGIN_SDK_SHA256=50667c6e0c6c118fbb7155d096870bc477c6b1621641fb9b3695fda9f476490c
RUN --mount=type=bind,from=first-party-inputs,target=/first-party-inputs \
    test -n "$PLUGIN_SDK_WHEEL" && test -n "$PLUGIN_SDK_SHA256" \
    && printf '%s  %s\n' "$PLUGIN_SDK_SHA256" "$PLUGIN_SDK_WHEEL" | sha256sum -c - \
    && python -m pip install --no-deps "$PLUGIN_SDK_WHEEL"

COPY plugins/trainer/libreyolo/ /tmp/libreyolo-trainer/
RUN python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/libreyolo-trainer \
    && python -m pip install --no-deps /tmp/wheels/yoao_trainer_libreyolo-*.whl \
    && cp /tmp/libreyolo-trainer/manifest.json /app/manifest.json \
    && python -c "import importlib.util, torch, torchvision, libreyolo; assert torch.__version__ == '2.14.1+cu132'; assert torchvision.__version__ == '0.29.1+cu132'; assert libreyolo.__version__ == '1.6.0'; assert importlib.util.find_spec('ultralytics') is None" \
    && yoao-trainer-libreyolo describe > /dev/null \
    && rm -rf /tmp/libreyolo-trainer /tmp/wheels \
    && groupadd --gid 10001 yoao \
    && useradd --uid 10001 --gid 10001 --home-dir /tmp --no-create-home --shell /usr/sbin/nologin yoao \
    && chmod -R a+rX /opt/libreyolo-trainer-venv /app

USER 10001:10001

ENTRYPOINT ["yoao-trainer-libreyolo"]
