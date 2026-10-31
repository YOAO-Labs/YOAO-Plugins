# Build context: YOAO-Plugins root with the pinned submodule initialized.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel@sha256:a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17

ARG TORCH_ARCH="7.0;7.5;8.0;8.6"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    BUILD_WITH_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    MAX_JOBS=2 CUDA_HOME=/usr/local/cuda-12.1/ CUDA_VERSION=12.1 CC=gcc-10 CXX=g++-10
WORKDIR /app
RUN rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
       'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260601T000000Z/ jammy main universe' \
       'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260601T000000Z/ jammy-updates main universe' \
       'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260601T000000Z/ jammy-security main universe' > /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ninja-build gcc-10 g++-10 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY plugins/inference/grounded-sam2/build-requirements.lock plugins/inference/grounded-sam2/requirements.lock /tmp/
# The pinned base contains two overlapping typing-extensions distributions.
RUN python -m pip uninstall -y typing-extensions \
    && python -m pip uninstall -y typing-extensions \
    && python -m pip install --no-deps --require-hashes -r /tmp/build-requirements.lock \
    && python -m pip install --no-deps --require-hashes --no-build-isolation -r /tmp/requirements.lock
COPY plugins/inference/grounded-sam2/groundingdino-canonical-imports.patch /tmp/
RUN --mount=type=bind,source=plugins/inference/grounded-sam2/vendor/grounded-sam-2,target=/mnt/grounded-vendor,readonly \
    cp -a /mnt/grounded-vendor /tmp/grounded-vendor \
    && patch --batch --fuzz=0 -p1 -d /tmp/grounded-vendor -i /tmp/groundingdino-canonical-imports.patch \
    && test -f /tmp/grounded-vendor/setup.py && test -f /tmp/grounded-vendor/grounding_dino/setup.py \
    && python -m pip install --no-deps --no-build-isolation /tmp/grounded-vendor \
    && python -m pip install --no-deps --no-build-isolation /tmp/grounded-vendor/grounding_dino \
    && mkdir -p /usr/share/licenses/grounded-sam2-vendor \
    && cp /tmp/grounded-vendor/LICENSE* /usr/share/licenses/grounded-sam2-vendor/ \
    && cp /tmp/grounded-vendor/grounding_dino/LICENSE /usr/share/licenses/grounded-sam2-vendor/LICENSE_grounding_dino_subtree \
    && rm -rf /tmp/grounded-vendor \
    && python -c "import torch, sam2._C, groundingdino._C; from importlib.resources import files; from hydra import compose; names=('t','s','b+','l'); assert all(files('sam2').joinpath('configs','sam2.1',f'sam2.1_hiera_{n}.yaml').is_file() and compose(config_name=f'configs/sam2.1/sam2.1_hiera_{n}.yaml') for n in names); assert all(files('groundingdino.config').joinpath(n).is_file() for n in ('GroundingDINO_SwinT_OGC.py','GroundingDINO_SwinB_cfg.py'))"
COPY plugins/inference/grounded-sam2/first-party-requirements.lock /tmp/
RUN python -m pip install --no-deps --require-hashes -r /tmp/first-party-requirements.lock
COPY plugins/inference/grounded-sam2/*.py plugins/inference/grounded-sam2/pyproject.toml plugins/inference/grounded-sam2/README.md plugins/inference/grounded-sam2/LICENSE /tmp/grounded-sam2/
COPY plugins/inference/grounded-sam2/grounded_sam2_plugin/ /tmp/grounded-sam2/grounded_sam2_plugin/
COPY plugins/inference/grounded-sam2/scripts/ /tmp/grounded-sam2/scripts/
RUN python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/grounded-sam2 \
    && python -m pip install --no-deps /tmp/wheels/grounded_sam2_backend-*.whl \
    && python -c "import torch, torchvision; assert torch.__version__.split('+')[0] == '2.3.1'; assert torchvision.__version__.split('+')[0] == '0.18.1'" \
    && python -m pip check \
    && rm -rf /tmp/grounded-sam2 /tmp/wheels && mkdir -p /app/checkpoints /app/.cache/huggingface
EXPOSE 8001
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
