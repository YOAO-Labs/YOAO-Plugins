# Build context: YOAO-Plugins repository root. Initialize the pinned submodule first.
FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04@sha256:2a015be069bda4de48d677b6e3f271a2794560c7d788a39a18ecf218cae0751d

ARG USE_CUDA=1
ARG TORCH_ARCH="8.0;8.6;8.9;9.0"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/app/.cache/huggingface \
    AM_I_DOCKER=True BUILD_WITH_CUDA=${USE_CUDA} TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    CUDA_HOME=/usr/local/cuda PATH=/opt/python/cpython-3.12.11-linux-x86_64-gnu/bin:${PATH}
WORKDIR /app
RUN printf '%s\n' \
    'deb https://snapshot.ubuntu.com/ubuntu/20250604T000000Z/ jammy main universe' \
    'deb https://snapshot.ubuntu.com/ubuntu/20250604T000000Z/ jammy-updates main universe' \
    > /tmp/ubuntu-snapshot.list \
    && apt-get -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.list -o Dir::Etc::sourceparts=- -o Acquire::Check-Valid-Until=false update \
    && apt-get -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.list -o Dir::Etc::sourceparts=- -o Acquire::Check-Valid-Until=false install -y --no-install-recommends \
    ca-certificates ninja-build gcc g++ patch libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* /tmp/ubuntu-snapshot.list
COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install --install-dir /opt/python --no-bin 3.12.11 \
    && python3.12 -m ensurepip
COPY plugins/inference/sam3/*requirements.lock /tmp/
RUN python3.12 -m pip install --no-deps --require-hashes -r /tmp/build-requirements.lock \
    && python3.12 -m pip install --no-deps --require-hashes -r /tmp/cuda-requirements.lock \
    && python3.12 -m pip install --no-build-isolation --no-deps --require-hashes -r /tmp/requirements.lock \
    && python3.12 -m pip install --no-build-isolation --no-deps --require-hashes -r /tmp/fairscale-requirements.lock \
    && python3.12 -m pip install --no-deps --require-hashes -r /tmp/first-party-requirements.lock
COPY plugins/inference/sam3/ /tmp/sam3-plugin/
RUN test -f /tmp/sam3-plugin/vendor/sam3/pyproject.toml \
    && patch --batch --forward --fuzz=0 -d /tmp/sam3-plugin/vendor/sam3 -p1 < /tmp/sam3-plugin/patches/0001-request-scoped-autocast.patch \
    && python -m pip install --no-deps --no-build-isolation /tmp/sam3-plugin/vendor/sam3 \
    && python -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels /tmp/sam3-plugin \
    && python -m pip install --no-deps /tmp/wheels/sam3_backend-*.whl \
    && python -c "import sys,torch,torchvision; assert sys.version_info[:2] == (3, 12); assert torch.__version__.split('+')[0] == '2.7.1'; assert torchvision.__version__.split('+')[0] == '0.22.1'" \
    && mkdir -p /app/checkpoints /app/scripts \
    && cp /tmp/sam3-plugin/scripts/* /app/scripts/ \
    && rm -rf /tmp/sam3-plugin /tmp/wheels
EXPOSE 8002
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
