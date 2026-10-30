# LibreYOLO detection plugin

This service uses the native LibreYOLO 1.6.0 loader, not Ultralytics, and the existing detection SDK, model store and GPU lifecycle. It serves `detect` with YOLO9 t/s/m/c, full images and crops. Controlled checkpoints are independently validated against their immutable training receipts. The separately verified official pretrained files have native metadata version 1.1.1; only exact bytes in the canonical controlled asset manifest are accepted. No weights are downloaded by inference.

## Build and prepare

从插件仓库的指定提交构建私有 Linux x86-64 glibc / CPython 3.11 候选包。隔离构建环境需包含 `pip`、`build==1.3.0`、`setuptools==80.9.0`、`wheel==0.45.1`、`hatchling==1.27.0`、`git` 和 GNU `tar`。构建从 PyPI 和 PyTorch cu132 索引下载哈希锁定依赖；即使在 CPU 上推理，也包含 CUDA 链接的 Torch 和 NVIDIA 运行时 wheel。契约 wheel 和权威资产清单从经过校验的公开发布制品获取。

```sh
python scripts/build-libreyolo-candidate.py --revision <commit> --output /private/candidate.tar
```

An optional `--dependency-cache /private/wheel-cache` reuses downloaded wheels across builds. The producer disables pip's shared cache and verifies the locked hashes again when selecting wheels for each candidate. The cache is not part of the candidate.

归档包含由指定插件提交构建的 plugin 和 SDK wheel、经过校验的平台契约 wheel、完整哈希锁定 wheelhouse、`requirements.lock`、源码提交与 wheel 摘要，以及权威受控资产清单；不包含权重或平台 checkout。构建器规范化 wheel 和归档时间戳，但 wheel 兼容性取决于平台标签。这不意味着 OCI 镜像摘要可复现：基础镜像、apt 包和镜像构建器是独立输入。重新分发前须审查许可；私有候选包不是公开发布制品。

Transfer the expected archive SHA256 through a trusted independent channel. Verify that value before unpacking or running any artifact; the adjacent checksum alone is not an authenticity mechanism:

```sh
echo '<trusted-sha256>  candidate.tar' | sha256sum -c -
mkdir candidate && tar -xf candidate.tar -C candidate
python -m pip install --no-index --find-links candidate/wheelhouse --only-binary=:all: --require-hashes -r candidate/requirements.lock
python -m pip check
```

The wheel builder also accepts the canonical manifest explicitly when working directly from source:

```sh
python scripts/fetch-contracts.py /tmp/aap-contracts
python plugins/libreyolo/build-wheel.py /absolute/path/to/wheelhouse --manifest /tmp/aap-contracts/controlled-model-assets.json
```

清单嵌入插件 wheel，并复制到候选包中用于摘要核对；平台发布制品仍是唯一权威来源。在干净环境中从锁定 wheelhouse 安装后，显式准备模型：

```sh
aap-libreyolo prepare-assets --size t --asset-dir /private/libreyolo/assets
```

The command downloads only the pinned official source, bounds redirects and size, and checks SHA256 before publication. The default `t` must be prepared before serving; prepare s/m/c separately to expose those variants. Their original model repository licenses must be reviewed before use or redistribution. [LibreYOLO licensing](https://www.libreyolo.com/docs/licensing) identifies the library's code as MIT and YOLO9 COCO checkpoints as MIT, while identifying each exact weight repository as authoritative; code and weight licenses are distinct.

The upstream MIT declaration is not a license inventory for the complete image. The LibreYOLO distribution also includes third-party notices and bundled code with separate terms, including DINOv3; Torch and NVIDIA runtime components have their own licenses. Review the installed wheel's `NOTICE`, bundled license files and exact weight revisions before redistribution. The manifest's `license` field records the publisher's declaration, not an automatic legal review.

An installed developer project can compose its own detection provider by calling `aap_libreyolo_plugin.cli.main(provider_factory=MyProvider)`. The zero-argument factory returns an object with `load(model_id, device, config)` and `execution_facts()`; the latter should delegate to the native provider when wrapping it so measured runtime facts remain accurate. The generated project owns its model's load, predict and close methods.

For a local service, supply a private token file containing at least 32 characters and explicitly approved object origins/image hosts:

```sh
export AAP_PLUGIN_ASSET_ORIGINS=https://assets.example
export AAP_PLUGIN_IMAGE_HOSTS=assets.example
export AAP_PLUGIN_DATA_DIR=/private/libreyolo
export AAP_PLUGIN_TOKEN_FILE=/private/libreyolo/token
aap-libreyolo serve --device cpu --asset-dir /private/libreyolo/assets --host 127.0.0.1 --port 8000
```

CPU is a development profile, not GPU qualification. For CUDA set `AAP_PLUGIN_DEVICE=cuda:0`, `AAP_PLUGIN_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID` (physical `GPU-...`) and `AAP_PLUGIN_GPU_KEYRING_FILE`. The keyring is the existing JSON mapping from key ID to unpadded base64url raw Ed25519 public key; never put authority private keys in the plugin. The CUDA driver is queried for the physical UUID before serving. GPU requests require the existing signed admission and lifecycle contracts as well as the bearer token. CUDA 13.2 build and the 1.6.0/2.14.1+cu132/0.29.1+cu132 package baseline are required; execution is FP32 with half, AMP and TF32 disabled. Failed device checks do not fall back to CPU.

Managed CUDA rejects workloads from startup, even when a bearer token or a premature signed permit is present. Only the signed owner handshake can enter enforce mode. The explicitly local-exclusive developer and qualification commands retain their isolated legacy path; that path does not qualify a platform installation.

For an OCI artifact provide a digest-pinned base and a complete, separately locked wheelhouse (no Ultralytics):

```sh
aap-plugin build --base-image image@sha256:<digest> --wheelhouse /absolute/path/to/wheelhouse --requirements-lock /absolute/path/to/requirements.lock --entrypoint aap-libreyolo --tag registry.example/libreyolo:release --push
aap-libreyolo manifest registry.example/libreyolo:release --publisher YOUR_PUBLISHER --license REVIEWED_SPDX_EXPRESSION --device cuda --memory-mb 16384 --gpu-memory-mb 8192 --asset-origin https://assets.example --output manifest.json
```

The image runs as UID/GID 10001, listens on port 8000 and uses `/var/lib/aap-libreyolo` for private state. Its fixed entrypoint is `aap-libreyolo serve --host 0.0.0.0 --port 8000`. Mount prepared assets under `assets/`, with token and public keyring files accessible to that UID. The manifest generator reads the pushed image's actual `RepoDigest`; no placeholder digest qualifies for installation. Installing the artifact does not enable it in a project or prove GPU lifecycle qualification.

The installed CLI requires the publisher and license declaration explicitly; review the code and exact weights before supplying them. These commands run outside the platform checkout. The complete wheelhouse and hash-locked requirements must include the entrypoint distribution and all runtime dependencies.

## Qualification

原生 CPU 回归用例位于 `tests/test_native.py`；在安装了候选 wheelhouse 和 pytest 的干净 CPython 3.11 环境中运行。测试无需下载权重，会构造并重新加载真实的 t/s/m/c 模型。

For a local-exclusive developer check, use known real images (one yielding multiple detections and one yielding none):

```sh
aap-libreyolo probe --device cuda:0 --gpu-uuid GPU-... --asset-dir /private/libreyolo/assets --multi-image /private/multiple.jpg --empty-image /private/empty.jpg --output /private/local-probe.json
```

The probe verifies actual native input/model tensor placement and FP32 execution, rejects a missing asset and a nonexistent CUDA ordinal, and measures two real model load/inference/release cycles. It samples other GPU compute processes and fails closed if attribution is unavailable or the selected card is in use. It does not acquire a platform GPU lease; its report explicitly says `local-isolated` and cannot qualify a managed installation.

`aap-libreyolo-evidence --url URL --token-file FILE --scenario FILE --output FILE` accepts a controller-prepared scenario with fresh signed lifecycle/admission headers and records real HTTP responses and `nvidia-smi` observations for at least two prediction/unload cycles. A `kind: cancel_drain` scenario entry contains `predict_headers`, `predict_body`, `drain_headers`, and `drain_body`: the harness observes a live native borrower, cancels its HTTP request, requires the signed drain to report it active and not ready to unload, and waits until native work actually exits. Missing or unobservable cancellation evidence fails. Headers remain private and are not written to the output report. Review output against the existing lifecycle evidence thresholds; it does not issue authority credentials or itself approve a deployment. The training-return trial and project release remain separately required.

Ordinary scenario steps specify `method`, `path`, optional `headers`, `body`, and `expected_status` (default 200). Negative fault probes must specify their expected rejection status; the harness rejects any mismatch and requires two ordered successful prediction/unload pairs.

Managed lifecycle eligibility requires the existing shared lifecycle evidence report, authenticated administrator approval, and the constrained operator's actual Docker image inspection. The operator mounts the exact approved report read-only and sets `AAP_PLUGIN_QUALIFICATION_FILE`, `AAP_PLUGIN_QUALIFICATION_SHA256`, `AAP_PLUGIN_OCI_ARTIFACT`, and `AAP_PLUGIN_IMAGE_ID`; there is no Boolean qualification switch. The report is canonical JSON (`sort_keys=True`, compact separators, `ensure_ascii=True`, `allow_nan=False`) and its SHA256 is over those bytes. The plugin applies shared report thresholds and verifies the report against the image identity supplied by the operator, canonical asset manifest and pinned weight, single models pool, registry/resource binding, observed GPU UUID, driver, memory and CUDA FP32 runtime. The platform owner must separately validate and audit the administrator approval and inspected image; the process cannot independently attest its container image. Missing report means no managed lifecycle capability; invalid or partially supplied evidence fails startup.
