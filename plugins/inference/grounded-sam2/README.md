# Grounded-SAM-2 plugin

Distribution `yoao-grounded-sam2-plugin` 0.10.1 retains the platform backend's protocol 2.2 service, its distinct image and video model pools, and its stateful tracker sessions. The runtime baseline is Python 3.10, PyTorch 2.3.1, torchvision 0.18.1 and CUDA 12.1. Its pinned third-party source submodule is IDEA-Research/Grounded-SAM-2 at `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`; initialize recursively before a Docker build. SAM 2 and GroundingDINO source retain their upstream Apache-2.0 licenses and notices; the first-party MIT license does not override third-party source or model terms.

The image retains upstream license files under `/usr/share/licenses/grounded-sam2-vendor`, including the BSD-3-Clause cctorch notice. The submodule remains the authoritative source and license inventory.

`GET /setup`, `GET /health`, `GET /versions`, `POST /predict`, warmup and lifecycle endpoints are retained. Image inference covers text, point, positive/negative points, interactive boxes, batch boxes, masks and scribbles, with polygon or exact COCO RLE as applicable. Interactive requests retain multimask candidates, low-resolution logits and prompt/source identity. Video tracking retains forward/reverse windows, sessions, corrections, cross-window state and cancellation/draining of actual work. Setup/health do not load or download model weights. Auth and lifecycle verification remain governed by the existing backend protocol.

## Assets and access

In managed plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), assets reside under `AAP_PLUGIN_DATA_DIR/checkpoints`; the installer mounts this single approved data root. Standalone mode retains `CHECKPOINT_DIR`, defaulting to `/app/checkpoints`. SAM 2.1 supports `tiny`, `small`, `base_plus`, and `large`; GroundingDINO supports `T` and `B`. Tracker uses the selected SAM checkpoint; every image model advertises both axes because its pool constructs both SAM and DINO. All image variants additionally require `bert-base-uncased` under `checkpoints/bert-base-uncased/{config.json,tokenizer.json,tokenizer_config.json,vocab.txt,model.safetensors}`. SAM and DINO configs come from the pinned source. BERT and checkpoints are governed assets: catalog visibility and possession of credentials do not confer approval, download or redistribution rights. The process uses the explicit local BERT path and Hugging Face offline mode; missing files fail rather than download.

The administrator-side `scripts/download_checkpoints.py` is an explicit preparation tool, not a startup hook and not an approval mechanism. Its download sources and any preexisting files must be independently checked against the published catalog and approved before use. It downloads SAM and DINO checkpoints only, not the BERT files. With the container's default offline environment, an authorized administrator must explicitly change that environment in a separate preparation process; never place private model-governance credentials in the backend.

Catalog identities come from the corresponding public Hugging Face model repository metadata. The model cards for SAM 2.1, GroundingDINO and `google-bert/bert-base-uncased` declare Apache-2.0; rights and permissions must still be verified independently before distribution.

Plugin mode requires the installer-provided identity, route registry, data directory, device and private backend token. CUDA installations also require the public GPU admission keyring and matching allocated GPU identity; they reject CPU fallback. CPU plugin mode and standalone fallback retain their distinct behavior. `/setup` advertises managed GPU lifecycle only when approved evidence matches this installation; terminal retirement fences new work, waits for active operations, closes video sessions and drains both pools.

Persistent tracker state belongs to the exact video model-pool entry that opened it, rather than keeping a borrower between requests. Each start or continuation command borrows that entry only while executing. Idle unload, eviction and terminal drain invalidate its contexts and close their state before disposing the model; a token cannot attach to a later reload of the same model key. Failed or uncertain cleanup retains ownership and prevents an empty-pool proof.

## Build and validation

Before resolving or building, stage verified wheel artifacts from the two separately supplied candidate manifests at the repository root. The manifest digests must come from the artifact provider, not from a guessed release URL:

```sh
python3 scripts/fetch-contracts.py --manifest "$SDK_MANIFEST" --sha256 "$SDK_MANIFEST_SHA256" \
  --output .artifacts/python/first-party --package yoao-backend-runtime --package yoao-mask-utils
python3 scripts/fetch-contracts.py --manifest "$CONTRACT_MANIFEST" --sha256 "$CONTRACT_MANIFEST_SHA256" \
  --output .artifacts/python/first-party --package yoao-lifecycle-contract --package yoao-protocol-v2 --package yoao-task-contract
```

The relative wheel registrations in `pyproject.toml` and `uv.lock` require those exact staged files; these are non-editable artifacts, not source checkouts. In this directory, `uv lock --python 3.10` resolves the plugin and `python3.11 export_requirements.py` regenerates both Docker requirement locks, verifying the staged runtime and mask wheels against `uv.lock`. `build-requirements.lock` pins the build tools. The root-context Dockerfile pins the CUDA development base image by digest and installs the locked Python inputs, checked-out vendor source, and plugin wheel; the base provides Torch and torchvision. It does not provision model assets.

Provide a real contract manifest and its SHA256 as `CONTRACT_MANIFEST` and `CONTRACT_MANIFEST_SHA256` build arguments. The manifest is a URL or a path relative to the repository-root build context; local manifest and referenced wheels must be available in that context and not excluded by `Dockerfile.dockerignore`. The fetcher verifies the manifest and selected wheel hashes and installs only `yoao-lifecycle-contract==0.1.0`, `yoao-protocol-v2==0.2.0`, and `yoao-task-contract==0.1.0`; `pip check` enforces the plugin's versions. No editable platform checkout is required. Candidate wheel and sdist artifacts must be built and validated against these real inputs before any separately authorized publication; the renamed distributions are not assumed to have been published.

System build tools resolve against Ubuntu Jammy's signed `20260601T000000Z` snapshot, rather than floating package repositories.

Build with BuildKit enabled; it bind-mounts the pinned vendor source only for compilation instead of retaining a source layer.

The pinned base contains two overlapping `typing-extensions` distributions; the Docker build removes both before installing the single hashed locked version.

The pinned vendor's GroundingDINO wheel owns the `groundingdino` package, but its source imports a different namespace. `groundingdino-canonical-imports.patch` changes only those Python imports to the installed namespace; the image applies it with zero fuzz before building. For a standalone vendor build, from this directory, archive the pinned submodule into a disposable directory, apply the same patch, then install from that directory:

```sh
vendor_dir=$(mktemp -d)
git -C vendor/grounded-sam-2 archive b7a9c29f196edff0eb54dbe14588d7ae5e3dde28 | tar -x -C "$vendor_dir"
patch --batch --fuzz=0 -p1 -d "$vendor_dir" -i "$PWD/groundingdino-canonical-imports.patch"
python -m pip install --no-deps --no-build-isolation "$vendor_dir/grounding_dino"
```

SAM 2 YAMLs and both native extensions are checked from installed packages during the image build, without retaining a runtime source checkout. CUDA compilation is bounded by `MAX_JOBS=2`.

The operator-only `scripts/validate_managed_lifecycle.py` requires an isolated, explicitly authorized GPU allocation and matching `VALIDATION_GPU_UUID`, registry/resource identity, image digest, plugin artifact and asset-manifest digest. Set `VALIDATION_MODEL_APPROVAL_REF`, `VALIDATION_FIXTURE_APPROVAL_REF`, and `VALIDATION_WEIGHT_SHA256_JSON` mapping the selected SAM and DINO filenames and all five `bert-base-uncased/` filenames to their approved hashes. The script verifies these bytes before loading models, uses a disposable ephemeral signing authority and real loopback HTTP, and emits evidence bound to the exact installation. A passing report is not transferable to another image, GPU, pool topology, model selection or asset manifest.

Context priming also uses signed native admission and managed drain/unload before the two measured cycles; qualification never enables an unsigned legacy workload path. Tracker-session cleanup is measured in `workload_evidence`, not declared as an ORT graph-session topology. CUDA evidence records the loaded runtime version measured by the plugin, rather than PyTorch's build-time CUDA label. Authenticated setup exposes this measured runtime and physical GPU memory for the platform's independent activation check.

CPU-only tests can check package metadata, HTTP setup/health, protocol and lifecycle logic; these do not qualify model inference. Real qualification requires separately authorized model assets and GPU access, including exact masks/logits, multimask identity, all variant combinations, forward/reverse cross-window sessions, lifecycle cancellation and draining across both pools. Do not use GPU0.
