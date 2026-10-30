# Ultralytics YOLO backend plugin

The `yolo-backend` distribution serves the existing protocol-v2 native backend, not a detection-only SDK adapter. `/setup` exposes detection, instance segmentation, pose, oriented boxes, YOLO-World and YOLOE text detection, YOLOE text segmentation, YOLOE exemplar interaction, and ByteTrack/BoT-SORT video tracking. Supported series and size combinations are defined by `model_registry.py`; unsupported combinations remain errors. Managed Ultralytics detection and segmentation installations retain their immutable receipts, evaluation requests and native segmentation RLE. The platform owns task state, model approval and annotation writes.

## Build and runtime

From the repository root, the independent build context contains no platform checkout:

```sh
docker build -f plugins/inference/yolo/Dockerfile -t local-yolo-plugin .
```

The exec-form entrypoint is `uvicorn main:app --host 0.0.0.0 --port 8003 --workers 1`. The wheel distribution is `yolo-backend==0.1.0`; build it with `python -m build --wheel --sdist --no-isolation plugins/inference/yolo`. Flat modules (`main`, `schemas`, etc.) must not be co-installed with another flat backend in the same Python environment. The image uses a digest-pinned PyTorch/CUDA base, hash-locked third-party dependencies, the pinned Ultralytics CLIP git revision, and released backend-runtime and contract wheels pinned in `first-party-requirements.lock`. The development lock uses the same pinned contract release from Plugin-SDK. No model weights are included. Do not publish the image or dependencies without the necessary rights.

The image explicitly sets `CUDA_VERSION=13.2` to match its pinned base and packaged CUDA runtime; that base does not export the variable. Managed startup verifies this metadata against the actual runtime and driver. Qualification runners must not inject a missing image declaration or bypass those checks.

Plugin mode is selected by `AAP_PLUGIN_TOKEN_FILE`. Supply an independent, private Bearer token file (at least 32 characters), `AAP_PLUGIN_ID`, `AAP_PLUGIN_VERSION`, `AAP_PLUGIN_DATA_DIR`, `AAP_PLUGIN_DEVICE` (`cpu` or `cuda:0`), `AAP_PLUGIN_REGISTRY_ID` and matching `AAP_ROUTE_REGISTRY_ID`. The platform supplies its separate route-control credentials. `/health` is unauthenticated; `/setup`, model endpoints and CPU terminal drain require the plugin Bearer token. Managed installation and evaluation use that same plugin token, never the platform's first-party provider credential. Mutable checkpoints and receipts are under `AAP_PLUGIN_DATA_DIR/checkpoints`; prepared controlled assets are under `AAP_PLUGIN_DATA_DIR/model-assets`.

For a CPU-only disposable run, disable GPU visibility explicitly, use `--runtime=runc`, mount private token and data directories, and bind an unused loopback port. Set `YOLO_STRICT_OFFLINE=1` when network model downloads are forbidden. An empty data directory suffices for health and catalog discovery; it does not prove inference. Never start a GPU container or validator without an explicit allocation and acceptance window.

`/setup` model resource profiles reflect the configured device until inference resolves a device, then the latched runtime device; CPU size variants report zero VRAM. Discovery does not probe or allocate CUDA.

`YOLO_MODEL_POOL_CAP`, `YOLO_BUILD_TIMEOUT`, `YOLO_IDLE_UNLOAD_SECONDS`, `YOLO_IDLE_CHECK_INTERVAL` and `YOLO_STRICT_OFFLINE` retain their backend meanings. A CPU plugin advertises `/lifecycle/terminal-drain`, which is identity-bound and irreversible for that process. CUDA plugins require a bound GPU resource, physical GPU UUID and `AAP_PLUGIN_GPU_KEYRING_FILE`; they reject CPU fallback. They do not advertise managed GPU lifecycle until the installed image's approved evidence matches the actual installation, hardware, weights, runtime and pool topology. The legacy `YOLO_MANAGED_LIFECYCLE_VERIFIED` flag cannot qualify plugin mode.

## Asset preparation

The installed resource `yolo_plugin/pretrained-assets.json` is this plugin's versioned pretrained descriptor catalog; use `yolo_plugin.catalog.catalog_bytes()` and `catalog_sha256()` to consume its exact bytes and identity. Its `assets` array contains the 30 controlled Ultralytics detection and segmentation base models, not the entire stock inference matrix; `inference_assets` separately describes selected pose, OBB, open-vocabulary models and their text-encoder dependencies without declaring encoders to be trainable checkpoints. The controlled models' file names, sizes, SHA-256 digests and upstream locations derive from the historical contracts-0a21bc402432 manifest; inference asset identities are pinned to the listed sources. Weight licenses are unverified. Catalog visibility does not grant download, enablement or redistribution rights. The platform independently records approval and availability and derives its own storage keys.

After obtaining explicit approval and local bytes, prepare selected models with:

```sh
python -m scripts.prepare_assets --source-dir /secure/reviewed-weights \
  --target-dir /secure/plugin-data/model-assets --model yolo11n
```

The tool reads the installed catalog, verifies bytes and atomically prepares the digest cache; it never uploads or accesses object storage. The platform administrator owns S3 publication. `--model` may be repeated; omitting it selects all catalog entries. Stock weights outside this catalog can be prepared separately with `python -m scripts.download_weights --series yolo11 --target-dir /secure/plugin-data/checkpoints`; `--task`, `--size`, and `--all` select supported closed-set variants. These are explicit operator actions, not catalog-driven downloads. An existing controlled mapping that is missing or corrupt never falls back to an uncontrolled same-named file.

## Qualification and tests

Run `uv run --extra dev pytest tests/ -q` from `plugins/inference/yolo` in an isolated environment. Tests that replace model builders are not real model qualification. A real GPU lifecycle report must be collected against the exact candidate image, plugin identity, asset manifest, registry, GPU resource and physical card by the operator with an exclusive approved GPU window. `python -m scripts.validate_managed_lifecycle` uses a private ephemeral operator authority, real loopback HTTP, the original multi-task/tracker load, World and YOLOE text and exemplar requests in both cycles, and two-cycle unload checks. Controlled detection or segmentation fixtures add their immutable installation, native output and cold-reload checks. Required identity and stock inputs include `VALIDATION_GIT_COMMIT`, `VALIDATION_IMAGE_ID`, `VALIDATION_BACKEND_REGISTRY_ID`, `VALIDATION_GPU_RESOURCE_ID`, `VALIDATION_GPU_UUID`, `VALIDATION_MODEL_APPROVAL_REF`, `VALIDATION_FIXTURE_APPROVAL_REF`, and `VALIDATION_WEIGHT_SHA256_JSON` mapping the four tested stock filenames to their approved digests. The matching `AAP_PLUGIN_*` identity, token, image ID and manifest digest must be supplied.

Before qualification, the operator selects approved local weight files under `AAP_PLUGIN_DATA_DIR` with `python -m yolo_plugin.qualification_inventory --data-dir "$AAP_PLUGIN_DATA_DIR" --weight checkpoints/yolo11s.pt --weight checkpoints/yolo11s-seg.pt --weight checkpoints/yolo11s-pose.pt --weight checkpoints/yolo11s-obb.pt --weight checkpoints/yolov8s-worldv2.pt --weight checkpoints/yoloe-11s-seg.pt --weight checkpoints/mobileclip_blt.ts --weight checkpoints/weights/clip/ViT-B-32.pt`. Add a separate controlled checkpoint path when it is not one of these files. This records `qualification-weights.json` from actual bytes; startup and the validator independently recheck containment, size and SHA-256. The report's exact weight set must equal the independently selected inventory. The operator must obtain approval and stage each asset before running this command; neither this inventory nor catalog visibility grants approval.

Open-vocabulary qualification requires `VALIDATION_OPEN_IMAGE`, `VALIDATION_OPEN_WORLD_BASELINE`, `VALIDATION_OPEN_YOLOE_BASELINE`, `VALIDATION_OPEN_EXEMPLAR_BASELINE`, and `VALIDATION_OPEN_FIXTURE_SHA256_JSON` mapping `image`, `world`, `yoloe`, and `exemplar` to the approved frozen input digests. Baselines contain independent positive native outputs frozen before plugin comparison. `VALIDATION_OPEN_WEIGHT_SHA256_JSON` maps `yolov8s-worldv2.pt`, `yoloe-11s-seg.pt`, `mobileclip_blt.ts`, and `ViT-B-32.pt` to approved digests; `VALIDATION_OPEN_APPROVAL_REF` identifies their approval. Place these at the checkpoint paths listed above. The validator checks positive classes, geometry, encoder execution devices and model residency in both measured cycles and records actual outputs and approved artifacts. Use a network fence: upstream text encoders may otherwise download missing files independently of `YOLO_STRICT_OFFLINE`.

Positive tracking qualification requires `VALIDATION_TRACKER_VIDEO`, `VALIDATION_TRACKER_BYTETRACK_BASELINE`, `VALIDATION_TRACKER_BOTSORT_BASELINE`, and `VALIDATION_TRACKER_FIXTURE_SHA256_JSON` mapping `video`, `bytetrack`, and `botsort` to the approved frozen input digests. Both native baselines must identify the exact video and their algorithm. Both trackers execute in both measured cycles; positive frame continuity, box geometry, actual backend type and model device are checked. The stock BoT-SORT configuration has ReID disabled; do not silently acquire extra models.

GPU outputs are compared with CPU-frozen expectations using minimum box and polygon IoU 0.9 and maximum confidence delta 0.1; tracker frame boxes require IoU 0.9. Classes, result counts and tracker frame indices must match exactly. The report records these comparator tolerances and actual outputs; it does not claim bit-exact CPU/CUDA equality.

The validator emits canonical JSON and derives `passed` from measured checks. A coordinator-approved report and its digest must be mounted as `AAP_PLUGIN_QUALIFICATION_FILE` and `AAP_PLUGIN_QUALIFICATION_SHA256`; a report from a different image is never sufficient.

Torch qualification topology contains the exact Registry, GPU resource and native model pools, without an empty ONNX Runtime `session_counts` field. The pinned shared runtime validates this same shape during normal SDK activation. Changing the runtime image requires a fresh bound report; never edit an earlier report to make installation accept it.

Plugin health reports the actual model-pool capacity and measured GPU UUID, driver and total memory alongside performance counters; CPU installations report no GPU identity. Setup and qualification use the measured CUDA runtime, not the framework's compile-time CUDA label. The strict validator checks these activation facts before accepting inference cycles, and normal SDK activation remains a separate required check.

Optional controlled-model fixture inputs and validation rules are in `scripts/controlled_segmentation_validation.py`. Actual GPU validation, model accuracy, storage publication and platform workflow acceptance are separate from a CPU setup smoke.

## Licenses

First-party YOAO code is under `LICENSE` (MIT). The Ultralytics 8.4.183 dependency and the pinned Ultralytics CLIP fork at `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40` both carry AGPL-3.0; their upstream license text is retained in `AGPL-3.0.txt` and their source is available at https://github.com/ultralytics/ultralytics and https://github.com/ultralytics/CLIP. Other transitive dependencies have their own licenses, and model weights have separate terms. The combined runtime is not represented as MIT; downstream operators must assess distribution and network-service obligations and model rights.
