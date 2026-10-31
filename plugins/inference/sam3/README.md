# SAM3 inference backend

Independent `sam3-backend` 0.10.0 distribution. The HTTP service retains the platform's complete `/setup`, `/predict`, image and video sessions, lifecycle, admission, cancellation and drain contracts. It uses Python 3.12, PyTorch 2.7.1, torchvision 0.22.1 and CUDA 12.8. SAM3 inference is CUDA-only: a CPU is not a fallback.

## Capabilities and assets

| `/setup` model ID | Variant | Capability | Prepared file |
| --- | --- | --- | --- |
| `sam3-detection` | `model_variant=sam3` | PCS text detection | `checkpoints/sam3.pt` |
| `sam3-segmentation` | `model_variant=sam3` | PCS text segmentation | `checkpoints/sam3.pt` |
| `sam3-interactive-seg` | `model_variant=sam3` | points, box, mask/scribble seed, positive/negative exemplars | `checkpoints/sam3.pt` |
| `sam3-video-interactive-tracker` | `model_variant=sam3` | PVS point/box/mask-seeded forward/reverse tracking | `checkpoints/sam3.pt` |
| `sam3-video-tracker` | `model_variant=sam3` | SAM3.1 multiplex text tracking | `checkpoints/sam3.1_multiplex.pt` |

The existing APIs preserve exact masks/RLE and polygons, low-resolution logits and multimask candidates, prompt/source/object identity, cross-window sessions, lost/close semantics, and cancellation/drain of actual work and borrowers. Video responses are whole-window JSON, not an incremental frame stream. BPE vocabulary is included in the upstream SAM3 package; the local-checkpoint builders do not need Hugging Face configuration files. Keep vendor package data intact.

`sam3_plugin.catalog.descriptor()` and `deployment_selections(data_root)` use only the standard library and do not access or mutate assets. The published checkpoint hashes come from public Git LFS pointer texts whose Git blob SHA-1s match the upstream gated repositories' public tree entries; catalog `identity_evidence` records those blob IDs and pointer sources. This verifies metadata without downloading weights. Metadata does not authorize download, execution or redistribution.

## Access and licensing

The upstream submodule is `facebookresearch/sam3` at `4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b`; its `LICENSE` is the SAM License, not MIT. This plugin's `LICENSE` covers first-party code only. The SAM License governs SAM Materials and includes redistribution conditions; assess the exact license and preserve the license/notice before distributing source, images or weights. `facebook/sam3` and `facebook/sam3.1` are separately gated repositories: authorized administrators must independently obtain access and review/accept applicable terms for each. A token proves neither legal permission nor redistribution rights. No weights are included in this repository or built image.

## Build and run

Initialize submodules in the checkout. From the repository root, build with `docker build -f plugins/inference/sam3/Dockerfile .`. The image uses the pinned CUDA base digest and installs the separately hashed Python locks; the request-scoped autocast patch is applied with zero fuzz. It does not download checkpoints at startup. Native deployments mount the approved `AAP_PLUGIN_DATA_DIR` (normally `/var/lib/aap-plugin`) and place assets in its `checkpoints/` directory. Standalone deployments may instead set `CHECKPOINT_DIR` (default `/app/checkpoints`). Administrators may explicitly invoke `scripts/download_checkpoints.py` in an appropriately authorized preparation environment; that command requires separate gated access and credentials, and must not run as a governed service startup step. Credentials must not be baked into the image.

Native startup reads the instance token and GPU verification keyring files, binds signed lifecycle claims to the supplied registry/GPU resource identity, and rejects legacy unsigned GPU workloads. It advertises managed lifecycle only after the SDK validates approved qualification evidence against the actual three-pool topology, GPU identity, immutable image and both checkpoint identities. The signed terminal drain fences all three pools permanently and waits for builders, borrowers, operations and sessions to drain.

Real-GPU qualification runs in a separately authorized isolated validation process, never inside the governed service. To emit native-bound canonical evidence, set `VALIDATION_PLUGIN_ID`, `VALIDATION_PLUGIN_VERSION`, `VALIDATION_OCI_ARTIFACT`, `VALIDATION_ASSET_MANIFEST_SHA256`, `VALIDATION_REGISTRY_ID`, `VALIDATION_GPU_RESOURCE_ID`, `VALIDATION_IMAGE_ID`, `VALIDATION_GPU_UUID`, `CUDA_VERSION`, and the existing model/fixture approval and hash variables required by `scripts/validate_managed_lifecycle.py`. The resulting report still requires independent approval and its digest must be supplied by the installer; setting an old standalone lifecycle flag does not qualify native operation.

For standalone packaging, build wheel and sdist with `python -m build plugins/inference/sam3`; installing the service also requires the exact patched vendor source and first-party contract wheels represented by the locks. The included `uv.lock`, `requirements.lock` and `first-party-requirements.lock` describe the independent environment; do not use platform source paths as runtime dependencies. For managed lifecycle validation see `scripts/validate_managed_lifecycle.py` and its explicit approval prerequisites.

CPU-only packaging, metadata and HTTP checks do not qualify model inference. Real inference, every pool's two lifecycle cycles, forward/reverse cross-window identity, exact masks, cancellation and teardown require separately authorized GPU and checkpoint acceptance.
