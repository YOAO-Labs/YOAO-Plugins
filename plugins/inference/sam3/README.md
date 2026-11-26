# SAM3 Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-sam3-plugin` is a GPU-only inference plugin for YOAO built on SAM 3 and SAM 3.1 (`facebookresearch/sam3`). It implements YOAO ML Backend protocol v2.2 for text-prompted detection and segmentation, interactive image segmentation, interactive video tracking (PVS), and multiplex text video tracking.

## Capabilities and models

| `/setup` model ID | Task | Composition | Prompts | Inputs | Outputs | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| `sam3-detection` | `detect` | `atom` | `text` (PCS) | `full_image`, `crop` | `bbox` | `checkpoints/sam3.pt` |
| `sam3-segmentation` | `segment` | `composite` | `text` (PCS) | `full_image`, `crop` | `bbox`, `polygon` (`mask` / `both`) | `checkpoints/sam3.pt` |
| `sam3-interactive-seg` | `interactive_seg` | `atom` (interactive) | `point`, `interactive_box`, `mask`, `scribble`, `exemplar` | `full_image` + prompts | `polygon`, `mask` (multimask candidates and low-res logits) | `checkpoints/sam3.pt` |
| `sam3-video-interactive-tracker` | `tracker` | `composite` (PVS, interactive) | `point`, `interactive_box`, `correction_frame` | `video` + prompts | `bbox`, `polygon`, `mask` (forward/reverse tracking) | `checkpoints/sam3.pt` |
| `sam3-video-tracker` | `tracker` | `composite` (SAM 3.1 multiplex) | `text` | `video` | `bbox`, `polygon`, `mask` | `checkpoints/sam3.1_multiplex.pt` |

- All five models use `model_variant=sam3`. BPE vocabulary assets are packaged inside the upstream `sam3` package; local checkpoints do not require extra Hugging Face config files.
- Video tracking responses return whole-window JSON. Persistent PVS session contexts are bound to the active PVS pool entry and invalidated on idle unload, eviction, or terminal drain.

## Model weights and assets

No weights are bundled in this repository or built images, and the service never downloads checkpoints at startup. `sam3_plugin/pretrained-assets.json` pins the expected paths, sizes, and SHA-256 digests for `checkpoints/sam3.pt` (`facebook/sam3`) and `checkpoints/sam3.1_multiplex.pt` (`facebook/sam3.1`).

- In plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), mount approved weights under `AAP_PLUGIN_DATA_DIR/checkpoints`.
- In standalone mode, checkpoints default to `CHECKPOINT_DIR` (`/app/checkpoints`).
- Both Hugging Face repositories are separately gated. Authorized administrators must obtain access independently and may run `scripts/download_checkpoints.py` in a separate preparation environment; never bake credentials into container images.

## Configuration

The container entrypoint is `uvicorn main:app --host 0.0.0.0 --port 8002 --workers 1`, with `HOME=/tmp` so kernel caches work under non-root and read-only root filesystems.

| Variable | Default | Description |
| --- | --- | --- |
| `CHECKPOINT_DIR` | `/app/checkpoints` | Standalone checkpoint directory (overridden by `AAP_PLUGIN_DATA_DIR/checkpoints` in plugin mode) |
| `SAM3_SCORE_THRESHOLD` | `0.5` | Default score threshold |
| `SAM3_EMBEDDING_CACHE_SIZE` | `32` | Cached image embeddings for interactive segmentation |
| `SAM3_MODEL_POOL_BUILD_TIMEOUT` | `120` | Model pool cold-start wait timeout in seconds |
| `SAM3_IDLE_UNLOAD_SECONDS` | `600` | Idle unload threshold in seconds (`<=0` disables) |
| `SAM3_IDLE_CHECK_INTERVAL` | `60` | Idle check interval in seconds |
| `SAM3_VIDEO_MAX_WINDOW_FRAMES` | `16` | Maximum frames per SAM 3.1 multiplex tracking window |
| `SAM3_PVS_MAX_WINDOW_FRAMES` | `16` | Maximum frames per PVS interactive tracking window |
| `SAM3_PVS_MAX_CONTEXT_FRAMES` | `16` | Maximum frames per persistent PVS session context (enables session mode when greater than window size) |
| `SAM3_VIDEO_USE_FA3` | `0` | Set to `1` to enable FlashAttention-3 in video inference |
| `LOG_LEVEL` | `INFO` | Python logging level |

Managed plugin mode consumes `AAP_PLUGIN_TOKEN_FILE`, `AAP_PLUGIN_DATA_DIR`, `AAP_PLUGIN_DEVICE`, `AAP_PLUGIN_ASSET_MANIFEST_SHA256`, `AAP_PLUGIN_REGISTRY_ID`, `AAP_ROUTE_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID`, `AAP_PLUGIN_GPU_KEYRING_FILE`, and approved qualification evidence (`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`).

## Runtime requirements

- **Python**: `3.12` (`>=3.12,<3.13`)
- **ML stack**: `torch==2.7.1`, `torchvision==0.22.1`, CUDA `12.8`
- **Submodule**: initialize the `vendor/sam3` Git submodule before building the image.
- **Device policy**: CUDA is required. Build or device failures fail closed with no CPU fallback.

## License

First-party plugin code is licensed under [MIT](LICENSE). The upstream `facebookresearch/sam3` submodule and model weights (`facebook/sam3` and `facebook/sam3.1`) are governed by the **SAM License**, not MIT.

Administrators must independently obtain gated access and review and comply with the SAM License (including redistribution conditions and required notices) before distributing source, images, or weights.
