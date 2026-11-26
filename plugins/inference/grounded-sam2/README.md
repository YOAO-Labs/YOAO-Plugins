# Grounded-SAM2 Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-grounded-sam2-plugin` is an open-vocabulary detection, interactive image segmentation, and stateful video tracking plugin for YOAO built on GroundingDINO and SAM 2.1 (`IDEA-Research/Grounded-SAM-2`). It implements YOAO ML Backend protocol v2.2 with separate image and video model pools.

## Capabilities and models

| `/setup` model ID | Task | Composition | Prompts | Inputs | Outputs |
| --- | --- | --- | --- | --- | --- |
| `grounded-sam2-detection` | `detect` | `atom` (DINO) | `text` | `full_image`, `crop` | `bbox` |
| `grounded-sam2-segmentation` | `segment` | `composite` (DINO + SAM 2.1) | `text` | `full_image`, `crop` | `polygon` (`mask` / `both`) |
| `grounded-sam2-interactive-seg` | `interactive_seg` | `atom` (SAM 2.1, interactive) | `point`, `interactive_box`, `mask`, `scribble` | `full_image` + prompts | `polygon`, `mask` (multimask candidates and low-res logits) |
| `grounded-sam2-box-seg` | `segment` | `atom` (SAM 2.1, batchable) | `bbox` | `bbox_prompt`, `full_image` | `polygon` |
| `grounded-sam2-tracker` | `tracker` | `composite` (SAM 2.1 Video, interactive) | `point`, `bbox`, `correction_frame` | `video` + prompts | `bbox`, `polygon`, `mask` (forward/reverse window tracking) |

- **SAM 2.1 variants** (`sam_variant`): `tiny` (default), `small`, `base_plus`, `large` (`sam2.1_hiera_{tiny,small,base_plus,large}.pt`).
- **GroundingDINO variants** (`dino_variant`): `T` (default, `groundingdino_swint_ogc.pth`) and `B` (`groundingdino_swinb_cogcoor.pth`).
- Image models load both SAM 2.1 and GroundingDINO; the video tracker uses only the selected SAM 2.1 checkpoint in a dedicated video model pool with persistent session contexts.

## Model weights and assets

No checkpoints or tokenizers are bundled in this repository or built images. The service enforces Hugging Face offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and fails fast when required files are missing. `grounded_sam2_plugin/pretrained-assets.json` pins the official SAM 2.1, GroundingDINO, and `bert-base-uncased` files with SHA-256 digests.

- In plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), stage approved checkpoints under `AAP_PLUGIN_DATA_DIR/checkpoints`.
- In standalone mode, checkpoints default to `CHECKPOINT_DIR` (`/app/checkpoints`).
- Every image variant also requires local `bert-base-uncased` files under `checkpoints/bert-base-uncased/` (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `model.safetensors`).
- Administrators can run `scripts/download_checkpoints.py` in a separate authorized environment to fetch SAM 2.1 and GroundingDINO weights; BERT files must be staged separately.

## Configuration

The container entrypoint is `uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1`.

| Variable | Default | Description |
| --- | --- | --- |
| `CHECKPOINT_DIR` | `/app/checkpoints` | Standalone checkpoint directory (overridden by `AAP_PLUGIN_DATA_DIR/checkpoints` in plugin mode) |
| `SAM_VARIANT` | `tiny` | Default SAM 2.1 variant (`tiny`, `small`, `base_plus`, `large`) |
| `DINO_VARIANT` | `T` | Default GroundingDINO variant (`T`, `B`) |
| `BOX_THRESHOLD` | `0.35` | Default GroundingDINO box confidence threshold |
| `TEXT_THRESHOLD` | `0.25` | Default GroundingDINO text confidence threshold |
| `EMBEDDING_CACHE_SIZE` | `16` | Cached image embeddings for interactive segmentation |
| `MODEL_POOL_CAP` | `1` | Maximum resident image models |
| `VIDEO_MODEL_POOL_CAP` | `1` | Maximum resident video tracker models |
| `MODEL_POOL_BUILD_TIMEOUT` | `30` | Image model cold-start wait timeout in seconds |
| `VIDEO_MODEL_POOL_BUILD_TIMEOUT` | `60` | Video model cold-start wait timeout in seconds |
| `IDLE_UNLOAD_SECONDS` | `600` | Image pool idle unload threshold in seconds |
| `VIDEO_IDLE_UNLOAD_SECONDS` | `600` | Video pool idle unload threshold in seconds |
| `VIDEO_TRACKER_MAX_WINDOW_FRAMES` | `300` | Maximum frames per video tracking window |
| `VIDEO_TRACKER_MAX_CONTEXT_FRAMES` | `300` | Maximum frames per persistent video context (enables session mode when greater than window size) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Managed plugin mode consumes `AAP_PLUGIN_TOKEN_FILE`, `AAP_PLUGIN_DATA_DIR`, `AAP_PLUGIN_DEVICE`, `AAP_PLUGIN_ASSET_MANIFEST_SHA256`, and for CUDA deployments `AAP_PLUGIN_REGISTRY_ID`, `AAP_ROUTE_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID`, `AAP_PLUGIN_GPU_KEYRING_FILE`, `CUDA_VERSION`, and approved qualification evidence (`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`).

## Runtime requirements

- **Python**: `3.10` (`>=3.10,<3.11`)
- **ML stack**: `torch==2.3.1`, `torchvision==0.18.1`, CUDA `12.1`
- **Submodule**: initialize the `vendor/grounded-sam-2` Git submodule recursively before building the image.
- **Device policy**: managed CUDA plugin mode rejects CPU fallback.

## License

First-party plugin code is licensed under [MIT](LICENSE). Upstream SAM 2 and GroundingDINO code in `vendor/grounded-sam-2` retains Apache-2.0, and the container retains upstream notices (including the BSD-3-Clause `cctorch` notice) under `/usr/share/licenses/grounded-sam2-vendor`.

Model cards for SAM 2.1, GroundingDINO, and `google-bert/bert-base-uncased` declare Apache-2.0; operators must independently verify model and third-party dependency terms before use or redistribution.
