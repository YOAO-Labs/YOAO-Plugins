# Ultralytics YOLO Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-yolo-plugin` is a multi-task inference plugin for YOAO built on Ultralytics YOLO `8.4.183` and the pinned Ultralytics CLIP fork. It implements YOAO ML Backend protocol v2.2 with a pooled PyTorch runtime and supports both stock pretrained models and managed trained checkpoints.

## Capabilities and models

| `/setup` model ID | Task | Prompt | Inputs | Outputs | Supported series and sizes |
| --- | --- | --- | --- | --- | --- |
| `detect` | `detect` | `none` | `full_image`, `crop` | `bbox` | `yolov8` (`n/s/m/l/x`), `yolov9` (`t/s/m/c/e`), `yolov10` (`n/s/m/b/l/x`), `yolo11` (`n/s/m/l/x`, default `yolo11`/`s`), `yolo12` (`n/s/m/l/x`), `yolo26` (`n/s/m/l/x`), `rtdetr` (`l/x`) |
| `segment` | `segment` | `none` | `full_image`, `crop` | `polygon` (native RLE for managed models) | `yolov8` (`n/s/m/l/x`), `yolov9` (`c/e`), `yolo11` (`n/s/m/l/x`, default `yolo11`/`s`), `yolo26` (`n/s/m/l/x`) |
| `pose` | `pose` | `none` | `full_image`, `crop` | `keypoint` | `yolov8`, `yolo11` (default `yolo11`/`s`), `yolo26` (`n/s/m/l/x`) |
| `obb` | `obb` | `none` | `full_image`, `crop` | `rotated_bbox` | `yolov8`, `yolo11` (default `yolo11`/`s`), `yolo26` (`n/s/m/l/x`) |
| `detect-world` | `detect` | `text` | `full_image`, `crop` | `bbox` | `yolo-worldv2` (default `s`), `yolo-world` (`s/m/l/x`) |
| `detect-yoloe` | `detect` | `text` | `full_image`, `crop` | `bbox` | `yoloe-v8` (`s/m/l`), `yoloe-11` (default `s`, `s/m/l`), `yoloe-26` (`n/s/m/l/x`) |
| `segment-yoloe` | `segment` | `text` | `full_image`, `crop` | `polygon` (`mask` / `both`) | `yoloe-v8` (`s/m/l`), `yoloe-11` (default `s`, `s/m/l`), `yoloe-26` (`n/s/m/l/x`) |
| `exemplar-yoloe` | `interactive_seg` | `exemplar` | `full_image` | `bbox`, `polygon` | `yoloe-v8` (`s/m/l`), `yoloe-11` (default `s`, `s/m/l`), `yoloe-26` (`n/s/m/l/x`); multi-box positive prompts only |
| `track` | `tracker` | `none` | `video` | `bbox` with `track_id` | Reuses `detect` series and sizes with `bytetrack` or `botsort` (stock BoT-SORT has ReID disabled) |

- Supported series and size combinations are defined in `model_registry.py`; unsupported combinations return an error.
- Managed Ultralytics detection and segmentation checkpoints are verified against their immutable training receipts and support platform evaluation requests.

## Model weights and assets

No model weights are bundled in this repository or built images. `yolo_plugin/pretrained-assets.json` lists 30 controlled detection and segmentation base models (`yolov8`, `yolo11`, `yolo26` across `n/s/m/l/x`) plus 6 inference and text-encoder assets (`yolo11s-pose.pt`, `yolo11s-obb.pt`, `yolov8s-worldv2.pt`, `yoloe-11s-seg.pt`, `ViT-B-32.pt`, `mobileclip_blt.ts`).

- In plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), mutable checkpoints and receipts live under `AAP_PLUGIN_DATA_DIR/checkpoints`, and verified controlled assets live under `AAP_PLUGIN_DATA_DIR/model-assets`.
- In standalone mode, weights default to `YOLO_CHECKPOINTS_DIR` (`/app/checkpoints`).
- Explicit preparation helpers are available for operators: `python -m scripts.prepare_assets` (stages controlled assets from local approved files) and `python -m scripts.download_weights` (downloads selected closed-set stock weights).
- Set `YOLO_STRICT_OFFLINE=1` to forbid Ultralytics network weight downloads. Open-vocabulary text encoders still require local encoder files and network isolation in offline environments.

## Configuration

The container entrypoint is `uvicorn main:app --host 0.0.0.0 --port 8003 --workers 1`.

| Variable | Default | Description |
| --- | --- | --- |
| `AAP_PLUGIN_TOKEN_FILE` | Empty | Enables plugin mode when set; path to private Bearer token file (at least 32 characters) |
| `AAP_PLUGIN_DEVICE` / `YOLO_DEVICE` | `cuda:0` | Target device (`cpu` or `cuda:0`); plugin mode reads `AAP_PLUGIN_DEVICE` |
| `AAP_PLUGIN_DATA_DIR` / `YOLO_CHECKPOINTS_DIR` | `/app/checkpoints` | Asset root; plugin mode uses `AAP_PLUGIN_DATA_DIR` |
| `YOLO_MODEL_POOL_CAP` | `2` | Maximum resident models in the model pool |
| `YOLO_BUILD_TIMEOUT` | `30` | Cold-start build wait timeout in seconds |
| `YOLO_IDLE_UNLOAD_SECONDS` | `600` | Idle model pool unload threshold in seconds (`<=0` disables) |
| `YOLO_IDLE_CHECK_INTERVAL` | `60` | Idle check interval in seconds |
| `YOLO_STRICT_OFFLINE` | `0` | Set to `1` to disable network weight downloads |
| `YOLO_TRACKER_MAX_FRAMES` | `900` | Maximum frames per video tracking request |
| `LOG_LEVEL` | `INFO` | Python logging level |

Plugin mode also consumes `AAP_PLUGIN_ID`, `AAP_PLUGIN_VERSION`, `AAP_PLUGIN_REGISTRY_ID`, and `AAP_ROUTE_REGISTRY_ID`. Managed CUDA deployments additionally require `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID`, `AAP_PLUGIN_GPU_KEYRING_FILE`, and `CUDA_VERSION=13.2` (set by the image); managed GPU lifecycle is advertised only when approved qualification evidence (`AAP_PLUGIN_QUALIFICATION_FILE` and `AAP_PLUGIN_QUALIFICATION_SHA256`) matches the deployment.

## Runtime requirements

- **Python**: `>=3.11,<3.14`
- **ML stack**: `ultralytics==8.4.183`, pinned Ultralytics CLIP fork, `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132`, CUDA `13.2`
- **Device policy**: supports both CPU and CUDA. Managed CUDA plugin mode rejects CPU fallback.
- **Environment isolation**: flat top-level modules (`main`, `schemas`, `predictor`) must be installed in an isolated Python environment.

## License

First-party YOAO source code is licensed under [MIT](LICENSE). The `ultralytics==8.4.183` runtime dependency and pinned Ultralytics CLIP fork carry [AGPL-3.0](AGPL-3.0.txt) (upstream repositories: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) and [ultralytics/CLIP](https://github.com/ultralytics/CLIP)). Other transitive dependencies and model weights carry separate terms; the combined runtime is not MIT and operators must evaluate AGPL-3.0 and model rights independently.
