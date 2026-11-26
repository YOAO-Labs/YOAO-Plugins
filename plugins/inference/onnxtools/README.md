# ONNX Tools Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-onnxtools-plugin` is a two-stage vehicle attribute inference plugin for YOAO built on [onnxtools](https://github.com/yyq19990828/onnxtools) and ONNX Runtime `1.22.0`. It implements YOAO ML Backend protocol v2.2 and exposes both a combined detection-and-attribute pipeline and standalone atomic handles for multi-stage workflows.

## Capabilities and models

| `/setup` model ID | Task | Composition | Inputs | Outputs | ORT sessions |
| --- | --- | --- | --- | --- | --- |
| `vehicle-attr` | `detect` | `composite` (RT-DETR + PP-LCNet) | `full_image`, `crop` | `bbox` + `vehicle_type`, `color` | 2 |
| `vehicle-detect` | `detect` | `atom` (RT-DETR only) | `full_image`, `crop` | `bbox` | 1 |
| `vehicle-attr-classify` | `classify` | `atom` (PP-LCNet only) | `full_image`, `crop` | `vehicle_type`, `color` (no geometry) | 1 |

- **`vehicle_type`** (13 classes): `car`, `truck`, `bus`, `tanker`, `slagcar`, `fire_engine`, `mixer`, `ambulance`, `police_car`, `engineering_truck`, `hazardous_goods_vehicle`, `manned_sweeping_vehicle`, `school_bus`.
- **`color`** (11 classes): `black`, `white`, `gray`, `red`, `yellow`, `green`, `blue`, `purple`, `brown`, `pink`, `other`.
- `/setup` publishes `output_attribute_schema` for `vehicle-attr` and `vehicle-attr-classify` so projects can import the attribute options directly.

## Model weights and assets

No model weights are bundled in this repository or built images. `pretrained-model-assets.json` describes the two ONNX files (`rtdetr-2024080100.onnx` for vehicle detection and `va_260612.onnx` for attribute classification) with their expected sizes and SHA-256 digests.

- In plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), mount approved weights read-only under `AAP_PLUGIN_DATA_DIR/models`.
- In standalone mode, mount weights under `ONNXTOOLS_MODEL_DIR` (default `/app/models`).

## Configuration

The container entrypoint is `python3 -m uvicorn main:app --host 0.0.0.0 --port 8004 --workers 1`.

| Variable | Default | Description |
| --- | --- | --- |
| `ONNXTOOLS_MODEL_DIR` | `/app/models` | Standalone model directory (overridden by `AAP_PLUGIN_DATA_DIR/models` in plugin mode) |
| `ONNXTOOLS_DET_MODEL` | `rtdetr-2024080100.onnx` | Vehicle detection ONNX filename |
| `ONNXTOOLS_VA_MODEL` | `va_260612.onnx` | Vehicle attribute classification ONNX filename |
| `ONNXTOOLS_CONF_THRES` | `0.5` | Detection confidence threshold |
| `ONNXTOOLS_BUILD_TIMEOUT` | `30` | Cold-start wait timeout in seconds |
| `ONNXTOOLS_IDLE_UNLOAD_SECONDS` | `600` | Full-pool idle unload threshold in seconds (`<=0` disables) |
| `ONNXTOOLS_IDLE_CHECK_INTERVAL` | `60` | Idle check interval in seconds |
| `LOG_LEVEL` | `INFO` | Python logging level |

Managed plugin mode also consumes `AAP_PLUGIN_TOKEN_FILE`, `AAP_PLUGIN_DATA_DIR`, `AAP_PLUGIN_DEVICE`, `AAP_PLUGIN_REGISTRY_ID`, `AAP_ROUTE_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID`, `AAP_PLUGIN_GPU_KEYRING_FILE`, and approved qualification evidence (`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`) for CUDA lifecycle enforcement.

## Runtime requirements

- **Python**: `>=3.10`
- **ML stack**: pinned `onnxtools[inference]`, `onnxruntime-gpu==1.22.0` (runs on both CUDA and CPU), `numpy==2.2.6`, `opencv-contrib-python-headless==5.0.0.93`, CUDA `12.8` runtime base image
- **Device policy**: standalone mode may fall back to CPU when CUDA is unavailable. Managed CUDA plugin mode rejects whole-session CPU fallback.
- **Environment isolation**: flat top-level modules (`main`, `schemas`, `predictor`) must be installed in an isolated Python environment.

## License

First-party plugin code is licensed under [MIT](LICENSE). Upstream `onnxtools` and third-party runtime dependencies remain subject to their own licenses.

Provenance and licenses for `rtdetr-2024080100.onnx` and `va_260612.onnx` are unverified; `pretrained-model-assets.json` grants no rights to download, use, or redistribute those files, and administrators must verify rights independently before deployment.
