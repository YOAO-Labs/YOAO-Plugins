# RapidOCR Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-rapidocr-plugin` is an ONNX-based OCR inference plugin for YOAO built on RapidOCR `3.9.0` and ONNX Runtime `1.23.2`. It splits the `det -> cls -> rec` pipeline into reusable atomic models and an end-to-end composite model under YOAO ML Backend protocol v2.2.

## Capabilities and models

| `/setup` model ID | Task | Composition | Input | Output | Tunable parameters |
| --- | --- | --- | --- | --- | --- |
| `ocr-det` | `detect` | `atom` | `full_image` | `polygon` text regions | `box_thresh` (`0.5`), `unclip_ratio` (`1.6`) |
| `ocr-rec` | `ocr` | `atom` (runs internal `cls` orientation) | `crop` | `attributes.text`, `orientation` (`0`/`180`), `language` (`universal`/`en`) | None |
| `ocr-e2e` | `ocr` | `composite` (`det -> cls -> rec`) | `full_image` | `polygon` + `text`, `orientation`, `language` | `box_thresh` (`0.5`), `unclip_ratio` (`1.6`), `text_score` (`0.5`) |

- Textline orientation (`cls`, 0/180 degrees) is built into `ocr-rec` and `ocr-e2e` rather than exposed as a separate model. Platforms can also chain `ocr-det` into `ocr-rec` in a multi-stage pipeline.
- Select model variants via `context.model_variants` (`{version, size, lang}`; defaults to `v5`, `mobile`, `universal`):
  - **`ocr-det`**: PP-OCR `v5` (`mobile`, `server`) and `v6` (`tiny`, `small`, `medium`), language-agnostic.
  - **`ocr-rec` / `ocr-e2e`**: `universal` (Chinese and English) supports `v5` (`mobile`, `server`) and `v6` (`tiny`, `small`, `medium`); `en` (English) supports `v5` (`mobile`) only.
  - Internal `cls` selects `mobile` for `mobile`/`tiny`/`small` and `server` for `server`/`medium`.

## Model weights and assets

This plugin uses 13 ONNX files from RapidOCR `3.9.0` (5 detection, 2 orientation classification, 6 recognition). Weights are not bundled in this repository or built images.

- In plugin mode (`AAP_PLUGIN_TOKEN_FILE` set), verified weights reside under `AAP_PLUGIN_DATA_DIR/models` and are validated against `data/assets.json` generated with `python -m plugin_assets`.
- In standalone mode, weights are loaded from `RAPIDOCR_MODEL_DIR`.
- Operators can inspect the 13-file catalog and SHA-256 digests with `python -m pretrained_manifest`, or explicitly fetch weights in an authorized environment with `python -m download_models --models-dir /path/to/models`.

## Configuration

The container entrypoint is `python3 -m uvicorn main:app --host 0.0.0.0 --port 8005 --workers 1`.

| Variable | Default | Description |
| --- | --- | --- |
| `RAPIDOCR_MODEL_DIR` | GPU `/app/models`; CPU `/var/lib/aap-plugin/models` | Standalone weight root directory (overridden by `AAP_PLUGIN_DATA_DIR/models` in plugin mode) |
| `RAPIDOCR_DEVICE` | GPU `gpu`; CPU `cpu` | Execution device in standalone mode (plugin mode uses `AAP_PLUGIN_DEVICE`) |
| `RAPIDOCR_POOL_CAP` | `3` | Maximum composite engines in the pool (each holds `det`, `cls`, and `rec` ORT sessions) |
| `RAPIDOCR_BUILD_TIMEOUT` | `30` | Cold-start wait timeout in seconds |
| `RAPIDOCR_IDLE_UNLOAD_SECONDS` | `600` | Full-pool idle unload threshold in seconds (`<=0` disables) |
| `RAPIDOCR_IDLE_CHECK_INTERVAL` | `60` | Idle check interval in seconds |
| `LOG_LEVEL` | `INFO` | Python logging level |

Managed plugin mode reads `AAP_PLUGIN_TOKEN_FILE`, `AAP_PLUGIN_DATA_DIR`, `AAP_PLUGIN_DEVICE`, and `AAP_PLUGIN_ASSET_MANIFEST_SHA256`, plus GPU identity and verification bindings (`AAP_PLUGIN_REGISTRY_ID`, `AAP_ROUTE_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID`, `AAP_PLUGIN_GPU_KEYRING_FILE`) for CUDA deployments.

## Runtime requirements

- **Python**: `>=3.10`
- **Core dependencies**: `rapidocr==3.9.0`, `numpy==2.2.6`, `opencv-python-headless==5.0.0.93`
- **CPU vs. CUDA profiles** (mutually exclusive extras):
  - `cpu` (`Dockerfile.cpu`): `onnxruntime==1.23.2` with no NVIDIA wheels.
  - `cuda` (`Dockerfile`): `onnxruntime-gpu==1.23.2` and `nvidia-cudnn-cu12==9.10.2.21` on a CUDA `12.8` runtime base image (required for PP-OCRv6).
- **Fallback policy**: standalone `gpu` mode falls back to CPU only on explicit device errors. Managed CUDA plugin mode requires `CUDAExecutionProvider` as the primary provider on every session and rejects CPU fallback.

## License

First-party plugin code is licensed under [MIT](LICENSE). Upstream RapidOCR and third-party dependencies follow their own licenses.

According to [RapidOCR's model license notice](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/), `PP-OCRv6_det_small.onnx` and `PP-OCRv6_rec_small.onnx` are covered under Apache-2.0 (preserving PaddleOCR attribution and notices). The remaining 11 ONNX model files are outside that notice and require independent license verification before use or redistribution.
