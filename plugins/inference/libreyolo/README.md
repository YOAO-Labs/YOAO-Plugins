# LibreYOLO Inference Plugin

English | [简体中文](README.zh-CN.md)

`yoao-libreyolo-plugin` is an object detection inference plugin for YOAO built on the native LibreYOLO `1.6.0` loader (`yoao-libreyolo-native`), with no Ultralytics dependency. It uses the YOAO detection plugin SDK (`yoao-inference-plugin-sdk`) and shared GPU lifecycle runtime.

## Capabilities and models

| `/setup` model ID | Task | Inputs | Outputs | Variants / checkpoints |
| --- | --- | --- | --- | --- |
| `detect` | `detect` | `full_image`, `crop` | `bbox` | Pretrained YOLOv9 `t` (default), `s`, `m`, `c`; managed checkpoints (`trained:<model_version_id>`) |

- Pretrained variants map to `LibreYOLO9t.pt`, `LibreYOLO9s.pt`, `LibreYOLO9m.pt`, and `LibreYOLO9c.pt`. The `t` variant must be prepared before starting the service.
- Managed checkpoints use the `libreyolo-native-state-dict-1.0` format and are validated against immutable training receipts before loading.
- Per-request parameters support `conf` (default `0.25`), `iou` (default `0.45`), `max_det` (default `300`), and optional `classes` filtering.

## Model weights and assets

No weights are bundled in this repository or built images, and inference never downloads weights at runtime. The plugin ships its versioned pretrained catalog in `yoao_libreyolo_plugin/controlled-model-assets.json`, pinning the four official YOLOv9 files and their SHA-256 digests.

To stage and verify an approved pretrained weight before serving:

```sh
yoao-libreyolo prepare-assets --size t --asset-dir /var/lib/aap-libreyolo/assets
```

Repeat with `--size s`, `m`, or `c` to expose additional variants. Catalog visibility does not grant download, enablement, or redistribution rights.

## Configuration

The default container entrypoint is `yoao-libreyolo serve --host 0.0.0.0 --port 8000` (running as UID/GID `10001`).

| Variable | Default | Description |
| --- | --- | --- |
| `AAP_PLUGIN_TOKEN_FILE` | Required | Path to private Bearer token file (at least 32 characters) |
| `AAP_PLUGIN_DEVICE` | `cpu` | Target device (`cpu` or `cuda:0`) |
| `AAP_PLUGIN_ASSET_DIR` | `/var/lib/aap-libreyolo/assets` | Directory containing verified pretrained weights |
| `AAP_PLUGIN_DATA_DIR` | `/var/lib/aap-libreyolo` | Private state directory (stores managed models under `managed/`) |
| `AAP_PLUGIN_ASSET_ORIGINS` | Empty | Comma-separated approved origins for managed model downloads |
| `AAP_PLUGIN_IMAGE_HOSTS` | Empty | Comma-separated approved hosts for remote image fetches |

Managed CUDA deployments also require `AAP_PLUGIN_REGISTRY_ID`, `AAP_PLUGIN_GPU_RESOURCE_ID`, `AAP_PLUGIN_GPU_UUID` (physical `GPU-...`), and `AAP_PLUGIN_GPU_KEYRING_FILE` (Ed25519 public verification keyring JSON). Managed GPU lifecycle is advertised only when approved qualification evidence is mounted via `AAP_PLUGIN_QUALIFICATION_FILE`, `AAP_PLUGIN_QUALIFICATION_SHA256`, `AAP_PLUGIN_OCI_ARTIFACT`, and `AAP_PLUGIN_IMAGE_ID`.

For loopback-only local development on CPU, run `yoao-libreyolo serve --local --device cpu --host 127.0.0.1 --port 8000`.

## Runtime requirements

- **Python**: `>=3.11`
- **ML stack**: `libreyolo==1.6.0`, `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132` (CUDA `13.2` baseline)
- **Precision and fallback**: runs in FP32 with half precision, AMP, and TF32 disabled. Failed CUDA checks do not fall back to CPU.

## License

First-party plugin and native loader source code is licensed under [MIT](LICENSE). Upstream [LibreYOLO](https://www.libreyolo.com/docs/licensing) identifies its library code and YOLO9 COCO weights as MIT, while bundled third-party notices (including DINOv3), PyTorch, NVIDIA runtime components, and each weight repository remain subject to their own terms and must be reviewed before use or redistribution.
