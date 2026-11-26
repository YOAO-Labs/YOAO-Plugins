# LibreYOLO Trainer Plugin

English | [简体中文](README.zh-CN.md)

`yoao-trainer-libreyolo` (`yoao.trainer.libreyolo`) is the official MIT-licensed `yoao-trainer/v1` object detection trainer plugin built on LibreYOLO `1.6.0` (LibreYOLO9). It implements `yoao-extension-manifest/v1` with no Ultralytics dependency anywhere in code, packaging, or container images.

## Capabilities

- **Extension point and protocol**: `trainer` (`yoao-trainer/v1`, manifest capabilities `train`, `evaluate`, `export_weights`; `preview` is not declared)
- **Framework**: `libreyolo` `1.6.0`
- **Tasks**: `detect` (object detection only)
- **Architectures (4 total)**: `libreyolo9t`, `libreyolo9s`, `libreyolo9m`, `libreyolo9c` (`family: "libreyolo9"`, `end2end: false`)
- **Dataset contracts**: `ultralytics-yolo-1`, `ultralytics-bbox-1`
- **Serving compatibility**: `framework: "libreyolo"`, `min_version: "1.6.0"`, `format: "libreyolo-native-state-dict-1.0"`

## CLI subcommands

The container entrypoint is `yoao-trainer-libreyolo`:

```sh
# Emit TrainerCapabilitiesV1 JSON without GPU or network access
yoao-trainer-libreyolo describe

# Validate a training configuration and emit ValidateConfigResultV1 JSON
yoao-trainer-libreyolo validate-config --spec /work/input/config-spec.json

# Run offline deterministic training, evaluation, and receipt generation
yoao-trainer-libreyolo train --spec /work/input/spec.json --output /work/output
```

- Checkpoint loading enforces `torch.load(..., weights_only=True)`, and optimizer, EMA, RNG, or GradScaler restore mismatches fail fast rather than silently falling back.
- Training writes `libreyolo-data.yaml` under the output directory without mutating read-only dataset mounts, records `input-membership.json` when task identities are present, and reports canonical COCO metrics (`mAP50-95`, `mAP50`, `mAP75`, `AR100`).

## Runtime requirements

- **Python**: `>=3.11` (container image uses Python `3.12`)
- **ML stack**: `libreyolo==1.6.0`, `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132`, `pycocotools>=2.0.7,<3`, CUDA `13.2` (zero `ultralytics` dependency)
- **Resources and container**: `manifest.json` declares `requires_gpu: true` and `memory_mb: 8192`. The container runs as non-root UID/GID `10001` with `--network=none` and writable caches under `/tmp`.
- **Model weights**: base weights (`LibreYOLO9{t,s,m,c}.pt`) are not bundled in this repository or image; mount approved weights read-only at runtime.

## License

First-party plugin code is licensed under [MIT](LICENSE), and `manifest.json` declares `MIT` with no Ultralytics dependency. Third-party runtime dependencies (such as PyTorch, CUDA, `pycocotools`, and notices bundled with `libreyolo`) and pretrained weights remain subject to their own licenses.
