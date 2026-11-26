# Ultralytics YOLO Trainer Plugin

English | [简体中文](README.zh-CN.md)

`yoao-trainer-ultralytics` (`yoao.trainer.ultralytics`) is the official `yoao-trainer/v1` trainer plugin for Ultralytics YOLO `8.4.183`. It supports offline object detection and instance segmentation training, configuration validation, and CPU-only augmentation preview under `yoao-extension-manifest/v1`.

## Capabilities

- **Extension point and protocol**: `trainer` (`yoao-trainer/v1`, manifest capabilities `train`, `evaluate`, `preview`, `export_weights`)
- **Framework**: `ultralytics` `8.4.183`
- **Tasks**: `detect` (object detection) and `segment` (instance segmentation)
- **Architectures (30 total)**: `yolov8`, `yolo11`, and `yolo26` across scales `n`, `s`, `m`, `l`, and `x` for detection (`{family}{scale}`) and segmentation (`{family}{scale}-seg`). `yolo26` uses the end-to-end head (`nms=False`, no `dfl` loss gain).
- **Dataset contracts**: `ultralytics-yolo-1`, `ultralytics-bbox-1`, `ultralytics-simple-polygon-1`, `ultralytics-instance-rle-1`
- **Serving compatibility**: `framework: "ultralytics"`, `min_version: "8.4.183"`, `format: null` (segmentation uses the `aap-instance-rle-1` output contract)

## CLI subcommands

The container entrypoint is `yoao-trainer-ultralytics`:

```sh
# Emit TrainerCapabilitiesV1 JSON (architectures, base-asset SHA-256 catalog, parameter schema)
yoao-trainer-ultralytics describe

# Validate a training spec and emit ValidateConfigResultV1 JSON (exits 0 even when valid=false)
yoao-trainer-ultralytics validate-config --spec /work/input/config-spec.json

# Run offline fixed-batch training, evaluation, and receipt generation
yoao-trainer-ultralytics train --spec /work/input/spec.json --output /work/output

# Run CPU-only deterministic augmentation preview (preview-*.png and report.json)
yoao-trainer-ultralytics preview --spec /work/input/spec.json --output /work/output
```

- `train` runs offline with telemetry disabled, deterministic settings, resume-aware checkpointing, `input-membership.json`, monotonic `events.jsonl`, and atomic `result.json` receipts.
- `preview` enforces CPU-only execution (`CUDA_VISIBLE_DEVICES=""`) with memory, CPU time, and timeout limits.

## Runtime requirements

- **Python**: `>=3.11,<3.14` (container image uses Python `3.12`)
- **ML stack**: `ultralytics==8.4.183`, `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132`, CUDA `13.2`
- **Resources and container**: `manifest.json` declares `requires_gpu: true` and `memory_mb: 8192`. The container runs as non-root UID/GID `10001` with `--network=none` and writable caches under `/tmp`.
- **Model weights**: the 30 controlled base `.pt` weights are not bundled in this repository or image; mount approved weights read-only at runtime.

## License

First-party plugin code is licensed under [MIT](LICENSE). The container runtime pins `ultralytics==8.4.183`, which is licensed under [AGPL-3.0-only](AGPL-3.0.txt); `manifest.json` declares `AGPL-3.0-only` accordingly. Model weights and other third-party dependencies remain subject to their own terms.
