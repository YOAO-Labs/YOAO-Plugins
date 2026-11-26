# YOAO Ultralytics Trainer Plugin (`yoao.trainer.ultralytics`)

Official `yoao-trainer/v1` trainer plugin for Ultralytics YOLO `8.4.183`, supporting detection and instance segmentation across the 30 controlled `yolov8`, `yolo11`, and `yolo26` architectures (`n/s/m/l/x` × `detect/segment`).

## Capabilities

- **Protocol**: `yoao-trainer/v1` (`yoao-extension-manifest/v1`, `extension_point: "trainer"`).
- **Entrypoint**: `yoao-trainer-ultralytics` with subcommands:
  - `describe`: prints `TrainerCapabilitiesV1` JSON (30 architectures with pinned base-asset SHA-256 digests, Ultralytics `8.4.183` parameter schema, evaluator source SHA-256 descriptors, policy digests, and `serving_compatibility` with `format: null`).
  - `validate-config --spec <path>`: validates task, architecture, dataset contract, and cross-field rules (`optimizer`, `warmup_epochs`, `rect`, `close_mosaic`, `ultralytics-instance-rle-1` `workers=0`, `yolo26` `dfl`/`nms`/64M semantic class pixel cap) and prints `ValidateConfigResultV1` JSON (exit `0` even when `valid=false`).
  - `train --spec <path> --output <dir>`: runs offline fixed-batch training with offline font interception (`DejaVuSans.ttf` / `NotoSansCJK-Regular.ttc`), disabled telemetry, disabled first-epoch OOM batch halving, `aap_resume` historical-best checkpointing, `InstanceRLETrainer` support, retained-model FP32 tensor-identity verification, `input-membership.json`, monotonic `events.jsonl` (`TrainEventV1`), and atomic `result.json` (`TrainReceiptV1`).
  - `preview --spec <path> --output <dir>`: runs CPU-only deterministic augmentation preview (`RLIMIT_AS=16 GiB`, `RLIMIT_CPU=(30, 35)`, 45 s alarm, single-core affinity, `CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_initialized() is False`) with `NativeTrace`, emitting `preview-*.png` and `report.json` (`PreviewReportV1`).

## Building the OCI Image

Build from the `YOAO-Plugins` repository root with a BuildKit named context supplying the verified `yoao-plugin-sdk` wheel:

```sh
docker build \
  -f plugins/trainer/ultralytics/Dockerfile \
  --build-context first-party-inputs=.artifacts/python/first-party \
  -t yoao-trainer-ultralytics:1.0.0 \
  .
```

The container runs cleanly under arbitrary non-root `--user <uid>:<gid>` with `--network=none` and writable caches under `/tmp`.

## License

First-party plugin code is licensed under [MIT](LICENSE). The bundled container runtime pins `ultralytics==8.4.183`, which is licensed under [AGPL-3.0-only](AGPL-3.0.txt); the plugin manifest declares `AGPL-3.0-only` accordingly.
