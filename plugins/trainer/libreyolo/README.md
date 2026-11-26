# YOAO LibreYOLO Trainer Plugin (`yoao.trainer.libreyolo`)

Official MIT-licensed LibreYOLO9 (`libreyolo==1.6.0`) object detection trainer plugin implementing `yoao-trainer/v1` and `yoao-extension-manifest/v1`.

## License and Dependency Boundary

- First-party plugin code is licensed under [MIT](LICENSE).
- This plugin depends exclusively on `yoao-plugin-sdk`, `PyYAML`, and `libreyolo==1.6.0` (with PyTorch/CUDA supplied by the pinned container base image). It has **zero** `ultralytics` dependency or import anywhere in code, packaging, lockfiles, or container builds.
- Model weights are not bundled; controlled base weights (`LibreYOLO9{t,s,m,c}.pt`) remain administrator-approved assets mounted read-only at runtime.

## Supported Capabilities

- **Extension point**: `trainer` (`yoao-trainer/v1`)
- **Framework**: `libreyolo` `1.6.0`
- **Task**: `detect` (object detection only; `preview` is not declared)
- **Architectures**: `libreyolo9t`, `libreyolo9s`, `libreyolo9m`, `libreyolo9c` (`family: "libreyolo9"`, `end2end: false`)
- **Dataset contracts**: `ultralytics-yolo-1`, `ultralytics-bbox-1`
- **Serving compatibility**: `framework: "libreyolo"`, `format: "libreyolo-native-state-dict-1.0"`

## CLI (`yoao-trainer-libreyolo`)

```bash
# Emit TrainerCapabilitiesV1 JSON without GPU or network
yoao-trainer-libreyolo describe

# Validate a training configuration and emit ValidateConfigResultV1 JSON
yoao-trainer-libreyolo validate-config --spec /work/input/config-spec.json

# Run deterministic training, evaluation, and receipt generation
yoao-trainer-libreyolo train --spec /work/input/spec.json --output /work/output
```

## Security and Determinism Invariants

- All checkpoint reads enforce `torch.load(..., weights_only=True)`.
- `StrictResumeLogging` turns silent optimizer, EMA, RNG, or GradScaler restore warnings into hard failures.
- `CheckpointCallbacks` keeps `best.pt` free of `aap_resume` metadata while embedding verified historical best weights inside `checkpoint-NNNN.pt` alongside an atomic `checkpoint.json` pointer.
- `ObservedTrainer` and `ObservedValidator` verify exact dataset snapshot membership and emit `input-membership.json` when sample task identities are present.
- `output/libreyolo-data.yaml` is written inside the writable output directory without mutating the read-only dataset mount.
- Final evaluation strips `metrics/precision` and `metrics/recall` aliases from summaries and reports canonical `mAP50-95`, `mAP50`, `mAP75`, and `AR100`.

## Container Build

Build from the `YOAO-Plugins` repository root with the verified `yoao-plugin-sdk` wheel supplied via the `first-party-inputs` BuildKit context:

```bash
docker build \
  -f plugins/trainer/libreyolo/Dockerfile \
  --build-context first-party-inputs=.artifacts/python/first-party \
  -t yoao-trainer-libreyolo:1.0.0 \
  .
```

## Local CPU Tests

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  .artifacts/python/first-party/yoao_plugin_sdk-0.1.0-py3-none-any.whl \
  -e "plugins/trainer/libreyolo[dev]"
.venv/bin/pytest -q plugins/trainer/libreyolo/tests
```
