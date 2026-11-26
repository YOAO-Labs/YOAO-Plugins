"""End-to-end tests for the `train` CLI subcommand with mocked Ultralytics runtime."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import uuid

import numpy as np
import pytest
from yoao_plugin_sdk.trainer import (
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
)

from yoao_trainer_ultralytics.cli import main
from yoao_trainer_ultralytics.files import digest_file
from yoao_trainer_ultralytics.train import SUPERVISOR_OWNED_ARTIFACTS, run_train
from tests.conftest import (
    install_fake_ultralytics_modules,
    make_detect_dataset,
    make_fake_fonts,
    make_rle_dataset,
)


class _FakeOutOfMemoryError(RuntimeError):
    pass


def _install_fake_train_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_root: Path,
    classes: list[str],
    task: str = "detect",
    end2end: bool = False,
    is_rle: bool = False,
    raise_on_train: BaseException | None = None,
) -> None:
    """Install fake torch + ultralytics trainer/YOLO classes exercising callbacks and artifacts."""
    mods = install_fake_ultralytics_modules(monkeypatch, version="8.4.183")
    ultralytics_mod = mods["ultralytics"]

    # Fake torch with pickle-backed save/load and cuda.OutOfMemoryError
    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    torch_mod.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        OutOfMemoryError=_FakeOutOfMemoryError,
        is_initialized=lambda: False,
        memory_allocated=lambda: 0,
    )

    def _torch_load(source: Any, map_location: str = "cpu", weights_only: bool = False) -> Any:
        if isinstance(source, (str, Path)):
            return pickle.loads(Path(source).read_bytes())
        return pickle.loads(source.read())

    def _torch_save(obj: Any, stream: Any) -> None:
        if isinstance(stream, (str, Path)):
            Path(stream).write_bytes(pickle.dumps(obj))
        else:
            stream.write(pickle.dumps(obj))

    class _FakeArr:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = np.asarray(arr)

        def __getitem__(self, idx: Any) -> "_FakeArr":
            return _FakeArr(self._arr[idx])

        def float(self) -> np.ndarray:
            return self._arr.astype(np.float32)

    torch_mod.load = _torch_load  # type: ignore[attr-defined]
    torch_mod.save = _torch_save  # type: ignore[attr-defined]
    torch_mod.from_numpy = lambda arr: _FakeArr(arr)  # type: ignore[attr-defined]

    class _FakeTensorMask:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def gt_(self, thresh: float) -> "_FakeTensorMask":
            self._arr = self._arr > thresh
            return self

        def any(self) -> bool:
            return bool(self._arr.any())

    def _fake_interpolate(tensor: Any, size: tuple[int, int], mode: str = "bilinear", align_corners: bool = False) -> _FakeTensorMask:
        src = np.asarray(tensor)[0, 0]
        h, w = src.shape
        out_h, out_w = size
        ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5, 0.0, h - 1.0)
        xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5, 0.0, w - 1.0)
        y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
        wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
        top = (1.0 - wx) * src[np.ix_(y0, x0)] + wx * src[np.ix_(y0, x1)]
        bot = (1.0 - wx) * src[np.ix_(y1, x0)] + wx * src[np.ix_(y1, x1)]
        return _FakeTensorMask((1.0 - wy) * top + wy * bot)

    torch_mod.nn = SimpleNamespace(functional=SimpleNamespace(interpolate=_fake_interpolate))  # type: ignore[attr-defined]

    torch_utils_mod = ModuleType("torch.utils")
    torch_utils_data_mod = ModuleType("torch.utils.data")

    class FakeDataLoader:
        def __init__(self, dataset: Any, batch_size: int = 1, num_workers: int = 0, **kwargs: Any) -> None:
            self.dataset = dataset
            self.batch_size = batch_size
            self.num_workers = num_workers

    torch_utils_data_mod.DataLoader = FakeDataLoader  # type: ignore[attr-defined]
    torch_utils_mod.data = torch_utils_data_mod  # type: ignore[attr-defined]
    torch_mod.utils = torch_utils_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_mod)

    # Fake ultralytics.models.yolo.{detect,segment}
    models_mod = ModuleType("ultralytics.models")
    yolo_mod = ModuleType("ultralytics.models.yolo")
    detect_mod = ModuleType("ultralytics.models.yolo.detect")
    segment_mod = ModuleType("ultralytics.models.yolo.segment")
    loss_mod = ModuleType("ultralytics.utils.loss")

    class FakeBaseTrainer:
        def __init__(self, overrides: dict[str, Any]) -> None:
            self.args = SimpleNamespace(**overrides)
            self.save_dir = Path(overrides["project"]) / overrides["name"]
            self.save_dir.mkdir(parents=True, exist_ok=True)
            weights_dir = self.save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            self.last = weights_dir / "last.pt"
            self.best = weights_dir / "best.pt"
            self.epochs = int(overrides["epochs"])
            self.batch_size = int(overrides["batch"])
            self.start_epoch = 0
            self.epoch = 0
            self.resume = bool(overrides.get("resume"))
            self.callbacks: dict[str, list[Any]] = {}
            self.tloss = None
            self.metrics: dict[str, float] = {}
            self.optimizer = SimpleNamespace(param_groups=[{"lr": overrides["lr0"], "params": []}])
            self.model = SimpleNamespace(
                named_parameters=lambda: [("layer.weight", SimpleNamespace(requires_grad=True))],
                names={i: c for i, c in enumerate(classes)},
            )
            train_imgs = [str(p) for p in sorted((dataset_root / "images" / "train").glob("*"))]
            val_imgs = [str(p) for p in sorted((dataset_root / "images" / "val").glob("*"))]
            self.train_loader = FakeDataLoader(
                SimpleNamespace(im_files=train_imgs, rect=False, imgsz=overrides["imgsz"]),
                batch_size=self.batch_size,
                num_workers=overrides["workers"],
            )
            self.test_loader = FakeDataLoader(
                SimpleNamespace(im_files=val_imgs, rect=False, imgsz=overrides["imgsz"]),
                batch_size=self.batch_size,
                num_workers=overrides["workers"],
            )
            self.validator = SimpleNamespace(
                args=self.args,
                device="cpu",
                end2end=end2end,
                dataloader=self.test_loader,
            )

        def check_resume(self, overrides: dict[str, Any]) -> None:
            pass

        def run_callbacks(self, event: str) -> None:
            pass

        def preprocess_batch(self, batch: Any) -> Any:
            return batch

        def final_eval(self) -> Any:
            return None

    class DetectionTrainer(FakeBaseTrainer):
        pass

    class SegmentationTrainer(FakeBaseTrainer):
        pass

    class SegmentationValidator:
        def _prepare_batch(self, si: int, batch: Any) -> Any:
            return batch

    class v8SegmentationLoss:
        def loss(self, preds: Any, batch: Any) -> Any:
            return preds

    detect_mod.DetectionTrainer = DetectionTrainer  # type: ignore[attr-defined]
    segment_mod.SegmentationTrainer = SegmentationTrainer  # type: ignore[attr-defined]
    segment_mod.SegmentationValidator = SegmentationValidator  # type: ignore[attr-defined]
    loss_mod.v8SegmentationLoss = v8SegmentationLoss  # type: ignore[attr-defined]

    class FakeYOLO:
        def __init__(self, model_path: str, task: str = "detect") -> None:
            self.model_path = model_path
            self.task = task
            self.model = SimpleNamespace(names={i: c for i, c in enumerate(classes)})
            self._callbacks: dict[str, list[Any]] = {}
            self.trainer: Any = None

        def add_callback(self, event: str, fn: Any) -> None:
            self._callbacks.setdefault(event, []).append(fn)

        def _fire(self, event: str, arg: Any) -> None:
            for fn in self._callbacks.get(event, []):
                fn(arg)

        def train(self, trainer: type, **kwargs: Any) -> Any:
            if raise_on_train is not None:
                raise raise_on_train
            overrides = {**kwargs, "model": self.model_path}
            t = trainer(overrides)
            self.trainer = t
            t.check_resume(overrides)
            if is_rle and hasattr(t, "get_dataloader"):
                t.train_loader = t.get_dataloader(
                    str(dataset_root / "images" / "train"),
                    batch_size=t.batch_size,
                    rank=0,
                    mode="train",
                )
                t.test_loader = t.get_dataloader(
                    str(dataset_root / "images" / "val"),
                    batch_size=t.batch_size,
                    rank=0,
                    mode="val",
                )
                t.validator.dataloader = t.test_loader

            self._fire("on_train_start", t)
            for ep in range(t.epochs):
                t.epoch = ep
                t.run_callbacks("on_train_epoch_start")
                t.preprocess_batch({"img": np.zeros((t.batch_size, 3, 32, 32), dtype=np.float32)})
                self._fire("on_train_batch_end", t)
                self._fire("on_val_start", t.validator)
                t.metrics = (
                    {"metrics/mAP50(B)": 0.7 + 0.05 * ep, "metrics/mAP50(M)": 0.65 + 0.05 * ep}
                    if task == "segment"
                    else {"metrics/mAP50(B)": 0.7 + 0.05 * ep}
                )
                self._fire("on_val_end", t.validator)
                fitness = 0.7 + 0.05 * ep
                blob = {
                    "version": "8.4.183",
                    "train_args": {"task": "segment" if task == "segment" else "detect"},
                    "ema": SimpleNamespace(names={i: c for i, c in enumerate(classes)}),
                    "epoch": ep,
                    "best_fitness": fitness,
                    "train_metrics": {"fitness": fitness},
                    "optimizer": {"step": ep + 1},
                }
                t.last.write_bytes(pickle.dumps(blob))
                t.best.write_bytes(pickle.dumps(blob))
                self._fire("on_model_save", t)
                self._fire("on_fit_epoch_end", t)

            t.final_eval()
            if task == "segment":
                return SimpleNamespace(
                    box=SimpleNamespace(ap_class_index=list(range(len(classes)))),
                    class_result=lambda idx: (0.8, 0.75, 0.78, 0.52, 0.76, 0.72, 0.74, 0.49),
                    results_dict={
                        "metrics/mAP50(B)": 0.78,
                        "metrics/mAP50-95(B)": 0.52,
                        "metrics/mAP50(M)": 0.74,
                        "metrics/mAP50-95(M)": 0.49,
                    },
                )
            return SimpleNamespace(
                box=SimpleNamespace(ap_class_index=list(range(len(classes)))),
                class_result=lambda idx: (0.82, 0.77, 0.80, 0.55),
                results_dict={"metrics/mAP50(B)": 0.80, "metrics/mAP50-95(B)": 0.55},
            )

    ultralytics_mod.YOLO = FakeYOLO  # type: ignore[attr-defined]
    for name, mod in (
        ("ultralytics.models", models_mod),
        ("ultralytics.models.yolo", yolo_mod),
        ("ultralytics.models.yolo.detect", detect_mod),
        ("ultralytics.models.yolo.segment", segment_mod),
        ("ultralytics.utils.loss", loss_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _write_train_spec(
    path: Path,
    *,
    dataset_dir: Path,
    classes_digest: str,
    task: str = "detect",
    architecture_id: str = "yolo11n",
    dataset_contract: str = "ultralytics-bbox-1",
    parameters: dict[str, Any] | None = None,
) -> Path:
    spec = TrainSpecV1(
        protocol_version="yoao-trainer/v1",
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        framework_id="ultralytics",
        architecture_id=architecture_id,
        task=task,
        dataset_contract=dataset_contract,
        dataset_dir=str(dataset_dir),
        dataset_snapshot_digest="e" * 64,
        classes_digest=classes_digest,
        config_digest="f" * 64,
        parameters=parameters or {"epochs": 2, "batch": 1, "imgsz": 128, "pretrained": False},
    )
    path.write_text(spec.model_dump_json(), encoding="utf-8")
    return path


def test_train_cli_detect_emits_monotonic_events_and_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fake_fonts(tmp_path, monkeypatch)
    ds = make_detect_dataset(tmp_path / "dataset")
    _install_fake_train_runtime(monkeypatch, dataset_root=ds["root"], classes=ds["classes"], task="detect")
    monkeypatch.setattr("yoao_trainer_ultralytics.train.validate_architecture", None)

    spec_path = _write_train_spec(
        tmp_path / "train-spec.json",
        dataset_dir=ds["root"],
        classes_digest=ds["classes_digest"],
        task="detect",
        architecture_id="yolo11n",
        dataset_contract="ultralytics-bbox-1",
        parameters={"epochs": 2, "batch": 1, "imgsz": 128, "pretrained": False, "workers": 2},
    )
    out_dir = tmp_path / "output"
    rc = main(["train", "--spec", str(spec_path), "--output", str(out_dir)])
    assert rc == 0

    events_lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [TrainEventV1.model_validate_json(line) for line in events_lines]
    assert [ev.seq for ev in events] == list(range(1, len(events) + 1))
    assert any(ev.kind == "epoch" and ev.epoch == 2 for ev in events)
    assert any(ev.kind == "checkpoint" for ev in events)

    receipt = TrainReceiptV1.model_validate_json((out_dir / "result.json").read_text(encoding="utf-8"))
    assert receipt.status == "succeeded"
    assert receipt.serving_compatibility is not None
    assert receipt.serving_compatibility.framework == "ultralytics"
    assert receipt.serving_compatibility.format is None

    filenames = {art.filename for art in receipt.artifacts}
    assert not (filenames & SUPERVISOR_OWNED_ARTIFACTS)
    assert {"best.pt", "checkpoint-0002.pt", "metrics.json", "native-config.json", "input-membership.json"} == filenames

    by_name = {art.filename: art for art in receipt.artifacts}
    assert by_name["best.pt"].role == "weights"
    assert by_name["checkpoint-0002.pt"].role == "checkpoint"
    assert by_name["metrics.json"].role == "metrics"
    assert by_name["input-membership.json"].role == "input_membership"
    for art in receipt.artifacts:
        assert digest_file(out_dir / art.filename) == art.sha256


def test_train_cli_rle_segment_yolo26_produces_preflight_and_end2end_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fake_fonts(tmp_path, monkeypatch)
    ds = make_rle_dataset(tmp_path / "rle_ds", width=64, height=64)
    _install_fake_train_runtime(
        monkeypatch,
        dataset_root=ds["root"],
        classes=ds["classes"],
        task="segment",
        end2end=True,
        is_rle=True,
    )
    monkeypatch.setattr("yoao_trainer_ultralytics.train.validate_architecture", None)

    spec_path = _write_train_spec(
        tmp_path / "rle-spec.json",
        dataset_dir=ds["root"],
        classes_digest=ds["classes_digest"],
        task="segment",
        architecture_id="yolo26n-seg",
        dataset_contract="ultralytics-instance-rle-1",
        parameters={"epochs": 1, "batch": 1, "imgsz": 128, "pretrained": False, "workers": 0, "nms": False},
    )
    out_dir = tmp_path / "rle_out"
    rc = main(["train", "--spec", str(spec_path), "--output", str(out_dir)])
    assert rc == 0

    receipt = TrainReceiptV1.model_validate_json((out_dir / "result.json").read_text(encoding="utf-8"))
    assert receipt.status == "succeeded"
    assert receipt.serving_compatibility is not None
    assert receipt.serving_compatibility.framework == "ultralytics"
    assert receipt.serving_compatibility.format is None
    assert receipt.serving_compatibility.training_runtime["end2end"] is True
    assert receipt.serving_compatibility.training_runtime["nms"] is False

    filenames = {art.filename for art in receipt.artifacts}
    assert not (filenames & SUPERVISOR_OWNED_ARTIFACTS)
    assert {
        "best.pt",
        "checkpoint-0001.pt",
        "metrics.json",
        "native-config.json",
        "input-membership.json",
        "rle-consumer.json",
        "rle-train-preflight.json",
        "rle-val-preflight.json",
    } == filenames
    # Excluding input_membership (embedded by supervisor), uploaded plugin artifacts stay <= 7
    uploaded_count = sum(1 for art in receipt.artifacts if art.role != "input_membership")
    assert uploaded_count <= 7

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mask_consumer"]["bridge"] == "ultralytics-instance-rle-1"
    assert set(metrics["mask_resolution"]) == {"train", "val"}


def test_train_cli_emits_structured_failure_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fake_fonts(tmp_path, monkeypatch)
    ds = make_detect_dataset(tmp_path / "ds")

    # 1. Config mismatch (unknown param)
    bad_cfg_spec = _write_train_spec(
        tmp_path / "bad-cfg.json",
        dataset_dir=ds["root"],
        classes_digest=ds["classes_digest"],
        parameters={"bogus_option": 123},
    )
    out_cfg = tmp_path / "out_cfg"
    assert main(["train", "--spec", str(bad_cfg_spec), "--output", str(out_cfg)]) == 1
    rec_cfg = TrainReceiptV1.model_validate_json((out_cfg / "result.json").read_text(encoding="utf-8"))
    assert rec_cfg.status == "failed"
    assert rec_cfg.failure_code == "config_mismatch"

    # 2. CUDA OOM mapped to cuda_oom
    _install_fake_train_runtime(
        monkeypatch,
        dataset_root=ds["root"],
        classes=ds["classes"],
        raise_on_train=_FakeOutOfMemoryError("CUDA out of memory"),
    )
    monkeypatch.setattr("yoao_trainer_ultralytics.train.validate_architecture", None)
    oom_spec = _write_train_spec(
        tmp_path / "oom.json",
        dataset_dir=ds["root"],
        classes_digest=ds["classes_digest"],
    )
    out_oom = tmp_path / "out_oom"
    assert main(["train", "--spec", str(oom_spec), "--output", str(out_oom)]) == 1
    rec_oom = TrainReceiptV1.model_validate_json((out_oom / "result.json").read_text(encoding="utf-8"))
    assert rec_oom.status == "failed"
    assert rec_oom.failure_code == "cuda_oom"

    # 3. Runtime version mismatch
    install_fake_ultralytics_modules(monkeypatch, version="8.4.0")
    out_rt = tmp_path / "out_rt"
    assert main(["train", "--spec", str(oom_spec), "--output", str(out_rt)]) == 1
    rec_rt = TrainReceiptV1.model_validate_json((out_rt / "result.json").read_text(encoding="utf-8"))
    assert rec_rt.status == "failed"
    assert rec_rt.failure_code == "runtime_mismatch"

    # 4. Input mismatch (tampered dataset image)
    (ds["root"] / "images" / "train" / "sample_0.png").write_bytes(b"corrupted")
    out_in = tmp_path / "out_in"
    assert main(["train", "--spec", str(oom_spec), "--output", str(out_in)]) == 1
    rec_in = TrainReceiptV1.model_validate_json((out_in / "result.json").read_text(encoding="utf-8"))
    assert rec_in.status == "failed"
    assert rec_in.failure_code == "input_mismatch"
