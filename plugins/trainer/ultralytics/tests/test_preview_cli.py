"""Tests for CPU-only, bounded, deterministic Ultralytics augmentation preview."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import signal
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from yoao_plugin_sdk.trainer import (
    PreviewReportV1,
    PreviewSourceV1,
    PreviewSpecV1,
    digest_file,
)

from yoao_trainer_ultralytics import InputMismatchError, NativeRuntimeMismatch
from yoao_trainer_ultralytics.cli import main as cli_main
from yoao_trainer_ultralytics.policies import PREVIEW_LIMITS, PREVIEW_POLICY_DIGEST
from yoao_trainer_ultralytics.preview import (
    closed_mosaic,
    enforce_limits,
    run_preview_cli,
    verify_pool,
)
from tests.conftest import (
    install_fake_ultralytics_modules,
    make_fake_tensor,
    write_png,
)


def _make_preview_spec(input_dir: Path, *, count: int = 2, epoch: int = 1, policy_digest: str = PREVIEW_POLICY_DIGEST) -> tuple[PreviewSpecV1, Path]:
    sources: list[PreviewSourceV1] = []
    for i in range(2):
        img_rel = f"images/src_{i}.png"
        lbl_rel = f"labels/src_{i}.txt"
        img_path = write_png(input_dir / img_rel, 32, 32, (60 + i * 30, 120, 180))
        lbl_path = input_dir / lbl_rel
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        sources.append(
            PreviewSourceV1(
                task_id=f"task-{i}",
                filename=img_rel,
                sha256=digest_file(img_path),
                size_bytes=img_path.stat().st_size,
                width=32,
                height=32,
                box_count=1,
                label_path=lbl_rel,
                label_sha256=digest_file(lbl_path),
            )
        )
    spec = PreviewSpecV1(
        protocol_version="yoao-trainer/v1",
        preview_id="preview-test-1",
        framework_id="ultralytics",
        architecture_id="yolo11n",
        task="detect",
        config_digest="b" * 64,
        dataset_snapshot_digest="c" * 64,
        parameters={"epochs": 10, "batch": 2, "imgsz": 64, "close_mosaic": 2, "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0},
        native_values={"epochs": 10, "batch": 2, "imgsz": 64, "close_mosaic": 2, "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "multi_scale": 0.0},
        sources=sources,
        seed=42,
        epoch=epoch,
        count=count,
        policy_digest=policy_digest,
        input_digest="a" * 64,
        pool_summary={"classes": ["cat", "dog"]},
    )
    spec_path = input_dir / "preview-spec.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    return spec, spec_path


def _install_fake_preview_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_initialized: bool = False,
) -> None:
    mods = install_fake_ultralytics_modules(monkeypatch, version="8.4.183")
    ultralytics_mod = mods["ultralytics"]
    ultralytics_mod.__file__ = str(Path(__file__))  # type: ignore[attr-defined]

    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    torch_mod.set_num_threads = lambda n: None  # type: ignore[attr-defined]
    torch_mod.get_num_interop_threads = lambda: 1  # type: ignore[attr-defined]
    torch_mod.set_num_interop_threads = lambda n: None  # type: ignore[attr-defined]
    torch_mod.manual_seed = lambda seed: None  # type: ignore[attr-defined]
    torch_mod.device = lambda name: name  # type: ignore[attr-defined]
    torch_mod.cuda = SimpleNamespace(is_initialized=lambda: cuda_initialized)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    cfg_mod = ModuleType("ultralytics.cfg")
    cfg_mod.get_cfg = lambda default, overrides: SimpleNamespace(**{**default, **overrides})  # type: ignore[attr-defined]

    utils_mod = sys.modules["ultralytics.utils"]
    utils_mod.DEFAULT_CFG = {"mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}  # type: ignore[attr-defined]

    instance_mod = ModuleType("ultralytics.utils.instance")

    class Instances:
        def __init__(self, bboxes: np.ndarray) -> None:
            self.bboxes = bboxes

        def clip(self, w: int, h: int) -> None:
            self.bboxes = np.clip(self.bboxes, 0.0, 1.0)

        def remove_zero_area_boxes(self) -> np.ndarray:
            return np.ones((len(self.bboxes),), dtype=bool)

    instance_mod.Instances = Instances  # type: ignore[attr-defined]

    augment_mod = ModuleType("ultralytics.data.augment")

    class RandomPerspective:
        @staticmethod
        def box_candidates(b1: Any, b2: Any) -> np.ndarray:
            return np.ones((1,), dtype=bool)

    class Mosaic:
        def _cat_labels(self, labels: list[dict[str, Any]]) -> dict[str, Any]:
            return dict(labels[0])

    class MixUp:
        def apply_instances(self, labels: dict[str, Any], params: Any = None) -> dict[str, Any]:
            return labels

    class CutMix:
        def apply_instances(self, labels: dict[str, Any], params: Any = None) -> dict[str, Any]:
            return labels

    class Albumentations:
        def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
            return labels

    augment_mod.RandomPerspective = RandomPerspective  # type: ignore[attr-defined]
    augment_mod.Mosaic = Mosaic  # type: ignore[attr-defined]
    augment_mod.MixUp = MixUp  # type: ignore[attr-defined]
    augment_mod.CutMix = CutMix  # type: ignore[attr-defined]
    augment_mod.Albumentations = Albumentations  # type: ignore[attr-defined]

    build_mod = ModuleType("ultralytics.data.build")

    class FakePreviewDataset:
        def __init__(self, img_root: str, args: Any) -> None:
            self.files = sorted(Path(img_root).glob("*.png"))
            self.args = args
            self.mosaic_closed = False

        def __len__(self) -> int:
            return len(self.files)

        def close_mosaic(self, args: Any) -> None:
            self.mosaic_closed = True
            args.mosaic = 0.0

        def load_image(self, index: int) -> None:
            pass

        def get_image_and_label(self, index: int) -> dict[str, Any]:
            return {
                "im_file": str(self.files[index]),
                "cls": np.array([[0.0]], dtype=np.float32),
                "bboxes": np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32),
            }

        def __getitem__(self, index: int) -> dict[str, Any]:
            label = self.get_image_and_label(index)
            inst = Instances(np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32))
            inst.clip(self.args.imgsz, self.args.imgsz)
            inst.remove_zero_area_boxes()
            return label

        @staticmethod
        def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "img": make_fake_tensor(np.full((1, 3, 64, 64), 0.5, dtype=np.float32)),
                "bboxes": make_fake_tensor(items[0]["bboxes"]),
                "cls": make_fake_tensor(items[0]["cls"]),
            }

    build_mod.build_yolo_dataset = lambda args, img_root, batch, data, stride: FakePreviewDataset(img_root, args)  # type: ignore[attr-defined]

    models_mod = ModuleType("ultralytics.models")
    yolo_mod = ModuleType("ultralytics.models.yolo")
    detect_mod = ModuleType("ultralytics.models.yolo.detect")

    class DetectionTrainer:
        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            return batch

    detect_mod.DetectionTrainer = DetectionTrainer  # type: ignore[attr-defined]
    ultralytics_mod.YOLO = lambda path, task="detect": SimpleNamespace(model=SimpleNamespace(stride=make_fake_tensor(np.array([32]))))  # type: ignore[attr-defined]

    for name, mod in (
        ("ultralytics.cfg", cfg_mod),
        ("ultralytics.utils.instance", instance_mod),
        ("ultralytics.data.augment", augment_mod),
        ("ultralytics.data.build", build_mod),
        ("ultralytics.models", models_mod),
        ("ultralytics.models.yolo", yolo_mod),
        ("ultralytics.models.yolo.detect", detect_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def test_enforce_limits_requires_empty_cuda_visible_devices_and_applies_rlimits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(NativeRuntimeMismatch, match="CUDA_VISIBLE_DEVICES"):
        enforce_limits(apply_rlimits=False)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(NativeRuntimeMismatch, match="CUDA_VISIBLE_DEVICES"):
        enforce_limits(apply_rlimits=False)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    applied: list[tuple[int, tuple[int, int]]] = []
    alarms: list[int] = []
    monkeypatch.setattr(resource, "getrlimit", lambda name: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda name, val: applied.append((name, val)))
    monkeypatch.setattr(signal, "alarm", lambda sec: alarms.append(sec))
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1})
    pinned: list[set[int]] = []
    monkeypatch.setattr("os.sched_setaffinity", lambda pid, cpus: pinned.append(set(cpus)))

    enforce_limits(apply_rlimits=True)
    assert (resource.RLIMIT_AS, (PREVIEW_LIMITS["address_bytes"], PREVIEW_LIMITS["address_bytes"])) in applied
    assert (resource.RLIMIT_CPU, (PREVIEW_LIMITS["cpu_seconds"], PREVIEW_LIMITS["cpu_hard_seconds"])) in applied
    assert alarms == [PREVIEW_LIMITS["child_seconds"]]
    assert pinned == [{0}]


def test_verify_pool_and_closed_mosaic_boundaries(tmp_path: Path) -> None:
    spec, _ = _make_preview_spec(tmp_path / "pool")
    counts = verify_pool(spec, tmp_path / "pool")
    assert counts == {"task-0": 1, "task-1": 1}

    # Tampered image digest rejected
    (tmp_path / "pool" / "images" / "src_0.png").write_bytes(b"corrupt-png-bytes")
    with pytest.raises(InputMismatchError, match="bytes differ"):
        verify_pool(spec, tmp_path / "pool")

    assert closed_mosaic({"epochs": 10, "close_mosaic": 2}, {}, epoch=8) is False
    assert closed_mosaic({"epochs": 10, "close_mosaic": 2}, {}, epoch=9) is True
    assert closed_mosaic({"epochs": 10, "close_mosaic": 0}, {}, epoch=10) is False


def test_preview_cli_produces_pngs_and_report_and_blocks_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TRAINING_MODELS_DIR", str(tmp_path / "empty_models"))

    input_dir = tmp_path / "in"
    _, spec_path = _make_preview_spec(input_dir, count=2, epoch=9)
    out_dir = tmp_path / "out"
    _install_fake_preview_runtime(monkeypatch, cuda_initialized=False)

    rc = run_preview_cli(spec_path, out_dir, apply_rlimits=False)
    assert rc == 0
    report = PreviewReportV1.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report.status == "succeeded"
    assert report.cuda_initialized is False
    assert report.policy_digest == PREVIEW_POLICY_DIGEST
    assert report.details["mosaic_closed"] is True
    assert len(report.outputs) == 2
    for idx, out in enumerate(report.outputs):
        png = out_dir / f"preview-{idx}.png"
        assert png.is_file()
        assert out.filename == f"preview-{idx}.png"
        assert out.sha256 == digest_file(png)
        assert out.box_count == 1
        assert out.metadata["source_draws"] == [f"task-{idx}"]
        assert out.metadata["sources_read"] == [f"task-{idx}"]
        assert any(op["step"] == "clip" for op in out.metadata["operations"])

    # CUDA initialized unexpectedly fails with structured PreviewReportV1
    _install_fake_preview_runtime(monkeypatch, cuda_initialized=True)
    cuda_out = tmp_path / "cuda_out"
    assert run_preview_cli(spec_path, cuda_out, apply_rlimits=False) == 1
    failed = PreviewReportV1.model_validate_json((cuda_out / "report.json").read_text(encoding="utf-8"))
    assert failed.status == "failed"
    assert failed.cuda_initialized is True
    assert failed.failure_code == "runtime_mismatch"

    # Bad policy_digest fails cleanly
    bad_dir = tmp_path / "bad_policy"
    _, bad_spec = _make_preview_spec(bad_dir, policy_digest="0" * 64)
    bad_out = tmp_path / "bad_out"
    assert run_preview_cli(bad_spec, bad_out, apply_rlimits=False) == 1
    bad_rep = PreviewReportV1.model_validate_json((bad_out / "report.json").read_text(encoding="utf-8"))
    assert bad_rep.status == "failed"
    assert bad_rep.failure_code == "input_mismatch"


def _stub_preview_rlimits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource, "getrlimit", lambda name: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(resource, "setrlimit", lambda name, val: None)
    monkeypatch.setattr(signal, "alarm", lambda sec: 0)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    if hasattr(os, "sched_setaffinity"):
        monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None)


def test_cli_main_preview_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TRAINING_MODELS_DIR", str(tmp_path / "empty_models"))
    _stub_preview_rlimits(monkeypatch)
    _install_fake_preview_runtime(monkeypatch, cuda_initialized=False)

    _, spec_path = _make_preview_spec(tmp_path / "in", count=2, epoch=1)
    out_dir = tmp_path / "out"

    rc = cli_main(["preview", "--spec", str(spec_path), "--output", str(out_dir)])
    assert rc == 0
    report = PreviewReportV1.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report.status == "succeeded"
    assert report.cuda_initialized is False
    assert report.policy_digest == PREVIEW_POLICY_DIGEST
    assert len(report.outputs) == 2
    for idx, out in enumerate(report.outputs):
        png = out_dir / f"preview-{idx}.png"
        assert png.is_file()
        assert out.sha256 == digest_file(png)


def test_cli_main_preview_rejects_oversized_spec_before_reading(tmp_path: Path) -> None:
    oversized_spec = tmp_path / "oversized-spec.json"
    with oversized_spec.open("wb") as fh:
        fh.write(b"\xff\xfe{not-valid-json")
        fh.truncate(PREVIEW_LIMITS["report_bytes"] + 1)
    out_dir = tmp_path / "oversized_out"

    rc = cli_main(["preview", "--spec", str(oversized_spec), "--output", str(out_dir)])
    assert rc == 1
    report = PreviewReportV1.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report.status == "failed"
    assert report.failure_code == "input_mismatch"
    assert report.policy_digest == PREVIEW_POLICY_DIGEST
    assert report.cuda_initialized is False


def test_cli_main_preview_reports_malformed_json_with_policy_digest(tmp_path: Path) -> None:
    malformed_spec = tmp_path / "malformed-spec.json"
    malformed_spec.write_text("{not-valid-json", encoding="utf-8")
    out_dir = tmp_path / "malformed_out"

    rc = cli_main(["preview", "--spec", str(malformed_spec), "--output", str(out_dir)])
    assert rc == 1
    report = PreviewReportV1.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report.status == "failed"
    assert report.failure_code == "preview_failed"
    assert report.policy_digest == PREVIEW_POLICY_DIGEST
    assert report.cuda_initialized is False


def test_cli_main_preview_blocks_initialized_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TRAINING_MODELS_DIR", str(tmp_path / "empty_models"))
    _stub_preview_rlimits(monkeypatch)
    _install_fake_preview_runtime(monkeypatch, cuda_initialized=True)

    _, spec_path = _make_preview_spec(tmp_path / "in", count=1, epoch=1)
    out_dir = tmp_path / "cuda_out"

    rc = cli_main(["preview", "--spec", str(spec_path), "--output", str(out_dir)])
    assert rc == 1
    report = PreviewReportV1.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report.status == "failed"
    assert report.cuda_initialized is True
    assert report.failure_code == "runtime_mismatch"
    assert report.policy_digest == PREVIEW_POLICY_DIGEST
