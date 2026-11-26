"""End-to-end `train` CLI tests with mocked libreyolo/torch runtimes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import uuid

import pytest
from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    canonical_json_digest,
)

from yoao_trainer_libreyolo import SUPERVISOR_OWNED_ARTIFACTS
from yoao_trainer_libreyolo.catalog import parameter_schema
from yoao_trainer_libreyolo.cli import main
from yoao_trainer_libreyolo.files import digest_file


def _build_dataset(root: Path) -> tuple[Path, dict[str, Any], list[str]]:
    ds = root / "dataset"
    ds.mkdir()
    classes = ["car", "person"]
    (ds / "data.yaml").write_text(
        'train: images/train\nval: images/val\nnames: {"0": "car", "1": "person"}\n',
        encoding="utf-8",
    )
    (ds / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    samples = []
    for idx, (split, lbl_text) in enumerate(
        [("train", ""), ("val", "0 0.5 0.5 0.2 0.2\n")], start=1
    ):
        img = ds / f"images/{split}/img_{idx}.png"
        lbl = ds / f"labels/{split}/img_{idx}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(f"img-{split}-{idx}".encode("utf-8"))
        lbl.write_text(lbl_text, encoding="utf-8")
        samples.append(
            {
                "task_id": str(uuid.UUID(int=idx)),
                "split": split,
                "image_path": f"images/{split}/img_{idx}.png",
                "label_path": f"labels/{split}/img_{idx}.txt",
                "sha256": digest_file(img),
                "label_sha256": digest_file(lbl),
                "width": 64,
                "height": 64,
                "size_bytes": img.stat().st_size,
            }
        )
    manifest: dict[str, Any] = {
        "schema": 3,
        "task": "detect",
        "media_type": "image",
        "format": "ultralytics-yolo",
        "geometry": "normalized-bbox",
        "bridge": "ultralytics-bbox-1",
        "classes": classes,
        "validation_digest": "e" * 64,
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": "f" * 64,
        "samples": samples,
    }
    manifest["content_digest"] = canonical_json_digest(manifest)
    (ds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds, manifest, classes


def _install_mock_runtime(
    monkeypatch,
    tmp_path: Path,
    ds_dir: Path,
    *,
    libreyolo_version: str = "1.6.0",
    eval_backend: str | None = "pycocotools",
    fail_mode: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "cv2_threads": None,
        "torch_threads": None,
        "torch_interop": None,
        "deterministic": None,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "calls": [],
        "load_calls": [],
        "observed_val_paths": [str(ds_dir / "images/val/img_2.png")],
    }

    # Fake cv2
    fake_cv2 = ModuleType("cv2")
    fake_cv2.setNumThreads = lambda n: state.__setitem__("cv2_threads", n)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    # Fake torch
    fake_torch = ModuleType("torch")

    class FakeCudaOOM(RuntimeError):
        pass

    fake_cuda = SimpleNamespace(OutOfMemoryError=FakeCudaOOM)
    fake_cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    fake_torch.backends = SimpleNamespace(cudnn=fake_cudnn)  # type: ignore[attr-defined]
    fake_torch.set_num_threads = lambda n: state.__setitem__("torch_threads", n)  # type: ignore[attr-defined]
    fake_torch.set_num_interop_threads = lambda n: state.__setitem__("torch_interop", n)  # type: ignore[attr-defined]

    def set_det(flag: bool) -> None:
        state["deterministic"] = flag
        state["cudnn_benchmark"] = fake_cudnn.benchmark
        state["cudnn_deterministic"] = fake_cudnn.deterministic

    fake_torch.use_deterministic_algorithms = set_det  # type: ignore[attr-defined]

    def fake_load(source, *, map_location=None, weights_only=None):
        state["load_calls"].append(
            {"source": str(source), "map_location": map_location, "weights_only": weights_only}
        )
        assert weights_only is True, "weights_only=True must be enforced"
        if isinstance(source, (str, Path)):
            return pickle.loads(Path(source).read_bytes())
        return pickle.loads(source.read())

    def fake_save(obj, target):
        raw = pickle.dumps(obj)
        if isinstance(target, (str, Path)):
            Path(target).write_bytes(raw)
        else:
            target.write(raw)

    fake_torch.load = fake_load  # type: ignore[attr-defined]
    fake_torch.save = fake_save  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Fake pycocotools files for descriptor hashing
    fake_pycoco_dir = tmp_path / "fake_pycocotools"
    fake_pycoco_dir.mkdir(exist_ok=True)
    fake_pycoco = ModuleType("pycocotools")
    for sub in ("cocoeval", "coco", "mask"):
        sub_file = fake_pycoco_dir / f"{sub}.py"
        sub_file.write_text(f"# fake {sub}\n", encoding="utf-8")
        sub_mod = ModuleType(f"pycocotools.{sub}")
        sub_mod.__file__ = str(sub_file)
        setattr(fake_pycoco, sub, sub_mod)
        monkeypatch.setitem(sys.modules, f"pycocotools.{sub}", sub_mod)
    monkeypatch.setitem(sys.modules, "pycocotools", fake_pycoco)

    # Fake libreyolo package
    fake_libre_dir = tmp_path / "fake_libreyolo"
    fake_libre_dir.mkdir(exist_ok=True)
    libre_init = fake_libre_dir / "__init__.py"
    libre_init.write_text("# fake libreyolo\n", encoding="utf-8")

    fake_libre = ModuleType("libreyolo")
    fake_libre.__version__ = libreyolo_version  # type: ignore[attr-defined]
    fake_libre.__file__ = str(libre_init)

    fake_utils = ModuleType("libreyolo.utils")
    fake_ser = ModuleType("libreyolo.utils.serialization")

    def validate_checkpoint_metadata(blob, *, strict=True):
        if not isinstance(blob, dict):
            raise TypeError("checkpoint must be dict")
        for key in ("task", "model_family", "libreyolo_version", "size", "names"):
            if key not in blob:
                raise ValueError(f"missing {key}")

    fake_ser.validate_checkpoint_metadata = validate_checkpoint_metadata  # type: ignore[attr-defined]
    fake_utils.serialization = fake_ser  # type: ignore[attr-defined]
    fake_libre.utils = fake_utils  # type: ignore[attr-defined]

    fake_val_pkg = ModuleType("libreyolo.validation")
    fake_det_val = ModuleType("libreyolo.validation.detection_validator")

    class DetectionValidator:
        def _setup_dataloader(self):
            return SimpleNamespace(
                dataset=SimpleNamespace(img_files=list(state["observed_val_paths"]))
            )

    fake_det_val.DetectionValidator = DetectionValidator  # type: ignore[attr-defined]
    fake_val_pkg.detection_validator = fake_det_val  # type: ignore[attr-defined]
    fake_libre.validation = fake_val_pkg  # type: ignore[attr-defined]

    class FakeTrainer:
        def __init__(self) -> None:
            self.train_loader = SimpleNamespace(
                dataset=SimpleNamespace(
                    dataset=SimpleNamespace(
                        img_files=[str(ds_dir / "images/train/img_1.png")]
                    )
                )
            )
            self._epoch_validator = SimpleNamespace(
                dataloader=SimpleNamespace(
                    dataset=SimpleNamespace(
                        img_files=[str(ds_dir / "images/val/img_2.png")]
                    )
                )
            )

    class FakeLibreYOLO:
        task = "detect"
        names = {0: "car", 1: "person"}
        last_eval_backend = eval_backend

        def __init__(self, path: str, device: str) -> None:
            self.path = path
            self.device = device
            self.validator_class = DetectionValidator
            state["calls"].append(("load", path, device))

        def _trainer_class(self):
            return FakeTrainer

        def train(self, **kw: Any) -> None:
            state["calls"].append(("train", kw))
            if fail_mode == "cuda_oom":
                raise FakeCudaOOM("CUDA out of memory while allocating tensor")
            if fail_mode == "resume_warning":
                logging.getLogger("libreyolo.training.trainer").warning(
                    "Could not restore RNG state: mismatched device"
                )
                return
            self._trainer_class()()
            save_dir = Path(kw["project"]) / kw["name"]
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            config = {
                key: value
                for key, value in kw.items()
                if key not in {"callbacks", "loggers", "cuda_graph", "compile"}
            }
            config.update(
                resume=bool(kw["resume"]),
                num_classes=2,
                size="t",
                data_dir=None,
            )
            kw["callbacks"].on_train_start(
                SimpleNamespace(
                    save_dir=str(save_dir),
                    start_epoch=1,
                    task="detect",
                    config=config,
                )
            )
            ckpt_blob = {
                "model": {"weight": [1.0]},
                "task": "detect",
                "model_family": "yolo9",
                "libreyolo_version": "1.6.0",
                "size": "t",
                "names": ["car", "person"],
                "epoch": 0,
                "best_epoch": 1,
                "optimizer": {"state": {}, "param_groups": [{}]},
                "rng_state": {"torch": b"rng"},
            }
            (weights_dir / "last.pt").write_bytes(pickle.dumps(ckpt_blob))
            (weights_dir / "best.pt").write_bytes(pickle.dumps(ckpt_blob))
            kw["callbacks"].on_train_epoch_end(
                SimpleNamespace(
                    epoch=1,
                    save_dir=str(save_dir),
                    val_metrics={
                        "metrics/mAP50": 0.55,
                        "metrics/mAP50-95": 0.35,
                        "metrics/precision": 0.35,
                        "metrics/recall": 0.65,
                    },
                    train_loss=0.42,
                )
            )

        def val(self, **kw: Any) -> dict[str, float]:
            state["calls"].append(("val", kw))
            if fail_mode == "missing_final_val_member":
                state["observed_val_paths"] = []
            validator = self.validator_class()
            validator._setup_dataloader()
            return {
                "metrics/precision": 0.35,
                "metrics/recall": 0.65,
                "metrics/mAP50-95": 0.35,
                "metrics/mAP50": 0.55,
                "metrics/mAP75": 0.25,
                "metrics/AR100": 0.65,
            }

    fake_libre.LibreYOLO = FakeLibreYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libreyolo", fake_libre)
    monkeypatch.setitem(sys.modules, "libreyolo.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "libreyolo.utils.serialization", fake_ser)
    monkeypatch.setitem(sys.modules, "libreyolo.validation", fake_val_pkg)
    monkeypatch.setitem(
        sys.modules, "libreyolo.validation.detection_validator", fake_det_val
    )
    return state


def _make_train_spec(
    tmp_path: Path,
    ds_dir: Path,
    manifest: dict[str, Any],
    classes: list[str],
    *,
    parameters: dict[str, Any] | None = None,
    use_premerged_effective: bool = False,
) -> Path:
    base_weights = tmp_path / "LibreYOLO9t.pt"
    base_weights.write_bytes(b"base-weights")
    schema = parameter_schema()
    if use_premerged_effective:
        params: dict[str, Any] = {
            field["key"]: field["default"] for field in schema["fields"]
        }
        params.update(epochs=2, batch=1, imgsz=128, optimizer="adamw", no_aug_epochs=0)
    else:
        params = parameters if parameters is not None else {"epochs": 2, "batch": 1, "imgsz": 128}
    spec = TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        framework_id="libreyolo",
        task="detect",
        architecture_id="libreyolo9t",
        dataset_contract="ultralytics-yolo-1",
        dataset_dir=str(ds_dir),
        dataset_snapshot_digest=manifest["content_digest"],
        config_digest="a" * 64,
        classes_digest=canonical_json_digest(classes),
        schema_id=schema["schema_id"],
        schema_digest=schema["schema_digest"],
        parameters=params,
        base_weights_path=str(base_weights),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    return spec_path


@pytest.mark.parametrize("eval_backend", ["pycocotools", None])
@pytest.mark.parametrize("use_premerged_effective", [False, True])
def test_train_cli_end_to_end_emits_valid_events_artifacts_and_receipt(
    tmp_path: Path,
    monkeypatch,
    eval_backend: str | None,
    use_premerged_effective: bool,
) -> None:
    ds_dir, manifest, classes = _build_dataset(tmp_path)
    state = _install_mock_runtime(
        monkeypatch, tmp_path, ds_dir, eval_backend=eval_backend
    )
    spec_path = _make_train_spec(
        tmp_path,
        ds_dir,
        manifest,
        classes,
        use_premerged_effective=use_premerged_effective,
    )
    output_dir = tmp_path / "output"

    rc = main(["train", "--spec", str(spec_path), "--output", str(output_dir)])
    assert rc == 0

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["LIBREYOLO_FASTER_COCO_EVAL"] == "0"
    assert state["cv2_threads"] == 1
    assert state["torch_threads"] == 4
    assert state["torch_interop"] == 4
    assert state["deterministic"] is True
    assert state["cudnn_benchmark"] is False
    assert state["cudnn_deterministic"] is True
    assert all(call["weights_only"] is True for call in state["load_calls"])

    # Events strictly increase starting at seq=1 and strip precision/recall aliases.
    event_lines = (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [TrainEventV1.model_validate(json.loads(line)) for line in event_lines]
    assert [ev.seq for ev in events] == list(range(1, len(events) + 1))
    assert events[0].kind == "phase" and events[0].phase == "preparing"
    assert any(ev.kind == "phase" and ev.phase == "training" for ev in events)
    assert any(ev.kind == "phase" and ev.phase == "evaluating" for ev in events)
    epoch_ev = next(ev for ev in events if ev.kind == "epoch")
    assert epoch_ev.epoch == 1
    assert "metrics/precision" not in epoch_ev.metrics
    assert "metrics/recall" not in epoch_ev.metrics
    assert epoch_ev.metrics["train/loss"] == 0.42

    # Receipt validates cleanly with ZERO supervisor-owned artifact collisions.
    receipt = TrainReceiptV1.model_validate(
        json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    )
    assert receipt.status == "succeeded"
    assert receipt.serving_compatibility is not None
    assert receipt.serving_compatibility.framework == "libreyolo"
    assert receipt.serving_compatibility.format == "libreyolo-native-state-dict-1.0"

    by_role = {item.role: item for item in receipt.artifacts}
    assert set(by_role) == {
        "weights",
        "checkpoint",
        "metrics",
        "config",
        "input_membership",
    }
    for item in receipt.artifacts:
        assert item.filename not in SUPERVISOR_OWNED_ARTIFACTS
        artifact_path = output_dir / item.filename
        assert artifact_path.is_file()
        assert digest_file(artifact_path) == item.sha256
        assert artifact_path.stat().st_size == item.size_bytes

    # best.pt is clean of aap_resume while checkpoint-0001.pt carries aap_resume.
    best_blob = pickle.loads((output_dir / "best.pt").read_bytes())
    assert "aap_resume" not in best_blob
    cp_blob = pickle.loads((output_dir / by_role["checkpoint"].filename).read_bytes())
    assert cp_blob["aap_resume"]["best_sha256"] == by_role["weights"].sha256

    # metrics.json reports canonical summary + evaluator comparison descriptor.
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["summary"]) == {
        "metrics/mAP50-95",
        "metrics/mAP50",
        "metrics/mAP75",
        "metrics/AR100",
    }
    if eval_backend is None:
        assert "comparison" not in metrics
        assert metrics["comparison_unavailable_reason"] == "evaluator_not_recorded"
    else:
        assert metrics["comparison"]["candidate"]["weights_sha256"] == by_role["weights"].sha256
        assert metrics["comparison"]["evaluator"]["options"]["backend"] == "pycocotools"

    # input-membership.json matches observed dataset.
    membership = json.loads((output_dir / "input-membership.json").read_text(encoding="utf-8"))
    assert membership["counts"] == {"train": 1, "val": 1}


@pytest.mark.parametrize(
    "scenario,expected_code",
    [
        ("cuda_oom", "cuda_oom"),
        ("missing_final_val_member", "input_mismatch"),
        ("resume_warning", "native_exception"),
        ("wrong_runtime_version", "runtime_mismatch"),
        ("tampered_platform_param", "config_mismatch"),
    ],
)
def test_train_cli_emits_structured_failure_receipt_and_exits_nonzero(
    tmp_path: Path, monkeypatch, scenario: str, expected_code: str
) -> None:
    ds_dir, manifest, classes = _build_dataset(tmp_path)
    _install_mock_runtime(
        monkeypatch,
        tmp_path,
        ds_dir,
        libreyolo_version="1.5.0" if scenario == "wrong_runtime_version" else "1.6.0",
        fail_mode=scenario
        if scenario in {"cuda_oom", "missing_final_val_member", "resume_warning"}
        else None,
    )
    params = (
        {"epochs": 2, "batch": 1, "imgsz": 128, "amp": True}
        if scenario == "tampered_platform_param"
        else {"epochs": 2, "batch": 1, "imgsz": 128}
    )
    spec_path = _make_train_spec(
        tmp_path, ds_dir, manifest, classes, parameters=params
    )
    output_dir = tmp_path / f"out-{scenario}"

    rc = main(["train", "--spec", str(spec_path), "--output", str(output_dir)])
    assert rc != 0
    receipt = TrainReceiptV1.model_validate(
        json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    )
    assert receipt.status == "failed"
    assert receipt.failure_code == expected_code
    assert receipt.failure_message
    if scenario == "resume_warning":
        assert "restoration failed" in receipt.failure_message
