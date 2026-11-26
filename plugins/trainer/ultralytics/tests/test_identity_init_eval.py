"""Tests for native_identity, initialization, and evaluation contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np
import pytest

from yoao_plugin_sdk.trainer import canonical_json_digest, digest_file

from yoao_trainer_ultralytics import InputMismatchError
from yoao_trainer_ultralytics.catalog import model_spec
from yoao_trainer_ultralytics.dataset import INSTANCE_RLE_CONTRACT
from yoao_trainer_ultralytics.evaluation import (
    ultralytics_descriptor,
    ultralytics_report,
)
from yoao_trainer_ultralytics.initialization import (
    observation,
    prepare,
    tensor_identity,
    validate_input,
    validate_pin,
)
from yoao_trainer_ultralytics.native_identity import (
    matches_architecture_signature,
    validate_architecture_yaml,
)
from yoao_trainer_ultralytics.policies import initialization_policy_for


class _FakeModule:
    def __init__(self, state: dict[str, np.ndarray], *, sub_head: Any = None, names: list[str] | None = None) -> None:
        self._state = state
        self.model = [sub_head] if sub_head is not None else []
        self.names = names if names is not None else ["cat", "dog"]

    def state_dict(self) -> dict[str, np.ndarray]:
        return dict(self._state)

    def named_modules(self, remove_duplicate: bool = False):
        yield ("", self)


def _make_pin(
    weights_path: Path,
    *,
    task: str = "detect",
    base_model: str = "yolo11n",
    classes: list[str] | None = None,
) -> dict[str, Any]:
    classes = classes or ["cat", "dog"]
    spec = model_spec(base_model)
    policy = initialization_policy_for(spec.family, spec.task, "8.4.183")
    proj_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    weights_sha = digest_file(weights_path)
    evaluation: dict[str, Any] = {
        "report_sha256": "1" * 64,
        "report_digest": "2" * 64,
        "validation_digest": "3" * 64,
    }
    if task == "segment":
        evaluation.update(
            {
                "normalized_model_digest": "4" * 64,
                "descriptor_digest": "5" * 64,
                "weights_sha256": weights_sha,
            }
        )
    pin: dict[str, Any] = {
        "kind": "trained_model",
        "policy_digest": canonical_json_digest(policy),
        "project_id": proj_id,
        "model_version_id": str(uuid.uuid4()),
        "model_name": "detector-v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "dataset_id": str(uuid.uuid4()),
        "framework": "ultralytics",
        "framework_version": "8.4.183",
        "task": task,
        "model_family": spec.family,
        "base_model": base_model,
        "artifact": {
            "key": f"projects/{proj_id}/runs/{run_id}/attempts/{attempt_id}/best.pt",
            "role": "weights",
            "sha256": weights_sha,
            "size_bytes": weights_path.stat().st_size,
        },
        "classes": classes,
        "classes_digest": canonical_json_digest(classes),
        "bindings": list(classes),
        "binding_digest": canonical_json_digest({"classes": classes, "bindings": list(classes)}),
        "config_digest": "6" * 64,
        "dataset_digest": "7" * 64,
        "validation_digest": "3" * 64,
        "runtime": {
            "python": "3.12.11",
            "torch": "2.14.1+cu132",
            "torchvision": "0.29.1+cu132",
            "cuda": "13.2",
            "image_digest": "sha256:" + "8" * 64,
        },
        "evaluation": evaluation,
    }
    if task == "segment":
        pin["data_contract"] = dict(INSTANCE_RLE_CONTRACT)
    pin["source_digest"] = canonical_json_digest(pin)
    return pin


def test_validate_architecture_yaml_canonical_and_legacy_yolov8() -> None:
    canonical_v11 = {
        "ch": 3,
        "scales": {"n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024]},
        "backbone": [[-1, 1, "Conv", [64, 3, 2]]],
        "head": [[-1, 1, "Detect", ["nc"]]],
    }
    actual_v11 = {**canonical_v11, "scale": "n", "yaml_file": "/cfg/yolo11n.yaml"}
    validate_architecture_yaml(actual_v11, model_spec("yolo11n"), canonical_v11)

    with pytest.raises(ValueError, match="channel"):
        validate_architecture_yaml({**actual_v11, "ch": 1}, model_spec("yolo11n"), canonical_v11)
    with pytest.raises(ValueError, match="scale"):
        validate_architecture_yaml({**actual_v11, "scale": "s"}, model_spec("yolo11n"), canonical_v11)
    with pytest.raises(ValueError, match="legacy YOLOv8"):
        no_scale = {k: v for k, v in actual_v11.items() if k != "scale"}
        validate_architecture_yaml(no_scale, model_spec("yolo11n"), canonical_v11)

    canonical_v8 = {
        "ch": 3,
        "scales": {"n": [0.33, 0.25, 1024]},
        "backbone": [[-1, 1, "Conv", [2048, 3, 2]]],
        "head": [[-1, 1, "Detect", ["nc"]]],
    }
    legacy_v8 = {
        "ch": 3,
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "yaml_file": "yolov8n.yaml",
        "backbone": [[-1, 1, "Conv", [1024, 3, 2]]],
        "head": [[-1, 1, "Detect", ["nc"]]],
    }
    validate_architecture_yaml(legacy_v8, model_spec("yolov8n"), canonical_v8)


def test_matches_architecture_signature_matches_exact_and_transferred() -> None:
    def factory(base: str | tuple[str, int], nc: int) -> _FakeModule:
        if isinstance(base, tuple):
            return _FakeModule({"w": np.zeros((nc, 8), dtype=np.float32)})
        return _FakeModule({"w": np.zeros((nc, 4), dtype=np.float32)})

    exact = _FakeModule({"w": np.zeros((3, 4), dtype=np.float32)})
    transferred = _FakeModule({"w": np.zeros((3, 8), dtype=np.float32)})
    bad = _FakeModule({"w": np.zeros((3, 99), dtype=np.float32)})

    assert matches_architecture_signature(exact, "yolo11n", 3, factory, "8.4.183") is True
    assert matches_architecture_signature(transferred, "yolo11n", 3, factory, "8.4.183") is True
    assert matches_architecture_signature(bad, "yolo11n", 3, factory, "8.4.183") is False


def test_initialization_tensor_identity_pin_prepare_and_fresh_state(tmp_path: Path) -> None:
    w_fp32 = np.array([[1.0, -2.5], [0.25, 4.0]], dtype=np.float32)
    w_fp16 = w_fp32.astype(np.float16)
    w_perturbed = w_fp32.copy()
    w_perturbed[0, 0] = np.nextafter(w_perturbed[0, 0], np.float32(2.0))

    d32, c32 = tensor_identity(_FakeModule({"conv.weight": w_fp32}))
    d16, c16 = tensor_identity(_FakeModule({"conv.weight": w_fp16}))
    d_pert, _ = tensor_identity(_FakeModule({"conv.weight": w_perturbed}))
    assert (d32, c32) == (d16, c16)
    assert d32 != d_pert

    weights = tmp_path / "init.pt"
    weights.write_bytes(b"fake-weights-payload")
    seg_pin = _make_pin(weights, task="segment", base_model="yolo26n-seg")
    validate_pin(seg_pin)
    validate_input(seg_pin, init_weights_path=weights, resume_checkpoint_path=None)

    with pytest.raises(ValueError, match="source digest"):
        tampered = {**seg_pin, "model_name": "tampered"}
        validate_pin(tampered)

    head = _FakeModule({"proto.weight": w_fp32})
    loaded = _FakeModule({"backbone.weight": w_fp32, "proto.weight": w_fp32}, sub_head=head)
    wrapped_model = SimpleNamespace(model=loaded, task="segment")
    baseline = prepare(
        seg_pin,
        wrapped_model,
        classes=["cat", "dog"],
        architecture_validator=None,
    )
    assert baseline is not None
    assert "mask_head_tensor_digest" in baseline

    best_path = tmp_path / "best.pt"
    trainer = SimpleNamespace(
        model=loaded,
        start_epoch=0,
        optimizer=SimpleNamespace(state={}),
        scheduler=SimpleNamespace(last_epoch=-1),
        scaler=SimpleNamespace(state_dict=lambda: {}),
        ema=SimpleNamespace(updates=0),
        best_fitness=None,
        best=best_path,
    )
    ev = observation(seg_pin, baseline, trainer, resumed=False)
    assert ev is not None
    assert ev["loaded_tensor_digest"] == ev["source_tensor_digest"]
    assert ev["loaded_mask_head_tensor_digest"] == ev["mask_head_tensor_digest"]

    # Non-fresh state must fail
    trainer.optimizer.state = {0: {"momentum_buffer": 1}}
    with pytest.raises(InputMismatchError, match="fresh-state reset"):
        observation(seg_pin, baseline, trainer, resumed=False)


def test_evaluation_report_and_yolo26_end2end_descriptor() -> None:
    detect_result = SimpleNamespace(
        box=SimpleNamespace(ap_class_index=[0, 1]),
        class_result=lambda idx: (0.8, 0.7, 0.75, 0.5) if idx == 0 else (0.9, 0.85, 0.88, 0.6),
        results_dict={"metrics/mAP50(B)": 0.815, "metrics/mAP50-95(B)": 0.55},
    )
    rep = ultralytics_report(
        detect_result,
        task="detect",
        classes=["cat", "dog"],
        validation_digest="a" * 64,
        imgsz=640,
    )
    assert len(rep["per_class"]) == 2
    assert rep["per_class"][0]["name"] == "cat"
    assert set(rep["per_class"][0]) == {"index", "name", "precision", "recall", "mAP50", "mAP50-95"}

    seg_result = SimpleNamespace(
        box=SimpleNamespace(ap_class_index=[0]),
        class_result=lambda idx: (0.8, 0.7, 0.75, 0.5, 0.78, 0.68, 0.72, 0.48),
        results_dict={"metrics/mAP50(B)": 0.75, "metrics/mAP50(M)": 0.72},
    )
    seg_rep = ultralytics_report(
        seg_result,
        task="segment",
        classes=["person"],
        validation_digest="b" * 64,
        imgsz=640,
    )
    assert "mAP50(M)" in seg_rep["per_class"][0]
    assert "mAP50(B)" in seg_rep["per_class"][0]

    def _make_validator(*, end2end: bool, nms: Any) -> Any:
        return SimpleNamespace(
            args=SimpleNamespace(
                split="val",
                imgsz=640,
                conf=0.001,
                iou=0.7,
                max_det=300,
                augment=False,
                single_cls=False,
                classes=None,
                agnostic_nms=False,
                dnn=False,
                compile=False,
                nms=nms,
                quantize=None,
                channels_last=False,
                save_json=False,
                fraction=1.0,
                overlap_mask=False,
                mask_ratio=1,
            ),
            device="cpu",
            end2end=end2end,
            dataloader=SimpleNamespace(
                batch_size=1,
                num_workers=0,
                dataset=SimpleNamespace(rect=False, imgsz=640),
            ),
        )

    desc = ultralytics_descriptor(
        config_digest="c" * 64,
        architecture_id="yolo26n-seg",
        task="segment",
        classes=["person"],
        validation_digest="b" * 64,
        weights_sha256="d" * 64,
        validator=_make_validator(end2end=True, nms=False),
        metrics=seg_rep["summary"],
        is_rle=True,
    )
    assert desc["revision"] == 3
    assert desc["evaluator"]["options"]["end2end"] is True
    assert len(desc["evaluator"]["source_sha256"]) == 64

    with pytest.raises(ValueError, match="inference branch differs"):
        ultralytics_descriptor(
            config_digest="c" * 64,
            architecture_id="yolo26n-seg",
            task="segment",
            classes=["person"],
            validation_digest="b" * 64,
            weights_sha256="d" * 64,
            validator=_make_validator(end2end=False, nms=False),
            metrics=seg_rep["summary"],
            is_rle=True,
        )
