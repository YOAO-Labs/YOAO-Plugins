"""Tests for weights_only=True checkpoint loading, validation, StrictResumeLogging, and callbacks."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from yoao_plugin_sdk.trainer import (
    EventEmitter,
    InputMismatchError,
    NativeConfigurationMismatch,
)

from yoao_trainer_libreyolo.checkpoint import (
    STRICT_RESUME_WARNING_SUBSTRINGS,
    AttemptContext,
    CheckpointCallbacks,
    StrictResumeLogging,
    load_and_validate_resume_checkpoint,
    load_checkpoint_safe,
    validate_checkpoint,
)
from yoao_trainer_libreyolo.validation import resolve_train_parameters

CONFIG_DIGEST = "a" * 64


def _install_fake_torch_and_libreyolo(monkeypatch) -> list[dict[str, Any]]:
    load_calls: list[dict[str, Any]] = []

    fake_torch = ModuleType("torch")

    def fake_load(source, *, map_location=None, weights_only=None):
        load_calls.append({"source": source, "map_location": map_location, "weights_only": weights_only})
        if weights_only is not True:
            raise AssertionError("weights_only=True is mandatory on every torch.load call")
        if isinstance(source, (str, Path)):
            return pickle.loads(Path(source).read_bytes())
        return pickle.loads(source.read())

    def fake_save(obj, target):
        data = pickle.dumps(obj)
        if isinstance(target, (str, Path)):
            Path(target).write_bytes(data)
        else:
            target.write(data)

    fake_torch.load = fake_load  # type: ignore[attr-defined]
    fake_torch.save = fake_save  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_libreyolo = ModuleType("libreyolo")
    fake_utils = ModuleType("libreyolo.utils")
    fake_ser = ModuleType("libreyolo.utils.serialization")

    def validate_checkpoint_metadata(blob, *, strict=True):
        if not isinstance(blob, dict):
            raise TypeError("checkpoint must be a dict")
        required = ("task", "model_family", "libreyolo_version", "size", "names")
        for key in required:
            if key not in blob:
                raise ValueError(f"missing metadata field: {key}")

    fake_ser.validate_checkpoint_metadata = validate_checkpoint_metadata  # type: ignore[attr-defined]
    fake_utils.serialization = fake_ser  # type: ignore[attr-defined]
    fake_libreyolo.utils = fake_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libreyolo", fake_libreyolo)
    monkeypatch.setitem(sys.modules, "libreyolo.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "libreyolo.utils.serialization", fake_ser)
    return load_calls


def _make_blob(
    *,
    epoch: int = 0,
    best_epoch: int = 1,
    size: str = "t",
    names: Any = None,
    version: str = "1.6.0",
    family: str = "yolo9",
    task: str = "detect",
    optimizer: Any = None,
    rng_state: Any = None,
) -> dict[str, Any]:
    return {
        "model": {"weight": [1.0]},
        "task": task,
        "model_family": family,
        "libreyolo_version": version,
        "size": size,
        "names": ["car", "person"] if names is None else names,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "optimizer": {"state": {}, "param_groups": [{}]} if optimizer is None else optimizer,
        "rng_state": {"torch": b"rng"} if rng_state is None else rng_state,
    }


def test_load_checkpoint_safe_enforces_weights_only_true_at_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _install_fake_torch_and_libreyolo(monkeypatch)
    ckpt = tmp_path / "sample.pt"
    ckpt.write_bytes(pickle.dumps({"ok": True}))

    loaded = load_checkpoint_safe(ckpt)
    assert loaded == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "segment"),
        ("model_family", "yolo11"),
        ("libreyolo_version", "1.5.0"),
        ("size", "s"),
        ("names", {0: "person", 1: "car"}),
        ("names", {1: "car", 2: "person"}),
        ("optimizer", None),
        ("epoch", 5),
    ],
)
def test_validate_checkpoint_rejects_incompatible_metadata(
    monkeypatch, field: str, value: Any
) -> None:
    _install_fake_torch_and_libreyolo(monkeypatch)
    blob = _make_blob(epoch=0)
    blob[field] = value
    with pytest.raises(InputMismatchError):
        validate_checkpoint(
            blob,
            classes=["car", "person"],
            architecture_id="libreyolo9t",
            config_digest=CONFIG_DIGEST,
            total_epochs=3,
            epoch=1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_state",
        "bad_schema",
        "wrong_digest",
        "bad_best_sha",
        "empty_best_bytes",
        "completed_at_total",
        "callback_fitness",
        "missing_rng",
        "missing_optimizer",
        "dirty_embedded_best",
    ],
)
def test_validate_resume_checkpoint_rejects_corrupted_aap_resume(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    _install_fake_torch_and_libreyolo(monkeypatch)
    best_blob = _make_blob(epoch=0, best_epoch=1)
    if mutation == "dirty_embedded_best":
        best_blob["aap_resume"] = {"schema": 1}
    best_bytes = pickle.dumps(best_blob)
    blob = _make_blob(epoch=0, best_epoch=1)
    blob["aap_resume"] = {
        "schema": 1,
        "config_digest": CONFIG_DIGEST,
        "best_weights": best_bytes,
        "best_sha256": hashlib.sha256(best_bytes).hexdigest(),
    }
    if mutation == "missing_state":
        blob.pop("aap_resume")
    elif mutation == "bad_schema":
        blob["aap_resume"]["schema"] = 2
    elif mutation == "wrong_digest":
        blob["aap_resume"]["config_digest"] = "0" * 64
    elif mutation == "bad_best_sha":
        blob["aap_resume"]["best_sha256"] = "0" * 64
    elif mutation == "empty_best_bytes":
        blob["aap_resume"]["best_weights"] = b""
        blob["aap_resume"]["best_sha256"] = hashlib.sha256(b"").hexdigest()
    elif mutation == "completed_at_total":
        blob["epoch"] = 2
    elif mutation == "callback_fitness":
        blob["fitness_source"] = "callback"
    elif mutation == "missing_rng":
        blob["rng_state"] = None
    elif mutation == "missing_optimizer":
        blob["optimizer"] = {}

    resume_file = tmp_path / "resume.pt"
    resume_file.write_bytes(pickle.dumps(blob))
    with pytest.raises(InputMismatchError):
        load_and_validate_resume_checkpoint(
            resume_file,
            classes=["car", "person"],
            architecture_id="libreyolo9t",
            config_digest=CONFIG_DIGEST,
            total_epochs=3,
        )


@pytest.mark.parametrize("warning_substring", STRICT_RESUME_WARNING_SUBSTRINGS)
def test_strict_resume_logging_raises_on_all_four_warning_substrings(
    warning_substring: str,
) -> None:
    handler = StrictResumeLogging()
    record = logging.LogRecord(
        "libreyolo.training.trainer",
        logging.WARNING,
        "",
        0,
        f"{warning_substring}: state dict keys mismatch",
        (),
        None,
    )
    with pytest.raises(RuntimeError, match="Native checkpoint state restoration failed"):
        handler.emit(record)


def test_strict_resume_logging_allows_benign_records() -> None:
    handler = StrictResumeLogging()
    record = logging.LogRecord(
        "libreyolo.training.trainer",
        logging.INFO,
        "",
        0,
        "Epoch 1/3 complete",
        (),
        None,
    )
    handler.emit(record)


def test_checkpoint_callbacks_keep_best_pt_clean_and_strip_metric_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _install_fake_torch_and_libreyolo(monkeypatch)
    resolved = resolve_train_parameters({"epochs": 3, "batch": 1, "imgsz": 128})
    effective = resolved["effective_values"]

    first = tmp_path / "attempt1"
    (first / "framework/weights").mkdir(parents=True)
    ctx1 = AttemptContext(
        classes=["car", "person"],
        architecture_id="libreyolo9t",
        config_digest=CONFIG_DIGEST,
        total_epochs=3,
        effective_values=effective,
        data_yaml=str(first / "libreyolo-data.yaml"),
    )
    emitter1 = EventEmitter(first / "events.jsonl")
    cb1 = CheckpointCallbacks(ctx1, first, emitter=emitter1)

    start_config = {
        **effective,
        "data": ctx1.data_yaml,
        "project": str(first),
        "name": "framework",
        "exist_ok": False,
        "resume": False,
        "num_classes": 2,
        "size": "t",
        "data_dir": None,
    }
    cb1.on_train_start(
        SimpleNamespace(
            save_dir=str(first / "framework"),
            start_epoch=1,
            task="detect",
            config=start_config,
        )
    )
    assert cb1.native_config_written is True
    native_cfg = json.loads((first / "native-config.json").read_text(encoding="utf-8"))
    assert native_cfg["verification"]["verified"] is True

    # Tampered observed native config must fail loudly.
    bad_cfg = {**start_config, "amp": True}
    with pytest.raises(NativeConfigurationMismatch, match="amp"):
        cb1.on_train_start(
            SimpleNamespace(
                save_dir=str(first / "framework"),
                start_epoch=1,
                task="detect",
                config=bad_cfg,
            )
        )

    best_path = first / "framework/weights/best.pt"
    last_path = first / "framework/weights/last.pt"
    initial = _make_blob(epoch=0, best_epoch=1)
    best_path.write_bytes(pickle.dumps(initial))
    last_path.write_bytes(pickle.dumps(initial))
    original_best_bytes = best_path.read_bytes()

    cb1.on_train_epoch_end(
        SimpleNamespace(
            epoch=1,
            save_dir=str(first / "framework"),
            val_metrics={
                "metrics/mAP50": 0.45,
                "metrics/mAP50-95": 0.25,
                "metrics/precision": 0.25,
                "metrics/recall": 0.6,
            },
            train_loss=0.8,
        )
    )

    # best.pt stays untouched and free of aap_resume!
    assert best_path.read_bytes() == original_best_bytes
    assert "aap_resume" not in pickle.loads(best_path.read_bytes())

    # Committed checkpoint has aap_resume and pointer file matches.
    pointer = json.loads((first / "checkpoint.json").read_text(encoding="utf-8"))
    assert pointer["name"] == "checkpoint-0001.pt"
    assert pointer["epoch"] == 1
    committed_path = first / pointer["name"]
    committed_blob = load_and_validate_resume_checkpoint(
        committed_path,
        classes=["car", "person"],
        architecture_id="libreyolo9t",
        config_digest=CONFIG_DIGEST,
        total_epochs=3,
    )
    assert committed_blob["aap_resume"]["best_weights"] == original_best_bytes

    # Events strip precision/recall aliases and retain canonical metrics + train/loss.
    events = [
        json.loads(line)
        for line in (first / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    epoch_evt = next(ev for ev in events if ev["kind"] == "epoch")
    assert epoch_evt["epoch"] == 1
    assert "metrics/precision" not in epoch_evt["metrics"]
    assert "metrics/recall" not in epoch_evt["metrics"]
    assert epoch_evt["metrics"]["metrics/mAP50"] == 0.45
    assert epoch_evt["metrics"]["train/loss"] == 0.8

    # Resume into attempt 2 and verify historical best is restored cleanly and old checkpoint pruned.
    second = tmp_path / "attempt2"
    (second / "framework/weights").mkdir(parents=True)
    ctx2 = AttemptContext(
        classes=["car", "person"],
        architecture_id="libreyolo9t",
        config_digest=CONFIG_DIGEST,
        total_epochs=3,
        effective_values=effective,
        data_yaml=str(second / "libreyolo-data.yaml"),
    )
    cb2 = CheckpointCallbacks(
        ctx2, second, emitter=EventEmitter(second / "events.jsonl"), resume=committed_blob
    )
    cb2.commit_resume(committed_path)
    assert (second / "checkpoint-0001.pt").is_file()

    cb2.on_train_start(
        SimpleNamespace(
            save_dir=str(second / "framework"),
            start_epoch=2,
            task="detect",
            config={
                **effective,
                "data": ctx2.data_yaml,
                "project": str(second),
                "name": "framework",
                "exist_ok": False,
                "resume": True,
                "num_classes": 2,
                "size": "t",
                "data_dir": None,
            },
        )
    )
    restored_best = second / "framework/weights/best.pt"
    assert restored_best.read_bytes() == original_best_bytes
    assert "aap_resume" not in pickle.loads(restored_best.read_bytes())

    (second / "framework/weights/last.pt").write_bytes(
        pickle.dumps(_make_blob(epoch=1, best_epoch=1))
    )
    cb2.on_train_epoch_end(
        SimpleNamespace(
            epoch=2,
            save_dir=str(second / "framework"),
            val_metrics={"metrics/mAP50": 0.4, "metrics/precision": 0.2},
            train_loss=0.7,
        )
    )
    pointer2 = json.loads((second / "checkpoint.json").read_text(encoding="utf-8"))
    assert pointer2["name"] == "checkpoint-0002.pt"
    assert pointer2["epoch"] == 2
    assert not (second / "checkpoint-0001.pt").exists()
    assert (second / "checkpoint-0002.pt").is_file()
    assert all(call["weights_only"] is True for call in calls)
