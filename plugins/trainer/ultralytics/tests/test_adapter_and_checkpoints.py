"""Tests for Ultralytics adapter boundaries (fonts, telemetry, OOM retries) and checkpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from yoao_trainer_ultralytics import (
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
)
from yoao_trainer_ultralytics.adapter import (
    UltralyticsAttemptTrainer,
    configure_offline_fonts,
    disable_telemetry,
    require_qualified_runtime,
)
from yoao_trainer_ultralytics.checkpoints import UltralyticsCheckpoints
from tests.conftest import install_fake_ultralytics_modules, make_fake_fonts


def test_runtime_version_telemetry_and_offline_fonts_patch_all_three_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_ultralytics_modules(monkeypatch, version="8.4.182")
    with pytest.raises(NativeRuntimeMismatch, match="8.4.183"):
        require_qualified_runtime()

    mods = install_fake_ultralytics_modules(monkeypatch, version="8.4.183")
    assert require_qualified_runtime() == "8.4.183"

    disable_telemetry()
    assert mods["settings_log"] == [
        {"sync": False, "wandb": False, "mlflow": False, "tensorboard": False}
    ]

    ascii_font, unicode_font = make_fake_fonts(tmp_path, monkeypatch)
    configured = configure_offline_fonts()
    assert configured == {"Arial.ttf": ascii_font, "Arial.Unicode.ttf": unicode_font}
    for site in (mods["checks"], mods["data_utils"], mods["plotting"]):
        assert site.check_font("Arial.ttf") == ascii_font
        assert site.check_font("Arial.Unicode.ttf") == unicode_font
        with pytest.raises(NativeConfigurationMismatch, match="Unsupported Ultralytics training font"):
            site.check_font("ComicSans.ttf")

    monkeypatch.setenv("YOAO_ULTRALYTICS_ASCII_FONT", "relative/font.ttf")
    with pytest.raises(NativeConfigurationMismatch, match="installed local font file"):
        configure_offline_fonts()


def test_attempt_trainer_enforces_fixed_batch_resume_rebind_and_oom_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ultralytics_modules(monkeypatch, version="8.4.183")
    events_seen: list[str] = []

    class FakeBase:
        def __init__(self, *, mutate_batch: bool = False) -> None:
            self.args = SimpleNamespace(batch=4, project="/old", name="old", data="/old.yaml", exist_ok=True)
            self.resume = True
            self.epoch = 2
            self.start_epoch = 2
            self._oom_retries = 0
            self._mutate_batch = mutate_batch

        def check_resume(self, overrides: dict[str, Any]) -> None:
            if self._mutate_batch:
                self.args.batch = 2

        def run_callbacks(self, event: str) -> None:
            events_seen.append(event)

    class Combined(UltralyticsAttemptTrainer, FakeBase):
        pass

    trainer = Combined()
    overrides = {
        "batch": 4,
        "project": "/new/out",
        "name": "framework",
        "data": "/new/data.yaml",
        "exist_ok": False,
    }
    trainer.check_resume(overrides)
    assert trainer.args.project == "/new/out"
    assert trainer.args.name == "framework"
    assert trainer.args.data == "/new/data.yaml"
    assert trainer.args.exist_ok is False
    assert trainer.args.save_dir == "/new/out/framework"

    trainer.run_callbacks("on_train_epoch_start")
    assert trainer._oom_retries == 3
    assert events_seen == ["on_train_epoch_start"]

    bad_batch = Combined(mutate_batch=True)
    with pytest.raises(NativeConfigurationMismatch, match="changed the frozen batch"):
        bad_batch.check_resume(overrides)

    with pytest.raises(NativeConfigurationMismatch, match="fixed integer batch"):
        trainer.check_resume({**overrides, "batch": -1})


def test_checkpoints_preserve_aap_resume_best_and_unlink_previous_only_after_commit(
    tmp_path: Path,
) -> None:
    def fake_load(source: Any) -> Any:
        if isinstance(source, (str, Path)):
            return pickle.loads(Path(source).read_bytes())
        return pickle.loads(source.read())

    def fake_save(obj: Any, stream: Any) -> None:
        stream.write(pickle.dumps(obj))

    def make_blob(epoch: int, fitness: float, *, best_fitness: float | None = None, optimizer: Any = True) -> dict[str, Any]:
        return {
            "version": "8.4.183",
            "train_args": {"task": "detect"},
            "ema": SimpleNamespace(names={0: "cat", 1: "dog"}),
            "epoch": epoch,
            "best_fitness": fitness if best_fitness is None else best_fitness,
            "train_metrics": {"fitness": fitness},
            "optimizer": {"step": epoch + 1} if optimizer else None,
        }

    out1 = tmp_path / "attempt1"
    out1.mkdir()
    ckpt = UltralyticsCheckpoints(
        output_dir=out1,
        architecture_id="yolo11n",
        task="detect",
        classes=["cat", "dog"],
        epochs=5,
        config_digest="a" * 64,
        torch_load=fake_load,
        torch_save=fake_save,
        architecture_validator=None,
    )

    last_pt = out1 / "last.pt"
    best_pt = out1 / "best.pt"
    best_blob = make_blob(0, 0.75)
    best_pt.write_bytes(pickle.dumps(best_blob))
    last_pt.write_bytes(pickle.dumps(make_blob(0, 0.75)))
    ckpt.saved(SimpleNamespace(epoch=0, last=last_pt, best=best_pt))

    assert (out1 / "checkpoint-0001.pt").is_file()
    pointer1 = json.loads((out1 / "checkpoint.json").read_text(encoding="utf-8"))
    assert pointer1["name"] == "checkpoint-0001.pt"
    assert pointer1["epoch"] == 1

    # Epoch 2 has worse fitness (0.60), so historical best (0.75) from epoch 1 stays in aap_resume
    last_pt.write_bytes(pickle.dumps(make_blob(1, 0.60, best_fitness=0.75)))
    ckpt.saved(SimpleNamespace(epoch=1, last=last_pt, best=best_pt))

    assert not (out1 / "checkpoint-0001.pt").exists()
    ckpt2_path = out1 / "checkpoint-0002.pt"
    assert ckpt2_path.is_file()
    pointer2 = json.loads((out1 / "checkpoint.json").read_text(encoding="utf-8"))
    assert pointer2["name"] == "checkpoint-0002.pt"
    assert pointer2["epoch"] == 2

    # Resume in attempt2 restores exact historical best.pt bytes
    out2 = tmp_path / "attempt2"
    out2.mkdir()
    resumed = UltralyticsCheckpoints(
        output_dir=out2,
        architecture_id="yolo11n",
        task="detect",
        classes=["cat", "dog"],
        epochs=5,
        config_digest="a" * 64,
        resume_checkpoint_path=ckpt2_path,
        torch_load=fake_load,
        torch_save=fake_save,
        architecture_validator=None,
    )
    restored_best = out2 / "framework" / "weights" / "best.pt"
    resumed.started(SimpleNamespace(best=restored_best))
    assert restored_best.read_bytes() == best_pt.read_bytes()

    # Tampered config_digest in resume checkpoint is rejected
    with pytest.raises(InputMismatchError, match="frozen resume checkpoint"):
        UltralyticsCheckpoints(
            output_dir=tmp_path / "attempt_bad_cfg",
            architecture_id="yolo11n",
            task="detect",
            classes=["cat", "dog"],
            epochs=5,
            config_digest="f" * 64,
            resume_checkpoint_path=ckpt2_path,
            torch_load=fake_load,
            torch_save=fake_save,
            architecture_validator=None,
        )

    # Tampered best_sha256 in aap_resume is rejected
    tampered_path = tmp_path / "tampered.pt"
    payload = fake_load(ckpt2_path)
    payload["aap_resume"]["best_sha256"] = "0" * 64
    tampered_path.write_bytes(pickle.dumps(payload))
    with pytest.raises(InputMismatchError, match="Historical best weights"):
        UltralyticsCheckpoints(
            output_dir=tmp_path / "attempt_tampered",
            architecture_id="yolo11n",
            task="detect",
            classes=["cat", "dog"],
            epochs=5,
            config_digest="a" * 64,
            resume_checkpoint_path=tampered_path,
            torch_load=fake_load,
            torch_save=fake_save,
            architecture_validator=None,
        )
