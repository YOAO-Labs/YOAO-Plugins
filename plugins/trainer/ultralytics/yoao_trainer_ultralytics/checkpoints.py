"""Ultralytics checkpoints carrying verified historical best weights across resumes."""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path
from typing import Any, Callable

from . import FRAMEWORK_VERSION, InputMismatchError
from .files import atomic_bytes, atomic_json, immutable_copy
from .native_identity import validate_architecture
from .policies import SEGMENTATION


def _default_torch_io() -> tuple[Callable[..., Any], Callable[..., Any]]:
    import torch

    def load_fn(source: Any) -> Any:
        return torch.load(source, map_location="cpu", weights_only=False)

    def save_fn(obj: Any, stream: Any) -> None:
        torch.save(obj, stream)

    return load_fn, save_fn


class UltralyticsCheckpoints:
    """Carry clean historical best.pt inside aap_resume alongside full optimizer checkpoint."""

    def __init__(
        self,
        *,
        output_dir: Path,
        architecture_id: str,
        task: str,
        classes: list[str],
        epochs: int,
        config_digest: str,
        resume_checkpoint_path: str | Path | None = None,
        framework_version: str = FRAMEWORK_VERSION,
        torch_load: Callable[[Any], Any] | None = None,
        torch_save: Callable[[Any, Any], None] | None = None,
        architecture_validator: Callable[..., Any] | None = validate_architecture,
    ) -> None:
        self.output = output_dir
        self.architecture_id = architecture_id
        self.task = task
        self.classes = list(classes)
        self.epochs = int(epochs)
        self.config_digest = config_digest
        self.framework_version = framework_version
        self._torch_load = torch_load
        self._torch_save = torch_save
        self._validate_architecture = architecture_validator
        self.checkpoint: dict[str, Any] | None = None
        self.best_bytes: bytes | None = None

        if resume_checkpoint_path:
            resume_path = Path(resume_checkpoint_path)
            full = self._load(resume_path)
            self.validate_identity(full)
            epoch = full.get("epoch", -1) + 1
            if not full.get("optimizer") or not 0 < epoch < self.epochs:
                raise InputMismatchError(
                    "A complete unfinished native optimizer checkpoint is required"
                )
            state = full.get("aap_resume")
            if state is None:
                raise InputMismatchError(
                    "Resume checkpoint lacks verified aap_resume historical best weights"
                )
            if not isinstance(state, dict):
                raise InputMismatchError("Historical best receipt is invalid")
            best = state.get("best_weights")
            if (
                state.get("schema") != 1
                or state.get("config_digest") != self.config_digest
                or not isinstance(best, bytes)
                or not best
                or hashlib.sha256(best).hexdigest() != state.get("best_sha256")
            ):
                raise InputMismatchError(
                    "Historical best weights do not match the frozen resume checkpoint"
                )
            self.best_bytes = best
            selected = self._load(io.BytesIO(best))
            self.validate_identity(selected)
            if selected["epoch"] > full["epoch"] or selected.get(
                "train_metrics", {}
            ).get("fitness") != full.get("best_fitness"):
                raise InputMismatchError(
                    "Historical best fitness or epoch does not match the resume checkpoint"
                )
            self.checkpoint = immutable_copy(
                resume_path, self.output / f"checkpoint-{epoch:04d}.pt"
            )
            self.checkpoint.update(role="checkpoint", epoch=epoch)
            atomic_json(self.output / "checkpoint.json", self.checkpoint)

    def _load(self, source: Any) -> Any:
        if self._torch_load is not None:
            return self._torch_load(source)
        load_fn, _ = _default_torch_io()
        return load_fn(source)

    def _save(self, obj: Any, stream: Any) -> None:
        if self._torch_save is not None:
            self._torch_save(obj, stream)
            return
        _, save_fn = _default_torch_io()
        save_fn(obj, stream)

    def validate_identity(self, blob: dict[str, Any]) -> None:
        model = blob.get("ema")
        names = getattr(model, "names", None)
        if isinstance(names, dict) and set(names) == set(range(len(names))):
            names = [names[index] for index in range(len(names))]
        expected_task = "segment" if self.task == SEGMENTATION else "detect"
        if (
            blob.get("version") != self.framework_version
            or blob.get("train_args", {}).get("task") != expected_task
            or names != self.classes
            or type(blob.get("epoch")) is not int
            or blob["epoch"] < 0
            or type(blob.get("best_fitness")) not in {int, float}
            or not math.isfinite(blob["best_fitness"])
        ):
            raise InputMismatchError(
                "Native checkpoint version, task, class or epoch identity mismatch"
            )
        if self._validate_architecture is not None:
            try:
                self._validate_architecture(
                    model, self.architecture_id, classes=len(names)
                )
            except (ValueError, AttributeError, KeyError, TypeError) as exc:
                raise InputMismatchError("Unqualified checkpoint architecture") from exc

    def started(self, trainer: Any) -> None:
        if self.best_bytes is not None:
            atomic_bytes(Path(trainer.best), self.best_bytes)

    def saved(self, trainer: Any) -> None:
        previous = self.checkpoint
        epoch = trainer.epoch + 1
        full = self._load(trainer.last)
        self.validate_identity(full)
        if full.get("epoch") != trainer.epoch or not full.get("optimizer"):
            raise RuntimeError("Full training checkpoint was not saved")
        best = Path(trainer.best)
        if best.is_file():
            best_bytes = best.read_bytes()
            selected = self._load(io.BytesIO(best_bytes))
            self.validate_identity(selected)
            if selected["epoch"] > full["epoch"] or selected.get(
                "train_metrics", {}
            ).get("fitness") != full.get("best_fitness"):
                raise ValueError(
                    "Native best weights do not match the checkpoint selection"
                )
            full["aap_resume"] = {
                "schema": 1,
                "config_digest": self.config_digest,
                "best_weights": best_bytes,
                "best_sha256": hashlib.sha256(best_bytes).hexdigest(),
            }
        staging = self.output / "checkpoint-staging.pt"
        try:
            with staging.open("wb") as stream:
                self._save(full, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.checkpoint = immutable_copy(
                staging, self.output / f"checkpoint-{epoch:04d}.pt"
            )
        finally:
            staging.unlink(missing_ok=True)
        self.checkpoint.update(role="checkpoint", epoch=epoch)
        atomic_json(self.output / "checkpoint.json", self.checkpoint)
        if previous and previous["name"] != self.checkpoint["name"]:
            (self.output / previous["name"]).unlink(missing_ok=True)
