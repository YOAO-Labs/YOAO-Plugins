"""Safe weights_only=True checkpoint verification, strict resume logging, and callbacks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import logging
import math
import os
from pathlib import Path
from typing import Any

from yoao_plugin_sdk.trainer import canonical_json_digest

from . import FRAMEWORK_ID, FRAMEWORK_VERSION, NATIVE_CHECKPOINT_FAMILY
from .catalog import architecture_size
from .errors import InputMismatchError
from .evaluation import metrics_of, strip_metric_aliases
from .files import atomic_bytes, atomic_json, digest_file, immutable_copy
from .validation import normalized, verify_native_observation

STRICT_RESUME_WARNING_SUBSTRINGS = (
    "Could not load optimizer state",
    "Could not load EMA weights",
    "Could not restore RNG state",
    "Could not restore GradScaler state",
)

CHECKPOINT_RESUME_POLICY_DIGEST = canonical_json_digest(
    {
        "schema": 1,
        "framework": FRAMEWORK_ID,
        "framework_version": FRAMEWORK_VERSION,
        "weights_only": True,
        "strict_resume_logging": True,
        "embedded_best_weights": True,
        "clean_best_pt": True,
        "warning_substrings": list(STRICT_RESUME_WARNING_SUBSTRINGS),
    }
)


def load_checkpoint_safe(source: Path | str | io.BytesIO) -> Any:
    """Load a PyTorch checkpoint strictly with weights_only=True."""
    import torch

    return torch.load(source, map_location="cpu", weights_only=True)


def ordered_names(blob: dict[str, Any]) -> list[str]:
    names = blob["names"]
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Invalid native class indexes")
        return [names[i] for i in range(len(names))]
    return list(names)


def validate_checkpoint(
    blob: Any,
    *,
    classes: list[str],
    architecture_id: str,
    config_digest: str,
    total_epochs: int,
    epoch: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    try:
        validate_checkpoint_metadata(blob, strict=True)
    except (TypeError, ValueError) as exc:
        raise InputMismatchError(str(exc)) from exc

    expected_size = architecture_size(architecture_id)
    try:
        blob_names = ordered_names(blob)
    except (KeyError, TypeError, ValueError) as exc:
        raise InputMismatchError("Native checkpoint class names are invalid") from exc

    if (
        blob.get("task") != "detect"
        or blob.get("model_family") != NATIVE_CHECKPOINT_FAMILY
        or blob.get("libreyolo_version") != FRAMEWORK_VERSION
        or blob.get("size") != expected_size
        or blob_names != classes
    ):
        raise InputMismatchError(
            "Native checkpoint task, version, architecture size, or class identity mismatch"
        )

    if epoch is not None and (
        blob.get("epoch") != epoch - 1 or not blob.get("optimizer")
    ):
        raise InputMismatchError("Full native epoch and optimizer state required")

    if resume:
        state = blob.get("aap_resume")
        completed = blob.get("epoch", -1) + 1
        best = state.get("best_weights") if isinstance(state, dict) else None
        if (
            not isinstance(state, dict)
            or state.get("schema") != 1
            or state.get("config_digest") != config_digest
            or not isinstance(best, bytes)
            or not best
            or hashlib.sha256(best).hexdigest() != state.get("best_sha256")
            or not 0 < completed < total_epochs
            or blob.get("fitness_source") == "callback"
            or not blob.get("optimizer")
            or not blob.get("rng_state")
        ):
            raise InputMismatchError(
                "Only a complete, verified platform resume checkpoint is supported"
            )
    return blob


def load_and_validate_resume_checkpoint(
    resume_path: Path | str,
    *,
    classes: list[str],
    architecture_id: str,
    config_digest: str,
    total_epochs: int,
) -> dict[str, Any]:
    resume_blob = validate_checkpoint(
        load_checkpoint_safe(resume_path),
        classes=classes,
        architecture_id=architecture_id,
        config_digest=config_digest,
        total_epochs=total_epochs,
        resume=True,
    )
    embedded_best = load_checkpoint_safe(
        io.BytesIO(resume_blob["aap_resume"]["best_weights"])
    )
    validate_checkpoint(
        embedded_best,
        classes=classes,
        architecture_id=architecture_id,
        config_digest=config_digest,
        total_epochs=total_epochs,
    )
    if "aap_resume" in embedded_best:
        raise InputMismatchError("Embedded best weights must not contain aap_resume")
    return resume_blob


class StrictResumeLogging(logging.Handler):
    """Fail loudly if LibreYOLO warns about partial resume state restoration."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(text in message for text in STRICT_RESUME_WARNING_SUBSTRINGS):
            raise RuntimeError("Native checkpoint state restoration failed: " + message)


@dataclass(frozen=True)
class AttemptContext:
    classes: list[str]
    architecture_id: str
    config_digest: str
    total_epochs: int
    effective_values: dict[str, Any]
    data_yaml: str


class CheckpointCallbacks:
    """LibreYOLO training callbacks preserving clean best.pt and resumable checkpoints."""

    def __init__(
        self,
        context: AttemptContext,
        output: Path,
        *,
        emitter: Any,
        resume: dict[str, Any] | None = None,
        membership: Any = None,
    ) -> None:
        self.context = context
        self.output = output
        self.emitter = emitter
        self.resume = resume
        self.membership = membership
        self.checkpoint: dict[str, Any] | None = None
        self.native_config_written = False
        self.trainer: Any = None

    def commit_resume(self, path: Path | str) -> None:
        if self.resume is None:
            raise ValueError("commit_resume requires a validated resume checkpoint")
        epoch = self.resume["epoch"] + 1
        self.checkpoint = immutable_copy(
            Path(path), self.output / f"checkpoint-{epoch:04d}.pt"
        )
        self.checkpoint.update(role="checkpoint", epoch=epoch)
        atomic_json(self.output / "checkpoint.json", self.checkpoint)

    def on_train_start(self, event: Any) -> None:
        if self.membership is not None:
            if self.trainer is None:
                raise InputMismatchError("Native training input observation is missing")
            self.membership.observe(
                "train", self.trainer.train_loader.dataset.dataset.img_files
            )
        expected = (self.resume["epoch"] + 2) if self.resume else 1
        if (
            Path(event.save_dir).resolve() != (self.output / "framework").resolve()
            or event.start_epoch != expected
            or event.task != "detect"
        ):
            raise ValueError("Native resume directory or epoch mismatch")
        if self.resume:
            atomic_bytes(
                Path(event.save_dir) / "weights/best.pt",
                self.resume["aap_resume"]["best_weights"],
            )
        observed = normalized(dict(event.config))
        runtime_options = {
            "data": self.context.data_yaml,
            "project": str(self.output),
            "name": "framework",
            "exist_ok": False,
            "resume": bool(self.resume),
            "num_classes": len(self.context.classes),
            "size": architecture_size(self.context.architecture_id),
            "data_dir": None,
        }
        verification = verify_native_observation(
            self.context.effective_values,
            observed,
            runtime_options,
            num_classes=len(self.context.classes),
            size=architecture_size(self.context.architecture_id),
        )
        atomic_json(
            self.output / "native-config.json",
            {
                "config_digest": self.context.config_digest,
                "observed": observed,
                "runtime_options": runtime_options,
                "verification": verification,
                "evidence": {},
            },
        )
        self.native_config_written = True
        self.emitter.phase("training")

    def on_train_epoch_end(self, event: Any) -> None:
        import torch

        if self.membership is not None:
            if getattr(self.trainer, "_epoch_validator", None) is None:
                raise InputMismatchError("Native validation input observation is missing")
            self.membership.observe(
                "val", self.trainer._epoch_validator.dataloader.dataset.img_files
            )
        directory = Path(event.save_dir) / "weights"
        full = validate_checkpoint(
            load_checkpoint_safe(directory / "last.pt"),
            classes=self.context.classes,
            architecture_id=self.context.architecture_id,
            config_digest=self.context.config_digest,
            total_epochs=self.context.total_epochs,
            epoch=event.epoch,
        )
        best = directory / "best.pt"
        native_best = validate_checkpoint(
            load_checkpoint_safe(best),
            classes=self.context.classes,
            architecture_id=self.context.architecture_id,
            config_digest=self.context.config_digest,
            total_epochs=self.context.total_epochs,
        )
        if "aap_resume" in native_best:
            raise InputMismatchError("Native best.pt must remain clean of aap_resume")
        if native_best.get("epoch", -1) + 1 != full.get("best_epoch"):
            raise ValueError(
                "Historical best weights do not match checkpoint selection"
            )
        checkpoint_payload = dict(full)
        checkpoint_payload["aap_resume"] = {
            "schema": 1,
            "config_digest": self.context.config_digest,
            "best_weights": best.read_bytes(),
            "best_sha256": digest_file(best),
        }
        staging = self.output / "checkpoint-staging.pt"
        try:
            with staging.open("wb") as stream:
                torch.save(checkpoint_payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            previous = self.checkpoint
            self.checkpoint = immutable_copy(
                staging, self.output / f"checkpoint-{event.epoch:04d}.pt"
            )
        finally:
            staging.unlink(missing_ok=True)
        self.checkpoint.update(role="checkpoint", epoch=event.epoch)
        atomic_json(self.output / "checkpoint.json", self.checkpoint)
        if previous and previous["name"] != self.checkpoint["name"]:
            (self.output / previous["name"]).unlink(missing_ok=True)
        raw = metrics_of(event.val_metrics)
        metrics = strip_metric_aliases(raw)
        train_loss = float(event.train_loss)
        if math.isfinite(train_loss):
            metrics["train/loss"] = 0.0 if train_loss == 0.0 else train_loss
        self.emitter.epoch(
            epoch=event.epoch,
            total_epochs=self.context.total_epochs,
            metrics=metrics,
        )
        self.emitter.checkpoint(
            epoch=event.epoch,
            total_epochs=self.context.total_epochs,
            filename=self.checkpoint["name"],
            sha256=self.checkpoint["sha256"],
        )

    def on_train_end(self, event: Any) -> None:
        pass

    def on_train_exception(self, event: Any) -> None:
        pass
