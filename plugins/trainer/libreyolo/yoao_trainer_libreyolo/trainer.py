"""LibreYOLOAttemptTrainer and yoao-trainer/v1 train execution orchestrator."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    TrainArtifactV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    atomic_write_json,
    write_event_jsonl,
)

from . import (
    FRAMEWORK_ID,
    FRAMEWORK_VERSION,
    SUPERVISOR_OWNED_ARTIFACTS,
    SUPPORTED_DATASET_CONTRACTS,
    SUPPORTED_TASKS,
)
from .catalog import architecture_by_id, serving_compatibility
from .checkpoint import (
    AttemptContext,
    CheckpointCallbacks,
    StrictResumeLogging,
    load_and_validate_resume_checkpoint,
    load_checkpoint_safe,
    ordered_names,
)
from .dataset import validate_dataset, write_native_dataset_yaml
from .errors import (
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
    classify_failure,
)
from .evaluation import evaluation_options, libreyolo_descriptor, libreyolo_report
from .files import atomic_json, digest_file, immutable_copy
from .membership import InputMembership
from .validation import resolve_train_parameters


class EventEmitter:
    """Emits strictly monotonic TrainEventV1 JSONL lines to events.jsonl."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seq = 0

    def _emit(self, **kwargs: Any) -> TrainEventV1:
        self._seq += 1
        event = TrainEventV1(seq=self._seq, **kwargs)
        write_event_jsonl(self.path, event)
        return event

    def phase(
        self, phase: str, *, metrics: dict[str, Any] | None = None
    ) -> TrainEventV1:
        return self._emit(kind="phase", phase=phase, metrics=metrics or {})

    def epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> TrainEventV1:
        progress = min(1.0, max(0.0, float(epoch) / float(total_epochs)))
        return self._emit(
            kind="epoch",
            phase="training",
            epoch=epoch,
            total_epochs=total_epochs,
            progress=progress,
            metrics=metrics,
        )

    def checkpoint(
        self,
        *,
        epoch: int,
        total_epochs: int,
        filename: str,
        sha256: str,
    ) -> TrainEventV1:
        return self._emit(
            kind="checkpoint",
            phase="training",
            epoch=epoch,
            total_epochs=total_epochs,
            detail={"filename": filename, "sha256": sha256},
        )


def _artifact_entry(role: str, path: Path) -> TrainArtifactV1:
    if path.name in SUPERVISOR_OWNED_ARTIFACTS:
        raise ValueError(f"Refusing to declare supervisor-owned artifact: {path.name}")
    return TrainArtifactV1(
        role=role,  # type: ignore[arg-type]
        filename=path.name,
        sha256=digest_file(path),
        size_bytes=path.stat().st_size,
    )


class LibreYOLOAttemptTrainer:
    """Executes one isolated LibreYOLO 1.6.0 detection training attempt."""

    def run(self, spec: TrainSpecV1, output: Path) -> TrainReceiptV1:
        output.mkdir(parents=True, exist_ok=True)
        emitter = EventEmitter(output / "events.jsonl")
        emitter.phase("preparing")

        self._validate_spec_identity(spec)
        resolved = resolve_train_parameters(
            spec.parameters,
            schema_id=spec.schema_id,
            schema_digest=spec.schema_digest,
        )
        effective_values = resolved["effective_values"]
        canonical = resolved["canonical"]
        total_epochs = int(canonical["epochs"])

        dataset_dir = Path(spec.dataset_dir)
        manifest, classes = validate_dataset(
            dataset_dir,
            expected_classes_digest=spec.classes_digest,
            expected_snapshot_digest=spec.dataset_snapshot_digest,
        )
        membership = InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output,
            manifest=manifest,
            run_id=spec.run_id,
            attempt_id=spec.attempt_id,
            config_digest=spec.config_digest,
            dataset_snapshot_digest=spec.dataset_snapshot_digest,
        )
        native_yaml = write_native_dataset_yaml(dataset_dir, output, classes)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["LIBREYOLO_FASTER_COCO_EVAL"] = "0"

        import cv2
        import libreyolo
        from libreyolo import LibreYOLO
        import torch

        if getattr(libreyolo, "__version__", None) != FRAMEWORK_VERSION:
            raise NativeRuntimeMismatch(
                f"Unexpected LibreYOLO runtime version: {getattr(libreyolo, '__version__', None)!r}"
            )

        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        resume_blob = None
        if spec.resume_checkpoint_path is not None:
            resume_blob = load_and_validate_resume_checkpoint(
                spec.resume_checkpoint_path,
                classes=classes,
                architecture_id=spec.architecture_id,
                config_digest=spec.config_digest,
                total_epochs=total_epochs,
            )

        context = AttemptContext(
            classes=classes,
            architecture_id=spec.architecture_id,
            config_digest=spec.config_digest,
            total_epochs=total_epochs,
            effective_values=effective_values,
            data_yaml=str(native_yaml),
        )
        callbacks = CheckpointCallbacks(
            context,
            output,
            emitter=emitter,
            resume=resume_blob,
            membership=membership,
        )
        if resume_blob is not None and spec.resume_checkpoint_path is not None:
            callbacks.commit_resume(spec.resume_checkpoint_path)

        model_source = spec.resume_checkpoint_path or spec.base_weights_path
        if not model_source:
            raise InputMismatchError("Base weights or resume checkpoint path is required")
        model = LibreYOLO(model_source, device="cuda:0")
        if getattr(model, "task", None) != "detect":
            raise InputMismatchError("Controlled LibreYOLO base is not a detector")

        native_trainer_class = model._trainer_class()

        class ObservedTrainer(native_trainer_class):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                callbacks.trainer = self

        model._trainer_class = lambda: ObservedTrainer

        logger = logging.getLogger("libreyolo.training.trainer")
        handler = StrictResumeLogging()
        logger.addHandler(handler)
        try:
            arguments = {
                **effective_values,
                "data": str(native_yaml),
                "project": str(output),
                "name": "framework",
                "exist_ok": False,
                "resume": spec.resume_checkpoint_path or False,
                "callbacks": callbacks,
                "loggers": [],
                "cuda_graph": False,
                "compile": False,
            }
            model.train(**arguments)
        finally:
            logger.removeHandler(handler)

        best_src = output / "framework/weights/best.pt"
        if not best_src.is_file() or callbacks.checkpoint is None:
            raise RuntimeError(
                "Native training did not commit required weights and checkpoint"
            )
        final_best_blob = load_checkpoint_safe(best_src)
        if isinstance(final_best_blob, dict) and "aap_resume" in final_best_blob:
            raise InputMismatchError("Exported best.pt must remain clean of aap_resume")

        weights_info = immutable_copy(best_src, output / "best.pt")
        emitter.phase("evaluating")

        evaluated = LibreYOLO(str(output / "best.pt"), device="cuda:0")
        if ordered_names({"names": evaluated.names}) != classes:
            raise InputMismatchError("Trained native class order mismatch")

        from libreyolo.validation.detection_validator import DetectionValidator

        class ObservedValidator(DetectionValidator):
            def _setup_dataloader(self) -> Any:
                loader = super()._setup_dataloader()
                membership.observe("val", loader.dataset.img_files)
                return loader

        evaluated.validator_class = ObservedValidator
        eval_opts = evaluation_options(
            data_yaml=str(native_yaml),
            batch=int(canonical["batch"]),
            imgsz=int(canonical["imgsz"]),
            output=output,
        )
        raw_eval = evaluated.val(**eval_opts)
        eval_backend = getattr(evaluated, "last_eval_backend", None)
        validation_digest = manifest.get("validation_digest")
        report = libreyolo_report(
            raw_eval,
            batch=int(canonical["batch"]),
            imgsz=int(canonical["imgsz"]),
            weights_sha256=weights_info["sha256"],
            validation_digest=validation_digest,
            backend=eval_backend,
        )
        comparison = libreyolo_descriptor(
            config_digest=spec.config_digest,
            classes=classes,
            validation_digest=validation_digest,
            weights_sha256=weights_info["sha256"],
            backend=eval_backend,
            options=eval_opts,
            summary=report["summary"],
        )
        if comparison is not None:
            report["comparison"] = comparison
        else:
            report["comparison_unavailable_reason"] = "evaluator_not_recorded"

        membership_path = membership.complete()
        metrics_path = output / "metrics.json"
        atomic_json(metrics_path, report)

        artifacts = [
            _artifact_entry("weights", output / "best.pt"),
            _artifact_entry("checkpoint", output / callbacks.checkpoint["name"]),
            _artifact_entry("metrics", metrics_path),
        ]
        native_config_path = output / "native-config.json"
        if callbacks.native_config_written and native_config_path.is_file():
            artifacts.append(_artifact_entry("config", native_config_path))
        if membership_path is not None and membership_path.is_file():
            artifacts.append(_artifact_entry("input_membership", membership_path))

        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=artifacts,
            serving_compatibility=serving_compatibility(),
        )
        atomic_write_json(output / "result.json", receipt)
        return receipt

    @staticmethod
    def _validate_spec_identity(spec: TrainSpecV1) -> None:
        if spec.framework_id != FRAMEWORK_ID:
            raise NativeConfigurationMismatch(
                f"Unsupported framework_id {spec.framework_id!r}; expected {FRAMEWORK_ID!r}"
            )
        if spec.task not in SUPPORTED_TASKS:
            raise NativeConfigurationMismatch(
                f"Unsupported task {spec.task!r}; expected one of {list(SUPPORTED_TASKS)}"
            )
        if architecture_by_id(spec.architecture_id) is None:
            raise NativeConfigurationMismatch(
                f"Unsupported architecture_id {spec.architecture_id!r}"
            )
        if spec.dataset_contract not in SUPPORTED_DATASET_CONTRACTS:
            raise InputMismatchError(
                f"Unsupported dataset_contract {spec.dataset_contract!r}"
            )
        if spec.init_weights_path is not None:
            raise InputMismatchError(
                "LibreYOLO trainer does not support init_weights_path; use base_weights_path or resume_checkpoint_path"
            )
        if spec.resume_checkpoint_path is None and spec.base_weights_path is None:
            raise InputMismatchError(
                "Either base_weights_path or resume_checkpoint_path is required"
            )


def _write_terminal_receipt(output_dir: Path, receipt: TrainReceiptV1) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "result.json", receipt)
    except Exception:
        pass


def execute_train(spec_path: Path, output_dir: Path) -> int:
    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except ValueError:
        previous_handler = None

    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = TrainSpecV1.model_validate(raw)
        LibreYOLOAttemptTrainer().run(spec, output_dir)
        return 0
    except KeyboardInterrupt as exc:
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="cancelled",
            failure_message=(str(exc).strip() or "Training cancelled")[:2048],
        )
        _write_terminal_receipt(output_dir, receipt)
        return 130
    except Exception as exc:
        code = classify_failure(exc)
        message = (str(exc).strip() or type(exc).__name__)[:2048]
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            failure_code=code,
            failure_message=message,
        )
        _write_terminal_receipt(output_dir, receipt)
        print(f"yoao-trainer-libreyolo failed ({code}): {message}", file=sys.stderr)
        return 1
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGTERM, previous_handler)
            except ValueError:
                pass
