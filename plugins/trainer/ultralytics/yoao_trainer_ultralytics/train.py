"""Ultralytics yoao-trainer/v1 training execution entrypoint."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
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
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
    TrainerError,
)
from .adapter import (
    UltralyticsAttemptTrainer,
    configure_offline_fonts,
    disable_telemetry,
    require_qualified_runtime,
)
from .catalog import (
    model_spec,
    serving_compatibility_for,
)
from .checkpoints import UltralyticsCheckpoints
from .dataset import is_instance_rle, validate_dataset
from .evaluation import (
    UltralyticsTrainingEvaluation,
    ultralytics_descriptor,
    ultralytics_epoch_metrics,
    ultralytics_report,
)
from .files import atomic_json, digest_file, immutable_copy
from .initialization import (
    observation as initialization_observation,
    prepare as prepare_initialization,
    tensor_identity,
    validate_input as validate_initialization_input,
)
from .input_membership import InputMembership
from .native_identity import validate_architecture
from .policies import SEGMENTATION
from .validation import (
    native_values,
    normalized,
    validate_config,
    verify_native_configuration,
)

SUPERVISOR_OWNED_ARTIFACTS = frozenset(
    {"manifest.json", "train.log", "observability.json", "config.json", "result.json"}
)


class _CancelledTraining(BaseException):
    """Raised when SIGTERM/SIGINT interrupts training cleanly."""


class _EventStream:
    """Emit strictly increasing TrainEventV1 JSONL lines to events.jsonl."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / "events.jsonl"
        self._seq = 0

    def emit(
        self,
        kind: str,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        self._seq += 1
        event = TrainEventV1(
            seq=self._seq,
            kind=kind,  # type: ignore[arg-type]
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            progress=progress,
            metrics=metrics or {},
            detail=detail or {},
        )
        write_event_jsonl(self._path, event)
        return event


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "ultralytics": FRAMEWORK_VERSION,
    }
    for pkg in ("torch", "torchvision", "numpy", "pillow", "opencv-python"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def _artifact_entry(role: str, path: Path) -> TrainArtifactV1:
    if path.name in SUPERVISOR_OWNED_ARTIFACTS:
        raise NativeConfigurationMismatch(
            f"Refusing to declare supervisor-owned artifact {path.name!r} in TrainReceiptV1"
        )
    return TrainArtifactV1(
        role=role,  # type: ignore[arg-type]
        filename=path.name,
        sha256=digest_file(path),
        size_bytes=path.stat().st_size,
    )


def _classify_failure(exc: BaseException) -> tuple[str, str]:
    torch_mod = sys.modules.get("torch")
    oom_cls = getattr(getattr(torch_mod, "cuda", None), "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        code = "cuda_oom"
    elif isinstance(exc, TrainerError):
        code = exc.failure_code
    else:
        code = "native_exception"
    message = str(exc).strip() or type(exc).__name__
    return code, message[:2048]


def _resolve_model_path(
    spec: TrainSpecV1,
    effective_values: dict[str, Any],
    *,
    has_init_pin: bool,
) -> str:
    pretrained = bool(effective_values.get("pretrained", True))
    if not pretrained and (spec.init_weights_path or has_init_pin):
        raise InputMismatchError("Random initialization cannot load project weights")
    if spec.resume_checkpoint_path:
        path = Path(spec.resume_checkpoint_path)
        if not path.is_file():
            raise InputMismatchError(
                f"Resume checkpoint does not exist: {spec.resume_checkpoint_path}"
            )
        return str(path)
    if not pretrained:
        return f"{spec.architecture_id}.yaml"
    if spec.init_weights_path:
        path = Path(spec.init_weights_path)
        if not path.is_file():
            raise InputMismatchError(
                f"Initialization weights do not exist: {spec.init_weights_path}"
            )
        return str(path)
    if spec.base_weights_path:
        path = Path(spec.base_weights_path)
        if not path.is_file():
            raise InputMismatchError(
                f"Base weights file does not exist: {spec.base_weights_path}"
            )
        return str(path)
    return f"{spec.architecture_id}.yaml"


def _load_optional_init_pin(
    spec_path: Path | None, spec_parameters: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cleaned = dict(spec_parameters)
    pin = cleaned.pop("initialization", None)
    if pin is None and spec_path is not None:
        sidecar = spec_path.parent / "initialization.json"
        if sidecar.is_file():
            pin = json.loads(sidecar.read_text(encoding="utf-8"))
    if pin is not None and not isinstance(pin, dict):
        raise InputMismatchError("initialization pin must be a JSON object")
    return pin, cleaned


_DEFAULT_ARCH_VALIDATOR = object()


def run_train(
    spec: TrainSpecV1,
    output_dir: Path,
    *,
    spec_path: Path | None = None,
    architecture_validator: Any = _DEFAULT_ARCH_VALIDATOR,
) -> TrainReceiptV1:
    """Execute one Ultralytics training attempt and write events.jsonl and result.json."""
    if architecture_validator is _DEFAULT_ARCH_VALIDATOR:
        architecture_validator = validate_architecture
    output_dir.mkdir(parents=True, exist_ok=True)
    events = _EventStream(output_dir)
    events.emit("phase", phase="preparing")

    if spec.framework_id != FRAMEWORK_ID:
        raise NativeConfigurationMismatch(
            f"Unsupported framework_id {spec.framework_id!r}; expected {FRAMEWORK_ID!r}"
        )

    config_directory = Path(
        os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / "ultralytics"))
    )
    config_directory.mkdir(parents=True, exist_ok=True)

    init_pin, raw_parameters = _load_optional_init_pin(spec_path, spec.parameters)

    manifest, classes, data_yaml = validate_dataset(
        Path(spec.dataset_dir),
        task=spec.task,
        dataset_contract=spec.dataset_contract,
        expected_classes_digest=spec.classes_digest,
    )

    validation = validate_config(
        {
            "task": spec.task,
            "architecture_id": spec.architecture_id,
            "dataset_contract": spec.dataset_contract,
            "parameters": raw_parameters,
            "schema_id": spec.schema_id,
            "schema_digest": spec.schema_digest,
            "classes": classes,
            "initialization": bool(init_pin or spec.init_weights_path),
        },
        allow_effective_passthrough=True,
    )
    if not validation.valid:
        joined = "; ".join(
            f"{issue.field or issue.code}: {issue.message}"
            for issue in validation.issues
        )
        raise NativeConfigurationMismatch(f"Invalid training configuration: {joined}")

    assert validation.schema_id is not None and validation.schema_digest is not None
    effective = dict(validation.effective_values)
    arch = model_spec(spec.architecture_id)
    segmentation = spec.task == SEGMENTATION
    native_task = "segment" if segmentation else "detect"
    raster = is_instance_rle(spec.dataset_contract)
    total_epochs = int(effective["epochs"])

    membership: InputMembership | None = None
    if "samples" in manifest:
        dataset_id = str(
            manifest.get("dataset_id")
            or manifest.get("version_id")
            or spec.dataset_snapshot_digest
        )
        dataset_sha256 = str(
            manifest.get("dataset_sha256")
            or manifest.get("package_sha256")
            or spec.dataset_snapshot_digest
        )
        membership = InputMembership(
            dataset_root=Path(spec.dataset_dir),
            output_dir=output_dir,
            manifest=manifest,
            run_id=spec.run_id,
            attempt_id=spec.attempt_id,
            config_digest=spec.config_digest,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
        )

    require_qualified_runtime()
    configure_offline_fonts()
    disable_telemetry()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer

    trainer_class: type = SegmentationTrainer if segmentation else DetectionTrainer
    rle_consumer_path: Path | None = None
    if raster:
        from .instance_rle import build_instance_rle_trainer_class, consumer_evidence

        trainer_class = build_instance_rle_trainer_class()
        rle_consumer_path = output_dir / "rle-consumer.json"
        atomic_json(rle_consumer_path, consumer_evidence())

    class AttemptTrainer(UltralyticsAttemptTrainer, trainer_class):  # type: ignore[valid-type,misc]
        def preprocess_batch(self, batch: Any) -> Any:
            result = super().preprocess_batch(batch)
            self._platform_batch_images = int(result["img"].shape[0])
            return result

        def final_eval(self) -> Any:
            self._platform_final_eval = True
            events.emit("phase", phase="evaluating")
            try:
                return super().final_eval()
            finally:
                self._platform_final_eval = False

    checkpoints = UltralyticsCheckpoints(
        output_dir=output_dir,
        architecture_id=spec.architecture_id,
        task=spec.task,
        classes=classes,
        epochs=total_epochs,
        config_digest=spec.config_digest,
        resume_checkpoint_path=spec.resume_checkpoint_path,
        architecture_validator=architecture_validator,
    )

    evaluate_train = bool(effective.get("evaluate_train", False))
    training_evaluation = (
        UltralyticsTrainingEvaluation(output_dir, membership)
        if evaluate_train
        else None
    )

    model_path = _resolve_model_path(
        spec, effective, has_init_pin=(init_pin is not None)
    )
    validate_initialization_input(
        init_pin,
        init_weights_path=spec.init_weights_path,
        resume_checkpoint_path=spec.resume_checkpoint_path,
    )
    model = YOLO(model_path, task=native_task)
    if model.task != native_task:
        raise InputMismatchError("Loaded weights do not match the frozen native task")
    if architecture_validator is not None:
        try:
            architecture_validator(model.model, spec.architecture_id)
        except (ValueError, AttributeError, KeyError, TypeError) as exc:
            raise InputMismatchError("Unqualified native base architecture") from exc

    initialization_baseline = prepare_initialization(
        init_pin,
        model,
        classes=classes,
        resume_checkpoint_path=spec.resume_checkpoint_path,
        architecture_validator=architecture_validator,
    )
    bare_init_fingerprint: tuple[str, int] | None = None
    if (
        init_pin is None
        and spec.init_weights_path
        and not spec.resume_checkpoint_path
        and getattr(model, "model", None) is not None
        and hasattr(model.model, "state_dict")
    ):
        bare_init_fingerprint = tensor_identity(model.model)

    runtime_options = {
        "data": str(data_yaml),
        "model": model_path,
        "project": str(output_dir),
        "name": "framework",
        "exist_ok": False,
        "resume": spec.resume_checkpoint_path or False,
    }
    native_config_path = output_dir / "native-config.json"

    batch_window_start = time.monotonic()
    batch_window_images = 0

    def on_train_start(trainer: Any) -> None:
        nonlocal batch_window_start, batch_window_images
        if membership is not None:
            membership.observe("train", trainer.train_loader.dataset.im_files)
            membership.observe("val", trainer.test_loader.dataset.im_files)
        checkpoints.started(trainer)
        observed_args = normalized(vars(trainer.args))
        verification = verify_native_configuration(
            effective,
            observed_args,
            runtime_options,
            schema_id=validation.schema_id or "",
            schema_digest=validation.schema_digest or "",
        )
        evidence: dict[str, Any] = {
            "optimizer": type(trainer.optimizer).__name__,
            "optimizer_groups": [
                {
                    key: normalized(val)
                    for key, val in group.items()
                    if key != "params"
                }
                for group in trainer.optimizer.param_groups
            ],
            "frozen_parameters": [
                name
                for name, parameter in trainer.model.named_parameters()
                if not parameter.requires_grad
            ],
        }
        if initialization_baseline is not None:
            evidence["initialization"] = initialization_observation(
                init_pin,
                initialization_baseline,
                trainer,
                resumed=bool(spec.resume_checkpoint_path),
            )
        elif bare_init_fingerprint is not None:
            loaded_model = getattr(trainer.model, "module", trainer.model)
            loaded_fp, loaded_count = tensor_identity(loaded_model)
            if (loaded_fp, loaded_count) != bare_init_fingerprint:
                raise InputMismatchError(
                    "Weights-only initialization tensor fingerprint changed on load"
                )
            evidence["initialization"] = {
                "mode": "weights_only",
                "source_tensor_digest": bare_init_fingerprint[0],
                "loaded_tensor_digest": loaded_fp,
                "tensor_count": loaded_count,
                "start_epoch": int(trainer.start_epoch),
            }
        atomic_json(
            native_config_path,
            {
                "config_digest": spec.config_digest,
                "observed": observed_args,
                "runtime_options": runtime_options,
                "verification": verification,
                "evidence": evidence,
            },
        )
        batch_window_start = time.monotonic()
        batch_window_images = 0
        events.emit("phase", phase="training", total_epochs=total_epochs)

    def on_train_batch_end(trainer: Any) -> None:
        nonlocal batch_window_start, batch_window_images
        images = getattr(trainer, "_platform_batch_images", None)
        if type(images) is int and images > 0:
            batch_window_images += images
        now = time.monotonic()
        elapsed = now - batch_window_start
        if elapsed >= 5.0 and batch_window_images > 0:
            detail: dict[str, Any] = {
                "batch_throughput": round(batch_window_images / elapsed, 3)
            }
            torch_mod = sys.modules.get("torch")
            if (
                torch_mod is not None
                and hasattr(torch_mod, "cuda")
                and torch_mod.cuda.is_initialized()
            ):
                detail["cuda_memory_mb"] = round(
                    torch_mod.cuda.memory_allocated() / (1024 * 1024), 2
                )
            epoch_val = max(0, int(getattr(trainer, "epoch", 0)))
            events.emit(
                "progress",
                phase="training",
                epoch=min(epoch_val, total_epochs),
                total_epochs=total_epochs,
                progress=min(1.0, epoch_val / max(total_epochs, 1)),
                detail=detail,
            )
            batch_window_start = now
            batch_window_images = 0

    def on_val_start(validator: Any) -> None:
        if membership is not None:
            membership.observe("val", validator.dataloader.dataset.im_files)
        events.emit("phase", phase="evaluating")

    def on_val_end(validator: Any) -> None:
        events.emit("phase", phase="training")

    def on_model_save(trainer: Any) -> None:
        checkpoints.saved(trainer)
        if checkpoints.checkpoint is not None:
            events.emit(
                "checkpoint",
                phase="training",
                epoch=checkpoints.checkpoint["epoch"],
                total_epochs=total_epochs,
                detail={
                    "filename": checkpoints.checkpoint["name"],
                    "sha256": checkpoints.checkpoint["sha256"],
                },
            )

    def on_fit_epoch_end(trainer: Any) -> None:
        if (
            not getattr(trainer, "_platform_final_eval", False)
            and 0 <= trainer.epoch < trainer.epochs
        ):
            metrics = ultralytics_epoch_metrics(trainer)
            if training_evaluation is not None:
                events.emit("phase", phase="evaluating")
                try:
                    metrics.update(training_evaluation(trainer))
                finally:
                    events.emit("phase", phase="training")
            completed = int(trainer.epoch) + 1
            events.emit(
                "epoch",
                phase="training",
                epoch=completed,
                total_epochs=int(trainer.epochs),
                progress=min(1.0, completed / max(int(trainer.epochs), 1)),
                metrics=metrics,
            )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_val_start", on_val_start)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    train_arguments = {
        **native_values(effective),
        "data": str(data_yaml),
        "project": str(output_dir),
        "name": "framework",
        "exist_ok": False,
        "resume": spec.resume_checkpoint_path or False,
    }
    result = model.train(trainer=AttemptTrainer, **train_arguments)

    best_src = output_dir / "framework" / "weights" / "best.pt"
    if not best_src.is_file() or checkpoints.checkpoint is None:
        raise RuntimeError(
            "Training did not produce required weights (best.pt) and checkpoint"
        )
    weights = immutable_copy(best_src, output_dir / "best.pt")

    validation_digest = str(
        manifest.get("validation_digest") or spec.dataset_snapshot_digest
    )
    report = ultralytics_report(
        result,
        task=spec.task,
        classes=classes,
        validation_digest=validation_digest,
        imgsz=int(effective["imgsz"]),
    )
    if raster:
        from .instance_rle import consumer_evidence

        report["mask_consumer"] = consumer_evidence()
        preflight_map: dict[str, Any] = {}
        for split in ("train", "val"):
            pf_path = output_dir / "framework" / f"rle-{split}-preflight.json"
            if pf_path.is_file():
                preflight_map[split] = json.loads(pf_path.read_text(encoding="utf-8"))
        if preflight_map:
            report["mask_resolution"] = preflight_map

    report["comparison"] = ultralytics_descriptor(
        config_digest=spec.config_digest,
        architecture_id=spec.architecture_id,
        task=spec.task,
        classes=classes,
        validation_digest=validation_digest,
        weights_sha256=weights["sha256"],
        validator=model.trainer.validator,
        metrics=report["summary"],
        is_rle=raster,
        data_contract_meta=manifest.get("data_contract"),
    )

    metrics_path = output_dir / "metrics.json"
    atomic_json(metrics_path, report)

    membership_path: Path | None = None
    if membership is not None:
        membership.complete()
        membership_path = output_dir / "input-membership.json"

    artifacts: list[TrainArtifactV1] = [
        _artifact_entry("weights", output_dir / weights["name"]),
        _artifact_entry("checkpoint", output_dir / checkpoints.checkpoint["name"]),
        _artifact_entry("metrics", metrics_path),
    ]
    if native_config_path.is_file():
        artifacts.append(_artifact_entry("config", native_config_path))
    if membership_path is not None and membership_path.is_file():
        artifacts.append(_artifact_entry("input_membership", membership_path))
    if raster:
        for extra_path in (
            rle_consumer_path,
            output_dir / "rle-train-preflight.json",
            output_dir / "rle-val-preflight.json",
        ):
            if extra_path is not None and extra_path.is_file():
                artifacts.append(_artifact_entry("config", extra_path))

    receipt = TrainReceiptV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="succeeded",
        artifacts=artifacts,
        serving_compatibility=serving_compatibility_for(
            arch, runtime_versions=_runtime_versions()
        ),
    )
    atomic_write_json(output_dir / "result.json", receipt)
    return receipt


def run_train_cli(spec_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_handlers: dict[int, Any] = {}

    def _handle_cancel(signum: int, frame: Any) -> None:
        raise _CancelledTraining(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_cancel)
        except (ValueError, OSError):
            pass

    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = TrainSpecV1.model_validate(raw)
        run_train(spec, output_dir, spec_path=spec_path)
        return 0
    except _CancelledTraining as exc:
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="cancelled",
            artifacts=[],
            failure_message=str(exc)[:2048],
        )
        atomic_write_json(output_dir / "result.json", receipt)
        return 130
    except Exception as exc:
        code, message = _classify_failure(exc)
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            artifacts=[],
            failure_code=code,
            failure_message=message,
        )
        atomic_write_json(output_dir / "result.json", receipt)
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
