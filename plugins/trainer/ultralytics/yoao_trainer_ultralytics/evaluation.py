"""Evaluator descriptors, source SHA-256 identity, report projection, and RNG-isolated train eval."""

from __future__ import annotations

from copy import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import math
from pathlib import Path
from typing import Any

from . import FRAMEWORK_ID, FRAMEWORK_VERSION
from .catalog import model_spec
from .files import canonical_digest
from .policies import SEGMENTATION
from .validation import normalized


def _optional_version(pkg: str, default: str = "unavailable") -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return default


def metrics_of(raw: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): (0.0 if float(value) == 0.0 else float(value))
        for key, value in raw.items()
        if not isinstance(value, bool)
        and isinstance(value, (float, int))
        and math.isfinite(float(value))
    }


def ultralytics_epoch_metrics(trainer: Any) -> dict[str, float]:
    """Include epoch-averaged training losses beside validation metrics."""
    metrics = metrics_of(trainer.metrics or {})
    if trainer.tloss is not None:
        metrics.update(
            metrics_of(trainer.label_loss_items(trainer.tloss, prefix="train"))
        )
    return metrics


def descriptor(
    *,
    config_digest: str,
    task: str,
    classes: list[str],
    validation_digest: str,
    weights_sha256: str,
    implementation: str,
    source_files: list[str | Path],
    options: dict[str, Any],
    metrics: dict[str, float],
    is_rle: bool = False,
) -> dict[str, Any]:
    segmentation = task == SEGMENTATION
    return {
        "revision": 3 if is_rle else (2 if segmentation else 1),
        "config_digest": config_digest,
        "candidate": {
            "weights_sha256": weights_sha256,
            "selection": "native_best",
            "best_epoch": None,
        },
        "benchmark": {
            "validation_digest": validation_digest,
            "classes": classes,
            "class_space": "frozen_native",
        },
        "evaluator": {
            "framework": FRAMEWORK_ID,
            "version": FRAMEWORK_VERSION,
            "task": "instance-segmentation" if segmentation else "detection",
            "implementation": implementation,
            "source_sha256": canonical_digest(
                [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files]
            ),
            "options": normalized(options),
        },
        "semantics": {
            (key if segmentation else key.removesuffix("(B)")): {
                "unit": "fraction",
                "direction": "higher",
                "minimum": 0,
                "maximum": 1,
                "unavailable_below": 0,
            }
            for key in metrics
            if key.startswith("metrics/")
        },
    }


def ultralytics_descriptor(
    *,
    config_digest: str,
    architecture_id: str,
    task: str,
    classes: list[str],
    validation_digest: str,
    weights_sha256: str,
    validator: Any,
    metrics: dict[str, float],
    is_rle: bool = False,
    data_contract_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = vars(validator.args)
    fields = (
        "split",
        "imgsz",
        "conf",
        "iou",
        "max_det",
        "augment",
        "single_cls",
        "classes",
        "agnostic_nms",
        "dnn",
        "compile",
        "nms",
        "quantize",
        "channels_last",
        "save_json",
        "fraction",
    )
    options: dict[str, Any] = {key: args.get(key) for key in fields}
    segmentation = task == SEGMENTATION
    if segmentation:
        options.update({key: args.get(key) for key in ("overlap_mask", "mask_ratio")})
        process = getattr(validator, "process", None)
        if process is not None:
            options["mask_processing"] = f"{process.__module__}.{process.__name__}"
    options.update(
        half=args.get("quantize") == 16,
        device=str(validator.device),
        batch=validator.dataloader.batch_size,
        workers=validator.dataloader.num_workers,
        rect=validator.dataloader.dataset.rect,
        imgsz=validator.dataloader.dataset.imgsz,
        torch_version=_optional_version("torch"),
        torchvision_version=_optional_version("torchvision"),
        end2end=bool(getattr(validator, "end2end", False)),
    )
    expected = model_spec(architecture_id)
    if options["end2end"] != expected.end2end or (
        expected.end2end and options["nms"] is not False
    ):
        raise ValueError("Observed evaluator inference branch differs from frozen model")

    source_files: list[str | Path] = []
    try:
        from ultralytics.data import augment, dataset, utils as data_utils
        from ultralytics.engine import validator as engine
        from ultralytics.utils import metrics as native_metrics, ops

        source_files.extend(
            [
                inspect.getfile(type(validator)),
                engine.__file__,
                native_metrics.__file__,
                ops.__file__,
                augment.__file__,
                dataset.__file__,
            ]
        )
        if segmentation:
            source_files.append(data_utils.__file__)
        if is_rle:
            from ultralytics.utils import loss
            from .instance_rle import InstanceRLEDataset, consumer_evidence

            options["mask_consumer"] = consumer_evidence()
            if data_contract_meta is not None:
                options["data_contract"] = data_contract_meta
            source_files.extend([inspect.getfile(InstanceRLEDataset), loss.__file__])
    except (ImportError, TypeError, OSError):
        source_files = [Path(__file__)]
        if is_rle:
            from .instance_rle import InstanceRLEDataset, consumer_evidence

            options["mask_consumer"] = consumer_evidence()
            if data_contract_meta is not None:
                options["data_contract"] = data_contract_meta
            source_files.append(Path(inspect.getfile(InstanceRLEDataset)))

    return descriptor(
        config_digest=config_digest,
        task=task,
        classes=classes,
        validation_digest=validation_digest,
        weights_sha256=weights_sha256,
        implementation=f"{type(validator).__module__}.{type(validator).__name__}",
        source_files=source_files,
        options=options,
        metrics=metrics,
        is_rle=is_rle,
    )


def describe_evaluators() -> dict[str, dict[str, Any]]:
    """Build static/runtime evaluator descriptors for TrainerCapabilitiesV1."""
    eval_file = Path(__file__).resolve()
    rle_file = eval_file.parent / "instance_rle.py"
    detect_sources: list[Path] = [eval_file]
    segment_sources: list[Path] = [eval_file, rle_file]
    runtime_available = False
    try:
        from ultralytics.data import augment, dataset, utils as data_utils
        from ultralytics.engine import validator as engine
        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.models.yolo.segment import SegmentationValidator
        from ultralytics.utils import loss, metrics as native_metrics, ops

        runtime_available = True
        common = [
            Path(engine.__file__),
            Path(native_metrics.__file__),
            Path(ops.__file__),
            Path(augment.__file__),
            Path(dataset.__file__),
        ]
        detect_sources = [Path(inspect.getfile(DetectionValidator)), *common, eval_file]
        segment_sources = [
            Path(inspect.getfile(SegmentationValidator)),
            *common,
            Path(data_utils.__file__),
            Path(loss.__file__),
            rle_file,
            eval_file,
        ]
    except ImportError:
        pass

    def _source_sha(paths: list[Path]) -> str:
        return canonical_digest(
            [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        )

    return {
        "detect": {
            "framework": FRAMEWORK_ID,
            "framework_version": FRAMEWORK_VERSION,
            "task": "detection",
            "implementation": "ultralytics.models.yolo.detect.val.DetectionValidator",
            "runtime_available": runtime_available,
            "source_sha256": _source_sha(detect_sources),
        },
        "segment": {
            "framework": FRAMEWORK_ID,
            "framework_version": FRAMEWORK_VERSION,
            "task": "instance-segmentation",
            "implementation": "ultralytics.models.yolo.segment.val.SegmentationValidator",
            "rle_implementation": "yoao_trainer_ultralytics.instance_rle.InstanceRLEValidator",
            "runtime_available": runtime_available,
            "source_sha256": _source_sha(segment_sources),
        },
    }


def ultralytics_report(
    result: Any,
    *,
    task: str,
    classes: list[str],
    validation_digest: str,
    imgsz: int,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    segmentation = task == SEGMENTATION
    fields = ("precision", "recall", "mAP50", "mAP50-95")
    if segmentation:
        fields = tuple(f"{key}({kind})" for kind in ("B", "M") for key in fields)
    per_class: list[dict[str, Any]] = []
    for position, class_index in enumerate(result.box.ap_class_index):
        index = int(class_index)
        values = [float(value) for value in result.class_result(position)]
        if not 0 <= index < len(classes) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("Invalid class evaluation result")
        per_class.append(
            {
                "index": index,
                "name": classes[index],
                **dict(zip(fields, values, strict=True)),
            }
        )
    report: dict[str, Any] = {
        "summary": metrics_of(result.results_dict),
        "per_class": per_class,
        "validation_digest": validation_digest,
        "evaluation": {
            "framework": FRAMEWORK_ID,
            "version": FRAMEWORK_VERSION,
            "task": "instance-segmentation" if segmentation else "detection",
            "split": "val",
            "imgsz": imgsz,
            "independent_test_set": False,
        },
    }
    if comparison is not None:
        report["comparison"] = comparison
    return report


class UltralyticsTrainingEvaluation:
    """Evaluate frozen training members with the validation recipe, RNG-isolated."""

    def __init__(self, output_dir: Path, membership: Any = None) -> None:
        self.output = output_dir
        self.membership = membership
        self.validator = None

    def __call__(self, trainer: Any) -> dict[str, float]:
        import random
        import numpy as np
        import torch

        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = [trainer.device.index] if trainer.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                return self.evaluate(trainer)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def evaluate(self, trainer: Any) -> dict[str, float]:
        if self.validator is None:
            args = copy(trainer.validator.args)
            args.split = "train"
            args.rect = trainer.validator.dataloader.dataset.rect
            args.augment = args.plots = args.save_json = args.save_txt = False
            args.workers = 0
            validator = type(trainer.validator)(
                save_dir=self.output / "training-evaluation", args=args
            )
            validator.data, validator.device = trainer.data, trainer.device
            validator.stride, validator.names = trainer.stride, trainer.model.names
            validator.dataloader = validator.get_dataloader(
                trainer.data["train"], trainer.batch_size
            )
            if self.membership is not None:
                self.membership.observe("train", validator.dataloader.dataset.im_files)
            self.validator = validator
        return {
            f"train/{key}": value
            for key, value in metrics_of(self.validator(trainer=trainer)).items()
            if key.startswith("metrics/")
        }
