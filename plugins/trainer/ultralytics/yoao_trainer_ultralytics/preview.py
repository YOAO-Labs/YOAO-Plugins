"""CPU-only, bounded, deterministic Ultralytics augmentation preview."""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import resource
import signal
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from yoao_plugin_sdk.trainer import (
    PREVIEW_MAX_EDGE,
    PREVIEW_MAX_OUTPUT_BOXES,
    PREVIEW_MAX_OUTPUT_BYTES,
    TRAINER_PROTOCOL_VERSION,
    PreviewOutputV1,
    PreviewReportV1,
    PreviewSpecV1,
    atomic_write_json,
    canonical_json_digest,
    classify_failure,
    digest_file,
    sanitize_failure_message,
)

from . import (
    FRAMEWORK_ID,
    FRAMEWORK_VERSION,
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
)
from .adapter import disable_telemetry, require_qualified_runtime
from .catalog import model_spec
from .policies import (
    DETECTION,
    PREVIEW_LIMITS,
    PREVIEW_POLICY,
    PREVIEW_POLICY_DIGEST,
)


def _clamp_rlimit(limit_name: int, soft: int, hard: int) -> None:
    try:
        cur_soft, cur_hard = resource.getrlimit(limit_name)
    except (ValueError, OSError):
        return
    if cur_hard != resource.RLIM_INFINITY:
        hard = min(hard, cur_hard)
        soft = min(soft, hard)
    if cur_soft != resource.RLIM_INFINITY:
        soft = min(soft, cur_soft)
    resource.setrlimit(limit_name, (soft, hard))


def enforce_limits(*, apply_rlimits: bool = True) -> None:
    """Enforce CPU-only boundary and OS resource limits before any ML framework import."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise NativeRuntimeMismatch(
            "A preview child must hide CUDA (CUDA_VISIBLE_DEVICES='') before framework imports"
        )
    if not apply_rlimits:
        return
    _clamp_rlimit(
        resource.RLIMIT_AS,
        PREVIEW_LIMITS["address_bytes"],
        PREVIEW_LIMITS["address_bytes"],
    )
    _clamp_rlimit(
        resource.RLIMIT_CPU,
        PREVIEW_LIMITS["cpu_seconds"],
        PREVIEW_LIMITS["cpu_hard_seconds"],
    )
    signal.alarm(PREVIEW_LIMITS["child_seconds"])
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = os.sched_getaffinity(0)
        if allowed:
            os.sched_setaffinity(0, {min(allowed)})


class NativeTrace:
    """Observe native augmentation operations without mutating arrays or RNG."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.operations: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def observe(self):
        from ultralytics.data.augment import (
            Albumentations,
            CutMix,
            MixUp,
            Mosaic,
            RandomPerspective,
        )
        from ultralytics.utils.instance import Instances

        originals: list[tuple[Any, str, Any]] = []

        def patch(owner: Any, key: str, wrapper: Any, *, static: bool = False) -> None:
            original = inspect.getattr_static(owner, key)
            originals.append((owner, key, original))
            callable_original = getattr(owner, key)
            replacement = wrapper(callable_original)
            setattr(owner, key, staticmethod(replacement) if static else replacement)

        def clip(original):
            def call(instance, *args, **kwargs):
                before = instance.bboxes.copy()
                result = original(instance, *args, **kwargs)
                changed = int(np.any(before != instance.bboxes, axis=1).sum())
                self.operations.append(
                    {"step": "clip", "clipped": changed, "filtered": 0}
                )
                return result

            return call

        def zero_area(original):
            def call(instance, *args, **kwargs):
                mask = original(instance, *args, **kwargs)
                self.operations.append(
                    {
                        "step": "zero_area",
                        "clipped": 0,
                        "filtered": int((~mask).sum()),
                    }
                )
                return mask

            return call

        def candidates(original):
            def call(*args, **kwargs):
                mask = original(*args, **kwargs)
                self.operations.append(
                    {
                        "step": "perspective_candidates",
                        "clipped": 0,
                        "filtered": int((~mask).sum()),
                    }
                )
                return mask

            return call

        def mosaic(original):
            def call(transform, labels):
                sources = [
                    source for label in labels for source in label["_aap_sources"]
                ]
                result = original(transform, labels)
                result["_aap_sources"] = sources
                self.operations.append({"step": "mosaic", "clipped": 0, "filtered": 0})
                return result

            return call

        def mix(original, step: str):
            def call(transform, labels, params=None):
                sources = list(labels["_aap_sources"])
                other = labels["mix_labels"][0]
                applied = not (params or {}).get("skip", False)
                filtered = (
                    len(other["cls"]) - len(params["indexes2"])
                    if applied and step == "cutmix"
                    else 0
                )
                if applied:
                    sources += other["_aap_sources"]
                result = original(transform, labels, params)
                result["_aap_sources"] = sources
                self.operations.append(
                    {
                        "step": step if applied else step + "_skipped",
                        "clipped": 0,
                        "filtered": filtered,
                    }
                )
                return result

            return call

        def albumentations(original):
            def call(transform, labels):
                before = len(labels["cls"])
                result = original(transform, labels)
                self.operations.append(
                    {
                        "step": "albumentations",
                        "clipped": 0,
                        "filtered": before - len(result["cls"]),
                    }
                )
                return result

            return call

        patch(Instances, "clip", clip)
        patch(Instances, "remove_zero_area_boxes", zero_area)
        patch(RandomPerspective, "box_candidates", candidates, static=True)
        patch(Mosaic, "_cat_labels", mosaic)
        patch(MixUp, "apply_instances", lambda orig: mix(orig, "mixup"))
        patch(CutMix, "apply_instances", lambda orig: mix(orig, "cutmix"))
        patch(Albumentations, "__call__", albumentations)
        try:
            yield
        finally:
            for owner, key, original in reversed(originals):
                setattr(owner, key, original)


def verify_pool(spec: PreviewSpecV1, input_dir: Path) -> dict[str, int]:
    """Verify preview source images and labels match frozen digests and bounds."""
    resolved_root = input_dir.resolve()
    total_bytes = 0
    total_pixels = 0
    total_boxes = 0
    label_counts: dict[str, int] = {}
    for source in spec.sources:
        image = input_dir / source.filename
        if (
            image.is_symlink()
            or not image.resolve().is_relative_to(resolved_root)
            or not image.is_file()
        ):
            raise InputMismatchError(f"Missing or escaping preview image: {source.filename}")
        if (
            image.stat().st_size != source.size_bytes
            or image.stat().st_size > PREVIEW_LIMITS["source_bytes"]
            or digest_file(image) != source.sha256
        ):
            raise InputMismatchError("Preview source image bytes differ from frozen spec")
        with Image.open(image) as opened:
            decoded = ImageOps.exif_transpose(opened)
            width, height = decoded.size
            if (width, height) != (source.width, source.height):
                raise InputMismatchError("Preview source image dimensions differ")
            pixels = width * height
            if pixels > PREVIEW_LIMITS["source_pixels"]:
                raise InputMismatchError("Preview source pixel bound exceeded")
            decoded.load()
        lines_count = 0
        if source.label_path is not None:
            label = input_dir / source.label_path
            if (
                label.is_symlink()
                or not label.resolve().is_relative_to(resolved_root)
                or not label.is_file()
            ):
                raise InputMismatchError(
                    f"Missing or escaping preview label: {source.label_path}"
                )
            if digest_file(label) != source.label_sha256:
                raise InputMismatchError("Preview source label digest differs")
            lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
            lines_count = len(lines)
            if lines_count != source.box_count or lines_count > PREVIEW_LIMITS["source_boxes"]:
                raise InputMismatchError("Preview source box count differs or exceeds bound")
        label_counts[source.task_id] = lines_count
        total_bytes += image.stat().st_size
        total_pixels += pixels
        total_boxes += lines_count
    if (
        total_bytes > PREVIEW_LIMITS["pool_bytes"]
        or total_pixels > PREVIEW_LIMITS["pool_pixels"]
        or total_boxes > PREVIEW_LIMITS["pool_boxes"]
    ):
        raise InputMismatchError("Preview pool bound exceeded")
    return label_counts


def closed_mosaic(parameters: dict[str, Any], native_vals: dict[str, Any], epoch: int) -> bool:
    epochs = int(native_vals.get("epochs") or parameters.get("epochs") or 1)
    tail = int(native_vals.get("close_mosaic") or parameters.get("close_mosaic") or 0)
    return bool(tail and epoch - 1 >= epochs - tail)


def _infer_class_count(spec: PreviewSpecV1, input_dir: Path) -> int:
    classes = spec.pool_summary.get("classes")
    if isinstance(classes, list) and classes:
        return len(classes)
    max_cls = -1
    for source in spec.sources:
        if source.label_path:
            label_file = input_dir / source.label_path
            if label_file.is_file():
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        max_cls = max(max_cls, int(parts[0]))
    return max(max_cls + 1, 1)


def _resolve_preview_native_values(spec: PreviewSpecV1) -> dict[str, Any]:
    values = dict(spec.native_values)
    if not values:
        values = {
            **spec.parameters,
            "copy_paste": 0.0,
        }
    values.pop("evaluate_train", None)
    values.update({"device": "cpu", "workers": 0, "task": "detect"})
    values.setdefault("copy_paste", 0.0)
    values.setdefault("imgsz", int(spec.parameters.get("imgsz", 640)))
    values.setdefault("batch", int(spec.parameters.get("batch", 8)))
    values.setdefault("epochs", int(spec.parameters.get("epochs", 50)))
    values.setdefault("multi_scale", float(spec.parameters.get("multi_scale", 0.0)))
    values.setdefault("close_mosaic", int(spec.parameters.get("close_mosaic", 10)))
    values.setdefault("mosaic", float(spec.parameters.get("mosaic", 1.0)))
    values.setdefault("mixup", float(spec.parameters.get("mixup", 0.0)))
    values.setdefault("cutmix", float(spec.parameters.get("cutmix", 0.0)))
    return values


def _pipeline_sha256(ultralytics_mod: Any) -> str:
    root = Path(ultralytics_mod.__file__).parent
    if not root.is_dir():
        return canonical_json_digest({"plugin_preview": digest_file(Path(__file__))})
    return canonical_json_digest(
        {
            str(path.relative_to(root)): digest_file(path)
            for path in sorted(root.rglob("*.py"))
        }
    )


def run_preview(
    spec: PreviewSpecV1,
    input_dir: Path,
    output_dir: Path,
    *,
    apply_rlimits: bool = True,
    traced: bool = True,
) -> PreviewReportV1:
    """Run deterministic CPU-only native Ultralytics augmentation preview."""
    enforce_limits(apply_rlimits=apply_rlimits)

    if spec.framework_id != FRAMEWORK_ID:
        raise NativeConfigurationMismatch(
            f"Unsupported preview framework_id: {spec.framework_id!r}"
        )
    if spec.task != DETECTION:
        raise NativeConfigurationMismatch(
            f"Augmentation preview supports only detect task, got {spec.task!r}"
        )
    if spec.policy_digest != PREVIEW_POLICY_DIGEST:
        raise InputMismatchError("Preview policy_digest differs from qualified policy")
    arch = model_spec(spec.architecture_id)
    if arch.task != DETECTION:
        raise NativeConfigurationMismatch(
            f"Preview architecture {arch.id!r} is not a detection model"
        )

    native_vals = _resolve_preview_native_values(spec)
    total_epochs = int(native_vals.get("epochs", 1))
    if spec.epoch > total_epochs:
        raise NativeConfigurationMismatch("Preview epoch exceeds configured epochs")
    if float(native_vals["imgsz"]) * (1.0 + float(native_vals["multi_scale"])) > PREVIEW_MAX_EDGE:
        raise NativeConfigurationMismatch(
            "Configured imgsz * (1 + multi_scale) exceeds preview output edge limit"
        )

    label_counts = verify_pool(spec, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG

    require_qualified_runtime()
    disable_telemetry()

    try:
        import cv2

        cv2.setNumThreads(1)
        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = "unavailable"

    torch.set_num_threads(1)
    if hasattr(torch, "get_num_interop_threads") and torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)

    base_path: Path | None = None
    expected_base_sha: str | None = None
    stride = 32
    models_dir = Path(os.environ.get("TRAINING_MODELS_DIR", "/models"))
    if arch.base_asset_filename:
        candidate = models_dir / arch.base_asset_filename
        if candidate.is_file():
            if arch.base_asset_sha256 and digest_file(candidate) != arch.base_asset_sha256:
                raise InputMismatchError("Controlled preview base weights digest mismatch")
            base_path = candidate
            expected_base_sha = arch.base_asset_sha256
            model = YOLO(str(candidate), task="detect")
            stride = max(int(model.model.stride.max()), 32)

    default_keys = (
        set(DEFAULT_CFG.keys())
        if isinstance(DEFAULT_CFG, dict)
        else set(vars(DEFAULT_CFG).keys())
    )
    cfg_overrides = (
        {key: val for key, val in native_vals.items() if key in default_keys}
        if len(default_keys) >= 20
        else dict(native_vals)
    )
    args = get_cfg(DEFAULT_CFG, cfg_overrides)
    observed = {key: getattr(args, key) for key in cfg_overrides if hasattr(args, key)}
    for key, val in cfg_overrides.items():
        if key in observed and observed[key] != val:
            raise NativeConfigurationMismatch(
                f"Native preview parameter observation mismatch for {key!r}"
            )

    num_classes = _infer_class_count(spec, input_dir)
    raw_classes = spec.pool_summary.get("classes")
    classes = (
        list(raw_classes)
        if isinstance(raw_classes, list) and len(raw_classes) == num_classes
        else [f"class_{idx}" for idx in range(num_classes)]
    )

    first_image = input_dir / spec.sources[0].filename
    img_root = str(first_image.parent)
    dataset = build_yolo_dataset(
        args,
        img_root,
        batch=int(native_vals["batch"]),
        data={"names": dict(enumerate(classes)), "nc": len(classes)},
        stride=stride,
    )
    if len(dataset) != len(spec.sources):
        raise InputMismatchError("Native validation removed a frozen preview pool member")

    by_path = {
        str((input_dir / source.filename).resolve()): source
        for source in spec.sources
    }
    trace = NativeTrace()
    original_get = dataset.get_image_and_label

    def get_label(index: int) -> dict[str, Any]:
        result = original_get(index)
        source = by_path[str(Path(result["im_file"]).resolve())]
        result["_aap_sources"] = [source.task_id]
        trace.reads.append(source.task_id)
        trace.operations.append(
            {
                "step": "native_label_scan",
                "clipped": 0,
                "filtered": label_counts[source.task_id] - len(result["cls"]),
            }
        )
        return result

    dataset.get_image_and_label = get_label
    mosaic_closed_flag = closed_mosaic(spec.parameters, native_vals, spec.epoch)
    if mosaic_closed_flag:
        dataset.close_mosaic(args)
    for index in range(len(dataset)):
        dataset.load_image(index)

    trainer = object.__new__(DetectionTrainer)
    trainer.args, trainer.device, trainer.stride = args, torch.device("cpu"), stride

    outputs: list[PreviewOutputV1] = []
    with trace.observe() if traced else contextlib.nullcontext():
        for index in range(spec.count):
            trace.reads, trace.operations = [], []
            result = dataset[index % len(dataset)]
            sources = list(result["_aap_sources"])
            batch = trainer.preprocess_batch(dataset.collate_fn([result]))
            image = (
                (batch["img"][0].clamp(0, 1) * 255)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            height, width = int(image.shape[0]), int(image.shape[1])
            boxes = batch["bboxes"].tolist()
            indexes = batch["cls"].view(-1).tolist()
            if (
                max(width, height) > PREVIEW_MAX_EDGE
                or len(boxes) > PREVIEW_MAX_OUTPUT_BOXES
                or (boxes and not np.isfinite(np.asarray(boxes)).all())
                or any(
                    type(v) not in (int, float)
                    or int(v) != v
                    or not 0 <= v < len(classes)
                    for v in indexes
                )
            ):
                raise InputMismatchError("Native preview output violates bounds")
            filename = f"preview-{index}.png"
            out_path = output_dir / filename
            tmp_path = output_dir / f".tmp-{filename}"
            try:
                Image.fromarray(image).save(tmp_path, format="PNG", compress_level=3)
                os.replace(tmp_path, out_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            size_bytes = out_path.stat().st_size
            if size_bytes > PREVIEW_MAX_OUTPUT_BYTES:
                raise InputMismatchError("Native preview PNG exceeds byte bound")
            outputs.append(
                PreviewOutputV1(
                    filename=filename,
                    sha256=digest_file(out_path),
                    size_bytes=size_bytes,
                    width=width,
                    height=height,
                    box_count=len(boxes),
                    metadata={
                        "boxes": [
                            {"class_index": int(c), "xywh": box}
                            for c, box in zip(indexes, boxes, strict=True)
                        ],
                        "source_draws": sources,
                        "sources_read": list(trace.reads),
                        "operations": list(trace.operations),
                        "filtered": sum(item["filtered"] for item in trace.operations),
                        "clipped": sum(item["clipped"] for item in trace.operations),
                    },
                )
            )

    if torch.cuda.is_initialized():
        raise NativeRuntimeMismatch("Native preview unexpectedly initialized CUDA")
    if base_path is not None and expected_base_sha is not None:
        if digest_file(base_path) != expected_base_sha:
            raise InputMismatchError("Preview modified controlled base weights")

    transform_parameters = {
        key: getattr(args, key)
        for key in ("mosaic", "mixup", "cutmix", "copy_paste")
    }
    report = PreviewReportV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="succeeded",
        input_digest=spec.input_digest,
        policy_digest=spec.policy_digest,
        cuda_initialized=False,
        outputs=outputs,
        warnings=[],
        runtime={
            "framework": FRAMEWORK_ID,
            "framework_version": getattr(ultralytics, "__version__", FRAMEWORK_VERSION),
            "pipeline_sha256": _pipeline_sha256(ultralytics),
            "torch": getattr(torch, "__version__", "unknown"),
            "numpy": np.__version__,
            "opencv": opencv_version,
            "pillow": importlib.metadata.version("pillow"),
            "stride": stride,
            "cuda_initialized": False,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        details={
            "policy": PREVIEW_POLICY,
            "mosaic_closed": mosaic_closed_flag,
            "native_parameters": dict(spec.native_values) if spec.native_values else observed,
            "transform_parameters": transform_parameters,
        },
    )
    report_path = output_dir / "report.json"
    atomic_write_json(report_path, report)
    if report_path.stat().st_size > PREVIEW_LIMITS["report_bytes"]:
        raise InputMismatchError("Preview report exceeds byte bound")
    return report


def run_preview_cli(
    spec_path: Path,
    output_dir: Path,
    *,
    apply_rlimits: bool = True,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_digest = "0" * 64
    policy_digest = PREVIEW_POLICY_DIGEST
    try:
        if spec_path.stat().st_size > PREVIEW_LIMITS["report_bytes"]:
            raise InputMismatchError("Preview spec exceeds byte bound")
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = PreviewSpecV1.model_validate(raw)
        input_digest = spec.input_digest
        policy_digest = spec.policy_digest
        run_preview(spec, spec_path.parent, output_dir, apply_rlimits=apply_rlimits)
        return 0
    except Exception as exc:
        code = classify_failure(exc)
        if code == "native_exception":
            code = "preview_failed"
        message = sanitize_failure_message(exc, type(exc).__name__)
        cuda_init = False
        import sys

        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and hasattr(torch_mod, "cuda"):
            try:
                cuda_init = bool(torch_mod.cuda.is_initialized())
            except Exception:
                cuda_init = False
        failed_report = PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            input_digest=input_digest,
            policy_digest=policy_digest,
            cuda_initialized=cuda_init,
            outputs=[],
            failure_code=code,
            failure_message=message,
        )
        atomic_write_json(output_dir / "report.json", failed_report)
        return 1
