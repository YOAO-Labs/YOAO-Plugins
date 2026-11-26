"""Canonical LibreYOLO detection evaluation options, metrics projection, and descriptors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
from typing import Any

from yoao_plugin_sdk.trainer import canonical_json_digest

from . import FRAMEWORK_ID, FRAMEWORK_VERSION
from .validation import normalized

CANONICAL_METRIC_KEYS = (
    "metrics/mAP50-95",
    "metrics/mAP50",
    "metrics/mAP75",
    "metrics/AR100",
)

NATIVE_ALIAS_SEMANTICS = {
    "metrics/precision": "mAP50-95",
    "metrics/recall": "AR100",
}


def metrics_of(raw: dict[str, Any] | None) -> dict[str, float]:
    if not raw:
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        if (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and math.isfinite(float(value))
        ):
            as_float = float(value)
            result[str(key)] = 0.0 if as_float == 0.0 else as_float
    return result


def strip_metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
    """Drop LibreYOLO precision/recall aliases so summaries report true mAP/AR semantics."""
    return {
        key: value
        for key, value in metrics.items()
        if "precision" not in key and "recall" not in key
    }


def evaluation_options(
    *, data_yaml: str, batch: int, imgsz: int, output: Path
) -> dict[str, Any]:
    return {
        "data": data_yaml,
        "split": "val",
        "batch": batch,
        "imgsz": imgsz,
        "conf": 0.001,
        "iou": 0.65,
        "max_det": 300,
        "eval_max_det": 100,
        "half": False,
        "workers": 2,
        "device": "cuda:0",
        "augment": False,
        "plots": False,
        "save_json": False,
        "faster_coco_eval": False,
        "allow_download_scripts": False,
        "project": str(output),
        "name": "evaluation",
        "exist_ok": False,
    }


def libreyolo_report(
    raw: dict[str, Any],
    *,
    batch: int,
    imgsz: int,
    weights_sha256: str,
    validation_digest: str | None,
    backend: str | None,
) -> dict[str, Any]:
    native = metrics_of(raw)
    if any(key not in native for key in CANONICAL_METRIC_KEYS):
        raise ValueError("Native detection evaluation is incomplete")
    options = evaluation_options(data_yaml="", batch=batch, imgsz=imgsz, output=Path("."))
    for key in ("data", "project", "name", "exist_ok"):
        options.pop(key)
    return {
        "summary": {key: native[key] for key in CANONICAL_METRIC_KEYS},
        "native_raw": native,
        "per_class": [],
        "per_class_available": False,
        "native_alias_semantics": dict(NATIVE_ALIAS_SEMANTICS),
        "validation_digest": validation_digest,
        "evaluation": {
            "framework": FRAMEWORK_ID,
            "version": FRAMEWORK_VERSION,
            "task": "detection",
            "evaluator": backend,
            "weights_sha256": weights_sha256,
            "parameters": options,
            "independent_test_set": False,
            "deterministic_algorithms": True,
            "bitwise_resume_guaranteed": False,
        },
    }


def _hash_source_files(paths: list[Path]) -> str:
    return canonical_json_digest(
        [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    )


def _optional_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_find_spec(module_name: str) -> Any:
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def evaluator_source_sha256() -> str:
    """Hash installed libreyolo + pycocotools sources without importing torch, or fall back to plugin module hash."""
    sources: list[Path] = []
    libreyolo_spec = _safe_find_spec("libreyolo")
    if libreyolo_spec is not None and libreyolo_spec.origin:
        sources.extend(sorted(Path(libreyolo_spec.origin).parent.rglob("*.py")))
    if _safe_find_spec("pycocotools") is not None:
        for module_name in ("pycocotools.cocoeval", "pycocotools.coco", "pycocotools.mask"):
            spec = _safe_find_spec(module_name)
            if spec is not None and spec.origin:
                sources.append(Path(spec.origin))
    if not sources:
        sources = [Path(__file__).resolve()]
    return _hash_source_files(sources)


def evaluator_descriptor_summary() -> dict[str, Any]:
    return {
        "framework": FRAMEWORK_ID,
        "version": FRAMEWORK_VERSION,
        "task": "detect",
        "implementation": "libreyolo.val/pycocotools.COCOeval",
        "metrics": list(CANONICAL_METRIC_KEYS),
        "native_alias_semantics": dict(NATIVE_ALIAS_SEMANTICS),
        "per_class_available": False,
        "deterministic_algorithms": True,
        "source_sha256": evaluator_source_sha256(),
    }


def libreyolo_descriptor(
    *,
    config_digest: str,
    classes: list[str],
    validation_digest: str | None,
    weights_sha256: str,
    backend: str | None,
    options: dict[str, Any],
    summary: dict[str, float],
) -> dict[str, Any] | None:
    if backend is None:
        return None
    if not isinstance(backend, str) or not (
        backend == "pycocotools" or backend.startswith("pycocotools ")
    ):
        raise ValueError("Unexpected final evaluator backend")
    import libreyolo
    from pycocotools import coco, cocoeval, mask

    root = Path(libreyolo.__file__).parent
    clean_options = {
        key: value
        for key, value in options.items()
        if key not in {"data", "project", "name", "exist_ok"}
    }
    clean_options.update(
        backend=backend,
        pycocotools_version=_optional_version("pycocotools") or "unknown",
        torch_version=_optional_version("torch") or "unknown",
        torchvision_version=_optional_version("torchvision") or "unknown",
    )
    sources = sorted(root.rglob("*.py")) + [
        Path(cocoeval.__file__),
        Path(coco.__file__),
        Path(mask.__file__),
    ]
    return {
        "revision": 1,
        "config_digest": config_digest,
        "candidate": {
            "weights_sha256": weights_sha256,
            "selection": "native_best",
            "best_epoch": None,
        },
        "benchmark": {
            "validation_digest": validation_digest,
            "classes": list(classes),
            "class_space": "frozen_native",
        },
        "evaluator": {
            "framework": FRAMEWORK_ID,
            "version": FRAMEWORK_VERSION,
            "task": "detection",
            "implementation": "libreyolo.val/pycocotools.COCOeval",
            "source_sha256": _hash_source_files(sources),
            "options": normalized(clean_options),
        },
        "semantics": {
            key: {
                "unit": "fraction",
                "direction": "higher",
                "minimum": 0,
                "maximum": 1,
                "unavailable_below": 0,
            }
            for key in summary
            if key.startswith("metrics/")
        },
    }
