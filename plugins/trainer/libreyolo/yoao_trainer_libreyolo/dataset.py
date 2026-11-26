"""Frozen detection dataset validation and non-mutating native data.yaml projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yoao_plugin_sdk.trainer import (
    SOURCE_LINEAGE_CONTRACT,
    InputMismatchError,
    atomic_write_bytes,
    ordered_names,
    validate_frozen_manifest_core,
    validate_sample_files,
    validate_source_lineage,
    validate_yolo_bbox_label,
)

_DETECTION_TASKS = frozenset({"detect", "image-detection"})


def _validate_manifest_task(manifest: dict[str, Any]) -> None:
    schema = manifest.get("schema")
    task = manifest.get("task")
    bridge = manifest.get("bridge")
    geometry = manifest.get("geometry")
    if schema is None:
        if task is not None and task not in _DETECTION_TASKS:
            raise InputMismatchError(
                f"LibreYOLO only supports detection datasets; got task={task!r}"
            )
        return
    if type(schema) is not int:
        raise InputMismatchError("Dataset manifest schema must be an integer")
    if schema == 1:
        if (
            manifest.get("task", "image-detection") != "image-detection"
            or "geometry" in manifest
            or "bridge" in manifest
            or manifest.get("format", "ultralytics-detection") != "ultralytics-detection"
        ):
            raise InputMismatchError("Unsupported schema-1 dataset contract for LibreYOLO")
        return
    if schema == 3:
        if (
            task != "detect"
            or manifest.get("media_type") != "image"
            or manifest.get("format") != "ultralytics-yolo"
            or geometry != "normalized-bbox"
            or bridge != "ultralytics-bbox-1"
        ):
            raise InputMismatchError(
                "LibreYOLO only supports schema-3 detection bbox datasets"
            )
        return
    raise InputMismatchError(
        f"Unsupported dataset manifest schema {schema!r} for LibreYOLO"
    )


def validate_dataset(
    dataset_dir: Path,
    *,
    expected_classes_digest: str,
    expected_snapshot_digest: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a read-only frozen YOLO detection dataset without mutating it."""
    root = dataset_dir.resolve()
    data_yaml_path = root / "data.yaml"
    if data_yaml_path.is_symlink():
        raise InputMismatchError("Frozen dataset data.yaml must not be a symlink")
    if not data_yaml_path.is_file():
        raise InputMismatchError("Frozen dataset is missing data.yaml")
    try:
        data_yaml_config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputMismatchError(f"Failed to parse frozen dataset data.yaml: {exc}") from exc
    if not isinstance(data_yaml_config, dict):
        raise InputMismatchError(
            "Unsupported frozen dataset paths or download configuration"
        )

    manifest, ordered = validate_frozen_manifest_core(
        root,
        expected_classes_digest=expected_classes_digest,
        expected_snapshot_digest=expected_snapshot_digest,
        data_yaml_config=data_yaml_config,
        require_classes_file=True,
    )
    _validate_manifest_task(manifest)
    validated = validate_sample_files(root, manifest, require_splits=("train", "val"))
    for _sample, _image_path, label_path in validated:
        validate_yolo_bbox_label(label_path.read_text(encoding="utf-8"), len(ordered))
    return manifest, ordered


def write_native_dataset_yaml(
    dataset_dir: Path, output_dir: Path, classes: list[str]
) -> Path:
    """Write output/libreyolo-data.yaml pointing at read-only dataset_dir without mutating dataset_dir."""
    source_root = dataset_dir.resolve()
    target = output_dir / "libreyolo-data.yaml"
    rendered = yaml.safe_dump(
        {
            "train": str(source_root / "images/train"),
            "val": str(source_root / "images/val"),
            "names": list(classes),
        },
        allow_unicode=True,
    ).encode("utf-8")
    atomic_write_bytes(target, rendered)
    return target


__all__ = [
    "SOURCE_LINEAGE_CONTRACT",
    "ordered_names",
    "validate_dataset",
    "validate_source_lineage",
    "write_native_dataset_yaml",
]
