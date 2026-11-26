"""Frozen detection dataset validation and non-mutating native data.yaml projection."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

import yaml
from yoao_plugin_sdk.trainer import canonical_json_digest

from .errors import InputMismatchError
from .files import atomic_bytes, digest_file

SOURCE_LINEAGE_CONTRACT = {"schema": 1, "scope": "accepted-reference"}
_LINEAGE_FIELDS = frozenset(
    {
        "source_id",
        "location_id",
        "path_digest",
        "scan_run_id",
        "accepted_revision_id",
        "revision_number",
        "previous_revision_id",
        "accepted_at",
        "scan_completed_at",
        "content_verified_at",
        "source_config_revision",
        "location_config_revision",
    }
)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_DETECTION_TASKS = frozenset({"detect", "image-detection"})


def ordered_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            indexed = {int(key): value for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise InputMismatchError("Invalid dataset class indexes") from exc
        if set(indexed) != set(range(len(indexed))):
            raise InputMismatchError("Invalid dataset class indexes")
        ordered = [indexed[i] for i in range(len(indexed))]
    elif isinstance(names, list):
        ordered = list(names)
    else:
        raise InputMismatchError("Dataset class names must be a list or indexed mapping")
    if not ordered or any(not isinstance(item, str) or not item for item in ordered):
        raise InputMismatchError("Dataset class names must be non-empty strings")
    return ordered


def _require_uuid(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str):
        raise InputMismatchError("Frozen source lineage identifier must be a UUID")
    try:
        parsed = str(uuid.UUID(value))
    except ValueError as exc:
        raise InputMismatchError("Frozen source lineage identifier must be a UUID") from exc
    if parsed != value:
        raise InputMismatchError("Frozen source lineage identifier must be a canonical UUID")


def _require_timestamp(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 64 or "T" not in value:
        raise InputMismatchError("Frozen source lineage timestamp must be timezone-aware ISO-8601")
    norm = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(norm)
    except ValueError as exc:
        raise InputMismatchError(
            "Frozen source lineage timestamp must be timezone-aware ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputMismatchError("Frozen source lineage timestamp must be timezone-aware ISO-8601")


def validate_source_lineage(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise InputMismatchError("Frozen dataset manifest must be a JSON object")
    declared = False
    if "source_lineage_contract" in manifest:
        contract = manifest["source_lineage_contract"]
        if (
            not isinstance(contract, dict)
            or type(contract.get("schema")) is not int
            or contract != SOURCE_LINEAGE_CONTRACT
        ):
            raise InputMismatchError("Unknown frozen source lineage contract")
        declared = True
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise InputMismatchError("Frozen dataset samples must be a list")
    for member in samples:
        if not isinstance(member, dict):
            raise InputMismatchError("Frozen sample must be a JSON object")
        if "source_lineage" not in member:
            if declared and member.get("asset_reference") is not None:
                raise InputMismatchError("Referenced sample is missing declared source lineage")
            continue
        if not declared:
            raise InputMismatchError("Source lineage present without explicit contract")
        if member.get("asset_reference") is None:
            raise InputMismatchError("Source lineage requires an asset reference")
        value = member["source_lineage"]
        if not isinstance(value, dict) or set(value) != _LINEAGE_FIELDS:
            raise InputMismatchError("Source lineage fields are incomplete or unauthorized")
        for key in ("source_id", "location_id", "scan_run_id"):
            _require_uuid(value[key])
        for key in ("accepted_revision_id", "previous_revision_id"):
            _require_uuid(value[key], nullable=True)
        if not isinstance(value["path_digest"], str) or not _SHA256_RE.fullmatch(
            value["path_digest"]
        ):
            raise InputMismatchError("Source lineage path_digest must be a SHA-256 hex string")
        for key in ("source_config_revision", "location_config_revision"):
            if type(value[key]) is not int or value[key] < 1:
                raise InputMismatchError("Source lineage revision must be a positive integer")
        rev = value["revision_number"]
        if rev is not None and (type(rev) is not int or rev < 1):
            raise InputMismatchError("Source lineage revision_number must be a positive integer")
        for key in ("accepted_at", "scan_completed_at", "content_verified_at"):
            _require_timestamp(value[key])


def _validate_manifest_task(manifest: dict[str, Any]) -> None:
    schema = manifest.get("schema")
    task = manifest.get("task")
    bridge = manifest.get("bridge")
    geometry = manifest.get("geometry")
    if schema is None:
        if task is not None and task not in _DETECTION_TASKS:
            raise InputMismatchError(f"LibreYOLO only supports detection datasets; got task={task!r}")
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
            raise InputMismatchError("LibreYOLO only supports schema-3 detection bbox datasets")
        return
    raise InputMismatchError(f"Unsupported dataset manifest schema {schema!r} for LibreYOLO")


def validate_dataset(
    dataset_dir: Path,
    *,
    expected_classes_digest: str,
    expected_snapshot_digest: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a read-only frozen YOLO detection dataset without mutating it."""
    root = dataset_dir.resolve()
    data_yaml_path = root / "data.yaml"
    classes_path = root / "classes.json"
    manifest_path = root / "manifest.json"
    for required in (data_yaml_path, classes_path, manifest_path):
        if not required.is_file():
            raise InputMismatchError(f"Frozen dataset is missing {required.name}")

    config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"train", "val", "names"}
        or config["train"] != "images/train"
        or config["val"] != "images/val"
    ):
        raise InputMismatchError("Unsupported frozen dataset paths or download configuration")

    ordered = ordered_names(config["names"])
    classes_file = json.loads(classes_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise InputMismatchError("Frozen dataset manifest must be a JSON object")

    validate_source_lineage(manifest)
    _validate_manifest_task(manifest)

    if ordered != manifest.get("classes") or classes_file != ordered:
        raise InputMismatchError("Frozen dataset class order mismatch")
    if canonical_json_digest(ordered) != expected_classes_digest:
        raise InputMismatchError("Frozen dataset classes digest does not match spec.classes_digest")
    if (
        expected_snapshot_digest is not None
        and isinstance(manifest.get("content_digest"), str)
        and manifest["content_digest"] != expected_snapshot_digest
    ):
        raise InputMismatchError(
            "Frozen dataset content_digest does not match spec.dataset_snapshot_digest"
        )

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise InputMismatchError("Frozen dataset manifest must contain samples")

    seen: set[str] = set()
    splits: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") not in {"train", "val"}:
            raise InputMismatchError("Invalid frozen dataset sample split")
        split = sample["split"]
        splits.add(split)
        raw_paths = (sample.get("image_path"), sample.get("label_path"))
        for index, raw in enumerate(raw_paths):
            if not isinstance(raw, str):
                raise InputMismatchError("Invalid frozen sample path")
            rel = PurePosixPath(raw)
            expected_root = "images" if index == 0 else "labels"
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or "\\" in raw
                or len(rel.parts) != 3
                or rel.parts[:2] != (expected_root, split)
                or raw in seen
            ):
                raise InputMismatchError("Invalid frozen sample path")
            seen.add(raw)
        image_path = root / raw_paths[0]
        label_path = root / raw_paths[1]
        if (
            image_path.stem != label_path.stem
            or not image_path.is_file()
            or not label_path.is_file()
            or image_path.is_symlink()
            or label_path.is_symlink()
            or not image_path.resolve().is_relative_to(root)
            or not label_path.resolve().is_relative_to(root)
            or digest_file(image_path) != sample.get("sha256")
            or digest_file(label_path) != sample.get("label_sha256")
        ):
            raise InputMismatchError("Frozen image or explicit label file missing or changed")
        text = label_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            values = line.split()
            if len(values) != 5:
                raise InputMismatchError("Invalid detection label")
            try:
                class_index = int(values[0])
                x, y, w, h = map(float, values[1:])
            except ValueError as exc:
                raise InputMismatchError("Invalid numeric detection label") from exc
            if (
                not 0 <= class_index < len(ordered)
                or not all(math.isfinite(v) for v in (x, y, w, h))
                or min(w, h) <= 0
                or x - w / 2 < -1e-8
                or y - h / 2 < -1e-8
                or x + w / 2 > 1 + 1e-8
                or y + h / 2 > 1 + 1e-8
            ):
                raise InputMismatchError("Invalid normalized detection label")

    if splits != {"train", "val"}:
        raise InputMismatchError("Frozen training and validation partitions required")
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
    atomic_bytes(target, rendered)
    return target
