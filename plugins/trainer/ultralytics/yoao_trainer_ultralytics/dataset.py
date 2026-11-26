"""Frozen dataset validation for detection, simple-polygon segmentation, and instance RLE."""

from __future__ import annotations

from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any
import uuid

import yaml

from . import DatasetContractError
from .files import canonical_digest, digest_file
from .policies import (
    DETECTION,
    INSTANCE_RLE_LIMITS,
    LEGACY_DETECTION,
    LEGACY_SEGMENTATION,
    SEGMENTATION,
)

LEGACY_SEGMENTATION_CONTRACT = {
    "schema": 2,
    "task": LEGACY_SEGMENTATION,
    "geometry": "normalized-simple-polygon",
    "bridge": "ultralytics-simple-polygon-1",
}
DETECTION_CONTRACT = {
    "schema": 3,
    "task": DETECTION,
    "media_type": "image",
    "format": "ultralytics-yolo",
    "geometry": "normalized-bbox",
    "bridge": "ultralytics-bbox-1",
}
SEGMENTATION_CONTRACT = {
    "schema": 3,
    "task": SEGMENTATION,
    "media_type": "image",
    "format": "ultralytics-yolo",
    "geometry": "normalized-simple-polygon",
    "bridge": "ultralytics-simple-polygon-1",
}
INSTANCE_RLE_CONTRACT = {
    "schema": 4,
    "task": SEGMENTATION,
    "media_type": "image",
    "format": "aap-instance-rle",
    "geometry": "instance-mask",
    "bridge": "ultralytics-instance-rle-1",
}

MAX_VERTICES = 512
MAX_INSTANCES = 300
MAX_IMAGE_VERTICES = 8192

SOURCE_LINEAGE_CONTRACT = {"schema": 1, "scope": "accepted-reference"}
_LINEAGE_FIELDS = {
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
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def normalize_task(task: str | None) -> str:
    if task in {DETECTION, LEGACY_DETECTION}:
        return DETECTION
    if task in {SEGMENTATION, LEGACY_SEGMENTATION}:
        return SEGMENTATION
    raise DatasetContractError("unsupported_task", f"Unsupported task: {task!r}")


def is_instance_rle(contract: dict[str, Any] | str | None) -> bool:
    if isinstance(contract, str):
        return contract == INSTANCE_RLE_CONTRACT["bridge"]
    if isinstance(contract, dict):
        return contract.get("bridge") == INSTANCE_RLE_CONTRACT["bridge"]
    return False


def data_contract(record: dict[str, Any]) -> dict[str, Any]:
    schema = record.get("schema")
    if type(schema) is int and schema == 1:
        if (
            record.get("task", LEGACY_DETECTION) != LEGACY_DETECTION
            or "geometry" in record
            or "bridge" in record
            or record.get("format", "ultralytics-detection") != "ultralytics-detection"
        ):
            raise DatasetContractError(
                "unknown_data_contract", "Unknown detection dataset contract"
            )
        return {"schema": 1, "task": LEGACY_DETECTION}
    if (
        type(schema) is int
        and schema == 2
        and all(
            record.get(key) == value
            for key, value in LEGACY_SEGMENTATION_CONTRACT.items()
        )
        and record.get("format", "ultralytics-segmentation")
        == "ultralytics-segmentation"
    ):
        return dict(LEGACY_SEGMENTATION_CONTRACT)
    if type(schema) is int and schema == 3:
        expected = {
            DETECTION: DETECTION_CONTRACT,
            SEGMENTATION: SEGMENTATION_CONTRACT,
        }.get(record.get("task"))
        if expected and all(
            record.get(key) == value for key, value in expected.items()
        ):
            return dict(expected)
    if (
        type(schema) is int
        and schema == 4
        and all(
            record.get(key) == value for key, value in INSTANCE_RLE_CONTRACT.items()
        )
    ):
        return dict(INSTANCE_RLE_CONTRACT)
    raise DatasetContractError(
        "unknown_data_contract",
        "Unknown training task, geometry, or bridge dataset contract",
    )


def _check_uuid(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str):
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        )
    try:
        parsed = str(uuid.UUID(value))
    except ValueError as exc:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        ) from exc
    if parsed != value:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        )


def _check_timestamp(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 64 or "T" not in value:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        )
    norm = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(norm)
    except ValueError as exc:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        )


def validate_source_lineage(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise DatasetContractError(
            "invalid_source_lineage", "Frozen manifest must be a JSON object"
        )
    declared = False
    if "source_lineage_contract" in record:
        contract = record["source_lineage_contract"]
        if (
            not isinstance(contract, dict)
            or type(contract.get("schema")) is not int
            or contract != SOURCE_LINEAGE_CONTRACT
        ):
            raise DatasetContractError(
                "unknown_source_lineage_contract",
                "Unknown source_lineage_contract",
            )
        declared = True
    members = record.get("samples", [])
    if not isinstance(members, list):
        raise DatasetContractError("invalid_source_lineage", "samples must be a list")
    for member in members:
        if not isinstance(member, dict):
            raise DatasetContractError(
                "invalid_source_lineage", "Each sample must be an object"
            )
        if "source_lineage" not in member:
            if declared and member.get("asset_reference") is not None:
                raise DatasetContractError(
                    "invalid_source_lineage",
                    "Referenced sample is missing declared source_lineage",
                )
            continue
        if not declared:
            raise DatasetContractError(
                "invalid_source_lineage",
                "source_lineage present without source_lineage_contract",
            )
        if member.get("asset_reference") is None:
            raise DatasetContractError(
                "invalid_source_lineage",
                "source_lineage must belong to a referenced asset",
            )
        value = member["source_lineage"]
        if not isinstance(value, dict) or set(value) != _LINEAGE_FIELDS:
            raise DatasetContractError(
                "invalid_source_lineage",
                "Incomplete or unauthorized source_lineage fields",
            )
        for key in ("source_id", "location_id", "scan_run_id"):
            _check_uuid(value[key])
        for key in ("accepted_revision_id", "previous_revision_id"):
            _check_uuid(value[key], nullable=True)
        if not isinstance(value["path_digest"], str) or not _SHA256_RE.fullmatch(
            value["path_digest"]
        ):
            raise DatasetContractError(
                "invalid_source_lineage", "path_digest must be a SHA-256 hex digest"
            )
        for key in ("source_config_revision", "location_config_revision"):
            if type(value[key]) is not int or value[key] < 1:
                raise DatasetContractError(
                    "invalid_source_lineage", f"{key} must be a positive integer"
                )
        rev = value["revision_number"]
        if rev is not None and (type(rev) is not int or rev < 1):
            raise DatasetContractError(
                "invalid_source_lineage", "revision_number must be positive or None"
            )
        for key in ("accepted_at", "scan_completed_at", "content_verified_at"):
            _check_timestamp(value[key])


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) > 1e-14:
        return value
    ax, ay, bx, by, cx, cy = map(Fraction, (*a, *b, *c))
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        and _cross(a, b, c) == 0
    )


def polygon_area(points: list[tuple[float, float]]) -> float:
    terms = [
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(points, points[1:] + points[:1], strict=True)
    ]
    value = math.fsum(terms)
    if abs(value) <= 1e-12:
        value = sum(
            Fraction(a[0]) * Fraction(b[1]) - Fraction(b[0]) * Fraction(a[1])
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
    return float(abs(value) / 2)


def _validate_ring(points: list[tuple[float, float]]) -> None:
    if len(set(points)) != len(points) or polygon_area(points) == 0:
        raise DatasetContractError(
            "invalid_polygon", "Polygon has duplicate vertices or zero area"
        )
    count = len(points)
    for i, b in enumerate(points):
        a, c = points[i - 1], points[(i + 1) % count]
        if _cross(a, b, c) == 0 and (_on_segment(a, b, c) or _on_segment(b, c, a)):
            raise DatasetContractError(
                "invalid_polygon", "Polygon edges cannot fold back or overlap"
            )
    for i, a in enumerate(points):
        b = points[(i + 1) % count]
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            c, d = points[j], points[(j + 1) % count]
            if (
                max(a[0], b[0]) < min(c[0], d[0])
                or max(c[0], d[0]) < min(a[0], b[0])
                or max(a[1], b[1]) < min(c[1], d[1])
                or max(c[1], d[1]) < min(a[1], b[1])
            ):
                continue
            ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
            cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
            if (
                (ab_c * ab_d < 0 and cd_a * cd_b < 0)
                or _on_segment(a, b, c)
                or _on_segment(a, b, d)
                or _on_segment(c, d, a)
                or _on_segment(c, d, b)
            ):
                raise DatasetContractError(
                    "invalid_polygon", "Polygon cannot self-intersect or self-touch"
                )


def _canonical_ring(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    start = points.index(min(points))
    forward = tuple(points[start:] + points[:start])
    return min(forward, (forward[0], *reversed(forward[1:])))


def polygon_points(
    raw: Any,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= MAX_VERTICES + 1:
        raise DatasetContractError("polygon_limit", "Polygon requires 3..512 vertices")
    points: list[tuple[float, float]] = []
    for point in raw:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(
                type(v) not in {int, float} or not 0 <= v <= 1 or not math.isfinite(v)
                for v in point
            )
        ):
            raise DatasetContractError(
                "invalid_polygon", "Polygon coordinates must be finite values in [0, 1]"
            )
        points.append((float(point[0]) if point[0] else 0.0, float(point[1]) if point[1] else 0.0))
    if points[-1] == points[0]:
        points.pop()
    if not 3 <= len(points) <= MAX_VERTICES:
        raise DatasetContractError("polygon_limit", "Polygon requires 3..512 vertices")
    _validate_ring(points)
    native = [
        (
            struct.unpack("f", struct.pack("f", p[0]))[0],
            struct.unpack("f", struct.pack("f", p[1]))[0],
        )
        for p in points
    ]
    try:
        _validate_ring(native)
    except DatasetContractError as exc:
        raise DatasetContractError(
            "native_polygon_collapse",
            "Polygon degenerates after float32 conversion",
        ) from exc
    return _canonical_ring(points), _canonical_ring(native)


def segmentation_labels(
    annotations: list[dict[str, Any]], classes: list[str]
) -> tuple[str, list[dict[str, Any]]]:
    if len(annotations) > MAX_INSTANCES:
        raise DatasetContractError(
            "instance_limit", f"At most {MAX_INSTANCES} instances per image"
        )
    rows: list[tuple[str, str, int, int]] = []
    signatures: set[tuple[int, tuple[tuple[float, float], ...]]] = set()
    ids: set[str] = set()
    vertices = 0
    for annotation in annotations:
        geometry = annotation.get("geometry", {})
        if (
            annotation.get("tool_unit_id") != "region"
            or geometry.get("type") != "polygon"
            or geometry.get("holes", []) != []
            or set(geometry) - {"type", "points", "holes"}
            or annotation.get("class_name") not in classes
        ):
            raise DatasetContractError(
                "unsupported_annotation",
                "Segmentation requires hole-free polygon region annotations in frozen classes",
            )
        identifier, version = annotation.get("id"), annotation.get("version")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in ids
            or type(version) is not int
            or version < 1
        ):
            raise DatasetContractError(
                "invalid_instance_identity",
                "Segmentation instance requires unique id and positive integer version",
            )
        ids.add(identifier)
        points, native = polygon_points(geometry.get("points"))
        vertices += len(points)
        if vertices > MAX_IMAGE_VERTICES:
            raise DatasetContractError(
                "polygon_limit", f"At most {MAX_IMAGE_VERTICES} vertices per image"
            )
        index = classes.index(annotation["class_name"])
        signature = (index, native)
        if signature in signatures:
            raise DatasetContractError(
                "duplicate_instance",
                "Duplicate polygon of same class after float32 quantization",
            )
        signatures.add(signature)
        line = (
            str(index)
            + " "
            + " ".join(format(v, ".17g") for p in points for v in p)
        )
        rows.append((line, identifier, version, index))
    rows.sort()
    text = "\n".join(row[0] for row in rows) + ("\n" if rows else "")
    instances = [
        {
            "line": line_num,
            "annotation_id": identifier,
            "annotation_version": version,
            "class_index": index,
            "line_sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
        for line_num, (value, identifier, version, index) in enumerate(rows, 1)
    ]
    return text, instances


def instance_rle_budget(
    width: int, height: int, instances: int, encoded_bytes: int = 0
) -> None:
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise DatasetContractError(
            "mask_dimensions", "Mask training requires positive integer image dimensions"
        )
    if width * height > INSTANCE_RLE_LIMITS["max_image_pixels"]:
        raise DatasetContractError(
            "mask_pixel_limit", "Image exceeds 16M pixel budget"
        )
    if type(instances) is not int or not 0 <= instances <= MAX_INSTANCES:
        raise DatasetContractError(
            "instance_limit", f"Image exceeds {MAX_INSTANCES} instance budget"
        )
    if width * height * instances > INSTANCE_RLE_LIMITS["max_original_mask_elements"]:
        raise DatasetContractError(
            "mask_decode_limit",
            "Original-resolution mask decode work exceeds 128M elements",
        )
    if (
        type(encoded_bytes) is not int
        or not 0 <= encoded_bytes <= INSTANCE_RLE_LIMITS["max_sample_rle_bytes"]
    ):
        raise DatasetContractError(
            "mask_encoded_limit", "Sample RLE total exceeds 16 MiB budget"
        )


def instance_rle_bytes(rle: Any) -> bytes:
    if not isinstance(rle, dict) or rle.get("encoding") != "coco_rle":
        raise DatasetContractError("invalid_rle", "Only canonical COCO RLE is accepted")
    size, counts = rle.get("size"), rle.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise DatasetContractError("invalid_rle", "RLE size must be [height, width]")
    instance_rle_budget(size[1], size[0], 1)
    if (
        not isinstance(counts, list)
        or not counts
        or len(counts) > 1_000_000
        or any(type(v) is not int or v < 0 for v in counts)
        or sum(counts) != size[0] * size[1]
    ):
        raise DatasetContractError(
            "invalid_rle", "RLE counts must be bounded non-negative runs covering image"
        )
    data = json.dumps(
        {"encoding": "coco_rle", "size": size, "counts": counts},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(data) > 4 * 1024 * 1024:
        raise DatasetContractError(
            "invalid_rle", "Single instance canonical RLE exceeds 4 MiB"
        )
    return data


def instance_rle_evidence(rle: Any) -> dict[str, Any]:
    encoded = instance_rle_bytes(rle)
    height, width = rle["size"]
    hasher = hashlib.sha256(struct.pack(">II", height, width))
    offset, area = 0, 0
    xmin, ymin, xmax, ymax = width, height, -1, -1
    for index, count in enumerate(rle["counts"]):
        value = index % 2
        chunk = bytes([value]) * min(count, 65536)
        remaining = count
        while remaining:
            take = min(remaining, len(chunk))
            hasher.update(chunk[:take])
            remaining -= take
        if value and count:
            first_x, first_y = divmod(offset, height)
            last_x, last_y = divmod(offset + count - 1, height)
            xmin, xmax = min(xmin, first_x), max(xmax, last_x)
            ymin = min(ymin, first_y if first_x == last_x else 0)
            ymax = max(ymax, last_y if first_x == last_x else height - 1)
            area += count
        offset += count
    if not area:
        raise DatasetContractError("empty_instance", "Instance mask cannot be empty")
    return {
        "size": [height, width],
        "area_pixels": area,
        "bbox_xyxy": [xmin, ymin, xmax + 1, ymax + 1],
        "pixel_sha256": hasher.hexdigest(),
        "rle_sha256": hashlib.sha256(encoded).hexdigest(),
        "rle_bytes": len(encoded),
    }


def validate_instance_rle_sample(
    label: Path, sample: dict[str, Any], classes: list[str]
) -> None:
    limit = INSTANCE_RLE_LIMITS["max_sample_label_bytes"]
    if label.stat().st_size > limit:
        raise DatasetContractError(
            "mask_encoded_limit", "Frozen RLE sample exceeds encoded object budget"
        )
    payload = json.loads(label.read_text(encoding="utf-8"))
    width, height = sample["width"], sample["height"]
    mapping = sample.get("instances")
    if (
        payload.get("schema") != 1
        or payload.get("size") != [height, width]
        or not isinstance(mapping, list)
        or not isinstance(payload.get("instances"), list)
        or len(mapping) != len(payload["instances"])
    ):
        raise DatasetContractError(
            "invalid_rle_receipt",
            "Frozen RLE instance receipt missing or dimensions differ",
        )
    instance_rle_budget(width, height, len(mapping))
    seen: set[str] = set()
    encoded = 0
    semantics: list[tuple[int, str]] = []
    for instance, source in zip(payload["instances"], mapping, strict=True):
        identifier = instance.get("annotation_id")
        version = instance.get("annotation_version")
        index = instance.get("class_index")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
            or type(version) is not int
            or version < 1
            or type(index) is not int
            or not 0 <= index < len(classes)
            or instance.get("source_geometry") not in {"polygon", "raster_mask"}
        ):
            raise DatasetContractError(
                "invalid_instance_identity", "Invalid frozen RLE instance identity"
            )
        seen.add(identifier)
        evidence = instance_rle_evidence(instance.get("rle"))
        if evidence["size"] != [height, width] or any(
            instance.get(key) != value for key, value in evidence.items()
        ):
            raise DatasetContractError(
                "rle_evidence_mismatch", "Frozen RLE evidence mismatch"
            )
        if {k: v for k, v in instance.items() if k != "rle"} != source:
            raise DatasetContractError(
                "rle_mapping_mismatch", "Frozen RLE line-to-source mapping changed"
            )
        encoded += evidence["rle_bytes"]
        semantics.append((index, evidence["pixel_sha256"]))
    instance_rle_budget(width, height, len(mapping), encoded)
    if sample.get("semantic_label_sha256") != canonical_digest(sorted(semantics)):
        raise DatasetContractError(
            "rle_semantic_mismatch", "Frozen RLE semantic label digest mismatch"
        )


def validate_dataset(
    dataset_dir: Path,
    *,
    task: str,
    dataset_contract: str,
    expected_classes_digest: str,
    expected_snapshot_digest: str | None = None,
) -> tuple[dict[str, Any], list[str], Path]:
    """Validate frozen dataset directory, returning (manifest, classes, data_yaml)."""
    root = dataset_dir.resolve()
    data_yaml = root / "data.yaml"
    manifest_path = root / "manifest.json"
    classes_path = root / "classes.json"
    if not data_yaml.is_file() or not manifest_path.is_file():
        raise DatasetContractError(
            "missing_dataset_files", "Frozen dataset requires data.yaml and manifest.json"
        )
    yaml_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if (
        not isinstance(yaml_cfg, dict)
        or set(yaml_cfg) != {"train", "val", "names"}
        or yaml_cfg["train"] != "images/train"
        or yaml_cfg["val"] != "images/val"
    ):
        raise DatasetContractError(
            "invalid_data_yaml",
            "Unsupported frozen dataset paths or download configuration in data.yaml",
        )
    names = yaml_cfg["names"]
    if isinstance(names, dict):
        parsed_names = {int(k): v for k, v in names.items()}
        if set(parsed_names) != set(range(len(parsed_names))):
            raise DatasetContractError("invalid_classes", "Invalid dataset class indexes")
        classes = [parsed_names[i] for i in range(len(parsed_names))]
    elif isinstance(names, list):
        classes = list(names)
    else:
        raise DatasetContractError("invalid_classes", "Invalid dataset class names")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_lineage(manifest)

    if (
        expected_snapshot_digest is not None
        and manifest.get("content_digest") is not None
        and manifest.get("content_digest") != expected_snapshot_digest
    ):
        raise DatasetContractError(
            "snapshot_digest_mismatch",
            "Frozen manifest content_digest differs from TrainSpecV1.dataset_snapshot_digest",
        )

    if manifest.get("classes") != classes:
        raise DatasetContractError(
            "class_mismatch", "Frozen manifest classes differ from data.yaml"
        )
    if classes_path.is_file():
        if json.loads(classes_path.read_text(encoding="utf-8")) != classes:
            raise DatasetContractError(
                "class_mismatch", "Frozen classes.json differs from data.yaml"
            )
    if canonical_digest(classes) != expected_classes_digest:
        raise DatasetContractError(
            "classes_digest_mismatch",
            "Frozen dataset classes digest differs from TrainSpecV1.classes_digest",
        )

    contract = data_contract(manifest)
    manifest_task = normalize_task(contract["task"])
    if manifest_task != task:
        raise DatasetContractError(
            "task_mismatch",
            f"Frozen dataset task {manifest_task!r} differs from spec task {task!r}",
        )
    rle = is_instance_rle(contract)
    if rle != is_instance_rle(dataset_contract):
        raise DatasetContractError(
            "contract_mismatch",
            f"Dataset manifest bridge differs from requested dataset_contract {dataset_contract!r}",
        )

    segmentation = manifest_task == SEGMENTATION
    seen: set[str] = set()
    splits: set[str] = set()
    for sample in manifest["samples"]:
        split = sample["split"]
        splits.add(split)
        paths = [sample["image_path"], sample["label_path"]]
        for idx, raw in enumerate(paths):
            rel = PurePosixPath(raw)
            expected_root = "images" if idx == 0 else "labels"
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or len(rel.parts) != 3
                or rel.parts[:2] != (expected_root, split)
                or raw in seen
            ):
                raise DatasetContractError("invalid_sample_path", f"Invalid sample path: {raw}")
            seen.add(raw)
        image, label = root / paths[0], root / paths[1]
        if (
            image.stem != label.stem
            or not image.is_file()
            or not label.is_file()
            or not image.resolve().is_relative_to(root)
            or not label.resolve().is_relative_to(root)
            or digest_file(image) != sample["sha256"]
            or digest_file(label) != sample["label_sha256"]
        ):
            raise DatasetContractError(
                "sample_digest_mismatch",
                "Frozen image or explicit label file missing or changed",
            )
        if rle:
            validate_instance_rle_sample(label, sample, classes)
            continue
        text = label.read_text(encoding="utf-8")
        if segmentation:
            lines = text.splitlines()
            mapping = sample.get("instances")
            if not isinstance(mapping, list) or len(mapping) != len(lines):
                raise DatasetContractError(
                    "missing_instance_receipt",
                    "Frozen segmentation instance receipt missing",
                )
            annotations = []
            for line, source in zip(lines, mapping, strict=True):
                values = line.split()
                if len(values) < 7 or len(values) % 2 != 1:
                    raise DatasetContractError(
                        "invalid_label", "Invalid segmentation label line"
                    )
                cls_idx = int(values[0])
                if not 0 <= cls_idx < len(classes):
                    raise DatasetContractError(
                        "invalid_label", "Invalid segmentation class index"
                    )
                coords = list(map(float, values[1:]))
                annotations.append(
                    {
                        "id": source["annotation_id"],
                        "version": source["annotation_version"],
                        "class_name": classes[cls_idx],
                        "tool_unit_id": "region",
                        "geometry": {
                            "type": "polygon",
                            "points": [
                                coords[i : i + 2] for i in range(0, len(coords), 2)
                            ],
                        },
                    }
                )
            canonical_text, receipt = segmentation_labels(annotations, classes)
            if canonical_text != text or receipt != mapping:
                raise DatasetContractError(
                    "segmentation_receipt_mismatch",
                    "Frozen segmentation label or instance receipt changed",
                )
            continue
        for line in text.splitlines():
            values = line.split()
            if len(values) != 5:
                raise DatasetContractError("invalid_label", "Invalid detection label line")
            cls_idx = int(values[0])
            x, y, w, h = map(float, values[1:])
            if (
                not 0 <= cls_idx < len(classes)
                or not all(math.isfinite(v) for v in (x, y, w, h))
                or min(w, h) <= 0
                or x - w / 2 < -1e-8
                or y - h / 2 < -1e-8
                or x + w / 2 > 1 + 1e-8
                or y + h / 2 > 1 + 1e-8
            ):
                raise DatasetContractError(
                    "invalid_label", "Invalid normalized detection bounding box"
                )

    if splits != {"train", "val"}:
        raise DatasetContractError(
            "missing_splits", "Frozen dataset requires both train and val splits"
        )
    return manifest, classes, data_yaml
