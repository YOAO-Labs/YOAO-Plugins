"""Observe native LibreYOLO dataloader membership and emit input-membership.json."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

from yoao_plugin_sdk.trainer import atomic_write_json, canonical_json_digest

from .errors import InputMismatchError

MAX_MEMBERS = 20000
SPLITS = ("train", "val")
MEMBER_FIELDS = (
    "task_id",
    "split",
    "image_path",
    "label_path",
    "sha256",
    "label_sha256",
    "width",
    "height",
    "size_bytes",
)
POLICY_DIGEST = canonical_json_digest(
    {
        "schema": 1,
        "scope": "native-loader-membership",
        "member_fields": list(MEMBER_FIELDS),
        "splits": list(SPLITS),
        "max_members": MAX_MEMBERS,
    }
)
_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _is_uuid(value: Any) -> bool:
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _validate_member_path(value: Any, kind: str, split: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or len(value) > 1000
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("Invalid frozen input member path")
    path = PurePosixPath(value)
    if (
        path.as_posix() != value
        or len(path.parts) != 3
        or path.parts[:2] != (kind, split)
        or ".." in path.parts
    ):
        raise ValueError("Invalid frozen input member path")
    return path


def member_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid frozen input manifest")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_MEMBERS:
        raise ValueError("Frozen input members are missing or exceed the version limit")
    records: dict[str, dict[str, Any]] = {}
    labels: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") not in SPLITS:
            raise ValueError("Invalid frozen input member split")
        split = sample["split"]
        image = _validate_member_path(sample.get("image_path"), "images", split)
        label = _validate_member_path(sample.get("label_path"), "labels", split)
        if (
            not _is_uuid(sample.get("task_id"))
            or not _is_hash(sample.get("sha256"))
            or not _is_hash(sample.get("label_sha256"))
            or image.stem != label.stem
            or str(image) in records
            or str(label) in labels
        ):
            raise ValueError("Invalid or duplicate frozen input member")
        for field in ("width", "height", "size_bytes"):
            value = sample.get(field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Invalid frozen input member dimensions or size")
        member = {field: sample.get(field) for field in MEMBER_FIELDS}
        records[str(image)] = {
            "task_id": sample["task_id"],
            "member_digest": canonical_json_digest(member),
            "split": split,
        }
        labels.add(str(label))
    return records


def compact_members(records: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    return {
        split: sorted(
            (
                {"task_id": row["task_id"], "member_digest": row["member_digest"]}
                for row in records.values()
                if row["split"] == split
            ),
            key=lambda row: (row["member_digest"], row["task_id"]),
        )
        for split in SPLITS
    }


def observed_members(
    records: dict[str, dict[str, Any]], split: str, paths: list[str]
) -> list[dict[str, str]]:
    if (
        split not in SPLITS
        or not isinstance(paths, list)
        or len(paths) > MAX_MEMBERS
        or any(not isinstance(path, str) for path in paths)
    ):
        raise ValueError("Invalid native input members")
    expected = [path for path, row in records.items() if row["split"] == split]
    if Counter(paths) != Counter(expected):
        raise ValueError("Native loader members differ from the frozen manifest")
    return compact_members({path: records[path] for path in paths})[split]


def has_member_identities(manifest: dict[str, Any]) -> bool:
    samples = manifest.get("samples")
    return isinstance(samples, list) and bool(samples) and all(
        isinstance(sample, dict) and _is_uuid(sample.get("task_id"))
        for sample in samples
    )


class InputMembership:
    """Observes exact train/val image paths admitted by LibreYOLO loaders."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        output_dir: Path,
        manifest: dict[str, Any],
        run_id: str,
        attempt_id: str,
        config_digest: str,
        dataset_snapshot_digest: str,
        dataset_id: str | None = None,
        dataset_sha256: str | None = None,
    ) -> None:
        self.root = dataset_dir.resolve()
        self.output = output_dir
        self.manifest = manifest
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.config_digest = str(config_digest)
        self.dataset_snapshot_digest = str(dataset_snapshot_digest)
        self.dataset_id = dataset_id
        self.dataset_sha256 = dataset_sha256
        self.has_identities = has_member_identities(manifest)
        self.members: dict[str, Any] = {}
        try:
            if self.has_identities:
                self.records: dict[str, dict[str, Any]] = member_records(manifest)
                self.expected_paths: dict[str, list[str]] = {
                    split: [
                        path for path, row in self.records.items() if row["split"] == split
                    ]
                    for split in SPLITS
                }
            else:
                self.records = {}
                samples = manifest.get("samples") or []
                self.expected_paths = {
                    split: [
                        str(sample["image_path"])
                        for sample in samples
                        if isinstance(sample, dict) and sample.get("split") == split
                    ]
                    for split in SPLITS
                }
        except (ValueError, KeyError, TypeError) as exc:
            raise InputMismatchError(str(exc)) from exc

    def _normalize_paths(self, native_paths: Any) -> list[str]:
        if not isinstance(native_paths, (list, tuple)):
            raise ValueError("Native input paths must be a sequence")
        paths: list[str] = []
        for value in native_paths:
            if not isinstance(value, (str, Path)):
                raise ValueError("Native input path is invalid")
            path = Path(value)
            if not path.is_absolute():
                path = self.root / path
            paths.append(path.resolve().relative_to(self.root).as_posix())
        return paths

    def observe(self, split: str, native_paths: Any) -> None:
        try:
            if split not in SPLITS:
                raise ValueError(f"Unknown dataset split: {split!r}")
            paths = self._normalize_paths(native_paths)
            if self.has_identities:
                observed: Any = observed_members(self.records, split, paths)
            else:
                if Counter(paths) != Counter(self.expected_paths[split]):
                    raise ValueError("Native loader members differ from the frozen manifest")
                observed = sorted(paths)
            if split in self.members and self.members[split] != observed:
                raise ValueError("Native input members changed during execution")
            self.members[split] = observed
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise InputMismatchError(str(exc)) from exc

    def complete(self) -> Path | None:
        if set(self.members) != set(SPLITS):
            raise InputMismatchError("Native input observation is incomplete")
        if not self.has_identities:
            return None
        dataset_id = self.dataset_id or self.manifest.get("dataset_id")
        if not _is_uuid(dataset_id):
            dataset_id = self.dataset_snapshot_digest
        dataset_sha256 = self.dataset_sha256 or self.manifest.get("dataset_sha256")
        if not _is_hash(dataset_sha256):
            dataset_sha256 = self.dataset_snapshot_digest
        proof = {
            "schema": 1,
            "policy_digest": POLICY_DIGEST,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "config_digest": self.config_digest,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "members_digest": canonical_json_digest(self.members),
            "members": self.members,
            "counts": {split: len(self.members[split]) for split in SPLITS},
        }
        target = self.output / "input-membership.json"
        atomic_write_json(target, proof)
        return target
