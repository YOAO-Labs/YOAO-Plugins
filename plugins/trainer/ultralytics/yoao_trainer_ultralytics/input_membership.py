"""Observe native Ultralytics loader membership and write input-membership.json."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

from . import InputMismatchError
from .files import atomic_json, canonical_digest
from .policies import (
    INPUT_MEMBERSHIP_MAX_MEMBERS,
    INPUT_MEMBERSHIP_MEMBER_FIELDS,
    INPUT_MEMBERSHIP_POLICY_DIGEST,
    INPUT_MEMBERSHIP_SPLITS,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
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
    if not isinstance(samples, list) or not 1 <= len(samples) <= INPUT_MEMBERSHIP_MAX_MEMBERS:
        raise ValueError("Frozen input members are missing or exceed the version limit")
    records: dict[str, dict[str, Any]] = {}
    labels: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") not in INPUT_MEMBERSHIP_SPLITS:
            raise ValueError("Invalid frozen input member split")
        split = sample["split"]
        image = _validate_member_path(sample.get("image_path"), "images", split)
        label = _validate_member_path(sample.get("label_path"), "labels", split)
        if (
            not _is_uuid(sample.get("task_id"))
            or not _is_sha256(sample.get("sha256"))
            or not _is_sha256(sample.get("label_sha256"))
            or image.stem != label.stem
            or str(image) in records
            or str(label) in labels
        ):
            raise ValueError("Invalid or duplicate frozen input member")
        for field in ("width", "height", "size_bytes"):
            val = sample.get(field)
            if val is not None and (type(val) is not int or val <= 0):
                raise ValueError("Invalid frozen input member dimensions or size")
        member = {field: sample.get(field) for field in INPUT_MEMBERSHIP_MEMBER_FIELDS}
        records[str(image)] = {
            "task_id": sample["task_id"],
            "member_digest": canonical_digest(member),
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
        for split in INPUT_MEMBERSHIP_SPLITS
    }


def contract_for(manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid frozen input manifest")
    if "samples" not in manifest:
        return None
    members = compact_members(member_records(manifest))
    if not all(members.values()):
        raise ValueError("Frozen training and validation members are required")
    return {
        "schema": 1,
        "policy_digest": INPUT_MEMBERSHIP_POLICY_DIGEST,
        "members_digest": canonical_digest(members),
        "counts": {split: len(rows) for split, rows in members.items()},
    }


def observed_members(
    records: dict[str, dict[str, Any]], split: str, paths: list[str]
) -> list[dict[str, str]]:
    if (
        split not in INPUT_MEMBERSHIP_SPLITS
        or not isinstance(paths, list)
        or len(paths) > INPUT_MEMBERSHIP_MAX_MEMBERS
        or any(not isinstance(path, str) for path in paths)
    ):
        raise ValueError("Invalid native input members")
    expected = [path for path, row in records.items() if row["split"] == split]
    if Counter(paths) != Counter(expected):
        raise ValueError("Native loader members differ from the frozen manifest")
    return compact_members({path: records[path] for path in paths})[split]


class InputMembership:
    """Observe native dataset paths and produce input-membership.json on completion."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        output_dir: Path,
        manifest: dict[str, Any],
        run_id: str,
        attempt_id: str,
        config_digest: str,
        dataset_id: str,
        dataset_sha256: str,
    ) -> None:
        self.root = dataset_root.resolve()
        self.output = output_dir
        self.records = member_records(manifest)
        contract = contract_for(manifest)
        if contract is None:
            raise InputMismatchError("Frozen manifest lacks samples for membership proof")
        self.contract = contract
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.config_digest = str(config_digest)
        self.dataset_id = str(dataset_id)
        self.dataset_sha256 = str(dataset_sha256)
        self.members: dict[str, list[dict[str, str]]] = {}

    def observe(self, split: str, native_paths: list[str | Path]) -> None:
        try:
            paths: list[str] = []
            for value in native_paths:
                if not isinstance(value, (str, Path)):
                    raise ValueError("Native input path is invalid")
                path = Path(value)
                if not path.is_absolute():
                    path = self.root / path
                paths.append(path.resolve().relative_to(self.root).as_posix())
            observed = observed_members(self.records, split, paths)
            if split in self.members and self.members[split] != observed:
                raise ValueError("Native input members changed during execution")
            self.members[split] = observed
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise InputMismatchError(str(exc)) from exc

    def complete(self) -> dict[str, Any]:
        if set(self.members) != set(INPUT_MEMBERSHIP_SPLITS):
            raise InputMismatchError("Native input observation is incomplete")
        members_digest = canonical_digest(self.members)
        if members_digest != self.contract["members_digest"]:
            raise InputMismatchError("Observed members differ from frozen contract")
        proof: dict[str, Any] = {
            "schema": 1,
            "policy_digest": INPUT_MEMBERSHIP_POLICY_DIGEST,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "config_digest": self.config_digest,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "members_digest": members_digest,
            "members": self.members,
            "counts": {split: len(self.members[split]) for split in INPUT_MEMBERSHIP_SPLITS},
        }
        atomic_json(self.output / "input-membership.json", proof)
        return proof
