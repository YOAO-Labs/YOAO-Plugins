"""Operator-selected, content-verified qualification weights in plugin data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from aap_backend_runtime import artifact_evidence

INVENTORY_NAME = "qualification-weights.json"


def _contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Weight paths must be relative to the plugin data directory")
    path = Path(name)
    if path.as_posix() != name or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Weight paths must be canonical relative paths")
    target = (root / path).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("Weight must be inside the plugin data directory")
    return target


def prepare_inventory(data_dir: Path, paths: list[str]) -> Path:
    """Freeze operator-selected existing local bytes; never download weights."""
    root = data_dir.resolve(strict=True)
    if not root.is_dir() or not paths or len(set(paths)) != len(paths):
        raise ValueError("Select unique weight paths in an existing data directory")
    records = []
    for name in sorted(paths):
        item = artifact_evidence(_contained(root, name), kind="weight")
        records.append({"path": name, "sha256": item["sha256"], "size_bytes": item["size_bytes"]})
    destination = root / INVENTORY_NAME
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump({"version": 1, "weights": records}, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def read_inventory(data_dir: Path) -> list[dict]:
    """Verify the selected inventory and its current contained bytes fail closed."""
    root = data_dir.resolve(strict=True)
    source = root / INVENTORY_NAME
    if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root):
        raise ValueError("Weight inventory must be inside the plugin data directory")
    payload = json.loads(source.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {"version", "weights"} or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Invalid qualification weight inventory")
    records = payload["weights"]
    if not isinstance(records, list) or not records:
        raise ValueError("Qualification weight inventory is empty")
    seen = set()
    verified = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Invalid qualification weight entry")
        name = record["path"]
        if not isinstance(name, str) or name in seen:
            raise ValueError("Qualification weight paths must be unique")
        seen.add(name)
        observed = artifact_evidence(_contained(root, name), kind="weight")
        if type(record["size_bytes"]) is not int or record["size_bytes"] != observed["size_bytes"] or record["sha256"] != observed["sha256"]:
            raise ValueError("Qualification weight bytes differ from selected inventory")
        verified.append({"path": name, "sha256": observed["sha256"], "size_bytes": observed["size_bytes"]})
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Record approved local qualification weights")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weight", action="append", required=True, help="Path relative to data directory; repeat")
    args = parser.parse_args()
    print(prepare_inventory(args.data_dir, args.weight))


if __name__ == "__main__":
    main()
