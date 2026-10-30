"""Prepare selected plugin-owned pretrained assets from approved local files.

Catalog visibility grants no download, enablement or redistribution rights.
Platform administrators own object-store publication; this tool uses local bytes only.
"""
from __future__ import annotations

import argparse
from functools import partial
from importlib.resources import as_file
import json
from pathlib import Path
import sys

from training_contract.assets import prepare_asset, read_manifest
from yolo_plugin.catalog import catalog_resource


def _file_chunks(path):
    with path.open("rb") as source:
        yield from iter(lambda: source.read(1024 * 1024), b"")


def prepare_local(source_dir, target_dir, models=None):
    with as_file(catalog_resource()) as manifest:
        references = {item.filename: item for item in read_manifest(manifest)}
        entries = {
            item["base_model"]: references[item["filename"]]
            for item in json.loads(manifest.read_bytes())["assets"]
            if item.get("framework") == "ultralytics"
        }
    if not entries:
        raise ValueError("Plugin catalog contains no Ultralytics assets")
    selected = set(models) if models else set(entries)
    if not selected.issubset(entries):
        raise ValueError("Unknown Ultralytics asset selection")
    for name in sorted(selected):
        reference = entries[name]
        if reference.size_bytes is None:
            raise ValueError("Asset preparation requires size_bytes")
        candidate = source_dir / f"{reference.sha256}.pt"
        if not candidate.exists():
            candidate = source_dir / reference.filename
        prepare_asset(target_dir, reference, partial(_file_chunks, candidate))
        print(f"ready {reference.filename} sha256={reference.sha256}")
    print(f"done: {len(selected)} assets ready in {target_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--model", action="append", help="Canonical base model ID; repeat to select a subset")
    args = parser.parse_args()
    try:
        prepare_local(args.source_dir, args.target_dir, args.model)
    except Exception as exc:
        print(f"Asset preparation failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
