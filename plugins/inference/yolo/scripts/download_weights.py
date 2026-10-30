#!/usr/bin/env python3
"""Download stock closed-set YOLO weights for local plugin use."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model_registry import MODEL_MATRIX, UnsupportedVariantError, iter_supported_combinations, resolve_weight_filename


def _download_one(task: str, series: str, size: str, target_dir: Path) -> bool:
    try:
        filename = resolve_weight_filename(task, series, size)
    except UnsupportedVariantError as exc:
        print(f"skip {task}/{series}/{size}: {exc}", file=sys.stderr)
        return False
    if (target_dir / filename).exists():
        print(f"hit  {filename}")
        return True
    from ultralytics import YOLO
    cwd = os.getcwd()
    try:
        os.chdir(target_dir)
        print(f"pull {filename} ...")
        YOLO(filename)
        return True
    except Exception as exc:
        print(f"fail {filename}: {exc}", file=sys.stderr)
        return False
    finally:
        os.chdir(cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(MODEL_MATRIX), default=None)
    parser.add_argument("--series")
    parser.add_argument("--size")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--target-dir", default=os.environ.get("YOLO_CHECKPOINTS_DIR", "/app/checkpoints"))
    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    combos = iter_supported_combinations()
    if not args.all:
        combos = [c for c in combos if (not args.task or c[0] == args.task) and (not args.series or c[1] == args.series) and (not args.size or c[2] == args.size)]
    ok = sum(_download_one(task, series, size, target_dir) for task, series, size in combos)
    print(f"done: {ok}/{len(combos)} weights ready in {target_dir}")
    return 0 if ok == len(combos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
