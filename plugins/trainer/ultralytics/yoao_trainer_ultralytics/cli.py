"""CLI entrypoint for the official Ultralytics yoao-trainer/v1 plugin."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    TrainerCapabilitiesV1,
    canonical_json_bytes,
)

from . import FRAMEWORK_ID, FRAMEWORK_VERSION
from .catalog import (
    SUPPORTED_DATASET_CONTRACTS,
    SUPPORTED_TASKS,
    build_parameter_schema_catalog,
    serving_compatibility_catalog,
    trainer_architectures,
)
from .evaluation import describe_evaluators
from .policies import policy_digests
from .validation import validate_config


def _collect_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "ultralytics": FRAMEWORK_VERSION,
    }
    for pkg in ("torch", "torchvision", "numpy", "pillow", "opencv-python"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def build_capabilities() -> TrainerCapabilitiesV1:
    return TrainerCapabilitiesV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        framework_id=FRAMEWORK_ID,
        framework_version=FRAMEWORK_VERSION,
        runtime_versions=_collect_runtime_versions(),
        supported_tasks=list(SUPPORTED_TASKS),
        supported_dataset_contracts=list(SUPPORTED_DATASET_CONTRACTS),
        architectures=trainer_architectures(),
        parameter_schema=build_parameter_schema_catalog(),
        evaluator_descriptors=describe_evaluators(),
        policy_digests=policy_digests(),
        serving_compatibility=serving_compatibility_catalog(),
    )


def _cmd_describe() -> int:
    caps = build_capabilities()
    sys.stdout.write(canonical_json_bytes(caps).decode("utf-8") + "\n")
    return 0


def _cmd_validate_config(spec_path: Path) -> int:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    result = validate_config(raw)
    sys.stdout.write(canonical_json_bytes(result).decode("utf-8") + "\n")
    return 0


def _cmd_train(spec_path: Path, output_dir: Path) -> int:
    from .train import run_train_cli

    return run_train_cli(spec_path, output_dir)


def _cmd_preview(spec_path: Path, output_dir: Path) -> int:
    from .preview import run_preview_cli

    return run_preview_cli(spec_path, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yoao-trainer-ultralytics",
        description="Official Ultralytics YOLO 8.4.183 trainer plugin (yoao-trainer/v1).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="Print TrainerCapabilitiesV1 JSON to stdout")

    validate_p = sub.add_parser(
        "validate-config", help="Validate training configuration against Ultralytics 8.4.183"
    )
    validate_p.add_argument("--spec", type=Path, required=True)

    train_p = sub.add_parser("train", help="Execute a frozen yoao-trainer/v1 training attempt")
    train_p.add_argument("--spec", type=Path, required=True)
    train_p.add_argument("--output", type=Path, required=True)

    preview_p = sub.add_parser("preview", help="Run CPU-only augmentation preview")
    preview_p.add_argument("--spec", type=Path, required=True)
    preview_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "describe":
        return _cmd_describe()
    if args.command == "validate-config":
        return _cmd_validate_config(args.spec)
    if args.command == "train":
        return _cmd_train(args.spec, args.output)
    if args.command == "preview":
        return _cmd_preview(args.spec, args.output)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
