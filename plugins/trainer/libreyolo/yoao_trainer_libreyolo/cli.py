"""CLI entrypoint (`yoao-trainer-libreyolo`) implementing yoao-trainer/v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from yoao_plugin_sdk.trainer import canonical_json_bytes

from .catalog import capabilities
from .trainer import execute_train
from .validation import validate_config


def _print_canonical(payload: object) -> None:
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8") + "\n")  # type: ignore[arg-type]
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yoao-trainer-libreyolo",
        description="Official MIT-licensed LibreYOLO9 trainer plugin (yoao-trainer/v1).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "describe",
        help="Emit TrainerCapabilitiesV1 JSON without GPU or network access.",
    )

    validate_p = sub.add_parser(
        "validate-config",
        help="Validate a LibreYOLO 1.6.0 config spec and emit ValidateConfigResultV1 JSON.",
    )
    validate_p.add_argument("--spec", type=Path, required=True)

    train_p = sub.add_parser(
        "train",
        help="Execute a deterministic LibreYOLO 1.6.0 training attempt.",
    )
    train_p.add_argument("--spec", type=Path, required=True)
    train_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "describe":
        _print_canonical(capabilities())
        return 0

    if args.command == "validate-config":
        try:
            payload = json.loads(args.spec.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Failed to read validation spec: {exc}", file=sys.stderr)
            return 2
        result = validate_config(payload)
        _print_canonical(result)
        return 0

    if args.command == "train":
        return execute_train(args.spec, args.output)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
