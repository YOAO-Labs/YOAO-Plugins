"""CLI entrypoint for the official Ultralytics yoao-trainer/v1 plugin."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import platform
from typing import Any

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    PreviewReportV1,
    PreviewSpecV1,
    TrainerCapabilitiesV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    build_trainer_parser,
    optional_package_version,
    run_trainer_cli,
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

_CLI_PROG = "yoao-trainer-ultralytics"
_CLI_DESCRIPTION = "Official Ultralytics YOLO 8.4.183 trainer plugin (yoao-trainer/v1)."


def collect_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "ultralytics": FRAMEWORK_VERSION,
    }
    for pkg in ("torch", "torchvision", "numpy", "pillow", "opencv-python"):
        pkg_version = optional_package_version(pkg)
        if pkg_version is not None:
            versions[pkg] = pkg_version
    return versions


def build_capabilities() -> TrainerCapabilitiesV1:
    return TrainerCapabilitiesV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        framework_id=FRAMEWORK_ID,
        framework_version=FRAMEWORK_VERSION,
        runtime_versions=collect_runtime_versions(),
        supported_tasks=list(SUPPORTED_TASKS),
        supported_dataset_contracts=list(SUPPORTED_DATASET_CONTRACTS),
        architectures=trainer_architectures(),
        parameter_schema=build_parameter_schema_catalog(),
        evaluator_descriptors=describe_evaluators(),
        policy_digests=policy_digests(),
        serving_compatibility=serving_compatibility_catalog(),
    )


class UltralyticsTrainerPlugin:
    def describe(self) -> TrainerCapabilitiesV1:
        return build_capabilities()

    def validate_config(self, payload: Any) -> ValidateConfigResultV1:
        return validate_config(payload)

    def train(
        self,
        spec: TrainSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> TrainReceiptV1:
        from .train import run_train

        return run_train(spec, output_dir, spec_path=spec_path)

    def preview(
        self,
        spec: PreviewSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> PreviewReportV1:
        from .preview import run_preview

        input_dir = spec_path.parent if spec_path is not None else output_dir
        return run_preview(spec, input_dir, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    arg_list = list(argv) if argv is not None else None
    parser = build_trainer_parser(
        prog=_CLI_PROG,
        description=_CLI_DESCRIPTION,
        supports_preview=True,
    )
    try:
        args = parser.parse_args(arg_list)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    if args.command == "preview":
        from .preview import run_preview_cli

        return run_preview_cli(args.spec, args.output)

    return run_trainer_cli(
        UltralyticsTrainerPlugin(),
        prog=_CLI_PROG,
        description=_CLI_DESCRIPTION,
        supports_preview=True,
        argv=arg_list,
    )


if __name__ == "__main__":
    raise SystemExit(main())
