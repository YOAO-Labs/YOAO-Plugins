"""CLI entrypoint (`yoao-trainer-libreyolo`) implementing yoao-trainer/v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yoao_plugin_sdk.trainer import (
    TrainerCapabilitiesV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    run_trainer_cli,
)

from .catalog import capabilities
from .validation import validate_config


class LibreYOLOTrainerPlugin:
    """yoao-trainer/v1 plugin adapter for LibreYOLO 1.6.0."""

    def describe(self) -> TrainerCapabilitiesV1:
        return capabilities()

    def validate_config(self, payload: Any) -> ValidateConfigResultV1:
        return validate_config(payload)

    def train(
        self,
        spec: TrainSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> TrainReceiptV1:
        from .trainer import LibreYOLOAttemptTrainer

        return LibreYOLOAttemptTrainer().run(spec, output_dir, spec_path=spec_path)


def main(argv: list[str] | None = None) -> int:
    return run_trainer_cli(
        LibreYOLOTrainerPlugin(),
        prog="yoao-trainer-libreyolo",
        description="Official MIT-licensed LibreYOLO9 trainer plugin (yoao-trainer/v1).",
        supports_preview=False,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
