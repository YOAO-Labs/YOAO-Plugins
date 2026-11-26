"""Fixed-batch, telemetry-disabled, and offline-font boundaries for Ultralytics 8.4.183."""

from __future__ import annotations

import os
from pathlib import Path

from . import (
    FRAMEWORK_VERSION,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
)

_QUALIFIED_VERSION = FRAMEWORK_VERSION
_FIRST_EPOCH_OOM_RETRY_LIMIT = 3
_FONT_PATHS: dict[str, tuple[str, str]] = {
    "Arial.ttf": (
        "YOAO_ULTRALYTICS_ASCII_FONT",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    "Arial.Unicode.ttf": (
        "YOAO_ULTRALYTICS_UNICODE_FONT",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ),
}


def require_qualified_runtime() -> str:
    from ultralytics import __version__

    if __version__ != _QUALIFIED_VERSION:
        raise NativeRuntimeMismatch(
            f"Ultralytics trainer plugin requires {_QUALIFIED_VERSION}, got {__version__}"
        )
    return __version__


def disable_telemetry() -> None:
    require_qualified_runtime()
    from ultralytics import settings

    settings.update(
        {"sync": False, "wandb": False, "mlflow": False, "tensorboard": False}
    )


def configure_offline_fonts() -> dict[str, Path]:
    """Patch all three Ultralytics font-lookup call sites to use local installed fonts."""
    require_qualified_runtime()
    from ultralytics.data import utils as data_utils
    from ultralytics.utils import checks, plotting

    fonts: dict[str, Path] = {}
    for name, (variable, default) in _FONT_PATHS.items():
        path = Path(os.environ.get(variable, default)).expanduser()
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.R_OK):
            raise NativeConfigurationMismatch(
                f"{variable} must point to an installed local font file: {path}"
            )
        fonts[name] = path

    def local_font(font: str = "Arial.ttf") -> Path:
        try:
            return fonts[font]
        except KeyError as exc:
            raise NativeConfigurationMismatch(
                f"Unsupported Ultralytics training font: {font}"
            ) from exc

    checks.check_font = data_utils.check_font = plotting.check_font = local_font
    return fonts


class UltralyticsAttemptTrainer:
    """Mixin enforcing fixed batch size, attempt output ownership, and no OOM batch halving."""

    def check_resume(self, overrides: dict) -> None:
        require_qualified_runtime()
        frozen_batch = overrides.get("batch")
        if type(frozen_batch) is not int or frozen_batch < 1:
            raise NativeConfigurationMismatch(
                "Ultralytics requires a fixed integer batch"
            )
        super().check_resume(overrides)
        if self.args.batch != frozen_batch:
            raise NativeConfigurationMismatch("Ultralytics changed the frozen batch")
        if self.resume:
            for field in ("project", "name", "data", "exist_ok"):
                setattr(self.args, field, overrides[field])
            self.args.save_dir = str(Path(overrides["project"]) / overrides["name"])

    def run_callbacks(self, event: str) -> None:
        if event == "on_train_epoch_start" and self.epoch == self.start_epoch:
            self._oom_retries = _FIRST_EPOCH_OOM_RETRY_LIMIT
        super().run_callbacks(event)
