"""Official Ultralytics YOLO 8.4.183 trainer plugin for YOAO."""

from __future__ import annotations

from yoao_plugin_sdk.trainer import (
    DatasetContractError,
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
    TrainerError,
)

__version__ = "1.0.0"

FRAMEWORK_ID = "ultralytics"
FRAMEWORK_VERSION = "8.4.183"
LEGACY_ULTRALYTICS_VERSION = "8.4.177"

__all__ = [
    "FRAMEWORK_ID",
    "FRAMEWORK_VERSION",
    "LEGACY_ULTRALYTICS_VERSION",
    "DatasetContractError",
    "InputMismatchError",
    "NativeConfigurationMismatch",
    "NativeRuntimeMismatch",
    "TrainerError",
    "__version__",
]
