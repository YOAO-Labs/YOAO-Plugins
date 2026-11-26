"""Official Ultralytics YOLO 8.4.183 trainer plugin for YOAO."""

from __future__ import annotations

__version__ = "1.0.0"

FRAMEWORK_ID = "ultralytics"
FRAMEWORK_VERSION = "8.4.183"
LEGACY_ULTRALYTICS_VERSION = "8.4.177"


class TrainerError(ValueError):
    """Base structured failure for the Ultralytics trainer plugin."""

    failure_code = "native_exception"

    def __init__(self, message: str, *, failure_code: str | None = None) -> None:
        super().__init__(message)
        if failure_code is not None:
            self.failure_code = failure_code


class InputMismatchError(TrainerError):
    failure_code = "input_mismatch"


class NativeConfigurationMismatch(TrainerError):
    failure_code = "config_mismatch"


class NativeRuntimeMismatch(NativeConfigurationMismatch):
    failure_code = "runtime_mismatch"


class DatasetContractError(InputMismatchError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message, failure_code="input_mismatch")
        self.code = code


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
