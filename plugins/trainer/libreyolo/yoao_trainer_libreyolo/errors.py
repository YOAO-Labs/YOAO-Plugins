"""Typed domain exceptions mapped to yoao-trainer/v1 failure codes."""

from __future__ import annotations

import sys


class InputMismatchError(ValueError):
    failure_code = "input_mismatch"


class NativeConfigurationMismatch(ValueError):
    failure_code = "config_mismatch"


class NativeRuntimeMismatch(NativeConfigurationMismatch):
    failure_code = "runtime_mismatch"


def classify_failure(exc: BaseException) -> str:
    torch = sys.modules.get("torch")
    if torch is not None:
        cuda_mod = getattr(torch, "cuda", None)
        oom_cls = getattr(cuda_mod, "OutOfMemoryError", None)
        if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
            return "cuda_oom"
    code = getattr(exc, "failure_code", None)
    if isinstance(code, str) and code:
        return code
    return "native_exception"
