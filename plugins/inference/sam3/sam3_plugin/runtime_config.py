"""Resolve the native plugin's approved data root and CUDA device."""
from __future__ import annotations

import os


def checkpoint_dir() -> str:
    if os.environ.get("AAP_PLUGIN_TOKEN_FILE"):
        return os.path.join(os.environ["AAP_PLUGIN_DATA_DIR"], "checkpoints")
    if root := os.environ.get("AAP_PLUGIN_DATA_DIR"):
        return os.path.join(root, "checkpoints")
    return os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")


def configured_device() -> str:
    device = os.environ["AAP_PLUGIN_DEVICE"] if os.environ.get("AAP_PLUGIN_TOKEN_FILE") else "cuda:0"
    if device != "cuda:0":
        raise RuntimeError("SAM3 requires the plugin CUDA device cuda:0; CPU inference is unsupported")
    return device
