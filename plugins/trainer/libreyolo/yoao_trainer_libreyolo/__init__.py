"""Official MIT-licensed LibreYOLO9 trainer plugin (yoao-trainer/v1)."""

from __future__ import annotations

from yoao_plugin_sdk.trainer import SUPERVISOR_OWNED_ARTIFACTS

PLUGIN_ID = "yoao.trainer.libreyolo"
PLUGIN_VERSION = "1.0.0"
FRAMEWORK_ID = "libreyolo"
FRAMEWORK_VERSION = "1.6.0"
SERVING_FORMAT = "libreyolo-native-state-dict-1.0"
MODEL_FAMILY = "libreyolo9"
NATIVE_CHECKPOINT_FAMILY = "yolo9"
SUPPORTED_TASKS = ("detect",)
SUPPORTED_DATASET_CONTRACTS = ("ultralytics-yolo-1", "ultralytics-bbox-1")

__all__ = [
    "FRAMEWORK_ID",
    "FRAMEWORK_VERSION",
    "MODEL_FAMILY",
    "NATIVE_CHECKPOINT_FAMILY",
    "PLUGIN_ID",
    "PLUGIN_VERSION",
    "SERVING_FORMAT",
    "SUPERVISOR_OWNED_ARTIFACTS",
    "SUPPORTED_DATASET_CONTRACTS",
    "SUPPORTED_TASKS",
]
