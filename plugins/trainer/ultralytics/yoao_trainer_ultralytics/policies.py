"""Frozen policy definitions and canonical SHA-256 digests for Ultralytics 8.4.183."""

from __future__ import annotations

from typing import Any

from yoao_plugin_sdk.trainer import (
    MAX_MEMBERS as INPUT_MEMBERSHIP_MAX_MEMBERS,
    MEMBER_FIELDS as INPUT_MEMBERSHIP_MEMBER_FIELDS,
    POLICY_DIGEST as INPUT_MEMBERSHIP_POLICY_DIGEST,
    SPLITS as INPUT_MEMBERSHIP_SPLITS,
    canonical_json_digest,
)

from . import FRAMEWORK_ID, FRAMEWORK_VERSION, LEGACY_ULTRALYTICS_VERSION

DETECTION = "detect"
SEGMENTATION = "segment"
LEGACY_DETECTION = "image-detection"
LEGACY_SEGMENTATION = "image-instance-segmentation"

MAX_WEIGHTS_BYTES = 2 * 1024**3
MAX_SEMANTIC_CLASS_PIXELS = 64_000_000

INSTANCE_RLE_LIMITS: dict[str, int] = {
    "max_image_pixels": 16_000_000,
    "max_instances": 300,
    "max_sample_rle_bytes": 16 * 1024 * 1024,
    "max_sample_label_bytes": 32 * 1024 * 1024,
    "max_dataset_bytes": 20 * 1024**3,
    "max_package_bytes": 20 * 1024**3,
    "max_original_mask_elements": 128_000_000,
    "max_batch_mask_elements": 64_000_000,
    "workers": 0,
}

PREVIEW_REVISION = 1
PREVIEW_PURPOSE = "augmentation-preview"
PREVIEW_LIMITS: dict[str, int] = {
    "sources": 16,
    "candidate_scan": 512,
    "outputs": 4,
    "source_bytes": 8 * 1024**2,
    "pool_bytes": 64 * 1024**2,
    "source_pixels": 16_000_000,
    "pool_pixels": 32_000_000,
    "source_boxes": 256,
    "pool_boxes": 4096,
    "output_edge": 2560,
    "output_boxes": 4096,
    "output_bytes": 24 * 1024**2,
    "report_bytes": 8 * 1024**2,
    "address_bytes": 16 * 1024**3,
    "cpu_seconds": 30,
    "cpu_hard_seconds": 35,
    "child_seconds": 45,
    "execution_seconds": 60,
    "threads": 1,
    "ttl_seconds": 24 * 60 * 60,
    "put_seconds": 60,
}
PREVIEW_POLICY: dict[str, Any] = {
    "revision": PREVIEW_REVISION,
    "purpose": PREVIEW_PURPOSE,
    "pool_selection": "sha256-seed-task-v1",
    "buffer_policy": "native-sequential-pool-load-v1",
    "limits": PREVIEW_LIMITS,
}
PREVIEW_POLICY_DIGEST = canonical_json_digest(PREVIEW_POLICY)

INPUT_MEMBERSHIP_POLICY: dict[str, Any] = {
    "schema": 1,
    "scope": "native-loader-membership",
    "member_fields": list(INPUT_MEMBERSHIP_MEMBER_FIELDS),
    "splits": list(INPUT_MEMBERSHIP_SPLITS),
    "max_members": INPUT_MEMBERSHIP_MAX_MEMBERS,
}

BUDGET_POLICY: dict[str, Any] = {
    "revision": 1,
    "clock": "monotonic",
    "scope": "spawn-through-confirmed-process-group-exit",
    "allowance": "reserve-current-study-balance-before-enqueue",
    "unknown": "block-admission-and-resume",
    "overrun": "count-in-full",
}
BUDGET_POLICY_DIGEST = canonical_json_digest(BUDGET_POLICY)

LEGACY_INIT_POLICY: dict[str, Any] = {
    "revision": 1,
    "purpose": "trained-model-initialization",
    "framework": FRAMEWORK_ID,
    "framework_version": LEGACY_ULTRALYTICS_VERSION,
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "python": "3.12",
    "cuda": "13.2",
    "task": LEGACY_DETECTION,
    "model_family": "yolo11",
    "class_identity": "exact-native-order",
    "architecture": "same-controlled-variant",
    "transfer": "all-state-tensors",
    "state": "fresh-training",
    "maximum_bytes": MAX_WEIGHTS_BYTES,
}
LEGACY_INIT_POLICY_DIGEST = canonical_json_digest(LEGACY_INIT_POLICY)

LEGACY_SEGMENTATION_INIT_POLICY: dict[str, Any] = {
    **LEGACY_INIT_POLICY,
    "revision": 2,
    "task": SEGMENTATION,
    "qualified_models": ["yolo11n-seg"],
    "source_data_identity": "exact-frozen-contract-and-bridge",
    "evaluation_identity": "raw-report-and-normalized-registration-separately-frozen",
}
LEGACY_SEGMENTATION_INIT_POLICY_DIGEST = canonical_json_digest(
    LEGACY_SEGMENTATION_INIT_POLICY
)

INSTANCE_RLE_POLICY: dict[str, Any] = {
    "bridge": "ultralytics-instance-rle-1",
    "framework": FRAMEWORK_ID,
    "version": FRAMEWORK_VERSION,
    "limits": dict(INSTANCE_RLE_LIMITS),
    "rasterizer": "aap-pixel-center-even-odd-1",
    "transform": "aap-letterbox-nearest-1",
    "loss_mask_resize": "truth-full-input-plane;prototype-bilinear-upsample-align-corners-false",
    "validation_mask_resize": "torch.bilinear-align-corners-false-threshold-0.5-to-prototype-plane",
    "semantic_target": "smallest-covering-instance;first-input-on-equal-area;background-zero",
    "max_semantic_class_pixels": MAX_SEMANTIC_CLASS_PIXELS,
}
INSTANCE_RLE_POLICY_DIGEST = canonical_json_digest(INSTANCE_RLE_POLICY)


def initialization_policy_for(
    family: str,
    task: str,
    framework_version: str = FRAMEWORK_VERSION,
) -> dict[str, Any]:
    segmentation = task == SEGMENTATION
    if framework_version == LEGACY_ULTRALYTICS_VERSION:
        return (
            dict(LEGACY_SEGMENTATION_INIT_POLICY)
            if segmentation
            else dict(LEGACY_INIT_POLICY)
        )
    if framework_version != FRAMEWORK_VERSION:
        raise ValueError(
            f"Unsupported Ultralytics initialization version: {framework_version}"
        )
    base = LEGACY_SEGMENTATION_INIT_POLICY if segmentation else LEGACY_INIT_POLICY
    return {
        **base,
        "revision": 3,
        "framework_version": FRAMEWORK_VERSION,
        "task": task,
        "model_family": family,
        "qualified_models": [
            f"{family}{scale}{'-seg' if segmentation else ''}" for scale in "nsmlx"
        ],
    }


INITIALIZATION_POLICY_CATALOG: dict[str, str] = {
    f"{family}-{task}": canonical_json_digest(
        initialization_policy_for(family, task, FRAMEWORK_VERSION)
    )
    for family in ("yolov8", "yolo11", "yolo26")
    for task in (DETECTION, SEGMENTATION)
}
INITIALIZATION_POLICY_DIGEST = canonical_json_digest(INITIALIZATION_POLICY_CATALOG)


def policy_digests() -> dict[str, str]:
    return {
        "initialization": INITIALIZATION_POLICY_DIGEST,
        "preview": PREVIEW_POLICY_DIGEST,
        "input_membership": INPUT_MEMBERSHIP_POLICY_DIGEST,
        "instance_rle": INSTANCE_RLE_POLICY_DIGEST,
        "budget": BUDGET_POLICY_DIGEST,
    }
