"""Ultralytics 8.4.183 model catalog, pretrained asset manifest, and parameter schemas."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any

from yoao_plugin_sdk.trainer import ServingCompatibilityV1, TrainerArchitectureV1

from . import FRAMEWORK_ID, FRAMEWORK_VERSION
from .files import canonical_digest
from .policies import DETECTION, SEGMENTATION

SUPPORTED_TASKS = (DETECTION, SEGMENTATION)
SUPPORTED_DATASET_CONTRACTS = (
    "ultralytics-yolo-1",
    "ultralytics-bbox-1",
    "ultralytics-simple-polygon-1",
    "ultralytics-instance-rle-1",
)
DETECT_DATASET_CONTRACTS = frozenset(
    {"ultralytics-yolo-1", "ultralytics-bbox-1", "ultralytics-yolo"}
)
SEGMENT_POLYGON_DATASET_CONTRACTS = frozenset(
    {"ultralytics-yolo-1", "ultralytics-simple-polygon-1", "ultralytics-yolo"}
)
SEGMENT_RLE_DATASET_CONTRACTS = frozenset({"ultralytics-instance-rle-1"})
SEGMENT_DATASET_CONTRACTS = (
    SEGMENT_POLYGON_DATASET_CONTRACTS | SEGMENT_RLE_DATASET_CONTRACTS
)

CANONICAL_PARAMETERS = ("epochs", "batch", "imgsz", "seed", "lr0", "patience")
MODEL_FAMILIES = ("yolov8", "yolo11", "yolo26")
MODEL_SCALES = ("n", "s", "m", "l", "x")

GROUPS: dict[str, str] = {
    "basic": "Basic Training",
    "optimizer": "Optimizer and Learning Rate",
    "augmentation": "Data Augmentation",
    "loss": "Loss Weights",
    "finetuning": "Fine-tuning",
    "evaluation": "Evaluation and Artifacts",
    "performance": "Performance and Reproducibility",
    "platform": "Platform-Managed",
    "other": "Other Tasks and Modes",
}


def _number(
    low: int | float,
    high: int | float,
    *,
    integer: bool = False,
    positive: bool = False,
    multiple: int | None = None,
) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "type": "integer" if integer else "number",
        "maximum": high,
    }
    rule["exclusiveMinimum" if positive else "minimum"] = low
    if multiple is not None:
        rule["multipleOf"] = multiple
    return rule


def _enum(*values: Any) -> dict[str, Any]:
    return {"enum": list(values)}


BOOL_RULE: dict[str, Any] = {"type": "boolean"}
PROB_RULE: dict[str, Any] = _number(0, 1)
SCALE_PAIR_RULE: dict[str, Any] = {
    "type": "array",
    "items": _number(0, 3, positive=True),
    "minItems": 2,
    "maxItems": 2,
    "ordered": True,
}

REVIEWED_FIELDS: dict[str, tuple[str, str, dict[str, Any], Any]] = {
    "epochs": ("Epochs", "basic", _number(1, 1000, integer=True), 50),
    "batch": ("Fixed Batch Size", "basic", _number(1, 64, integer=True), 8),
    "imgsz": (
        "Input Image Size",
        "basic",
        _number(128, 1280, integer=True, multiple=32),
        640,
    ),
    "seed": ("Random Seed", "performance", _number(0, 2147483647, integer=True), 42),
    "patience": ("Early Stopping Patience", "basic", _number(0, 1000, integer=True), 50),
    "pretrained": ("Use Pretrained Weights", "basic", BOOL_RULE, True),
    "lr0": ("Initial Learning Rate", "optimizer", _number(0, 0.1, positive=True), 0.01),
    "lrf": ("Final Learning Rate Factor", "optimizer", PROB_RULE, None),
    "optimizer": ("Optimizer", "optimizer", _enum("SGD", "Adam", "AdamW"), "SGD"),
    "momentum": ("Momentum / Adam beta1", "optimizer", _number(0, 0.9999), None),
    "weight_decay": ("Weight Decay", "optimizer", _number(0, 0.1), None),
    "warmup_epochs": ("Warmup Epochs", "optimizer", _number(0, 100), None),
    "warmup_momentum": ("Warmup Momentum", "optimizer", PROB_RULE, None),
    "warmup_bias_lr": ("Warmup Bias Learning Rate", "optimizer", _number(0, 1), None),
    "cos_lr": ("Cosine LR Schedule", "optimizer", BOOL_RULE, None),
    "nbs": (
        "Nominal Batch Size",
        "optimizer",
        _number(1, 256, integer=True),
        None,
    ),
    "box": ("Box Loss Gain", "loss", _number(0, 100, positive=True), None),
    "cls": ("Classification Loss Gain", "loss", _number(0, 100, positive=True), None),
    "cls_pw": ("Class Positive Weight", "loss", PROB_RULE, None),
    "dfl": ("DFL Loss Gain", "loss", _number(0, 100, positive=True), None),
    "rect": ("Rectangular Batching", "augmentation", BOOL_RULE, None),
    "close_mosaic": (
        "Close Mosaic Epochs",
        "augmentation",
        _number(0, 1000, integer=True),
        None,
    ),
    "multi_scale": ("Multi-Scale Range", "augmentation", _number(0, 0.5), None),
    "hsv_h": ("HSV Hue", "augmentation", PROB_RULE, None),
    "hsv_s": ("HSV Saturation", "augmentation", PROB_RULE, None),
    "hsv_v": ("HSV Value", "augmentation", PROB_RULE, None),
    "degrees": ("Rotation Degrees", "augmentation", _number(0, 180), None),
    "translate": ("Translation Fraction", "augmentation", PROB_RULE, None),
    "scale": (
        "Scale Factor or Range",
        "augmentation",
        {"anyOf": [PROB_RULE, SCALE_PAIR_RULE]},
        None,
    ),
    "shear": ("Shear Degrees", "augmentation", _number(0, 45), None),
    "perspective": ("Perspective Distortion", "augmentation", _number(0, 0.001), None),
    "flipud": ("Vertical Flip Probability", "augmentation", PROB_RULE, None),
    "fliplr": ("Horizontal Flip Probability", "augmentation", PROB_RULE, None),
    "bgr": ("BGR Channel Swap Probability", "augmentation", PROB_RULE, None),
    "mosaic": ("Mosaic Probability", "augmentation", PROB_RULE, None),
    "mixup": ("MixUp Probability", "augmentation", PROB_RULE, None),
    "cutmix": ("CutMix Probability", "augmentation", PROB_RULE, None),
}

MANAGED_FIELDS: frozenset[str] = frozenset(
    "task mode model data device workers project name exist_ok pretrained cls_remap "
    "save save_period resume amp cache deterministic fraction verbose val split conf "
    "iou max_det save_json plots profile cfg nms single_cls classes agnostic_nms".split()
)

UNQUALIFIED_FIELDS: frozenset[str] = frozenset(
    "time compile channels_last quantize distill_model dis freeze".split()
)

OTHER_FIELDS: frozenset[str] = frozenset(
    "overlap_mask mask_ratio dropout pose kobj rle angle dlog dgrad dlam copy_paste "
    "copy_paste_mode auto_augment erasing retina_masks dnn source vid_stride "
    "stream_buffer visualize augment embed show save_frames save_txt save_conf "
    "save_crop show_labels show_conf show_boxes line_width format optimize dynamic "
    "simplify opset workspace tracker".split()
)

FIXED_FIELDS: dict[str, Any] = {
    "task": "detect",
    "mode": "train",
    "device": 0,
    "workers": 2,
    "amp": False,
    "cache": False,
    "deterministic": True,
    "save": True,
    "save_period": -1,
    "plots": False,
    "cls_remap": True,
    "val": True,
    "split": "val",
    "single_cls": False,
    "classes": None,
    "fraction": 1.0,
    "profile": False,
    "save_json": False,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "agnostic_nms": False,
    "nms": None,
    "verbose": True,
    "cfg": None,
}

SEGMENTATION_FIXED_FIELDS: dict[str, Any] = {
    "task": "segment",
    "overlap_mask": False,
    "mask_ratio": 1,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "multi_scale": 0.0,
    "rect": False,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 0,
}

RLE_FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    set(SEGMENTATION_FIXED_FIELDS) | {"workers", "cache", "rect"}
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    scale: str
    task: str
    head: str
    end2end: bool = False
    base_asset_filename: str | None = None
    base_asset_sha256: str | None = None

    @property
    def yaml_filename(self) -> str:
        return f"{self.id}.yaml"


@lru_cache(maxsize=1)
def load_native_inventory() -> dict[str, Any]:
    raw = json.loads(
        files(__package__)
        .joinpath("native_inventory_current.json")
        .read_text(encoding="utf-8")
    )
    return raw[FRAMEWORK_ID]


@lru_cache(maxsize=1)
def load_pretrained_assets() -> dict[str, dict[str, Any]]:
    raw = json.loads(
        files(__package__)
        .joinpath("pretrained-assets.json")
        .read_text(encoding="utf-8")
    )
    assets = raw.get("assets", [])
    by_id: dict[str, dict[str, Any]] = {}
    for entry in assets:
        if entry.get("framework") != FRAMEWORK_ID:
            continue
        model_id = str(entry["base_model"]).lower()
        by_id[model_id] = dict(entry)
    return by_id


@lru_cache(maxsize=1)
def all_model_specs() -> tuple[ModelSpec, ...]:
    assets = load_pretrained_assets()
    specs: list[ModelSpec] = []
    for family in MODEL_FAMILIES:
        for task in SUPPORTED_TASKS:
            for scale in MODEL_SCALES:
                model_id = f"{family}{scale}{'-seg' if task == SEGMENTATION else ''}"
                asset = assets.get(model_id)
                head = (
                    ("Segment26" if family == "yolo26" else "Segment")
                    if task == SEGMENTATION
                    else "Detect"
                )
                specs.append(
                    ModelSpec(
                        id=model_id,
                        family=family,
                        scale=scale,
                        task=task,
                        head=head,
                        end2end=(family == "yolo26"),
                        base_asset_filename=asset["filename"] if asset else None,
                        base_asset_sha256=asset["sha256"] if asset else None,
                    )
                )
    return tuple(specs)


def model_spec(architecture_id: str) -> ModelSpec:
    if not isinstance(architecture_id, str):
        raise ValueError("architecture_id must be a string")
    lookup = {item.id: item for item in all_model_specs()}
    try:
        return lookup[architecture_id.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported Ultralytics architecture: {architecture_id!r}"
        ) from None


def trainer_architectures() -> list[TrainerArchitectureV1]:
    return [
        TrainerArchitectureV1(
            id=spec.id,
            family=spec.family,
            tasks=[spec.task],
            end2end=spec.end2end,
            base_asset_filename=spec.base_asset_filename,
            base_asset_sha256=spec.base_asset_sha256,
        )
        for spec in all_model_specs()
    ]


@lru_cache(maxsize=32)
def parameter_schema_for(architecture_id: str) -> dict[str, Any]:
    spec = model_spec(architecture_id)
    inventory = load_native_inventory()
    segmentation = spec.task == SEGMENTATION
    family = spec.family
    revision = 5

    result: dict[str, Any] = {
        "schema_id": f"{FRAMEWORK_ID}-{inventory['version']}-{family}-{spec.task}-{revision}",
        "framework": FRAMEWORK_ID,
        "framework_version": inventory["version"],
        "adapter_revision": 3,
        "task": spec.task,
        "model_family": family,
        "source": inventory["source"],
        "source_sha256": inventory["source_sha256"],
        "groups": dict(GROUPS),
        "fields": [],
    }
    for key, native in inventory["fields"].items():
        field: dict[str, Any] = {
            "key": key,
            **native,
            "native_default": native["default"],
            "aliases": [],
        }
        if segmentation and key in SEGMENTATION_FIXED_FIELDS:
            field.update(
                label=REVIEWED_FIELDS.get(key, (key,))[0],
                group="platform",
                status="platform-managed",
                rules={},
                default=SEGMENTATION_FIXED_FIELDS[key],
                reason="Fixed mask and transform semantics for segmentation qualification",
            )
        elif key in REVIEWED_FIELDS:
            label, group, rule, default = REVIEWED_FIELDS[key]
            field.update(
                label=label,
                group=group,
                status="editable",
                rules=deepcopy(rule),
                default=native["default"] if default is None else default,
                reason=(
                    f"Controlled {family} {spec.task} parameter bounded by platform contract"
                ),
            )
        elif key in MANAGED_FIELDS:
            field.update(
                label=key,
                group="platform",
                status="platform-managed",
                rules={},
                default=FIXED_FIELDS.get(key, native["default"]),
                reason="Managed by platform dataset, device, evaluation, or artifact contract",
            )
        elif key in UNQUALIFIED_FIELDS:
            field.update(
                label=key,
                group="performance",
                status="unqualified",
                rules={},
                reason="Native option is not qualified for controlled execution",
            )
        elif key in OTHER_FIELDS:
            field.update(
                label=key,
                group="other",
                status="inapplicable",
                rules={},
                reason="Used only by other tasks or modes",
            )
        else:
            raise ValueError(f"Unreviewed native Ultralytics field: {key}")

        if family == "yolo26" and key == "dfl":
            field.update(
                status="inapplicable",
                rules={},
                reason="YOLO26 uses reg_max=1 and does not apply DFL loss gain",
            )
        if family == "yolo26" and key == "nms":
            field["default"] = False
            field["reason"] = "Fixed to end-to-end inference branch (nms=False)"
        if family == "yolo26" and segmentation and key == "batch":
            field["default"] = 1
        result["fields"].append(field)

    result["fields"].append(
        {
            "key": "evaluate_train",
            "label": "Evaluate Train Split Each Epoch",
            "group": "evaluation",
            "status": "editable",
            "rules": dict(BOOL_RULE),
            "default": False,
            "native_default": None,
            "native_type": "bool",
            "defined_in": "platform",
            "aliases": [],
            "reason": (
                "Run native validator on train split without augmentation each epoch; "
                "excluded from best-checkpoint selection"
            ),
        }
    )
    result["schema_digest"] = canonical_digest(result)
    return deepcopy(result)


def build_parameter_schema_catalog() -> dict[str, Any]:
    default_schema = parameter_schema_for("yolo11n")
    per_family_task: dict[str, dict[str, Any]] = {}
    for family in MODEL_FAMILIES:
        for task in SUPPORTED_TASKS:
            sample_id = f"{family}n{'-seg' if task == SEGMENTATION else ''}"
            schema = parameter_schema_for(sample_id)
            per_family_task[f"{family}:{task}"] = {
                "schema_id": schema["schema_id"],
                "schema_digest": schema["schema_digest"],
                "model_family": family,
                "task": task,
                "field_count": len(schema["fields"]),
                "editable_count": sum(
                    1 for field in schema["fields"] if field["status"] == "editable"
                ),
                "fields": schema["fields"],
            }
    return {
        "schema_id": default_schema["schema_id"],
        "schema_digest": default_schema["schema_digest"],
        "source": default_schema["source"],
        "source_sha256": default_schema["source_sha256"],
        "groups": default_schema["groups"],
        "fields": default_schema["fields"],
        "canonical_parameters": list(CANONICAL_PARAMETERS),
        "schemas": per_family_task,
        "constraints": {
            "rle_contract": {
                "dataset_contract": "ultralytics-instance-rle-1",
                "forced_workers": 0,
                "forbidden_fields": sorted(RLE_FORBIDDEN_FIELDS),
            },
            "yolo26": {
                "end2end": True,
                "forced_nms": False,
                "inapplicable_fields": ["dfl"],
                "max_semantic_class_pixels": 64_000_000,
            },
        },
    }


_DETECTION_SOURCE_SHA256: dict[str, str] = {
    "ultralytics.utils.ops": "7b15cd980abbd578a8be91661c2399b2cbdea0207f1e61b0fb74cfac659bed77",
}
_SEGMENTATION_SOURCE_SHA256: dict[str, str] = {
    "ultralytics.models.yolo.segment.predict": "66371a017252aa2e8606ba13dbc6c76307088d69159a5f46970138a9412846c5",
    "ultralytics.utils.ops": "7b15cd980abbd578a8be91661c2399b2cbdea0207f1e61b0fb74cfac659bed77",
}


def _runtime_qualification(*, task: str, end2end: bool) -> dict[str, Any]:
    if task == SEGMENTATION:
        return {
            "revision": 2,
            "ultralytics": FRAMEWORK_VERSION,
            "torch": "2.14.1+cu132",
            "torchvision": "0.29.1+cu132",
            "retina_masks": True,
            "mask_processing": "ultralytics.utils.ops.process_mask_native",
            "mask_threshold": 0.0,
            "decoder": "aap-coco-rle-column-major-1",
            "source_sha256": dict(_SEGMENTATION_SOURCE_SHA256),
            "end2end": end2end,
            "nms": False if end2end else None,
        }
    return {
        "revision": 2,
        "ultralytics": FRAMEWORK_VERSION,
        "torch": "2.14.1+cu132",
        "torchvision": "0.29.1+cu132",
        "source_sha256": dict(_DETECTION_SOURCE_SHA256),
        "end2end": end2end,
        "nms": False if end2end else None,
    }


def serving_compatibility_catalog() -> list[ServingCompatibilityV1]:
    model_sizes = {spec.id: spec.scale for spec in all_model_specs()}
    entries: list[ServingCompatibilityV1] = []
    for families, end2end in ((["yolov8", "yolo11"], False), (["yolo26"], True)):
        for task in SUPPORTED_TASKS:
            runtime_qual = _runtime_qualification(task=task, end2end=end2end)
            training_runtime: dict[str, Any] = {
                "framework": FRAMEWORK_ID,
                "framework_version": FRAMEWORK_VERSION,
                "task": task,
                "model_families": list(families),
                "model_sizes": {
                    spec.id: model_sizes[spec.id]
                    for spec in all_model_specs()
                    if spec.family in families and spec.task == task
                },
                "end2end": end2end,
                "nms": False if end2end else None,
                "runtime_qualification": runtime_qual,
            }
            if task == SEGMENTATION:
                training_runtime["output_contract"] = "aap-instance-rle-1"
            entries.append(
                ServingCompatibilityV1(
                    framework=FRAMEWORK_ID,
                    format=None,
                    training_runtime=training_runtime,
                    compatible_serving_runtimes=[
                        {
                            "framework": FRAMEWORK_ID,
                            "format": None,
                            "min_version": FRAMEWORK_VERSION,
                            "end2end": end2end,
                            "nms": False if end2end else None,
                        }
                    ],
                )
            )
    return entries


def serving_compatibility_for(
    spec: ModelSpec,
    *,
    runtime_versions: dict[str, str] | None = None,
) -> ServingCompatibilityV1:
    training_runtime: dict[str, Any] = {
        "framework": FRAMEWORK_ID,
        "framework_version": FRAMEWORK_VERSION,
        "task": spec.task,
        "model_family": spec.family,
        "model_size": spec.scale,
        "end2end": spec.end2end,
        "nms": False if spec.end2end else None,
        "runtime_qualification": _runtime_qualification(
            task=spec.task, end2end=spec.end2end
        ),
    }
    if spec.task == SEGMENTATION:
        training_runtime["output_contract"] = "aap-instance-rle-1"
    if runtime_versions:
        training_runtime["runtime_versions"] = dict(runtime_versions)
    return ServingCompatibilityV1(
        framework=FRAMEWORK_ID,
        format=None,
        training_runtime=training_runtime,
        compatible_serving_runtimes=[
            {
                "framework": FRAMEWORK_ID,
                "format": None,
                "min_version": FRAMEWORK_VERSION,
                "end2end": spec.end2end,
                "nms": False if spec.end2end else None,
            }
        ],
    )
