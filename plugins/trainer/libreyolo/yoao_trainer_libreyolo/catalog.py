"""LibreYOLO 1.6.0 architectures, parameter catalog, and capabilities."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import importlib.metadata
from importlib.resources import files
import json
import platform
from typing import Any

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    ServingCompatibilityV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    canonical_json_digest,
)

from . import (
    FRAMEWORK_ID,
    FRAMEWORK_VERSION,
    MODEL_FAMILY,
    SERVING_FORMAT,
    SUPPORTED_DATASET_CONTRACTS,
    SUPPORTED_TASKS,
)

SCHEMA_ID = "libreyolo-1.6.0-yolo9-detect-3"

GROUPS: dict[str, str] = {
    "basic": "基础训练",
    "optimizer": "优化器与学习率",
    "augmentation": "数据增强",
    "loss": "损失函数",
    "finetuning": "微调",
    "evaluation": "验证与产物",
    "performance": "性能与复现",
    "platform": "平台与执行",
    "other": "其他任务与模式",
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


BOOL: dict[str, Any] = {"type": "boolean"}
PROB: dict[str, Any] = _number(0, 1)
SCALE_PAIR: dict[str, Any] = {
    "type": "array",
    "items": _number(0, 3, positive=True),
    "minItems": 2,
    "maxItems": 2,
    "ordered": True,
}

REVIEWED_FIELDS: dict[str, tuple[str, str, dict[str, Any], Any]] = {
    "epochs": ("训练轮次", "basic", _number(1, 1000, integer=True), 50),
    "batch": ("固定批大小", "basic", _number(1, 64, integer=True), 8),
    "imgsz": ("输入尺寸", "basic", _number(128, 1280, integer=True, multiple=32), 640),
    "seed": ("随机种子", "performance", _number(0, 2147483647, integer=True), 42),
    "patience": ("早停耐心轮次", "basic", _number(0, 1000, integer=True), 50),
    "lr0": ("初始学习率", "optimizer", _number(0, 0.1, positive=True), 0.01),
    "momentum": ("动量 / Adam β1", "optimizer", _number(0, 0.9999), None),
    "weight_decay": ("权重衰减", "optimizer", _number(0, 0.1), None),
    "warmup_momentum": ("预热动量", "optimizer", PROB, None),
    "nbs": (
        "标称批大小（梯度累积）",
        "optimizer",
        {"anyOf": [{"type": "null"}, _number(1, 256, integer=True)]},
        None,
    ),
    "degrees": ("随机旋转角度", "augmentation", _number(0, 180), None),
    "translate": ("随机平移比例", "augmentation", PROB, None),
    "shear": ("随机剪切角度", "augmentation", _number(0, 45), None),
    "perspective": ("透视幅度", "augmentation", _number(0, 0.001), None),
    "flipud": ("上下翻转概率", "augmentation", PROB, None),
    "optimizer": ("优化器", "optimizer", _enum("sgd", "adam", "adamw"), "sgd"),
    "nesterov": ("Nesterov 动量", "optimizer", BOOL, None),
    "scheduler": ("学习率计划", "optimizer", _enum("linear", "cos", "warmcos"), None),
    "warmup_epochs": ("预热轮次", "optimizer", _number(0, 100, integer=True), None),
    "warmup_lr_start": ("预热起始学习率", "optimizer", _number(0, 0.1), None),
    "no_aug_epochs": (
        "末尾关闭增强的轮次",
        "augmentation",
        _number(0, 1000, integer=True),
        None,
    ),
    "min_lr_ratio": ("最低学习率比例", "optimizer", PROB, None),
    "mosaic_prob": ("Mosaic 概率", "augmentation", PROB, None),
    "mixup_prob": ("MixUp 概率", "augmentation", PROB, None),
    "hsv_prob": ("HSV 扰动概率", "augmentation", PROB, None),
    "flip_prob": ("左右翻转概率", "augmentation", PROB, None),
    "mosaic_scale": ("Mosaic 缩放区间", "augmentation", SCALE_PAIR, None),
    "ema": ("指数移动平均权重", "optimizer", BOOL, None),
    "ema_decay": ("EMA 衰减系数", "optimizer", _number(0, 1, positive=True), None),
    "min_samples": (
        "每轮数据集样本数下限",
        "basic",
        _number(0, 20000, integer=True),
        None,
    ),
    "class_balanced": ("按类别平衡采样", "basic", BOOL, None),
    "max_labels": (
        "每张图片最大训练框数",
        "basic",
        _number(1, 1000, integer=True),
        None,
    ),
    "aux_weight": ("辅助检测头损失权重", "loss", _number(0, 10), None),
    "letterbox_pad": (
        "Letterbox 填充位置",
        "augmentation",
        _enum("topleft", "center"),
        "center",
    ),
    "val_loss": ("计算验证损失", "evaluation", BOOL, None),
}

MANAGED_FIELDS: frozenset[str] = frozenset(
    "size num_classes data data_dir device sync_bn amp amp_dtype project name "
    "exist_ok save_period eval_interval max_det eval_max_det faster_coco_eval "
    "save_plots plot_samples workers cache resume log_interval "
    "allow_download_scripts single_cls classes".split()
)

UNQUALIFIED_FIELDS: frozenset[str] = frozenset(
    "cuda_graph compile distill_model dis distill_loss_type distill_mask_ratio "
    "distill_tau distill_normalize average_best export_check precise_bn profile "
    "profile_then_stop profile_warmup profile_steps profile_trace profile_open "
    "freeze".split()
)

INAPPLICABLE_FIELDS: frozenset[str] = frozenset(
    "mask_downsample_ratio copy_paste copy_paste_mode rot90 auto_augment "
    "erasing cutmix scale crop_pct cls_pw class_weights lora mixup_scale "
    "mixup".split()
)

FIXED_VALUES: dict[str, Any] = {
    "device": "cuda:0",
    "workers": 2,
    "amp": False,
    "cache": False,
    "amp_dtype": "float16",
    "sync_bn": False,
    "eval_interval": 1,
    "save_period": 0,
    "save_plots": False,
    "max_det": 300,
    "eval_max_det": 100,
    "faster_coco_eval": False,
    "plot_samples": 8,
    "log_interval": 10,
    "allow_download_scripts": False,
    "single_cls": False,
    "classes": None,
}

LABELS: dict[str, str] = {
    "size": "模型规格",
    "num_classes": "类别数量",
    "data": "固定数据集",
    "data_dir": "数据目录",
    "device": "执行设备",
    "sync_bn": "同步批归一化",
    "amp": "混合精度",
    "amp_dtype": "混合精度类型",
    "project": "输出根目录",
    "name": "输出子目录",
    "exist_ok": "允许已有目录",
    "save_period": "周期保存",
    "eval_interval": "验证间隔",
    "max_det": "最大检测数",
    "eval_max_det": "COCO 评估检测数",
    "faster_coco_eval": "加速 COCO 评估",
    "save_plots": "原生绘图",
    "plot_samples": "绘图样本数",
    "workers": "数据加载进程数",
    "cache": "数据缓存",
    "resume": "断点恢复",
    "log_interval": "日志间隔",
    "allow_download_scripts": "数据下载脚本",
    "single_cls": "单类别模式",
    "classes": "类别过滤",
    "compile": "编译模型",
    "cuda_graph": "CUDA Graph",
    "distill_model": "蒸馏教师模型",
    "dis": "蒸馏损失权重",
    "mixup_scale": "MixUp 缩放区间",
    "lora": "LoRA",
    "cls_pw": "类别正样本加权",
    "class_weights": "类别加权",
    "freeze": "冻结层或模块（需验证具体命中）",
}


@lru_cache(maxsize=1)
def controlled_assets() -> tuple[dict[str, Any], ...]:
    raw = json.loads(
        files("yoao_trainer_libreyolo")
        .joinpath("controlled-model-assets.json")
        .read_text(encoding="utf-8")
    )
    return tuple(raw["assets"])


@lru_cache(maxsize=1)
def native_inventory() -> dict[str, Any]:
    return json.loads(
        files("yoao_trainer_libreyolo")
        .joinpath("native-inventory.json")
        .read_text(encoding="utf-8")
    )


def architectures() -> list[TrainerArchitectureV1]:
    return [
        TrainerArchitectureV1(
            id=asset["base_model"],
            family=MODEL_FAMILY,
            tasks=[asset["task"]],
            end2end=False,
            base_asset_filename=asset["filename"],
            base_asset_sha256=asset["sha256"],
        )
        for asset in controlled_assets()
    ]


def architecture_by_id(architecture_id: str) -> TrainerArchitectureV1 | None:
    if not isinstance(architecture_id, str):
        return None
    lowered = architecture_id.strip().lower()
    for arch in architectures():
        if arch.id.lower() == lowered:
            return arch
    return None


def architecture_size(architecture_id: str) -> str:
    arch = architecture_by_id(architecture_id)
    if arch is None:
        raise ValueError(f"Unsupported LibreYOLO architecture: {architecture_id!r}")
    return arch.id[-1].lower()


@lru_cache(maxsize=1)
def _cached_parameter_schema() -> dict[str, Any]:
    inventory = native_inventory()
    if inventory["version"] != FRAMEWORK_VERSION:
        raise ValueError("Native inventory version does not match pinned LibreYOLO")
    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "framework": FRAMEWORK_ID,
        "framework_version": FRAMEWORK_VERSION,
        "adapter_revision": 3,
        "task": "detect",
        "model_family": MODEL_FAMILY,
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
        if key in REVIEWED_FIELDS:
            label, group, rule, default = REVIEWED_FIELDS[key]
            field.update(
                label=label,
                group=group,
                status="editable",
                editable=True,
                rules=deepcopy(rule),
                default=native["default"] if default is None else default,
                reason="已验证 LibreYOLO9 目标检测训练路径；取值范围按平台资格收敛",
            )
            if "type" in rule:
                field["type"] = rule["type"]
            elif "enum" in rule and all(isinstance(v, str) for v in rule["enum"]):
                field["type"] = "string"
        elif key in MANAGED_FIELDS:
            field.update(
                label=LABELS.get(key, key),
                group="platform",
                status="platform-managed",
                editable=False,
                rules={},
                default=FIXED_VALUES.get(key, native["default"]),
                reason="由平台固定数据、设备、评测或产物协议管理，不能提交覆盖",
            )
        elif key in UNQUALIFIED_FIELDS:
            field.update(
                label=LABELS.get(key, key),
                group="performance",
                status="unqualified",
                editable=False,
                rules={},
                reason="原生功能需要额外执行、恢复或产物验证；当前不可编辑",
            )
        elif key in INAPPLICABLE_FIELDS:
            field.update(
                label=LABELS.get(key, key),
                group="other",
                status="inapplicable",
                editable=False,
                rules={},
                reason="其他任务/模式专用，或当前检测路径不读取该值",
            )
        else:
            raise ValueError(f"Unreviewed native LibreYOLO field: {key}")
        if key == "mixup_prob":
            field["aliases"] = ["mixup"]
        result["fields"].append(field)
    result["schema_digest"] = canonical_json_digest(result)
    return result


def parameter_schema() -> dict[str, Any]:
    return deepcopy(_cached_parameter_schema())


def serving_compatibility() -> ServingCompatibilityV1:
    return ServingCompatibilityV1(
        framework=FRAMEWORK_ID,
        format=SERVING_FORMAT,
        training_runtime={
            "framework": FRAMEWORK_ID,
            "framework_version": FRAMEWORK_VERSION,
            "task": "detect",
            "model_families": [MODEL_FAMILY, "yolo9"],
            "serving_model_family": "yolo9",
            "model_sizes": {
                asset["base_model"].lower(): asset["base_model"][-1].lower()
                for asset in controlled_assets()
            },
            "runtime_qualification": {
                "revision": 1,
                "libreyolo": FRAMEWORK_VERSION,
                "torch": "2.14.1+cu132",
                "torchvision": "0.29.1+cu132",
                "device": "cpu",
            },
        },
        compatible_serving_runtimes=[
            {
                "framework": FRAMEWORK_ID,
                "format": SERVING_FORMAT,
                "min_version": FRAMEWORK_VERSION,
            }
        ],
    )


def _optional_dist_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "libreyolo": _optional_dist_version("libreyolo") or FRAMEWORK_VERSION,
    }
    for pkg in ("torch", "torchvision", "pycocotools"):
        detected = _optional_dist_version(pkg)
        if detected is not None:
            versions[pkg] = detected
    return versions


def capabilities() -> TrainerCapabilitiesV1:
    from .checkpoint import CHECKPOINT_RESUME_POLICY_DIGEST
    from .evaluation import evaluator_descriptor_summary
    from .membership import POLICY_DIGEST as INPUT_MEMBERSHIP_POLICY_DIGEST

    return TrainerCapabilitiesV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        framework_id=FRAMEWORK_ID,
        framework_version=FRAMEWORK_VERSION,
        runtime_versions=runtime_versions(),
        supported_tasks=list(SUPPORTED_TASKS),
        supported_dataset_contracts=list(SUPPORTED_DATASET_CONTRACTS),
        architectures=architectures(),
        parameter_schema=parameter_schema(),
        evaluator_descriptors={"detect": evaluator_descriptor_summary()},
        policy_digests={
            "input_membership": INPUT_MEMBERSHIP_POLICY_DIGEST,
            "checkpoint_resume": CHECKPOINT_RESUME_POLICY_DIGEST,
        },
        serving_compatibility=[serving_compatibility()],
    )
