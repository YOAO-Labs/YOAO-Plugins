"""Exact native architecture topology and tensor-shape identity checks."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .catalog import ModelSpec, model_spec


def validate_architecture_yaml(
    actual: Any, spec: ModelSpec, canonical: dict[str, Any]
) -> None:
    if not isinstance(actual, dict) or any(
        actual.get(key, 3) != 3 for key in ("ch", "channels")
    ):
        raise ValueError("Native input-channel identity differs")
    expected_graph = {key: canonical[key] for key in ("backbone", "head")}
    if actual.get("scale") is not None:
        if (
            actual["scale"] != spec.scale
            or Path(actual.get("yaml_file", "")).name != spec.yaml_filename
            or actual.get("scales") != canonical.get("scales")
        ):
            raise ValueError("Native family or scale metadata differs")
    else:
        if spec.family != "yolov8":
            raise ValueError("Only the explicit legacy YOLOv8 YAML layout is supported")
        depth, width, maximum = canonical["scales"][spec.scale]
        if (
            actual.get("depth_multiple") != depth
            or actual.get("width_multiple") != width
            or (
                actual.get("yaml_file")
                and Path(actual["yaml_file"]).name != spec.yaml_filename
            )
        ):
            raise ValueError("Legacy native scale metadata differs")
        expected_graph = deepcopy(expected_graph)
        for layer in expected_graph["backbone"] + expected_graph["head"]:
            if layer[2] in {"Conv", "C2f", "SPPF"}:
                layer[3][0] = min(layer[3][0], maximum)
    if any(actual.get(key) != value for key, value in expected_graph.items()):
        raise ValueError("Native backbone or head topology differs")


def architecture_signature(model: Any) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    return (
        tuple(
            (name, type(module).__module__, type(module).__qualname__)
            for name, module in model.named_modules(remove_duplicate=False)
        ),
        tuple(
            (name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()
        ),
    )


@lru_cache(maxsize=128)
def reference_signature(
    base_model: str | tuple[str, int],
    classes: int,
    factory: Callable[[str | tuple[str, int], int], Any],
    framework_version: str,
) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Cache compact identity data; factory owns model construction and RNG isolation."""
    return architecture_signature(factory(base_model, classes))


def matches_architecture_signature(
    model: Any,
    base_model: str,
    classes: int,
    factory: Callable[[str | tuple[str, int], int], Any],
    framework_version: str,
) -> bool:
    actual = architecture_signature(model)
    return actual == reference_signature(
        base_model, classes, factory, framework_version
    ) or actual == reference_signature(
        (base_model, 80), classes, factory, framework_version
    )


def _reference_model(base_model: str | tuple[str, int], classes: int) -> Any:
    import torch
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel

    source_classes = None
    if isinstance(base_model, tuple):
        base_model, source_classes = base_model
    native_type = SegmentationModel if base_model.endswith("-seg") else DetectionModel
    with torch.random.fork_rng(devices=[]), torch.device("cpu"), torch.no_grad():
        model = native_type(base_model + ".yaml", nc=classes, verbose=False)
        if source_classes is not None:
            source = native_type(base_model + ".yaml", nc=source_classes, verbose=False)
            model.load(source, verbose=False)
        return model


def validate_architecture(
    model: Any, base_model: str, *, classes: int | None = None
) -> ModelSpec:
    from ultralytics import __version__
    from ultralytics.nn.modules import head as native_heads
    from ultralytics.nn.tasks import yaml_model_load

    spec = model_spec(base_model)
    architecture = model.yaml
    head = model.model[-1]
    validate_architecture_yaml(architecture, spec, yaml_model_load(spec.yaml_filename))
    if (
        type(head) is not getattr(native_heads, spec.head)
        or (classes is not None and head.nc != classes)
        or (getattr(head, "one2one_cv2", None) is not None) != spec.end2end
        or (spec.end2end and head.reg_max != 1)
        or not matches_architecture_signature(
            model,
            spec.id,
            head.nc,
            _reference_model,
            __version__,
        )
    ):
        raise ValueError(
            "Unqualified native model family, scale, head or inference branch"
        )
    return spec
