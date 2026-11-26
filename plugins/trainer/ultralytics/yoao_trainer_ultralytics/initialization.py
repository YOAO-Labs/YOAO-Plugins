"""Retained-model weights-only initialization and fresh-state tensor identity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import uuid

import numpy as np

from . import FRAMEWORK_ID, InputMismatchError
from .catalog import model_spec
from .dataset import data_contract, normalize_task
from .files import canonical_digest, digest_file
from .native_identity import validate_architecture
from .policies import (
    LEGACY_INIT_POLICY,
    LEGACY_INIT_POLICY_DIGEST,
    LEGACY_SEGMENTATION_INIT_POLICY,
    LEGACY_SEGMENTATION_INIT_POLICY_DIGEST,
    MAX_WEIGHTS_BYTES,
    SEGMENTATION,
    initialization_policy_for,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_VARIANTS = {f"yolo11{scale}" for scale in "nsmlx"}

PIN_KEYS = frozenset(
    {
        "kind",
        "policy_digest",
        "project_id",
        "model_version_id",
        "model_name",
        "run_id",
        "attempt_id",
        "dataset_id",
        "framework",
        "framework_version",
        "task",
        "model_family",
        "base_model",
        "artifact",
        "classes",
        "classes_digest",
        "bindings",
        "binding_digest",
        "config_digest",
        "dataset_digest",
        "validation_digest",
        "runtime",
        "evaluation",
        "source_digest",
    }
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def tensor_identity(model: Any) -> tuple[str, int]:
    """Compute a deterministic FP32-normalized SHA-256 fingerprint of every state tensor."""
    identity = hashlib.sha256()
    state = model.state_dict()
    for name, value in sorted(state.items()):
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            dtype_str = f"torch.{arr.dtype}"
            shape = list(arr.shape)
            raw_bytes = arr.tobytes()
        else:
            tensor = value.detach().cpu().contiguous()
            if tensor.is_floating_point():
                tensor = tensor.float()
            dtype_str = str(tensor.dtype)
            shape = list(tensor.shape)
            raw_bytes = tensor.numpy().tobytes()
        identity.update(
            json.dumps([name, shape, dtype_str], separators=(",", ":")).encode("utf-8")
        )
        identity.update(raw_bytes)
    return identity.hexdigest(), len(state)


def policy_for_pin(pin: Any) -> dict[str, Any]:
    if not isinstance(pin, dict):
        raise ValueError("Initialization pin must be a JSON object")
    if pin.get("policy_digest") == LEGACY_INIT_POLICY_DIGEST:
        return LEGACY_INIT_POLICY
    if pin.get("policy_digest") == LEGACY_SEGMENTATION_INIT_POLICY_DIGEST:
        return LEGACY_SEGMENTATION_INIT_POLICY
    base_model = pin.get("base_model")
    framework_version = pin.get("framework_version")
    if isinstance(base_model, str) and isinstance(framework_version, str):
        spec = model_spec(base_model)
        policy = initialization_policy_for(spec.family, spec.task, framework_version)
        if pin.get("policy_digest") == canonical_digest(policy):
            return policy
        v3_policy = {
            "framework": FRAMEWORK_ID,
            "framework_version": framework_version,
            "task": spec.task,
            "model_family": spec.family,
            "base_model": spec.id,
        }
        if pin.get("policy_digest") == canonical_digest(v3_policy):
            return {**policy, **v3_policy, "_v3_runtime": True}
    raise ValueError("Initialization source policy digest is not qualified")


def artifact_root(pin: dict[str, Any]) -> str:
    key = pin["artifact"]["key"]
    logical = (
        f"projects/{pin['project_id']}/runs/{pin['run_id']}/attempts/{pin['attempt_id']}/"
    )
    if not isinstance(key, str):
        raise ValueError("Invalid initialization artifact key")
    if key.startswith(logical):
        return ""
    marker = "_aap_managed/training/"
    root, separator, suffix = key.rpartition(marker + logical)
    prefix = root + marker
    if (
        not separator
        or not suffix
        or prefix.startswith("__aap_reference__/")
        or len(prefix) > 266
        or len(prefix.encode("utf-8")) > 707
        or len(key.encode("utf-8")) > 1024
        or "\\" in key
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("Invalid initialization artifact key")
    return prefix


def validate_pin(pin: Any) -> dict[str, Any]:
    policy = policy_for_pin(pin)
    segmentation = normalize_task(policy["task"]) == SEGMENTATION
    expected_keys = PIN_KEYS | ({"data_contract"} if segmentation else set())
    if set(pin) != expected_keys:
        raise ValueError("Initialization pin fields are incomplete or unexpected")
    if (
        pin["kind"] != "trained_model"
        or pin["framework"] != FRAMEWORK_ID
        or any(
            pin[key] != policy[key]
            for key in ("framework", "framework_version", "task", "model_family")
        )
        or not isinstance(pin["base_model"], str)
        or pin["base_model"] not in policy.get("qualified_models", _LEGACY_VARIANTS)
    ):
        raise ValueError("Initialization source is not qualified for this runtime")
    if segmentation and (
        normalize_task(data_contract(pin["data_contract"])["task"]) != SEGMENTATION
        or data_contract(pin["data_contract"]) != pin["data_contract"]
    ):
        raise ValueError("Segmentation initialization requires exact frozen data_contract")
    for key in ("project_id", "model_version_id", "run_id", "attempt_id", "dataset_id"):
        if not isinstance(pin[key], str) or str(uuid.UUID(pin[key])) != pin[key]:
            raise ValueError(f"Initialization pin {key} must be a canonical UUID")
    if not isinstance(pin["model_name"], str) or not 0 < len(pin["model_name"]) <= 120:
        raise ValueError("Invalid initialization model_name")
    for key in (
        "source_digest",
        "classes_digest",
        "binding_digest",
        "config_digest",
        "dataset_digest",
        "validation_digest",
    ):
        if not _is_sha256(pin[key]):
            raise ValueError(f"Invalid initialization digest for {key}")
    artifact = pin["artifact"]
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"key", "role", "sha256", "size_bytes"}
        or artifact["role"] != "weights"
        or not isinstance(artifact["key"], str)
        or len(artifact["key"]) > 500
        or not _is_sha256(artifact["sha256"])
        or type(artifact["size_bytes"]) is not int
        or not 0 < artifact["size_bytes"] <= MAX_WEIGHTS_BYTES
    ):
        raise ValueError("Invalid initialization artifact metadata")
    artifact_root(pin)
    classes, bindings = pin["classes"], pin["bindings"]
    if (
        not isinstance(classes, list)
        or not 0 < len(classes) <= 10000
        or not all(isinstance(v, str) and 0 < len(v) <= 2000 for v in classes)
        or len(set(classes)) != len(classes)
        or not isinstance(bindings, list)
        or len(bindings) != len(classes)
        or not all(isinstance(v, str) and 0 < len(v) <= 2000 for v in bindings)
        or canonical_digest(classes) != pin["classes_digest"]
        or canonical_digest({"classes": classes, "bindings": bindings})
        != pin["binding_digest"]
    ):
        raise ValueError("Initialization classes or binding identity mismatch")
    runtime = pin["runtime"]
    if not isinstance(runtime, dict):
        raise ValueError("Initialization runtime environment is not qualified")
    if policy.get("_v3_runtime"):
        if set(runtime) != {"python", "torch", "torchvision", "cuda", "image_digest"}:
            raise ValueError("Initialization runtime environment is not qualified")
    elif (
        set(runtime) != {"python", "torch", "torchvision", "cuda", "image_digest"}
        or not all(isinstance(v, str) and 0 < len(v) <= 150 for v in runtime.values())
        or runtime["torch"].split("+", 1)[0] != policy["torch"]
        or runtime["torchvision"].split("+", 1)[0] != policy["torchvision"]
        or re.fullmatch(r"3\.12\.\d+", runtime["python"]) is None
        or runtime["cuda"] != policy["cuda"]
        or not runtime["image_digest"].startswith("sha256:")
        or not _is_sha256(runtime["image_digest"][7:])
    ):
        raise ValueError("Initialization runtime environment is not qualified")
    evaluation = pin["evaluation"]
    expected_eval_keys = {"report_sha256", "report_digest", "validation_digest"} | (
        {"normalized_model_digest", "descriptor_digest", "weights_sha256"}
        if segmentation
        else set()
    )
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != expected_eval_keys
        or not all(_is_sha256(v) for v in evaluation.values())
        or evaluation["validation_digest"] != pin["validation_digest"]
        or (segmentation and evaluation["weights_sha256"] != artifact["sha256"])
        or canonical_digest({k: v for k, v in pin.items() if k != "source_digest"})
        != pin["source_digest"]
    ):
        raise ValueError("Initialization evaluation or source digest mismatch")
    return pin


def verify_evidence(
    pin: dict[str, Any], evidence: Any, *, resumed: bool
) -> None:
    validate_pin(pin)
    if not isinstance(evidence, dict) or (
        evidence.get("policy_digest") != pin["policy_digest"]
        or evidence.get("source_digest") != pin["source_digest"]
        or evidence.get("source_sha256") != pin["artifact"]["sha256"]
        or evidence.get("mode") != ("checkpoint_resume" if resumed else "weights_only")
    ):
        raise ValueError("Initialization evidence differs from pinned source or mode")
    if resumed:
        if type(evidence.get("start_epoch")) is not int or evidence["start_epoch"] <= 0:
            raise ValueError("Resumed run must restore positive start_epoch")
        return
    if (
        not _is_sha256(evidence.get("source_tensor_digest"))
        or evidence.get("source_tensor_digest") != evidence.get("loaded_tensor_digest")
        or type(evidence.get("tensor_count")) is not int
        or evidence["tensor_count"] <= 0
        or type(evidence.get("start_epoch")) is not int
        or evidence["start_epoch"] != 0
        or type(evidence.get("optimizer_state_count")) is not int
        or evidence["optimizer_state_count"] != 0
        or type(evidence.get("scheduler_last_epoch")) is not int
        or evidence["scheduler_last_epoch"] not in {-1, 0}
        or evidence.get("scaler_state") != {}
        or type(evidence.get("ema_updates")) is not int
        or evidence["ema_updates"] != 0
        or evidence.get("best_fitness") not in (None, 0.0)
        or evidence.get("best_weights_present") is not False
    ):
        raise ValueError(
            "Weights-only initialization failed full tensor transfer or fresh-state reset"
        )
    if normalize_task(policy_for_pin(pin)["task"]) == SEGMENTATION and (
        not _is_sha256(evidence.get("mask_head_tensor_digest"))
        or evidence["mask_head_tensor_digest"]
        != evidence.get("loaded_mask_head_tensor_digest")
        or type(evidence.get("mask_head_tensor_count")) is not int
        or evidence["mask_head_tensor_count"] <= 0
    ):
        raise ValueError("Segmentation initialization failed mask head tensor transfer")


def validate_input(
    init_pin: dict[str, Any] | None,
    *,
    init_weights_path: str | Path | None,
    resume_checkpoint_path: str | Path | None,
) -> None:
    if init_pin is None:
        return
    try:
        validate_pin(init_pin)
        if not resume_checkpoint_path:
            if not init_weights_path:
                raise ValueError("init_weights_path is required for initialization pin")
            source = Path(init_weights_path)
            if (
                not source.is_file()
                or source.stat().st_size != init_pin["artifact"]["size_bytes"]
                or digest_file(source) != init_pin["artifact"]["sha256"]
            ):
                raise ValueError(
                    "Native initialization artifact size or digest differs"
                )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InputMismatchError(str(exc)) from exc


def prepare(
    init_pin: dict[str, Any] | None,
    model: Any,
    *,
    classes: list[str],
    resume_checkpoint_path: str | Path | None = None,
    architecture_validator: Any = validate_architecture,
) -> dict[str, Any] | None:
    if init_pin is None:
        return None
    try:
        validate_pin(init_pin)
        common = {
            "policy_digest": init_pin["policy_digest"],
            "source_digest": init_pin["source_digest"],
            "source_sha256": init_pin["artifact"]["sha256"],
        }
        if resume_checkpoint_path:
            return {**common, "mode": "checkpoint_resume"}
        names = getattr(model.model, "names", None)
        ordered = (
            [names.get(index) for index in range(len(names))]
            if isinstance(names, dict)
            else names
        )
        segmentation = normalize_task(init_pin["task"]) == SEGMENTATION
        if architecture_validator is not None:
            architecture_validator(
                model.model, init_pin["base_model"], classes=len(ordered)
            )
        expected_task = "segment" if segmentation else "detect"
        try:
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel

            expected_type = SegmentationModel if segmentation else DetectionModel
            type_ok = type(model.model) is expected_type
        except ImportError:
            type_ok = True
        if (
            not type_ok
            or model.task != expected_task
            or ordered != init_pin["classes"]
            or ordered != classes
        ):
            raise ValueError(
                "Native initialization artifact, architecture, or class identity differs"
            )
        fingerprint, count = tensor_identity(model.model)
        head_identity = (
            tensor_identity(model.model.model[-1]) if segmentation else None
        )
        return {
            **common,
            **(
                {
                    "mask_head_tensor_digest": head_identity[0],
                    "mask_head_tensor_count": head_identity[1],
                }
                if head_identity is not None
                else {}
            ),
            "mode": "weights_only",
            "source_tensor_digest": fingerprint,
            "tensor_count": count,
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InputMismatchError(str(exc)) from exc


def observation(
    init_pin: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    trainer: Any,
    *,
    resumed: bool,
) -> dict[str, Any] | None:
    if baseline is None or init_pin is None:
        return None
    evidence = {**baseline, "start_epoch": int(trainer.start_epoch)}
    if baseline["mode"] == "weights_only":
        model = getattr(trainer.model, "module", trainer.model)
        fingerprint, count = tensor_identity(model)
        if count != baseline["tensor_count"]:
            raise InputMismatchError("Native initialization state tensor count changed")
        if "mask_head_tensor_digest" in baseline:
            head_fingerprint, head_count = tensor_identity(model.model[-1])
            if head_count != baseline["mask_head_tensor_count"]:
                raise InputMismatchError(
                    "Native initialization mask head tensor count changed"
                )
            evidence["loaded_mask_head_tensor_digest"] = head_fingerprint
        evidence.update(
            loaded_tensor_digest=fingerprint,
            optimizer_state_count=len(trainer.optimizer.state),
            scheduler_last_epoch=trainer.scheduler.last_epoch,
            scaler_state=trainer.scaler.state_dict(),
            ema_updates=trainer.ema.updates,
            best_fitness=(
                float(trainer.best_fitness)
                if trainer.best_fitness is not None
                else None
            ),
            best_weights_present=Path(trainer.best).exists(),
        )
    try:
        verify_evidence(init_pin, evidence, resumed=resumed)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InputMismatchError(str(exc)) from exc
    return evidence
