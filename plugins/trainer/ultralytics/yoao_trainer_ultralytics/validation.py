"""Strict JSON validation and cross-field rules for Ultralytics 8.4.183."""

from __future__ import annotations

from typing import Any

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    ValidateConfigResultV1,
    ValidationIssueV1,
    normalized,
    validate_value,
)

from . import FRAMEWORK_ID, FRAMEWORK_VERSION, NativeConfigurationMismatch
from .catalog import (
    CANONICAL_PARAMETERS,
    DETECT_DATASET_CONTRACTS,
    FIXED_FIELDS,
    RLE_FORBIDDEN_FIELDS,
    SEGMENT_DATASET_CONTRACTS,
    SEGMENT_RLE_DATASET_CONTRACTS,
    SEGMENTATION_FIXED_FIELDS,
    SUPPORTED_TASKS,
    ModelSpec,
    model_spec,
    parameter_schema_for,
)
from .policies import MAX_SEMANTIC_CLASS_PIXELS, SEGMENTATION


def _fixed_values_for(spec: ModelSpec, *, is_rle: bool) -> dict[str, Any]:
    fixed = dict(FIXED_FIELDS)
    if spec.end2end:
        fixed["nms"] = False
    if spec.task == SEGMENTATION:
        fixed.update(SEGMENTATION_FIXED_FIELDS)
    if is_rle:
        fixed["workers"] = 0
    return fixed


def native_values(effective_values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in effective_values.items()
        if key != "evaluate_train"
    }


def _extract_num_classes(payload: dict[str, Any]) -> tuple[int | None, ValidationIssueV1 | None]:
    raw_nc = payload.get("num_classes")
    if raw_nc is not None:
        if type(raw_nc) is not int or raw_nc < 1:
            return None, ValidationIssueV1(
                code="invalid_value",
                field="num_classes",
                message="num_classes must be a positive integer",
            )
        return raw_nc, None
    classes = payload.get("classes")
    if classes is not None:
        if (
            not isinstance(classes, list)
            or not classes
            or not all(isinstance(item, str) and item.strip() for item in classes)
        ):
            return None, ValidationIssueV1(
                code="invalid_value",
                field="classes",
                message="classes must be a non-empty list of strings",
            )
        return len(classes), None
    return None, None


def validate_config(
    payload: dict[str, Any],
    *,
    allow_effective_passthrough: bool = False,
) -> ValidateConfigResultV1:
    """Validate a config request and return a ValidateConfigResultV1."""
    if not isinstance(payload, dict):
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            issues=[
                ValidationIssueV1(
                    code="invalid_spec",
                    message="Validation spec must be a JSON object",
                )
            ],
        )

    issues: list[ValidationIssueV1] = []
    task = payload.get("task")
    if type(task) is not str or task not in SUPPORTED_TASKS:
        issues.append(
            ValidationIssueV1(
                code="unsupported_task",
                field="task",
                message=f"Unsupported task {task!r}; expected one of {list(SUPPORTED_TASKS)}",
            )
        )

    architecture_id = payload.get("architecture_id") or payload.get("base_model")
    spec: ModelSpec | None = None
    if type(architecture_id) is not str or not architecture_id.strip():
        issues.append(
            ValidationIssueV1(
                code="missing_architecture",
                field="architecture_id",
                message="architecture_id is required",
            )
        )
    else:
        try:
            spec = model_spec(architecture_id)
        except ValueError as exc:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_architecture",
                    field="architecture_id",
                    message=str(exc),
                )
            )
        else:
            if type(task) is str and task in SUPPORTED_TASKS and spec.task != task:
                issues.append(
                    ValidationIssueV1(
                        code="architecture_task_mismatch",
                        field="architecture_id",
                        message=(
                            f"Architecture {spec.id!r} supports task {spec.task!r}, "
                            f"not {task!r}"
                        ),
                    )
                )

    dataset_contract = payload.get("dataset_contract")
    is_rle = dataset_contract in SEGMENT_RLE_DATASET_CONTRACTS
    if type(dataset_contract) is not str or not dataset_contract.strip():
        issues.append(
            ValidationIssueV1(
                code="missing_dataset_contract",
                field="dataset_contract",
                message="dataset_contract is required",
            )
        )
    elif type(task) is str and task in SUPPORTED_TASKS:
        allowed_contracts = (
            SEGMENT_DATASET_CONTRACTS
            if task == SEGMENTATION
            else DETECT_DATASET_CONTRACTS
        )
        if dataset_contract not in allowed_contracts:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_dataset_contract",
                    field="dataset_contract",
                    message=(
                        f"dataset_contract {dataset_contract!r} is not supported for "
                        f"task {task!r}"
                    ),
                )
            )

    if issues or spec is None:
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            issues=issues,
        )

    schema = parameter_schema_for(spec.id)
    schema_id = schema["schema_id"]
    schema_digest = schema["schema_digest"]

    supplied_schema_id = payload.get("schema_id")
    supplied_schema_digest = payload.get("schema_digest")
    if supplied_schema_id is not None and supplied_schema_id != schema_id:
        issues.append(
            ValidationIssueV1(
                code="schema_mismatch",
                field="schema_id",
                message=f"schema_id {supplied_schema_id!r} does not match {schema_id!r}",
            )
        )
    if supplied_schema_digest is not None and supplied_schema_digest != schema_digest:
        issues.append(
            ValidationIssueV1(
                code="schema_mismatch",
                field="schema_digest",
                message="schema_digest does not match current Ultralytics 8.4.183 schema",
            )
        )

    submitted = payload.get("parameters")
    if submitted is None and "values" in payload:
        submitted = payload["values"]
    if submitted is None:
        submitted = {}
    if not isinstance(submitted, dict) or len(submitted) > 150:
        issues.append(
            ValidationIssueV1(
                code="invalid_parameters",
                field="parameters",
                message="parameters must be a bounded JSON object (<= 150 keys)",
            )
        )
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            schema_id=schema_id,
            schema_digest=schema_digest,
            issues=issues,
        )

    fields = {field["key"]: field for field in schema["fields"]}
    fixed = _fixed_values_for(spec, is_rle=is_rle)

    user_values: dict[str, Any] = {}
    for key, value in submitted.items():
        if type(key) is not str:
            issues.append(
                ValidationIssueV1(
                    code="unknown_parameter",
                    message=f"Parameter key must be a string: {key!r}",
                )
            )
            continue
        field = fields.get(key)
        if field is None:
            issues.append(
                ValidationIssueV1(
                    code="unknown_parameter",
                    field=key,
                    message=f"Unknown Ultralytics 8.4.183 parameter: {key}",
                )
            )
            continue

        status = field["status"]
        if status != "editable":
            if allow_effective_passthrough:
                expected_val = fixed[key] if key in fixed else field["default"]
                if normalized(value) == normalized(expected_val):
                    continue
                if key == "workers" and is_rle:
                    issues.append(
                        ValidationIssueV1(
                            code="rle_incompatible_parameter",
                            field=key,
                            message="ultralytics-instance-rle-1 requires workers=0",
                        )
                    )
                    continue
            if is_rle and key in RLE_FORBIDDEN_FIELDS:
                issues.append(
                    ValidationIssueV1(
                        code="rle_incompatible_parameter",
                        field=key,
                        message=(
                            f"Parameter {key!r} cannot be overridden under "
                            "ultralytics-instance-rle-1"
                        ),
                    )
                )
            elif status == "platform-managed":
                issues.append(
                    ValidationIssueV1(
                        code="platform_managed_parameter",
                        field=key,
                        message=f"Parameter {key!r} is platform-managed and read-only",
                    )
                )
            elif status == "inapplicable":
                issues.append(
                    ValidationIssueV1(
                        code="inapplicable_parameter",
                        field=key,
                        message=field.get("reason")
                        or f"Parameter {key!r} is inapplicable for {spec.id}",
                    )
                )
            else:
                issues.append(
                    ValidationIssueV1(
                        code="unqualified_parameter",
                        field=key,
                        message=field.get("reason")
                        or f"Parameter {key!r} is not qualified for execution",
                    )
                )
            continue

        try:
            validate_value(value, field["rules"])
        except ValueError as exc:
            issues.append(
                ValidationIssueV1(
                    code="invalid_value",
                    field=key,
                    message=f"{key}: {exc}",
                )
            )
            continue

        if allow_effective_passthrough and normalized(value) == normalized(
            field["default"]
        ):
            # In supervisor-forwarded effective_values dumps, defaults may be echoed back.
            continue
        user_values[key] = value

    effective: dict[str, Any] = {
        field["key"]: field["default"]
        for field in schema["fields"]
        if field["status"] == "editable"
    }
    effective.update(user_values)
    origins: dict[str, str] = {
        key: "user" if key in user_values else "reviewed-default" for key in effective
    }
    effective.update(fixed)
    for key in fixed:
        origins[key] = (
            "platform:ultralytics-instance-rle-1"
            if is_rle and key == "workers"
            else "platform"
        )

    inactive: dict[str, str] = {}

    if not issues:
        if effective["optimizer"] == "SGD" and effective["momentum"] <= 0:
            issues.append(
                ValidationIssueV1(
                    code="invalid_combination",
                    field="momentum",
                    message="SGD requires momentum > 0 for native Nesterov",
                )
            )

        warmup_limit = max(effective["epochs"] - 1, 0)
        if effective["warmup_epochs"] > warmup_limit:
            if "warmup_epochs" in user_values:
                issues.append(
                    ValidationIssueV1(
                        code="invalid_combination",
                        field="warmup_epochs",
                        message="warmup_epochs cannot exceed epochs - 1",
                    )
                )
            else:
                effective["warmup_epochs"] = warmup_limit
                origins["warmup_epochs"] = "dependency:epochs"

        if effective["warmup_epochs"] == 0:
            inactive["warmup_bias_lr"] = "Unused when warmup_epochs is 0"
            inactive["warmup_momentum"] = "Unused when warmup_epochs is 0"

        if effective["optimizer"] != "SGD":
            inactive["warmup_momentum"] = (
                "Current optimizer does not use SGD momentum warmup"
            )

        if effective["rect"]:
            for key in ("mosaic", "mixup", "cutmix"):
                inactive[key] = "Rectangular batch training disables this augmentation"
                effective[key] = 0.0
                origins[key] = "dependency:rect"

        if effective["close_mosaic"] >= effective["epochs"]:
            if "close_mosaic" in user_values:
                for key in ("mosaic", "mixup", "cutmix"):
                    inactive[key] = "close_mosaic covers the entire training run"
            else:
                effective["close_mosaic"] = max(0, effective["epochs"] - 1)
                origins["close_mosaic"] = "dependency:epochs"

        for key in sorted(user_values.keys() & inactive.keys()):
            if allow_effective_passthrough and normalized(user_values[key]) == normalized(
                effective[key]
            ):
                continue
            issues.append(
                ValidationIssueV1(
                    code="inactive_parameter",
                    field=key,
                    message=inactive[key],
                )
            )

        if payload.get("initialization") and not effective.get("pretrained", True):
            issues.append(
                ValidationIssueV1(
                    code="initialization_conflict",
                    field="pretrained",
                    message="Random initialization (pretrained=False) conflicts with retained-model initialization",
                )
            )

    num_classes, nc_issue = _extract_num_classes(payload)
    if nc_issue is not None:
        issues.append(nc_issue)
    elif (
        not issues
        and spec.family == "yolo26"
        and spec.task == SEGMENTATION
        and num_classes is not None
    ):
        semantic_pixels = effective["batch"] * num_classes * (effective["imgsz"] ** 2)
        if semantic_pixels > MAX_SEMANTIC_CLASS_PIXELS:
            issues.append(
                ValidationIssueV1(
                    code="semantic_pixel_budget_exceeded",
                    field="batch",
                    message=(
                        f"YOLO26 segmentation semantic targets exceed {MAX_SEMANTIC_CLASS_PIXELS} "
                        f"class pixels: batch={effective['batch']}, num_classes={num_classes}, "
                        f"imgsz={effective['imgsz']}"
                    ),
                )
            )

    if issues:
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            schema_id=schema_id,
            schema_digest=schema_digest,
            issues=issues,
        )

    resolved_config = {
        "framework_id": FRAMEWORK_ID,
        "framework_version": FRAMEWORK_VERSION,
        "architecture_id": spec.id,
        "task": spec.task,
        "dataset_contract": dataset_contract,
        "end2end": spec.end2end,
        "evaluate_train": bool(effective.get("evaluate_train", False)),
        "canonical_parameters": {
            key: effective[key] for key in CANONICAL_PARAMETERS
        },
        "native_values": native_values(effective),
    }
    return ValidateConfigResultV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        valid=True,
        schema_id=schema_id,
        schema_digest=schema_digest,
        resolved_config=resolved_config,
        effective_values=effective,
        origins=origins,
        inactive=inactive,
        issues=[],
    )


def verify_native_configuration(
    effective_values: dict[str, Any],
    observed: dict[str, Any],
    runtime_options: dict[str, Any],
    *,
    schema_id: str,
    schema_digest: str,
) -> dict[str, Any]:
    """Verify Ultralytics resolved args match our frozen effective values and runtime options."""
    if not isinstance(observed, dict) or not observed:
        raise NativeConfigurationMismatch(
            "Native resolved configuration observation is required"
        )
    allowed_runtime = {"data", "model", "project", "name", "exist_ok", "resume"}
    if set(runtime_options) != allowed_runtime or (
        set(runtime_options) & set(effective_values)
    ):
        raise NativeConfigurationMismatch(
            "Runtime materialization cannot override effective parameters"
        )
    if (
        runtime_options["name"] != "framework"
        or runtime_options["exist_ok"] is not False
    ):
        raise NativeConfigurationMismatch("Native output ownership mismatch")

    expected = {**native_values(effective_values), **runtime_options}
    differences: list[str] = []
    for key, value in expected.items():
        actual = normalized(observed.get(key))
        expected_val = normalized(value)
        if key == "device":
            actual, expected_val = str(actual), str(expected_val)
        if key not in observed or actual != expected_val:
            differences.append(key)
    if differences:
        raise NativeConfigurationMismatch(
            "Native resolved configuration mismatch: " + ", ".join(sorted(differences))
        )
    return {
        "verified": True,
        "checked_fields": sorted(expected),
        "schema_id": schema_id,
        "schema_digest": schema_digest,
        "framework": FRAMEWORK_ID,
        "framework_version": FRAMEWORK_VERSION,
    }
