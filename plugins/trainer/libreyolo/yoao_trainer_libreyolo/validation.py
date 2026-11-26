"""Strict LibreYOLO 1.6.0 config validation and native observation checks."""

from __future__ import annotations

from typing import Any

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    NativeConfigurationMismatch,
    ValidateConfigResultV1,
    ValidationIssueV1,
    normalized,
    validate_value,
)

from . import (
    FRAMEWORK_ID,
    SUPPORTED_DATASET_CONTRACTS,
    SUPPORTED_TASKS,
)
from .catalog import FIXED_VALUES, architecture_by_id, parameter_schema

CANONICAL = ("epochs", "batch", "imgsz", "seed", "lr0", "patience")
ALLOWED_RUNTIME_OPTIONS = frozenset(
    {
        "data",
        "project",
        "name",
        "exist_ok",
        "resume",
        "num_classes",
        "size",
        "data_dir",
    }
)


def _apply_cross_field_rules(
    effective: dict[str, Any],
    origins: dict[str, str],
    user_values: dict[str, Any],
    issues: list[ValidationIssueV1],
) -> dict[str, str]:
    inactive: dict[str, str] = {}
    if effective["warmup_epochs"] == 0:
        inactive.update(
            {
                key: "Unused when warmup_epochs is 0"
                for key in ("warmup_lr_start", "warmup_momentum")
            }
        )
    if not effective["ema"]:
        inactive["ema_decay"] = "Unused when ema is disabled"
    if effective["optimizer"] != "sgd":
        inactive.update(
            {
                key: "Unused by non-SGD optimizers"
                for key in ("momentum", "nesterov", "warmup_momentum")
            }
        )
    elif effective["nesterov"] and effective["momentum"] <= 0:
        issues.append(
            ValidationIssueV1(
                code="cross_field_conflict",
                field="momentum",
                message="nesterov=True with optimizer='sgd' requires momentum > 0",
            )
        )
    if effective["scheduler"] != "linear":
        inactive["warmup_momentum"] = "Only used by linear scheduler with SGD"
    if (
        effective["warmup_epochs"] > 0
        and effective["warmup_lr_start"] > effective["lr0"]
    ):
        if "warmup_lr_start" in user_values:
            issues.append(
                ValidationIssueV1(
                    code="cross_field_conflict",
                    field="warmup_lr_start",
                    message="warmup_lr_start must not exceed lr0",
                )
            )
        else:
            effective["warmup_lr_start"] = effective["lr0"]
            origins["warmup_lr_start"] = "dependency:lr0"
    if effective["mosaic_prob"] == 0:
        inactive.update(
            {
                key: "Only used when mosaic_prob > 0"
                for key in (
                    "degrees",
                    "translate",
                    "mosaic_scale",
                    "shear",
                    "perspective",
                    "mixup_prob",
                )
            }
        )
    if effective["no_aug_epochs"] >= effective["epochs"]:
        if "no_aug_epochs" in user_values:
            for key in (
                "mosaic_prob",
                "mixup_prob",
                "degrees",
                "translate",
                "mosaic_scale",
                "shear",
                "perspective",
            ):
                inactive[key] = "Disabled because no_aug_epochs covers all epochs"
        else:
            effective["no_aug_epochs"] = max(0, effective["epochs"] - 1)
            origins["no_aug_epochs"] = "dependency:epochs"
    return inactive


def validate_config(payload: Any) -> ValidateConfigResultV1:
    schema = parameter_schema()
    schema_id = schema["schema_id"]
    schema_digest = schema["schema_digest"]
    issues: list[ValidationIssueV1] = []

    if not isinstance(payload, dict):
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            schema_id=schema_id,
            schema_digest=schema_digest,
            issues=[
                ValidationIssueV1(
                    code="invalid_spec",
                    message="Validation payload must be a JSON object",
                )
            ],
        )

    framework_id = payload.get("framework_id")
    if framework_id is not None and framework_id != FRAMEWORK_ID:
        issues.append(
            ValidationIssueV1(
                code="unsupported_framework",
                field="framework_id",
                message=f"Unsupported framework_id {framework_id!r}; expected {FRAMEWORK_ID!r}",
            )
        )

    task = payload.get("task")
    if type(task) is not str or task not in SUPPORTED_TASKS:
        issues.append(
            ValidationIssueV1(
                code="unsupported_task",
                field="task",
                message=f"LibreYOLO trainer supports only tasks {list(SUPPORTED_TASKS)}; got {task!r}",
            )
        )

    raw_arch = payload.get("architecture_id")
    arch = architecture_by_id(raw_arch) if isinstance(raw_arch, str) else None
    if arch is None:
        issues.append(
            ValidationIssueV1(
                code="unsupported_architecture",
                field="architecture_id",
                message=f"Unsupported LibreYOLO architecture {raw_arch!r}",
            )
        )

    dataset_contract = payload.get("dataset_contract")
    if (
        type(dataset_contract) is not str
        or dataset_contract not in SUPPORTED_DATASET_CONTRACTS
    ):
        issues.append(
            ValidationIssueV1(
                code="unsupported_dataset_contract",
                field="dataset_contract",
                message=(
                    f"Unsupported dataset_contract {dataset_contract!r}; "
                    f"expected one of {list(SUPPORTED_DATASET_CONTRACTS)}"
                ),
            )
        )

    req_schema_id = payload.get("schema_id")
    req_schema_digest = payload.get("schema_digest")
    if req_schema_id is not None and req_schema_id != schema_id:
        issues.append(
            ValidationIssueV1(
                code="schema_mismatch",
                field="schema_id",
                message=f"Unknown schema_id {req_schema_id!r}; expected {schema_id!r}",
            )
        )
    if req_schema_digest is not None and req_schema_digest != schema_digest:
        issues.append(
            ValidationIssueV1(
                code="schema_mismatch",
                field="schema_digest",
                message="schema_digest does not match current LibreYOLO 1.6.0 schema",
            )
        )

    submitted = payload.get("parameters", payload.get("values", {}))
    if not isinstance(submitted, dict) or len(submitted) > 150:
        issues.append(
            ValidationIssueV1(
                code="invalid_parameters",
                field="parameters",
                message="parameters must be a JSON object with at most 150 keys",
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
    aliases = {
        alias: field["key"]
        for field in schema["fields"]
        for alias in field["aliases"]
    }
    user_values: dict[str, Any] = {}
    for supplied, value in submitted.items():
        if not isinstance(supplied, str):
            issues.append(
                ValidationIssueV1(
                    code="invalid_parameters",
                    message="Parameter keys must be strings",
                )
            )
            continue
        key = aliases.get(supplied, supplied)
        field = fields.get(key)
        if field is None or field["status"] != "editable":
            issues.append(
                ValidationIssueV1(
                    code="unsupported_parameter",
                    field=supplied,
                    message=f"Parameter {supplied!r} is unknown, platform-managed, or inapplicable",
                )
            )
            continue
        if key in user_values:
            issues.append(
                ValidationIssueV1(
                    code="duplicate_parameter_alias",
                    field=key,
                    message=f"Parameter {key!r} and its alias cannot both be supplied",
                )
            )
            continue
        try:
            validate_value(value, field["rules"])
        except ValueError as exc:
            issues.append(
                ValidationIssueV1(
                    code="invalid_parameter_value",
                    field=key,
                    message=str(exc),
                )
            )
            continue
        user_values[key] = value

    effective = {
        field["key"]: field["default"]
        for field in schema["fields"]
        if field["status"] == "editable"
    }
    effective.update(user_values)
    origins = {
        key: "user" if key in user_values else "reviewed-default" for key in effective
    }
    effective.update(FIXED_VALUES)
    origins.update({key: "platform" for key in FIXED_VALUES})

    inactive = _apply_cross_field_rules(effective, origins, user_values, issues)
    for key in sorted(user_values.keys() & inactive.keys()):
        issues.append(
            ValidationIssueV1(
                code="inactive_parameter",
                field=key,
                message=inactive[key],
            )
        )

    if issues:
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=False,
            schema_id=schema_id,
            schema_digest=schema_digest,
            inactive=inactive,
            issues=issues,
        )

    resolved_config = {
        "framework_id": FRAMEWORK_ID,
        "task": task,
        "architecture_id": arch.id,
        "dataset_contract": dataset_contract,
        "parameters": {key: effective[key] for key in CANONICAL},
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


def resolve_train_parameters(
    spec_parameters: dict[str, Any],
    *,
    schema_id: str | None = None,
    schema_digest: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize TrainSpecV1.parameters (editable overrides or pre-merged effective values)."""
    schema = parameter_schema()
    if schema_id is not None and schema_id != schema["schema_id"]:
        raise NativeConfigurationMismatch(
            f"Unknown schema_id {schema_id!r}; expected {schema['schema_id']!r}"
        )
    if schema_digest is not None and schema_digest != schema["schema_digest"]:
        raise NativeConfigurationMismatch(
            "schema_digest does not match LibreYOLO 1.6.0 schema"
        )
    if not isinstance(spec_parameters, dict) or len(spec_parameters) > 150:
        raise NativeConfigurationMismatch("parameters must be a bounded JSON object")

    fields = {field["key"]: field for field in schema["fields"]}
    aliases = {
        alias: field["key"]
        for field in schema["fields"]
        for alias in field["aliases"]
    }
    editable_overrides: dict[str, Any] = {}
    supplied_editable: dict[str, Any] = {}
    for supplied, value in spec_parameters.items():
        if not isinstance(supplied, str):
            raise NativeConfigurationMismatch("Parameter keys must be strings")
        key = aliases.get(supplied, supplied)
        field = fields.get(key)
        if field is None:
            raise NativeConfigurationMismatch(f"Unknown LibreYOLO parameter: {supplied!r}")
        if field["status"] != "editable":
            expected_default = field["default"]
            if normalized(value) != normalized(expected_default):
                raise NativeConfigurationMismatch(
                    f"Non-editable parameter {supplied!r} cannot override platform value"
                )
            continue
        if key in supplied_editable:
            if normalized(value) != normalized(supplied_editable[key]):
                raise NativeConfigurationMismatch(
                    f"Conflicting alias and native key supplied for {key!r}"
                )
            continue
        try:
            validate_value(value, field["rules"])
        except ValueError as exc:
            raise NativeConfigurationMismatch(f"{key}: {exc}") from exc
        supplied_editable[key] = value
        if normalized(value) != normalized(field["default"]):
            editable_overrides[key] = value

    effective = {
        field["key"]: field["default"]
        for field in schema["fields"]
        if field["status"] == "editable"
    }
    effective.update(supplied_editable)
    origins = {
        key: "user" if key in editable_overrides else "reviewed-default"
        for key in effective
    }
    effective.update(FIXED_VALUES)
    origins.update({key: "platform" for key in FIXED_VALUES})

    issues: list[ValidationIssueV1] = []
    inactive = _apply_cross_field_rules(effective, origins, editable_overrides, issues)
    for key in sorted(editable_overrides.keys() & inactive.keys()):
        issues.append(
            ValidationIssueV1(
                code="inactive_parameter",
                field=key,
                message=inactive[key],
            )
        )
    if issues:
        detail = "; ".join(
            f"{issue.field}: {issue.message}" if issue.field else issue.message
            for issue in issues
        )
        raise NativeConfigurationMismatch(detail)
    return {
        "schema_id": schema["schema_id"],
        "schema_digest": schema["schema_digest"],
        "effective_values": effective,
        "origins": origins,
        "inactive": inactive,
        "canonical": {key: effective[key] for key in CANONICAL},
    }


def verify_native_observation(
    effective_values: dict[str, Any],
    observed: dict[str, Any],
    runtime_options: dict[str, Any],
    *,
    num_classes: int,
    size: str,
) -> dict[str, Any]:
    """Verify LibreYOLO's resolved TrainConfig matches the frozen effective configuration."""
    if not isinstance(observed, dict) or not observed:
        raise NativeConfigurationMismatch(
            "Native resolved configuration observation is required"
        )
    if set(runtime_options) != ALLOWED_RUNTIME_OPTIONS or (
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
    if (
        runtime_options["num_classes"] != num_classes
        or runtime_options["size"] != size
        or runtime_options["data_dir"] is not None
    ):
        raise NativeConfigurationMismatch(
            "Native dataset or model derived identity mismatch"
        )
    expected = {**effective_values, **runtime_options}
    differences: list[str] = []
    for key, value in expected.items():
        actual = normalized(observed.get(key))
        if key not in observed or actual != normalized(value):
            differences.append(key)
    if differences:
        raise NativeConfigurationMismatch(
            "Native resolved configuration mismatch: " + ", ".join(sorted(differences))
        )
    schema = parameter_schema()
    return {
        "verified": True,
        "checked_fields": sorted(expected),
        "schema_id": schema["schema_id"],
        "schema_digest": schema["schema_digest"],
        "framework": FRAMEWORK_ID,
        "framework_version": schema["framework_version"],
        "task": schema["task"],
        "model_family": schema["model_family"],
    }
