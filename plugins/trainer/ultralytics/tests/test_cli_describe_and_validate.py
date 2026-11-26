"""CLI tests for `describe` and `validate-config` subcommands."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
from yoao_plugin_sdk.trainer import (
    TrainerCapabilitiesV1,
    ValidateConfigResultV1,
)

from yoao_trainer_ultralytics.catalog import load_pretrained_assets, parameter_schema_for
from yoao_trainer_ultralytics.cli import main

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _run_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: dict) -> ValidateConfigResultV1:
    spec_path = tmp_path / "validate-spec.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    code = main(["validate-config", "--spec", str(spec_path)])
    assert code == 0
    captured = capsys.readouterr()
    return ValidateConfigResultV1.model_validate_json(captured.out)


def test_describe_emits_valid_capabilities_with_30_architectures_and_null_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["describe"])
    assert code == 0
    raw_text = capsys.readouterr().out
    raw = json.loads(raw_text)
    caps = TrainerCapabilitiesV1.model_validate(raw)

    assert caps.protocol_version == "yoao-trainer/v1"
    assert caps.framework_id == "ultralytics"
    assert caps.framework_version == "8.4.183"
    assert caps.supported_tasks == ["detect", "segment"]
    assert "ultralytics-yolo-1" in caps.supported_dataset_contracts
    assert "ultralytics-instance-rle-1" in caps.supported_dataset_contracts

    assert len(caps.architectures) == 30
    assets = load_pretrained_assets()
    by_id = {arch.id: arch for arch in caps.architectures}
    for family in ("yolov8", "yolo11", "yolo26"):
        for scale in ("n", "s", "m", "l", "x"):
            for suffix, expected_task in (("", "detect"), ("-seg", "segment")):
                arch_id = f"{family}{scale}{suffix}"
                arch = by_id[arch_id]
                assert arch.family == family
                assert arch.tasks == [expected_task]
                assert arch.end2end is (family == "yolo26")
                assert arch.base_asset_filename == assets[arch_id]["filename"]
                assert arch.base_asset_sha256 == assets[arch_id]["sha256"]
                assert SHA256_RE.fullmatch(arch.base_asset_sha256 or "")

    assert set(caps.evaluator_descriptors) == {"detect", "segment"}
    for task_key in ("detect", "segment"):
        assert SHA256_RE.fullmatch(caps.evaluator_descriptors[task_key]["source_sha256"])

    for digest_key in ("initialization", "preview", "input_membership", "instance_rle", "budget"):
        assert digest_key in caps.policy_digests
        assert SHA256_RE.fullmatch(caps.policy_digests[digest_key])
    assert (
        caps.policy_digests["input_membership"]
        == "1ad70c6b02157e1b4c33646027309a72e73aaa5eba693e7a5fe52ce8f9bb71b4"
    )

    assert len(caps.serving_compatibility) == 4
    for compat in caps.serving_compatibility:
        assert compat.framework == "ultralytics"
        assert compat.format is None
        assert compat.training_runtime["framework"] == "ultralytics"
        assert compat.training_runtime["framework_version"] == "8.4.183"
        for rt in compat.compatible_serving_runtimes:
            assert rt["framework"] == "ultralytics"
            assert rt["format"] is None
            assert rt["min_version"] == "8.4.183"
    assert '"format":null' in raw_text
    assert '"format":""' not in raw_text
    assert {c.training_runtime["end2end"] for c in caps.serving_compatibility} == {False, True}


def test_validate_config_accepts_detect_rle_and_yolo26_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    detect = _run_validate(
        tmp_path,
        capsys,
        {
            "task": "detect",
            "architecture_id": "yolo11n",
            "dataset_contract": "ultralytics-yolo-1",
            "parameters": {"epochs": 10, "batch": 4, "imgsz": 640, "optimizer": "SGD", "momentum": 0.9},
        },
    )
    assert detect.valid is True
    assert detect.issues == []
    assert detect.effective_values["epochs"] == 10
    assert detect.effective_values["workers"] == 2
    assert detect.effective_values["close_mosaic"] == 9
    assert detect.origins["close_mosaic"] == "dependency:epochs"
    assert detect.resolved_config is not None
    assert detect.resolved_config["end2end"] is False

    rle = _run_validate(
        tmp_path,
        capsys,
        {
            "task": "segment",
            "architecture_id": "yolo11n-seg",
            "dataset_contract": "ultralytics-instance-rle-1",
            "parameters": {"epochs": 5, "batch": 2, "imgsz": 640},
        },
    )
    assert rle.valid is True
    assert rle.effective_values["workers"] == 0
    assert rle.origins["workers"] == "platform:ultralytics-instance-rle-1"
    assert rle.effective_values["overlap_mask"] is False
    assert rle.effective_values["mask_ratio"] == 1

    y26 = _run_validate(
        tmp_path,
        capsys,
        {
            "task": "segment",
            "architecture_id": "yolo26n-seg",
            "dataset_contract": "ultralytics-instance-rle-1",
            "parameters": {"epochs": 4, "batch": 2, "imgsz": 640},
            "num_classes": 4,
        },
    )
    assert y26.valid is True
    assert y26.effective_values["nms"] is False
    assert y26.effective_values["workers"] == 0
    assert y26.resolved_config is not None
    assert y26.resolved_config["end2end"] is True


@pytest.mark.parametrize(
    ("payload", "expected_code", "expected_field"),
    [
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"not_a_real_param": 1},
            },
            "unknown_parameter",
            "not_a_real_param",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"epochs": "10"},
            },
            "invalid_value",
            "epochs",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"optimizer": "SGD", "momentum": 0},
            },
            "invalid_combination",
            "momentum",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"optimizer": "AdamW", "warmup_momentum": 0.8},
            },
            "inactive_parameter",
            "warmup_momentum",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"warmup_epochs": 0, "warmup_bias_lr": 0.1},
            },
            "inactive_parameter",
            "warmup_bias_lr",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"epochs": 5, "warmup_epochs": 5},
            },
            "invalid_combination",
            "warmup_epochs",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"rect": True, "mosaic": 0.5},
            },
            "inactive_parameter",
            "mosaic",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"epochs": 5, "close_mosaic": 5, "mixup": 0.2},
            },
            "inactive_parameter",
            "mixup",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo26n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"dfl": 1.5},
            },
            "inapplicable_parameter",
            "dfl",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo26n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"nms": True},
            },
            "platform_managed_parameter",
            "nms",
        ),
        (
            {
                "task": "segment",
                "architecture_id": "yolo26n-seg",
                "dataset_contract": "ultralytics-instance-rle-1",
                "parameters": {"batch": 16, "imgsz": 640},
                "num_classes": 12,
            },
            "semantic_pixel_budget_exceeded",
            "batch",
        ),
        (
            {
                "task": "segment",
                "architecture_id": "yolo11n-seg",
                "dataset_contract": "ultralytics-instance-rle-1",
                "parameters": {"workers": 2},
            },
            "rle_incompatible_parameter",
            "workers",
        ),
        (
            {
                "task": "segment",
                "architecture_id": "yolo11n-seg",
                "dataset_contract": "ultralytics-instance-rle-1",
                "parameters": {"mosaic": 0.5},
            },
            "rle_incompatible_parameter",
            "mosaic",
        ),
        (
            {
                "task": "segment",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-instance-rle-1",
                "parameters": {},
            },
            "architecture_task_mismatch",
            "architecture_id",
        ),
        (
            {
                "task": "detect",
                "architecture_id": "yolo11n",
                "dataset_contract": "ultralytics-yolo-1",
                "parameters": {"pretrained": False},
                "initialization": True,
            },
            "initialization_conflict",
            "pretrained",
        ),
    ],
)
def test_validate_config_rejects_invalid_and_cross_field_violations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: dict,
    expected_code: str,
    expected_field: str,
) -> None:
    result = _run_validate(tmp_path, capsys, payload)
    assert result.valid is False
    assert any(
        issue.code == expected_code and issue.field == expected_field
        for issue in result.issues
    ), result.issues


def test_validate_config_enforces_schema_identity_and_rect_auto_disable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = parameter_schema_for("yolo11n")
    ok = _run_validate(
        tmp_path,
        capsys,
        {
            "task": "detect",
            "architecture_id": "yolo11n",
            "dataset_contract": "ultralytics-yolo-1",
            "schema_id": schema["schema_id"],
            "schema_digest": schema["schema_digest"],
            "parameters": {"rect": True},
        },
    )
    assert ok.valid is True
    assert ok.effective_values["mosaic"] == 0.0
    assert ok.origins["mosaic"] == "dependency:rect"
    assert "mosaic" in ok.inactive

    stale = _run_validate(
        tmp_path,
        capsys,
        {
            "task": "detect",
            "architecture_id": "yolo11n",
            "dataset_contract": "ultralytics-yolo-1",
            "schema_id": "stale-id",
            "schema_digest": "0" * 64,
            "parameters": {},
        },
    )
    assert stale.valid is False
    assert {issue.code for issue in stale.issues} == {"schema_mismatch"}
