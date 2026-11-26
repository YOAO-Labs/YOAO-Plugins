"""Tests for `describe` and `validate-config` CLI commands and LibreYOLO 1.6.0 schema rules."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from yoao_plugin_sdk.trainer import (
    TrainerCapabilitiesV1,
    ValidateConfigResultV1,
    canonical_json_digest,
)

from yoao_trainer_libreyolo.catalog import controlled_assets
from yoao_trainer_libreyolo.cli import main
from yoao_trainer_libreyolo.membership import MEMBER_FIELDS, MAX_MEMBERS, POLICY_DIGEST, SPLITS

EXPECTED_ASSETS = {
    "libreyolo9t": (
        "LibreYOLO9t.pt",
        "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2",
    ),
    "libreyolo9s": (
        "LibreYOLO9s.pt",
        "4ea4a3591c27c66769702bdba5703920d2eb03bbb427e37ed461fcd65e083fb2",
    ),
    "libreyolo9m": (
        "LibreYOLO9m.pt",
        "efa83041e2aeed114a2fcd5364d19359781c225079fa8f470399380eb96961cc",
    ),
    "libreyolo9c": (
        "LibreYOLO9c.pt",
        "567197cb51e967a09aab8efc9955f937bec693135896c321d373c93c751a4552",
    ),
}


def test_describe_cli_emits_valid_capabilities_and_exact_assets(capsys) -> None:
    rc = main(["describe"])
    assert rc == 0
    raw = json.loads(capsys.readouterr().out)
    caps = TrainerCapabilitiesV1.model_validate(raw)

    assert caps.protocol_version == "yoao-trainer/v1"
    assert caps.framework_id == "libreyolo"
    assert caps.framework_version == "1.6.0"
    assert caps.supported_tasks == ["detect"]
    assert "ultralytics-yolo-1" in caps.supported_dataset_contracts
    assert "ultralytics-bbox-1" in caps.supported_dataset_contracts

    arch_map = {arch.id: arch for arch in caps.architectures}
    assert set(arch_map) == set(EXPECTED_ASSETS)
    for arch_id, (filename, sha256) in EXPECTED_ASSETS.items():
        arch = arch_map[arch_id]
        assert arch.family == "libreyolo9"
        assert arch.tasks == ["detect"]
        assert arch.end2end is False
        assert arch.base_asset_filename == filename
        assert arch.base_asset_sha256 == sha256

    assert len(controlled_assets()) == 4
    assert caps.parameter_schema["schema_id"] == "libreyolo-1.6.0-yolo9-detect-3"
    assert len(caps.parameter_schema["schema_digest"]) == 64
    assert len(caps.parameter_schema["fields"]) == 92

    assert "detect" in caps.evaluator_descriptors
    assert len(caps.evaluator_descriptors["detect"]["source_sha256"]) == 64

    expected_input_policy = canonical_json_digest(
        {
            "schema": 1,
            "scope": "native-loader-membership",
            "member_fields": list(MEMBER_FIELDS),
            "splits": list(SPLITS),
            "max_members": MAX_MEMBERS,
        }
    )
    assert caps.policy_digests["input_membership"] == POLICY_DIGEST == expected_input_policy
    assert "checkpoint_resume" in caps.policy_digests

    assert len(caps.serving_compatibility) == 1
    compat = caps.serving_compatibility[0]
    assert compat.framework == "libreyolo"
    assert compat.format == "libreyolo-native-state-dict-1.0"
    assert compat.training_runtime == {
        "framework": "libreyolo",
        "framework_version": "1.6.0",
    }
    assert compat.compatible_serving_runtimes == [
        {
            "framework": "libreyolo",
            "format": "libreyolo-native-state-dict-1.0",
            "min_version": "1.6.0",
        }
    ]


def test_installed_console_script_describe_runs_cleanly() -> None:
    exe = shutil.which("yoao-trainer-libreyolo")
    if exe is None:
        exe = str(Path(sys.executable).parent / "yoao-trainer-libreyolo")
    assert Path(exe).is_file(), f"Console script not installed at {exe}"
    proc = subprocess.run([exe, "describe"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    caps = TrainerCapabilitiesV1.model_validate(json.loads(proc.stdout))
    assert caps.framework_id == "libreyolo"


@pytest.mark.parametrize("dataset_contract", ["ultralytics-yolo-1", "ultralytics-bbox-1"])
def test_validate_config_cli_accepts_valid_specs_and_clamps_dependencies(
    tmp_path: Path, capsys, dataset_contract: str
) -> None:
    spec_path = tmp_path / "valid.json"
    spec_path.write_text(
        json.dumps(
            {
                "task": "detect",
                "architecture_id": "libreyolo9t",
                "dataset_contract": dataset_contract,
                "parameters": {
                    "epochs": 5,
                    "batch": 4,
                    "imgsz": 320,
                    "lr0": 0.00005,
                    "optimizer": "adamw",
                    "mixup": 0.1,
                    "nbs": None,
                },
            }
        ),
        encoding="utf-8",
    )
    rc = main(["validate-config", "--spec", str(spec_path)])
    assert rc == 0
    result = ValidateConfigResultV1.model_validate(json.loads(capsys.readouterr().out))
    assert result.valid is True
    assert result.issues == []
    assert result.schema_id == "libreyolo-1.6.0-yolo9-detect-3"
    assert result.effective_values["epochs"] == 5
    assert result.effective_values["mixup_prob"] == 0.1
    assert result.effective_values["no_aug_epochs"] == 4
    assert result.origins["no_aug_epochs"] == "dependency:epochs"
    assert result.effective_values["warmup_lr_start"] == 0.00005
    assert result.origins["warmup_lr_start"] == "dependency:lr0"
    assert result.effective_values["amp"] is False
    assert result.origins["amp"] == "platform"
    assert "momentum" in result.inactive
    assert "nesterov" in result.inactive
    assert result.resolved_config["architecture_id"] == "libreyolo9t"


@pytest.mark.parametrize(
    "spec_override,expected_code,expected_field",
    [
        ({"task": "segment"}, "unsupported_task", "task"),
        ({"architecture_id": "libreyolo9e"}, "unsupported_architecture", "architecture_id"),
        ({"architecture_id": "yolo11n"}, "unsupported_architecture", "architecture_id"),
        (
            {"dataset_contract": "ultralytics-instance-rle-1"},
            "unsupported_dataset_contract",
            "dataset_contract",
        ),
        ({"parameters": {"unknown_knob": 1}}, "unsupported_parameter", "unknown_knob"),
        ({"parameters": {"amp": True}}, "unsupported_parameter", "amp"),
        ({"parameters": {"compile": True}}, "unsupported_parameter", "compile"),
        ({"parameters": {"epochs": 0}}, "invalid_parameter_value", "epochs"),
        ({"parameters": {"epochs": True}}, "invalid_parameter_value", "epochs"),
        ({"parameters": {"imgsz": 641}}, "invalid_parameter_value", "imgsz"),
        (
            {"parameters": {"mosaic_scale": [1.5, 0.5]}},
            "invalid_parameter_value",
            "mosaic_scale",
        ),
        (
            {"parameters": {"optimizer": "sgd", "nesterov": True, "momentum": 0.0}},
            "cross_field_conflict",
            "momentum",
        ),
        (
            {"parameters": {"optimizer": "adamw", "momentum": 0.9}},
            "inactive_parameter",
            "momentum",
        ),
        (
            {"parameters": {"warmup_epochs": 0, "warmup_lr_start": 0.001}},
            "inactive_parameter",
            "warmup_lr_start",
        ),
        (
            {"parameters": {"mosaic_prob": 0.0, "degrees": 5.0}},
            "inactive_parameter",
            "degrees",
        ),
        (
            {"parameters": {"lr0": 0.001, "warmup_epochs": 2, "warmup_lr_start": 0.01}},
            "cross_field_conflict",
            "warmup_lr_start",
        ),
        (
            {"parameters": {"epochs": 5, "no_aug_epochs": 5, "mosaic_prob": 0.5}},
            "inactive_parameter",
            "mosaic_prob",
        ),
        (
            {"parameters": {"mixup": 0.2, "mixup_prob": 0.2}},
            "duplicate_parameter_alias",
            "mixup_prob",
        ),
    ],
)
def test_validate_config_cli_rejects_invalid_specs_with_exit_zero(
    tmp_path: Path,
    capsys,
    spec_override: dict,
    expected_code: str,
    expected_field: str,
) -> None:
    base = {
        "task": "detect",
        "architecture_id": "libreyolo9t",
        "dataset_contract": "ultralytics-yolo-1",
        "parameters": {},
    }
    base.update(spec_override)
    spec_path = tmp_path / "invalid.json"
    spec_path.write_text(json.dumps(base), encoding="utf-8")

    rc = main(["validate-config", "--spec", str(spec_path)])
    assert rc == 0
    result = ValidateConfigResultV1.model_validate(json.loads(capsys.readouterr().out))
    assert result.valid is False
    assert any(
        issue.code == expected_code and issue.field == expected_field
        for issue in result.issues
    ), result.issues
