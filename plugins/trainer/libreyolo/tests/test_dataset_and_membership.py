"""Tests for frozen dataset validation, non-mutating data.yaml, membership receipts, and metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import pytest
import yaml
from yoao_plugin_sdk.trainer import (
    POLICY_DIGEST,
    InputMembership,
    InputMismatchError,
    canonical_json_digest,
    digest_file,
)

from yoao_trainer_libreyolo.dataset import validate_dataset, write_native_dataset_yaml
from yoao_trainer_libreyolo.evaluation import libreyolo_report


def _make_dataset(tmp_path: Path, *, with_task_ids: bool = True) -> tuple[Path, dict, list[str]]:
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    classes = ["car", "person"]
    (ds_dir / "data.yaml").write_text(
        'train: images/train\nval: images/val\nnames: {"0": "car", "1": "person"}\n',
        encoding="utf-8",
    )
    (ds_dir / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    samples = []
    for idx, (split, label_text) in enumerate(
        [("train", ""), ("val", "0 0.5 0.5 0.2 0.2\n")], start=1
    ):
        img = ds_dir / f"images/{split}/sample_{idx}.png"
        lbl = ds_dir / f"labels/{split}/sample_{idx}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(f"image-bytes-{split}-{idx}".encode("utf-8"))
        lbl.write_text(label_text, encoding="utf-8")
        row = {
            "split": split,
            "image_path": f"images/{split}/sample_{idx}.png",
            "label_path": f"labels/{split}/sample_{idx}.txt",
            "sha256": digest_file(img),
            "label_sha256": digest_file(lbl),
            "width": 64,
            "height": 64,
            "size_bytes": img.stat().st_size,
        }
        if with_task_ids:
            row["task_id"] = str(uuid.UUID(int=idx))
        samples.append(row)
    manifest = {
        "schema": 3,
        "task": "detect",
        "media_type": "image",
        "format": "ultralytics-yolo",
        "geometry": "normalized-bbox",
        "bridge": "ultralytics-bbox-1",
        "classes": classes,
        "validation_digest": "e" * 64,
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": "f" * 64,
        "samples": samples,
    }
    manifest["content_digest"] = canonical_json_digest(manifest)
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds_dir, manifest, classes


def test_validate_dataset_accepts_explicit_negatives_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    ds_dir, manifest, classes = _make_dataset(tmp_path)
    classes_digest = canonical_json_digest(classes)

    loaded_manifest, loaded_classes = validate_dataset(
        ds_dir,
        expected_classes_digest=classes_digest,
        expected_snapshot_digest=manifest["content_digest"],
    )
    assert loaded_classes == classes
    assert len(loaded_manifest["samples"]) == 2

    # Wrong classes_digest fails.
    with pytest.raises(InputMismatchError, match="classes digest"):
        validate_dataset(ds_dir, expected_classes_digest="0" * 64)

    # Missing label fails.
    train_label = ds_dir / "labels/train/sample_1.txt"
    train_label.unlink()
    with pytest.raises(InputMismatchError, match="missing or changed"):
        validate_dataset(ds_dir, expected_classes_digest=classes_digest)
    train_label.write_text("", encoding="utf-8")

    # Changed class order fails.
    (ds_dir / "classes.json").write_text(json.dumps(["person", "car"]), encoding="utf-8")
    with pytest.raises(InputMismatchError, match="differs from dataset classes"):
        validate_dataset(ds_dir, expected_classes_digest=classes_digest)
    (ds_dir / "classes.json").write_text(json.dumps(classes), encoding="utf-8")

    # Segmentation manifest is rejected.
    seg_manifest = {
        **manifest,
        "task": "segment",
        "geometry": "normalized-simple-polygon",
        "bridge": "ultralytics-simple-polygon-1",
    }
    (ds_dir / "manifest.json").write_text(json.dumps(seg_manifest), encoding="utf-8")
    with pytest.raises(InputMismatchError, match="detection"):
        validate_dataset(ds_dir, expected_classes_digest=classes_digest)


def test_write_native_dataset_yaml_does_not_mutate_readonly_dataset_dir(
    tmp_path: Path,
) -> None:
    ds_dir, _manifest, classes = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    original_yaml = (ds_dir / "data.yaml").read_bytes()

    os.chmod(ds_dir, 0o555)
    try:
        native_yaml = write_native_dataset_yaml(ds_dir, output_dir, classes)
        assert native_yaml == output_dir / "libreyolo-data.yaml"
        parsed = yaml.safe_load(native_yaml.read_text(encoding="utf-8"))
        assert parsed["names"] == ["car", "person"]
        assert parsed["train"] == str(ds_dir.resolve() / "images/train")
        assert parsed["val"] == str(ds_dir.resolve() / "images/val")
        assert (ds_dir / "data.yaml").read_bytes() == original_yaml
        assert not (ds_dir / "libreyolo-data.yaml").exists()
    finally:
        os.chmod(ds_dir, 0o755)


def test_input_membership_produces_canonical_receipt_and_rejects_bad_paths(
    tmp_path: Path,
) -> None:
    ds_dir, manifest, _classes = _make_dataset(tmp_path, with_task_ids=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    config_digest = "c" * 64

    observer = InputMembership(
        dataset_dir=ds_dir,
        output_dir=output_dir,
        manifest=manifest,
        run_id=run_id,
        attempt_id=attempt_id,
        config_digest=config_digest,
        dataset_snapshot_digest=manifest["content_digest"],
    )
    observer.observe("train", [str(ds_dir / "images/train/sample_1.png")])
    observer.observe("val", [ds_dir / "images/val/sample_2.png"])
    membership_result = observer.complete()

    assert membership_result is not None
    receipt_path = membership_result.path
    assert receipt_path == output_dir / "input-membership.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == 1
    assert receipt["policy_digest"] == POLICY_DIGEST
    assert receipt["run_id"] == run_id
    assert receipt["attempt_id"] == attempt_id
    assert receipt["config_digest"] == config_digest
    assert receipt["dataset_id"] == manifest["dataset_id"]
    assert receipt["dataset_sha256"] == manifest["dataset_sha256"]
    assert receipt["counts"] == {"train": 1, "val": 1}
    assert receipt["members_digest"] == canonical_json_digest(receipt["members"])

    # Rejects missing / extra / wrong-split / outside-root paths.
    bad_observer = InputMembership(
        dataset_dir=ds_dir,
        output_dir=tmp_path / "bad_out",
        manifest=manifest,
        run_id=run_id,
        attempt_id=attempt_id,
        config_digest=config_digest,
        dataset_snapshot_digest=manifest["content_digest"],
    )
    for bad_paths in (
        [],
        [ds_dir / "images/val/sample_2.png"],
        [ds_dir / "images/train/sample_1.png", ds_dir / "images/train/sample_1.png"],
        [tmp_path / "outside.png"],
    ):
        with pytest.raises(InputMismatchError):
            bad_observer.observe("train", bad_paths)
    with pytest.raises(InputMismatchError, match="incomplete"):
        bad_observer.complete()


def test_libreyolo_report_strips_precision_recall_aliases() -> None:
    raw = {
        "metrics/precision": 0.31,
        "metrics/recall": 0.62,
        "metrics/mAP50-95": 0.31,
        "metrics/mAP50": 0.54,
        "metrics/mAP75": 0.22,
        "metrics/AR100": 0.62,
    }
    report = libreyolo_report(
        raw,
        batch=4,
        imgsz=320,
        weights_sha256="d" * 64,
        validation_digest="e" * 64,
        backend="pycocotools",
    )
    assert report["summary"] == {
        "metrics/mAP50-95": 0.31,
        "metrics/mAP50": 0.54,
        "metrics/mAP75": 0.22,
        "metrics/AR100": 0.62,
    }
    assert "metrics/precision" not in report["summary"]
    assert "metrics/recall" not in report["summary"]
    assert report["native_raw"] == raw
    assert report["per_class"] == []
    assert report["per_class_available"] is False
    assert report["evaluation"]["weights_sha256"] == "d" * 64
    assert report["evaluation"]["evaluator"] == "pycocotools"

    with pytest.raises(ValueError, match="incomplete"):
        libreyolo_report(
            {"metrics/precision": 0.3},
            batch=4,
            imgsz=320,
            weights_sha256="d" * 64,
            validation_digest="e" * 64,
            backend=None,
        )
