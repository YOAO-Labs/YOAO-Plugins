"""Instance-RLE semantic target, transform, budget, and preflight tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yoao_trainer_ultralytics import DatasetContractError, InputMismatchError
from yoao_trainer_ultralytics.dataset import validate_dataset
from yoao_trainer_ultralytics.instance_rle import (
    InstanceRLEDataset,
    decode_rle,
    mask_box,
    semantic_target_array,
    transform_mask,
)
from tests.conftest import encode_rle, make_rle_dataset


def test_semantic_target_uses_smallest_area_and_first_input_tie_break() -> None:
    large = np.zeros((8, 8), dtype=np.uint8)
    large[1:7, 1:7] = 1  # area 36, class 2
    small = np.zeros((8, 8), dtype=np.uint8)
    small[2:5, 2:5] = 1  # area 9, class 5
    tie_a = np.zeros((8, 8), dtype=np.uint8)
    tie_a[5:7, 5:7] = 1  # area 4, class 3 (first)
    tie_b = np.zeros((8, 8), dtype=np.uint8)
    tie_b[5:7, 5:7] = 1  # area 4, class 1 (second, equal area)

    sem = semantic_target_array(
        np.stack([large, small, tie_a, tie_b], axis=0), [2, 5, 3, 1]
    )
    assert sem.dtype == np.float32
    assert sem[0, 0] == 0.0  # uncovered background is zero
    assert sem[1, 1] == 2.0  # covered only by large
    assert sem[3, 3] == 5.0  # smaller instance overrides larger covering instance
    assert sem[5, 5] == 3.0  # equal-area tie keeps first input


def test_decode_rle_transform_mask_and_mask_box_roundtrip() -> None:
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[4:12, 0:8] = 1
    rle = encode_rle(mask)
    decoded = decode_rle(rle)
    np.testing.assert_array_equal(decoded, mask)

    transformed = transform_mask(rle, 32, flip=False)
    assert transformed.shape == (32, 32)
    # 16x32 letterboxed into 32x32 adds top=8, bottom=8 padding
    assert transformed[12:20, 0:8].all()
    assert not transformed[:12, :].any()

    flipped = transform_mask(rle, 32, flip=True)
    assert flipped[12:20, 24:32].all()
    assert not flipped[:, :24].any()

    box = mask_box(transformed)
    assert box is not None
    assert pytest.approx(box, abs=1e-6) == (0.125, 0.5, 0.25, 0.25)
    assert mask_box(np.zeros((32, 32), dtype=np.uint8)) is None


def test_rle_dataset_validates_budgets_items_and_preflight_vanishing(
    tmp_path: Path,
) -> None:
    # 1. Valid dataset + item extraction
    ds_info = make_rle_dataset(tmp_path / "valid", width=64, height=64)
    manifest, classes, _ = validate_dataset(
        ds_info["root"],
        task="segment",
        dataset_contract="ultralytics-instance-rle-1",
        expected_classes_digest=ds_info["classes_digest"],
    )
    assert classes == ["person", "car"]
    assert len(manifest["samples"]) == 2

    args = SimpleNamespace(
        imgsz=64,
        nms=False,
        fliplr=0.0,
        bgr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
    ds = InstanceRLEDataset(ds_info["root"], args, "train", batch_size=1)
    item = ds[0]
    assert item["img"].shape == (3, 64, 64)
    assert item["masks"].shape == (1, 64, 64)
    assert item["sem_masks"].shape == (64, 64)
    pf = ds.preflight()
    assert pf["resize_vanished_count"] == 0
    assert pf["validator_vanished_count"] == 0

    # 2. Semantic class pixel budget (>64M) rejected when nms=False
    huge_args = SimpleNamespace(imgsz=1280, nms=False)
    with pytest.raises(InputMismatchError, match="64M class pixels"):
        InstanceRLEDataset(ds_info["root"], huge_args, "train", batch_size=20)

    # 3. Max batch mask elements (>64M) rejected
    big_masks_root = tmp_path / "batch_overflow"
    big_mask = np.zeros((32, 32), dtype=np.uint8)
    big_mask[2:30, 2:30] = 1
    make_rle_dataset(
        big_masks_root,
        width=32,
        height=32,
        masks_by_split={"train": [(0, big_mask)] * 40, "val": [(0, big_mask)]},
    )
    with pytest.raises(InputMismatchError, match="64M training mask elements"):
        InstanceRLEDataset(
            big_masks_root,
            SimpleNamespace(imgsz=1280, nms=None),
            "train",
            batch_size=1,
        )

    # 4. Preflight detects validator_vanished (1px mask at imgsz=64 disappears at 16x16 bilinear >0.5)
    tiny_mask = np.zeros((64, 64), dtype=np.uint8)
    tiny_mask[32, 32] = 1
    tiny_info = make_rle_dataset(
        tmp_path / "val_vanish",
        width=64,
        height=64,
        masks_by_split={"train": [(0, tiny_mask)], "val": [(0, tiny_mask)]},
    )
    tiny_ds = InstanceRLEDataset(
        tiny_info["root"], SimpleNamespace(imgsz=64, nms=None), "train", batch_size=1
    )
    tiny_pf = tiny_ds.preflight()
    assert tiny_pf["resize_vanished_count"] == 0
    assert tiny_pf["validator_vanished_count"] == 1

    # 5. Preflight detects resize_vanished (1px mask at 256x256 downsampled to 32x32 nearest)
    vanish_mask = np.zeros((256, 256), dtype=np.uint8)
    vanish_mask[1, 1] = 1
    vanish_info = make_rle_dataset(
        tmp_path / "resize_vanish",
        width=256,
        height=256,
        masks_by_split={"train": [(0, vanish_mask)], "val": [(0, vanish_mask)]},
    )
    vanish_ds = InstanceRLEDataset(
        vanish_info["root"], SimpleNamespace(imgsz=32, nms=None), "train", batch_size=1
    )
    vanish_pf = vanish_ds.preflight()
    assert vanish_pf["resize_vanished_count"] == 1


def test_validate_dataset_rejects_corrupt_rle_semantic_digest(tmp_path: Path) -> None:
    ds_info = make_rle_dataset(tmp_path / "corrupt")
    manifest_path = ds_info["root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["semantic_label_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match="semantic label digest mismatch"):
        validate_dataset(
            ds_info["root"],
            task="segment",
            dataset_contract="ultralytics-instance-rle-1",
            expected_classes_digest=ds_info["classes_digest"],
        )
