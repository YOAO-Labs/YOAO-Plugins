"""Shared synthetic dataset, spec, and mocked-ML fixtures for Ultralytics trainer tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import uuid

import numpy as np
from PIL import Image
import pytest
import yaml

from yoao_plugin_sdk.trainer import canonical_json_digest, digest_file

from yoao_trainer_ultralytics.dataset import (
    DETECTION_CONTRACT,
    INSTANCE_RLE_CONTRACT,
    instance_rle_evidence,
)


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2D binary uint8 mask to canonical column-major COCO RLE."""
    height, width = mask.shape
    flat = np.asarray(mask != 0, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    expected = 0
    run = 0
    for val in flat.tolist():
        if val == expected:
            run += 1
        else:
            counts.append(run)
            expected = 1 - expected
            run = 1
    counts.append(run)
    return {"encoding": "coco_rle", "size": [height, width], "counts": counts}


def write_png(path: Path, width: int = 32, height: int = 32, color: tuple[int, int, int] = (80, 120, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path, format="PNG")
    return path


def make_detect_dataset(root: Path, *, classes: list[str] | None = None) -> dict[str, Any]:
    classes = classes or ["cat", "dog"]
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for split, idx in (("train", 0), ("val", 1)):
        img_rel = f"images/{split}/sample_{idx}.png"
        lbl_rel = f"labels/{split}/sample_{idx}.txt"
        img_path = write_png(root / img_rel, 32, 32, (40 + idx * 40, 100, 150))
        lbl_path = root / lbl_rel
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        samples.append(
            {
                "task_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{split}-{idx}")),
                "split": split,
                "image_path": img_rel,
                "label_path": lbl_rel,
                "sha256": digest_file(img_path),
                "label_sha256": digest_file(lbl_path),
                "width": 32,
                "height": 32,
                "size_bytes": img_path.stat().st_size,
            }
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": {i: c for i, c in enumerate(classes)}}
        ),
        encoding="utf-8",
    )
    (root / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    manifest = {
        **DETECTION_CONTRACT,
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": "a" * 64,
        "validation_digest": "b" * 64,
        "classes": classes,
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "root": root,
        "manifest": manifest,
        "classes": classes,
        "classes_digest": canonical_json_digest(classes),
    }


def make_rle_dataset(
    root: Path,
    *,
    classes: list[str] | None = None,
    width: int = 32,
    height: int = 32,
    masks_by_split: dict[str, list[tuple[int, np.ndarray]]] | None = None,
) -> dict[str, Any]:
    classes = classes or ["person", "car"]
    root.mkdir(parents=True, exist_ok=True)
    if masks_by_split is None:
        base_mask = np.zeros((height, width), dtype=np.uint8)
        base_mask[4:28, 4:28] = 1
        masks_by_split = {
            "train": [(0, base_mask)],
            "val": [(0, base_mask)],
        }

    samples: list[dict[str, Any]] = []
    for idx, (split, entries) in enumerate(masks_by_split.items()):
        img_rel = f"images/{split}/sample_{idx}.png"
        lbl_rel = f"labels/{split}/sample_{idx}.json"
        img_path = write_png(root / img_rel, width, height, (50 + idx * 30, 90, 140))
        payload_instances: list[dict[str, Any]] = []
        manifest_instances: list[dict[str, Any]] = []
        semantics: list[tuple[int, str]] = []
        for inst_idx, (class_idx, mask) in enumerate(entries):
            rle = encode_rle(mask)
            ev = instance_rle_evidence(rle)
            meta = {
                "annotation_id": f"ann-{split}-{inst_idx}",
                "annotation_version": 1,
                "class_index": class_idx,
                "source_geometry": "raster_mask",
                **ev,
            }
            manifest_instances.append(meta)
            payload_instances.append({**meta, "rle": rle})
            semantics.append((class_idx, ev["pixel_sha256"]))
        lbl_path = root / lbl_rel
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text(
            json.dumps({"schema": 1, "size": [height, width], "instances": payload_instances}),
            encoding="utf-8",
        )
        samples.append(
            {
                "task_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rle-{split}-{idx}")),
                "split": split,
                "image_path": img_rel,
                "label_path": lbl_rel,
                "sha256": digest_file(img_path),
                "label_sha256": digest_file(lbl_path),
                "width": width,
                "height": height,
                "size_bytes": img_path.stat().st_size,
                "instances": manifest_instances,
                "semantic_label_sha256": canonical_json_digest(sorted(semantics)),
            }
        )

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": {i: c for i, c in enumerate(classes)}}
        ),
        encoding="utf-8",
    )
    (root / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    manifest = {
        **INSTANCE_RLE_CONTRACT,
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": "c" * 64,
        "validation_digest": "d" * 64,
        "data_contract": dict(INSTANCE_RLE_CONTRACT),
        "classes": classes,
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "root": root,
        "manifest": manifest,
        "classes": classes,
        "classes_digest": canonical_json_digest(classes),
    }


def make_fake_fonts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    ascii_font = tmp_path / "DejaVuSans.ttf"
    unicode_font = tmp_path / "NotoSansCJK-Regular.ttc"
    ascii_font.write_bytes(b"fake-dejavu-font")
    unicode_font.write_bytes(b"fake-noto-font")
    monkeypatch.setenv("YOAO_ULTRALYTICS_ASCII_FONT", str(ascii_font))
    monkeypatch.setenv("YOAO_ULTRALYTICS_UNICODE_FONT", str(unicode_font))
    return ascii_font, unicode_font


def install_fake_ultralytics_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "8.4.183",
) -> dict[str, Any]:
    """Install minimal fake ultralytics namespace modules into sys.modules."""
    settings_log: list[dict[str, Any]] = []

    class FakeSettings:
        def update(self, mapping: dict[str, Any]) -> None:
            settings_log.append(dict(mapping))

    ultralytics_mod = ModuleType("ultralytics")
    ultralytics_mod.__version__ = version  # type: ignore[attr-defined]
    ultralytics_mod.settings = FakeSettings()  # type: ignore[attr-defined]
    ultralytics_mod.__path__ = []  # type: ignore[attr-defined]

    data_mod = ModuleType("ultralytics.data")
    data_mod.__path__ = []  # type: ignore[attr-defined]
    data_utils_mod = ModuleType("ultralytics.data.utils")
    data_utils_mod.check_font = lambda font="Arial.ttf": Path("/broken")  # type: ignore[attr-defined]

    utils_mod = ModuleType("ultralytics.utils")
    utils_mod.__path__ = []  # type: ignore[attr-defined]
    checks_mod = ModuleType("ultralytics.utils.checks")
    checks_mod.check_font = lambda font="Arial.ttf": Path("/broken")  # type: ignore[attr-defined]
    plotting_mod = ModuleType("ultralytics.utils.plotting")
    plotting_mod.check_font = lambda font="Arial.ttf": Path("/broken")  # type: ignore[attr-defined]

    ultralytics_mod.data = data_mod  # type: ignore[attr-defined]
    ultralytics_mod.utils = utils_mod  # type: ignore[attr-defined]
    data_mod.utils = data_utils_mod  # type: ignore[attr-defined]
    utils_mod.checks = checks_mod  # type: ignore[attr-defined]
    utils_mod.plotting = plotting_mod  # type: ignore[attr-defined]

    for name, mod in (
        ("ultralytics", ultralytics_mod),
        ("ultralytics.data", data_mod),
        ("ultralytics.data.utils", data_utils_mod),
        ("ultralytics.utils", utils_mod),
        ("ultralytics.utils.checks", checks_mod),
        ("ultralytics.utils.plotting", plotting_mod),
    ):
        monkeypatch.setitem(__import__("sys").modules, name, mod)

    return {
        "ultralytics": ultralytics_mod,
        "checks": checks_mod,
        "data_utils": data_utils_mod,
        "plotting": plotting_mod,
        "settings_log": settings_log,
    }


def make_fake_tensor(array: np.ndarray) -> Any:
    """Minimal torch-like tensor wrapper backed by numpy for CPU tests."""
    arr = np.asarray(array)

    class _FakeTensor:
        def __init__(self, data: np.ndarray) -> None:
            self._data = data
            self.shape = data.shape
            self.dtype = f"torch.{data.dtype}"

        def detach(self) -> "_FakeTensor":
            return self

        def cpu(self) -> "_FakeTensor":
            return self

        def contiguous(self) -> "_FakeTensor":
            return _FakeTensor(np.ascontiguousarray(self._data))

        def is_floating_point(self) -> bool:
            return bool(np.issubdtype(self._data.dtype, np.floating))

        def float(self) -> "_FakeTensor":
            return _FakeTensor(self._data.astype(np.float32))

        def numpy(self) -> np.ndarray:
            return self._data

        def numel(self) -> int:
            return int(self._data.size)

        def clamp(self, lo: float, hi: float) -> "_FakeTensor":
            return _FakeTensor(np.clip(self._data, lo, hi))

        def __mul__(self, other: float) -> "_FakeTensor":
            return _FakeTensor(self._data * other)

        def round(self) -> "_FakeTensor":
            return _FakeTensor(np.round(self._data))

        def byte(self) -> "_FakeTensor":
            return _FakeTensor(self._data.astype(np.uint8))

        def permute(self, *dims: int) -> "_FakeTensor":
            return _FakeTensor(np.transpose(self._data, dims))

        def view(self, *shape: int) -> "_FakeTensor":
            return _FakeTensor(self._data.reshape(shape))

        def tolist(self) -> Any:
            return self._data.tolist()

        def __getitem__(self, item: Any) -> "_FakeTensor":
            return _FakeTensor(self._data[item])

        def max(self) -> int:
            return int(self._data.max())

    return _FakeTensor(arr)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
