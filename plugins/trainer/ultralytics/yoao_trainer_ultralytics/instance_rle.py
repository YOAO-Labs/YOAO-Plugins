"""Qualified instance-RLE bitmap consumer with bounded instance and semantic targets."""

from __future__ import annotations

from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from yoao_plugin_sdk.trainer import atomic_write_json

from . import InputMismatchError
from .dataset import instance_rle_budget
from .policies import INSTANCE_RLE_LIMITS, MAX_SEMANTIC_CLASS_PIXELS

MAX_LABEL_BYTES = INSTANCE_RLE_LIMITS["max_sample_label_bytes"]


def semantic_target_array(masks: np.ndarray, classes: list[int] | np.ndarray) -> np.ndarray:
    """Match native Format: smallest covering instance, first input on equal-area ties."""
    height, width = masks.shape[-2:]
    result = np.zeros((height, width), dtype=np.float32)
    smallest = np.full((height, width), height * width + 1, dtype=np.int64)
    for mask, native_class in zip(masks, classes, strict=True):
        area = int(mask.sum())
        covered = (mask != 0) & (area < smallest)
        result[covered] = native_class
        smallest[covered] = area
    return result


def semantic_target(masks: np.ndarray, classes: list[int] | np.ndarray) -> Any:
    """Return torch.Tensor when torch is installed, otherwise float32 ndarray."""
    result = semantic_target_array(masks, classes)
    try:
        import torch

        return torch.from_numpy(result)
    except ImportError:
        return result


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode one original-size uint8 mask (column-major Fortran order)."""
    height, width = rle["size"]
    flat = np.empty(height * width, dtype=np.uint8)
    offset = 0
    for index, count in enumerate(rle["counts"]):
        flat[offset : offset + count] = index % 2
        offset += count
    return flat.reshape((height, width), order="F")


def _resize_nearest(array: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(array, size_wh, interpolation=cv2.INTER_NEAREST)
    except ImportError:
        img = Image.fromarray(array)
        return np.asarray(img.resize(size_wh, resample=Image.Resampling.NEAREST))


def _resize_linear(array: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(array, size_wh, interpolation=cv2.INTER_LINEAR)
    except ImportError:
        img = Image.fromarray(array)
        return np.asarray(img.resize(size_wh, resample=Image.Resampling.BILINEAR))


def _pad_letterbox(
    image: np.ndarray,
    *,
    top: int,
    bottom: int,
    left: int,
    right: int,
    value: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    try:
        import cv2

        return cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
        )
    except ImportError:
        h, w, c = image.shape
        out = np.empty((h + top + bottom, w + left + right, c), dtype=image.dtype)
        out[:] = value
        out[top : top + h, left : left + w] = image
        return out


def _bilinear_quarter_threshold(mask: np.ndarray, imgsz: int) -> bool:
    """Check whether bilinear downsample to (imgsz//4, imgsz//4) with align_corners=False retains >0.5 pixels."""
    quarter = imgsz // 4
    try:
        import torch

        target = torch.nn.functional.interpolate(
            torch.from_numpy(mask)[None, None].float(),
            (quarter, quarter),
            mode="bilinear",
            align_corners=False,
        ).gt_(0.5)
        return bool(target.any())
    except ImportError:
        src = mask.astype(np.float32)
        h, w = src.shape
        ys = (np.arange(quarter, dtype=np.float32) + 0.5) * (h / quarter) - 0.5
        xs = (np.arange(quarter, dtype=np.float32) + 0.5) * (w / quarter) - 0.5
        ys = np.clip(ys, 0.0, h - 1.0)
        xs = np.clip(xs, 0.0, w - 1.0)
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.floor(xs).astype(np.int64)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        top = (1.0 - wx) * src[np.ix_(y0, x0)] + wx * src[np.ix_(y0, x1)]
        bot = (1.0 - wx) * src[np.ix_(y1, x0)] + wx * src[np.ix_(y1, x1)]
        interp = (1.0 - wy) * top + wy * bot
        return bool((interp > 0.5).any())


def transform_mask(rle: dict[str, Any], imgsz: int, *, flip: bool = False) -> np.ndarray:
    height, width = rle["size"]
    ratio = min(imgsz / height, imgsz / width)
    resized = (round(width * ratio), round(height * ratio))
    left, top = (imgsz - resized[0]) // 2, (imgsz - resized[1]) // 2
    original = decode_rle(rle)
    scaled = _resize_nearest(original, resized)
    del original
    target = np.zeros((imgsz, imgsz), dtype=np.uint8)
    target[top : top + resized[1], left : left + resized[0]] = scaled
    del scaled
    return np.ascontiguousarray(target[:, ::-1]) if flip else target


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (
        (x0 + x1) / (2 * width),
        (y0 + y1) / (2 * height),
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def read_payload(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_LABEL_BYTES:
        raise InputMismatchError("Frozen RLE sample exceeds encoded object budget")
    return json.loads(path.read_text(encoding="utf-8"))


class InstanceRLEDataset:
    """No original-size mask cache, worker processes, prefetch, or contour stage."""

    rect = False
    cache = False
    mosaic = False

    def __init__(
        self,
        root: str | Path,
        args: Any,
        split: str,
        batch_size: int,
        *,
        augment: bool | None = None,
    ) -> None:
        self.root, self.args, self.split = Path(root), args, split
        self.augment = split == "train" if augment is None else augment
        self.imgsz = args.imgsz
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.semantic = getattr(args, "nms", None) is False
        if (
            self.semantic
            and batch_size * len(manifest["classes"]) * (self.imgsz**2)
            > MAX_SEMANTIC_CLASS_PIXELS
        ):
            raise InputMismatchError(
                "Semantic targets exceed 64M class pixels; reduce batch, classes or imgsz"
            )
        self.samples = [
            sample for sample in manifest["samples"] if sample["split"] == split
        ]
        self.im_files = [
            str(self.root / sample["image_path"]) for sample in self.samples
        ]
        self.labels = []
        for sample in self.samples:
            instance_rle_budget(
                sample["width"], sample["height"], len(sample["instances"])
            )
            self.labels.append(
                {
                    "cls": np.array(
                        [[item["class_index"]] for item in sample["instances"]],
                        dtype=np.float32,
                    ).reshape(-1, 1),
                    "bboxes": np.array(
                        [
                            [
                                (item["bbox_xyxy"][0] + item["bbox_xyxy"][2])
                                / (2 * sample["width"]),
                                (item["bbox_xyxy"][1] + item["bbox_xyxy"][3])
                                / (2 * sample["height"]),
                                (item["bbox_xyxy"][2] - item["bbox_xyxy"][0])
                                / sample["width"],
                                (item["bbox_xyxy"][3] - item["bbox_xyxy"][1])
                                / sample["height"],
                            ]
                            for item in sample["instances"]
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 4),
                    "im_file": str(self.root / sample["image_path"]),
                    "shape": (sample["height"], sample["width"]),
                }
            )
        largest = sorted(
            (len(sample["instances"]) for sample in self.samples), reverse=True
        )[:batch_size]
        if (
            sum(largest) * (self.imgsz**2)
            > INSTANCE_RLE_LIMITS["max_batch_mask_elements"]
        ):
            raise InputMismatchError(
                "RLE batch exceeds 64M training mask elements; reduce batch or imgsz"
            )
        self._hsv = None

    def _apply_hsv(self, image: np.ndarray) -> np.ndarray:
        if self._hsv is None:
            try:
                from ultralytics.data.augment import RandomHSV

                self._hsv = RandomHSV(
                    self.args.hsv_h, self.args.hsv_s, self.args.hsv_v
                )
            except ImportError:
                self._hsv = False
        if self._hsv:
            return self._hsv({"img": image})["img"]
        return image

    def __len__(self) -> int:
        return len(self.samples)

    def close_mosaic(self, hyp: Any = None) -> None:
        pass

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.item(index, augment=self.augment)

    def item(self, index: int, *, augment: bool = False) -> dict[str, Any]:
        sample = self.samples[index]
        payload = read_payload(self.root / sample["label_path"])
        height, width = sample["height"], sample["width"]
        with Image.open(self.im_files[index]) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))[
                :, :, ::-1
            ].copy()
        if image.shape[:2] != (height, width):
            raise InputMismatchError(
                "Frozen raster image dimensions differ from mask coordinates"
            )
        ratio = min(self.imgsz / height, self.imgsz / width)
        resized = (round(width * ratio), round(height * ratio))
        left, top = (self.imgsz - resized[0]) // 2, (self.imgsz - resized[1]) // 2
        image = _resize_linear(image, resized)
        image = _pad_letterbox(
            image,
            top=top,
            bottom=self.imgsz - resized[1] - top,
            left=left,
            right=self.imgsz - resized[0] - left,
            value=(114, 114, 114),
        )
        flip = augment and random.random() < self.args.fliplr
        if augment:
            image = self._apply_hsv(image)
        if flip:
            image = image[:, ::-1]
        if augment and random.random() < self.args.bgr:
            image = image[:, :, ::-1]
        masks = np.empty(
            (len(payload["instances"]), self.imgsz, self.imgsz), dtype=np.uint8
        )
        boxes: list[tuple[float, float, float, float]] = []
        classes: list[list[int]] = []
        for position, instance in enumerate(payload["instances"]):
            mask = transform_mask(instance["rle"], self.imgsz, flip=flip)
            box = mask_box(mask)
            if box is None:
                raise InputMismatchError(
                    f"Instance {instance['annotation_id']} vanishes at training resolution"
                )
            masks[position] = mask
            del mask
            boxes.append(box)
            classes.append([instance["class_index"]])
        count = len(boxes)
        img_chw = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        sem = semantic_target(masks, [item[0] for item in classes])
        try:
            import torch

            return {
                "img": torch.from_numpy(img_chw),
                "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
                "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "masks": torch.from_numpy(masks),
                "sem_masks": sem,
                "batch_idx": torch.zeros(count),
                "im_file": self.im_files[index],
                "ori_shape": (height, width),
                "resized_shape": (self.imgsz, self.imgsz),
                "ratio_pad": ((resized[1] / height, resized[0] / width), (left, top)),
            }
        except ImportError:
            return {
                "img": img_chw,
                "cls": np.array(classes, dtype=np.float32).reshape(-1, 1),
                "bboxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "masks": masks,
                "sem_masks": sem,
                "batch_idx": np.zeros(count, dtype=np.float32),
                "im_file": self.im_files[index],
                "ori_shape": (height, width),
                "resized_shape": (self.imgsz, self.imgsz),
                "ratio_pad": ((resized[1] / height, resized[0] / width), (left, top)),
            }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        from ultralytics.data.dataset import YOLODataset

        elements = sum(item["masks"].numel() for item in batch)
        if elements > INSTANCE_RLE_LIMITS["max_batch_mask_elements"]:
            raise InputMismatchError("RLE batch exceeds 64M training mask elements")
        return YOLODataset.collate_fn(batch)

    def preflight(self) -> dict[str, Any]:
        """Stream originals; detect loss of tiny instances before optimization."""
        vanished: list[dict[str, Any]] = []
        validation_vanished: list[dict[str, Any]] = []
        vanished_count = 0
        validation_vanished_count = 0
        for sample in self.samples:
            payload = read_payload(self.root / sample["label_path"])
            for instance in payload["instances"]:
                mask = transform_mask(instance["rle"], self.imgsz)
                identity = {
                    "task_id": sample["task_id"],
                    "annotation_id": instance["annotation_id"],
                    "annotation_version": instance["annotation_version"],
                }
                if not mask.any():
                    vanished_count += 1
                    if len(vanished) < 1000:
                        vanished.append(identity)
                elif not _bilinear_quarter_threshold(mask, self.imgsz):
                    validation_vanished_count += 1
                    if len(validation_vanished) < 1000:
                        validation_vanished.append(identity)
        return {
            "samples": len(self.samples),
            "resize_vanished": vanished,
            "resize_vanished_count": vanished_count,
            "validator_vanished_count": validation_vanished_count,
            "detail_limit": 1000,
            "validator_vanished": validation_vanished,
            "mask_plane": [self.imgsz, self.imgsz],
            "validator_plane": [self.imgsz // 4, self.imgsz // 4],
        }


def build_instance_rle_validator_class() -> type:
    from ultralytics.models.yolo.segment import SegmentationValidator

    class InstanceRLEValidator(SegmentationValidator):
        def build_dataset(self, img_path, mode="val", batch=None):
            return InstanceRLEDataset(
                Path(img_path).parent.parent,
                self.args,
                self.args.split,
                batch or self.args.batch,
                augment=False,
            )

    return InstanceRLEValidator


def build_instance_rle_trainer_class() -> type:
    from torch.utils.data import DataLoader
    from ultralytics.models.yolo.segment import SegmentationTrainer

    rle_validator_cls = build_instance_rle_validator_class()

    class InstanceRLETrainer(SegmentationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            return InstanceRLEDataset(
                Path(img_path).parent.parent,
                self.args,
                mode,
                batch or self.batch_size,
            )

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if rank not in {-1, 0} or self.args.workers != 0:
                raise InputMismatchError(
                    "RLE first revision requires single-device workers=0"
                )
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            evidence = dataset.preflight()
            atomic_write_json(self.save_dir / f"rle-{mode}-preflight.json", evidence)
            atomic_write_json(self.save_dir.parent / f"rle-{mode}-preflight.json", evidence)
            if evidence["resize_vanished"]:
                raise InputMismatchError(
                    "Instances disappear at selected input resolution; inspect RLE preflight"
                )
            return DataLoader(
                dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=mode == "train",
                num_workers=0,
                pin_memory=False,
                collate_fn=dataset.collate_fn,
            )

        def get_validator(self):
            return rle_validator_cls(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    return InstanceRLETrainer


def consumer_evidence() -> dict[str, Any]:
    try:
        from ultralytics.models.yolo.segment import SegmentationValidator
        from ultralytics.utils.loss import v8SegmentationLoss

        loss_sha = hashlib.sha256(
            inspect.getsource(v8SegmentationLoss.loss).encode("utf-8")
        ).hexdigest()
        validator_sha = hashlib.sha256(
            inspect.getsource(SegmentationValidator._prepare_batch).encode("utf-8")
        ).hexdigest()
    except ImportError:
        plugin_src = Path(__file__).read_bytes()
        loss_sha = hashlib.sha256(plugin_src).hexdigest()
        validator_sha = loss_sha

    return {
        "bridge": "ultralytics-instance-rle-1",
        "workers": 0,
        "prefetch": False,
        "transform": "aap-letterbox-nearest-1",
        "rasterizer": "aap-pixel-center-even-odd-1",
        "loss_mask_processing": "truth-full-input-plane;prototype-bilinear-upsample-align-corners-false",
        "validator_mask_processing": "truth-bilinear-align-corners-false-threshold-0.5-at-input-quarter",
        "loss_source_sha256": loss_sha,
        "validator_source_sha256": validator_sha,
        "limits": dict(INSTANCE_RLE_LIMITS),
        "semantic_target": "smallest-covering-instance;first-input-on-equal-area;background-zero",
        "max_semantic_class_pixels": MAX_SEMANTIC_CLASS_PIXELS,
    }


def __getattr__(name: str) -> Any:
    if name == "InstanceRLEValidator":
        return build_instance_rle_validator_class()
    if name == "InstanceRLETrainer":
        return build_instance_rle_trainer_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
