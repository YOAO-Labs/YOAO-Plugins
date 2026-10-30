"""Independent positive open-vocabulary workloads for GPU lifecycle qualification."""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from aap_backend_runtime import artifact_evidence


INPUTS = {
    "image": "VALIDATION_OPEN_IMAGE",
    "world": "VALIDATION_OPEN_WORLD_BASELINE",
    "yoloe": "VALIDATION_OPEN_YOLOE_BASELINE",
    "exemplar": "VALIDATION_OPEN_EXEMPLAR_BASELINE",
}
WEIGHTS = {
    "yolov8s-worldv2.pt": "yolov8s-worldv2.pt",
    "yoloe-11s-seg.pt": "yoloe-11s-seg.pt",
    "mobileclip_blt.ts": "mobileclip_blt.ts",
    "ViT-B-32.pt": "weights/clip/ViT-B-32.pt",
}


def _required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _mapping(name, keys):
    value = json.loads(_required(name))
    if not isinstance(value, dict) or set(value) != set(keys) or any(
        not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
        for sha in value.values()
    ):
        raise ValueError(f"{name} must contain the exact approved SHA-256 mapping")
    return value


def _iou_box(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0


def _polygon_iou(left, right, width, height):
    masks = []
    for points in (left, right):
        if len(points) < 3 or any(len(p) != 2 or not 0 <= p[0] <= width or not 0 <= p[1] <= height for p in points):
            raise ValueError("Open-vocabulary polygon exceeds the original image")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(np.asarray(points)).astype(np.int32)], 1)
        masks.append(mask)
    intersection = np.count_nonzero(masks[0] & masks[1])
    union = np.count_nonzero(masks[0] | masks[1])
    return intersection / union if union else 0


def compare_output(result, baseline, *, output, normalized, dimensions):
    """Compare all positive HTTP geometry to pre-frozen native expectations."""
    width, height = dimensions
    rows = baseline["results"]
    if not rows or not isinstance(result, list):
        raise AssertionError("Open-vocabulary output must be positive")
    boxes = [item for item in result if item.get("type") == "rectanglelabels"]
    polygons = [item for item in result if item.get("type") == "polygonlabels"]
    expected_boxes = rows if output in {"box", "both"} else []
    expected_polygons = [row for row in rows if len(row.get("polygon_xy", [])) >= 3] if output in {"mask", "both"} else []
    if output in {"mask", "both"} and not expected_polygons:
        raise AssertionError("Open-vocabulary native baseline must contain a positive polygon")
    if len(boxes) != len(expected_boxes) or len(polygons) != len(expected_polygons) or len(result) != len(boxes) + len(polygons):
        raise AssertionError("Open-vocabulary result counts differ from native baseline")
    scale = 1 if normalized else 100
    for item, expected in zip(boxes, expected_boxes, strict=True):
        value = item["value"]
        actual = [value["x"] * width / scale, value["y"] * height / scale, (value["x"] + value["width"]) * width / scale, (value["y"] + value["height"]) * height / scale]
        if item.get("native_class_index") != expected["class_index"] or value.get("rectanglelabels") != [expected["class_name"]] or not math.isfinite(item["score"]) or abs(item["score"] - expected["score"]) > 0.1 or any(not math.isfinite(v) for v in actual) or not (0 <= actual[0] < actual[2] <= width and 0 <= actual[1] < actual[3] <= height) or _iou_box(actual, expected["box_xyxy"]) < 0.9:
            raise AssertionError("Open-vocabulary box differs from native baseline")
    for item, expected in zip(polygons, expected_polygons, strict=True):
        value = item["value"]
        points = [[x * width / scale, y * height / scale] for x, y in value["points"]]
        if value.get("polygonlabels") != [expected["class_name"]] or not math.isfinite(item["score"]) or abs(item["score"] - expected["score"]) > 0.1 or _polygon_iou(points, expected["polygon_xy"], width, height) < 0.9:
            raise AssertionError("Open-vocabulary polygon differs from native baseline")
    return {"boxes": len(boxes), "polygons": len(polygons)}


class EncoderProbe:
    """Observe actual ephemeral text encoder devices without changing inference."""

    def __init__(self):
        from ultralytics.nn.text_model import CLIP, MobileCLIPTS

        self.events = []
        self.originals = []
        for cls, name in ((CLIP, "clip"), (MobileCLIPTS, "mobileclip")):
            original = cls.encode_text
            self.originals.append((cls, original))

            def observed(instance, tokens, *args, _original=original, _name=name, **kwargs):
                output = _original(instance, tokens, *args, **kwargs)
                self.events.append({"encoder": _name, "model_device": str(instance.device), "input_device": str(tokens.device), "output_device": str(output.device)})
                return output

            cls.encode_text = observed

    def close(self):
        for cls, original in self.originals:
            cls.encode_text = original


class OpenVocabularyFixture:
    def __init__(self, root, checkpoint_dir, model_assets_dir):
        expected_inputs = _mapping("VALIDATION_OPEN_FIXTURE_SHA256_JSON", INPUTS)
        expected_weights = _mapping("VALIDATION_OPEN_WEIGHT_SHA256_JSON", WEIGHTS)
        self.model_approval = _required("VALIDATION_OPEN_APPROVAL_REF")
        self.fixture_approval = _required("VALIDATION_FIXTURE_APPROVAL_REF")
        self.root = root / "open-inputs"
        self.root.mkdir()
        self.inputs = {}
        self.artifacts = []
        for role, env in INPUTS.items():
            source = Path(_required(env))
            dest = self.root / ("image.bin" if role == "image" else f"{role}.json")
            shutil.copyfile(source, dest)
            evidence = artifact_evidence(dest, kind="fixture", approval_ref=self.fixture_approval)
            if evidence["sha256"] != expected_inputs[role]:
                raise ValueError("Frozen open-vocabulary input differs from approval")
            self.artifacts.append(evidence)
            self.inputs[role] = dest
        with Image.open(self.inputs["image"]) as image:
            self.dimensions = image.size
            image.verify()
        self.baselines = {role: json.loads(self.inputs[role].read_bytes()) for role in ("world", "yoloe", "exemplar")}
        if self.baselines["world"].get("text") != "car" or self.baselines["yoloe"].get("text") != "car":
            raise ValueError("Expected frozen positive car text baselines")
        if self.baselines["world"].get("image_sha256") != expected_inputs["image"] or self.baselines["yoloe"].get("image_sha256") != expected_inputs["image"]:
            raise ValueError("Open-vocabulary baseline image identity differs")
        from training_contract.assets import resolve_prepared_asset
        from ultralytics.utils import WEIGHTS_DIR
        from yolo_plugin.catalog import catalog_bytes

        catalog = {item["filename"]: item for item in json.loads(catalog_bytes())["inference_assets"]}

        encoder_root = Path(WEIGHTS_DIR)
        if not encoder_root.is_absolute():
            encoder_root = checkpoint_dir / encoder_root
        if (encoder_root / "clip/ViT-B-32.pt").resolve() != (checkpoint_dir / WEIGHTS["ViT-B-32.pt"]).resolve():
            raise ValueError("CLIP encoder location differs from selected weight inventory")
        self.placement = {}
        self.weight_artifacts = []
        for name, relative in WEIGHTS.items():
            evidence = artifact_evidence(checkpoint_dir / relative, kind="weight", approval_ref=self.model_approval)
            if evidence["sha256"] != expected_weights[name] or evidence["sha256"] != catalog[name]["sha256"] or evidence["size_bytes"] != catalog[name]["size_bytes"]:
                raise ValueError("Open-vocabulary weight differs from approval")
            if name in {"yolov8s-worldv2.pt", "yoloe-11s-seg.pt"}:
                prepared = resolve_prepared_asset(model_assets_dir, name)
                if prepared is not None and artifact_evidence(prepared, kind="weight")["sha256"] != evidence["sha256"]:
                    raise ValueError("Prepared open-vocabulary model differs from selected weights")
                self.placement[name] = "prepared" if prepared is not None else "checkpoint"
            else:
                self.placement[name] = "checkpoint"
            self.weight_artifacts.append(evidence)
        self.probe = EncoderProbe()
        self.cycles = []

    async def run(self, checked_post, headers, snapshot, cycle):
        before = len(self.probe.events)
        outputs = {}
        for role, model, series, output, baseline, key in (
            ("world", "detect-world", "yolo-worldv2", "box", "world", "yolo-worldv2/s/openvocab"),
            ("yoloe_detect", "detect-yoloe", "yoloe-11", "box", "yoloe", "yoloe-11/s/openvocab"),
            ("yoloe_segment", "segment-yoloe", "yoloe-11", "mask", "yoloe", "yoloe-11/s/openvocab"),
        ):
            body = {"tasks": [{"id": role, "file_path": str(self.inputs["image"])}], "context": {"type": "text", "model_id": model, "text": "car", "output": output, "variants": {"series": series, "size": "s"}}}
            response = await checked_post("/predict", body, headers())
            item = response["results"][0]
            details = compare_output(item["result"], self.baselines[baseline], output=output, normalized=False, dimensions=self.dimensions)
            self._check_pool(await snapshot(), key)
            outputs[role] = {**details, "results": item["result"]}
        exemplar = self.baselines["exemplar"]
        body = {"task": {"id": "exemplar", "file_path": str(self.inputs["image"])}, "context": {"type": "exemplar", "model_id": "exemplar-yoloe", "output": "both", "variants": {"series": "yoloe-11", "size": "s"}, "exemplars": [{"bbox": exemplar["exemplar_bbox_normalized"], "label": True}]}}
        response = await checked_post("/predict", body, headers())
        details = compare_output(response["result"], exemplar, output="both", normalized=True, dimensions=self.dimensions)
        self._check_pool(await snapshot(), "yoloe-11/s/openvocab_vp")
        outputs["exemplar"] = {**details, "results": response["result"]}
        events = self.probe.events[before:]
        if {event["encoder"] for event in events} != {"clip", "mobileclip"} or any(not all(event[key].startswith("cuda") for key in ("model_device", "input_device", "output_device")) for event in events):
            raise AssertionError("Actual text encoder execution did not remain on CUDA")
        self.cycles.append({"cycle": cycle, "outputs": outputs, "encoder_devices": events})

    @staticmethod
    def _check_pool(snapshot, key):
        if key not in {entry["key"] for entry in snapshot["loaded_keys"]} or snapshot["gpu_resident"] is not True or not str(snapshot["device"]).startswith("cuda") or snapshot["builders"] or snapshot["borrowers"] or snapshot["waiters"]:
            raise AssertionError("Open-vocabulary model is not resident on the expected GPU pool key")

    def evidence(self):
        if [cycle["cycle"] for cycle in self.cycles] != [1, 2]:
            raise AssertionError("Open-vocabulary workloads require both measured cycles")
        return self.weight_artifacts + self.artifacts, {"placement": self.placement, "cycles": self.cycles}
