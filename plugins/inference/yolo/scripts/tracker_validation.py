"""Positive tracking workloads bound to independently frozen native baselines."""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import cv2

from aap_backend_runtime import artifact_evidence
from scripts.open_vocabulary_validation import _iou_box

INPUTS = {
    "video": "VALIDATION_TRACKER_VIDEO",
    "bytetrack": "VALIDATION_TRACKER_BYTETRACK_BASELINE",
    "botsort": "VALIDATION_TRACKER_BOTSORT_BASELINE",
}


def compare_tracks(actual, baseline, dimensions):
    width, height = dimensions
    expected = [item for item in baseline["tracks"].values() if item["class_name"] == "car"]
    observed = [item for item in actual if item.get("class_name") == "car"]
    if not expected or len(observed) != len(expected) or len(actual) != len(observed):
        raise AssertionError("Positive tracker results differ from native baseline")
    ids = set()
    for track, reference in zip(observed, expected, strict=True):
        track_id = track.get("track_id")
        if track.get("type") != "video_track_bbox" or type(track_id) is not int or track_id in ids or not math.isfinite(track["score"]) or not 0 <= track["score"] <= 1:
            raise AssertionError("Tracker identity or score is invalid")
        ids.add(track_id)
        frames = track["keyframes"]
        expected_frames = reference["frames"]
        if len(frames) < 2 or [frame["frame_index"] for frame in frames] != [frame["frame_index"] for frame in expected_frames]:
            raise AssertionError("Tracker frame continuity differs from native baseline")
        for frame, native in zip(frames, expected_frames, strict=True):
            box = frame["bbox"]
            coords = [box["x"] * width, box["y"] * height, (box["x"] + box["w"]) * width, (box["y"] + box["h"]) * height]
            if any(not math.isfinite(value) for value in coords) or not (0 <= coords[0] < coords[2] <= width and 0 <= coords[1] < coords[3] <= height) or not math.isfinite(frame["score"]) or not 0 <= frame["score"] <= 1 or _iou_box(coords, native["xyxy"]) < 0.9:
                raise AssertionError("Tracker frame geometry differs from native baseline")
    return {"tracks": len(observed), "frames": sum(len(item["keyframes"]) for item in observed)}


class TrackerFixture:
    def __init__(self, root, approval_ref):
        raw = os.environ.get("VALIDATION_TRACKER_FIXTURE_SHA256_JSON", "")
        expected = json.loads(raw)
        if not isinstance(expected, dict) or set(expected) != set(INPUTS) or any(not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) for sha in expected.values()):
            raise ValueError("Tracker fixture requires exact approved SHA-256 mapping")
        target = root / "tracker-inputs"
        target.mkdir()
        self.inputs = {}
        self.artifacts = []
        for role, env in INPUTS.items():
            name = os.environ.get(env, "").strip()
            if not name:
                raise RuntimeError(f"{env} is required")
            path = target / ("video.mp4" if role == "video" else f"{role}.json")
            shutil.copyfile(Path(name), path)
            evidence = artifact_evidence(path, kind="fixture", approval_ref=approval_ref)
            if evidence["sha256"] != expected[role]:
                raise ValueError("Tracker fixture differs from approved SHA-256")
            self.artifacts.append(evidence)
            self.inputs[role] = path
        self.baselines = {name: json.loads(self.inputs[name].read_bytes()) for name in ("bytetrack", "botsort")}
        for name, value in self.baselines.items():
            if value.get("algorithm") != name or value.get("video_sha256") != expected["video"]:
                raise ValueError("Tracker baseline algorithm or video identity differs")
        capture = cv2.VideoCapture(str(self.inputs["video"]))
        try:
            self.dimensions = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not capture.isOpened() or min(self.dimensions) <= 0:
                raise ValueError("Tracker video cannot be decoded")
        finally:
            capture.release()
        self.cycles = []

    async def run(self, checked_post, headers, pool, series, size, cycle):
        outputs = {}
        for name, expected_type in (("bytetrack", "BYTETracker"), ("botsort", "BOTSORT")):
            body = {"tasks": [{"id": f"tracker-{name}", "file_path": str(self.inputs["video"])}], "context": {"type": "tracker", "variants": {"series": series, "size": size}, "params": {"tracker": name}}}
            response = await checked_post("/predict", body, headers())
            items = response["results"][0]["result"]
            details = compare_tracks(items, self.baselines[name], self.dimensions)
            async with pool.borrow("tracker", series, size) as lease:
                trackers = getattr(getattr(lease.model, "predictor", None), "trackers", [])
                if len(trackers) != 1 or type(trackers[0]).__name__ != expected_type or not str(lease.model.device).startswith("cuda"):
                    raise AssertionError("Actual tracker backend or model device differs")
            state = await pool.snapshot()
            if f"{series}/{size}/tracker" not in {entry["key"] for entry in state["loaded_keys"]} or state["gpu_resident"] is not True or state["builders"] or state["borrowers"] or state["waiters"]:
                raise AssertionError("Tracker pool is not idle and GPU resident")
            outputs[name] = {**details, "backend": expected_type, "model_device": str(lease.model.device), "results": items}
        self.cycles.append({"cycle": cycle, "outputs": outputs})

    def evidence(self):
        if [cycle["cycle"] for cycle in self.cycles] != [1, 2]:
            raise AssertionError("Both tracker variants require two measured cycles")
        return self.artifacts, {"cycles": self.cycles}
