"""Positive multi-frame tracker qualification must preserve native geometry."""
import copy

import pytest

from scripts.tracker_validation import compare_tracks


def test_tracker_requires_positive_continuous_matching_geometry():
    native = {"tracks": {"7": {"class_name": "car", "frames": [
        {"frame_index": 0, "xyxy": [10, 20, 70, 80]},
        {"frame_index": 1, "xyxy": [12, 20, 72, 80]},
    ]}}}
    actual = [{"type": "video_track_bbox", "track_id": 1, "class_name": "car", "score": 0.9, "keyframes": [
        {"frame_index": 0, "bbox": {"x": 0.1, "y": 0.2, "w": 0.6, "h": 0.6}, "score": 0.9},
        {"frame_index": 1, "bbox": {"x": 0.12, "y": 0.2, "w": 0.6, "h": 0.6}, "score": 0.9},
    ]}]
    assert compare_tracks(actual, native, (100, 100)) == {"tracks": 1, "frames": 2}
    lost_frame = copy.deepcopy(actual)
    lost_frame[0]["keyframes"].pop()
    with pytest.raises(AssertionError, match="continuity"):
        compare_tracks(lost_frame, native, (100, 100))
    wrong_box = copy.deepcopy(actual)
    wrong_box[0]["keyframes"][1]["bbox"]["x"] = 0.4
    with pytest.raises(AssertionError, match="geometry"):
        compare_tracks(wrong_box, native, (100, 100))
    with pytest.raises(AssertionError, match="Positive"):
        compare_tracks([], native, (100, 100))
