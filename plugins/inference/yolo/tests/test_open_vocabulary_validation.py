"""Independent geometry expectations for open-vocabulary qualification."""
import copy

import pytest

from scripts.open_vocabulary_validation import compare_output


@pytest.fixture
def native():
    return {"results": [{"class_index": 0, "class_name": "car", "score": 0.9, "box_xyxy": [10, 20, 70, 80], "polygon_xy": [[10, 20], [70, 20], [70, 80], [10, 80]]}]}


def test_native_box_and_polygon_geometry_must_match(native):
    results = [
        {"type": "rectanglelabels", "native_class_index": 0, "score": 0.9, "value": {"x": 10, "y": 20, "width": 60, "height": 60, "rectanglelabels": ["car"]}},
        {"type": "polygonlabels", "score": 0.9, "value": {"points": [[10, 20], [70, 20], [70, 80], [10, 80]], "polygonlabels": ["car"]}},
    ]
    assert compare_output(results, native, output="both", normalized=False, dimensions=(100, 100)) == {"boxes": 1, "polygons": 1}
    wrong_class = copy.deepcopy(results)
    wrong_class[0]["value"]["rectanglelabels"] = ["truck"]
    with pytest.raises(AssertionError, match="box"):
        compare_output(wrong_class, native, output="both", normalized=False, dimensions=(100, 100))
    wrong_polygon = copy.deepcopy(results)
    wrong_polygon[1]["value"]["points"] = [[0, 0], [5, 0], [5, 5], [0, 5]]
    with pytest.raises(AssertionError, match="polygon"):
        compare_output(wrong_polygon, native, output="both", normalized=False, dimensions=(100, 100))
    with pytest.raises(AssertionError, match="counts"):
        compare_output(results[:1], native, output="both", normalized=False, dimensions=(100, 100))
