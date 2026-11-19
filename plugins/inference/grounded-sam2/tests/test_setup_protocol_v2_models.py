"""Grounded-SAM2 协议 v2 多模型目录测试。

验证 /setup 的 models[] 将文本检测、文本分割、视频追踪、交互分割和批量框分割
各声明为独立模型，供平台协议能力目录按任务归类。
"""

from __future__ import annotations

from main import _seeds_from_ctx, setup


def _correction_prompt() -> dict:
    return {
        "type": "correction_frame",
        "frame_index": 12,
        "direction": "forward",
        "output_geometry": "mask",
        "mask_prompt": {
            "rle": {
                "encoding": "coco_rle",
                "size": [2, 3],
                "counts": [1, 2, 3],
            },
            "source_annotation_id": "annotation-1",
            "source_version": 3,
            "source_digest": "a" * 64,
        },
    }


def test_setup_top_level_infra_is_pytorch():
    data = setup()
    assert data["infra"] == "pytorch"


def test_setup_protocol_version_v22():
    data = setup()
    assert data["protocol_version"] == "2.2"
    assert data["compat_protocol_versions"] == ["2.1", "2.0"]


def test_setup_exposes_five_models():
    """目录声明五个模型，包括批量框分割原子。"""
    data = setup()
    assert isinstance(data["models"], list)
    assert len(data["models"]) == 5


def test_box_seg_model_is_public_non_interactive_atom():
    """批量框分割是公开、非交互的原子能力，输入 bbox 并输出 polygon。"""
    data = setup()
    box_seg = next(m for m in data["models"] if m["id"] == "grounded-sam2-box-seg")
    assert box_seg["task"] == "segment"
    assert box_seg["is_interactive"] is False
    assert box_seg["supported_prompts"] == ["bbox"]
    assert box_seg["supported_geometric_outputs"] == ["polygon"]
    assert box_seg["composition"] == "atom"
    # public: 不声明 visibility(平台缺省 public), 与 internal 区分。
    assert box_seg.get("visibility") in (None, "public")


def test_composition_annotations():
    """各模型在 composition 中声明原子能力或内部编排。"""
    by_id = {m["id"]: m for m in setup()["models"]}
    assert by_id["grounded-sam2-detection"]["composition"] == "atom"
    assert by_id["grounded-sam2-segmentation"]["composition"] == "composite"
    assert by_id["grounded-sam2-interactive-seg"]["composition"] == "atom"
    assert by_id["grounded-sam2-tracker"]["composition"] == "composite"
    assert by_id["grounded-sam2-box-seg"]["composition"] == "atom"


def test_setup_models_cover_protocol_tasks():
    data = setup()
    tasks = {m["task"] for m in data["models"]}
    assert tasks == {"detect", "segment", "interactive_seg", "tracker"}


def test_setup_models_declare_supported_inputs():
    """各模型在 supported_inputs 中显式声明输入契约。"""
    by_id = {m["id"]: m for m in setup()["models"]}
    assert by_id["grounded-sam2-detection"]["supported_inputs"] == [
        "full_image",
        "crop",
    ]
    assert by_id["grounded-sam2-segmentation"]["supported_inputs"] == [
        "full_image",
        "crop",
    ]
    assert by_id["grounded-sam2-interactive-seg"]["supported_inputs"] == [
        "bbox_prompt",
        "point_prompt",
        "mask_prompt",
        "scribble_prompt",
        "full_image",
    ]
    assert by_id["grounded-sam2-tracker"]["supported_inputs"] == [
        "video",
        "point_prompt",
        "bbox_prompt",
        "mask_prompt",
    ]
    assert by_id["grounded-sam2-box-seg"]["supported_inputs"] == [
        "bbox_prompt",
        "full_image",
    ]


def test_setup_models_declare_output_attribute_types():
    """文本检测和文本分割声明类别；交互、追踪和批量框分割不声明自产属性。score 不计入属性。"""
    by_id = {m["id"]: m for m in setup()["models"]}
    assert by_id["grounded-sam2-detection"]["output_attribute_types"] == ["class"]
    assert by_id["grounded-sam2-segmentation"]["output_attribute_types"] == ["class"]
    # 交互分割 / 视频追踪 / 框→mask 细化: 透传或无类别产出, 不声明。
    assert "output_attribute_types" not in by_id["grounded-sam2-interactive-seg"]
    assert "output_attribute_types" not in by_id["grounded-sam2-tracker"]
    assert "output_attribute_types" not in by_id["grounded-sam2-box-seg"]


def test_setup_models_declare_resource_profile():
    """批量模型声明 batchable=True，交互和视频追踪逐次运行；均不声明显存。"""
    by_id = {m["id"]: m for m in setup()["models"]}
    assert by_id["grounded-sam2-detection"]["resource_profile"] == {
        "device": "gpu",
        "batchable": True,
    }
    assert by_id["grounded-sam2-segmentation"]["resource_profile"] == {
        "device": "gpu",
        "batchable": True,
    }
    assert by_id["grounded-sam2-box-seg"]["resource_profile"] == {
        "device": "gpu",
        "batchable": True,
    }
    assert by_id["grounded-sam2-interactive-seg"]["resource_profile"] == {
        "device": "gpu",
        "batchable": False,
    }
    assert by_id["grounded-sam2-tracker"]["resource_profile"] == {
        "device": "gpu",
        "batchable": False,
    }


def test_setup_each_model_carries_infra_pytorch():
    data = setup()
    for m in data["models"]:
        assert m["infra"] == "pytorch"
        assert m["model_family"] == "grounded-sam2"


def test_detection_model_text_to_bbox():
    data = setup()
    det = next(m for m in data["models"] if m["task"] == "detect")
    assert det["supported_prompts"] == ["text"]
    assert det["supported_geometric_outputs"] == ["bbox"]
    assert det["is_interactive"] is False


def test_segmentation_model_text_to_polygon():
    data = setup()
    seg = next(m for m in data["models"] if m["task"] == "segment")
    assert seg["supported_prompts"] == ["text"]
    assert seg["supported_geometric_outputs"] == ["polygon"]


def test_interactive_seg_model_point_box_to_native_mask():
    data = setup()
    inter = next(m for m in data["models"] if m["task"] == "interactive_seg")
    # 图像交互使用 interactive_box，单框生成单个 mask。
    assert set(inter["supported_prompts"]) == {
        "point",
        "interactive_box",
        "mask",
        "scribble",
    }
    assert inter["supported_geometric_outputs"] == ["polygon", "mask"]
    assert inter["is_interactive"] is True


def test_image_mask_and_scribble_prompts_are_advertised_only_on_consumer():
    by_id = {m["id"]: m for m in setup()["models"]}
    interactive = by_id["grounded-sam2-interactive-seg"]
    assert {"mask", "scribble"}.issubset(interactive["supported_prompts"])
    assert {"mask_prompt", "scribble_prompt"}.issubset(interactive["supported_inputs"])
    assert "correction_frame" not in interactive["supported_prompts"]
    assert interactive["supported_geometric_outputs"] == ["polygon", "mask"]

    tracker = by_id["grounded-sam2-tracker"]
    assert "correction_frame" in tracker["supported_prompts"]
    assert "point" in tracker["supported_prompts"]
    assert {"mask", "scribble"}.isdisjoint(tracker["supported_prompts"])
    assert "mask_prompt" in tracker["supported_inputs"]
    assert "point_prompt" in tracker["supported_inputs"]
    assert "scribble_prompt" not in tracker["supported_inputs"]
    assert "video" in tracker["supported_inputs"]
    assert tracker["max_window_frames"] > 0


def test_tracker_model_sam2_video():
    data = setup()
    tracker = next(m for m in data["models"] if m["task"] == "tracker")
    assert tracker["supported_trackers"] == ["sam2_video"]
    assert tracker["supported_geometric_outputs"] == ["bbox", "polygon", "mask"]


def test_tracker_preserves_validated_correction_frame_seed():
    prompt = _correction_prompt()

    assert _seeds_from_ctx({"seeds": [{"obj_id": 7, "prompts": [prompt]}]}) == [
        {"obj_id": 7, "prompts": [prompt]}
    ]


def test_top_level_back_compat_fields_unchanged():
    """顶层 supported_prompts / supported_trackers 保留, 供未升级平台合成隐式单 model."""
    data = setup()
    # 图像交互框为 interactive_box；追踪和批量框分割的 bbox 不在顶层声明。
    assert set(data["supported_prompts"]) == {
        "point",
        "interactive_box",
        "mask",
        "scribble",
        "text",
    }
    assert data["supported_trackers"] == ["sam2_video"]


def test_setup_params_schema_platform_roles():
    props = setup()["params"]["properties"]
    assert props["box_threshold"]["x-platform-role"] == "confidence"
    assert props["text_threshold"]["x-platform-role"] == "textThreshold"
    assert props["simplify_tolerance"]["x-platform-role"] == "simplifyTolerance"
