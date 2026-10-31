import pytest

from grounded_sam2_plugin.catalog import deployment_selections


def test_image_selections_include_both_models_and_bert(tmp_path):
    selections = deployment_selections(tmp_path)
    assert len(selections) == 36
    model, variants, assets = next(
        row for row in selections
        if row[0] == "grounded-sam2-interactive-seg"
        and row[1] == {"sam_variant": "small", "dino_variant": "B"}
    )
    paths = {asset["path"] for asset in assets}
    assert paths == {
        "checkpoints/sam2.1_hiera_small.pt",
        "checkpoints/groundingdino_swinb_cogcoor.pth",
        "checkpoints/bert-base-uncased/config.json",
        "checkpoints/bert-base-uncased/tokenizer.json",
        "checkpoints/bert-base-uncased/tokenizer_config.json",
        "checkpoints/bert-base-uncased/vocab.txt",
        "checkpoints/bert-base-uncased/model.safetensors",
    }
    assert all(len(asset["sha256"]) == 64 and asset["size_bytes"] > 0 for asset in assets)
    tracker = next(row for row in selections if row[0] == "grounded-sam2-tracker" and row[1] == {"sam_variant": "small"})
    assert [item["path"] for item in tracker[2]] == ["checkpoints/sam2.1_hiera_small.pt"]


def test_catalog_rejects_symlink_outside_data_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "checkpoints").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes data root"):
        deployment_selections(root)
