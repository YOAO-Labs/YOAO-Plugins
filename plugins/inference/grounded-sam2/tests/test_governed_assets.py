import hashlib
import json

import pytest

import main


def test_image_preflight_requires_all_local_bert_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CHECKPOINT_DIR", str(tmp_path))
    bert = tmp_path / "bert-base-uncased"
    bert.mkdir()
    monkeypatch.setattr(main, "BERT_DIR", str(bert))
    config = tmp_path / "dino-config.py"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("DINO_CONFIG_PATH", str(config))
    (tmp_path / main.SAM2_CONFIGS["tiny"][1]).touch()
    (tmp_path / main.DINO_CONFIGS["T"][1]).touch()
    with pytest.raises(FileNotFoundError, match="BERT asset"):
        main._preflight_image_model("tiny", "T")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"):
        (bert / name).touch()
    main._preflight_image_model("tiny", "T")


def test_qualification_report_rejects_changed_asset_bytes(tmp_path, monkeypatch):
    data = b"approved test asset"
    weight = tmp_path / "model.bin"
    weight.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    asset = {"path": "model.bin", "sha256": digest, "size_bytes": len(data)}
    monkeypatch.setattr(
        main, "deployment_selections",
        lambda _root: [("grounded-sam2-segmentation", {}, [asset])],
    )
    report = json.dumps({"artifacts": [{"kind": "weight", **asset}]}).encode()
    report_path = tmp_path / "report.json"
    report_path.write_bytes(report)
    monkeypatch.setenv("AAP_PLUGIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AAP_PLUGIN_QUALIFICATION_FILE", str(report_path))
    monkeypatch.setenv("AAP_PLUGIN_QUALIFICATION_SHA256", hashlib.sha256(report).hexdigest())
    assert main._qualification_weights() == [{"sha256": digest, "size_bytes": len(data)}]
    weight.write_bytes(b"changed test asset")
    with pytest.raises(ValueError, match="differs"):
        main._qualification_weights()
