"""Plugin asset preparation uses approved local bytes against its packaged catalog."""
import hashlib
import json

import pytest
from training_contract.assets import AssetReference, AssetUnavailable, resolve_asset
from scripts import prepare_assets


@pytest.fixture
def assets(tmp_path, monkeypatch):
    content = b"reviewed weight"
    item = AssetReference(hashlib.sha256(content).hexdigest(), "yolo11n.pt", len(content))
    source = tmp_path / "source"
    source.mkdir()
    (source / item.filename).write_bytes(content)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "assets": [{**item.metadata(item.size_bytes), "framework": "ultralytics", "base_model": "yolo11n"}]}))
    monkeypatch.setattr(prepare_assets, "catalog_resource", lambda: manifest)
    return item, source, tmp_path / "cache"


def test_local_import_validates_and_materializes_bytes(assets):
    item, source, target = assets
    prepare_assets.prepare_local(source, target, ["yolo11n"])
    assert resolve_asset(target, item).read_bytes() == b"reviewed weight"


def test_corrupt_asset_never_enters_cache(assets):
    item, source, target = assets
    (source / item.filename).write_bytes(b"corrupt weight")
    with pytest.raises(AssetUnavailable):
        prepare_assets.prepare_local(source, target, ["yolo11n"])
    assert not (target / f"{item.sha256}.pt").exists()


def test_unknown_model_cannot_be_prepared(assets):
    _, source, target = assets
    with pytest.raises(ValueError, match="Unknown Ultralytics"):
        prepare_assets.prepare_local(source, target, ["unknown"])
    assert not target.exists()
