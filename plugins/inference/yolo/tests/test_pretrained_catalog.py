"""The installed descriptor catalog stays consistent with training model identities."""
import json
from importlib.resources import as_file

from training_contract.assets import read_manifest
from training_contract.models import models_for_framework, supports_version
from yolo_plugin.catalog import catalog_bytes, catalog_resource


def test_catalog_covers_controlled_ultralytics_models_without_platform_authority():
    payload = json.loads(catalog_bytes())
    entries = payload["assets"]
    specs = {spec.id: spec for spec in models_for_framework("ultralytics")}
    assert {entry["base_model"] for entry in entries} == set(specs)
    with as_file(catalog_resource()) as path:
        references = read_manifest(path)
    assert len(references) == len(specs)
    for entry in entries:
        spec = specs[entry["base_model"]]
        assert entry["framework"] == spec.framework
        assert entry["task"] == spec.task
        assert entry["model_family"] == spec.family
        assert entry["model_size"] == spec.scale
        assert entry["license"] == "unverified"
        assert "object_key" not in entry
        assert entry["compatibility"]["versions"] == [version for version in ("8.4.177", "8.4.183") if supports_version(spec, version)]
        assert entry["source_url"].startswith("https://github.com/ultralytics/assets/releases/download/")
        assert entry["size_bytes"] > 0


def test_qualification_inference_assets_have_declared_roles_and_encoder_dependencies():
    from scripts.open_vocabulary_validation import WEIGHTS

    payload = json.loads(catalog_bytes())
    inference = {entry["filename"]: entry for entry in payload["inference_assets"]}
    assert set(WEIGHTS) <= set(inference)
    for entry in inference.values():
        assert entry["license"] == "unverified"
        assert entry["size_bytes"] > 0
        assert len(entry["sha256"]) == 64
        assert entry["source_url"].startswith("https://")
        assert "object_key" not in entry
    for entry in inference.values():
        if entry["role"] == "open_vocabulary_model":
            assert entry["capabilities"]
            for filename in entry["dependencies"]:
                assert inference[filename]["role"] == "text_encoder"
                assert set(inference[filename]["used_by"]) <= set(entry["capabilities"])
