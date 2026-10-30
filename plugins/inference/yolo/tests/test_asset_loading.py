"""Controlled built-in weights must use the prepared read-only asset identity."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training_contract.assets import AssetReference, AssetUnavailable, prepare_asset


@pytest.fixture
def builtin(monkeypatch, tmp_path):
    import main

    assets = tmp_path / "read-only-assets"
    checkpoints = tmp_path / "backend-state"
    checkpoints.mkdir()
    monkeypatch.setattr(main, "MODEL_ASSETS_DIR", assets)
    monkeypatch.setattr(main, "CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setattr(main, "STRICT_OFFLINE", False)
    monkeypatch.setattr(main, "_move_model_to_effective_device", lambda model: model)
    loader = MagicMock(return_value=SimpleNamespace(task="segment"))
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", loader)
    return main, assets, checkpoints, loader


def prepared(assets):
    content = b"verified segmentation model"
    reference = AssetReference(
        hashlib.sha256(content).hexdigest(), "yolo11n-seg.pt", len(content)
    )
    return prepare_asset(assets, reference, lambda: [content])


def test_builtin_loads_shared_blob_without_copying_it_to_backend_state(builtin):
    main, assets, checkpoints, loader = builtin
    path = prepared(assets)
    assert main._build_model("segment", "yolo11", "n") is loader.return_value
    loader.assert_called_once_with(str(path))
    assert list(checkpoints.iterdir()) == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_required_manifest_asset_never_uses_legacy_file_or_online_download(
    builtin, corrupt
):
    main, assets, checkpoints, loader = builtin
    path = prepared(assets)
    (checkpoints / "yolo11n-seg.pt").write_bytes(b"uncontrolled legacy file")
    if corrupt:
        path.chmod(0o644)
        path.write_bytes(b"corrupt")
    else:
        path.unlink()
    with pytest.raises(AssetUnavailable):
        main._build_model("segment", "yolo11", "n")
    loader.assert_not_called()


def test_unmapped_legacy_weight_retains_existing_local_loading(builtin):
    main, _assets, checkpoints, loader = builtin
    path = checkpoints / "yolo11n-seg.pt"
    path.write_bytes(b"legacy weights")
    main._build_model("segment", "yolo11", "n")
    loader.assert_called_once_with(str(path))


