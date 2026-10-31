import pytest

from sam3_plugin.runtime_config import checkpoint_dir, configured_device


def test_native_data_root_overrides_standalone_checkpoint_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AAP_PLUGIN_TOKEN_FILE", "/run/plugin-token")
    monkeypatch.setenv("AAP_PLUGIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DIR", "/unapproved")
    monkeypatch.setenv("AAP_PLUGIN_DEVICE", "cuda:0")
    assert checkpoint_dir() == str(tmp_path / "checkpoints")
    assert configured_device() == "cuda:0"


def test_native_rejects_cpu_device(monkeypatch):
    monkeypatch.setenv("AAP_PLUGIN_TOKEN_FILE", "/run/plugin-token")
    monkeypatch.setenv("AAP_PLUGIN_DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="CPU inference is unsupported"):
        configured_device()
