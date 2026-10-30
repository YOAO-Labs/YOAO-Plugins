"""Exercise native plugin authentication and irreversible CPU drain in a fresh process."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import pytest


def test_plugin_auth_managed_access_and_terminal_drain(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("plugin-credential-" + "x" * 48)
    registry = "12345678-1234-4234-8234-123456789abc"
    env = os.environ.copy()
    env.update(
        AAP_PLUGIN_TOKEN_FILE=str(token_file),
        AAP_PLUGIN_DATA_DIR=str(tmp_path / "data"),
        AAP_PLUGIN_DEVICE="cpu",
        AAP_PLUGIN_ID="yolo",
        AAP_PLUGIN_VERSION="0.1.0",
        AAP_PLUGIN_REGISTRY_ID=registry,
        AAP_ROUTE_REGISTRY_ID=registry,
        CUDA_VISIBLE_DEVICES="-1",
        NVIDIA_VISIBLE_DEVICES="none",
    )
    for name in ("AAP_PLUGIN_GPU_RESOURCE_ID", "AAP_PLUGIN_GPU_UUID", "AAP_PLUGIN_QUALIFICATION_FILE", "AAP_PLUGIN_QUALIFICATION_SHA256", "AAP_ROUTE_CONTROL_TOKEN", "AAP_ROUTE_CONTROL_TOKEN_FILE", "AAP_BACKEND_AUTH_TOKEN", "YOLO_MANAGED_MODELS_TOKEN"):
        env.pop(name, None)
    script = textwrap.dedent('''
        from pathlib import Path
        from fastapi.testclient import TestClient
        import os
        from yolo_plugin.catalog import catalog_sha256
        os.environ["AAP_PLUGIN_ASSET_MANIFEST_SHA256"] = catalog_sha256()
        import main
        token = Path(os.environ["AAP_PLUGIN_TOKEN_FILE"]).read_text()
        auth = {"Authorization": "Bearer " + token}
        with TestClient(main.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["compute"]["configured_device"] == "cpu"
            assert client.get("/setup").status_code == 403
            assert client.get("/setup", headers={"Authorization": "Bearer wrong"}).status_code == 403
            setup = client.get("/setup", headers=auth)
            assert setup.status_code == 200
            assert setup.json()["plugin_identity"] == {"plugin_id": "yolo", "version": "0.1.0"}
            assert setup.json()["terminal_lifecycle"] == {"terminal_drain_endpoint": "/lifecycle/terminal-drain"}
            assert "managed_lifecycle" not in setup.json()
            profile = client.get("/managed-evaluation/profile", headers=auth)
            assert profile.status_code == 200, profile.text
            body = {"registry_id": os.environ["AAP_PLUGIN_REGISTRY_ID"], "boot_id": health.json()["residency"]["boot_id"], "operation_id": "stop-1"}
            assert client.post("/lifecycle/terminal-drain", json=body).status_code == 403
            response = client.post("/lifecycle/terminal-drain", json=body, headers=auth)
            assert response.status_code == 200, response.text
            assert response.json()["unloaded"] is True
            assert client.post("/lifecycle/terminal-drain", json=body, headers=auth).json() == response.json()
            assert client.post("/warmup", json={"task": "detect", "variants": {"series": "yolo11", "size": "n"}}, headers=auth).status_code == 503
        os.environ["AAP_PLUGIN_ASSET_MANIFEST_SHA256"] = "0" * 64
        try:
            with TestClient(main.app):
                raise AssertionError("mismatched catalog was accepted")
        except RuntimeError as exc:
            assert "asset manifest" in str(exc)
    ''')
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=90)


def test_qualification_weight_inventory_uses_actual_contained_bytes(tmp_path, monkeypatch):
    import main

    root = tmp_path / "plugin"
    root.mkdir()
    weight = root / "weights.pt"
    weight.write_bytes(b"actual weight")
    monkeypatch.setenv("AAP_PLUGIN_DATA_DIR", str(root))
    monkeypatch.setenv("AAP_PLUGIN_QUALIFICATION_WEIGHTS_JSON", json.dumps([str(weight)]))
    assert main._plugin_qualification_weights(required=True) == [
        {"sha256": hashlib.sha256(b"actual weight").hexdigest(), "size_bytes": len(b"actual weight")}
    ]
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    link = root / "link.pt"
    link.symlink_to(outside)
    monkeypatch.setenv("AAP_PLUGIN_QUALIFICATION_WEIGHTS_JSON", json.dumps([str(link)]))
    with pytest.raises(ValueError, match="inside the plugin data directory"):
        main._plugin_qualification_weights(required=True)
