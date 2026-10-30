from __future__ import annotations

import hashlib
import json

import pytest

import plugin_assets


def test_asset_verification_rejects_tampering_and_unapproved_files(
    tmp_path, monkeypatch
):
    content = b"approved model"
    digest = hashlib.sha256(content).hexdigest()
    relative = "PP-OCRv6/det/example.onnx"
    monkeypatch.setattr(plugin_assets, "_known_assets", lambda: {relative: digest})
    model = tmp_path / "models" / relative
    model.parent.mkdir(parents=True)
    model.write_bytes(content)
    manifest = json.dumps({"files": [{"path": relative, "sha256": digest, "license": "test-license", "license_source": "https://example.org/license"}]}).encode()
    (tmp_path / "assets.json").write_bytes(manifest)
    manifest_digest = hashlib.sha256(manifest).hexdigest()

    plugin_assets.verify_assets(tmp_path, manifest_digest)
    model.write_bytes(b"altered model")
    with pytest.raises(ValueError, match="digest"):
        plugin_assets.verify_assets(tmp_path, manifest_digest)
    model.write_bytes(content)
    (model.parent / "unapproved.onnx").write_bytes(content)
    with pytest.raises(ValueError, match="unapproved"):
        plugin_assets.verify_assets(tmp_path, manifest_digest)
    with pytest.raises(RuntimeError, match="manifest digest"):
        plugin_assets.verify_assets(tmp_path, "0" * 64)


def test_manifest_preparation_requires_per_asset_license(tmp_path, monkeypatch, capsys):
    import sys

    path = "PP-OCRv6/det/PP-OCRv6_det_small.onnx"
    output = tmp_path / "assets.json"
    args = ["plugin_assets", "--artifact", "local@sha256:" + "a" * 64,
            "--publisher", "example", "--license", "MIT", "--sdk-compatibility", ">=0.1",
            "--device", "cpu", "--memory-mb", "1024", "--asset", path,
            "--asset-manifest", str(output)]
    monkeypatch.setattr(sys, "argv", args + ["--asset-license", f"{path}=Apache-2.0=https://example.org/notice"])
    assert plugin_assets.main() == 0
    entry = json.loads(output.read_text())["files"][0]
    assert entry["license"] == "Apache-2.0"
    assert entry["license_source"] == "https://example.org/notice"
    assert json.loads(capsys.readouterr().out)["asset_manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()

    monkeypatch.setattr(sys, "argv", args + ["--asset-license", "other=MIT=https://example.org"])
    with pytest.raises(SystemExit):
        plugin_assets.main()
