"""Behavioral coverage for explicit, verified artifact selection."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("fetch_contracts", Path(__file__).with_name("fetch-contracts.py"))
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def inputs(tmp_path):
    files = []
    for name in ("yoao_task_contract-0.1.0-py3-none-any.whl", "yoao_training_contract-0.1.0-py3-none-any.whl"):
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        files.append({"file": name, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = tmp_path / "manifest.json"
    data = json.dumps({"files": files}).encode()
    manifest.write_bytes(data)
    return manifest, hashlib.sha256(data).hexdigest(), files


def test_filters_wheels_from_verified_local_manifest(tmp_path):
    manifest, digest, files = inputs(tmp_path)
    output = tmp_path / "output"
    fetcher.fetch(str(manifest), digest, output, ["YOAO.task_contract"])
    assert (output / files[0]["file"]).read_bytes() == files[0]["file"].encode()
    assert not (output / files[1]["file"]).exists()


def test_missing_package_or_tampered_bytes_fail_before_output(tmp_path):
    manifest, digest, files = inputs(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="absent"):
        fetcher.fetch(str(manifest), digest, output, ["yoao-missing"])
    with pytest.raises(ValueError, match="hash mismatch: manifest"):
        fetcher.fetch(str(manifest), "0" * 64, output)
    (tmp_path / files[1]["file"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        fetcher.fetch(str(manifest), digest, output)
    assert not output.exists()
