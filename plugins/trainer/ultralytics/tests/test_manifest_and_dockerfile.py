"""Manifest, Dockerfile, lockfile, and lazy-import boundary tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from yoao_plugin_sdk.manifest import load_manifest_file

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_ROOT / "manifest.json"
DOCKERFILE_PATH = PLUGIN_ROOT / "Dockerfile"
LOCKFILE_PATH = PLUGIN_ROOT / "requirements.lock"


def test_manifest_passes_sdk_and_platform_contract() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "artifact" not in raw
    assert "source_tier" not in raw
    assert raw["resource"] == {"memory_mb": 8192}
    assert len(raw["description"]) <= 280

    manifest = load_manifest_file(MANIFEST_PATH)
    assert manifest.manifest_schema == "yoao-extension-manifest/v1"
    assert manifest.extension_point == "trainer"
    assert manifest.plugin_id == "yoao.trainer.ultralytics"
    assert manifest.version == "1.0.0"
    assert manifest.display_name == "Ultralytics YOLO Trainer"
    assert manifest.homepage == "https://github.com/YOAO-Labs/YOAO-Plugins"
    assert manifest.publisher == "YOAO Official"
    assert manifest.license == "AGPL-3.0-only"
    assert manifest.protocol_version == "yoao-trainer/v1"
    assert manifest.entrypoint == ["yoao-trainer-ultralytics"]
    assert manifest.capabilities == ["train", "evaluate", "preview", "export_weights"]
    assert manifest.resource.model_dump() == {"memory_mb": 8192}
    assert manifest.extension == {
        "framework_id": "ultralytics",
        "framework_version": "8.4.183",
        "requires_gpu": True,
    }


def test_dockerfile_label_matches_manifest_and_enforces_runtime_invariants() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    match = re.search(r"LABEL\s+io\.yoao\.plugin\.manifest='([^']+)'", dockerfile)
    assert match is not None, "Dockerfile missing io.yoao.plugin.manifest label"
    embedded = json.loads(match.group(1))
    assert embedded == manifest

    assert (
        "FROM mirror.gcr.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime"
        "@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722"
        in dockerfile
    )
    for pkg in (
        "fonts-dejavu-core",
        "fonts-noto-cjk",
        "libgl1",
        "libglib2.0-0",
        "ca-certificates",
        "python3.12-venv",
    ):
        assert pkg in dockerfile

    assert "YOLO_CONFIG_DIR=/tmp/ultralytics" in dockerfile
    assert "MPLCONFIGDIR=/tmp/matplotlib" in dockerfile
    assert "HOME=/tmp" in dockerfile
    assert "YOLO_AUTOINSTALL=false" in dockerfile
    assert "chmod -R a+rX" in dockerfile
    assert "ultralytics.__version__ == '8.4.183'" in dockerfile
    assert 'ENTRYPOINT ["yoao-trainer-ultralytics"]' in dockerfile


def test_requirements_lock_pins_ultralytics_and_build_tools_with_hashes() -> None:
    lines = [
        line.strip()
        for line in LOCKFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines
    by_name = {line.split("==", 1)[0]: line for line in lines}
    assert by_name["ultralytics"].startswith("ultralytics==8.4.183 ")
    assert "wheel" in by_name
    assert "setuptools" in by_name
    for excluded in ("pytest", "torch", "torchvision", "yoao-plugin-sdk", "yoao-trainer-ultralytics"):
        assert excluded not in by_name
    for line in lines:
        assert "--hash=sha256:" in line


def test_package_and_describe_do_not_eagerly_import_torch_or_ultralytics() -> None:
    script = (
        "import sys; "
        "import yoao_trainer_ultralytics; "
        "from yoao_trainer_ultralytics.cli import build_capabilities; "
        "caps = build_capabilities(); "
        "assert len(caps.architectures) == 30; "
        "assert 'torch' not in sys.modules; "
        "assert 'ultralytics' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
