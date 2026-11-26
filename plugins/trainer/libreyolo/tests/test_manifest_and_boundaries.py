"""Manifest, Dockerfile label, MIT/zero-ultralytics boundary, and weights_only AST guards."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from yoao_plugin_sdk.manifest import load_manifest_file

from yoao_trainer_libreyolo.catalog import capabilities

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PLUGIN_ROOT / "yoao_trainer_libreyolo"
MANIFEST_PATH = PLUGIN_ROOT / "manifest.json"
DOCKERFILE_PATH = PLUGIN_ROOT / "Dockerfile"
PYPROJECT_PATH = PLUGIN_ROOT / "pyproject.toml"
REQUIREMENTS_LOCK_PATH = PLUGIN_ROOT / "requirements.lock"
LICENSE_PATH = PLUGIN_ROOT / "LICENSE"


def _extract_dockerfile_manifest_label(dockerfile_text: str) -> str:
    match = re.search(
        r"^LABEL\s+io\.yoao\.plugin\.manifest='([^']+)'\s*$",
        dockerfile_text,
        re.MULTILINE,
    )
    assert match is not None, "Dockerfile must declare LABEL io.yoao.plugin.manifest='...'"
    return match.group(1)


def test_manifest_validates_with_sdk_and_platform_constraints() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest_file(MANIFEST_PATH)

    assert manifest.manifest_schema == "yoao-extension-manifest/v1"
    assert manifest.extension_point == "trainer"
    assert manifest.plugin_id == "yoao.trainer.libreyolo"
    assert manifest.version == "1.0.0"
    assert manifest.display_name == "LibreYOLO Trainer"
    assert manifest.description is not None and len(manifest.description) <= 280
    assert manifest.homepage == "https://github.com/YOAO-Labs/YOAO-Plugins"
    assert manifest.publisher == "YOAO Official"
    assert manifest.license == "MIT"
    assert manifest.protocol_version == "yoao-trainer/v1"
    assert manifest.entrypoint == ["yoao-trainer-libreyolo"]
    assert manifest.capabilities == ["train", "evaluate", "export_weights"]
    assert "preview" not in manifest.capabilities
    assert manifest.resource is not None and manifest.resource.memory_mb == 8192
    assert set(raw["resource"]) == {"memory_mb"}
    assert manifest.extension == {
        "framework_id": "libreyolo",
        "framework_version": "1.6.0",
        "requires_gpu": True,
    }
    assert "artifact" not in raw
    assert "source_tier" not in raw
    assert "source_tier" not in raw["extension"]
    assert manifest.artifact is None

    caps = capabilities()
    assert manifest.extension["framework_id"] == caps.framework_id
    assert manifest.extension["framework_version"] == caps.framework_version


def test_dockerfile_label_matches_manifest_and_hardens_runtime() -> None:
    dockerfile_text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    raw_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    label_raw = _extract_dockerfile_manifest_label(dockerfile_text)

    assert json.loads(label_raw) == raw_manifest
    assert label_raw == json.dumps(raw_manifest, separators=(",", ":"), ensure_ascii=False)

    assert (
        "FROM mirror.gcr.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime"
        "@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722"
    ) in dockerfile_text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in dockerfile_text
    assert "LIBREYOLO_FASTER_COCO_EVAL=0" in dockerfile_text
    assert "HOME=/tmp" in dockerfile_text
    assert "MPLCONFIGDIR=/tmp/matplotlib" in dockerfile_text
    assert "chmod -R a+rX" in dockerfile_text
    assert "USER 10001:10001" in dockerfile_text
    assert 'ENTRYPOINT ["yoao-trainer-libreyolo"]' in dockerfile_text
    assert "assert libreyolo.__version__ == '1.6.0'" in dockerfile_text
    assert "assert importlib.util.find_spec('ultralytics') is None" in dockerfile_text


def test_zero_ultralytics_imports_or_dependencies_anywhere() -> None:
    # 1. AST import guard across all non-test Python files in this plugin.
    py_files = [
        *sorted(PACKAGE_ROOT.rglob("*.py")),
        PLUGIN_ROOT / "export_requirements.py",
    ]
    assert py_files
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.lower().startswith("ultralytics"), (
                        f"Forbidden ultralytics import in {path}: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").lower()
                assert not module.startswith("ultralytics"), (
                    f"Forbidden ultralytics from-import in {path}: {node.module}"
                )

    # 2. Packaging dependencies guard (pyproject.toml).
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    all_deps: list[str] = list(pyproject.get("project", {}).get("dependencies", []))
    for extra_deps in pyproject.get("project", {}).get("optional-dependencies", {}).values():
        all_deps.extend(extra_deps)
    for dep in all_deps:
        pkg_name = re.split(r"[<>=!~;\[\s]", dep.strip(), maxsplit=1)[0].lower()
        assert pkg_name != "ultralytics", f"Forbidden ultralytics dependency in pyproject.toml: {dep}"
    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    assert all(k.lower() != "ultralytics" for k in uv_sources)

    # 3. Hash-locked requirements guard (requirements.lock).
    lock_text = REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8")
    locked_packages: set[str] = set()
    for line in lock_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        pkg = stripped.split("==", 1)[0].strip().lower().replace("_", "-")
        assert pkg != "ultralytics", f"Forbidden ultralytics entry in requirements.lock: {line}"
        locked_packages.add(pkg)
    assert "libreyolo" in locked_packages
    assert "libreyolo==1.6.0" in lock_text

    # 4. Manifest & LICENSE guard.
    assert "ultralytics" not in MANIFEST_PATH.read_text(encoding="utf-8").lower()
    assert "MIT License" in LICENSE_PATH.read_text(encoding="utf-8")


def test_every_torch_load_call_passes_weights_only_true_in_ast() -> None:
    torch_load_count = 0
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_torch_load = (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            )
            if not is_torch_load:
                continue
            torch_load_count += 1
            weights_only_kw = [
                kw for kw in node.keywords if kw.arg == "weights_only"
            ]
            assert len(weights_only_kw) == 1, f"torch.load missing weights_only kwarg in {path}"
            val = weights_only_kw[0].value
            assert isinstance(val, ast.Constant) and val.value is True, (
                f"torch.load must pass weights_only=True in {path}"
            )
    assert torch_load_count >= 1


def test_describe_and_cli_import_do_not_eagerly_import_ml_libraries() -> None:
    code = (
        "import io, json, sys; "
        "from contextlib import redirect_stdout; "
        "from yoao_trainer_libreyolo.cli import main; "
        "buf = io.StringIO(); "
        "rc = 0\n"
        "with redirect_stdout(buf):\n"
        "    rc = main(['describe'])\n"
        "assert rc == 0\n"
        "payload = json.loads(buf.getvalue())\n"
        "assert payload['framework_id'] == 'libreyolo'\n"
        "for mod in ('torch', 'torchvision', 'libreyolo', 'cv2', 'ultralytics'):\n"
        "    assert mod not in sys.modules, f'unexpected eager import: {mod}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
