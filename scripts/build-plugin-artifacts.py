#!/usr/bin/env python3
"""Build publisher inputs solely from a named committed plugin source revision."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("plugins/inference/libreyolo/native",)
PLUGINS = ("libreyolo", "yolo", "rapidocr", "onnxtools", "grounded-sam2", "sam3")
CATALOGS = {
    "libreyolo": ("aap_libreyolo_plugin", "aap_libreyolo_plugin/controlled-model-assets.json"),
    "yolo": ("yolo_backend", "yolo_plugin/pretrained-assets.json"),
    "rapidocr": ("rapidocr_backend", None),
    "onnxtools": ("onnxtools_backend", "/share/onnxtools-backend/pretrained-model-assets.json"),
    "grounded-sam2": ("grounded_sam2_backend", "grounded_sam2_plugin/pretrained-assets.json"),
    "sam3": ("sam3_backend", "sam3_plugin/pretrained-assets.json"),
}


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_submodules(revision):
    """Record vendor identities from the same commit as the source archive."""
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=ROOT)
    pins = {}
    for entry in tree.split(b"\0"):
        if entry.startswith(b"160000 commit "):
            metadata, path = entry.split(b"\t", 1)
            pins[path.decode("utf-8")] = metadata.split()[2].decode("ascii")
    if not pins:
        return []
    declarations = subprocess.check_output(
        ["git", "config", "--blob", f"{revision}:.gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
        cwd=ROOT, text=True,
    )
    result = []
    for declaration in declarations.splitlines():
        key, path = declaration.split(maxsplit=1)
        if path not in pins:
            continue
        url = subprocess.check_output(
            ["git", "config", "--blob", f"{revision}:.gitmodules", "--get", key[:-5] + ".url"],
            cwd=ROOT, text=True,
        ).strip()
        result.append({"path": path, "url": url, "commit": pins.pop(path)})
    if pins:
        raise ValueError(f"Submodules lack a URL declaration: {sorted(pins)}")
    return sorted(result, key=lambda item: item["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument(
        "--plugin",
        action="append",
        choices=PLUGINS,
        help="plugin to build (repeat to select several; default: libreyolo)",
    )
    args = parser.parse_args()
    plugins = tuple(f"plugins/inference/{name}" for name in dict.fromkeys(args.plugin or ["libreyolo"]))
    packages = PACKAGES if "plugins/inference/libreyolo" in plugins else ()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=ROOT, text=True).strip()
    epoch = subprocess.check_output(["git", "show", "-s", "--format=%ct", revision], cwd=ROOT, text=True).strip()
    env = {**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONHASHSEED": "0"}
    archive = output / "plugin-source.tar"
    run("git", "archive", "--format=tar", "--output", str(archive), revision, cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="plugin-artifacts-") as temp:
        source = Path(temp) / "source"
        source.mkdir()
        # git archive produces only paths from the local committed tree.
        with tarfile.open(archive) as tar:
            tar.extractall(source, filter="data")
        for package in packages:
            run(sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(output), str(source / package), env=env)
        for plugin in plugins:
            run(sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(output), str(source / plugin), env=env)
            name = Path(plugin).name
            distribution, member_suffix = CATALOGS[name]
            wheel, = output.glob(f"{distribution}-*.whl")
            if name == "rapidocr":
                # Import the built wheel, never the publisher's source checkout.
                catalog = subprocess.check_output(
                    [sys.executable, "-I", "-c",
                     "import json,sys; sys.path.insert(0,sys.argv[1]); "
                     "from pretrained_manifest import descriptor; "
                     "print(json.dumps(descriptor(),indent=2,sort_keys=True))",
                     str(wheel)],
                    env=env,
                )
            else:
                with zipfile.ZipFile(wheel) as built:
                    member, = (item for item in built.namelist() if item.endswith(member_suffix))
                    catalog = built.read(member)
            (output / f"{name}-pretrained-assets.json").write_bytes(catalog)
    files = [{"file": path.name, "sha256": sha(path), "size": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {"schema_version": 1, "source_revision": revision, "source_date_epoch": int(epoch), "source_repository": "https://github.com/YOAO-Labs/YOAO-Plugins", "source_submodules": source_submodules(revision), "source_archive_excludes_submodules": True, "license": "MIT", "license_scope": "First-party source only; third-party dependencies and model assets retain their own terms.", "license_files": [f"{p}/LICENSE" for p in (*packages, *plugins)], "plugins": list(plugins), "builders": {name: importlib.metadata.version(name) for name in ("build", "setuptools", "wheel")}, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS"), encoding="ascii")
    print(f"Publisher inputs: {output} (source {revision})")


if __name__ == "__main__":
    main()
