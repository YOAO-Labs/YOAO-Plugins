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

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("packages/backend-runtime", "packages/mask-utils", "packages/libreyolo-native", "packages/inference-plugin-sdk")
PLUGIN = "plugins/libreyolo"


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
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
        contracts = Path(temp) / "contracts"
        run(sys.executable, str(source / "scripts/fetch-contracts.py"), str(contracts))
        for package in PACKAGES:
            run(sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(output), str(source / package), env=env)
        run(sys.executable, str(source / PLUGIN / "build-wheel.py"), str(output), "--sdist", "--manifest", str(contracts / "controlled-model-assets.json"), env=env)
    files = [{"file": path.name, "sha256": sha(path), "size": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {"schema_version": 1, "source_revision": revision, "source_date_epoch": int(epoch), "source_repository": "https://github.com/YOAO-Labs/YOAO-Plugins", "license": "MIT", "license_files": [f"{p}/LICENSE" for p in (*PACKAGES, PLUGIN)], "builders": {name: importlib.metadata.version(name) for name in ("build", "setuptools", "wheel")}, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS"), encoding="ascii")
    print(f"Publisher inputs: {output} (source {revision})")


if __name__ == "__main__":
    main()
