#!/usr/bin/env python3
"""Build a private LibreYOLO candidate from an explicitly named git commit.

Run with CPython 3.11 on glibc Linux x86_64, with git, uv, tar, pip,
setuptools 80.9.0, wheel 0.45.1, hatchling 1.27.0 and build 1.3.0
installed. The build requires network access to the public package indexes;
consumption does not.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("plugins/inference/libreyolo/native",)
PLUGIN = "plugins/inference/libreyolo"
LOCK = "plugins/inference/libreyolo/third-party-requirements.lock"
ASSETS = "plugins/inference/libreyolo/aap_libreyolo_plugin/controlled-model-assets.json"
SDK_BASE = "https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/python-6118885dd686/"
SDK_WHEELS = {
    "aap_backend_runtime-0.1.0-py3-none-any.whl": "13d25849cafdb53d8a0b0feffa8b3505e9d8393db9e3ee093e7cbb5f4f9896e8",
    "aap_inference_plugin_sdk-0.1.0-py3-none-any.whl": "707a1d623ad858f383aa55c411bc898621378e633dbf52a6f4e8f74512e86e78",
}


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as wheel:
        files = [
            name
            for name in wheel.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(files) != 1:
            raise ValueError(f"wheel has invalid metadata: {path.name}")
        metadata = email.parser.BytesParser().parsebytes(wheel.read(files[0]))
    name, version = metadata["Name"], metadata["Version"]
    if not name or not version:
        raise ValueError(f"wheel is missing name or version: {path.name}")
    return re.sub(r"[-_.]+", "-", name).lower(), version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="committed source revision")
    parser.add_argument(
        "--output", required=True, type=Path, help="new candidate .tar path"
    )
    parser.add_argument(
        "--dependency-cache",
        type=Path,
        help="optional reusable directory for downloaded, hash-verified wheels",
    )
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 11)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.libc_ver()[0] != "glibc"
    ):
        parser.error("candidate target requires CPython 3.11 on glibc Linux x86_64")
    for name, expected in (
        ("setuptools", "80.9.0"),
        ("wheel", "0.45.1"),
        ("build", "1.3.0"),
        ("hatchling", "1.27.0"),
    ):
        if importlib.metadata.version(name) != expected:
            parser.error(f"builder requires {name}=={expected}")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        parser.error("output and checksum must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.revision + "^{commit}"],
        cwd=ROOT,
        text=True,
    ).strip()
    epoch = subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", revision], cwd=ROOT, text=True
    ).strip()
    env = {**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="aap-candidate-") as tmp:
        temporary = Path(tmp)
        source = temporary / "source"
        source.mkdir()
        source_tar = temporary / "source.tar"
        run(
            "git",
            "archive",
            "--format=tar",
            "--output",
            str(source_tar),
            revision,
            cwd=ROOT,
        )
        # Match publisher extraction permissions regardless of the caller's umask.
        old_umask = os.umask(0o022)
        try:
            run("tar", "-xf", str(source_tar), "-C", str(source))
        finally:
            os.umask(old_umask)
        source_tar.unlink()
        candidate = temporary / "candidate"
        wheelhouse = candidate / "wheelhouse"
        wheelhouse.mkdir(parents=True)
        exported = source / LOCK
        downloads = (
            args.dependency_cache.resolve()
            if args.dependency_cache
            else temporary / "downloads"
        )
        downloads.mkdir(parents=True, exist_ok=True)
        contracts = temporary / "contracts"
        run(sys.executable, str(source / "scripts/fetch-contracts.py"), str(contracts))
        for wheel in contracts.glob("*.whl"):
            shutil.copyfile(wheel, wheelhouse / wheel.name)
        for name, expected in SDK_WHEELS.items():
            with urlopen(SDK_BASE + name, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"SDK artifact hash mismatch: {name}")
            (wheelhouse / name).write_bytes(data)
        run(
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "download",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu132",
            "--find-links",
            str(downloads),
            "--dest",
            str(downloads),
            "-r",
            str(exported),
        )
        run(
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "download",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--no-index",
            "--find-links",
            str(downloads),
            "--dest",
            str(wheelhouse),
            "-r",
            str(exported),
        )
        for package in PACKAGES:
            run(
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheelhouse),
                str(source / package),
                env=env,
            )
        run(
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
            str(source / PLUGIN),
            env=env,
        )
        shutil.copyfile(source / ASSETS, candidate / "controlled-model-assets.json")
        inventory = []
        seen = set()
        for wheel in sorted(wheelhouse.glob("*.whl")):
            name, version = wheel_metadata(wheel)
            if name in seen:
                raise ValueError(f"multiple wheels for {name}")
            seen.add(name)
            inventory.append(
                {
                    "name": name,
                    "version": version,
                    "file": wheel.name,
                    "sha256": sha256(wheel),
                }
            )
        if (
            "aap-libreyolo-plugin" not in seen
            or "aap-inference-plugin-sdk" not in seen
            or "ai-annotation-sdk" in seen
            or "ultralytics" in seen
        ):
            raise ValueError("unexpected candidate distribution set")
        (candidate / "requirements.lock").write_text(
            "".join(
                f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
                for item in inventory
            ),
            encoding="utf-8",
        )
        (candidate / "inventory.json").write_text(
            json.dumps(
                {
                    "source_revision": revision,
                    "target": "cpython-3.11-linux-x86_64-glibc",
                    "profile": "CPU execution; cu132 Torch dependencies",
                    "controlled_asset_manifest_sha256": sha256(
                        candidate / "controlled-model-assets.json"
                    ),
                    "wheels": inventory,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        run(
            "tar",
            "--sort=name",
            "--mtime=@" + epoch,
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--mode=u+rwX,go+rX,go-w",
            "--format=gnu",
            "-cf",
            str(output),
            "-C",
            str(candidate),
            ".",
        )
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(f"Candidate SHA256: {digest}")


if __name__ == "__main__":
    main()
