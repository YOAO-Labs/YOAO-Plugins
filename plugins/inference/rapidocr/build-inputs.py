#!/usr/bin/env python3
"""Build a standalone wheel and verified, checkout-independent image context."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

PLUGIN = Path(__file__).resolve().parent
ROOT = PLUGIN.parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="new directory for independent image inputs")
parser.add_argument("--manifest", required=True, help="contract manifest path or URL")
parser.add_argument("--manifest-sha256", required=True, help="contract manifest SHA256")
parser.add_argument("--runtime-wheel", required=True, type=Path, help="local yoao-backend-runtime wheel")
parser.add_argument("--runtime-sha256", required=True, help="runtime wheel SHA256")
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
wheels = out / "wheels"
wheels.mkdir()
subprocess.run(
    [
        sys.executable, str(ROOT / "scripts/fetch-contracts.py"),
        "--manifest", args.manifest,
        "--sha256", args.manifest_sha256,
        "--output", str(wheels),
        "--package", "yoao-task-contract",
        "--package", "yoao-protocol-v2",
        "--package", "yoao-lifecycle-contract",
    ],
    check=True,
)
if not args.runtime_wheel.name.startswith("yoao_backend_runtime-") or args.runtime_wheel.suffix != ".whl":
    raise ValueError("expected a yoao-backend-runtime wheel")
runtime = args.runtime_wheel.read_bytes()
if hashlib.sha256(runtime).hexdigest() != args.runtime_sha256:
    raise ValueError("backend runtime artifact hash mismatch")
(wheels / args.runtime_wheel.name).write_bytes(runtime)
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(PLUGIN)], check=True)
descriptor = subprocess.check_output([sys.executable, str(PLUGIN / "pretrained_manifest.py")])
(out / "pretrained-models.json").write_bytes(descriptor)
for name in ("Dockerfile", "Dockerfile.cpu", "requirements-build.txt", "requirements-cpu.txt", "requirements-runtime.txt"):
    shutil.copyfile(PLUGIN / name, out / name)
shutil.copyfile(PLUGIN / "scripts/validate_managed_lifecycle.py", out / "validate_managed_lifecycle.py")
print(f"Standalone image inputs: {out}")
