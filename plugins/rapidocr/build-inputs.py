#!/usr/bin/env python3
"""Build a standalone wheel and verified, checkout-independent image context."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

PLUGIN = Path(__file__).resolve().parent
ROOT = PLUGIN.parents[1]
RUNTIME_URL = "https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/python-6118885dd686/aap_backend_runtime-0.1.0-py3-none-any.whl"
RUNTIME_SHA256 = "13d25849cafdb53d8a0b0feffa8b3505e9d8393db9e3ee093e7cbb5f4f9896e8"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="new directory for independent image inputs")
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
wheels = out / "wheels"
wheels.mkdir()
subprocess.run([sys.executable, str(ROOT / "scripts/fetch-contracts.py"), str(wheels)], check=True)
with urlopen(RUNTIME_URL, timeout=120) as response:
    runtime = response.read()
if hashlib.sha256(runtime).hexdigest() != RUNTIME_SHA256:
    raise ValueError("backend runtime artifact hash mismatch")
(wheels / RUNTIME_URL.rsplit("/", 1)[-1]).write_bytes(runtime)
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(PLUGIN)], check=True)
for name in ("Dockerfile", "Dockerfile.cpu", "requirements-build.txt", "requirements-cpu.txt", "requirements-runtime.txt"):
    shutil.copyfile(PLUGIN / name, out / name)
shutil.copyfile(PLUGIN / "scripts/validate_managed_lifecycle.py", out / "validate_managed_lifecycle.py")
print(f"Standalone image inputs: {out}")
