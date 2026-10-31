"""Export hash-pinned third-party dependencies from uv.lock (Python 3.11+)."""
from pathlib import Path
import tomllib

root = Path(__file__).parent
lock = tomllib.loads((root / "uv.lock").read_text())
excluded = {
    "grounded-sam2-backend", "aap-backend-runtime", "aap-lifecycle-contract",
    "aap-protocol-v2", "aap-task-contract", "anno-mask-utils", "torch",
    "torchvision", "triton", "intel-openmp", "mkl", "tbb",
}
lines = ["# Generated from uv.lock; Torch/CUDA are supplied by the pinned base image."]
for package in lock["package"]:
    name = package["name"]
    if name in excluded or (name.startswith("nvidia-") and name != "nvidia-ml-py"):
        continue
    hashes = sorted({item["hash"] for item in [package.get("sdist", {}), *package.get("wheels", [])] if "hash" in item})
    if not hashes:
        raise ValueError(f"No immutable distributions for {name}")
    markers = package.get("resolution-markers", [])
    marker = " ; " + " or ".join(f"({value})" for value in markers) if markers else ""
    lines.append(f"{name}=={package['version']}{marker} " + " ".join(f"--hash={value}" for value in hashes))
(root / "requirements.lock").write_text("\n".join(lines) + "\n")
