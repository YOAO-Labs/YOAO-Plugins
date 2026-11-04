"""Export hash-pinned third-party dependencies from uv.lock (Python 3.11+)."""
import hashlib
from pathlib import Path
import tomllib

root = Path(__file__).parent
lock = tomllib.loads((root / "uv.lock").read_text())
excluded = {
    "yoao-grounded-sam2-plugin", "yoao-backend-runtime", "yoao-lifecycle-contract",
    "yoao-protocol-v2", "yoao-task-contract", "yoao-mask-utils", "torch",
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

first_party = {"yoao-backend-runtime", "yoao-mask-utils"}
selected = {package["name"]: package for package in lock["package"] if package["name"] in first_party}
if selected.keys() != first_party:
    raise ValueError("Missing first-party packages in uv.lock")
lines = ["# Generated from uv.lock; wheels are staged with scripts/fetch-contracts.py."]
for name in sorted(first_party):
    package = selected[name]
    wheel, = package["wheels"]
    filename = wheel["filename"]
    path = (root / package["source"]["path"]).resolve()
    if path.name != filename or hashlib.sha256(path.read_bytes()).hexdigest() != wheel["hash"].removeprefix("sha256:"):
        raise ValueError(f"Staged wheel differs from uv.lock: {name}")
    lines.append(f"file:///mnt/build-context/.artifacts/python/first-party/{filename} --hash={wheel['hash']}")
(root / "first-party-requirements.lock").write_text("\n".join(lines) + "\n")
