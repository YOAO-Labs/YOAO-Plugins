"""Create a native installation intent only for an inspected digest-pinned image.

Requires the published aap-inference-contract and its dependencies. This emits
no approval: the platform must authorize both the installation and its assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from aap_inference_contract.plugin import PluginManifest

ENTRYPOINT = ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="locally inspected image with one repository digest")
    parser.add_argument("--asset-manifest", type=Path, default=Path(__file__).resolve().with_name("pretrained-model-assets.json"))
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--license", required=True, help="publisher's license declaration for the plugin artifact")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", "--", args.image], text=True))
    if len(inspected) != 1:
        parser.error("image inspection must return exactly one image")
    image = inspected[0]
    digests = image.get("RepoDigests") or []
    if len(digests) != 1:
        parser.error("image must have exactly one inspected repository digest")
    config = image.get("Config") or {}
    if config.get("Entrypoint") != ENTRYPOINT or config.get("Cmd") not in (None, []):
        parser.error("image entrypoint differs from the native ONNXTools service")
    assets = args.asset_manifest.read_bytes()
    manifest = PluginManifest.model_validate({
        "plugin_id": "onnxtools", "version": "0.3.0", "publisher": args.publisher,
        "license": args.license, "artifact": digests[0], "sdk_compatibility": "0.1.0",
        "entrypoint": ENTRYPOINT, "capabilities": ["inference", "detection"],
        "models": [
            {"id": model_id, "task": task, "is_interactive": False,
             "supported_prompts": ["none"], "supported_inputs": ["full_image", "crop"],
             "supported_geometric_outputs": outputs}
            for model_id, task, outputs in (
                ("vehicle-attr", "detect", ["bbox"]),
                ("vehicle-detect", "detect", ["bbox"]),
                ("vehicle-attr-classify", "classify", []),
            )
        ],
        "pools": {name: {"cap": 1} for name in ("pipeline", "detector", "va")},
        "resource": {"device": args.device, "memory_mb": args.memory_mb, "gpu_memory_mb": args.gpu_memory_mb},
        "asset_manifest_sha256": hashlib.sha256(assets).hexdigest(),
        "allowed_asset_sources": [], "managed_model_compatibility": [],
    })
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
