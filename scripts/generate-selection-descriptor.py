#!/usr/bin/env python3
"""Describe plugin-owned model dependencies for one native installation; never approve or acquire assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
HEX = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def safe_path(root, path):
    if not isinstance(path, str) or not path or "\\" in path or any(part in ("", ".", "..") for part in path.split("/")) or PurePosixPath(path).is_absolute():
        raise ValueError(f"Noncanonical asset path: {path!r}")
    if not (root / path).resolve().is_relative_to(root):
        raise ValueError(f"Asset path escapes data directory: {path!r}")
    return path


def asset(root, metadata, path):
    sha, size = metadata.get("sha256"), metadata.get("size_bytes")
    if not isinstance(sha, str) or not HEX.fullmatch(sha) or type(size) is not int or size <= 0:
        raise ValueError(f"Incomplete published identity for {path}")
    return {"sha256": sha, "size_bytes": size, "path": safe_path(root, path)}


def selections_libreyolo(root):
    sys.path.insert(0, str(ROOT / "plugins/inference/libreyolo"))
    from aap_libreyolo_plugin.assets import assets
    return [
        ("detect", {"series": "yolov9", "size": size}, [asset(root, item, "assets/" + item["filename"])])
        for size, item in assets().items()
    ]


def selections_ocr(root):
    sys.path.insert(0, str(ROOT / "plugins/inference/rapidocr"))
    from pretrained_manifest import descriptor
    published = descriptor()
    groups = {}
    for item in published["assets"]:
        for model in item["models"]:
            variants = model["variants"]
            key = (model["model_id"], tuple(sorted(variants.items())))
            groups.setdefault(key, []).append(asset(root, item, "models/" + item["path"]))
    return [(model, dict(variants), assets) for (model, variants), assets in groups.items()]


def selections_onnx(root):
    published = json.loads((ROOT / "plugins/inference/onnxtools/pretrained-model-assets.json").read_bytes())
    groups = {}
    for item in published["assets"]:
        for model in item["compatible_model_ids"]:
            groups.setdefault(model, []).append(asset(root, item, "models/" + item["filename"]))
    return [(model, {}, assets) for model, assets in groups.items()]


def selections_yolo(root):
    sys.path.insert(0, str(ROOT / "plugins/inference/yolo"))
    import model_registry as registry
    from training_contract.assets import load_manifest
    published = json.loads((ROOT / "plugins/inference/yolo/yolo_plugin/pretrained-assets.json").read_bytes())
    by_name = {}
    for item in published["assets"] + published["inference_assets"]:
        name = item["filename"]
        if name in by_name:
            raise ValueError(f"Duplicate catalog filename: {name}")
        by_name[name] = item
    safe_path(root, "model-assets/assets.json")
    prepared = {entry.filename: entry for entry in load_manifest(root / "model-assets")}

    def weight(name):
        item = by_name[name]
        if name not in prepared:
            path = "checkpoints/" + name
        else:
            entry = prepared[name]
            if entry.sha256 != item["sha256"] or entry.size_bytes != item["size_bytes"]:
                raise ValueError(f"Prepared identity differs from published catalog: {name}")
            blob = "model-assets/" + item["sha256"] + ".pt"
            safe_path(root, blob)
            blob_path = root / blob
            path = blob if blob_path.exists() or blob_path.is_symlink() else "model-assets/" + name
        return asset(root, item, path)

    result = []
    for task, series, size in registry.iter_supported_combinations():
        filename = registry.resolve_weight_filename(task, series, size)
        if filename in by_name:
            result.append(("track" if task == "tracker" else task, {"series": series, "size": size}, [weight(filename)]))
    for model, matrix, encoder in (
        ("detect-world", registry.OPENVOCAB_WORLD_SERIES, "ViT-B-32.pt"),
        ("detect-yoloe", registry.OPENVOCAB_YOLOE_SERIES, "mobileclip_blt.ts"),
        ("segment-yoloe", registry.OPENVOCAB_YOLOE_SERIES, "mobileclip_blt.ts"),
        ("exemplar-yoloe", registry.OPENVOCAB_YOLOE_SERIES, None),
    ):
        for series, sizes in matrix.items():
            for size in sizes:
                filename = registry.resolve_openvocab_weight_filename(series, size)
                if filename not in by_name or (encoder and encoder not in by_name):
                    continue
                assets = [weight(filename)]
                if encoder:
                    encoder_path = "checkpoints/weights/clip/ViT-B-32.pt" if model == "detect-world" else "checkpoints/mobileclip_blt.ts"
                    assets.append(asset(root, by_name[encoder], encoder_path))
                result.append((model, {"series": series, "size": size}, assets))
    return result


def build(plugin, version, artifact, manifest_sha, root):
    if not version or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", artifact):
        raise ValueError("An immutable OCI artifact and plugin version are required")
    if not HEX.fullmatch(manifest_sha):
        raise ValueError("Invalid installation asset manifest SHA256")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Data root must be a directory")
    groups = {"libreyolo": selections_libreyolo, "yolo": selections_yolo, "rapidocr": selections_ocr, "onnxtools": selections_onnx}[plugin](root)
    seen, paths, selections = set(), {}, []
    for model, variants, assets in groups:
        key = (model, tuple(sorted(variants.items())))
        if key in seen or not assets:
            raise ValueError(f"Duplicate or empty selection: {key}")
        seen.add(key)
        local_paths = set()
        for item in assets:
            path = item["path"]
            identity = (item["sha256"], item["size_bytes"])
            if path in local_paths or (path in paths and paths[path] != identity):
                raise ValueError(f"Duplicate or conflicting asset path: {path}")
            local_paths.add(path)
            paths[path] = identity
        selections.append({"model_id": model, "variants": variants, "asset_free": False, "assets": sorted(assets, key=lambda item: item["path"])})
    selections.sort(key=lambda item: (item["model_id"], tuple(sorted(item["variants"].items()))))
    return {"schema_version": 1, "plugin_id": plugin, "plugin_version": version, "artifact": artifact, "asset_manifest_sha256": manifest_sha, "selections": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", choices=("libreyolo", "yolo", "rapidocr", "onnxtools"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--asset-manifest-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir.resolve(strict=True)
    if args.output.resolve().is_relative_to(root):
        parser.error("output must be outside the installation data directory")
    data = canonical(build(args.plugin, args.version, args.artifact, args.asset_manifest_sha256, root))
    with args.output.open("xb") as output:
        output.write(data)
    print(f"{hashlib.sha256(data).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
