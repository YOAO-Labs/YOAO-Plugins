"""Packaged pretrained identities and complete deployment dependencies."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path


def catalog_bytes():
    return files("grounded_sam2_plugin").joinpath("pretrained-assets.json").read_bytes()


def catalog_sha256():
    return hashlib.sha256(catalog_bytes()).hexdigest()


def descriptor():
    return json.loads(catalog_bytes())


def deployment_selections(data_root):
    """Describe dependencies, without approving, acquiring, or modifying assets."""
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Data root must be a directory")
    metadata = descriptor()["assets"]
    sam = {a["variant"]: a for a in metadata if a["kind"] == "sam"}
    dino = {a["variant"]: a for a in metadata if a["kind"] == "dino"}
    bert = [a for a in metadata if a["kind"] == "bert"]
    expected = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"}
    if {a["filename"] for a in bert} != expected:
        raise ValueError("BERT asset identities are incomplete; cannot describe image deployments")

    def asset(item):
        parts = ("checkpoints", "bert-base-uncased", item["filename"]) if item["kind"] == "bert" else ("checkpoints", item["filename"])
        path = "/".join(parts)
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"Asset path escapes data root: {path}")
        return {"path": path, "sha256": item["sha256"], "size_bytes": item["size_bytes"]}

    result = []
    for sv, sam_asset in sam.items():
        result.append(("grounded-sam2-tracker", {"sam_variant": sv}, [asset(sam_asset)]))
        for dv, dino_asset in dino.items():
            dependencies = [asset(sam_asset), asset(dino_asset), *(asset(a) for a in bert)]
            variants = {"sam_variant": sv, "dino_variant": dv}
            for model in ("grounded-sam2-detection", "grounded-sam2-segmentation", "grounded-sam2-interactive-seg", "grounded-sam2-box-seg"):
                result.append((model, variants.copy(), [item.copy() for item in dependencies]))
    return result
