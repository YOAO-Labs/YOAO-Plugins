"""Published SAM3 asset metadata, not permission to acquire or run it."""
from __future__ import annotations

import json
from importlib.resources import files


def descriptor() -> dict:
    return json.loads(files("sam3_plugin").joinpath("pretrained-assets.json").read_text(encoding="utf-8"))


def deployment_selections(data_root):
    """Return selections only when publisher identities are independently verified."""
    result = []
    for item in descriptor()["assets"]:
        if not item["sha256"] or not item["size_bytes"]:
            raise ValueError(f"Unverified published asset identity: {item['path']}")
        asset = {key: item[key] for key in ("path", "sha256", "size_bytes")}
        for model in item["models"]:
            result.append((model["model_id"], model["variants"], [asset.copy()]))
    return result
