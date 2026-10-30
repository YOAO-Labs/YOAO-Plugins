"""Generate the versioned pretrained descriptor from the service catalog and pins.

This is publisher metadata, not permission to download, enable or redistribute.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import catalog
from download_models import MANIFEST

# Sizes measured against the exact pinned SHA256, not against mutable filenames.
_SIZES = {
    "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae": 4819576,
    "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad": 88118768,
    "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7": 1018508,
    "7d3c02ef6c7da8ae08b4347cc7695b2081aae68c325d64375724ecf39c99e743": 6776876,
    "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5": 16631306,
    "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a": 84577022,
    "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8": 7872351,
    "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5": 1829618,
    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f": 9929594,
    "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2": 62119454,
    "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b": 4489813,
    "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884": 21234383,
    "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b": 76629984,
}
_NOTICE = "https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/"
_COVERED = {
    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
}


def descriptor() -> dict:
    """Project every supported route and model asset without loading any weight."""
    assets = {}
    for version, component, url, digest in MANIFEST:
        relative = f"{version}/{component}/{url.rsplit('/', 1)[-1]}"
        assets[relative] = {
            "path": relative,
            "filename": Path(relative).name,
            "component": component,
            "size_bytes": _SIZES[digest],
            "sha256": digest,
            "source": url,
            "license": {
                "status": "verified" if digest in _COVERED else "unverified",
                "identifier": "Apache-2.0" if digest in _COVERED else None,
                "evidence": _NOTICE if digest in _COVERED else None,
            },
            "models": [],
        }
    combinations = {
        "ocr-det": [(v, s, None) for v, s in catalog._DET_COMBOS],
        "ocr-rec": catalog._REC_COMBOS,
        "ocr-e2e": catalog._REC_COMBOS,
    }
    for model, variants in combinations.items():
        for version, size, lang in variants:
            selected = {"version": version, "size": size}
            if lang is not None:
                selected["lang"] = lang
            engine = catalog.resolve(model, selected)
            for path in (engine.det_path, engine.cls_path, engine.rec_path):
                relative = os.path.relpath(path, catalog.MODELS_DIR).replace(os.sep, "/")
                assets[relative]["models"].append({"model_id": model, "variants": selected.copy()})
    return {
        "version": 1,
        "plugin_id": "rapidocr",
        "plugin_version": "0.1.0",
        "runtime_compatibility": {
            "rapidocr": "3.9.0",
            "onnxruntime": "1.23.2",
            "devices": ["cpu", "cuda"],
            "sessions_per_engine": ["det", "cls", "rec"],
        },
        "assets": list(assets.values()),
    }


if __name__ == "__main__":
    print(json.dumps(descriptor(), indent=2, sort_keys=True))
