#!/usr/bin/env python3
"""Fetch the immutable platform contract wheels and controlled asset manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE = "https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/contracts-0a21bc402432/"
MANIFEST_SHA256 = "b38851f5e736a760021c226039562c2afc9efd21c8ab229f0bb403bab7eca292"
ASSETS_SHA256 = "2a173e4b606321dda236e6615296c1632f495d1edb3f975cd79bfba9529e075f"
REVISION = "0a21bc402432aa93a80fcb9f12d2f532e7109635"


def get(name, expected):
    if Path(name).name != name:
        raise ValueError("invalid contract artifact filename")
    with urlopen(BASE + name, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"contract artifact hash mismatch: {name}")
    return data


def fetch(output):
    manifest = json.loads(get("manifest.json", MANIFEST_SHA256))
    if manifest["source_revision"] != REVISION:
        raise ValueError("contract source revision mismatch")
    output.mkdir(parents=True, exist_ok=True)
    files = [item for item in manifest["files"] if item["file"].endswith(".whl") or item["file"] == "controlled-model-assets.json"]
    if len(files) != 6 or next(item for item in files if item["file"] == "controlled-model-assets.json")["sha256"] != ASSETS_SHA256:
        raise ValueError("unexpected contract artifact inventory")
    for item in files:
        (output / item["file"]).write_bytes(get(item["file"], item["sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fetch(parser.parse_args().output)
