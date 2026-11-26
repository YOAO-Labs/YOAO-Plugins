"""Export hashed third-party wheels from uv.lock; Torch/CUDA use the pinned base image."""

from __future__ import annotations

from pathlib import Path
import tomllib

EXCLUDED = frozenset(
    {
        "yoao-trainer-libreyolo",
        "yoao-plugin-sdk",
        "torch",
        "torchvision",
        "triton",
        "cuda-toolkit",
        "cuda-bindings",
        "cuda-pathfinder",
    }
)
FORBIDDEN = frozenset({"ultralytics"})


def export_lock(root: Path | None = None) -> Path:
    base = Path(__file__).parent if root is None else Path(root)
    lock = tomllib.loads((base / "uv.lock").read_text(encoding="utf-8"))
    lines = [
        "# Generated from uv.lock; Torch/CUDA binaries come from the pinned Docker base image.",
        "# MIT boundary invariant: ultralytics is strictly forbidden.",
    ]
    for package in lock["package"]:
        name = package["name"]
        normalized = name.lower().replace("_", "-")
        if normalized in FORBIDDEN:
            raise ValueError(f"Forbidden AGPL dependency in LibreYOLO lock: {name}")
        if normalized in EXCLUDED or normalized.startswith("nvidia-"):
            continue
        hashes = sorted(
            {
                artifact["hash"]
                for artifact in [package.get("sdist", {}), *package.get("wheels", [])]
                if "hash" in artifact
            }
        )
        if not hashes:
            raise ValueError(f"No immutable distribution hashes for {name}")
        markers = package.get("resolution-markers", [])
        marker = " ; " + " or ".join(f"({value})" for value in markers) if markers else ""
        lines.append(
            f"{name}=={package['version']}{marker} "
            + " ".join(f"--hash={value}" for value in hashes)
        )
    target = base / "requirements.lock"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    export_lock()
