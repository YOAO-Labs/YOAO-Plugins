#!/usr/bin/env python3
"""Administrator-invoked SAM 2.1 and GroundingDINO checkpoint preparation.

This tool is never run by service startup. It does not grant approval, check
asset identities or prepare BERT; verify all files against the approved
manifest independently before making them available to the backend.

SAM_VARIANT and DINO_VARIANT select the primary pair; optional
PREFETCH_SAM_VARIANTS and PREFETCH_DINO_VARIANTS select additional variants.
CHECKPOINT_DIR defaults to /app/checkpoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))
SAM_VARIANT = os.getenv("SAM_VARIANT", "tiny")
DINO_VARIANT = os.getenv("DINO_VARIANT", "T")


def _parse_variants(raw: str | None) -> list[str]:
    """逗号分隔解析, 去空白 / 去空项 / 去重保序."""
    if not raw:
        return []
    seen: dict[str, None] = {}
    for part in raw.split(","):
        v = part.strip()
        if v and v not in seen:
            seen[v] = None
    return list(seen)


# (filename, hf_repo_id, hf_filename)
SAM2_FILES = {
    "tiny": (
        "sam2.1_hiera_tiny.pt",
        "facebook/sam2.1-hiera-tiny",
        "sam2.1_hiera_tiny.pt",
    ),
    "small": (
        "sam2.1_hiera_small.pt",
        "facebook/sam2.1-hiera-small",
        "sam2.1_hiera_small.pt",
    ),
    "base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "facebook/sam2.1-hiera-base-plus",
        "sam2.1_hiera_base_plus.pt",
    ),
    "large": (
        "sam2.1_hiera_large.pt",
        "facebook/sam2.1-hiera-large",
        "sam2.1_hiera_large.pt",
    ),
}

DINO_FILES = {
    "T": (
        "groundingdino_swint_ogc.pth",
        "ShilongLiu/GroundingDINO",
        "groundingdino_swint_ogc.pth",
    ),
    "B": (
        "groundingdino_swinb_cogcoor.pth",
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
    ),
}


def _download(target: Path, repo_id: str, filename: str) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(
            f"[skip] {target.name} already exists ({target.stat().st_size // 1024} KB)"
        )
        return
    print(f"[download] {repo_id}/{filename} → {target}")
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target.parent),
        local_dir_use_symlinks=False,
    )
    cached_path = Path(cached)
    if cached_path != target:
        cached_path.replace(target)
    print(f"[done] {target} ({target.stat().st_size // 1024} KB)")


def main() -> int:
    # primary: primary pair; prefetch: extras only; both: all requested files.
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode not in ("primary", "prefetch", "both"):
        print(
            f"ERROR: unknown mode={mode!r}; want primary|prefetch|both", file=sys.stderr
        )
        return 1

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if SAM_VARIANT not in SAM2_FILES:
        print(f"ERROR: unknown SAM_VARIANT={SAM_VARIANT}", file=sys.stderr)
        return 1
    if DINO_VARIANT not in DINO_FILES:
        print(f"ERROR: unknown DINO_VARIANT={DINO_VARIANT}", file=sys.stderr)
        return 1

    # 主变体: 必须成功; 额外预拉变体 (PREFETCH): best-effort.
    sam_extra = [
        v
        for v in _parse_variants(os.getenv("PREFETCH_SAM_VARIANTS"))
        if v != SAM_VARIANT
    ]
    dino_extra = [
        v
        for v in _parse_variants(os.getenv("PREFETCH_DINO_VARIANTS"))
        if v != DINO_VARIANT
    ]
    for v in sam_extra:
        if v not in SAM2_FILES:
            print(f"ERROR: unknown PREFETCH SAM variant={v}", file=sys.stderr)
            return 1
    for v in dino_extra:
        if v not in DINO_FILES:
            print(f"ERROR: unknown PREFETCH DINO variant={v}", file=sys.stderr)
            return 1

    # (target, repo_id, filename, mandatory)
    plan: list[tuple[Path, str, str, bool]] = []
    if mode in ("primary", "both"):
        plan += [
            (
                CHECKPOINT_DIR / SAM2_FILES[SAM_VARIANT][0],
                SAM2_FILES[SAM_VARIANT][1],
                SAM2_FILES[SAM_VARIANT][2],
                True,
            ),
            (
                CHECKPOINT_DIR / DINO_FILES[DINO_VARIANT][0],
                DINO_FILES[DINO_VARIANT][1],
                DINO_FILES[DINO_VARIANT][2],
                True,
            ),
        ]
    if mode in ("prefetch", "both"):
        for v in sam_extra:
            plan.append(
                (
                    CHECKPOINT_DIR / SAM2_FILES[v][0],
                    SAM2_FILES[v][1],
                    SAM2_FILES[v][2],
                    False,
                )
            )
        for v in dino_extra:
            plan.append(
                (
                    CHECKPOINT_DIR / DINO_FILES[v][0],
                    DINO_FILES[v][1],
                    DINO_FILES[v][2],
                    False,
                )
            )
        if sam_extra or dino_extra:
            print(f"[prefetch] extra SAM={sam_extra} DINO={dino_extra}")

    for target, repo_id, filename, mandatory in plan:
        try:
            _download(target, repo_id, filename)
        except Exception as exc:  # noqa: BLE001
            if mandatory:
                print(
                    f"ERROR: failed to fetch primary {repo_id}/{filename}: {exc}",
                    file=sys.stderr,
                )
                return 1
            print(
                f"[warn] prefetch failed for {repo_id}/{filename}: {exc} — "
                "运行期请求该变体会 503, 容器仍按主变体启动.",
                file=sys.stderr,
            )

    print(f"[ok] checkpoints ready in {CHECKPOINT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
