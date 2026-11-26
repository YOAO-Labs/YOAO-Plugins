"""Atomic, fsynced file helpers and content-addressed artifact copies."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from yoao_plugin_sdk.trainer import atomic_write_json, canonical_json_digest


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    atomic_write_json(path, value, mode=mode)


def atomic_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    fd_open = True
    try:
        with os.fdopen(fd, "wb") as target:
            fd_open = False
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
        sync_directory(path.parent)
    except BaseException:
        if fd_open:
            os.close(fd)
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def immutable_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_file(source) != digest_file(destination):
            raise ValueError("Immutable artifact already exists with different content")
    else:
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(destination.parent))
        fd_open = True
        try:
            with source.open("rb") as original, os.fdopen(fd, "wb") as copy_stream:
                fd_open = False
                shutil.copyfileobj(original, copy_stream, length=1024 * 1024)
                copy_stream.flush()
                os.fsync(copy_stream.fileno())
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, destination)
            sync_directory(destination.parent)
        except BaseException:
            if fd_open:
                os.close(fd)
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
    return {
        "name": destination.name,
        "sha256": digest_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def canonical_digest(value: Any) -> str:
    return canonical_json_digest(value)
