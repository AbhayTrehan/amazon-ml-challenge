"""Small content-addressed artifact helpers for shared retrieval caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_key(dependencies: Any) -> str:
    return hashlib.sha256(canonical_json(dependencies)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, default=float).encode("utf-8") + b"\n")


def complete_manifest(path: Path, identity: dict, artifacts: list[Path], metadata: dict) -> dict:
    """Publish COMPLETE only after every declared artifact exists and hashes."""
    missing = [str(item) for item in artifacts if not item.is_file()]
    if missing:
        raise RuntimeError(f"cannot complete manifest; missing artifacts: {missing}")
    manifest = {
        "state": "COMPLETE",
        "content_key": content_key(identity),
        "identity": identity,
        "artifacts": [
            {"path": str(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in artifacts
        ],
        "metadata": metadata,
    }
    atomic_json(path, manifest)
    return manifest


def validate_complete(path: Path) -> dict | None:
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("state") != "COMPLETE" or manifest.get("content_key") != content_key(manifest.get("identity")):
        return None
    for item in manifest.get("artifacts", []):
        artifact = Path(item["path"])
        if not artifact.is_file() or artifact.stat().st_size != item["bytes"] or sha256_file(artifact) != item["sha256"]:
            return None
    return manifest
