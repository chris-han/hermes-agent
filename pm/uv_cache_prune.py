"""Prune UV caches to locked runtime and declared core build requirements.

A shipped payload and rolling CI cache retain distributions named in uv.lock
or core build-system.requires. Tools such as wheel must survive even when
absent from the runtime graph, because a writable repair rebuilds core metadata.
Other identifiable distributions are discarded; unknown layouts stay intact.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


def _name_key(name: str) -> str:
    """PEP 503 distribution identity, also used for wheel dist-info names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_package_names(source_repo: Path) -> set[str]:
    """Dist names the lock can resolve; the exactness contract for a cache."""
    import tomllib

    data = tomllib.loads((source_repo / "uv.lock").read_text(encoding="utf-8"))
    return {_name_key(entry["name"]) for entry in data["package"]}


def _build_package_names(source_repo: Path) -> set[str]:
    """Core editable rebuilds also need the declared PEP 517 backend tools."""
    import tomllib

    manifest = source_repo / "pyproject.toml"
    if not manifest.is_file():
        return set()
    document = tomllib.loads(manifest.read_text(encoding="utf-8-sig"))
    names = set()
    for requirement in document.get("build-system", {}).get("requires", []):
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match is None:
            raise ValueError(f"invalid build requirement: {requirement!r}")
        names.add(_name_key(match.group(1)))
    return names


def prune_uv_cache_to_lock(cache: Path, source_repo: Path) -> int:
    """Delete cache entries for dists the lock cannot resolve.

    Unidentifiable buckets (no dist-info) survive — pruning fails open for
    unknown layouts, never for identifiable stale pins. Returns the pruned
    entry count for the build log.
    """
    keep = lock_package_names(source_repo) | _build_package_names(source_repo)
    dist_info = re.compile(r"([A-Za-z0-9_.]+?)-\d[^-]*\.dist-info")

    def dist_name(bucket: Path) -> str | None:
        for marker_file in bucket.glob("*.dist-info"):
            match = dist_info.match(marker_file.name)
            if match:
                return _name_key(match.group(1))
        return None

    pruned = 0
    archive = cache / "archive-v0"
    if archive.is_dir():
        for bucket in archive.iterdir():
            if not bucket.is_dir():
                continue
            name = dist_name(bucket)
            if name is not None and name not in keep:
                shutil.rmtree(bucket, ignore_errors=True)
                pruned += 1
    for family_dir in (*cache.glob("wheels-v*/pypi"), *cache.glob("sdists-v*/pypi")):
        if not family_dir.is_dir():
            continue
        for entry in family_dir.iterdir():
            if entry.is_dir() and _name_key(entry.name) not in keep:
                shutil.rmtree(entry, ignore_errors=True)
                pruned += 1
    return pruned
