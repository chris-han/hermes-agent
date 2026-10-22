from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from runtime_paths import workspace_session_segment


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stream_sha256(path: Path) -> tuple[str, int, float]:
    started = time.monotonic()
    digest = hashlib.sha256()
    bytes_processed = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            bytes_processed += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), bytes_processed, time.monotonic() - started


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source_hash, source_bytes, elapsed = _stream_sha256(source)
    item = {
        "source_path": str(source),
        "destination_path": str(destination),
        "file_size": source_bytes,
        "sha256": source_hash,
        "bytes_processed": source_bytes,
        "elapsed_seconds": elapsed,
        "action": "copy",
        "status": "planned",
    }
    if destination.exists():
        dest_hash, dest_bytes, dest_elapsed = _stream_sha256(destination)
        item["destination_bytes_processed"] = dest_bytes
        item["destination_elapsed_seconds"] = dest_elapsed
        if dest_hash != source_hash or dest_bytes != source_bytes:
            item["status"] = "failed_hash_mismatch"
            raise ValueError(f"destination exists with different content: {destination}")
        item["status"] = "verified"
        return item

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    dest_hash, dest_bytes, dest_elapsed = _stream_sha256(destination)
    item["destination_bytes_processed"] = dest_bytes
    item["destination_elapsed_seconds"] = dest_elapsed
    if dest_hash != source_hash or dest_bytes != source_bytes:
        item["status"] = "failed_verification"
        raise ValueError(f"copied file failed verification: {destination}")
    item["status"] = "verified"
    return item


def _manifest_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _manifest_path(session_root: Path, timestamp: str) -> Path:
    manifest_dir = session_root / "artifacts" / "migrations"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"{timestamp}-session-layout.json"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _existing_active_manifest(session_root: Path) -> Path | None:
    manifest_dir = session_root / "artifacts" / "migrations"
    if not manifest_dir.exists():
        return None
    for path in sorted(manifest_dir.glob("*-session-layout.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in {"planned", "copied", "verified"}:
            return path
    return None


def _chunks(items: list[tuple[Path, Path]], batch_size: int) -> Iterable[list[tuple[Path, Path]]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _source_destination_pairs(
    sessions_root: Path,
    normalized_root: Path,
    *,
    canonical_session_id: str,
    session_segment: str,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    pairs: list[tuple[Path, Path]] = []
    delete_candidates: list[str] = []

    legacy_root = sessions_root / canonical_session_id
    if legacy_root.exists():
        for source in sorted(path for path in legacy_root.rglob("*") if path.is_file()):
            pairs.append((source, normalized_root / source.relative_to(legacy_root)))
        delete_candidates.append(str(legacy_root))

    encoded = quote(canonical_session_id, safe="-_.")
    for source in (
        sessions_root / f"{encoded}.jsonl",
        sessions_root / f"session_{encoded}.json",
        sessions_root / f"session_{encoded}.trajectory.jsonl",
    ):
        if not source.exists():
            continue
        if source.name.endswith(".trajectory.jsonl"):
            destination = normalized_root / "logs" / f"{session_segment}.trajectory.jsonl"
        elif source.name.endswith(".jsonl"):
            destination = normalized_root / "logs" / f"{session_segment}.jsonl"
        else:
            destination = normalized_root / "logs" / f"{session_segment}.json"
        pairs.append((source, destination))
        delete_candidates.append(str(source))

    return pairs, delete_candidates


def migrate_workspace_session_layout(
    workspace_root: Path,
    *,
    workspace_id: str,
    canonical_session_id: str,
    maintenance_gate_verified: bool,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Copy a legacy prefixed workspace session layout into normalized form.

    This helper is intentionally copy-only. Legacy roots remain compatibility
    read inputs until an operator separately approves deletion after verifying
    the committed manifest.
    """
    if not maintenance_gate_verified:
        raise ValueError("maintenance gate must be verified before session layout migration")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    root = Path(workspace_root)
    session_segment = workspace_session_segment(workspace_id, canonical_session_id)
    sessions_root = root / "sessions"
    normalized_root = sessions_root / session_segment
    active_manifest = _existing_active_manifest(normalized_root)
    manifest_file = active_manifest or _manifest_path(normalized_root, _manifest_timestamp())

    manifest: dict[str, Any] = {
        "canonical_session_id": canonical_session_id,
        "session_filesystem_segment": session_segment,
        "workspace_id": workspace_id,
        "workspace_root": str(root),
        "manifest_path": str(manifest_file),
        "created_at": _now_iso(),
        "batch_size": batch_size,
        "status": "planned",
        "items": [],
        "delete_candidates": [],
    }

    source_pairs, delete_candidates = _source_destination_pairs(
        sessions_root,
        normalized_root,
        canonical_session_id=canonical_session_id,
        session_segment=session_segment,
    )
    manifest["delete_candidates"] = delete_candidates
    manifest["total_items"] = len(source_pairs)
    _write_manifest(manifest_file, manifest)

    try:
        for batch_index, batch in enumerate(_chunks(source_pairs, batch_size), start=1):
            manifest["current_batch"] = batch_index
            manifest["status"] = "copied"
            _write_manifest(manifest_file, manifest)
            for source, destination in batch:
                manifest["items"].append(_copy_verified(source, destination))
                manifest["status"] = "verified"
                _write_manifest(manifest_file, manifest)
    except Exception:
        manifest["status"] = "failed"
        manifest["updated_at"] = _now_iso()
        _write_manifest(manifest_file, manifest)
        raise

    for child in ("runs", "uploads", "artifacts", "logs"):
        (normalized_root / child).mkdir(parents=True, exist_ok=True)
    (normalized_root / "artifacts" / "tmp" / "hermes-results").mkdir(
        parents=True, exist_ok=True
    )

    manifest["status"] = "committed"
    manifest["updated_at"] = _now_iso()
    _write_manifest(manifest_file, manifest)
    return manifest


def rollback_workspace_session_layout(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") == "committed":
        raise ValueError("committed session layout migrations require a reverse manifest")
    deleted: list[str] = []
    for item in reversed(manifest.get("items") or []):
        destination = Path(str(item.get("destination_path") or ""))
        if destination.exists():
            destination.unlink()
            deleted.append(str(destination))
    manifest["status"] = "rolled_back"
    manifest["rolled_back_at"] = _now_iso()
    manifest["rolled_back_destinations"] = deleted
    _write_manifest(Path(manifest_path), manifest)
    return manifest
