from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_HEALTH_MARKER_VERSION = 1
_CHECK_MODES = {"auto", "quick", "full"}


def _health_marker_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}.health.json")


def _unlink_if_possible(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _db_metadata(db_path: Path) -> tuple[int, int]:
    with sqlite3.connect(str(db_path)) as conn:
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    return schema_version, user_version


def _marker_payload(db_path: Path) -> dict[str, int]:
    stat = db_path.stat()
    schema_version, user_version = _db_metadata(db_path)
    return {
        "version": _HEALTH_MARKER_VERSION,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "schema_version": schema_version,
        "user_version": user_version,
    }


def record_clean_eos_shutdown(db_path: Path) -> Path | None:
    """Record a cheap, one-run health proof after graceful runtime shutdown."""
    from storage.hosted_postgres import enabled
    if enabled():
        # PostgreSQL owns checkpoint/recovery. Legacy files are preserved sources.
        return None
    db_path = db_path.expanduser()
    marker = _health_marker_path(db_path)
    temporary = marker.with_suffix(f"{marker.suffix}.tmp")
    try:
        if not db_path.exists() or db_path.stat().st_size == 0:
            _unlink_if_possible(marker)
            return marker
        with sqlite3.connect(str(db_path)) as conn:
            checkpoint = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        if checkpoint and (
            int(checkpoint[0]) != 0 or int(checkpoint[1]) != int(checkpoint[2])
        ):
            _unlink_if_possible(marker)
            return marker
        payload = _marker_payload(db_path)
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(marker)
    except (OSError, ValueError, TypeError, sqlite3.DatabaseError):
        _unlink_if_possible(temporary)
        _unlink_if_possible(marker)
    return marker


def _clean_marker_matches(db_path: Path) -> bool:
    marker = _health_marker_path(db_path)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = _marker_payload(db_path)
    except (FileNotFoundError, OSError, ValueError, sqlite3.DatabaseError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def _run_check(db_path: Path, kind: str) -> str:
    pragma = "quick_check(1)" if kind == "quick" else "integrity_check"
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(f"PRAGMA {pragma}").fetchone()
    return "" if not row else str(row[0]).lower()


def _is_confirmed_corruption(exc: sqlite3.DatabaseError) -> bool:
    corruption_codes = {
        getattr(sqlite3, "SQLITE_CORRUPT", 11),
        getattr(sqlite3, "SQLITE_NOTADB", 26),
    }
    code = getattr(exc, "sqlite_errorcode", None)
    return isinstance(code, int) and (code & 0xFF) in corruption_codes


def quarantine_malformed_eos_db(
    db_path: Path,
    *,
    check_mode: str = "full",
    consume_clean_marker: bool = False,
) -> Path | None:
    """Move a malformed existing eos.db aside so bootstrap can recreate it."""
    if check_mode not in _CHECK_MODES:
        raise ValueError(f"unsupported eos.db check mode: {check_mode}")
    from storage.hosted_postgres import enabled
    if enabled():
        return None
    db_path = db_path.expanduser()
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None

    try:
        marker_matches = check_mode == "auto" and _clean_marker_matches(db_path)
        if marker_matches:
            result = "ok"
        elif check_mode == "full":
            result = _run_check(db_path, "full")
        else:
            try:
                result = _run_check(db_path, "quick")
            except sqlite3.DatabaseError as exc:
                if not _is_confirmed_corruption(exc):
                    raise
                result = "corrupt"
            if result != "ok":
                result = _run_check(db_path, "full")
    except sqlite3.DatabaseError as exc:
        if _is_confirmed_corruption(exc):
            return _quarantine(db_path)
        raise
    finally:
        if consume_clean_marker:
            _health_marker_path(db_path).unlink(missing_ok=True)

    if result != "ok":
        return _quarantine(db_path)
    return None


def _quarantine(db_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = db_path.with_name(f"{db_path.name}.corrupt-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = db_path.with_name(f"{db_path.name}.corrupt-{timestamp}-{suffix}")
        suffix += 1
    db_path.replace(candidate)
    for sidecar_suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(f"{db_path.name}{sidecar_suffix}")
        if not sidecar.exists():
            continue
        sidecar_candidate = candidate.with_name(f"{candidate.name}{sidecar_suffix}")
        sidecar.replace(sidecar_candidate)
    marker = _health_marker_path(db_path)
    if marker.exists():
        marker.replace(candidate.with_name(f"{candidate.name}.health.json"))
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine malformed repo-local eos.db before bootstrap replace.",
    )
    parser.add_argument("--db-path", default=".semantier-home/eos.db")
    args = parser.parse_args(argv)

    quarantined = quarantine_malformed_eos_db(Path(args.db_path))
    if quarantined:
        print(f"[bootstrap] quarantined malformed eos.db: {quarantined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
