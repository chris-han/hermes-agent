"""Version-and-sentinel guards for idempotent SQLite store schemas."""

from __future__ import annotations

import sqlite3
from collections.abc import Collection
from datetime import datetime, timezone

from storage import hosted_postgres as hosted


def schema_is_current(
    conn: sqlite3.Connection,
    *,
    schema_name: str,
    schema_version: int,
    sentinels: Collection[str],
) -> bool:
    if hosted.is_postgres(conn):
        # Hosted markers are migration-owned; inspect without triggering a
        # missing-relation error which would poison the caller's transaction.
        if conn.execute("SELECT to_regclass('semantier_schema_versions')").fetchone()[0] is None:
            return False
        row = conn.execute(
            "SELECT schema_version FROM semantier_schema_versions WHERE schema_name = %s",
            (schema_name,),
        ).fetchone()
        if row is None or int(row["schema_version"]) != schema_version:
            return False
        present = {
            row[0] for row in conn.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename = ANY(%s)",
                (sorted(sentinels),),
            )
        }
        return present == set(sentinels)
    try:
        row = conn.execute(
            "SELECT schema_version FROM semantier_schema_versions WHERE schema_name = ?",
            (schema_name,),
        ).fetchone()
        if row is None or int(row["schema_version"]) != schema_version:
            return False
        placeholders = ",".join("?" for _ in sentinels)
        present = {
            str(item["name"])
            for item in conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' "
                f"AND name IN ({placeholders})",
                tuple(sorted(sentinels)),
            ).fetchall()
        }
        return present == set(sentinels)
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return False


def mark_schema_current(
    conn: sqlite3.Connection,
    *,
    schema_name: str,
    schema_version: int,
) -> None:
    if hosted.is_postgres(conn):
        raise hosted.HostedStorageUnavailable("hosted schema markers require an explicit owner migration")
    conn.execute(
        "INSERT INTO semantier_schema_versions(schema_name, schema_version, updated_at) "
        "VALUES (?, ?, ?) ON CONFLICT(schema_name) DO UPDATE SET "
        "schema_version=excluded.schema_version, updated_at=excluded.updated_at",
        (
            schema_name,
            schema_version,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
