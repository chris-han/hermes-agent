from __future__ import annotations

import sqlite3
from storage import hosted_postgres as hosted
from dataclasses import dataclass
from typing import Mapping


CURRENT_SCHEMA_VERSION = 1
CURRENT_WORKFLOW_CONTRACT_VERSION = 1
CURRENT_CONTINUATION_PAYLOAD_VERSION = 1
CURRENT_EFFECT_CONTRACT_VERSION = 1
LEGACY_VERSION = 0
MIGRATION_REQUIRED = "MIGRATION_REQUIRED"


class MigrationRequired(RuntimeError):
    def __init__(self, artifact: str, versions: Mapping[str, int]) -> None:
        self.artifact = artifact
        self.versions = dict(versions)
        super().__init__(
            f"{MIGRATION_REQUIRED}: {artifact} has unsupported versions {self.versions}"
        )


@dataclass(frozen=True)
class ArtifactVersionSpec:
    table: str
    columns: tuple[str, ...]


ARTIFACT_VERSION_SPECS: dict[str, ArtifactVersionSpec] = {
    "event": ArtifactVersionSpec(
        table="semantier_durable_events",
        columns=("schema_version", "workflow_contract_version"),
    ),
    "timer": ArtifactVersionSpec(
        table="semantier_timer_occurrences",
        columns=("schema_version", "workflow_contract_version"),
    ),
    "effect": ArtifactVersionSpec(
        table="semantier_effect_outbox",
        columns=(
            "schema_version",
            "workflow_contract_version",
            "effect_contract_version",
        ),
    ),
    "continuation": ArtifactVersionSpec(
        table="semantier_internal_continuations",
        columns=(
            "schema_version",
            "workflow_contract_version",
            "continuation_payload_version",
        ),
    ),
}


def _current_for(column: str) -> int:
    if column == "schema_version":
        return CURRENT_SCHEMA_VERSION
    if column == "workflow_contract_version":
        return CURRENT_WORKFLOW_CONTRACT_VERSION
    if column == "continuation_payload_version":
        return CURRENT_CONTINUATION_PAYLOAD_VERSION
    if column == "effect_contract_version":
        return CURRENT_EFFECT_CONTRACT_VERSION
    raise KeyError(column)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    if hosted.is_postgres(conn):
        return conn.execute(
            "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename=%s", (table,)
        ).fetchone() is not None
    row = conn.execute(
        hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", "SELECT 1 FROM sqlite_master WHERE type='table' AND name=%s"),
        (table,),
    ).fetchone()
    return row is not None


def ensure_version_columns(conn: sqlite3.Connection, artifact: str) -> None:
    spec = ARTIFACT_VERSION_SPECS[artifact]
    if not table_exists(conn, spec.table):
        return
    if hosted.is_postgres(conn):
        existing = {row[0] for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s", (spec.table,)
        )}
        if not set(spec.columns).issubset(existing):
            raise MigrationRequired(artifact, {name: LEGACY_VERSION for name in spec.columns})
        return
    existing = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({spec.table})").fetchall()
    }
    for column in spec.columns:
        if column in existing:
            continue
        conn.execute(
            f"ALTER TABLE {spec.table} "
            f"ADD COLUMN {column} INTEGER NOT NULL DEFAULT {LEGACY_VERSION}"
        )


def current_versions(artifact: str) -> dict[str, int]:
    spec = ARTIFACT_VERSION_SPECS[artifact]
    return {column: _current_for(column) for column in spec.columns}


def assert_supported_versions(
    artifact: str,
    values: Mapping[str, int],
) -> None:
    expected = current_versions(artifact)
    actual = {column: int(values[column]) for column in expected}
    if actual != expected:
        raise MigrationRequired(artifact, actual)


def assert_row_supported(artifact: str, row) -> None:
    spec = ARTIFACT_VERSION_SPECS[artifact]
    try:
        values = {column: int(row[column]) for column in spec.columns}
    except (KeyError, IndexError, TypeError):
        values = {column: LEGACY_VERSION for column in spec.columns}
    assert_supported_versions(artifact, values)


def migrate_v0_to_v1(conn: sqlite3.Connection, artifact: str) -> int:
    """Apply the declared identity-preserving v0 -> v1 metadata migration.

    This migration changes only version metadata. It must not rewrite event
    identity, accepted sequence, effect identity/idempotency, timer identity,
    continuation identity, or domain payload/state.
    """

    spec = ARTIFACT_VERSION_SPECS[artifact]
    if not table_exists(conn, spec.table):
        return 0
    ensure_version_columns(conn, artifact)

    current = current_versions(artifact)
    select_cols = ", ".join(spec.columns)
    rows = conn.execute(
        f"SELECT {select_cols} FROM {spec.table}"
    ).fetchall()

    for row in rows:
        values = {column: int(row[idx]) for idx, column in enumerate(spec.columns)}
        if values == current:
            continue
        if all(value == LEGACY_VERSION for value in values.values()):
            continue
        raise MigrationRequired(artifact, values)

    assignments = ", ".join(hosted.sql_query(f"{column}=?", f'{column}=%s') for column in spec.columns)
    legacy_predicate = " AND ".join(hosted.sql_query(f"{column}=?", f'{column}=%s') for column in spec.columns)
    params = [
        *[current[column] for column in spec.columns],
        *[LEGACY_VERSION for _ in spec.columns],
    ]
    cursor = conn.execute(
        f"UPDATE {spec.table} SET {assignments} WHERE {legacy_predicate}",
        params,
    )
    return int(cursor.rowcount)


def migrate_all_v0_to_v1(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        artifact: migrate_v0_to_v1(conn, artifact)
        for artifact in ARTIFACT_VERSION_SPECS
    }
