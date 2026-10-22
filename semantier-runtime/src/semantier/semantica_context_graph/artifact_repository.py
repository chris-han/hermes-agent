"""Workspace-scoped, append-only ContextGraph artifact storage.

Semantica owns graph serialization.  Semantier owns the manifest, lifecycle,
tenant binding, hashes, and evaluation pins.  This module is infrastructure;
it is intentionally not registered as a Hermes plugin or toolset.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal


from storage.hosted_postgres import (
    enabled,
    connect_authority,
    mapped_store,
    require_tables,
    begin_write,
    integrity_errors,
)
from storage.ontology_build_store import ontology_build_store_path


ArtifactKind = Literal["candidate", "evaluation_baseline", "authoritative"]
AuthorityState = Literal["candidate", "authoritative"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    organization_id: str
    workspace_id: str
    domain_id: str
    graph_id: str
    graph_version: str
    artifact_kind: ArtifactKind
    authority_state: AuthorityState
    run_mode: str | None
    parent_artifact_id: str | None
    lineage_id: str | None
    lineage_root_artifact_id: str
    content_hash: str
    semantica_commit: str
    source_manifest_hash: str
    grounding_manifest_hash: str | None
    evaluation_manifest_hash: str | None
    governance_ref: str | None
    created_by_user_id: str
    created_at: str
    relative_artifact_path: str
    lifecycle_state: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "domain_id": self.domain_id,
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "artifact_kind": self.artifact_kind,
            "authority_state": self.authority_state,
            "run_mode": self.run_mode,
            "parent_artifact_id": self.parent_artifact_id,
            "lineage_id": self.lineage_id,
            "lineage_root_artifact_id": self.lineage_root_artifact_id,
            "content_hash": self.content_hash,
            "semantica_commit": self.semantica_commit,
            "source_manifest_hash": self.source_manifest_hash,
            "grounding_manifest_hash": self.grounding_manifest_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "governance_ref": self.governance_ref,
            "created_by_user_id": self.created_by_user_id,
            "created_at": self.created_at,
            "relative_artifact_path": self.relative_artifact_path,
            "lifecycle_state": self.lifecycle_state,
        }


class ArtifactRepositoryError(RuntimeError):
    """Base error for fail-closed artifact operations."""


class ArtifactNotFound(ArtifactRepositoryError):
    """The requested artifact is not in the authenticated repository."""


class ArtifactConflict(ArtifactRepositoryError):
    """An immutable identity was reused with different content or metadata."""


def validate_contextgraph_connection(connection, manifest_store_path, eos_path):
    if not enabled():
        return
    import psycopg
    from psycopg.pq import TransactionStatus
    from storage.hosted_postgres import database_url

    if (
        not isinstance(connection, psycopg.Connection)
        or connection.info.transaction_status != TransactionStatus.INTRANS
    ):
        raise ArtifactRepositoryError("active genuine PostgreSQL transaction required")
    expected = psycopg.conninfo.conninfo_to_dict(database_url())
    if any(
        str(expected[key]) != str(getattr(connection.info, key))
        for key in ("host", "port", "dbname")
        if key in expected
    ):
        raise ArtifactRepositoryError("borrowed PostgreSQL database mismatch")
    allowed = {
        mapped_store("standalone", manifest_store_path)["schema"],
        mapped_store("eos", eos_path)["schema"],
    }
    row = connection.execute("SELECT current_schema()").fetchone()
    namespace = next(iter(row.values())) if hasattr(row, "values") else row[0]
    if namespace not in allowed:
        raise ArtifactRepositoryError("borrowed PostgreSQL namespace mismatch")


class ContextGraphArtifactRepository:
    """Store immutable graph JSON and its authoritative lifecycle manifest."""

    _process_locks: dict[tuple[str, str, str], threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        domain_id: str = "tender-sensitive",
        eos_path: str | Path | None = None,
    ) -> None:
        if not organization_id or not workspace_id or not domain_id:
            raise ValueError(
                "organization_id, workspace_id, and domain_id are required"
            )
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.domain_id = domain_id
        self.eos_path = Path(
            eos_path
            or os.environ.get("SEMANTIER_EOS_DB_PATH")
            or self.workspace_root / "eos.db"
        ).resolve()
        self._manifest_store_path = ontology_build_store_path(self.workspace_root)
        graph_root = (self.workspace_root / ".semantier" / "context-graphs").resolve()
        self.root = (graph_root / domain_id).resolve()
        if graph_root not in self.root.parents:
            raise ValueError("domain_id escapes ContextGraph workspace root")
        self.authoritative_root = self.root / "authoritative"
        self.baselines_root = self.root / "baselines"
        self.candidates_root = self.root / "candidates"
        self.tmp_root = self.root / "tmp"
        self.manifest_path = self.root / "manifest.sqlite3"
        for path in (
            self.authoritative_root,
            self.baselines_root,
            self.candidates_root,
            self.tmp_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._initialize_manifest()

    @property
    def _lock(self) -> threading.RLock:
        key = (self.organization_id, self.workspace_id, self.domain_id)
        with self._process_locks_guard:
            return self._process_locks.setdefault(key, threading.RLock())

    def _query(self, sqlite, postgres):
        if not enabled():
            return sqlite
        from psycopg import sql

        schemas = {
            "manifest": sql.Identifier(
                mapped_store("standalone", self._manifest_store_path)["schema"]
            )
        }
        if "{eos}" in postgres:
            schemas["eos"] = sql.Identifier(
                mapped_store("eos", self.eos_path)["schema"]
            )
        return sql.SQL(postgres).format(**schemas)

    def _validate_connection(self, connection):
        return validate_contextgraph_connection(
            connection, self._manifest_store_path, self.eos_path
        )

    @contextmanager
    def _checkpoint_connection(self, connection=None):
        if connection is not None:
            self._validate_connection(connection)
            yield connection
        else:
            with self._connect() as owned:
                yield owned

    def _connect(self):
        if enabled():
            return connect_authority("standalone", self._manifest_store_path)
        connection = sqlite3.connect(self.manifest_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_manifest(self) -> None:
        if enabled():
            with self._connect() as conn:
                require_tables(
                    conn,
                    "graph_artifacts",
                    "runtime_feedback",
                    "artifact_lifecycle_events_local",
                )
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    graph_version TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN ('candidate', 'evaluation_baseline', 'authoritative')),
                    authority_state TEXT NOT NULL CHECK (authority_state IN ('candidate', 'authoritative')),
                    run_mode TEXT,
                    parent_artifact_id TEXT,
                    lineage_root_artifact_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    semantica_commit TEXT NOT NULL,
                    source_manifest_hash TEXT NOT NULL,
                    grounding_manifest_hash TEXT,
                    evaluation_manifest_hash TEXT,
                    governance_ref TEXT,
                    created_by_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    relative_artifact_path TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL DEFAULT 'active',
                    UNIQUE (organization_id, workspace_id, domain_id, graph_version)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_artifacts_scope ON graph_artifacts (organization_id, workspace_id, domain_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_artifacts_lineage ON graph_artifacts (lineage_root_artifact_id)"
            )
            graph_artifact_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(graph_artifacts)"
                ).fetchall()
            }
            if "lineage_id" not in graph_artifact_columns:
                connection.execute(
                    "ALTER TABLE graph_artifacts ADD COLUMN lineage_id TEXT"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_artifacts_lineage_id "
                "ON graph_artifacts (lineage_id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    graph_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    note TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runtime_feedback_artifact "
                "ON runtime_feedback (organization_id, workspace_id, domain_id, artifact_id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_lifecycle_events_local (
                    event_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    origin_id TEXT,
                    operation_id TEXT,
                    actor_id TEXT NOT NULL,
                    previous_state TEXT,
                    next_state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifact_lifecycle_events_local_artifact "
                "ON artifact_lifecycle_events_local (artifact_id, created_at)"
            )

    def _row(self, artifact_id: str, *, connection=None) -> ArtifactMetadata:
        with self._checkpoint_connection(connection) as connection:
            row = connection.execute(
                self._query(
                    "SELECT * FROM graph_artifacts WHERE artifact_id = ? AND organization_id = ? AND workspace_id = ? AND domain_id = ?",
                    "SELECT * FROM {manifest}.graph_artifacts WHERE artifact_id = %s AND organization_id = %s AND workspace_id = %s AND domain_id = %s",
                ),
                (artifact_id, self.organization_id, self.workspace_id, self.domain_id),
            ).fetchone()
        if row is None:
            raise ArtifactNotFound(
                "artifact is not available in the authenticated workspace"
            )
        return ArtifactMetadata(**dict(row))

    def get(self, artifact_id: str) -> ArtifactMetadata:
        return self._row(artifact_id)

    def list_lineage(self, graph_id: str | None = None) -> list[ArtifactMetadata]:
        query = "SELECT * FROM graph_artifacts WHERE organization_id = ? AND workspace_id = ? AND domain_id = ?"
        pg_query = "SELECT * FROM {manifest}.graph_artifacts WHERE organization_id = %s AND workspace_id = %s AND domain_id = %s"
        params: list[Any] = [self.organization_id, self.workspace_id, self.domain_id]
        if graph_id is not None:
            query += " AND graph_id = ?"
            pg_query += " AND graph_id = %s"
            params.append(graph_id)
        query += " ORDER BY created_at, artifact_id"
        pg_query += " ORDER BY created_at, artifact_id"
        with self._connect() as connection:
            rows = connection.execute(self._query(query, pg_query), params).fetchall()
        return [ArtifactMetadata(**dict(row)) for row in rows]

    # --------------------------------------------- Plan A M2 lineage surface

    def record_artifact_origin(
        self,
        *,
        origin_id: str,
        workspace_id: str,
        organization_id: str,
        source_id: str,
        lineage_id: str,
        extraction_run_id: str | None,
        candidate_graph_id: str | None,
        artifact_id: str | None,
        source_identity_ref: str | None,
        relation_kind: str,
        origin_kind: str = "attached",
        authority_state: str = "candidate",
        content_hash: str,
        created_at: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """Plan A M2 (D2, D4): persist an authoritative lineage edge.

        The caller may pass an explicit connection so this insert runs inside
        the same transaction as the artifact-tombstone chain (D7). The
        authority_state column is denormalized so M3's authoritative-block
        check is single-table.
        """
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("lineage_id is required")
        if lineage_id != source_id:
            raise ValueError(
                "lineage_id must equal source_id for source-backed records"
            )
        if organization_id != self.organization_id or workspace_id != self.workspace_id:
            raise ValueError("artifact origin scope mismatch")
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if parsed_created_at.tzinfo is None:
            raise ValueError("artifact origin created_at must be timezone-aware")
        created_at = parsed_created_at.astimezone(UTC).isoformat()
        if connection is not None:
            self._validate_connection(connection)
        conn = connection if connection is not None else self._connect()
        own = connection is None
        try:
            conn.execute(
                self._query(
                    "INSERT INTO contextgraph_artifact_origins ("
                    " origin_id, workspace_id, organization_id, source_id, lineage_id,"
                    " extraction_run_id, candidate_graph_id, artifact_id,"
                    " source_identity_ref, relation_kind, origin_kind,"
                    " authority_state, content_hash, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    "INSERT INTO {eos}.contextgraph_artifact_origins ( origin_id, workspace_id, organization_id, source_id, lineage_id, extraction_run_id, candidate_graph_id, artifact_id, source_identity_ref, relation_kind, origin_kind, authority_state, content_hash, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                ),
                (
                    origin_id,
                    workspace_id,
                    organization_id,
                    source_id,
                    lineage_id,
                    extraction_run_id,
                    candidate_graph_id,
                    artifact_id,
                    source_identity_ref,
                    relation_kind,
                    origin_kind,
                    authority_state,
                    content_hash,
                    created_at,
                ),
            )
            if own:
                conn.commit()
        finally:
            if own:
                conn.close()

    def tombstone_artifact_chain(
        self,
        *,
        source_id: str,
        operation_id: str,
        actor_id: str,
        reason: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[str]:
        """Plan A M2 (D7): tombstone every artifact linked to source_id.

        Runs in one transaction on the supplied connection (or opens one if
        omitted). Returns the list of tombstoned artifact_ids. Each
        transition appends one row to .
        """
        if connection is not None:
            self._validate_connection(connection)
        conn = connection if connection is not None else self._connect()
        own = connection is None
        try:
            rows = conn.execute(
                self._query(
                    "SELECT artifact_id, lifecycle_state FROM graph_artifacts "
                    "WHERE organization_id = ? AND workspace_id = ? AND domain_id = ? "
                    "AND COALESCE(lineage_id, lineage_root_artifact_id) = ? "
                    "AND lifecycle_state = 'active'",
                    "SELECT artifact_id, lifecycle_state FROM {manifest}.graph_artifacts WHERE organization_id = %s AND workspace_id = %s AND domain_id = %s AND COALESCE(lineage_id, lineage_root_artifact_id) = %s AND lifecycle_state = 'active'",
                ),
                (self.organization_id, self.workspace_id, self.domain_id, source_id),
            ).fetchall()
            tombstoned: list[str] = []
            now = datetime.now(UTC).isoformat()
            for row in rows:
                artifact_id = row["artifact_id"]
                conn.execute(
                    self._query(
                        "UPDATE graph_artifacts SET lifecycle_state = 'deleted' "
                        "WHERE artifact_id = ?",
                        "UPDATE {manifest}.graph_artifacts SET lifecycle_state = 'deleted' WHERE artifact_id = %s AND organization_id=%s AND workspace_id=%s AND domain_id=%s",
                    ),
                    (
                        artifact_id,
                        self.organization_id,
                        self.workspace_id,
                        self.domain_id,
                    )
                    if enabled()
                    else (artifact_id,),
                )
                conn.execute(
                    self._query(
                        "INSERT INTO artifact_lifecycle_events_local ("
                        " event_id, artifact_id, origin_id, operation_id, actor_id,"
                        " previous_state, next_state, stage, reason, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        "INSERT INTO {manifest}.artifact_lifecycle_events_local ( organization_id,workspace_id,domain_id,event_id, artifact_id, origin_id, operation_id, actor_id, previous_state, next_state, stage, reason, created_at) VALUES (%s,%s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    ),
                    (
                        *(
                            (self.organization_id, self.workspace_id, self.domain_id)
                            if enabled()
                            else ()
                        ),
                        f"evt_{uuid.uuid4().hex}",
                        artifact_id,
                        None,
                        operation_id,
                        actor_id,
                        "active",
                        "deleted",
                        "artifacts_tombstoned",
                        reason,
                        now,
                    ),
                )
                tombstoned.append(artifact_id)
            if own:
                conn.commit()
            return tombstoned
        finally:
            if own:
                conn.close()

    def create_checkpoint(
        self,
        graph: Any,
        *,
        lineage_id: str,
        graph_id: str,
        graph_version: str,
        artifact_kind: ArtifactKind,
        semantica_commit: str,
        source_manifest_hash: str,
        created_by_user_id: str,
        parent_artifact_id: str | None = None,
        run_mode: str | None = None,
        grounding_manifest_hash: str | None = None,
        evaluation_manifest_hash: str | None = None,
        governance_ref: str | None = None,
        artifact_id: str | None = None,
        connection=None,
    ) -> ArtifactMetadata:
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("lineage_id is required")
        if artifact_kind == "evaluation_baseline" and run_mode != "evaluation_baseline":
            raise ValueError(
                "evaluation_baseline artifacts require run_mode=evaluation_baseline"
            )
        if artifact_kind == "authoritative" and run_mode != "authoritative":
            raise ValueError("authoritative artifacts require run_mode=authoritative")
        if artifact_kind != "authoritative" and run_mode == "authoritative":
            raise ValueError(
                "only authoritative artifacts may use run_mode=authoritative"
            )
        if getattr(graph, "graph_id", None) != graph_id:
            raise ArtifactConflict(
                "graph_id metadata does not match native ContextGraph graph_id"
            )
        artifact_id = artifact_id or f"artifact_{uuid.uuid4().hex}"
        with self._lock, self._checkpoint_connection(connection) as writer:
            if enabled():
                writer.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    (
                        str(self._manifest_store_path)
                        + "\x1f"
                        + self.organization_id
                        + "\x1f"
                        + self.workspace_id
                        + "\x1f"
                        + self.domain_id
                        + "\x1f"
                        + graph_version,
                    ),
                )
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.tmp_root, suffix=".graph.json", delete=False
                ) as handle:
                    temp_path = Path(handle.name)
                graph.save_to_file(str(temp_path))
                payload = temp_path.read_bytes()
                content_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
                folder = {
                    "candidate": self.candidates_root,
                    "evaluation_baseline": self.baselines_root,
                    "authoritative": self.authoritative_root,
                }[artifact_kind]
                relative_path = (
                    Path(folder.relative_to(self.root)) / f"{graph_version}.graph.json"
                )
                final_path = (self.root / relative_path).resolve()
                if folder.resolve() not in final_path.parents:
                    raise ArtifactRepositoryError(
                        "graph_version escapes artifact directory"
                    )
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    final_path.exists()
                    and f"sha256:{hashlib.sha256(final_path.read_bytes()).hexdigest()}"
                    != content_hash
                ):
                    raise ArtifactConflict(
                        "immutable graph_version already contains different content"
                    )
                if not final_path.exists():
                    with (
                        temp_path.open("rb") as source,
                        tempfile.NamedTemporaryFile(
                            dir=final_path.parent,
                            prefix=f".{final_path.name}.",
                            suffix=".tmp",
                            delete=False,
                        ) as destination,
                    ):
                        destination_path = Path(destination.name)
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            destination.write(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                    os.replace(destination_path, final_path)
                    try:
                        directory_fd = os.open(final_path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        pass
                created_at = _utc_now()
                lineage_root = lineage_id
                if parent_artifact_id:
                    parent = self._row(parent_artifact_id, connection=writer)
                    if parent.graph_id != graph_id:
                        raise ArtifactConflict(
                            "parent artifact belongs to a different graph"
                        )
                    parent_lineage_id = (
                        parent.lineage_id or parent.lineage_root_artifact_id
                    )
                    if parent_lineage_id != lineage_id:
                        raise ArtifactConflict(
                            "parent artifact belongs to a different lineage"
                        )
                metadata = ArtifactMetadata(
                    artifact_id=artifact_id,
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    domain_id=self.domain_id,
                    graph_id=graph_id,
                    graph_version=graph_version,
                    artifact_kind=artifact_kind,
                    authority_state="authoritative"
                    if artifact_kind == "authoritative"
                    else "candidate",
                    run_mode=run_mode,
                    parent_artifact_id=parent_artifact_id,
                    lineage_id=lineage_id,
                    lineage_root_artifact_id=lineage_root,
                    content_hash=content_hash,
                    semantica_commit=semantica_commit,
                    source_manifest_hash=source_manifest_hash,
                    grounding_manifest_hash=grounding_manifest_hash,
                    evaluation_manifest_hash=evaluation_manifest_hash,
                    governance_ref=governance_ref,
                    created_by_user_id=created_by_user_id,
                    created_at=created_at,
                    relative_artifact_path=relative_path.as_posix(),
                    lifecycle_state="active",
                )
                try:
                    begin_write(writer) if connection is None else None
                    fields = metadata.as_dict()
                    writer.execute(
                        self._query(
                            """INSERT INTO graph_artifacts (
                        artifact_id, organization_id, workspace_id, domain_id,
                        graph_id, graph_version, artifact_kind, authority_state,
                        run_mode, parent_artifact_id, lineage_id,
                        lineage_root_artifact_id, content_hash, semantica_commit,
                        source_manifest_hash, grounding_manifest_hash,
                        evaluation_manifest_hash, governance_ref,
                        created_by_user_id, created_at, relative_artifact_path,
                        lifecycle_state
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            "INSERT INTO {manifest}.graph_artifacts (\n                            artifact_id, organization_id, workspace_id, domain_id,\n                            graph_id, graph_version, artifact_kind, authority_state,\n                            run_mode, parent_artifact_id, lineage_id,\n                            lineage_root_artifact_id, content_hash, semantica_commit,\n                            source_manifest_hash, grounding_manifest_hash,\n                            evaluation_manifest_hash, governance_ref,\n                            created_by_user_id, created_at, relative_artifact_path,\n                            lifecycle_state\n                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        ),
                        tuple(
                            fields[name]
                            for name in (
                                "artifact_id",
                                "organization_id",
                                "workspace_id",
                                "domain_id",
                                "graph_id",
                                "graph_version",
                                "artifact_kind",
                                "authority_state",
                                "run_mode",
                                "parent_artifact_id",
                                "lineage_id",
                                "lineage_root_artifact_id",
                                "content_hash",
                                "semantica_commit",
                                "source_manifest_hash",
                                "grounding_manifest_hash",
                                "evaluation_manifest_hash",
                                "governance_ref",
                                "created_by_user_id",
                                "created_at",
                                "relative_artifact_path",
                                "lifecycle_state",
                            )
                        ),
                    )
                    writer.commit() if connection is None else None
                except integrity_errors() as exc:
                    writer.rollback() if connection is None else None
                    raise ArtifactConflict(
                        "graph_version or artifact_id already exists"
                    ) from exc
                return metadata
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    def artifact_path(self, artifact_id: str) -> Path:
        metadata = self._row(artifact_id)
        path = (self.root / metadata.relative_artifact_path).resolve()
        if self.root not in path.parents:
            raise ArtifactRepositoryError("artifact path escaped repository root")
        return path

    def load_graph(self, artifact_id: str, *, graph_factory: Any | None = None) -> Any:
        metadata = self._row(artifact_id)
        path = self.artifact_path(artifact_id)
        payload = path.read_bytes()
        if f"sha256:{hashlib.sha256(payload).hexdigest()}" != metadata.content_hash:
            raise ArtifactRepositoryError("artifact content hash mismatch")
        if graph_factory is None:
            from semantica.context import ContextGraph

            graph_factory = ContextGraph
        graph = graph_factory()
        graph.load_from_file(str(path))
        # Semantica's JSON serializer flattens these node fields into the
        # persisted ``properties`` object, while its loader also restores the
        # corresponding first-class attributes.  Remove only those synthetic
        # copies so a save/load cycle preserves the native graph exactly.
        for node in graph.nodes.values():
            properties = getattr(node, "properties", None)
            if not isinstance(properties, dict):
                continue
            synthetic = {
                "content": getattr(node, "content", None),
                "valid_from": getattr(node, "valid_from", None),
                "valid_until": getattr(node, "valid_until", None),
            }
            for field, value in synthetic.items():
                if field in properties and properties[field] == value:
                    properties.pop(field)
        return graph

    def pin_evaluation_baseline(
        self,
        artifact_id: str,
        *,
        evaluation_run_id: str,
        evaluation_manifest_hash: str,
    ) -> ArtifactMetadata:
        source = self._row(artifact_id)
        if source.artifact_kind not in {"candidate", "evaluation_baseline"}:
            raise ArtifactConflict(
                "authoritative artifact cannot be pinned as Phase1-001 baseline"
            )
        if source.artifact_kind == "evaluation_baseline":
            if source.evaluation_manifest_hash != evaluation_manifest_hash:
                raise ArtifactConflict("baseline evaluation manifest hash mismatch")
            return source
        graph = self.load_graph(artifact_id)
        return self.create_checkpoint(
            graph,
            lineage_id=source.lineage_id or source.lineage_root_artifact_id,
            graph_id=source.graph_id,
            graph_version=f"{source.graph_version}-baseline-{evaluation_run_id}",
            artifact_kind="evaluation_baseline",
            run_mode="evaluation_baseline",
            semantica_commit=source.semantica_commit,
            source_manifest_hash=source.source_manifest_hash,
            grounding_manifest_hash=source.grounding_manifest_hash,
            evaluation_manifest_hash=evaluation_manifest_hash,
            created_by_user_id=source.created_by_user_id,
            parent_artifact_id=source.artifact_id,
        )

    def record_feedback(
        self,
        *,
        artifact_id: str,
        graph_version: str,
        content_hash: str,
        feedback_type: str,
        note: str,
        source_refs: list[str],
        actor_id: str,
    ) -> str:
        metadata = self._row(artifact_id)
        if (
            metadata.graph_version != graph_version
            or metadata.content_hash != content_hash
        ):
            raise ArtifactConflict("feedback graph binding is stale")
        if not feedback_type or not note or not actor_id:
            raise ValueError("feedback_type, note, and actor_id are required")
        feedback_id = f"feedback_{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                self._query(
                    "INSERT INTO runtime_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    "INSERT INTO {manifest}.runtime_feedback VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                ),
                (
                    feedback_id,
                    self.organization_id,
                    self.workspace_id,
                    self.domain_id,
                    artifact_id,
                    graph_version,
                    content_hash,
                    feedback_type,
                    note,
                    json.dumps(
                        sorted(set(source_refs)),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    actor_id,
                    _utc_now(),
                ),
            )
        return feedback_id

    def list_feedback(self, artifact_id: str) -> list[dict[str, Any]]:
        self._row(artifact_id)
        with self._connect() as connection:
            rows = connection.execute(
                self._query(
                    "SELECT * FROM runtime_feedback WHERE artifact_id = ? AND organization_id = ? AND workspace_id = ? AND domain_id = ? ORDER BY created_at, feedback_id",
                    "SELECT * FROM {manifest}.runtime_feedback WHERE artifact_id = %s AND organization_id = %s AND workspace_id = %s AND domain_id = %s ORDER BY created_at, feedback_id",
                ),
                (artifact_id, self.organization_id, self.workspace_id, self.domain_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_manifest(self) -> str:
        rows = [metadata.as_dict() for metadata in self.list_lineage()]
        return json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
