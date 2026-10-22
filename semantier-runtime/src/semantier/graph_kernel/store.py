from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from contracts.normative_conflict import (
    ConflictCoreV1,
    NormativeCompatibilityResultV1,
    NormativeConflictRevalidationSuccessorV1,
    NormativeConflictRevalidationTargetV1,
)
from eos.artifact_hashing import canonical_json, content_hash
from eos.db import ensure_graph_ingestion_schema, get_conn, now_iso

from .ids import assertion_id as deterministic_assertion_id
from .ids import node_id as deterministic_node_id
from .models import GraphAssertion, GraphNode, GraphSnapshot


GRAPH_AUTHORITY_REQUIRED = "GRAPH_AUTHORITY_REQUIRED"
GRAPH_SCOPE_REQUIRED = "GRAPH_SCOPE_REQUIRED"
GRAPH_CROSS_ORG_REJECTED = "GRAPH_CROSS_ORG_REJECTED"
GRAPH_SNAPSHOT_REQUIRED = "GRAPH_SNAPSHOT_REQUIRED"
GRAPH_ASSERTION_NOT_ACTIVATABLE = "GRAPH_ASSERTION_NOT_ACTIVATABLE"
GRAPH_ASSERTION_INVALID_TRANSITION = "GRAPH_ASSERTION_INVALID_TRANSITION"
GRAPH_UNMANAGED_DB_PATH_REJECTED = "GRAPH_UNMANAGED_DB_PATH_REJECTED"


class GraphKernelStore:
    def __init__(self, db_path: str | Path | None = None, *, _allow_offline_path: bool = False) -> None:
        if db_path is not None and not _allow_offline_path:
            raise ValueError(GRAPH_UNMANAGED_DB_PATH_REJECTED)
        self._db_path = Path(db_path) if db_path is not None else None
        self._allow_offline_path = _allow_offline_path
        self.ensure_schema()

    @classmethod
    def for_offline_path(cls, db_path: str | Path) -> GraphKernelStore:
        return cls(db_path, _allow_offline_path=True)

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled() and self._db_path is not None:
            raise hosted.HostedStorageUnavailable('offline SQLite graph operation requires explicit local backend')
        if self._db_path is None:
            return get_conn()
        if not self._allow_offline_path:
            raise ValueError(GRAPH_UNMANAGED_DB_PATH_REJECTED)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'graph_assertion_events', 'graph_assertions', 'graph_governance_events', 'graph_index_freshness', 'graph_ingestion_runs', 'graph_node_search', 'graph_nodes', 'graph_projection_contexts', 'graph_snapshot_assertions', 'graph_snapshots', 'graph_source_index_manifests')
                return
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                  node_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  node_type TEXT NOT NULL,
                  canonical_key TEXT NOT NULL,
                  display_name TEXT,
                  properties_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, node_type, canonical_key)
                );

                CREATE TABLE IF NOT EXISTS graph_assertions (
                  assertion_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  subject_id TEXT NOT NULL,
                  predicate TEXT NOT NULL,
                  object_id TEXT NOT NULL,
                  assertion_class TEXT NOT NULL,
                  semantic_tier TEXT,
                  authority_role TEXT NOT NULL,
                  governance_state TEXT NOT NULL,
                  valid_from TEXT,
                  valid_to TEXT,
                  source_ref TEXT NOT NULL,
                  source_anchor TEXT,
                  extraction_method TEXT NOT NULL,
                  extractor_version TEXT NOT NULL,
                  confidence REAL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(subject_id) REFERENCES graph_nodes(node_id),
                  FOREIGN KEY(object_id) REFERENCES graph_nodes(node_id)
                );

                CREATE TABLE IF NOT EXISTS graph_assertion_events (
                  event_id TEXT PRIMARY KEY,
                  assertion_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_payload_json TEXT NOT NULL,
                  prior_event_hash TEXT,
                  event_hash TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES graph_assertions(assertion_id)
                );

                CREATE TABLE IF NOT EXISTS graph_snapshots (
                  graph_version_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  ontology_version TEXT NOT NULL,
                  resolver_version TEXT NOT NULL,
                  active_assertion_set_hash TEXT NOT NULL,
                  snapshot_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_snapshot_assertions (
                  graph_version_id TEXT NOT NULL,
                  assertion_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  PRIMARY KEY(graph_version_id, assertion_id),
                  FOREIGN KEY(graph_version_id) REFERENCES graph_snapshots(graph_version_id),
                  FOREIGN KEY(assertion_id) REFERENCES graph_assertions(assertion_id)
                );
                CREATE INDEX IF NOT EXISTS idx_graph_snapshot_assertions_assertion
                  ON graph_snapshot_assertions(assertion_id, graph_version_id);

                CREATE TABLE IF NOT EXISTS graph_projection_contexts (
                  projection_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  source_ref TEXT NOT NULL,
                  focus_ref TEXT,
                  graph_snapshot_ref TEXT,
                  as_of TEXT,
                  lens TEXT NOT NULL,
                  context_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_projection_contexts_org_workspace
                  ON graph_projection_contexts(organization_id, workspace_id, updated_at);

                CREATE TABLE IF NOT EXISTS graph_governance_events (
                  event_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  target_object_ref TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_payload_json TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_governance_events_target
                  ON graph_governance_events(organization_id, target_object_ref, occurred_at);

                CREATE TABLE IF NOT EXISTS graph_ingestion_runs (
                  ingestion_run_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  source_ref TEXT NOT NULL,
                  trigger_type TEXT NOT NULL,
                  requested_by TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  requested_fingerprint TEXT NOT NULL,
                  observed_source_hash TEXT,
                  prior_successful_run_ref TEXT,
                  status TEXT NOT NULL,
                  lease_owner TEXT,
                  lease_expires_at TEXT,
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  max_attempts INTEGER NOT NULL DEFAULT 3,
                  next_retry_at TEXT,
                  error_code TEXT,
                  error_detail_ref TEXT,
                  resulting_graph_snapshot_ref TEXT,
                  resulting_graph_snapshot_hash TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT,
                  UNIQUE(organization_id, source_ref, idempotency_key),
                  FOREIGN KEY(prior_successful_run_ref) REFERENCES graph_ingestion_runs(ingestion_run_id),
                  FOREIGN KEY(resulting_graph_snapshot_ref) REFERENCES graph_snapshots(graph_version_id)
                );

                CREATE TABLE IF NOT EXISTS graph_source_index_manifests (
                  organization_id TEXT NOT NULL,
                  source_ref TEXT NOT NULL,
                  workspace_id TEXT,
                  index_fingerprint TEXT NOT NULL,
                  source_hash TEXT NOT NULL,
                  normalized_structure_hash TEXT NOT NULL,
                  extractor_version TEXT NOT NULL,
                  structure_schema_version TEXT NOT NULL,
                  ontology_compatibility_version TEXT NOT NULL,
                  graph_projection_version TEXT NOT NULL,
                  last_successful_run_ref TEXT NOT NULL,
                  graph_snapshot_ref TEXT NOT NULL,
                  graph_snapshot_hash TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, source_ref),
                  FOREIGN KEY(last_successful_run_ref) REFERENCES graph_ingestion_runs(ingestion_run_id),
                  FOREIGN KEY(graph_snapshot_ref) REFERENCES graph_snapshots(graph_version_id)
                );

                CREATE TABLE IF NOT EXISTS graph_index_freshness (
                  organization_id TEXT NOT NULL,
                  source_ref TEXT NOT NULL,
                  workspace_id TEXT,
                  status TEXT NOT NULL,
                  indexed_at TEXT,
                  source_fingerprint TEXT,
                  last_successful_index_ref TEXT,
                  graph_snapshot_ref TEXT,
                  graph_snapshot_hash TEXT,
                  error_code TEXT,
                  error_detail_ref TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, source_ref)
                );
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject
                  ON graph_assertions(organization_id, subject_id, predicate, governance_state);
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject_ordered
                  ON graph_assertions(organization_id, subject_id, predicate, governance_state, object_id);
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_object
                  ON graph_assertions(organization_id, object_id, predicate, governance_state);
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_class_state
                  ON graph_assertions(organization_id, assertion_class, governance_state);
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_effective
                  ON graph_assertions(organization_id, valid_from, valid_to);
                CREATE INDEX IF NOT EXISTS idx_graph_assertion_events_stream
                  ON graph_assertion_events(assertion_id, occurred_at);
                """
            )
            ensure_graph_ingestion_schema(conn)
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS graph_node_search
                    USING fts5(node_id UNINDEXED, organization_id UNINDEXED, content)
                    """
                )
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_node_search (
                      node_id TEXT PRIMARY KEY,
                      organization_id TEXT NOT NULL,
                      content TEXT NOT NULL
                    )
                    """
                )

    def upsert_node(
        self,
        *,
        organization_id: str,
        node_type: str,
        canonical_key: str,
        display_name: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> GraphNode:
        properties = properties or {}
        payload = {
            "organization_id": organization_id,
            "node_type": node_type,
            "canonical_key": canonical_key,
            "display_name": display_name,
            "properties": properties,
        }
        content = content_hash(payload)
        nid = deterministic_node_id(
            organization_id=organization_id,
            node_type=node_type,
            canonical_key=canonical_key,
        )
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_nodes (
                  node_id, organization_id, node_type, canonical_key,
                  display_name, properties_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                  display_name = excluded.display_name,
                  properties_json = excluded.properties_json,
                  content_hash = excluded.content_hash
                """, """
                INSERT INTO graph_nodes (
                  node_id, organization_id, node_type, canonical_key,
                  display_name, properties_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(node_id) DO UPDATE SET
                  display_name = excluded.display_name,
                  properties_json = excluded.properties_json,
                  content_hash = excluded.content_hash
                """),
                (
                    nid,
                    organization_id,
                    node_type,
                    canonical_key,
                    display_name,
                    canonical_json(properties),
                    content,
                    now_iso(),
                ),
            )
            search_content = " ".join(
                [
                    node_type,
                    canonical_key,
                    display_name or "",
                    canonical_json(properties),
                ]
            )
            conn.execute(hosted.sql_query("DELETE FROM graph_node_search WHERE node_id = ?", 'DELETE FROM graph_node_search WHERE node_id = %s'), (nid,))
            conn.execute(
                hosted.sql_query("INSERT INTO graph_node_search (node_id, organization_id, content) VALUES (?, ?, ?)", 'INSERT INTO graph_node_search (node_id, organization_id, content) VALUES (%s, %s, %s)'),
                (nid, organization_id, search_content),
            )
        return GraphNode(nid, organization_id, node_type, canonical_key, display_name, properties, content)

    def propose_assertion(
        self,
        *,
        organization_id: str,
        subject_id: str,
        predicate: str,
        object_id: str,
        assertion_class: str,
        semantic_tier: str | None,
        authority_role: str,
        governance_state: str,
        valid_from: str | None,
        valid_to: str | None,
        source_ref: str,
        source_anchor: str | None,
        extraction_method: str,
        extractor_version: str,
        confidence: float | None = None,
        actor_ref: str = "graph-kernel",
    ) -> GraphAssertion:
        if governance_state == "active" and not self._auto_activatable(assertion_class, authority_role, extraction_method):
            raise ValueError(GRAPH_AUTHORITY_REQUIRED)
        aid = deterministic_assertion_id(
            organization_id=organization_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            source_ref=source_ref,
            source_anchor=source_anchor,
        )
        payload = {
            "organization_id": organization_id,
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "assertion_class": assertion_class,
            "semantic_tier": semantic_tier,
            "authority_role": authority_role,
            "governance_state": governance_state,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "source_ref": source_ref,
            "source_anchor": source_anchor,
            "extraction_method": extraction_method,
            "extractor_version": extractor_version,
            "confidence": confidence,
        }
        assertion_hash = content_hash(payload)
        with self._connect() as conn:
            subject = conn.execute(hosted.sql_query("SELECT organization_id FROM graph_nodes WHERE node_id = ?", 'SELECT organization_id FROM graph_nodes WHERE node_id = %s'), (subject_id,)).fetchone()
            obj = conn.execute(hosted.sql_query("SELECT organization_id FROM graph_nodes WHERE node_id = ?", 'SELECT organization_id FROM graph_nodes WHERE node_id = %s'), (object_id,)).fetchone()
            if subject is None or obj is None:
                raise KeyError("subject_id or object_id not found")
            if subject["organization_id"] != organization_id or obj["organization_id"] != organization_id:
                raise ValueError(GRAPH_CROSS_ORG_REJECTED)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_assertions (
                  assertion_id, organization_id, subject_id, predicate, object_id,
                  assertion_class, semantic_tier, authority_role, governance_state,
                  valid_from, valid_to, source_ref, source_anchor, extraction_method,
                  extractor_version, confidence, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assertion_id) DO UPDATE SET
                  governance_state = excluded.governance_state,
                  content_hash = excluded.content_hash
                """, """
                INSERT INTO graph_assertions (
                  assertion_id, organization_id, subject_id, predicate, object_id,
                  assertion_class, semantic_tier, authority_role, governance_state,
                  valid_from, valid_to, source_ref, source_anchor, extraction_method,
                  extractor_version, confidence, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(assertion_id) DO UPDATE SET
                  governance_state = excluded.governance_state,
                  content_hash = excluded.content_hash
                """),
                (
                    aid, organization_id, subject_id, predicate, object_id,
                    assertion_class, semantic_tier, authority_role, governance_state,
                    valid_from, valid_to, source_ref, source_anchor, extraction_method,
                    extractor_version, confidence, assertion_hash, now_iso(),
                ),
            )
            self._record_event(
                conn,
                assertion_id=aid,
                event_type=governance_state,
                actor_ref=actor_ref,
                payload={"content_hash": assertion_hash},
            )
        return GraphAssertion(
            aid, organization_id, subject_id, predicate, object_id,
            assertion_class, semantic_tier, authority_role, governance_state,
            valid_from, valid_to, source_ref, source_anchor, extraction_method,
            extractor_version, confidence, assertion_hash,
        )

    def activate_assertion(self, *, assertion_id: str, actor_ref: str, governance_event_ref: str) -> GraphAssertion:
        with self._connect() as conn:
            row = conn.execute(hosted.sql_query("SELECT * FROM graph_assertions WHERE assertion_id = ?", 'SELECT * FROM graph_assertions WHERE assertion_id = %s'), (assertion_id,)).fetchone()
            if row is None:
                raise KeyError(assertion_id)
            if row["governance_state"] not in {"validated", "approved"}:
                raise ValueError(GRAPH_ASSERTION_NOT_ACTIVATABLE)
            conn.execute(hosted.sql_query("UPDATE graph_assertions SET governance_state = 'active' WHERE assertion_id = ?", "UPDATE graph_assertions SET governance_state = 'active' WHERE assertion_id = %s"), (assertion_id,))
            self._record_event(
                conn,
                assertion_id=assertion_id,
                event_type="activated",
                actor_ref=actor_ref,
                payload={"governance_event_ref": governance_event_ref},
            )
            updated = conn.execute(hosted.sql_query("SELECT * FROM graph_assertions WHERE assertion_id = ?", 'SELECT * FROM graph_assertions WHERE assertion_id = %s'), (assertion_id,)).fetchone()
        return self._assertion_obj(updated)

    def validate_assertion(self, *, assertion_id: str, actor_ref: str, governance_event_ref: str) -> GraphAssertion:
        return self._transition_assertion(
            assertion_id=assertion_id,
            actor_ref=actor_ref,
            governance_event_ref=governance_event_ref,
            allowed_from={"candidate"},
            target_state="validated",
            event_type="validated",
        )

    def approve_assertion(self, *, assertion_id: str, actor_ref: str, governance_event_ref: str) -> GraphAssertion:
        return self._transition_assertion(
            assertion_id=assertion_id,
            actor_ref=actor_ref,
            governance_event_ref=governance_event_ref,
            allowed_from={"validated"},
            target_state="approved",
            event_type="approved",
        )

    def reject_assertion(self, *, assertion_id: str, actor_ref: str, governance_event_ref: str) -> GraphAssertion:
        return self._transition_assertion(
            assertion_id=assertion_id,
            actor_ref=actor_ref,
            governance_event_ref=governance_event_ref,
            allowed_from={"candidate", "validated", "approved"},
            target_state="rejected",
            event_type="rejected",
        )

    def deprecate_assertion(
        self,
        *,
        assertion_id: str,
        actor_ref: str,
        governance_event_ref: str,
        deprecated_at: str | None = None,
    ) -> GraphAssertion:
        return self._transition_assertion(
            assertion_id=assertion_id,
            actor_ref=actor_ref,
            governance_event_ref=governance_event_ref,
            allowed_from={"candidate", "validated", "approved", "active"},
            target_state="deprecated",
            event_type="deprecated",
            valid_to=deprecated_at or now_iso(),
        )

    def record_assertion_governance_event(
        self,
        *,
        organization_id: str,
        assertion_id: str,
        event_type: str,
        actor_ref: str,
        governance_event_ref: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_payload = dict(payload or {})
        event_payload["governance_event_ref"] = governance_event_ref
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT organization_id
                FROM graph_assertions
                WHERE assertion_id = ?
                """, """
                SELECT organization_id
                FROM graph_assertions
                WHERE assertion_id = %s
                """),
                (assertion_id,),
            ).fetchone()
            if row is None:
                raise KeyError(assertion_id)
            if row["organization_id"] != organization_id:
                raise ValueError(GRAPH_CROSS_ORG_REJECTED)
            return self._record_event(
                conn,
                assertion_id=assertion_id,
                event_type=event_type,
                actor_ref=actor_ref,
                payload=event_payload,
            )

    def record_graph_governance_event(
        self,
        *,
        organization_id: str,
        target_object_ref: str,
        event_type: str,
        actor_ref: str,
        governance_event_ref: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_payload = dict(payload or {})
        event_payload["governance_event_ref"] = governance_event_ref
        timestamp = now_iso()
        event_hash = content_hash(
            {
                "organization_id": organization_id,
                "target_object_ref": target_object_ref,
                "event_type": event_type,
                "actor_ref": actor_ref,
                "payload": event_payload,
            }
        )
        event_id = f"graph_governance_event_{uuid4().hex}"
        with self._connect() as conn:
            assertion = conn.execute(
                hosted.sql_query("""
                SELECT organization_id
                FROM graph_assertions
                WHERE assertion_id = ?
                """, """
                SELECT organization_id
                FROM graph_assertions
                WHERE assertion_id = %s
                """),
                (target_object_ref,),
            ).fetchone()
            node = conn.execute(
                hosted.sql_query("""
                SELECT organization_id
                FROM graph_nodes
                WHERE node_id = ?
                """, """
                SELECT organization_id
                FROM graph_nodes
                WHERE node_id = %s
                """),
                (target_object_ref,),
            ).fetchone()
            owner = assertion or node
            if owner is None:
                raise KeyError(target_object_ref)
            if owner["organization_id"] != organization_id:
                raise ValueError(GRAPH_CROSS_ORG_REJECTED)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_governance_events (
                  event_id, organization_id, target_object_ref, event_type,
                  actor_ref, event_payload_json, event_hash, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO graph_governance_events (
                  event_id, organization_id, target_object_ref, event_type,
                  actor_ref, event_payload_json, event_hash, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    organization_id,
                    target_object_ref,
                    event_type,
                    actor_ref,
                    canonical_json(event_payload),
                    event_hash,
                    timestamp,
                ),
            )
        return event_id

    def deprecate_assertions_for_source(
        self,
        *,
        organization_id: str,
        source_ref: str,
        actor_ref: str,
        deprecated_at: str | None = None,
    ) -> list[str]:
        deprecated_at = deprecated_at or now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT assertion_id
                FROM graph_assertions
                WHERE organization_id = ?
                  AND source_ref = ?
                  AND governance_state IN ('candidate', 'validated', 'approved', 'active')
                ORDER BY assertion_id
                """, """
                SELECT assertion_id
                FROM graph_assertions
                WHERE organization_id = %s
                  AND source_ref = %s
                  AND governance_state IN ('candidate', 'validated', 'approved', 'active')
                ORDER BY assertion_id
                """),
                (organization_id, source_ref),
            ).fetchall()
            assertion_ids = [row["assertion_id"] for row in rows]
            for assertion_id in assertion_ids:
                conn.execute(
                    hosted.sql_query("""
                    UPDATE graph_assertions
                    SET governance_state = 'deprecated',
                        valid_to = COALESCE(valid_to, ?)
                    WHERE assertion_id = ?
                    """, """
                    UPDATE graph_assertions
                    SET governance_state = 'deprecated',
                        valid_to = COALESCE(valid_to, %s)
                    WHERE assertion_id = %s
                    """),
                    (deprecated_at, assertion_id),
                )
                self._record_event(
                    conn,
                    assertion_id=assertion_id,
                    event_type="deprecated",
                    actor_ref=actor_ref,
                    payload={"source_ref": source_ref, "deprecated_at": deprecated_at},
                )
        return assertion_ids

    def save_projection_context(
        self,
        *,
        projection_id: str,
        organization_id: str,
        workspace_id: str | None,
        source_ref: str,
        focus_ref: str | None,
        graph_snapshot_ref: str | None,
        as_of: str | None,
        lens: str,
    ) -> dict[str, Any]:
        payload = {
            "projection_id": projection_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_ref": source_ref,
            "focus_ref": focus_ref,
            "graph_snapshot_ref": graph_snapshot_ref,
            "as_of": as_of,
            "lens": lens,
        }
        context_hash = content_hash(payload)
        timestamp = now_iso()
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_projection_contexts (
                  projection_id, organization_id, workspace_id, source_ref,
                  focus_ref, graph_snapshot_ref, as_of, lens, context_hash,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(projection_id) DO UPDATE SET
                  organization_id = excluded.organization_id,
                  workspace_id = excluded.workspace_id,
                  source_ref = excluded.source_ref,
                  focus_ref = excluded.focus_ref,
                  graph_snapshot_ref = excluded.graph_snapshot_ref,
                  as_of = excluded.as_of,
                  lens = excluded.lens,
                  context_hash = excluded.context_hash,
                  updated_at = excluded.updated_at
                """, """
                INSERT INTO graph_projection_contexts (
                  projection_id, organization_id, workspace_id, source_ref,
                  focus_ref, graph_snapshot_ref, as_of, lens, context_hash,
                  created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(projection_id) DO UPDATE SET
                  organization_id = excluded.organization_id,
                  workspace_id = excluded.workspace_id,
                  source_ref = excluded.source_ref,
                  focus_ref = excluded.focus_ref,
                  graph_snapshot_ref = excluded.graph_snapshot_ref,
                  as_of = excluded.as_of,
                  lens = excluded.lens,
                  context_hash = excluded.context_hash,
                  updated_at = excluded.updated_at
                """),
                (
                    projection_id,
                    organization_id,
                    workspace_id,
                    source_ref,
                    focus_ref,
                    graph_snapshot_ref,
                    as_of,
                    lens,
                    context_hash,
                    timestamp,
                    timestamp,
                ),
            )
        return {**payload, "context_hash": context_hash, "updated_at": timestamp}

    def get_projection_context(
        self,
        *,
        organization_id: str,
        projection_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        filters = [hosted.sql_query("organization_id = ?", 'organization_id = %s'), hosted.sql_query("projection_id = ?", 'projection_id = %s')]
        params: list[Any] = [organization_id, projection_id]
        if workspace_id is not None:
            filters.append(hosted.sql_query("(workspace_id IS NULL OR workspace_id = ?)", '(workspace_id IS NULL OR workspace_id = %s)'))
            params.append(workspace_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT *
                FROM graph_projection_contexts
                WHERE {" AND ".join(filters)}
                """,
                tuple(params),
            ).fetchone()
        return dict(row) if row else None

    def get_neighbors(
        self,
        *,
        organization_id: str,
        node_id: str,
        predicates: list[str],
        governance_states: list[str] | None = None,
        as_of: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not organization_id:
            raise ValueError(GRAPH_SCOPE_REQUIRED)
        if not predicates:
            raise ValueError("predicates are required")
        states = governance_states or ["active"]
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT a.*, n.node_type AS object_type, n.canonical_key AS object_key
                FROM graph_assertions a
                JOIN graph_nodes n ON n.node_id = a.object_id
                WHERE a.organization_id = ?
                  AND a.subject_id = ?
                  AND a.predicate IN ({",".join("?" for _ in predicates)})
                  AND a.governance_state IN ({",".join("?" for _ in states)})
                ORDER BY a.predicate, a.object_id
                LIMIT ?
                """, f"\n                SELECT a.*, n.node_type AS object_type, n.canonical_key AS object_key\n                FROM graph_assertions a\n                JOIN graph_nodes n ON n.node_id = a.object_id\n                WHERE a.organization_id = %s\n                  AND a.subject_id = %s\n                  AND a.predicate IN ({','.join(('%s' for _ in predicates))})\n                  AND a.governance_state IN ({','.join(('%s' for _ in states))})\n                ORDER BY a.predicate, a.object_id\n                LIMIT %s\n                "),
                (organization_id, node_id, *predicates, *states, limit),
            ).fetchall()
        return [self._assertion_dict(row) for row in rows if _effective(row, as_of)]

    def trace_lineage(
        self,
        *,
        organization_id: str,
        start_node_id: str,
        predicates: list[str],
        max_depth: int,
        governance_states: list[str] | None = None,
        snapshot_id: str | None = None,
        as_of: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if not organization_id:
            raise ValueError(GRAPH_SCOPE_REQUIRED)
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not predicates:
            raise ValueError("predicates are required")
        states = governance_states or ["active"]
        results: list[dict[str, Any]] = []
        frontier: list[tuple[str, tuple[str, ...]]] = [(start_node_id, (start_node_id,))]
        with self._connect() as conn:
            for depth in range(1, max_depth + 1):
                if not frontier or len(results) >= limit:
                    break
                subjects = sorted({node_id for node_id, _path in frontier})
                paths_by_subject: dict[str, list[tuple[str, ...]]] = {subject: [] for subject in subjects}
                for node_id, path in frontier:
                    paths_by_subject[node_id].append(path)

                remaining = limit - len(results)
                row_cap = max(remaining * 4, remaining)
                rows = conn.execute(
                    hosted.sql_query(f"""
                    SELECT a.assertion_id, a.subject_id, a.object_id, a.predicate
                    FROM graph_assertions a
                    {snapshot_join(snapshot_id)}
                    WHERE a.organization_id = ?
                      AND a.subject_id IN ({",".join("?" for _ in subjects)})
                      AND a.predicate IN ({",".join("?" for _ in predicates)})
                      AND a.governance_state IN ({",".join("?" for _ in states)})
                      {effective_predicate(as_of)}
                    ORDER BY a.subject_id, a.predicate, a.object_id, a.assertion_id
                    LIMIT ?
                    """, f"\n                    SELECT a.assertion_id, a.subject_id, a.object_id, a.predicate\n                    FROM graph_assertions a\n                    {snapshot_join(snapshot_id)}\n                    WHERE a.organization_id = %s\n                      AND a.subject_id IN ({','.join(('%s' for _ in subjects))})\n                      AND a.predicate IN ({','.join(('%s' for _ in predicates))})\n                      AND a.governance_state IN ({','.join(('%s' for _ in states))})\n                      {effective_predicate(as_of)}\n                    ORDER BY a.subject_id, a.predicate, a.object_id, a.assertion_id\n                    LIMIT %s\n                    "),
                    tuple(
                        ([snapshot_id] if snapshot_id is not None else [])
                        + [organization_id]
                        + subjects
                        + [*predicates, *states]
                        + ([as_of, as_of] if as_of is not None else [])
                        + [row_cap]
                    ),
                ).fetchall()

                next_frontier: list[tuple[str, tuple[str, ...]]] = []
                for row in rows:
                    for path in paths_by_subject[row["subject_id"]]:
                        object_id = row["object_id"]
                        if object_id in path:
                            continue
                        next_path = (*path, object_id)
                        results.append(
                            {
                                "assertion_id": row["assertion_id"],
                                "subject_id": row["subject_id"],
                                "object_id": object_id,
                                "predicate": row["predicate"],
                                "depth": depth,
                                "path": ">".join(next_path),
                            }
                        )
                        if len(results) >= limit:
                            break
                        next_frontier.append((object_id, next_path))
                    if len(results) >= limit:
                        break
                frontier = next_frontier
        return results

    def search_nodes(self, *, organization_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if hosted.is_postgres(conn):
                prefix = re.fullmatch(r'\w+\*', query, flags=re.UNICODE) is not None
                query_function = 'to_tsquery' if prefix else 'websearch_to_tsquery'
                term = query[:-1] + ':*' if prefix else query
                rows = conn.execute(f"""
                    SELECT s.node_id,n.node_type,n.canonical_key,n.display_name
                    FROM graph_node_search s JOIN graph_nodes n ON n.node_id=s.node_id
                    WHERE s.organization_id=%s
                      AND to_tsvector('simple',s.content) @@ {query_function}('simple',%s)
                    LIMIT %s
                """, (organization_id,term,limit)).fetchall()
                return [dict(row) for row in rows]
            try:
                rows = conn.execute(
                    hosted.sql_query("""
                    SELECT s.node_id, n.node_type, n.canonical_key, n.display_name
                    FROM graph_node_search s
                    JOIN graph_nodes n ON n.node_id = s.node_id
                    WHERE s.organization_id = ?
                      AND graph_node_search MATCH ?
                    LIMIT ?
                    """, """
                    SELECT s.node_id, n.node_type, n.canonical_key, n.display_name
                    FROM graph_node_search s
                    JOIN graph_nodes n ON n.node_id = s.node_id
                    WHERE s.organization_id = %s
                      AND graph_node_search MATCH %s
                    LIMIT %s
                    """),
                    (organization_id, query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    hosted.sql_query("""
                    SELECT s.node_id, n.node_type, n.canonical_key, n.display_name
                    FROM graph_node_search s
                    JOIN graph_nodes n ON n.node_id = s.node_id
                    WHERE s.organization_id = ?
                      AND s.content LIKE ?
                    LIMIT ?
                    """, """
                    SELECT s.node_id, n.node_type, n.canonical_key, n.display_name
                    FROM graph_node_search s
                    JOIN graph_nodes n ON n.node_id = s.node_id
                    WHERE s.organization_id = %s
                      AND s.content LIKE %s
                    LIMIT %s
                    """),
                    (organization_id, f"%{query}%", limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def materialize_snapshot(
        self,
        *,
        organization_id: str,
        ontology_version: str,
        resolver_version: str,
        governance_states: list[str] | None = None,
    ) -> GraphSnapshot:
        states = governance_states or ["active"]
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT assertion_id, content_hash
                FROM graph_assertions
                WHERE organization_id = ?
                  AND governance_state IN ({",".join("?" for _ in states)})
                ORDER BY assertion_id
                """, f"\n                SELECT assertion_id, content_hash\n                FROM graph_assertions\n                WHERE organization_id = %s\n                  AND governance_state IN ({','.join(('%s' for _ in states))})\n                ORDER BY assertion_id\n                "),
                (organization_id, *states),
            ).fetchall()
            assertion_ids = [row["assertion_id"] for row in rows]
            active_hash = content_hash({"assertions": [(row["assertion_id"], row["content_hash"]) for row in rows]})
            snapshot_payload = {
                "organization_id": organization_id,
                "ontology_version": ontology_version,
                "resolver_version": resolver_version,
                "active_assertion_set_hash": active_hash,
                "assertion_ids": assertion_ids,
            }
            snapshot_hash = content_hash(snapshot_payload)
            graph_version_id = f"graph_v_{snapshot_hash[:32]}"
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO graph_snapshots (
                  graph_version_id, organization_id, ontology_version,
                  resolver_version, active_assertion_set_hash, snapshot_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO graph_snapshots (
                  graph_version_id, organization_id, ontology_version,
                  resolver_version, active_assertion_set_hash, snapshot_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    graph_version_id, organization_id, ontology_version,
                    resolver_version, active_hash, snapshot_hash, now_iso(),
                ),
            )
            for ordinal, assertion_id in enumerate(assertion_ids):
                conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO graph_snapshot_assertions (
                      graph_version_id, assertion_id, ordinal
                    ) VALUES (?, ?, ?)
                    """, """
                    INSERT INTO graph_snapshot_assertions (
                      graph_version_id, assertion_id, ordinal
                    ) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING"""),
                    (graph_version_id, assertion_id, ordinal),
                )
        return GraphSnapshot(
            graph_version_id=graph_version_id,
            organization_id=organization_id,
            ontology_version=ontology_version,
            resolver_version=resolver_version,
            active_assertion_set_hash=active_hash,
            assertion_ids=assertion_ids,
            snapshot_hash=snapshot_hash,
        )

    def compare_snapshots(self, *, organization_id: str, left_snapshot_id: str, right_snapshot_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            for snapshot_id in (left_snapshot_id, right_snapshot_id):
                row = conn.execute(
                    hosted.sql_query("SELECT organization_id FROM graph_snapshots WHERE graph_version_id = ?", 'SELECT organization_id FROM graph_snapshots WHERE graph_version_id = %s'),
                    (snapshot_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(GRAPH_SNAPSHOT_REQUIRED)
                if row["organization_id"] != organization_id:
                    raise ValueError(GRAPH_CROSS_ORG_REJECTED)
            left = {
                row["assertion_id"]
                for row in conn.execute(
                    hosted.sql_query("SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = ?", 'SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = %s'),
                    (left_snapshot_id,),
                ).fetchall()
            }
            right = {
                row["assertion_id"]
                for row in conn.execute(
                    hosted.sql_query("SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = ?", 'SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = %s'),
                    (right_snapshot_id,),
                ).fetchall()
            }
        payload = {
            "left_snapshot_id": left_snapshot_id,
            "right_snapshot_id": right_snapshot_id,
            "added_assertion_ids": sorted(right - left),
            "removed_assertion_ids": sorted(left - right),
        }
        return {**payload, "comparison_hash": content_hash(payload)}

    def find_conflict_evidence(
        self,
        *,
        organization_id: str,
        node_id: str,
        governance_states: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        states = governance_states or ["candidate", "validated", "approved", "active"]
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT *
                FROM graph_assertions
                WHERE organization_id = ?
                  AND predicate = 'CONFLICTS_WITH'
                  AND governance_state IN ({",".join("?" for _ in states)})
                  AND (subject_id = ? OR object_id = ?)
                ORDER BY governance_state, assertion_id
                LIMIT ?
                """, f"\n                SELECT *\n                FROM graph_assertions\n                WHERE organization_id = %s\n                  AND predicate = 'CONFLICTS_WITH'\n                  AND governance_state IN ({','.join(('%s' for _ in states))})\n                  AND (subject_id = %s OR object_id = %s)\n                ORDER BY governance_state, assertion_id\n                LIMIT %s\n                "),
                (organization_id, *states, node_id, node_id, limit),
            ).fetchall()
        conflicts = [self._assertion_dict(row) for row in rows]
        payload = {
            "organization_id": organization_id,
            "node_id": node_id,
            "conflict_assertion_ids": [item["assertion_id"] for item in conflicts],
        }
        return {
            **payload,
            "conflicts": conflicts,
            "evidence_hash": content_hash(payload),
        }

    def find_normative_conflict_impact(
        self,
        *,
        organization_id: str,
        rule_ref: str,
    ) -> dict[str, Any]:
        """Return projected conflict analyses affected by one normative rule."""
        with self._connect() as conn:
            rule = conn.execute(
                hosted.sql_query("""
                SELECT node_id
                FROM graph_nodes
                WHERE organization_id = ?
                  AND node_type = 'normative_rule'
                  AND canonical_key = ?
                """, """
                SELECT node_id
                FROM graph_nodes
                WHERE organization_id = %s
                  AND node_type = 'normative_rule'
                  AND canonical_key = %s
                """),
                (organization_id, rule_ref),
            ).fetchone()
            affected: list[dict[str, Any]] = []
            if rule is not None:
                cores = conn.execute(
                    hosted.sql_query("""
                    SELECT DISTINCT n.node_id, n.canonical_key, n.properties_json
                    FROM graph_assertions a
                    JOIN graph_nodes n ON n.node_id = a.subject_id
                    WHERE a.organization_id = ?
                      AND a.predicate = 'MEMBER_OF_CONFLICT'
                      AND a.governance_state = 'active'
                      AND a.object_id = ?
                      AND n.node_type = 'normative_conflict_core'
                    ORDER BY n.canonical_key
                    """, """
                    SELECT DISTINCT n.node_id, n.canonical_key, n.properties_json
                    FROM graph_assertions a
                    JOIN graph_nodes n ON n.node_id = a.subject_id
                    WHERE a.organization_id = %s
                      AND a.predicate = 'MEMBER_OF_CONFLICT'
                      AND a.governance_state = 'active'
                      AND a.object_id = %s
                      AND n.node_type = 'normative_conflict_core'
                    ORDER BY n.canonical_key
                    """),
                    (organization_id, rule["node_id"]),
                ).fetchall()
                for core in cores:
                    properties = json.loads(core["properties_json"])
                    resolutions = conn.execute(
                        hosted.sql_query("""
                        SELECT DISTINCT n.canonical_key
                        FROM graph_assertions a
                        JOIN graph_nodes n ON n.node_id = a.object_id
                        WHERE a.organization_id = ?
                          AND a.subject_id = ?
                          AND a.predicate = 'RESOLVED_BY'
                          AND a.governance_state = 'active'
                          AND n.node_type = 'normative_conflict_resolution'
                        ORDER BY n.canonical_key
                        """, """
                        SELECT DISTINCT n.canonical_key
                        FROM graph_assertions a
                        JOIN graph_nodes n ON n.node_id = a.object_id
                        WHERE a.organization_id = %s
                          AND a.subject_id = %s
                          AND a.predicate = 'RESOLVED_BY'
                          AND a.governance_state = 'active'
                          AND n.node_type = 'normative_conflict_resolution'
                        ORDER BY n.canonical_key
                        """),
                        (organization_id, core["node_id"]),
                    ).fetchall()
                    affected.append({
                        "conflict_core_ref": core["canonical_key"],
                        "conflict_core_hash": properties["conflict_core_hash"],
                        "normative_set_ref": properties["normative_set_ref"],
                        "context_ref": properties["context_ref"],
                        "resolution_refs": [row["canonical_key"] for row in resolutions],
                    })
        affected.sort(key=lambda item: item["conflict_core_ref"].encode("utf-8"))
        payload = {
            "organization_id": organization_id,
            "rule_ref": rule_ref,
            "affected_conflicts": affected,
        }
        return {**payload, "impact_hash": content_hash(payload)}

    def build_normative_conflict_revalidation_targets(
        self,
        *,
        organization_id: str,
        changed_rule_ref: str,
        changed_release_ref: str,
        changed_release_hash: str,
        reason: str,
    ) -> tuple[NormativeConflictRevalidationTargetV1, ...]:
        """Compile deterministic revalidation work from the projected impact index."""
        impact = self.find_normative_conflict_impact(
            organization_id=organization_id,
            rule_ref=changed_rule_ref,
        )
        targets: list[NormativeConflictRevalidationTargetV1] = []
        for affected in impact["affected_conflicts"]:
            payload = {
                "schema_version": "normative_conflict_revalidation_target.v1",
                "organization_id": organization_id,
                "changed_rule_ref": changed_rule_ref,
                "changed_release_ref": changed_release_ref,
                "changed_release_hash": changed_release_hash,
                "reason": reason,
                "conflict_core_ref": affected["conflict_core_ref"],
                "conflict_core_hash": affected["conflict_core_hash"],
                "normative_set_ref": affected["normative_set_ref"],
                "context_ref": affected["context_ref"],
                "prior_resolution_refs": tuple(affected["resolution_refs"]),
            }
            digest = content_hash(payload)
            targets.append(
                NormativeConflictRevalidationTargetV1(
                    target_ref="normative-conflict-revalidation-target:" + digest,
                    target_hash="sha256:" + digest,
                    **payload,
                )
            )
        return tuple(sorted(targets, key=lambda item: item.conflict_core_ref.encode("utf-8")))

    def link_normative_conflict_revalidation_successor(
        self,
        *,
        target: NormativeConflictRevalidationTargetV1,
        compatibility_result: NormativeCompatibilityResultV1,
        successor_core: ConflictCoreV1 | None,
    ) -> NormativeConflictRevalidationSuccessorV1:
        """Create append-only lineage from one prior conflict to a revalidation result."""
        if target.organization_id == "":
            raise ValueError("revalidation target organization is required")
        if compatibility_result.outcome == "CONFLICT":
            if successor_core is None:
                raise ValueError("conflicting revalidation requires successor core")
            if successor_core.compatibility_result_ref != compatibility_result.result_ref:
                raise ValueError("successor core/result mismatch")
            prior = self.find_normative_conflict_impact(
                organization_id=target.organization_id,
                rule_ref=target.changed_rule_ref,
            )
            prior_entry = next(
                (
                    item for item in prior["affected_conflicts"]
                    if item["conflict_core_ref"] == target.conflict_core_ref
                ),
                None,
            )
            if prior_entry is None:
                raise ValueError("revalidation target prior conflict is not projected")
            with self._connect() as conn:
                prior_members = {
                    row["canonical_key"]
                    for row in conn.execute(
                        hosted.sql_query("""
                        SELECT r.canonical_key
                        FROM graph_assertions a
                        JOIN graph_nodes c ON c.node_id = a.subject_id
                        JOIN graph_nodes r ON r.node_id = a.object_id
                        WHERE a.organization_id = ?
                          AND c.canonical_key = ?
                          AND c.node_type = 'normative_conflict_core'
                          AND a.predicate = 'MEMBER_OF_CONFLICT'
                          AND a.governance_state = 'active'
                          AND r.node_type = 'normative_rule'
                        """, """
                        SELECT r.canonical_key
                        FROM graph_assertions a
                        JOIN graph_nodes c ON c.node_id = a.subject_id
                        JOIN graph_nodes r ON r.node_id = a.object_id
                        WHERE a.organization_id = %s
                          AND c.canonical_key = %s
                          AND c.node_type = 'normative_conflict_core'
                          AND a.predicate = 'MEMBER_OF_CONFLICT'
                          AND a.governance_state = 'active'
                          AND r.node_type = 'normative_rule'
                        """),
                        (target.organization_id, target.conflict_core_ref),
                    ).fetchall()
                }
            outcome = (
                "CONFLICT_PERSISTS"
                if prior_members == set(successor_core.member_rule_refs)
                else "CONFLICT_MEMBERSHIP_CHANGED"
            )
            successor_core_ref = successor_core.conflict_core_ref
        else:
            if successor_core is not None:
                raise ValueError("non-conflict revalidation cannot carry successor core")
            outcome = (
                "CONFLICT_DISAPPEARS"
                if compatibility_result.outcome == "COMPATIBLE"
                else "UNKNOWN"
            )
            successor_core_ref = None

        payload = {
            "schema_version": "normative_conflict_revalidation_successor.v1",
            "target_ref": target.target_ref,
            "prior_conflict_core_ref": target.conflict_core_ref,
            "successor_compatibility_result_ref": compatibility_result.result_ref,
            "successor_compatibility_result_hash": compatibility_result.result_hash,
            "successor_normative_set_ref": compatibility_result.normative_set_ref,
            "outcome": outcome,
            "successor_conflict_core_ref": successor_core_ref,
        }
        digest = content_hash(payload)
        return NormativeConflictRevalidationSuccessorV1(
            successor_ref="normative-conflict-revalidation-successor:" + digest,
            successor_hash="sha256:" + digest,
            **payload,
        )

    def project_normative_conflict_revalidation_successor(
        self,
        *,
        organization_id: str,
        successor: NormativeConflictRevalidationSuccessorV1,
    ) -> dict[str, Any]:
        """Project append-only revalidation lineage without changing conflict truth."""
        prior = self.upsert_node(
            organization_id=organization_id,
            node_type="normative_conflict_core",
            canonical_key=successor.prior_conflict_core_ref,
            display_name=successor.prior_conflict_core_ref,
        )
        successor_node = self.upsert_node(
            organization_id=organization_id,
            node_type="normative_conflict_revalidation_successor",
            canonical_key=successor.successor_ref,
            display_name=successor.successor_ref,
            properties={
                "successor_hash": successor.successor_hash,
                "outcome": successor.outcome,
                "successor_compatibility_result_ref": successor.successor_compatibility_result_ref,
                "successor_conflict_core_ref": successor.successor_conflict_core_ref,
            },
        )
        assertion = self.propose_assertion(
            organization_id=organization_id,
            subject_id=prior.node_id,
            predicate="REVALIDATED_BY",
            object_id=successor_node.node_id,
            assertion_class="structural",
            semantic_tier=None,
            authority_role="structural",
            governance_state="active",
            valid_from=None,
            valid_to=None,
            source_ref=successor.successor_ref,
            source_anchor=successor.target_ref,
            extraction_method="parser",
            extractor_version="graph-kernel.projection.v1",
            confidence=None,
        )
        payload = {
            "successor_node_id": successor_node.node_id,
            "lineage_assertion_id": assertion.assertion_id,
            "successor_ref": successor.successor_ref,
        }
        return {**payload, "projection_hash": content_hash(payload)}

    def read_normative_conflict_graph(
        self,
        *,
        organization_id: str,
        conflict_core_ref: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            core = conn.execute(
                hosted.sql_query("""SELECT node_id, properties_json FROM graph_nodes
                   WHERE organization_id = ? AND node_type = 'normative_conflict_core'
                     AND canonical_key = ?""", """SELECT node_id, properties_json FROM graph_nodes
                   WHERE organization_id = %s AND node_type = 'normative_conflict_core'
                     AND canonical_key = %s"""),
                (organization_id, conflict_core_ref),
            ).fetchone()
            if core is None:
                raise KeyError(conflict_core_ref)
            props = json.loads(core["properties_json"])
            members = [
                row["canonical_key"] for row in conn.execute(
                    hosted.sql_query("""SELECT DISTINCT r.canonical_key
                       FROM graph_assertions a JOIN graph_nodes r ON r.node_id = a.object_id
                       WHERE a.organization_id = ? AND a.subject_id = ?
                         AND a.predicate = 'MEMBER_OF_CONFLICT'
                         AND a.governance_state = 'active'
                         AND r.node_type = 'normative_rule'
                       ORDER BY r.canonical_key""", """SELECT DISTINCT r.canonical_key
                       FROM graph_assertions a JOIN graph_nodes r ON r.node_id = a.object_id
                       WHERE a.organization_id = %s AND a.subject_id = %s
                         AND a.predicate = 'MEMBER_OF_CONFLICT'
                         AND a.governance_state = 'active'
                         AND r.node_type = 'normative_rule'
                       ORDER BY r.canonical_key"""),
                    (organization_id, core["node_id"]),
                ).fetchall()
            ]
            resolutions = [
                row["canonical_key"] for row in conn.execute(
                    hosted.sql_query("""SELECT DISTINCT r.canonical_key
                       FROM graph_assertions a JOIN graph_nodes r ON r.node_id = a.object_id
                       WHERE a.organization_id = ? AND a.subject_id = ?
                         AND a.predicate = 'RESOLVED_BY'
                         AND a.governance_state = 'active'
                       ORDER BY r.canonical_key""", """SELECT DISTINCT r.canonical_key
                       FROM graph_assertions a JOIN graph_nodes r ON r.node_id = a.object_id
                       WHERE a.organization_id = %s AND a.subject_id = %s
                         AND a.predicate = 'RESOLVED_BY'
                         AND a.governance_state = 'active'
                       ORDER BY r.canonical_key"""),
                    (organization_id, core["node_id"]),
                ).fetchall()
            ]
            successors = []
            for row in conn.execute(
                hosted.sql_query("""SELECT DISTINCT s.canonical_key, s.properties_json
                   FROM graph_assertions a JOIN graph_nodes s ON s.node_id = a.object_id
                   WHERE a.organization_id = ? AND a.subject_id = ?
                     AND a.predicate = 'REVALIDATED_BY'
                     AND a.governance_state = 'active'
                   ORDER BY s.canonical_key""", """SELECT DISTINCT s.canonical_key, s.properties_json
                   FROM graph_assertions a JOIN graph_nodes s ON s.node_id = a.object_id
                   WHERE a.organization_id = %s AND a.subject_id = %s
                     AND a.predicate = 'REVALIDATED_BY'
                     AND a.governance_state = 'active'
                   ORDER BY s.canonical_key"""),
                (organization_id, core["node_id"]),
            ).fetchall():
                item = json.loads(row["properties_json"])
                successors.append({
                    "successor_ref": row["canonical_key"],
                    "outcome": item["outcome"],
                    "successor_conflict_core_ref": item.get("successor_conflict_core_ref"),
                    "successor_compatibility_result_ref": item["successor_compatibility_result_ref"],
                })
        payload = {
            "conflict_core_ref": conflict_core_ref,
            "conflict_core_hash": props.get("conflict_core_hash"),
            "normative_set_ref": props.get("normative_set_ref"),
            "context_ref": props.get("context_ref"),
            "member_rule_refs": members,
            "evidence_refs": sorted(props.get("evidence_refs", []), key=lambda x: x.encode("utf-8")),
            "resolution_refs": resolutions,
            "successors": successors,
        }
        return {**payload, "read_model_hash": content_hash(payload)}

    def compare_normative_conflict_cores(
        self,
        *,
        organization_id: str,
        left_conflict_core_ref: str,
        right_conflict_core_ref: str,
    ) -> dict[str, Any]:
        left = self.read_normative_conflict_graph(
            organization_id=organization_id, conflict_core_ref=left_conflict_core_ref
        )
        right = self.read_normative_conflict_graph(
            organization_id=organization_id, conflict_core_ref=right_conflict_core_ref
        )
        left_members = set(left["member_rule_refs"])
        right_members = set(right["member_rule_refs"])
        ordered = lambda values: sorted(values, key=lambda x: x.encode("utf-8"))
        payload = {
            "left_conflict_core_ref": left_conflict_core_ref,
            "right_conflict_core_ref": right_conflict_core_ref,
            "left_member_rule_refs": ordered(left_members),
            "right_member_rule_refs": ordered(right_members),
            "added_rule_refs": ordered(right_members - left_members),
            "removed_rule_refs": ordered(left_members - right_members),
            "shared_rule_refs": ordered(left_members & right_members),
        }
        return {**payload, "comparison_hash": content_hash(payload)}

    def find_precedence_evidence(
        self,
        *,
        organization_id: str,
        start_node_id: str,
        max_depth: int = 5,
        governance_states: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        paths = self.trace_lineage(
            organization_id=organization_id,
            start_node_id=start_node_id,
            predicates=["SUPERSEDES", "CONSTRAINS", "GOVERNED_BY", "APPLIES_TO"],
            max_depth=max_depth,
            governance_states=governance_states or ["approved", "active"],
            limit=limit,
        )
        payload = {
            "organization_id": organization_id,
            "start_node_id": start_node_id,
            "path_assertion_ids": [item["assertion_id"] for item in paths],
            "max_depth": max_depth,
        }
        return {**payload, "paths": paths, "evidence_hash": content_hash(payload)}

    def integrity_check(self) -> str:
        if hosted.enabled():
            raise hosted.HostedStorageUnavailable('hosted physical integrity uses PostgreSQL operator checks')
        with self._connect() as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def backup_to_offline_path(self, backup_path: str | Path) -> None:
        if hosted.enabled():
            raise hosted.HostedStorageUnavailable('use explicit PostgreSQL backup administration in hosted mode')
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source:
            target = sqlite3.connect(str(backup_path))
            try:
                source.backup(target)
            finally:
                target.close()

    @classmethod
    def restore_offline_backup(cls, *, backup_path: str | Path, target_path: str | Path) -> GraphKernelStore:
        if hosted.enabled():
            raise hosted.HostedStorageUnavailable('offline SQLite restore requires explicit local backend')
        backup_path = Path(backup_path)
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source = sqlite3.connect(str(backup_path))
        try:
            target = sqlite3.connect(str(target_path))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        return cls.for_offline_path(target_path)

    def _record_event(
        self,
        conn: sqlite3.Connection,
        *,
        assertion_id: str,
        event_type: str,
        actor_ref: str,
        payload: dict[str, Any],
    ) -> str:
        prior = conn.execute(
            hosted.sql_query("""
            SELECT event_hash
            FROM graph_assertion_events
            WHERE assertion_id = ?
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT 1
            """, """
            SELECT event_hash
            FROM graph_assertion_events
            WHERE assertion_id = %s
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT 1
            """),
            (assertion_id,),
        ).fetchone()
        prior_hash = prior["event_hash"] if prior else None
        event_payload = {
            "assertion_id": assertion_id,
            "event_type": event_type,
            "actor_ref": actor_ref,
            "payload": payload,
            "prior_event_hash": prior_hash,
        }
        event_id = f"graph_assertion_event_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO graph_assertion_events (
              event_id, assertion_id, event_type, actor_ref, event_payload_json,
              prior_event_hash, event_hash, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO graph_assertion_events (
              event_id, assertion_id, event_type, actor_ref, event_payload_json,
              prior_event_hash, event_hash, occurred_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                event_id,
                assertion_id,
                event_type,
                actor_ref,
                canonical_json(payload),
                prior_hash,
                content_hash(event_payload),
                now_iso(),
            ),
        )
        return event_id

    def _auto_activatable(self, assertion_class: str, authority_role: str, extraction_method: str) -> bool:
        return (
            assertion_class == "structural"
            and authority_role == "structural"
            and extraction_method in {"parser", "runtime_observation"}
        )

    def _assertion_obj(self, row: sqlite3.Row) -> GraphAssertion:
        return GraphAssertion(
            assertion_id=row["assertion_id"],
            organization_id=row["organization_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            assertion_class=row["assertion_class"],
            semantic_tier=row["semantic_tier"],
            authority_role=row["authority_role"],
            governance_state=row["governance_state"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            source_ref=row["source_ref"],
            source_anchor=row["source_anchor"],
            extraction_method=row["extraction_method"],
            extractor_version=row["extractor_version"],
            confidence=row["confidence"],
            content_hash=row["content_hash"],
        )

    def _assertion_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        return result

    def _transition_assertion(
        self,
        *,
        assertion_id: str,
        actor_ref: str,
        governance_event_ref: str,
        allowed_from: set[str],
        target_state: str,
        event_type: str,
        valid_to: str | None = None,
    ) -> GraphAssertion:
        with self._connect() as conn:
            row = conn.execute(hosted.sql_query("SELECT * FROM graph_assertions WHERE assertion_id = ?", 'SELECT * FROM graph_assertions WHERE assertion_id = %s'), (assertion_id,)).fetchone()
            if row is None:
                raise KeyError(assertion_id)
            if row["governance_state"] not in allowed_from:
                raise ValueError(GRAPH_ASSERTION_INVALID_TRANSITION)
            conn.execute(
                hosted.sql_query("""
                UPDATE graph_assertions
                SET governance_state = ?,
                    valid_to = COALESCE(?, valid_to)
                WHERE assertion_id = ?
                """, """
                UPDATE graph_assertions
                SET governance_state = %s,
                    valid_to = COALESCE(%s, valid_to)
                WHERE assertion_id = %s
                """),
                (target_state, valid_to, assertion_id),
            )
            self._record_event(
                conn,
                assertion_id=assertion_id,
                event_type=event_type,
                actor_ref=actor_ref,
                payload={"governance_event_ref": governance_event_ref},
            )
            updated = conn.execute(hosted.sql_query("SELECT * FROM graph_assertions WHERE assertion_id = ?", 'SELECT * FROM graph_assertions WHERE assertion_id = %s'), (assertion_id,)).fetchone()
        return self._assertion_obj(updated)


def _effective(row: sqlite3.Row, as_of: str | None) -> bool:
    if as_of is None:
        return True
    if row["valid_from"] and row["valid_from"] > as_of:
        return False
    if row["valid_to"] and row["valid_to"] <= as_of:
        return False
    return True


def snapshot_predicate(snapshot_id: str | None) -> str:
    if snapshot_id is None:
        return ""
    return hosted.sql_query("AND assertion_id IN (SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = ?)", 'AND assertion_id IN (SELECT assertion_id FROM graph_snapshot_assertions WHERE graph_version_id = %s)')


def snapshot_join(snapshot_id: str | None) -> str:
    if snapshot_id is None:
        return ""
    return hosted.sql_query("JOIN graph_snapshot_assertions gsa ON gsa.assertion_id = a.assertion_id AND gsa.graph_version_id = ?", 'JOIN graph_snapshot_assertions gsa ON gsa.assertion_id = a.assertion_id AND gsa.graph_version_id = %s')


def effective_predicate(as_of: str | None) -> str:
    if as_of is None:
        return ""
    return hosted.sql_query("AND (a.valid_from IS NULL OR a.valid_from <= ?) AND (a.valid_to IS NULL OR a.valid_to > ?)", 'AND (a.valid_from IS NULL OR a.valid_from <= %s) AND (a.valid_to IS NULL OR a.valid_to > %s)')
