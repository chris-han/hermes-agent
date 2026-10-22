from __future__ import annotations

from storage import hosted_postgres as hosted

import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import now_iso

from .store import GraphKernelStore


GRAPH_INGESTION_STATUS_TERMINAL = {
    "no_change",
    "updated",
    "failed_retryable",
    "failed_terminal",
    "superseded",
}

INDEXED_NO_CHANGE = "INDEXED_NO_CHANGE"
INDEXED_UPDATED = "INDEXED_UPDATED"
INDEX_FAILED_RETRYABLE = "INDEX_FAILED_RETRYABLE"
INDEX_FAILED_TERMINAL = "INDEX_FAILED_TERMINAL"
DEFAULT_EXTRACTOR_VERSION = "graph_ingestion_coordinator.v1"
DEFAULT_STRUCTURE_SCHEMA_VERSION = "source_structure.v1"
DEFAULT_ONTOLOGY_COMPATIBILITY_VERSION = "ontology_compat.v1"
DEFAULT_GRAPH_PROJECTION_VERSION = "graph_projection.v1"
GRAPH_INDEX_TRIGGER_TYPES = {
    "upload",
    "source_registered",
    "source_replaced",
    "legal_monitor",
    "extractor_changed",
    "ontology_changed",
    "scheduled_reconciliation",
    "manual",
}


class GraphIngestionCoordinator:
    def __init__(self, store: GraphKernelStore | None = None) -> None:
        self.store = store or GraphKernelStore()

    @classmethod
    def for_offline_path(cls, db_path: str | Path) -> GraphIngestionCoordinator:
        return cls(GraphKernelStore.for_offline_path(db_path))

    def enqueue_run(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_ref: str,
        trigger_type: str,
        requested_by: str,
        idempotency_key: str,
        requested_fingerprint: str,
        observed_source_hash: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        now = now_iso()
        run_id = f"graph_ingestion_run_{uuid4().hex}"
        with self.store._connect() as conn:
            if not hosted.is_postgres(conn):
                conn.execute("PRAGMA foreign_keys = ON")
            prior = conn.execute(
                hosted.sql_query("""
                SELECT last_successful_run_ref
                FROM graph_source_index_manifests
                WHERE organization_id = ?
                  AND source_ref = ?
                """, '\n                SELECT last_successful_run_ref\n                FROM graph_source_index_manifests\n                WHERE organization_id = %s\n                  AND source_ref = %s\n                '),
                (organization_id, source_ref),
            ).fetchone()
            prior_successful_run_ref = prior["last_successful_run_ref"] if prior else None
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO graph_ingestion_runs (
                  ingestion_run_id, organization_id, workspace_id, source_ref,
                  trigger_type, requested_by, idempotency_key, requested_fingerprint,
                  observed_source_hash, prior_successful_run_ref, status, attempt_count, max_attempts,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?)
                """, "\n                INSERT INTO graph_ingestion_runs (\n                  ingestion_run_id, organization_id, workspace_id, source_ref,\n                  trigger_type, requested_by, idempotency_key, requested_fingerprint,\n                  observed_source_hash, prior_successful_run_ref, status, attempt_count, max_attempts,\n                  created_at, updated_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued', 0, %s, %s, %s) ON CONFLICT DO NOTHING"),
                (
                    run_id,
                    organization_id,
                    workspace_id,
                    source_ref,
                    trigger_type,
                    requested_by,
                    idempotency_key,
                    requested_fingerprint,
                    observed_source_hash,
                    prior_successful_run_ref,
                    max_attempts,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM graph_ingestion_runs
                WHERE organization_id = ?
                  AND source_ref = ?
                  AND idempotency_key = ?
                """, '\n                SELECT *\n                FROM graph_ingestion_runs\n                WHERE organization_id = %s\n                  AND source_ref = %s\n                  AND idempotency_key = %s\n                '),
                (organization_id, source_ref, idempotency_key),
            ).fetchone()
        return _row_dict(row)

    def index_source(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_ref: str,
        source_text: str,
        trigger_type: str,
        requested_by: str,
        idempotency_key: str,
        extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
        structure_schema_version: str = DEFAULT_STRUCTURE_SCHEMA_VERSION,
        ontology_compatibility_version: str = DEFAULT_ONTOLOGY_COMPATIBILITY_VERSION,
        graph_projection_version: str = DEFAULT_GRAPH_PROJECTION_VERSION,
        ontology_version: str = "ontology:v1",
        resolver_version: str = "resolver:v1",
        lease_owner: str = "graph-ingestion-coordinator",
        lease_expires_at: str | None = None,
    ) -> dict[str, Any]:
        if trigger_type not in GRAPH_INDEX_TRIGGER_TYPES:
            raise ValueError("GRAPH_INDEX_TRIGGER_TYPE_UNSUPPORTED")
        fingerprint = build_index_fingerprint(
            source_text=source_text,
            extractor_version=extractor_version,
            structure_schema_version=structure_schema_version,
            ontology_compatibility_version=ontology_compatibility_version,
            graph_projection_version=graph_projection_version,
        )
        run = self.enqueue_run(
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_ref=source_ref,
            trigger_type=trigger_type,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            requested_fingerprint=fingerprint["index_fingerprint"],
            observed_source_hash=fingerprint["source_hash"],
        )
        if run["status"] in GRAPH_INGESTION_STATUS_TERMINAL:
            return {"outcome": _external_outcome(run["status"]), "run": run, "freshness": self.get_freshness(organization_id=organization_id, source_ref=source_ref)}

        lease_expires_at = lease_expires_at or "9999-12-31T23:59:59+00:00"
        claimed = self.claim_next_run(
            organization_id=organization_id,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            source_ref=source_ref,
        )
        if claimed is None:
            run = self.get_run(run["ingestion_run_id"])
            return {"outcome": "INDEX_ALREADY_CLAIMED", "run": run, "freshness": self.get_freshness(organization_id=organization_id, source_ref=source_ref)}
        self.mark_indexing(ingestion_run_id=claimed["ingestion_run_id"], lease_owner=lease_owner)

        prior_manifest = self.get_manifest(organization_id=organization_id, source_ref=source_ref)
        if prior_manifest and prior_manifest["index_fingerprint"] == fingerprint["index_fingerprint"]:
            completed = self.complete_no_change(
                ingestion_run_id=claimed["ingestion_run_id"],
                lease_owner=lease_owner,
            )
            self._upsert_freshness(
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_ref=source_ref,
                status="fresh",
                indexed_at=completed["completed_at"],
                source_fingerprint=fingerprint["index_fingerprint"],
                last_successful_index_ref=prior_manifest["last_successful_run_ref"],
                graph_snapshot_ref=prior_manifest["graph_snapshot_ref"],
                graph_snapshot_hash=prior_manifest["graph_snapshot_hash"],
                error_code=None,
                error_detail_ref=None,
            )
            return {"outcome": INDEXED_NO_CHANGE, "run": completed, "freshness": self.get_freshness(organization_id=organization_id, source_ref=source_ref)}

        snapshot = self._rebuild_source_subgraph(
            organization_id=organization_id,
            source_ref=source_ref,
            source_text=source_text,
            fingerprint=fingerprint,
            ontology_version=ontology_version,
            resolver_version=resolver_version,
            deprecate_prior=prior_manifest is not None,
        )
        completed = self.complete_updated(
            ingestion_run_id=claimed["ingestion_run_id"],
            lease_owner=lease_owner,
            graph_snapshot_ref=snapshot.graph_version_id,
            graph_snapshot_hash=snapshot.snapshot_hash,
        )
        self._upsert_manifest(
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_ref=source_ref,
            fingerprint=fingerprint,
            last_successful_run_ref=completed["ingestion_run_id"],
            graph_snapshot_ref=snapshot.graph_version_id,
            graph_snapshot_hash=snapshot.snapshot_hash,
        )
        self._upsert_freshness(
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_ref=source_ref,
            status="fresh",
            indexed_at=completed["completed_at"],
            source_fingerprint=fingerprint["index_fingerprint"],
            last_successful_index_ref=completed["ingestion_run_id"],
            graph_snapshot_ref=snapshot.graph_version_id,
            graph_snapshot_hash=snapshot.snapshot_hash,
            error_code=None,
            error_detail_ref=None,
        )
        return {"outcome": INDEXED_UPDATED, "run": completed, "snapshot": snapshot.to_record(), "freshness": self.get_freshness(organization_id=organization_id, source_ref=source_ref)}

    def index_upload_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="upload", **kwargs)

    def index_source_registered_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="source_registered", **kwargs)

    def index_source_replaced_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="source_replaced", **kwargs)

    def index_legal_monitor_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="legal_monitor", **kwargs)

    def index_extractor_changed_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="extractor_changed", **kwargs)

    def index_ontology_changed_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="ontology_changed", **kwargs)

    def index_manual_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.index_source(trigger_type="manual", **kwargs)

    def reconcile_sources(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        sources: list[dict[str, Any]],
        requested_by: str = "scheduled-reconciliation",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for source in sorted(sources, key=lambda item: str(item["source_ref"])):
            source_ref = str(source["source_ref"])
            source_text = str(source.get("source_text") or "")
            fingerprint = build_index_fingerprint(
                source_text=source_text,
                extractor_version=str(source.get("extractor_version") or DEFAULT_EXTRACTOR_VERSION),
                structure_schema_version=str(source.get("structure_schema_version") or DEFAULT_STRUCTURE_SCHEMA_VERSION),
                ontology_compatibility_version=str(source.get("ontology_compatibility_version") or DEFAULT_ONTOLOGY_COMPATIBILITY_VERSION),
                graph_projection_version=str(source.get("graph_projection_version") or DEFAULT_GRAPH_PROJECTION_VERSION),
            )
            manifest = self.get_manifest(organization_id=organization_id, source_ref=source_ref)
            freshness = self.get_freshness(organization_id=organization_id, source_ref=source_ref)
            if manifest and manifest["index_fingerprint"] == fingerprint["index_fingerprint"] and freshness and freshness["status"] == "fresh":
                continue
            results.append(
                self.index_source(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    source_ref=source_ref,
                    source_text=source_text,
                    trigger_type="scheduled_reconciliation",
                    requested_by=requested_by,
                    idempotency_key=f"scheduled:{source_ref}:{fingerprint['index_fingerprint']}",
                    extractor_version=fingerprint["extractor_version"],
                    structure_schema_version=fingerprint["structure_schema_version"],
                    ontology_compatibility_version=fingerprint["ontology_compatibility_version"],
                    graph_projection_version=fingerprint["graph_projection_version"],
                )
            )
        return results

    def retry_due_failures(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        sources: list[dict[str, Any]],
        now: str,
        requested_by: str = "graph-ingestion-retry",
    ) -> list[dict[str, Any]]:
        by_ref = {str(source["source_ref"]): source for source in sources}
        with self.store._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM graph_ingestion_runs
                WHERE organization_id = ?
                  AND status = 'failed_retryable'
                  AND next_retry_at IS NOT NULL
                  AND next_retry_at <= ?
                  AND attempt_count < max_attempts
                ORDER BY next_retry_at, ingestion_run_id
                """, "\n                SELECT *\n                FROM graph_ingestion_runs\n                WHERE organization_id = %s\n                  AND status = 'failed_retryable'\n                  AND next_retry_at IS NOT NULL\n                  AND next_retry_at <= %s\n                  AND attempt_count < max_attempts\n                ORDER BY next_retry_at, ingestion_run_id\n                "),
                (organization_id, now),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            source = by_ref.get(row["source_ref"])
            if source is None:
                continue
            results.append(
                self.index_source(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    source_ref=row["source_ref"],
                    source_text=str(source.get("source_text") or ""),
                    trigger_type=str(source.get("trigger_type") or "manual"),
                    requested_by=requested_by,
                    idempotency_key=f"retry:{row['ingestion_run_id']}:{row['attempt_count'] + 1}",
                    extractor_version=str(source.get("extractor_version") or DEFAULT_EXTRACTOR_VERSION),
                    structure_schema_version=str(source.get("structure_schema_version") or DEFAULT_STRUCTURE_SCHEMA_VERSION),
                    ontology_compatibility_version=str(source.get("ontology_compatibility_version") or DEFAULT_ONTOLOGY_COMPATIBILITY_VERSION),
                    graph_projection_version=str(source.get("graph_projection_version") or DEFAULT_GRAPH_PROJECTION_VERSION),
                )
            )
        return results

    def get_run(self, ingestion_run_id: str) -> dict[str, Any]:
        with self.store._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = ?", 'SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = %s'),
                (ingestion_run_id,),
            ).fetchone()
        return _row_dict(row)

    def get_manifest(self, *, organization_id: str, source_ref: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM graph_source_index_manifests
                WHERE organization_id = ?
                  AND source_ref = ?
                """, '\n                SELECT *\n                FROM graph_source_index_manifests\n                WHERE organization_id = %s\n                  AND source_ref = %s\n                '),
                (organization_id, source_ref),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_freshness(self, *, organization_id: str, source_ref: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM graph_index_freshness
                WHERE organization_id = ?
                  AND source_ref = ?
                """, '\n                SELECT *\n                FROM graph_index_freshness\n                WHERE organization_id = %s\n                  AND source_ref = %s\n                '),
                (organization_id, source_ref),
            ).fetchone()
        return dict(row) if row is not None else None

    def claim_next_run(
        self,
        *,
        organization_id: str,
        lease_owner: str,
        lease_expires_at: str,
        source_ref: str | None = None,
    ) -> dict[str, Any] | None:
        now = now_iso()
        with self.store._connect() as conn:
            if not hosted.is_postgres(conn):
                conn.execute("PRAGMA foreign_keys = ON")
            filters = [hosted.sql_query("organization_id = ?", "organization_id = %s"), "status = 'queued'"]
            params: list[Any] = [organization_id]
            if source_ref is not None:
                filters.append(hosted.sql_query("source_ref = ?", "source_ref = %s"))
                params.append(source_ref)
            row = conn.execute(
                hosted.sql_query(f"""
                SELECT ingestion_run_id
                FROM graph_ingestion_runs
                WHERE {" AND ".join(filters)}
                ORDER BY created_at, ingestion_run_id
                LIMIT 1
                """, f"\n                SELECT ingestion_run_id\n                FROM graph_ingestion_runs\n                WHERE {' AND '.join(filters)}\n                ORDER BY created_at, ingestion_run_id\n                LIMIT 1\n                "),
                params,
            ).fetchone()
            if row is None:
                return None
            run_id = row["ingestion_run_id"]
            updated = conn.execute(
                hosted.sql_query("""
                UPDATE graph_ingestion_runs
                SET status = 'claimed',
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempt_count = attempt_count + 1,
                    updated_at = ?
                WHERE ingestion_run_id = ?
                  AND status = 'queued'
                """, "\n                UPDATE graph_ingestion_runs\n                SET status = 'claimed',\n                    lease_owner = %s,\n                    lease_expires_at = %s,\n                    attempt_count = attempt_count + 1,\n                    updated_at = %s\n                WHERE ingestion_run_id = %s\n                  AND status = 'queued'\n                "),
                (lease_owner, lease_expires_at, now, run_id),
            ).rowcount
            if updated != 1:
                return None
            claimed = conn.execute(
                hosted.sql_query("SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = ?", 'SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = %s'),
                (run_id,),
            ).fetchone()
        return _row_dict(claimed)

    def reclaim_expired_leases(self, *, organization_id: str, now: str) -> list[str]:
        with self.store._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT ingestion_run_id
                FROM graph_ingestion_runs
                WHERE organization_id = ?
                  AND status IN ('claimed', 'indexing')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?
                ORDER BY ingestion_run_id
                """, "\n                SELECT ingestion_run_id\n                FROM graph_ingestion_runs\n                WHERE organization_id = %s\n                  AND status IN ('claimed', 'indexing')\n                  AND lease_expires_at IS NOT NULL\n                  AND lease_expires_at <= %s\n                ORDER BY ingestion_run_id\n                "),
                (organization_id, now),
            ).fetchall()
            run_ids = [row["ingestion_run_id"] for row in rows]
            if run_ids:
                conn.execute(
                    hosted.sql_query(f"""
                    UPDATE graph_ingestion_runs
                    SET status = 'queued',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE ingestion_run_id IN ({",".join("?" for _ in run_ids)})
                    """, f"\n                    UPDATE graph_ingestion_runs\n                    SET status = 'queued',\n                        lease_owner = NULL,\n                        lease_expires_at = NULL,\n                        updated_at = %s\n                    WHERE ingestion_run_id IN ({','.join(('%s' for _ in run_ids))})\n                    "),
                    (now, *run_ids),
                )
        return run_ids

    def mark_indexing(self, *, ingestion_run_id: str, lease_owner: str) -> dict[str, Any]:
        return self._transition_claimed(
            ingestion_run_id=ingestion_run_id,
            lease_owner=lease_owner,
            status="indexing",
            updates={},
        )

    def complete_no_change(self, *, ingestion_run_id: str, lease_owner: str) -> dict[str, Any]:
        return self._finish(
            ingestion_run_id=ingestion_run_id,
            lease_owner=lease_owner,
            status="no_change",
            updates={},
        )

    def complete_updated(
        self,
        *,
        ingestion_run_id: str,
        lease_owner: str,
        graph_snapshot_ref: str,
        graph_snapshot_hash: str,
    ) -> dict[str, Any]:
        return self._finish(
            ingestion_run_id=ingestion_run_id,
            lease_owner=lease_owner,
            status="updated",
            updates={
                "resulting_graph_snapshot_ref": graph_snapshot_ref,
                "resulting_graph_snapshot_hash": graph_snapshot_hash,
            },
        )

    def fail_run(
        self,
        *,
        ingestion_run_id: str,
        lease_owner: str,
        retryable: bool,
        error_code: str,
        error_detail_ref: str,
        next_retry_at: str | None = None,
    ) -> dict[str, Any]:
        failed = self._finish(
            ingestion_run_id=ingestion_run_id,
            lease_owner=lease_owner,
            status="failed_retryable" if retryable else "failed_terminal",
            updates={
                "error_code": error_code,
                "error_detail_ref": error_detail_ref,
                "next_retry_at": next_retry_at,
            },
        )
        self._upsert_freshness(
            organization_id=failed["organization_id"],
            workspace_id=failed["workspace_id"],
            source_ref=failed["source_ref"],
            status="failed",
            indexed_at=None,
            source_fingerprint=failed["requested_fingerprint"],
            last_successful_index_ref=failed["prior_successful_run_ref"],
            graph_snapshot_ref=None,
            graph_snapshot_hash=None,
            error_code=error_code,
            error_detail_ref=error_detail_ref,
        )
        return failed

    def _transition_claimed(
        self,
        *,
        ingestion_run_id: str,
        lease_owner: str,
        status: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        now = now_iso()
        with self.store._connect() as conn:
            rowcount = conn.execute(
                hosted.sql_query("""
                UPDATE graph_ingestion_runs
                SET status = ?,
                    updated_at = ?
                WHERE ingestion_run_id = ?
                  AND lease_owner = ?
                  AND status = 'claimed'
                """, "\n                UPDATE graph_ingestion_runs\n                SET status = %s,\n                    updated_at = %s\n                WHERE ingestion_run_id = %s\n                  AND lease_owner = %s\n                  AND status = 'claimed'\n                "),
                (status, now, ingestion_run_id, lease_owner),
            ).rowcount
            if rowcount != 1:
                raise ValueError("GRAPH_INGESTION_RUN_NOT_CLAIMED")
            row = conn.execute(
                hosted.sql_query("SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = ?", 'SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = %s'),
                (ingestion_run_id,),
            ).fetchone()
        return _row_dict(row)

    def _rebuild_source_subgraph(
        self,
        *,
        organization_id: str,
        source_ref: str,
        source_text: str,
        fingerprint: dict[str, str],
        ontology_version: str,
        resolver_version: str,
        deprecate_prior: bool,
    ):
        if deprecate_prior:
            self.store.deprecate_assertions_for_source(
                organization_id=organization_id,
                source_ref=source_ref,
                actor_ref="graph-ingestion-coordinator",
            )
        source = self.store.upsert_node(
            organization_id=organization_id,
            node_type="source_document",
            canonical_key=source_ref,
            display_name=source_ref,
            properties={
                "source_hash": fingerprint["source_hash"],
                "normalized_structure_hash": fingerprint["normalized_structure_hash"],
                "extractor_version": fingerprint["extractor_version"],
            },
        )
        structure = self.store.upsert_node(
            organization_id=organization_id,
            node_type="source_structure",
            canonical_key=f"{source_ref}#structure",
            display_name=f"{source_ref} structure",
            properties={
                "structure_schema_version": fingerprint["structure_schema_version"],
                "source_preview": source_text[:120],
            },
        )
        self.store.propose_assertion(
            organization_id=organization_id,
            subject_id=source.node_id,
            predicate="CONTAINS",
            object_id=structure.node_id,
            assertion_class="structural",
            semantic_tier=None,
            authority_role="structural",
            governance_state="active",
            valid_from=None,
            valid_to=None,
            source_ref=source_ref,
            source_anchor=fingerprint["normalized_structure_hash"],
            extraction_method="parser",
            extractor_version=fingerprint["extractor_version"],
        )
        return self.store.materialize_snapshot(
            organization_id=organization_id,
            ontology_version=ontology_version,
            resolver_version=resolver_version,
        )

    def _upsert_manifest(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_ref: str,
        fingerprint: dict[str, str],
        last_successful_run_ref: str,
        graph_snapshot_ref: str,
        graph_snapshot_hash: str,
    ) -> None:
        now = now_iso()
        with self.store._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_source_index_manifests (
                  organization_id, source_ref, workspace_id, index_fingerprint,
                  source_hash, normalized_structure_hash, extractor_version,
                  structure_schema_version, ontology_compatibility_version,
                  graph_projection_version, last_successful_run_ref,
                  graph_snapshot_ref, graph_snapshot_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organization_id, source_ref) DO UPDATE SET
                  workspace_id = excluded.workspace_id,
                  index_fingerprint = excluded.index_fingerprint,
                  source_hash = excluded.source_hash,
                  normalized_structure_hash = excluded.normalized_structure_hash,
                  extractor_version = excluded.extractor_version,
                  structure_schema_version = excluded.structure_schema_version,
                  ontology_compatibility_version = excluded.ontology_compatibility_version,
                  graph_projection_version = excluded.graph_projection_version,
                  last_successful_run_ref = excluded.last_successful_run_ref,
                  graph_snapshot_ref = excluded.graph_snapshot_ref,
                  graph_snapshot_hash = excluded.graph_snapshot_hash,
                  updated_at = excluded.updated_at
                """, '\n                INSERT INTO graph_source_index_manifests (\n                  organization_id, source_ref, workspace_id, index_fingerprint,\n                  source_hash, normalized_structure_hash, extractor_version,\n                  structure_schema_version, ontology_compatibility_version,\n                  graph_projection_version, last_successful_run_ref,\n                  graph_snapshot_ref, graph_snapshot_hash, updated_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                ON CONFLICT(organization_id, source_ref) DO UPDATE SET\n                  workspace_id = excluded.workspace_id,\n                  index_fingerprint = excluded.index_fingerprint,\n                  source_hash = excluded.source_hash,\n                  normalized_structure_hash = excluded.normalized_structure_hash,\n                  extractor_version = excluded.extractor_version,\n                  structure_schema_version = excluded.structure_schema_version,\n                  ontology_compatibility_version = excluded.ontology_compatibility_version,\n                  graph_projection_version = excluded.graph_projection_version,\n                  last_successful_run_ref = excluded.last_successful_run_ref,\n                  graph_snapshot_ref = excluded.graph_snapshot_ref,\n                  graph_snapshot_hash = excluded.graph_snapshot_hash,\n                  updated_at = excluded.updated_at\n                '),
                (
                    organization_id,
                    source_ref,
                    workspace_id,
                    fingerprint["index_fingerprint"],
                    fingerprint["source_hash"],
                    fingerprint["normalized_structure_hash"],
                    fingerprint["extractor_version"],
                    fingerprint["structure_schema_version"],
                    fingerprint["ontology_compatibility_version"],
                    fingerprint["graph_projection_version"],
                    last_successful_run_ref,
                    graph_snapshot_ref,
                    graph_snapshot_hash,
                    now,
                ),
            )

    def _upsert_freshness(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_ref: str,
        status: str,
        indexed_at: str | None,
        source_fingerprint: str | None,
        last_successful_index_ref: str | None,
        graph_snapshot_ref: str | None,
        graph_snapshot_hash: str | None,
        error_code: str | None,
        error_detail_ref: str | None,
    ) -> None:
        now = now_iso()
        with self.store._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO graph_index_freshness (
                  organization_id, source_ref, workspace_id, status, indexed_at,
                  source_fingerprint, last_successful_index_ref, graph_snapshot_ref,
                  graph_snapshot_hash, error_code, error_detail_ref, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organization_id, source_ref) DO UPDATE SET
                  workspace_id = excluded.workspace_id,
                  status = excluded.status,
                  indexed_at = excluded.indexed_at,
                  source_fingerprint = excluded.source_fingerprint,
                  last_successful_index_ref = excluded.last_successful_index_ref,
                  graph_snapshot_ref = excluded.graph_snapshot_ref,
                  graph_snapshot_hash = excluded.graph_snapshot_hash,
                  error_code = excluded.error_code,
                  error_detail_ref = excluded.error_detail_ref,
                  updated_at = excluded.updated_at
                """, '\n                INSERT INTO graph_index_freshness (\n                  organization_id, source_ref, workspace_id, status, indexed_at,\n                  source_fingerprint, last_successful_index_ref, graph_snapshot_ref,\n                  graph_snapshot_hash, error_code, error_detail_ref, updated_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                ON CONFLICT(organization_id, source_ref) DO UPDATE SET\n                  workspace_id = excluded.workspace_id,\n                  status = excluded.status,\n                  indexed_at = excluded.indexed_at,\n                  source_fingerprint = excluded.source_fingerprint,\n                  last_successful_index_ref = excluded.last_successful_index_ref,\n                  graph_snapshot_ref = excluded.graph_snapshot_ref,\n                  graph_snapshot_hash = excluded.graph_snapshot_hash,\n                  error_code = excluded.error_code,\n                  error_detail_ref = excluded.error_detail_ref,\n                  updated_at = excluded.updated_at\n                '),
                (
                    organization_id,
                    source_ref,
                    workspace_id,
                    status,
                    indexed_at,
                    source_fingerprint,
                    last_successful_index_ref,
                    graph_snapshot_ref,
                    graph_snapshot_hash,
                    error_code,
                    error_detail_ref,
                    now,
                ),
            )

    def _finish(
        self,
        *,
        ingestion_run_id: str,
        lease_owner: str,
        status: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in GRAPH_INGESTION_STATUS_TERMINAL:
            raise ValueError("GRAPH_INGESTION_STATUS_NOT_TERMINAL")
        now = now_iso()
        with self.store._connect() as conn:
            current = conn.execute(
                hosted.sql_query("""
                SELECT status
                FROM graph_ingestion_runs
                WHERE ingestion_run_id = ?
                  AND lease_owner = ?
                  AND status IN ('claimed', 'indexing')
                """, "\n                SELECT status\n                FROM graph_ingestion_runs\n                WHERE ingestion_run_id = %s\n                  AND lease_owner = %s\n                  AND status IN ('claimed', 'indexing')\n                "),
                (ingestion_run_id, lease_owner),
            ).fetchone()
            if current is None:
                raise ValueError("GRAPH_INGESTION_RUN_NOT_CLAIMED")
            assignments = [
                hosted.sql_query("status = ?", "status = %s"),
                "lease_owner = NULL",
                "lease_expires_at = NULL",
                hosted.sql_query("updated_at = ?", "updated_at = %s"),
                hosted.sql_query("completed_at = ?", "completed_at = %s"),
            ]
            params: list[Any] = [status, now, now]
            for column, value in updates.items():
                assignments.append(hosted.sql_query(f"{column} = ?", f"{column} = %s"))
                params.append(value)
            params.extend([ingestion_run_id, lease_owner])
            conn.execute(
                hosted.sql_query(f"""
                UPDATE graph_ingestion_runs
                SET {", ".join(assignments)}
                WHERE ingestion_run_id = ?
                  AND lease_owner = ?
                """, f"\n                UPDATE graph_ingestion_runs\n                SET {', '.join(assignments)}\n                WHERE ingestion_run_id = %s\n                  AND lease_owner = %s\n                "),
                params,
            )
            row = conn.execute(
                hosted.sql_query("SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = ?", 'SELECT * FROM graph_ingestion_runs WHERE ingestion_run_id = %s'),
                (ingestion_run_id,),
            ).fetchone()
        return _row_dict(row)


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        raise ValueError("GRAPH_INGESTION_RUN_NOT_FOUND")
    return dict(row)


def build_index_fingerprint(
    *,
    source_text: str,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
    structure_schema_version: str = DEFAULT_STRUCTURE_SCHEMA_VERSION,
    ontology_compatibility_version: str = DEFAULT_ONTOLOGY_COMPATIBILITY_VERSION,
    graph_projection_version: str = DEFAULT_GRAPH_PROJECTION_VERSION,
) -> dict[str, str]:
    normalized_structure = " ".join(source_text.split())
    source_hash = content_hash(source_text)
    normalized_structure_hash = content_hash(
        {
            "normalized_structure": normalized_structure,
            "structure_schema_version": structure_schema_version,
        }
    )
    payload = {
        "source_hash": source_hash,
        "normalized_structure_hash": normalized_structure_hash,
        "extractor_version": extractor_version,
        "structure_schema_version": structure_schema_version,
        "ontology_compatibility_version": ontology_compatibility_version,
        "graph_projection_version": graph_projection_version,
    }
    return {**payload, "index_fingerprint": content_hash(canonical_json(payload))}


def _external_outcome(status: str) -> str:
    if status == "no_change":
        return INDEXED_NO_CHANGE
    if status == "updated":
        return INDEXED_UPDATED
    if status == "failed_retryable":
        return INDEX_FAILED_RETRYABLE
    if status == "failed_terminal":
        return INDEX_FAILED_TERMINAL
    return status
