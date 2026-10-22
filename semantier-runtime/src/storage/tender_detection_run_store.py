from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from eos.generic_replay_envelope_store import (
    SENSITIVE_DETECTION_TYPE,
    GenericReplayEnvelopeStore,
)
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["project_metadata"] = _loads_json(payload["project_metadata_json"])
    payload["requested_rule_families"] = _loads_json(payload["requested_rule_families_json"])
    payload["ignored_local_rules"] = _loads_json(payload["ignored_local_rules_json"])
    return payload


def _row_to_finding(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["source_anchor_refs"] = _loads_json(payload["source_anchor_refs_json"])
    if "triggered_rule_version_refs_json" in payload:
        payload["triggered_rule_version_refs"] = _loads_json(
            payload["triggered_rule_version_refs_json"]
        )
    if "conflicting_rule_version_refs_json" in payload:
        payload["conflicting_rule_version_refs"] = _loads_json(
            payload["conflicting_rule_version_refs_json"]
        )
    if "precedent_refs_json" in payload:
        payload["precedent_refs"] = _loads_json(payload["precedent_refs_json"]) or []
    if "decision_context_json" in payload:
        payload["decision_context"] = _loads_json(payload["decision_context_json"]) or {}
    if "frame_derivation_json" in payload:
        payload["frame_derivation"] = _loads_json(payload["frame_derivation_json"]) or {}
    payload["schema_version"] = payload.get("candidate_finding_schema_version") or "contextgraph_candidate_finding.v2"
    payload["escalation_flag"] = bool(payload["escalation_flag"])
    return payload


def _row_to_runtime_conflict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["rule_version_ids"] = _loads_json(payload["rule_version_ids_json"])
    payload["conflict_payload"] = _loads_json(payload["conflict_payload_json"])
    return payload


class TenderDetectionRunStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        self._db_path = path
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'tender_detection_dispositions','tender_detection_findings','tender_detection_runs','tender_detection_runtime_conflicts')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            prior_db_path = os.environ.get("SEMANTIER_EOS_DB_PATH")
            os.environ["SEMANTIER_EOS_DB_PATH"] = str(path)
            try:
                init_db()
            finally:
                if prior_db_path is None:
                    os.environ.pop("SEMANTIER_EOS_DB_PATH", None)
                else:
                    os.environ["SEMANTIER_EOS_DB_PATH"] = prior_db_path
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    @property
    def db_path(self) -> Path:
        """Authoritative SQLite path used by this store composition."""

        return self._db_path

    def create_run(
        self,
        *,
        run_id: str | None,
        tender_document_id: str,
        document_text: str,
        organization_id: str,
        workspace_id: str | None,
        project_metadata: dict[str, Any],
        review_purpose: str,
        caller_role: str,
        requested_rule_families: list[str],
        activation_set_snapshot_id: str,
        activation_resolver_policy_version: str,
        resolved_activation_set_hash: str,
        ignored_local_rules: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        runtime_conflicts: list[dict[str, Any]] | None = None,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
        request_hash: str | None = None,
    ) -> dict[str, Any]:
        normalized_run_id = run_id or f"tdr_{uuid4().hex}"
        now = _now_iso()
        source_document_hash = compute_content_hash(document_text)
        root_run_id = parent_run_id
        if parent_run_id:
            parent = self.get_run(parent_run_id)
            root_run_id = parent.get("root_run_id") or parent_run_id if parent else parent_run_id
        run_payload = {
            "run_id": normalized_run_id,
            "tender_document_id": tender_document_id,
            "document_text": document_text,
            "source_document_hash": source_document_hash,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "parent_run_id": parent_run_id,
            "root_run_id": root_run_id or normalized_run_id,
            "project_metadata": project_metadata,
            "review_purpose": review_purpose,
            "caller_role": caller_role,
            "requested_rule_families": requested_rule_families,
            "activation_set_snapshot_id": activation_set_snapshot_id,
            "activation_resolver_policy_version": activation_resolver_policy_version,
            "resolved_activation_set_hash": resolved_activation_set_hash,
            "ignored_local_rules": ignored_local_rules,
        }
        run_hash = compute_content_hash(canonical_json(run_payload))
        finding_set_hash = compute_content_hash(canonical_json(findings))
        runtime_conflict_set_hash = compute_content_hash(canonical_json(runtime_conflicts or []))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO tender_detection_runs (
                  run_id, tender_document_id, document_text, source_document_hash,
                  organization_id, workspace_id, parent_run_id, root_run_id,
                  project_metadata_json, review_purpose, caller_role,
                  requested_rule_families_json, activation_set_snapshot_id,
                  activation_resolver_policy_version, resolved_activation_set_hash,
                  ignored_local_rules_json, idempotency_key, request_hash, run_status,
                  finding_set_hash, runtime_conflict_set_hash, replay_status,
                  created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO tender_detection_runs (\n                  run_id, tender_document_id, document_text, source_document_hash,\n                  organization_id, workspace_id, parent_run_id, root_run_id,\n                  project_metadata_json, review_purpose, caller_role,\n                  requested_rule_families_json, activation_set_snapshot_id,\n                  activation_resolver_policy_version, resolved_activation_set_hash,\n                  ignored_local_rules_json, idempotency_key, request_hash, run_status,\n                  finding_set_hash, runtime_conflict_set_hash, replay_status,\n                  created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    normalized_run_id,
                    tender_document_id,
                    document_text,
                    source_document_hash,
                    organization_id,
                    workspace_id,
                    parent_run_id,
                    root_run_id or normalized_run_id,
                    _as_json(project_metadata),
                    review_purpose,
                    caller_role,
                    _as_json(requested_rule_families),
                    activation_set_snapshot_id,
                    activation_resolver_policy_version,
                    resolved_activation_set_hash,
                    _as_json(ignored_local_rules),
                    idempotency_key,
                    request_hash,
                    "FINALIZED",
                    finding_set_hash,
                    runtime_conflict_set_hash,
                    "REPLAY_PENDING",
                    now,
                    run_hash,
                ),
            )
            for finding in findings:
                self._insert_finding(normalized_run_id, tender_document_id, finding, now)
            for conflict in runtime_conflicts or []:
                self._insert_runtime_conflict(normalized_run_id, conflict, now)
            self._conn.commit()
        run = self.get_run(normalized_run_id)
        if run is None:
            raise ValueError(f"tender detection run not found after create: {normalized_run_id}")
        replay_binding = self._create_detection_replay_binding(run)
        self._mark_replay_created(normalized_run_id)
        run["detection_replay_binding"] = replay_binding
        return run

    def create_pending_run_idempotent(
        self,
        *,
        run_id: str | None,
        tender_document_id: str,
        document_text: str,
        organization_id: str,
        workspace_id: str | None,
        parent_run_id: str | None,
        project_metadata: dict[str, Any],
        review_purpose: str,
        caller_role: str,
        requested_rule_families: list[str],
        activation_set_snapshot_id: str,
        activation_resolver_policy_version: str,
        resolved_activation_set_hash: str,
        ignored_local_rules: list[dict[str, Any]],
        idempotency_key: str | None,
        request_hash: str,
    ) -> dict[str, Any]:
        normalized_run_id = run_id or f"tdr_{uuid4().hex}"
        source_document_hash = compute_content_hash(document_text)
        root_run_id = parent_run_id
        if parent_run_id:
            parent = self.get_run(parent_run_id)
            root_run_id = parent.get("root_run_id") or parent_run_id if parent else parent_run_id
        existing = None
        with self._lock:
            if idempotency_key:
                existing = self._conn.execute(
                    sql_query("""
                    SELECT * FROM tender_detection_runs
                    WHERE organization_id = ?
                      AND COALESCE(workspace_id, '') = COALESCE(?, '')
                      AND idempotency_key = ?
                    """, "\n                    SELECT * FROM tender_detection_runs\n                    WHERE organization_id = %s\n                      AND COALESCE(workspace_id, '') = COALESCE(%s, '')\n                      AND idempotency_key = %s\n                    "),
                    (organization_id, workspace_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    existing_payload = _row_to_run(existing)
                    if existing_payload.get("request_hash") != request_hash:
                        raise ValueError("DETECTION_IDEMPOTENCY_CONFLICT")
                    run = self.get_run(existing_payload["run_id"])
                    if run is None:
                        raise ValueError("DETECTION_RUN_NOT_FOUND")
                    return run
            now = _now_iso()
            run_payload = {
                "run_id": normalized_run_id,
                "tender_document_id": tender_document_id,
                "document_text": document_text,
                "source_document_hash": source_document_hash,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "parent_run_id": parent_run_id,
                "root_run_id": root_run_id or normalized_run_id,
                "project_metadata": project_metadata,
                "review_purpose": review_purpose,
                "caller_role": caller_role,
                "requested_rule_families": requested_rule_families,
                "activation_set_snapshot_id": activation_set_snapshot_id,
                "activation_resolver_policy_version": activation_resolver_policy_version,
                "resolved_activation_set_hash": resolved_activation_set_hash,
                "ignored_local_rules": ignored_local_rules,
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "run_status": "PENDING",
            }
            run_hash = compute_content_hash(canonical_json(run_payload))
            try:
                self._conn.execute(
                    sql_query("""
                    INSERT INTO tender_detection_runs (
                      run_id, tender_document_id, document_text, source_document_hash,
                      organization_id, workspace_id, parent_run_id, root_run_id,
                      project_metadata_json, review_purpose, caller_role,
                      requested_rule_families_json, activation_set_snapshot_id,
                      activation_resolver_policy_version, resolved_activation_set_hash,
                      ignored_local_rules_json, idempotency_key, request_hash, run_status,
                      replay_status, created_at, content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, '\n                    INSERT INTO tender_detection_runs (\n                      run_id, tender_document_id, document_text, source_document_hash,\n                      organization_id, workspace_id, parent_run_id, root_run_id,\n                      project_metadata_json, review_purpose, caller_role,\n                      requested_rule_families_json, activation_set_snapshot_id,\n                      activation_resolver_policy_version, resolved_activation_set_hash,\n                      ignored_local_rules_json, idempotency_key, request_hash, run_status,\n                      replay_status, created_at, content_hash\n                    )\n                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                    '),
                    (
                        normalized_run_id,
                        tender_document_id,
                        document_text,
                        source_document_hash,
                        organization_id,
                        workspace_id,
                        parent_run_id,
                        root_run_id or normalized_run_id,
                        _as_json(project_metadata),
                        review_purpose,
                        caller_role,
                        _as_json(requested_rule_families),
                        activation_set_snapshot_id,
                        activation_resolver_policy_version,
                        resolved_activation_set_hash,
                        _as_json(ignored_local_rules),
                        idempotency_key,
                        request_hash,
                        "PENDING",
                        "NOT_CREATED",
                        now,
                        run_hash,
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                if not idempotency_key:
                    raise
                existing = self._conn.execute(
                    sql_query("""
                    SELECT * FROM tender_detection_runs
                    WHERE organization_id = ?
                      AND COALESCE(workspace_id, '') = COALESCE(?, '')
                      AND idempotency_key = ?
                    """, "\n                    SELECT * FROM tender_detection_runs\n                    WHERE organization_id = %s\n                      AND COALESCE(workspace_id, '') = COALESCE(%s, '')\n                      AND idempotency_key = %s\n                    "),
                    (organization_id, workspace_id, idempotency_key),
                ).fetchone()
                if existing is None:
                    raise
                existing_payload = _row_to_run(existing)
                if existing_payload.get("request_hash") != request_hash:
                    raise ValueError("DETECTION_IDEMPOTENCY_CONFLICT")
                run = self.get_run(existing_payload["run_id"])
                if run is None:
                    raise ValueError("DETECTION_RUN_NOT_FOUND")
                return run
        run = self.get_run(normalized_run_id)
        if run is None:
            raise ValueError(f"tender detection run not found after pending create: {normalized_run_id}")
        return run

    def persist_findings(
        self,
        *,
        run_id: str,
        findings: list[dict[str, Any]],
        finding_set_hash: str,
    ) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        if run.get("run_status") == "FINALIZED":
            if run.get("finding_set_hash") != finding_set_hash:
                raise ValueError("FINDING_SET_HASH_MISMATCH")
            return
        now = _now_iso()
        with self._lock:
            self._conn.execute(sql_query("DELETE FROM tender_detection_findings WHERE run_id = ?", 'DELETE FROM tender_detection_findings WHERE run_id = %s'), (run_id,))
            for finding in findings:
                self._insert_finding(run_id, run["tender_document_id"], finding, now)
            self._conn.execute(
                sql_query("UPDATE tender_detection_runs SET finding_set_hash = ? WHERE run_id = ?", 'UPDATE tender_detection_runs SET finding_set_hash = %s WHERE run_id = %s'),
                (finding_set_hash, run_id),
            )
            self._conn.commit()

    def persist_runtime_conflicts(
        self,
        *,
        run_id: str,
        runtime_conflicts: list[dict[str, Any]],
        runtime_conflict_set_hash: str,
    ) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        if run.get("run_status") == "FINALIZED":
            if run.get("runtime_conflict_set_hash") != runtime_conflict_set_hash:
                raise ValueError("RUNTIME_CONFLICT_SET_HASH_MISMATCH")
            return
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                sql_query("DELETE FROM tender_detection_runtime_conflicts WHERE run_id = ?", 'DELETE FROM tender_detection_runtime_conflicts WHERE run_id = %s'),
                (run_id,),
            )
            for conflict in runtime_conflicts:
                self._insert_runtime_conflict(run_id, conflict, now)
            self._conn.execute(
                sql_query("UPDATE tender_detection_runs SET runtime_conflict_set_hash = ? WHERE run_id = ?", 'UPDATE tender_detection_runs SET runtime_conflict_set_hash = %s WHERE run_id = %s'),
                (runtime_conflict_set_hash, run_id),
            )
            self._conn.commit()

    def finalize_run(
        self,
        *,
        run_id: str,
        finding_set_hash: str,
        runtime_conflict_set_hash: str,
        status: str,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        if run.get("finding_set_hash") != finding_set_hash:
            raise ValueError("FINDING_SET_HASH_MISMATCH")
        if run.get("runtime_conflict_set_hash") != runtime_conflict_set_hash:
            raise ValueError("RUNTIME_CONFLICT_SET_HASH_MISMATCH")
        if run.get("run_status") == "FINALIZED" and run.get("replay_status") == "CREATED":
            replay_binding = self._create_detection_replay_binding(run)
            run["detection_replay_binding"] = replay_binding
            return run
        with self._lock:
            self._conn.execute(
                sql_query("""
                UPDATE tender_detection_runs
                SET run_status = ?, replay_status = ?
                WHERE run_id = ?
                """, '\n                UPDATE tender_detection_runs\n                SET run_status = %s, replay_status = %s\n                WHERE run_id = %s\n                '),
                ("REPLAY_PENDING", "REPLAY_PENDING", run_id),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        try:
            replay_binding = self._create_detection_replay_binding(run)
        except Exception:
            with self._lock:
                self._conn.execute(
                    sql_query("UPDATE tender_detection_runs SET run_status = ?, replay_status = ? WHERE run_id = ?", 'UPDATE tender_detection_runs SET run_status = %s, replay_status = %s WHERE run_id = %s'),
                    ("REPLAY_PENDING", "REPLAY_PENDING", run_id),
                )
                self._conn.commit()
            raise
        with self._lock:
            self._conn.execute(
                sql_query("UPDATE tender_detection_runs SET run_status = ?, replay_status = ? WHERE run_id = ?", 'UPDATE tender_detection_runs SET run_status = %s, replay_status = %s WHERE run_id = %s'),
                (status.upper(), "CREATED", run_id),
            )
            self._conn.commit()
        finalized = self.get_run(run_id)
        if finalized is None:
            raise ValueError("DETECTION_RUN_NOT_FOUND")
        finalized["detection_replay_binding"] = replay_binding
        return finalized

    def _mark_replay_created(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute(
                sql_query("UPDATE tender_detection_runs SET replay_status = ? WHERE run_id = ?", 'UPDATE tender_detection_runs SET replay_status = %s WHERE run_id = %s'),
                ("CREATED", run_id),
            )
            self._conn.commit()

    def _create_detection_replay_binding(self, run: dict[str, Any]) -> dict[str, Any]:
        project_metadata = run.get("project_metadata") or {}
        finding_set_hash = compute_content_hash(
            canonical_json(
                [
                    {
                        "finding_id": finding["finding_id"],
                        "target_anchor_ref": finding.get("target_anchor_ref"),
                        "start_offset": finding.get("start_offset"),
                        "end_offset": finding.get("end_offset"),
                        "issue_type": finding["issue_type"],
                        "triggered_rule_version_refs": finding.get(
                            "triggered_rule_version_refs"
                        )
                        or [],
                        "content_hash": finding["content_hash"],
                    }
                    for finding in run.get("findings", [])
                ]
            )
        )
        payload = {
            "run_id": run["run_id"],
            "target_document_hash": run["source_document_hash"],
            "target_source_hash": project_metadata.get("target_source_hash"),
            "target_normalized_document_artifact_ref": project_metadata.get(
                "target_normalized_document_artifact_ref"
            ),
            "target_normalized_document_artifact_hash": project_metadata.get(
                "target_normalized_document_artifact_hash"
            ),
            "document_schema_version": project_metadata.get("document_schema_version")
            or (
                "canonical_document.v2"
                if project_metadata.get("target_normalized_document_artifact_ref")
                else "plain_text.v1"
            ),
            "activation_set_snapshot_id": run["activation_set_snapshot_id"],
            "activation_resolver_policy_version": run[
                "activation_resolver_policy_version"
            ],
            "resolved_activation_set_hash": run["resolved_activation_set_hash"],
            "governed_rule_activation_ids": project_metadata.get(
                "governed_rule_activation_ids"
            )
            or [],
            "rule_version_refs": project_metadata.get("rule_version_refs") or [],
            "rule_version_set_hash": project_metadata.get("rule_version_set_hash"),
            "detector_version": "tender_document_review.v2",
            "matching_policy": "literal_substring_occurrence.v1",
            "normalization_policy": "casefold_display_match.v1",
            "semantic_profile_ids": project_metadata.get("semantic_profile_ids") or [],
            "finding_set_hash": finding_set_hash,
            "finding_refs": [
                {
                    "finding_id": finding["finding_id"],
                    "content_hash": finding["content_hash"],
                }
                for finding in run.get("findings", [])
            ],
            "runtime_conflict_set_hash": project_metadata.get("runtime_conflict_set_hash"),
            "runtime_conflict_refs": [
                {
                    "conflict_id": conflict["conflict_id"],
                    "content_hash": conflict["content_hash"],
                }
                for conflict in run.get("runtime_conflicts", [])
            ],
        }
        return GenericReplayEnvelopeStore(db_path=self._db_path).create_typed_replay_binding(
            replay_binding_id=f"sensitive_labeling_detection:{run['run_id']}",
            binding_type=SENSITIVE_DETECTION_TYPE,
            binding_schema_version="sensitive_labeling_detection_binding.v1",
            subject_type="tender_detection_run",
            subject_id=run["run_id"],
            organization_id=run["organization_id"],
            workspace_id=run.get("workspace_id"),
            payload=payload,
        )

    def _insert_runtime_conflict(
        self,
        run_id: str,
        conflict: dict[str, Any],
        created_at: str,
    ) -> None:
        conflict_id = str(conflict.get("conflict_id") or f"tdrc_{uuid4().hex}")
        payload = {
            "conflict_id": conflict_id,
            "run_id": run_id,
            **conflict,
        }
        conflict_hash = str(
            conflict.get("content_hash") or compute_content_hash(canonical_json(payload))
        )
        self._conn.execute(
            sql_query("""
            INSERT INTO tender_detection_runtime_conflicts (
              conflict_id, run_id, conflict_type, target_anchor_ref,
              rule_version_ids_json, start_offset, end_offset,
              conflict_payload_json, created_at, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO tender_detection_runtime_conflicts (\n              conflict_id, run_id, conflict_type, target_anchor_ref,\n              rule_version_ids_json, start_offset, end_offset,\n              conflict_payload_json, created_at, content_hash\n            )\n            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                conflict_id,
                run_id,
                str(conflict.get("conflict_type") or ""),
                conflict.get("target_anchor_ref"),
                _as_json(conflict.get("rule_version_ids") or []),
                conflict.get("start_offset"),
                conflict.get("end_offset"),
                _as_json(conflict),
                created_at,
                conflict_hash,
            ),
        )

    def _insert_finding(
        self,
        run_id: str,
        tender_document_id: str,
        finding: dict[str, Any],
        created_at: str,
    ) -> None:
        finding_id = str(finding.get("finding_id") or f"tdf_{uuid4().hex}")
        persisted_schema_version = "contextgraph_candidate_finding.v2"
        payload = {
            "finding_id": finding_id,
            "run_id": run_id,
            "tender_document_id": tender_document_id,
            "schema_version": persisted_schema_version,
            **finding,
        }
        payload["run_id"] = run_id
        payload["schema_version"] = persisted_schema_version
        finding_hash = compute_content_hash(canonical_json(payload))
        self._conn.execute(
            sql_query("""
            INSERT INTO tender_detection_findings (
              finding_id, run_id, tender_document_id, clause_id, clause_anchor,
              issue_type, matched_text, triggered_rule_version_id,
              source_anchor_refs_json, judgment_basis, severity, confidence,
              suggested_replacement, escalation_flag, resolver_evidence_ref,
              target_anchor_ref, target_element_kind, target_element_id,
              normalized_match, start_offset, end_offset, occurrence_index,
              target_element_hash, triggered_rule_version_refs_json,
              winning_rule_version_ref, conflicting_rule_version_refs_json,
              conflict_resolution_ref, target_evidence_ref,
              source_graph_version, source_graph_release_hash, source_graph_rule_id,
              matched_sensitive_concept_id, observed_expression, detection_method,
              semantic_relation, evidence_text, precedent_refs_json, decision_status,
              grounding_status, decision_context_json, candidate_finding_schema_version,
              semantic_profile_id, derivation_shape, frame_derivation_json,
              created_at, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO tender_detection_findings (\n              finding_id, run_id, tender_document_id, clause_id, clause_anchor,\n              issue_type, matched_text, triggered_rule_version_id,\n              source_anchor_refs_json, judgment_basis, severity, confidence,\n              suggested_replacement, escalation_flag, resolver_evidence_ref,\n              target_anchor_ref, target_element_kind, target_element_id,\n              normalized_match, start_offset, end_offset, occurrence_index,\n              target_element_hash, triggered_rule_version_refs_json,\n              winning_rule_version_ref, conflicting_rule_version_refs_json,\n              conflict_resolution_ref, target_evidence_ref,\n              source_graph_version, source_graph_release_hash, source_graph_rule_id,\n              matched_sensitive_concept_id, observed_expression, detection_method,\n              semantic_relation, evidence_text, precedent_refs_json, decision_status,\n              grounding_status, decision_context_json, candidate_finding_schema_version,\n              semantic_profile_id, derivation_shape, frame_derivation_json,\n              created_at, content_hash\n            )\n            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                finding_id,
                run_id,
                tender_document_id,
                str(finding.get("clause_id") or "document"),
                str(finding.get("clause_anchor") or "document"),
                str(finding["issue_type"]),
                str(finding["matched_text"]),
                str(finding["triggered_rule_version_id"]),
                _as_json(finding.get("source_anchor_refs") or []),
                str(finding["judgment_basis"]),
                str(finding.get("severity") or "medium"),
                float(finding.get("confidence") or 0.0),
                finding.get("suggested_replacement"),
                1 if finding.get("escalation_flag") else 0,
                str(finding["resolver_evidence_ref"]),
                finding.get("target_anchor_ref"),
                finding.get("target_element_kind"),
                finding.get("target_element_id"),
                finding.get("normalized_match"),
                finding.get("start_offset"),
                finding.get("end_offset"),
                finding.get("occurrence_index"),
                finding.get("target_element_hash"),
                _as_json(finding.get("triggered_rule_version_refs") or []),
                finding.get("winning_rule_version_ref"),
                _as_json(finding.get("conflicting_rule_version_refs") or []),
                finding.get("conflict_resolution_ref"),
                finding.get("target_evidence_ref"),
                finding.get("source_graph_version"),
                finding.get("source_graph_release_hash"),
                finding.get("source_graph_rule_id"),
                finding.get("matched_sensitive_concept_id")
                or finding.get("source_graph_rule_id")
                or finding.get("triggered_rule_version_id"),
                finding.get("observed_expression") or finding.get("matched_text"),
                finding.get("detection_method") or "exact",
                finding.get("semantic_relation") or "exact",
                finding.get("evidence_text") or finding.get("matched_text"),
                _as_json(finding.get("precedent_refs") or []),
                finding.get("decision_status")
                or ("escalate" if finding.get("escalation_flag") else "violation"),
                finding.get("grounding_status") or "ungrounded",
                _as_json(finding.get("decision_context") or {}),
                persisted_schema_version,
                finding.get("semantic_profile_id"),
                finding.get("derivation_shape"),
                _as_json(finding.get("frame_derivation") or {}),
                created_at,
                finding_hash,
            ),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._conn.execute(
            sql_query("SELECT * FROM tender_detection_runs WHERE run_id = ?", 'SELECT * FROM tender_detection_runs WHERE run_id = %s'),
            (run_id,),
        ).fetchone()
        if run is None:
            return None
        findings = self._conn.execute(
            sql_query("""
            SELECT * FROM tender_detection_findings
            WHERE run_id = ?
            ORDER BY
              created_at,
              target_anchor_ref IS NULL,
              target_anchor_ref,
              start_offset IS NULL,
              start_offset,
              occurrence_index IS NULL,
              occurrence_index,
              finding_id
            """, '\n            SELECT * FROM tender_detection_findings\n            WHERE run_id = %s\n            ORDER BY\n              created_at,\n              target_anchor_ref IS NULL,\n              target_anchor_ref,\n              start_offset IS NULL,\n              start_offset,\n              occurrence_index IS NULL,\n              occurrence_index,\n              finding_id\n            '),
            (run_id,),
        ).fetchall()
        payload = _row_to_run(run)
        payload["findings"] = [_row_to_finding(row) for row in findings]
        payload["runtime_conflicts"] = self.list_runtime_conflicts(run_id)
        payload["dispositions"] = self.list_dispositions(run_id)
        return payload

    def list_runtime_conflicts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT * FROM tender_detection_runtime_conflicts
            WHERE run_id = ?
            ORDER BY created_at, target_anchor_ref, start_offset, end_offset, conflict_id
            """, '\n            SELECT * FROM tender_detection_runtime_conflicts\n            WHERE run_id = %s\n            ORDER BY created_at, target_anchor_ref, start_offset, end_offset, conflict_id\n            '),
            (run_id,),
        ).fetchall()
        return [_row_to_runtime_conflict(row) for row in rows]

    def record_disposition(
        self,
        *,
        run_id: str,
        finding_id: str,
        disposition: str,
        decided_by: str,
        edited_replacement: str | None = None,
        rejection_rationale: str | None = None,
    ) -> dict[str, Any]:
        normalized = disposition.strip().lower()
        if normalized not in {"accepted", "rejected", "edited", "deferred", "escalated"}:
            raise ValueError("disposition must be accepted, rejected, edited, deferred, or escalated")
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"tender detection run not found: {run_id}")
        finding_ids = {finding["finding_id"] for finding in run["findings"]}
        if finding_id not in finding_ids:
            raise ValueError(f"finding not found for run: {finding_id}")
        now = _now_iso()
        disposition_id = f"tdd_{uuid4().hex}"
        payload = {
            "disposition_id": disposition_id,
            "run_id": run_id,
            "finding_id": finding_id,
            "disposition": normalized,
            "edited_replacement": edited_replacement,
            "rejection_rationale": rejection_rationale,
            "decided_by": decided_by,
            "decided_at": now,
        }
        payload["content_hash"] = compute_content_hash(canonical_json(payload))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO tender_detection_dispositions (
                  disposition_id, run_id, finding_id, disposition,
                  edited_replacement, rejection_rationale, decided_by, decided_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO tender_detection_dispositions (\n                  disposition_id, run_id, finding_id, disposition,\n                  edited_replacement, rejection_rationale, decided_by, decided_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    disposition_id,
                    run_id,
                    finding_id,
                    normalized,
                    edited_replacement,
                    rejection_rationale,
                    decided_by,
                    now,
                    payload["content_hash"],
                ),
            )
            self._conn.commit()
        return payload

    def list_dispositions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT * FROM tender_detection_dispositions
            WHERE run_id = ?
            ORDER BY decided_at, disposition_id
            """, '\n            SELECT * FROM tender_detection_dispositions\n            WHERE run_id = %s\n            ORDER BY decided_at, disposition_id\n            '),
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
