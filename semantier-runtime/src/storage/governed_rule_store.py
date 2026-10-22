from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db, verify_rule_governance_schema_ready
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock
from storage.governed_rule_activation_store import GovernedRuleActivationStore
from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore
from storage.policy_rule_candidate_store import PolicyRuleCandidateStore

RuleState = Literal["APPROVED", "ACTIVATED", "SUPERSEDED", "REJECTED"]


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _assert_ascii(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    hydrated = dict(row)
    for field in (
        "source_anchor_refs_json",
        "applicability_scope_json",
        "reviewer_edits_json",
        "approval_evidence_json",
        "test_evidence_json",
        "activation_refs_json",
    ):
        hydrated[field.removesuffix("_json")] = _loads_json(hydrated[field])
    return hydrated


class GovernedRuleStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        self._db_path=path
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'governed_rule_versions','knowledge_activation_set_snapshot','knowledge_source')
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
            verify_rule_governance_schema_ready(self._conn)
        self._lock = StoreMutationLock(self._conn)
        self._candidate_store = PolicyRuleCandidateStore(db_path=path)
        self._activation_store = GovernedRuleActivationStore(db_path=path)

    def create_approved_version(
        self,
        *,
        rule_version_id: str | None = None,
        rule_candidate_id: str,
        actor: str,
        actor_authority: str,
        supersedes_rule_version_id: str | None = None,
    ) -> dict[str, Any]:
        actor = _assert_ascii(actor, "actor")
        candidate = self._candidate_store.get_candidate(rule_candidate_id)
        if candidate is None:
            raise ValueError(f"rule candidate not found: {rule_candidate_id}")
        if candidate["candidate_state"] != "APPROVED":
            raise ValueError("only APPROVED candidates can become governed rule versions")
        self._assert_candidate_source_admissible(candidate)
        scope = candidate.get("applicability_scope") or {}
        source_graph_version = scope.get("source_graph_version")
        source_graph_release_hash = scope.get("source_graph_release_hash")
        source_graph_rule_id = scope.get("source_graph_rule_id")
        lineage = (source_graph_version, source_graph_release_hash, source_graph_rule_id)
        if any(value is not None for value in lineage) and not all(lineage):
            raise ValueError("graph rule lineage must be all-or-none")
        version_id = _assert_ascii(rule_version_id or f"grv_{uuid4().hex}", "rule_version_id")
        now = _now_iso()
        payload = {
            "rule_version_id": version_id,
            "rule_candidate_id": rule_candidate_id,
            "organization_id": candidate.get("organization_id"),
            "authority_domain": candidate.get("authority_domain"),
            "semantic_tier": candidate.get("semantic_tier"),
            "jurisdiction": candidate.get("jurisdiction"),
            "rule_family": candidate["rule_family"],
            "rule_state": "APPROVED",
            "source_graph_version": source_graph_version,
            "source_graph_release_hash": source_graph_release_hash,
            "source_graph_rule_id": source_graph_rule_id,
            "source_anchor_refs": candidate["source_anchor_refs"],
            "applicability_scope": candidate["applicability_scope"],
            "governed_rule_text": candidate["draft_rule_text"],
            "extracted_rationale": candidate["extracted_rationale"],
            "severity": candidate["severity"],
            "confidence": candidate["confidence"],
            "uncertainty_notes": candidate["uncertainty_notes"],
            "reviewer_edits": candidate["human_edits"],
            "approval_evidence": candidate["approval_evidence"],
            "test_evidence": candidate["test_evidence"],
            "activation_refs": [],
            "supersedes_rule_version_id": supersedes_rule_version_id,
        }
        version_hash = compute_content_hash(canonical_json(payload))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO governed_rule_versions (
                  rule_version_id, rule_candidate_id, organization_id,
                  authority_domain, semantic_tier, jurisdiction,
                  rule_family, rule_state,
                  source_graph_version, source_graph_release_hash, source_graph_rule_id,
                  source_anchor_refs_json, applicability_scope_json,
                  governed_rule_text, extracted_rationale, severity, confidence,
                  uncertainty_notes, reviewer_edits_json, approval_evidence_json,
                  test_evidence_json, activation_refs_json, supersedes_rule_version_id,
                  activated_at, activated_by, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """, '\n                INSERT INTO governed_rule_versions (\n                  rule_version_id, rule_candidate_id, organization_id,\n                  authority_domain, semantic_tier, jurisdiction,\n                  rule_family, rule_state,\n                  source_graph_version, source_graph_release_hash, source_graph_rule_id,\n                  source_anchor_refs_json, applicability_scope_json,\n                  governed_rule_text, extracted_rationale, severity, confidence,\n                  uncertainty_notes, reviewer_edits_json, approval_evidence_json,\n                  test_evidence_json, activation_refs_json, supersedes_rule_version_id,\n                  activated_at, activated_by, content_hash, created_at\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s)\n                '),
                (
                    version_id,
                    rule_candidate_id,
                    candidate.get("organization_id"),
                    candidate.get("authority_domain"),
                    candidate.get("semantic_tier"),
                    candidate.get("jurisdiction"),
                    candidate["rule_family"],
                    "APPROVED",
                    source_graph_version,
                    source_graph_release_hash,
                    source_graph_rule_id,
                    _as_json(candidate["source_anchor_refs"]),
                    _as_json(candidate["applicability_scope"]),
                    candidate["draft_rule_text"],
                    candidate["extracted_rationale"],
                    candidate["severity"],
                    candidate["confidence"],
                    candidate["uncertainty_notes"],
                    _as_json(candidate["human_edits"]),
                    _as_json(candidate["approval_evidence"]),
                    _as_json(candidate["test_evidence"]),
                    _as_json([]),
                    supersedes_rule_version_id,
                    version_hash,
                    now,
                ),
            )
            self._conn.commit()
            return self.get_rule_version(version_id)  # type: ignore[return-value]

    def _assert_candidate_source_admissible(self, candidate: dict[str, Any]) -> None:
        scope = candidate.get("applicability_scope") or {}
        if not isinstance(scope, dict):
            return
        knowledge_source_id = str(scope.get("knowledge_source_id") or "").strip()
        if not knowledge_source_id:
            return
        row = self._conn.execute(
            sql_query("SELECT * FROM knowledge_source WHERE knowledge_source_id = ?", 'SELECT * FROM knowledge_source WHERE knowledge_source_id = %s'),
            (knowledge_source_id,),
        ).fetchone()
        if row is None:
            raise ValueError("candidate knowledge_source_id not found")
        source = dict(row)
        if source.get("source_lifecycle_status") not in {"approved", "validated"}:
            raise ValueError("candidate knowledge source is not admissible for governed rule version")
        if source.get("source_kind") == "controlled_document":
            if source.get("semantic_purpose") != "sensitive_lexicon":
                raise ValueError("controlled document semantic_purpose is not admissible")
            KnowledgeSourceActivationStore(
                db_path=self._db_path
            ).resolve_controlled_document_compiler_route(knowledge_source_id)

    def activate_version(
        self,
        rule_version_id: str,
        *,
        activation_ref: str,
        actor: str,
        actor_authority: str,
        organization_id: str | None = None,
        activation_scope: Literal["organization", "workspace"] = "organization",
        workspace_id: str | None = None,
        effective_from: str | None = None,
        effective_to: str | None = None,
        approval_event_ref: str | None = None,
    ) -> dict[str, Any]:
        actor = _assert_ascii(actor, "actor")
        version_id = _assert_ascii(rule_version_id, "rule_version_id")
        activation_ref = _assert_ascii(activation_ref, "activation_ref")
        now = _now_iso()
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM governed_rule_versions WHERE rule_version_id = ?", 'SELECT * FROM governed_rule_versions WHERE rule_version_id = %s'),
                (version_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"governed rule version not found: {version_id}")
            if row["rule_state"] != "APPROVED":
                raise ValueError("only APPROVED governed rule versions can be activated")
        organization_id = organization_id or row["organization_id"]
        if not organization_id:
            raise ValueError("organization_id is required for governed rule activation")
        activation = self._activation_store.create_activation(
            activation_id=activation_ref,
            rule_version_id=version_id,
            organization_id=organization_id,
            activation_scope=activation_scope,
            workspace_id=workspace_id,
            effective_from=effective_from or now,
            effective_to=effective_to,
            activated_by=actor,
            actor_authority=actor_authority,
            approval_event_ref=approval_event_ref or f"approval_{version_id}",
            source_activation_ref=activation_ref,
        )
        self._candidate_store.mark_activated(
            row["rule_candidate_id"],
            actor=actor,
            actor_authority=actor_authority,
            activation_ref=activation_ref,
        )
        version = self.get_rule_version(version_id)
        if version is None:
            raise ValueError(f"governed rule version not found: {version_id}")
        version["activation"] = activation
        version["activation_refs"] = [activation_ref]
        return version

    def get_rule_version(self, rule_version_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM governed_rule_versions WHERE rule_version_id = ?", 'SELECT * FROM governed_rule_versions WHERE rule_version_id = %s'),
            (_assert_ascii(rule_version_id, "rule_version_id"),),
        ).fetchone()
        return _row_to_dict(row)

    def list_runtime_rules(self, *, rule_family: str | None = None) -> list[dict[str, Any]]:
        clauses = ["rule_state = 'ACTIVATED'"]
        values: list[Any] = []
        if rule_family is not None:
            clauses.append(sql_query("rule_family = ?", 'rule_family = %s'))
            values.append(_assert_ascii(rule_family, "rule_family"))
        rows = self._conn.execute(
            f"SELECT * FROM governed_rule_versions WHERE {' AND '.join(clauses)} ORDER BY created_at",
            values,
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def list_activation_snapshots(
        self, *, organization_id: str, workspace_id: str | None
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT *
            FROM knowledge_activation_set_snapshot
            WHERE organization_id = ?
              AND ((workspace_id = ?) OR (workspace_id IS NULL AND ? IS NULL))
            ORDER BY effective_at DESC, created_at DESC
            """, '\n            SELECT *\n            FROM knowledge_activation_set_snapshot\n            WHERE organization_id = %s\n              AND ((workspace_id = %s) OR (workspace_id IS NULL AND %s IS NULL))\n            ORDER BY effective_at DESC, created_at DESC\n            '),
            (organization_id, workspace_id, workspace_id),
        ).fetchall()
        return [
            {
                **dict(row),
                "included_activation_ids": _loads_json(row["included_activation_ids_json"]),
                "excluded_activation_evidence": _loads_json(
                    row["excluded_activation_evidence_json"]
                ),
            }
            for row in rows
        ]

    def list_rules_for_activation_snapshot(
        self,
        *,
        activation_set_snapshot_id: str,
        resolved_activation_set_hash: str,
        organization_id: str,
        workspace_id: str | None,
        requested_rule_families: list[str],
        effective_at: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        snapshot_id = _assert_ascii(activation_set_snapshot_id, "activation_set_snapshot_id")
        snapshot = self._conn.execute(
            sql_query("SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = ?", 'SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = %s'),
            (snapshot_id,),
        ).fetchone()
        if snapshot is None:
            raise ValueError(f"activation set snapshot not found: {snapshot_id}")
        if snapshot["organization_id"] != organization_id:
            raise ValueError("activation set snapshot organization mismatch")
        if (snapshot["workspace_id"] or None) != (workspace_id or None):
            raise ValueError("activation set snapshot workspace mismatch")
        snapshot_payload = {
            "organization_id": snapshot["organization_id"],
            "workspace_id": snapshot["workspace_id"],
            "principal_entitlement_ref": snapshot["principal_entitlement_ref"],
            "activation_resolver_policy_version": snapshot[
                "activation_resolver_policy_version"
            ],
            "effective_at": snapshot["effective_at"],
            "included_activation_ids": json.loads(snapshot["included_activation_ids_json"]),
            "excluded_activation_evidence": json.loads(
                snapshot["excluded_activation_evidence_json"]
            ),
            "prior_snapshot_hash": snapshot["prior_snapshot_hash"],
        }
        recomputed_snapshot_hash = compute_content_hash(snapshot_payload)
        if (
            snapshot["resolved_activation_set_hash"] != resolved_activation_set_hash
            or recomputed_snapshot_hash != resolved_activation_set_hash
        ):
            raise ValueError("activation set snapshot hash mismatch")
        activation_ids = json.loads(snapshot["included_activation_ids_json"])
        executable_activations = self._activation_store.resolve_executable_activations(
            activation_ids=activation_ids,
            organization_id=organization_id,
            workspace_id=workspace_id,
            effective_at=effective_at or snapshot["effective_at"],
        )
        activation_ids_by_rule_version: dict[str, list[str]] = {}
        for activation in executable_activations:
            activation_ids_by_rule_version.setdefault(str(activation["rule_version_id"]), []).append(
                str(activation["activation_id"])
            )
        rule_version_ids = [activation["rule_version_id"] for activation in executable_activations]
        if not rule_version_ids:
            return [], compute_content_hash([])
        placeholders = ", ".join([sql_query("?", '%s')] * len(rule_version_ids))
        rows = self._conn.execute(
            f"""
            SELECT * FROM governed_rule_versions
            WHERE rule_version_id IN ({placeholders})
            ORDER BY rule_version_id
            """,
            rule_version_ids,
        ).fetchall()
        requested = {_assert_ascii(item, "requested_rule_family") for item in requested_rule_families}
        rules = []
        for row in rows:
            rule = _row_to_dict(row)
            if rule is None:
                continue
            if requested and rule["rule_family"] not in requested:
                continue
            if rule.get("organization_id") != organization_id:
                raise ValueError("governed rule version organization mismatch")
            rule["governed_rule_activation_ids"] = sorted(
                activation_ids_by_rule_version.get(str(rule["rule_version_id"]), [])
            )
            rules.append(rule)
        version_hash_payload = [
            {
                "rule_version_id": rule["rule_version_id"],
                "content_hash": rule["content_hash"],
            }
            for rule in rules
        ]
        return rules, compute_content_hash(version_hash_payload)
