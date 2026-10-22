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
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

SourceKind = Literal["legal", "wiki", "research", "dataset", "controlled_document"]
SemanticPurpose = Literal["sensitive_lexicon"]
SemanticTier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
AuthorityRole = Literal["binding", "interpretive", "empirical", "advisory", "descriptive"]
SourceLifecycleStatus = Literal["proposed", "validated", "approved", "rejected", "deprecated"]
DatasetType = Literal["DEMO", "REAL"]
DatasetUsageRole = Literal["primary_analytics", "benchmark", "reference", "sandbox"]
DatasetAuthorityLevel = Literal["binding", "governed_default", "advisory", "sandbox"]
ContextControlLevel = Literal["none", "query_context", "prompt_context", "full_optional_context"]
ReadinessStatus = Literal["ready", "stale", "blocked", "deprecated"]
ActivationStatus = Literal["active", "inactive", "blocked", "superseded", "deprecated"]
WorkspaceOverridePolicy = Literal[
    "none",
    "retrieval_only",
    "prompt_context_only",
    "query_context_only",
    "optional_context",
    "governed_replacement",
]
ActivationDecision = Literal[
    "ACTIVATE_AUTHORITY",
    "DEACTIVATE_AUTHORITY",
    "SUPERSEDE",
    "BLOCK",
    "UNBLOCK",
    "DEPRECATE",
    "MIGRATE",
]
PreferenceScope = Literal["principal", "role", "workspace_default"]
PreferenceDecision = Literal[
    "ENABLE_RETRIEVAL",
    "DISABLE_RETRIEVAL",
    "ENABLE_PROMPT_CONTEXT",
    "DISABLE_PROMPT_CONTEXT",
    "ENABLE_QUERY_CONTEXT",
    "DISABLE_QUERY_CONTEXT",
    "RESET_TO_DEFAULT",
]

_DATASET_COMPATIBILITY: dict[
    DatasetUsageRole, tuple[tuple[set[DatasetAuthorityLevel], set[ContextControlLevel]], ...]
] = {
    "primary_analytics": (
        ({"binding", "governed_default"}, {"none", "query_context"}),
        ({"advisory"}, {"query_context", "full_optional_context"}),
    ),
    "benchmark": (
        ({"advisory", "sandbox"}, {"query_context", "full_optional_context"}),
    ),
    "reference": (
        ({"advisory", "sandbox"}, {"prompt_context", "query_context", "full_optional_context"}),
    ),
    "sandbox": (
        ({"sandbox"}, {"query_context", "full_optional_context"}),
    ),
}


def _db_path() -> Path:
    override = os.environ.get("SEMANTIER_EOS_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return sqlite_db_path("eos.db", ".semantier/eos.db").resolve()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _assert_ascii_identifier(value: str | None, field: str, *, required: bool = True) -> str | None:
    normalized = (value or "").strip()
    if not normalized:
        if required:
            raise ValueError(f"{field} is required")
        return None
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _assert_utc_timestamp(value: str | None, field: str, *, required: bool = False) -> str | None:
    normalized = (value or "").strip()
    if not normalized:
        if required:
            raise ValueError(f"{field} is required")
        return None
    parsed_text = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(parsed_text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware UTC ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be a timezone-aware UTC ISO-8601 timestamp")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _assert_dataset_compatibility(
    *,
    dataset_usage_role: DatasetUsageRole,
    authority_level: DatasetAuthorityLevel,
    user_context_control_level: ContextControlLevel,
) -> None:
    combinations = _DATASET_COMPATIBILITY.get(dataset_usage_role, ())
    if any(
        authority_level in allowed_authority and user_context_control_level in allowed_control
        for allowed_authority, allowed_control in combinations
    ):
        return
    raise ValueError(
        "dataset_usage_role, authority_level, and user_context_control_level are incompatible"
    )


def _allowed_dimensions(control_level: str) -> set[str]:
    if control_level == "none":
        return set()
    if control_level == "prompt_context":
        return {"retrieval", "prompt_context"}
    if control_level == "query_context":
        return {"query_context"}
    if control_level == "full_optional_context":
        return {"retrieval", "prompt_context", "query_context"}
    raise ValueError(f"unknown user_context_control_level: {control_level}")


class KnowledgeSourceActivationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'dataset_asset_version','knowledge_activation_migration_state','knowledge_activation_set_snapshot','knowledge_context_inclusion_preference','knowledge_context_inclusion_preference_events','knowledge_source','knowledge_source_activation','knowledge_source_activation_events')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            init_db(path)
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    def close(self) -> None:
        self._conn.close()

    def register_knowledge_source(
        self,
        *,
        knowledge_source_id: str,
        source_kind: SourceKind,
        semantic_tier: SemanticTier,
        authority_role: AuthorityRole,
        source_lifecycle_status: SourceLifecycleStatus,
        source_ref: str,
        source_hash: str,
        semantic_purpose: str | None = None,
        source_version: str | None = None,
        effective_from: str | None = None,
        effective_to: str | None = None,
        provenance_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_id = _assert_ascii_identifier(knowledge_source_id, "knowledge_source_id")
        source_kind = _assert_ascii_identifier(source_kind, "source_kind")  # type: ignore[assignment]
        semantic_purpose = _assert_ascii_identifier(
            semantic_purpose,
            "semantic_purpose",
            required=False,
        )
        if source_kind == "controlled_document" and not semantic_purpose:
            raise ValueError("controlled_document knowledge sources require semantic_purpose")
        if semantic_purpose and source_kind != "controlled_document":
            raise ValueError("semantic_purpose is supported only for controlled_document sources")
        if semantic_purpose and semantic_purpose != "sensitive_lexicon":
            raise ValueError("unknown semantic_purpose")
        source_version = _assert_ascii_identifier(source_version, "source_version", required=False)
        effective_from = _assert_utc_timestamp(effective_from, "effective_from")
        effective_to = _assert_utc_timestamp(effective_to, "effective_to")
        created_at = _now_iso()
        payload = {
            "knowledge_source_id": source_id,
            "source_kind": source_kind,
            "semantic_purpose": semantic_purpose,
            "semantic_tier": semantic_tier,
            "authority_role": authority_role,
            "source_lifecycle_status": source_lifecycle_status,
            "source_ref": source_ref,
            "source_version": source_version,
            "source_hash": source_hash,
            "effective_from": effective_from,
            "effective_to": effective_to,
            "provenance_manifest": provenance_manifest or {},
            "created_at": created_at,
        }
        row_hash = compute_content_hash(payload)
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source (
                  knowledge_source_id, source_kind, semantic_purpose, semantic_tier, authority_role,
                  source_lifecycle_status, source_ref, source_version, source_hash,
                  effective_from, effective_to, provenance_manifest_json, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source (\n                  knowledge_source_id, source_kind, semantic_purpose, semantic_tier, authority_role,\n                  source_lifecycle_status, source_ref, source_version, source_hash,\n                  effective_from, effective_to, provenance_manifest_json, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    source_id,
                    source_kind,
                    semantic_purpose,
                    semantic_tier,
                    authority_role,
                    source_lifecycle_status,
                    source_ref,
                    source_version,
                    source_hash,
                    effective_from,
                    effective_to,
                    _as_json(provenance_manifest or {}),
                    created_at,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM knowledge_source WHERE knowledge_source_id = ?", 'SELECT * FROM knowledge_source WHERE knowledge_source_id = %s'),
                (source_id,),
            ).fetchone())

    def resolve_controlled_document_compiler_route(
        self,
        knowledge_source_id: str,
    ) -> str:
        source_id = _assert_ascii_identifier(knowledge_source_id, "knowledge_source_id")
        row = self._conn.execute(
            sql_query("SELECT * FROM knowledge_source WHERE knowledge_source_id = ?", 'SELECT * FROM knowledge_source WHERE knowledge_source_id = %s'),
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"knowledge_source not found: {source_id}")
        if row["source_kind"] != "controlled_document":
            raise ValueError("CONTROLLED_DOCUMENT_SOURCE_REQUIRED")
        semantic_purpose = str(row["semantic_purpose"] or "").strip()
        if semantic_purpose == "sensitive_lexicon":
            return "sensitive_lexicon_docx.v1"
        raise ValueError("UNKNOWN_SEMANTIC_PURPOSE")

    def register_dataset_asset_version(
        self,
        *,
        dataset_asset_version_id: str,
        dataset_type: DatasetType,
        dataset_usage_role: DatasetUsageRole,
        authority_level: DatasetAuthorityLevel,
        user_context_control_level: ContextControlLevel,
        schema_contract_hash: str,
        lineage_manifest_hash: str,
        readiness_status: ReadinessStatus,
        quality_evidence_hash: str,
        dataset_key: str | None = None,
        dataset_version_id: str | None = None,
    ) -> dict[str, Any]:
        asset_id = _assert_ascii_identifier(dataset_asset_version_id, "dataset_asset_version_id")
        dataset_key = _assert_ascii_identifier(dataset_key, "dataset_key", required=False)
        dataset_version_id = _assert_ascii_identifier(dataset_version_id, "dataset_version_id", required=False)
        if dataset_type == "DEMO" and not dataset_key:
            raise ValueError("DEMO dataset assets require dataset_key")
        if dataset_type == "REAL" and not dataset_version_id:
            raise ValueError("REAL dataset assets require dataset_version_id")
        _assert_dataset_compatibility(
            dataset_usage_role=dataset_usage_role,
            authority_level=authority_level,
            user_context_control_level=user_context_control_level,
        )
        created_at = _now_iso()
        payload = {
            "dataset_asset_version_id": asset_id,
            "dataset_type": dataset_type,
            "dataset_key": dataset_key,
            "dataset_version_id": dataset_version_id,
            "dataset_usage_role": dataset_usage_role,
            "authority_level": authority_level,
            "user_context_control_level": user_context_control_level,
            "schema_contract_hash": schema_contract_hash,
            "lineage_manifest_hash": lineage_manifest_hash,
            "readiness_status": readiness_status,
            "quality_evidence_hash": quality_evidence_hash,
            "created_at": created_at,
        }
        row_hash = compute_content_hash(payload)
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT INTO dataset_asset_version (
                  dataset_asset_version_id, dataset_type, dataset_key, dataset_version_id,
                  dataset_usage_role, authority_level, user_context_control_level,
                  schema_contract_hash, lineage_manifest_hash, readiness_status,
                  quality_evidence_hash, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO dataset_asset_version (\n                  dataset_asset_version_id, dataset_type, dataset_key, dataset_version_id,\n                  dataset_usage_role, authority_level, user_context_control_level,\n                  schema_contract_hash, lineage_manifest_hash, readiness_status,\n                  quality_evidence_hash, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    asset_id,
                    dataset_type,
                    dataset_key,
                    dataset_version_id,
                    dataset_usage_role,
                    authority_level,
                    user_context_control_level,
                    schema_contract_hash,
                    lineage_manifest_hash,
                    readiness_status,
                    quality_evidence_hash,
                    created_at,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM dataset_asset_version WHERE dataset_asset_version_id = ?", 'SELECT * FROM dataset_asset_version WHERE dataset_asset_version_id = %s'),
                (asset_id,),
            ).fetchone())

    def activate_source(
        self,
        *,
        activation_id: str,
        knowledge_source_id: str,
        organization_id: str,
        activation_scope: Literal["organization", "workspace"],
        activation_status: ActivationStatus,
        workspace_override_policy: WorkspaceOverridePolicy,
        user_context_control_level: ContextControlLevel,
        decision: ActivationDecision,
        actor: str,
        actor_authority: str,
        reason: str,
        artifact_version_hash: str,
        dataset_asset_version_id: str | None = None,
        workspace_id: str | None = None,
        authority_source_manifest: dict[str, Any] | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        activation_id = _assert_ascii_identifier(activation_id, "activation_id")
        source_id = _assert_ascii_identifier(knowledge_source_id, "knowledge_source_id")
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        dataset_asset_version_id = _assert_ascii_identifier(
            dataset_asset_version_id,
            "dataset_asset_version_id",
            required=False,
        )
        if activation_scope == "workspace" and not workspace_id:
            raise ValueError("workspace activation requires workspace_id")
        scope_key = workspace_id if activation_scope == "workspace" else "organization"
        occurred_at = _assert_utc_timestamp(occurred_at, "occurred_at") or _now_iso()
        event_id = f"ksae_{uuid4().hex}"
        with self._lock, transaction_scope(self._conn):
            source = self._conn.execute(
                sql_query("SELECT * FROM knowledge_source WHERE knowledge_source_id = ?", 'SELECT * FROM knowledge_source WHERE knowledge_source_id = %s'),
                (source_id,),
            ).fetchone()
            if source is None:
                raise ValueError("knowledge_source_id does not exist")
            dataset_asset = None
            if str(source["source_kind"]) == "dataset":
                if not dataset_asset_version_id:
                    raise ValueError("dataset sources require dataset_asset_version_id")
                dataset_asset = self._conn.execute(
                    sql_query("SELECT * FROM dataset_asset_version WHERE dataset_asset_version_id = ?", 'SELECT * FROM dataset_asset_version WHERE dataset_asset_version_id = %s'),
                    (dataset_asset_version_id,),
                ).fetchone()
                if dataset_asset is None:
                    raise ValueError("dataset_asset_version_id does not exist")
                if user_context_control_level != dataset_asset["user_context_control_level"]:
                    raise ValueError("activation user_context_control_level must match dataset asset")
                if activation_status == "active":
                    duplicate = self._conn.execute(
                        sql_query("""
                        SELECT ksa.activation_id
                        FROM knowledge_source_activation ksa
                        JOIN dataset_asset_version dav
                          ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
                        WHERE ksa.organization_id = ?
                          AND ksa.scope_key = ?
                          AND ksa.activation_status = 'active'
                          AND dav.dataset_usage_role = ?
                        LIMIT 1
                        """, "\n                        SELECT ksa.activation_id\n                        FROM knowledge_source_activation ksa\n                        JOIN dataset_asset_version dav\n                          ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id\n                        WHERE ksa.organization_id = %s\n                          AND ksa.scope_key = %s\n                          AND ksa.activation_status = 'active'\n                          AND dav.dataset_usage_role = %s\n                        LIMIT 1\n                        "),
                        (organization_id, scope_key, dataset_asset["dataset_usage_role"]),
                    ).fetchone()
                    if duplicate:
                        raise ValueError("duplicate active dataset binding for dataset_usage_role")
            elif dataset_asset_version_id:
                raise ValueError("only dataset sources may link dataset_asset_version_id")

            prior_event_hash = self._conn.execute(
                sql_query("""
                SELECT content_hash
                FROM knowledge_source_activation_events
                WHERE knowledge_source_id = ?
                  AND organization_id = ?
                  AND COALESCE(workspace_id, '') = COALESCE(?, '')
                ORDER BY occurred_at DESC
                LIMIT 1
                """, "\n                SELECT content_hash\n                FROM knowledge_source_activation_events\n                WHERE knowledge_source_id = %s\n                  AND organization_id = %s\n                  AND COALESCE(workspace_id, '') = COALESCE(%s, '')\n                ORDER BY occurred_at DESC\n                LIMIT 1\n                "),
                (source_id, organization_id, workspace_id),
            ).fetchone()
            event_payload = {
                "activation_event_id": event_id,
                "knowledge_source_id": source_id,
                "dataset_asset_version_id": dataset_asset_version_id,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "prior_activation_event_hash": prior_event_hash["content_hash"] if prior_event_hash else None,
                "decision": decision,
                "actor": actor,
                "actor_authority": actor_authority,
                "reason": reason,
                "artifact_version_hash": artifact_version_hash,
                "occurred_at": occurred_at,
            }
            event_hash = compute_content_hash(event_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_activation_events (
                  activation_event_id, knowledge_source_id, dataset_asset_version_id,
                  organization_id, workspace_id, prior_activation_event_hash, decision,
                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source_activation_events (\n                  activation_event_id, knowledge_source_id, dataset_asset_version_id,\n                  organization_id, workspace_id, prior_activation_event_hash, decision,\n                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event_id,
                    source_id,
                    dataset_asset_version_id,
                    organization_id,
                    workspace_id,
                    event_payload["prior_activation_event_hash"],
                    decision,
                    actor,
                    actor_authority,
                    reason,
                    artifact_version_hash,
                    occurred_at,
                    event_hash,
                ),
            )
            created_at = _now_iso()
            activation_payload = {
                "activation_id": activation_id,
                "knowledge_source_id": source_id,
                "dataset_asset_version_id": dataset_asset_version_id,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "activation_scope": activation_scope,
                "scope_key": scope_key,
                "activation_status": activation_status,
                "workspace_override_policy": workspace_override_policy,
                "user_context_control_level": user_context_control_level,
                "activation_event_head_hash": event_hash,
                "authority_source_manifest": authority_source_manifest or {},
                "created_at": created_at,
            }
            row_hash = compute_content_hash(activation_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_activation (
                  activation_id, knowledge_source_id, dataset_asset_version_id,
                  organization_id, workspace_id, activation_scope, scope_key,
                  activation_status, workspace_override_policy, user_context_control_level,
                  activation_event_head_hash, authority_source_manifest_json, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source_activation (\n                  activation_id, knowledge_source_id, dataset_asset_version_id,\n                  organization_id, workspace_id, activation_scope, scope_key,\n                  activation_status, workspace_override_policy, user_context_control_level,\n                  activation_event_head_hash, authority_source_manifest_json, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    activation_id,
                    source_id,
                    dataset_asset_version_id,
                    organization_id,
                    workspace_id,
                    activation_scope,
                    scope_key,
                    activation_status,
                    workspace_override_policy,
                    user_context_control_level,
                    event_hash,
                    _as_json(authority_source_manifest or {}),
                    created_at,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM knowledge_source_activation WHERE activation_id = ?", 'SELECT * FROM knowledge_source_activation WHERE activation_id = %s'),
                (activation_id,),
            ).fetchone())

    def write_preference(
        self,
        *,
        preference_id: str,
        activation_id: str,
        organization_id: str,
        preference_scope: PreferenceScope,
        retrieval_enabled: bool,
        prompt_context_enabled: bool,
        query_context_enabled: bool,
        decision: PreferenceDecision,
        actor: str,
        reason: str,
        principal_id: str | None = None,
        role_id: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        preference_id = _assert_ascii_identifier(preference_id, "preference_id")
        activation_id = _assert_ascii_identifier(activation_id, "activation_id")
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        principal_id = _assert_ascii_identifier(principal_id, "principal_id", required=False)
        role_id = _assert_ascii_identifier(role_id, "role_id", required=False)
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        if preference_scope == "principal" and (not principal_id or role_id):
            raise ValueError("principal preference requires exactly principal_id")
        if preference_scope == "role" and (not role_id or principal_id):
            raise ValueError("role preference requires exactly role_id")
        if preference_scope == "workspace_default" and (principal_id or role_id):
            raise ValueError("workspace_default preference must not set principal_id or role_id")
        occurred_at = _assert_utc_timestamp(occurred_at, "occurred_at") or _now_iso()
        with self._lock, transaction_scope(self._conn):
            activation = self._conn.execute(
                sql_query("SELECT * FROM knowledge_source_activation WHERE activation_id = ?", 'SELECT * FROM knowledge_source_activation WHERE activation_id = %s'),
                (activation_id,),
            ).fetchone()
            if activation is None:
                raise ValueError("activation_id does not exist")
            if activation["organization_id"] != organization_id:
                raise ValueError("preference organization_id must match activation")
            allowed = _allowed_dimensions(str(activation["user_context_control_level"]))
            if not allowed:
                raise ValueError(
                    "preference is not permitted for activation user_context_control_level=none"
                )
            requested_enabled = {
                "retrieval": retrieval_enabled,
                "prompt_context": prompt_context_enabled,
                "query_context": query_context_enabled,
            }
            forbidden_enabled = [
                dimension
                for dimension, enabled in requested_enabled.items()
                if enabled and dimension not in allowed
            ]
            if forbidden_enabled:
                raise ValueError(
                    "preference enables dimensions outside activation user_context_control_level: "
                    + ",".join(sorted(forbidden_enabled))
                )

            prior_event_hash = self._conn.execute(
                sql_query("""
                SELECT preference_event_head_hash
                FROM knowledge_context_inclusion_preference
                WHERE preference_id = ?
                """, '\n                SELECT preference_event_head_hash\n                FROM knowledge_context_inclusion_preference\n                WHERE preference_id = %s\n                '),
                (preference_id,),
            ).fetchone()
            event_id = f"kcipe_{uuid4().hex}"
            event_payload = {
                "preference_event_id": event_id,
                "preference_id": preference_id,
                "activation_id": activation_id,
                "principal_id": principal_id,
                "role_id": role_id,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "prior_preference_event_hash": prior_event_hash["preference_event_head_hash"] if prior_event_hash else None,
                "decision": decision,
                "actor": actor,
                "reason": reason,
                "occurred_at": occurred_at,
            }
            event_hash = compute_content_hash(event_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_context_inclusion_preference_events (
                  preference_event_id, preference_id, activation_id, principal_id, role_id,
                  organization_id, workspace_id, prior_preference_event_hash, decision,
                  actor, reason, occurred_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_context_inclusion_preference_events (\n                  preference_event_id, preference_id, activation_id, principal_id, role_id,\n                  organization_id, workspace_id, prior_preference_event_hash, decision,\n                  actor, reason, occurred_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event_id,
                    preference_id,
                    activation_id,
                    principal_id,
                    role_id,
                    organization_id,
                    workspace_id,
                    event_payload["prior_preference_event_hash"],
                    decision,
                    actor,
                    reason,
                    occurred_at,
                    event_hash,
                ),
            )
            updated_at = _now_iso()
            pref_payload = {
                "preference_id": preference_id,
                "activation_id": activation_id,
                "principal_id": principal_id,
                "role_id": role_id,
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "preference_scope": preference_scope,
                "retrieval_enabled": bool(retrieval_enabled),
                "prompt_context_enabled": bool(prompt_context_enabled),
                "query_context_enabled": bool(query_context_enabled),
                "preference_event_head_hash": event_hash,
                "updated_at": updated_at,
            }
            row_hash = compute_content_hash(pref_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_context_inclusion_preference (
                  preference_id, activation_id, principal_id, role_id, organization_id,
                  workspace_id, preference_scope, retrieval_enabled, prompt_context_enabled,
                  query_context_enabled, preference_event_head_hash, updated_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(preference_id) DO UPDATE SET
                  retrieval_enabled = excluded.retrieval_enabled,
                  prompt_context_enabled = excluded.prompt_context_enabled,
                  query_context_enabled = excluded.query_context_enabled,
                  preference_event_head_hash = excluded.preference_event_head_hash,
                  updated_at = excluded.updated_at,
                  content_hash = excluded.content_hash
                """, '\n                INSERT INTO knowledge_context_inclusion_preference (\n                  preference_id, activation_id, principal_id, role_id, organization_id,\n                  workspace_id, preference_scope, retrieval_enabled, prompt_context_enabled,\n                  query_context_enabled, preference_event_head_hash, updated_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                ON CONFLICT(preference_id) DO UPDATE SET\n                  retrieval_enabled = excluded.retrieval_enabled,\n                  prompt_context_enabled = excluded.prompt_context_enabled,\n                  query_context_enabled = excluded.query_context_enabled,\n                  preference_event_head_hash = excluded.preference_event_head_hash,\n                  updated_at = excluded.updated_at,\n                  content_hash = excluded.content_hash\n                '),
                (
                    preference_id,
                    activation_id,
                    principal_id,
                    role_id,
                    organization_id,
                    workspace_id,
                    preference_scope,
                    int(bool(retrieval_enabled)),
                    int(bool(prompt_context_enabled)),
                    int(bool(query_context_enabled)),
                    event_hash,
                    updated_at,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM knowledge_context_inclusion_preference WHERE preference_id = ?", 'SELECT * FROM knowledge_context_inclusion_preference WHERE preference_id = %s'),
                (preference_id,),
            ).fetchone())

    def materialize_activation_set_snapshot(
        self,
        *,
        activation_set_snapshot_id: str,
        organization_id: str,
        principal_entitlement_ref: str,
        activation_resolver_policy_version: str,
        effective_at: str,
        included_activation_ids: list[str],
        excluded_activation_evidence: list[dict[str, Any]] | None = None,
        workspace_id: str | None = None,
        prior_snapshot_hash: str | None = None,
    ) -> dict[str, Any]:
        snapshot_id = _assert_ascii_identifier(
            activation_set_snapshot_id,
            "activation_set_snapshot_id",
        )
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        effective_at = _assert_utc_timestamp(effective_at, "effective_at", required=True)
        included_activation_ids = [
            _assert_ascii_identifier(item, "included_activation_id") or ""
            for item in included_activation_ids
        ]
        created_at = _now_iso()
        resolver_payload = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "principal_entitlement_ref": principal_entitlement_ref,
            "activation_resolver_policy_version": activation_resolver_policy_version,
            "effective_at": effective_at,
            "included_activation_ids": sorted(included_activation_ids),
            "excluded_activation_evidence": excluded_activation_evidence or [],
            "prior_snapshot_hash": prior_snapshot_hash,
        }
        resolved_hash = compute_content_hash(resolver_payload)
        row_payload = {
            "activation_set_snapshot_id": snapshot_id,
            **resolver_payload,
            "resolved_activation_set_hash": resolved_hash,
            "created_at": created_at,
        }
        row_hash = compute_content_hash(row_payload)
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_activation_set_snapshot (
                  activation_set_snapshot_id, organization_id, workspace_id,
                  principal_entitlement_ref, activation_resolver_policy_version,
                  effective_at, included_activation_ids_json,
                  excluded_activation_evidence_json, resolved_activation_set_hash,
                  prior_snapshot_hash, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_activation_set_snapshot (\n                  activation_set_snapshot_id, organization_id, workspace_id,\n                  principal_entitlement_ref, activation_resolver_policy_version,\n                  effective_at, included_activation_ids_json,\n                  excluded_activation_evidence_json, resolved_activation_set_hash,\n                  prior_snapshot_hash, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    snapshot_id,
                    organization_id,
                    workspace_id,
                    principal_entitlement_ref,
                    activation_resolver_policy_version,
                    effective_at,
                    canonical_json(sorted(included_activation_ids)),
                    canonical_json(excluded_activation_evidence or []),
                    resolved_hash,
                    prior_snapshot_hash,
                    created_at,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = ?", 'SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = %s'),
                (snapshot_id,),
            ).fetchone())

    def record_migration_state(
        self,
        *,
        organization_id: str,
        migration_status: Literal["pending", "activation_written", "complete", "failed"],
        source_legacy_fingerprint: str,
        started_at: str,
        activation_id: str | None = None,
        migration_event_ref: str | None = None,
        completed_at: str | None = None,
        last_error_code: str | None = None,
    ) -> dict[str, Any]:
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        activation_id = _assert_ascii_identifier(activation_id, "activation_id", required=False)
        started_at = _assert_utc_timestamp(started_at, "started_at", required=True)
        completed_at = _assert_utc_timestamp(completed_at, "completed_at")
        payload = {
            "organization_id": organization_id,
            "migration_status": migration_status,
            "source_legacy_fingerprint": source_legacy_fingerprint,
            "activation_id": activation_id,
            "migration_event_ref": migration_event_ref,
            "started_at": started_at,
            "completed_at": completed_at,
            "last_error_code": last_error_code,
        }
        row_hash = compute_content_hash(payload)
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_activation_migration_state (
                  organization_id, migration_status, source_legacy_fingerprint,
                  activation_id, migration_event_ref, started_at, completed_at,
                  last_error_code, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organization_id) DO UPDATE SET
                  migration_status = excluded.migration_status,
                  source_legacy_fingerprint = excluded.source_legacy_fingerprint,
                  activation_id = excluded.activation_id,
                  migration_event_ref = excluded.migration_event_ref,
                  started_at = excluded.started_at,
                  completed_at = excluded.completed_at,
                  last_error_code = excluded.last_error_code,
                  content_hash = excluded.content_hash
                """, '\n                INSERT INTO knowledge_activation_migration_state (\n                  organization_id, migration_status, source_legacy_fingerprint,\n                  activation_id, migration_event_ref, started_at, completed_at,\n                  last_error_code, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                ON CONFLICT(organization_id) DO UPDATE SET\n                  migration_status = excluded.migration_status,\n                  source_legacy_fingerprint = excluded.source_legacy_fingerprint,\n                  activation_id = excluded.activation_id,\n                  migration_event_ref = excluded.migration_event_ref,\n                  started_at = excluded.started_at,\n                  completed_at = excluded.completed_at,\n                  last_error_code = excluded.last_error_code,\n                  content_hash = excluded.content_hash\n                '),
                (
                    organization_id,
                    migration_status,
                    source_legacy_fingerprint,
                    activation_id,
                    migration_event_ref,
                    started_at,
                    completed_at,
                    last_error_code,
                    row_hash,
                ),
            )
            return dict(self._conn.execute(
                sql_query("SELECT * FROM knowledge_activation_migration_state WHERE organization_id = ?", 'SELECT * FROM knowledge_activation_migration_state WHERE organization_id = %s'),
                (organization_id,),
            ).fetchone())

    def get_migration_state(self, organization_id: str) -> dict[str, Any] | None:
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        with self._lock:
            return _row_dict(
                self._conn.execute(
                    sql_query("SELECT * FROM knowledge_activation_migration_state WHERE organization_id = ?", 'SELECT * FROM knowledge_activation_migration_state WHERE organization_id = %s'),
                    (organization_id,),
                ).fetchone()
            )

    def get_active_dataset_binding(
        self,
        *,
        organization_id: str,
        dataset_usage_role: str = "primary_analytics",
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        scope_keys = [workspace_id, "organization"] if workspace_id else ["organization"]
        with self._lock:
            for scope_key in scope_keys:
                if not scope_key:
                    continue
                rows = self._conn.execute(
                    sql_query("""
                    SELECT
                      ksa.*,
                      ks.source_kind,
                      ks.semantic_tier,
                      ks.authority_role,
                      dav.dataset_type,
                      dav.dataset_key,
                      dav.dataset_version_id,
                      dav.dataset_usage_role,
                      dav.authority_level,
                      dav.readiness_status,
                      dav.schema_contract_hash,
                      dav.lineage_manifest_hash,
                      dav.quality_evidence_hash
                    FROM knowledge_source_activation ksa
                    JOIN knowledge_source ks
                      ON ks.knowledge_source_id = ksa.knowledge_source_id
                    JOIN dataset_asset_version dav
                      ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
                    WHERE ksa.organization_id = ?
                      AND ksa.scope_key = ?
                      AND ksa.activation_status = 'active'
                      AND dav.dataset_usage_role = ?
                    ORDER BY ksa.created_at DESC
                    """, "\n                    SELECT\n                      ksa.*,\n                      ks.source_kind,\n                      ks.semantic_tier,\n                      ks.authority_role,\n                      dav.dataset_type,\n                      dav.dataset_key,\n                      dav.dataset_version_id,\n                      dav.dataset_usage_role,\n                      dav.authority_level,\n                      dav.readiness_status,\n                      dav.schema_contract_hash,\n                      dav.lineage_manifest_hash,\n                      dav.quality_evidence_hash\n                    FROM knowledge_source_activation ksa\n                    JOIN knowledge_source ks\n                      ON ks.knowledge_source_id = ksa.knowledge_source_id\n                    JOIN dataset_asset_version dav\n                      ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id\n                    WHERE ksa.organization_id = %s\n                      AND ksa.scope_key = %s\n                      AND ksa.activation_status = 'active'\n                      AND dav.dataset_usage_role = %s\n                    ORDER BY ksa.created_at DESC\n                    "),
                    (organization_id, scope_key, dataset_usage_role),
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError("ambiguous active dataset activation")
                if rows:
                    return dict(rows[0])
            return None

    def list_active_bindings(
        self,
        *,
        organization_id: str,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        scope_keys = ["organization"]
        if workspace_id:
            scope_keys.append(workspace_id)
        placeholders = ",".join(sql_query("?", '%s') for _ in scope_keys)
        with self._lock:
            rows = self._conn.execute(
                sql_query(f"""
                SELECT
                  ksa.*,
                  ks.source_kind,
                  ks.semantic_tier,
                  ks.authority_role,
                  ks.source_lifecycle_status,
                  ks.source_ref,
                  ks.source_version,
                  ks.source_hash,
                  ks.source_ref,
                  ks.provenance_manifest_json,
                  dav.dataset_type,
                  dav.dataset_key,
                  dav.dataset_version_id,
                  dav.dataset_usage_role,
                  dav.authority_level AS dataset_authority_level,
                  dav.readiness_status AS dataset_readiness_status,
                  dav.schema_contract_hash,
                  dav.lineage_manifest_hash,
                  dav.quality_evidence_hash
                FROM knowledge_source_activation ksa
                JOIN knowledge_source ks
                  ON ks.knowledge_source_id = ksa.knowledge_source_id
                LEFT JOIN dataset_asset_version dav
                  ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
                WHERE ksa.organization_id = ?
                  AND ksa.scope_key IN ({placeholders})
                  AND ksa.activation_status = 'active'
                ORDER BY ksa.scope_key, ksa.knowledge_source_id, ksa.created_at
                """, f"""
                SELECT
                  ksa.*,
                  ks.source_kind,
                  ks.semantic_tier,
                  ks.authority_role,
                  ks.source_lifecycle_status,
                  ks.source_ref,
                  ks.source_version,
                  ks.source_hash,
                  ks.source_ref,
                  ks.provenance_manifest_json,
                  dav.dataset_type,
                  dav.dataset_key,
                  dav.dataset_version_id,
                  dav.dataset_usage_role,
                  dav.authority_level AS dataset_authority_level,
                  dav.readiness_status AS dataset_readiness_status,
                  dav.schema_contract_hash,
                  dav.lineage_manifest_hash,
                  dav.quality_evidence_hash
                FROM knowledge_source_activation ksa
                JOIN knowledge_source ks
                  ON ks.knowledge_source_id = ksa.knowledge_source_id
                LEFT JOIN dataset_asset_version dav
                  ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
                WHERE ksa.organization_id = %s
                  AND ksa.scope_key IN ({placeholders})
                  AND ksa.activation_status = 'active'
                ORDER BY ksa.scope_key, ksa.knowledge_source_id, ksa.created_at
                """),
                (organization_id, *scope_keys),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_inclusion_preference(
        self,
        *,
        activation_id: str,
        organization_id: str,
        principal_id: str | None = None,
        role_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        activation_id = _assert_ascii_identifier(activation_id, "activation_id")
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        principal_id = _assert_ascii_identifier(principal_id, "principal_id", required=False)
        role_id = _assert_ascii_identifier(role_id, "role_id", required=False)
        workspace_id = _assert_ascii_identifier(workspace_id, "workspace_id", required=False)
        queries: list[tuple[str, tuple[Any, ...]]] = []
        if principal_id:
            queries.append(
                (
                    sql_query("""
                    SELECT * FROM knowledge_context_inclusion_preference
                    WHERE activation_id = ?
                      AND organization_id = ?
                      AND preference_scope = 'principal'
                      AND principal_id = ?
                      AND COALESCE(workspace_id, '') IN (COALESCE(?, ''), '')
                    ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(?, '') THEN 0 ELSE 1 END,
                             updated_at DESC
                    LIMIT 1
                    """, "\n                    SELECT * FROM knowledge_context_inclusion_preference\n                    WHERE activation_id = %s\n                      AND organization_id = %s\n                      AND preference_scope = 'principal'\n                      AND principal_id = %s\n                      AND COALESCE(workspace_id, '') IN (COALESCE(%s, ''), '')\n                    ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(%s, '') THEN 0 ELSE 1 END,\n                             updated_at DESC\n                    LIMIT 1\n                    "),
                    (activation_id, organization_id, principal_id, workspace_id, workspace_id),
                )
            )
        if role_id:
            queries.append(
                (
                    sql_query("""
                    SELECT * FROM knowledge_context_inclusion_preference
                    WHERE activation_id = ?
                      AND organization_id = ?
                      AND preference_scope = 'role'
                      AND role_id = ?
                      AND COALESCE(workspace_id, '') IN (COALESCE(?, ''), '')
                    ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(?, '') THEN 0 ELSE 1 END,
                             updated_at DESC
                    LIMIT 1
                    """, "\n                    SELECT * FROM knowledge_context_inclusion_preference\n                    WHERE activation_id = %s\n                      AND organization_id = %s\n                      AND preference_scope = 'role'\n                      AND role_id = %s\n                      AND COALESCE(workspace_id, '') IN (COALESCE(%s, ''), '')\n                    ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(%s, '') THEN 0 ELSE 1 END,\n                             updated_at DESC\n                    LIMIT 1\n                    "),
                    (activation_id, organization_id, role_id, workspace_id, workspace_id),
                )
            )
        queries.append(
            (
                sql_query("""
                SELECT * FROM knowledge_context_inclusion_preference
                WHERE activation_id = ?
                  AND organization_id = ?
                  AND preference_scope = 'workspace_default'
                  AND COALESCE(workspace_id, '') IN (COALESCE(?, ''), '')
                ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(?, '') THEN 0 ELSE 1 END,
                         updated_at DESC
                LIMIT 1
                """, "\n                SELECT * FROM knowledge_context_inclusion_preference\n                WHERE activation_id = %s\n                  AND organization_id = %s\n                  AND preference_scope = 'workspace_default'\n                  AND COALESCE(workspace_id, '') IN (COALESCE(%s, ''), '')\n                ORDER BY CASE WHEN COALESCE(workspace_id, '') = COALESCE(%s, '') THEN 0 ELSE 1 END,\n                         updated_at DESC\n                LIMIT 1\n                "),
                (activation_id, organization_id, workspace_id, workspace_id),
            )
        )
        with self._lock:
            for sql, params in queries:
                row = self._conn.execute(sql, params).fetchone()
                if row is not None:
                    return dict(row)
            return None

    def backfill_legacy_dataset_activation(
        self,
        *,
        organization_id: str,
        dataset_type: str | None,
        dataset_key: str | None = None,
        active_dataset_version_id: str | None = None,
        actor: str = "dataset_activation_migration",
    ) -> dict[str, Any] | None:
        organization_id = _assert_ascii_identifier(organization_id, "organization_id")
        normalized_dataset_type = (dataset_type or "").strip().upper()
        if normalized_dataset_type not in {"DEMO", "REAL"}:
            return None
        dataset_key = _assert_ascii_identifier(dataset_key, "dataset_key", required=False)
        active_dataset_version_id = _assert_ascii_identifier(
            active_dataset_version_id,
            "active_dataset_version_id",
            required=False,
        )
        if normalized_dataset_type == "DEMO" and not dataset_key:
            return None
        if normalized_dataset_type == "REAL" and not active_dataset_version_id:
            return None

        started_at = _now_iso()
        legacy_payload = {
            "organization_id": organization_id,
            "dataset_type": normalized_dataset_type,
            "dataset_key": dataset_key,
            "active_dataset_version_id": active_dataset_version_id,
        }
        legacy_fingerprint = compute_content_hash(legacy_payload)
        suffix = legacy_fingerprint[:16]
        source_id = f"ks_dataset_{organization_id}_{suffix}"
        asset_id = f"dav_{organization_id}_{suffix}"
        activation_id = f"ksa_{organization_id}_{suffix}"
        with self._lock, transaction_scope(self._conn):
            migration_state = self.get_migration_state(organization_id)
            if migration_state and migration_state.get("migration_status") in {
                "activation_written",
                "complete",
            }:
                return self.get_active_dataset_binding(organization_id=organization_id)
            existing = self.get_active_dataset_binding(organization_id=organization_id)
            if existing is not None:
                return existing

            source_hash = f"legacy:{legacy_fingerprint}"
            self._conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO knowledge_source (
                  knowledge_source_id, source_kind, semantic_tier, authority_role,
                  source_lifecycle_status, source_ref, source_version, source_hash,
                  effective_from, effective_to, provenance_manifest_json, created_at, content_hash
                )
                VALUES (?, 'dataset', 'T4', 'empirical', 'approved', ?, ?, ?, ?, NULL, ?, ?, ?)
                """, "\n                INSERT INTO knowledge_source (\n                  knowledge_source_id, source_kind, semantic_tier, authority_role,\n                  source_lifecycle_status, source_ref, source_version, source_hash,\n                  effective_from, effective_to, provenance_manifest_json, created_at, content_hash\n                )\n                VALUES (%s, 'dataset', 'T4', 'empirical', 'approved', %s, %s, %s, %s, NULL, %s, %s, %s)\n                 ON CONFLICT DO NOTHING"),
                (
                    source_id,
                    f"legacy_organization_dataset:{organization_id}",
                    active_dataset_version_id or dataset_key,
                    source_hash,
                    started_at,
                    canonical_json({"migration": "legacy_organization_dataset_fields"}),
                    started_at,
                    compute_content_hash(
                        {
                            "knowledge_source_id": source_id,
                            "legacy_fingerprint": legacy_fingerprint,
                            "created_at": started_at,
                        }
                    ),
                ),
            )
            self._conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO dataset_asset_version (
                  dataset_asset_version_id, dataset_type, dataset_key, dataset_version_id,
                  dataset_usage_role, authority_level, user_context_control_level,
                  schema_contract_hash, lineage_manifest_hash, readiness_status,
                  quality_evidence_hash, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, 'primary_analytics', 'governed_default', 'none', ?, ?, 'ready', ?, ?, ?)
                """, "\n                INSERT INTO dataset_asset_version (\n                  dataset_asset_version_id, dataset_type, dataset_key, dataset_version_id,\n                  dataset_usage_role, authority_level, user_context_control_level,\n                  schema_contract_hash, lineage_manifest_hash, readiness_status,\n                  quality_evidence_hash, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, 'primary_analytics', 'governed_default', 'none', %s, %s, 'ready', %s, %s, %s)\n                 ON CONFLICT DO NOTHING"),
                (
                    asset_id,
                    normalized_dataset_type,
                    dataset_key,
                    active_dataset_version_id,
                    f"legacy_schema:{legacy_fingerprint}",
                    f"legacy_lineage:{legacy_fingerprint}",
                    f"legacy_quality:{legacy_fingerprint}",
                    started_at,
                    compute_content_hash(
                        {
                            "dataset_asset_version_id": asset_id,
                            "legacy_fingerprint": legacy_fingerprint,
                            "created_at": started_at,
                        }
                    ),
                ),
            )
            event_id = f"ksae_{uuid4().hex}"
            event_payload = {
                "activation_event_id": event_id,
                "knowledge_source_id": source_id,
                "dataset_asset_version_id": asset_id,
                "organization_id": organization_id,
                "workspace_id": None,
                "prior_activation_event_hash": None,
                "decision": "MIGRATE",
                "actor": actor,
                "actor_authority": "system_migration",
                "reason": "legacy Organization dataset compatibility backfill",
                "artifact_version_hash": legacy_fingerprint,
                "occurred_at": started_at,
            }
            event_hash = compute_content_hash(event_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_activation_events (
                  activation_event_id, knowledge_source_id, dataset_asset_version_id,
                  organization_id, workspace_id, prior_activation_event_hash, decision,
                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash
                )
                VALUES (?, ?, ?, ?, NULL, NULL, 'MIGRATE', ?, 'system_migration', ?, ?, ?, ?)
                """, "\n                INSERT INTO knowledge_source_activation_events (\n                  activation_event_id, knowledge_source_id, dataset_asset_version_id,\n                  organization_id, workspace_id, prior_activation_event_hash, decision,\n                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, NULL, NULL, 'MIGRATE', %s, 'system_migration', %s, %s, %s, %s)\n                "),
                (
                    event_id,
                    source_id,
                    asset_id,
                    organization_id,
                    actor,
                    "legacy Organization dataset compatibility backfill",
                    legacy_fingerprint,
                    started_at,
                    event_hash,
                ),
            )
            activation_payload = {
                "activation_id": activation_id,
                "knowledge_source_id": source_id,
                "dataset_asset_version_id": asset_id,
                "organization_id": organization_id,
                "workspace_id": None,
                "activation_scope": "organization",
                "scope_key": "organization",
                "activation_status": "active",
                "workspace_override_policy": "none",
                "user_context_control_level": "none",
                "activation_event_head_hash": event_hash,
                "authority_source_manifest": {"migration_event_id": event_id},
                "created_at": started_at,
            }
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_activation (
                  activation_id, knowledge_source_id, dataset_asset_version_id,
                  organization_id, workspace_id, activation_scope, scope_key,
                  activation_status, workspace_override_policy, user_context_control_level,
                  activation_event_head_hash, authority_source_manifest_json, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, NULL, 'organization', 'organization', 'active', 'none', 'none', ?, ?, ?, ?)
                """, "\n                INSERT INTO knowledge_source_activation (\n                  activation_id, knowledge_source_id, dataset_asset_version_id,\n                  organization_id, workspace_id, activation_scope, scope_key,\n                  activation_status, workspace_override_policy, user_context_control_level,\n                  activation_event_head_hash, authority_source_manifest_json, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, NULL, 'organization', 'organization', 'active', 'none', 'none', %s, %s, %s, %s)\n                "),
                (
                    activation_id,
                    source_id,
                    asset_id,
                    organization_id,
                    event_hash,
                    canonical_json({"migration_event_id": event_id}),
                    started_at,
                    compute_content_hash(activation_payload),
                ),
            )
            migration_payload = {
                "organization_id": organization_id,
                "migration_status": "complete",
                "source_legacy_fingerprint": legacy_fingerprint,
                "activation_id": activation_id,
                "migration_event_ref": event_id,
                "started_at": started_at,
                "completed_at": started_at,
                "last_error_code": None,
            }
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_activation_migration_state (
                  organization_id, migration_status, source_legacy_fingerprint,
                  activation_id, migration_event_ref, started_at, completed_at,
                  last_error_code, content_hash
                )
                VALUES (?, 'complete', ?, ?, ?, ?, ?, NULL, ?)
                """, "\n                INSERT INTO knowledge_activation_migration_state (\n                  organization_id, migration_status, source_legacy_fingerprint,\n                  activation_id, migration_event_ref, started_at, completed_at,\n                  last_error_code, content_hash\n                )\n                VALUES (%s, 'complete', %s, %s, %s, %s, %s, NULL, %s)\n                "),
                (
                    organization_id,
                    legacy_fingerprint,
                    activation_id,
                    event_id,
                    started_at,
                    started_at,
                    compute_content_hash(migration_payload),
                ),
            )
            return self.get_active_dataset_binding(organization_id=organization_id)

    def get_activation(self, activation_id: str) -> dict[str, Any] | None:
        activation_id = _assert_ascii_identifier(activation_id, "activation_id")
        with self._lock:
            return _row_dict(
                self._conn.execute(
                    sql_query("SELECT * FROM knowledge_source_activation WHERE activation_id = ?", 'SELECT * FROM knowledge_source_activation WHERE activation_id = %s'),
                    (activation_id,),
                ).fetchone()
            )

    def transition_activation(
        self,
        *,
        activation_id: str,
        activation_status: ActivationStatus,
        decision: ActivationDecision,
        actor: str,
        actor_authority: str,
        reason: str,
        artifact_version_hash: str,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        activation_id = _assert_ascii_identifier(activation_id, "activation_id")
        occurred_at = _assert_utc_timestamp(occurred_at, "occurred_at") or _now_iso()
        with self._lock, transaction_scope(self._conn):
            activation = self._conn.execute(
                sql_query("SELECT * FROM knowledge_source_activation WHERE activation_id = ?", 'SELECT * FROM knowledge_source_activation WHERE activation_id = %s'),
                (activation_id,),
            ).fetchone()
            if activation is None:
                raise ValueError("activation_id does not exist")
            event_id = f"ksae_{uuid4().hex}"
            event_payload = {
                "activation_event_id": event_id,
                "knowledge_source_id": activation["knowledge_source_id"],
                "dataset_asset_version_id": activation["dataset_asset_version_id"],
                "organization_id": activation["organization_id"],
                "workspace_id": activation["workspace_id"],
                "prior_activation_event_hash": activation["activation_event_head_hash"],
                "decision": decision,
                "actor": actor,
                "actor_authority": actor_authority,
                "reason": reason,
                "artifact_version_hash": artifact_version_hash,
                "occurred_at": occurred_at,
            }
            event_hash = compute_content_hash(event_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_activation_events (
                  activation_event_id, knowledge_source_id, dataset_asset_version_id,
                  organization_id, workspace_id, prior_activation_event_hash, decision,
                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source_activation_events (\n                  activation_event_id, knowledge_source_id, dataset_asset_version_id,\n                  organization_id, workspace_id, prior_activation_event_hash, decision,\n                  actor, actor_authority, reason, artifact_version_hash, occurred_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event_id,
                    activation["knowledge_source_id"],
                    activation["dataset_asset_version_id"],
                    activation["organization_id"],
                    activation["workspace_id"],
                    activation["activation_event_head_hash"],
                    decision,
                    actor,
                    actor_authority,
                    reason,
                    artifact_version_hash,
                    occurred_at,
                    event_hash,
                ),
            )
            activation_payload = {
                "activation_id": activation_id,
                "knowledge_source_id": activation["knowledge_source_id"],
                "dataset_asset_version_id": activation["dataset_asset_version_id"],
                "organization_id": activation["organization_id"],
                "workspace_id": activation["workspace_id"],
                "activation_scope": activation["activation_scope"],
                "scope_key": activation["scope_key"],
                "activation_status": activation_status,
                "workspace_override_policy": activation["workspace_override_policy"],
                "user_context_control_level": activation["user_context_control_level"],
                "activation_event_head_hash": event_hash,
                "authority_source_manifest": json.loads(
                    activation["authority_source_manifest_json"] or "{}"
                ),
                "created_at": activation["created_at"],
            }
            row_hash = compute_content_hash(activation_payload)
            self._conn.execute(
                sql_query("""
                UPDATE knowledge_source_activation
                SET activation_status = ?,
                    activation_event_head_hash = ?,
                    content_hash = ?
                WHERE activation_id = ?
                """, '\n                UPDATE knowledge_source_activation\n                SET activation_status = %s,\n                    activation_event_head_hash = %s,\n                    content_hash = %s\n                WHERE activation_id = %s\n                '),
                (activation_status, event_hash, row_hash, activation_id),
            )
            return dict(
                self._conn.execute(
                    sql_query("SELECT * FROM knowledge_source_activation WHERE activation_id = ?", 'SELECT * FROM knowledge_source_activation WHERE activation_id = %s'),
                    (activation_id,),
                ).fetchone()
            )
