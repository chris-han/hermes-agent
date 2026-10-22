"""KGL storage for governed knowledge artifacts (v6 foundation).

Implements a deterministic lifecycle for knowledge ingestion:
PROPOSED -> VALIDATED -> APPROVED -> ACTIVE -> DEPRECATED
(with REJECTED terminal path).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.kgl_conflict_store import record_conflict_review
from eos.kgl_source_registration import normalize_source_registration
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

_DB_PATH = sqlite_db_path("eos.db", ".semantier/eos.db")

SourceType = Literal[
    "official_regulation",
    "official_taxonomy",
    "field_guide",
    "firm_guidance",
    "internal_policy",
    "runtime_observation",
]
AuthorityOrigin = Literal[
    "official_source",
    "interpretive_source",
    "internal_governance",
    "runtime_observation",
    "ontology_governance",
]
AuthorityDomain = Literal["accounting", "tax", "management", "compliance", "internal_control"]
AuthorityLevel = Literal["regulatory", "accounting", "management", "operational", "descriptive"]
SemanticTier = Literal["T2", "T3", "T4", "T5", "T6"]
IngestionStatus = Literal["PROPOSED", "VALIDATED", "APPROVED", "ACTIVE", "REJECTED", "DEPRECATED"]

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PROPOSED": {"VALIDATED", "REJECTED"},
    "VALIDATED": {"APPROVED", "REJECTED"},
    "APPROVED": {"ACTIVE", "REJECTED"},
    "ACTIVE": {"DEPRECATED"},
    "REJECTED": set(),
    "DEPRECATED": set(),
}

_TRANSITION_GUARDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("PROPOSED", "VALIDATED"): ("schema_validation_pass", "source_registration_present", "authority_origin_valid"),
    ("VALIDATED", "APPROVED"): ("precedence_check_complete", "no_unresolved_higher_tier_conflict"),
    ("APPROVED", "ACTIVE"): ("replay_dry_run_pass",),
    ("ACTIVE", "DEPRECATED"): ("supersession_candidate_exists_or_waiver_recorded",),
}


_SOURCE_TYPE_DEFAULT_ORIGIN: dict[SourceType, AuthorityOrigin] = {
    "official_regulation": "official_source",
    "official_taxonomy": "ontology_governance",
    "field_guide": "interpretive_source",
    "firm_guidance": "interpretive_source",
    "internal_policy": "internal_governance",
    "runtime_observation": "runtime_observation",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(_DB_PATH)))


def _assert_transition_guards(
    *,
    from_status: str,
    to_status: str,
    semantic_tier: str,
    guard_results: Optional[dict[str, bool]],
) -> None:
    if guard_results is None:
        return
    required_guards = list(_TRANSITION_GUARDS.get((from_status, to_status), ()))
    if to_status == "ACTIVE" and semantic_tier in {"T2", "T3"}:
        required_guards.append("authority_origin_verified")
    if to_status == "ACTIVE" and semantic_tier in {"T1", "T2"}:
        required_guards.append("non_interference_check_when_required")
    failed = [guard for guard in required_guards if not guard_results.get(guard)]
    if failed:
        raise ValueError(
            f"Transition {from_status}->{to_status} missing guards: {','.join(sorted(failed))}"
        )


def _default_authority_origin(source_type: SourceType) -> AuthorityOrigin:
    return _SOURCE_TYPE_DEFAULT_ORIGIN[source_type]


def _assert_authority_origin_compatibility(
    *,
    source_type: SourceType,
    authority_origin: AuthorityOrigin,
    semantic_tier: SemanticTier,
) -> None:
    expected_origin = _default_authority_origin(source_type)
    if authority_origin != expected_origin:
        raise ValueError(
            f"authority_origin {authority_origin} is incompatible with source_type {source_type}; "
            f"expected {expected_origin}"
        )

    if semantic_tier == "T2":
        if authority_origin != "official_source" or source_type != "official_regulation":
            raise ValueError("T2 authority must originate from official_regulation with official_source origin")
        return

    if semantic_tier == "T3":
        if authority_origin not in {"ontology_governance", "interpretive_source"}:
            raise ValueError("T3 authority must originate from ontology_governance or interpretive_source")
        if source_type not in {"official_taxonomy", "field_guide", "firm_guidance"}:
            raise ValueError("T3 authority must originate from official_taxonomy, field_guide, or firm_guidance")
        return

    if semantic_tier == "T4":
        if authority_origin != "internal_governance" or source_type != "internal_policy":
            raise ValueError("T4 authority must originate from internal_policy with internal_governance origin")
        return

    if semantic_tier in {"T5", "T6"} and authority_origin not in {"internal_governance", "runtime_observation"}:
        raise ValueError("T5/T6 artifacts must originate from internal_governance or runtime_observation")


class KGLStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'knowledge_artifacts','kgl_versions','kgl_approval_records')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    def propose_artifact(
        self,
        *,
        source_type: SourceType,
        authority_domain: AuthorityDomain,
        tag_authority_level: AuthorityLevel,
        semantic_tier: SemanticTier,
        source_ref: str,
        source_hash: str,
        authority_origin: AuthorityOrigin | None = None,
        source_registration: dict | None = None,
        extracted_claims: list[dict] | None = None,
        extracted_constraints: list[dict] | None = None,
        extracted_projection_rules: list[dict] | None = None,
    ) -> dict:
        """Insert a PROPOSED knowledge artifact.

        Idempotent by source_hash.
        """
        with self._lock:
            authority_origin = authority_origin or _default_authority_origin(source_type)
            if semantic_tier in {"T2", "T3"} and not str(source_hash or "").strip():
                raise ValueError("T2/T3 source_hash is required")
            if source_registration is None:
                if semantic_tier in {"T2", "T3"}:
                    raise ValueError("T2/T3 source_registration is required")
                normalized_registration = {
                    "source_uri": source_ref,
                    "jurisdiction": "UNSPECIFIED",
                    "effective_from": "1970-01-01",
                    "effective_to": None,
                    "source_version": source_hash,
                    "anchors": [],
                    "extraction_method": "unspecified",
                    "curator": "system",
                    "ambiguities": ["source_registration_not_supplied"],
                }
            else:
                normalized_registration = normalize_source_registration(
                    source_registration,
                    strict_authority=semantic_tier in {"T2", "T3"},
                )
            _assert_authority_origin_compatibility(
                source_type=source_type,
                authority_origin=authority_origin,
                semantic_tier=semantic_tier,
            )
            existing = self._conn.execute(
                sql_query("SELECT * FROM knowledge_artifacts WHERE source_hash = ? ORDER BY created_at DESC LIMIT 1", 'SELECT * FROM knowledge_artifacts WHERE source_hash = %s ORDER BY created_at DESC LIMIT 1'),
                (source_hash,),
            ).fetchone()
            if existing:
                return dict(existing)

            artifact_id = f"ka_{uuid4().hex}"
            now = _now_iso()
            claims_json = canonical_json(extracted_claims or [])
            constraints_json = canonical_json(extracted_constraints or [])
            rules_json = canonical_json(extracted_projection_rules or [])
            artifact_payload = {
                "artifact_id": artifact_id,
                "source_type": source_type,
                "authority_domain": authority_domain,
                "tag_authority_level": tag_authority_level,
                "semantic_tier": semantic_tier,
                "authority_origin": authority_origin,
                "source_ref": source_ref,
                "source_hash": source_hash,
                "source_registration": normalized_registration,
                "extracted_claims": extracted_claims or [],
                "extracted_constraints": extracted_constraints or [],
                "extracted_projection_rules": extracted_projection_rules or [],
            }
            artifact_content_hash = compute_content_hash(artifact_payload)
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_artifacts (
                  artifact_id, source_type, authority_domain, tag_authority_level, semantic_tier,
                  authority_origin, source_ref, source_hash, source_uri, jurisdiction, effective_from,
                  effective_to, source_version, source_anchors_json, extraction_method, curator,
                  ambiguities_json, extracted_claims_json, extracted_constraints_json,
                  extracted_projection_rules_json, ingestion_status, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PROPOSED', ?, ?)
                """, "\n                INSERT INTO knowledge_artifacts (\n                  artifact_id, source_type, authority_domain, tag_authority_level, semantic_tier,\n                  authority_origin, source_ref, source_hash, source_uri, jurisdiction, effective_from,\n                  effective_to, source_version, source_anchors_json, extraction_method, curator,\n                  ambiguities_json, extracted_claims_json, extracted_constraints_json,\n                  extracted_projection_rules_json, ingestion_status, created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PROPOSED', %s, %s)\n                "),
                (
                    artifact_id,
                    source_type,
                    authority_domain,
                    tag_authority_level,
                    semantic_tier,
                    authority_origin,
                    source_ref,
                    source_hash,
                    normalized_registration["source_uri"],
                    normalized_registration["jurisdiction"],
                    normalized_registration["effective_from"],
                    normalized_registration["effective_to"],
                    normalized_registration["source_version"],
                    canonical_json(normalized_registration["anchors"]),
                    normalized_registration["extraction_method"],
                    normalized_registration["curator"],
                    canonical_json(normalized_registration["ambiguities"]),
                    claims_json,
                    constraints_json,
                    rules_json,
                    now,
                    artifact_content_hash,
                ),
            )
            self._conn.commit()
            return self.get_artifact(artifact_id)  # type: ignore[return-value]

    def transition_artifact(
        self,
        artifact_id: str,
        to_status: IngestionStatus,
        *,
        actor: str,
        notes: str = "",
        actor_roles: Optional[Iterable[str]] = None,
        guard_results: Optional[dict[str, bool]] = None,
        precedence_review: Optional[dict] = None,
    ) -> dict:
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM knowledge_artifacts WHERE artifact_id = ?", 'SELECT * FROM knowledge_artifacts WHERE artifact_id = %s'),
                (artifact_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown artifact_id: {artifact_id}")
            from_status = str(row["ingestion_status"])
            if to_status not in _VALID_TRANSITIONS.get(from_status, set()):
                raise ValueError(f"Invalid transition {from_status} -> {to_status}")
            semantic_tier = str(row["semantic_tier"])
            _assert_transition_guards(
                from_status=from_status,
                to_status=to_status,
                semantic_tier=semantic_tier,
                guard_results=guard_results,
            )

            approved_by = row["approved_by"]
            if to_status in {"APPROVED", "ACTIVE"}:
                approved_by = actor

            self._conn.execute(
                sql_query("UPDATE knowledge_artifacts SET ingestion_status = ?, approved_by = ? WHERE artifact_id = ?", 'UPDATE knowledge_artifacts SET ingestion_status = %s, approved_by = %s WHERE artifact_id = %s'),
                (to_status, approved_by, artifact_id),
            )
            if to_status == "APPROVED" and precedence_review is not None:
                record_conflict_review(
                    artifact_id=artifact_id,
                    payload=precedence_review,
                    created_by=actor,
                    conn=self._conn,
                )
            self._conn.execute(
                sql_query("""
                INSERT INTO kgl_approval_records (record_id, artifact_id, from_status, to_status, actor, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO kgl_approval_records (record_id, artifact_id, from_status, to_status, actor, notes, created_at)\n                VALUES (%s, %s, %s, %s, %s, %s, %s)\n                '),
                (f"kglr_{uuid4().hex}", artifact_id, from_status, to_status, actor, notes, _now_iso()),
            )
            self._conn.commit()
            return self.get_artifact(artifact_id)  # type: ignore[return-value]

    def list_artifacts(self, *, status: Optional[IngestionStatus] = None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    sql_query("SELECT * FROM knowledge_artifacts WHERE ingestion_status = ? ORDER BY created_at DESC", 'SELECT * FROM knowledge_artifacts WHERE ingestion_status = %s ORDER BY created_at DESC'),
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge_artifacts ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM knowledge_artifacts WHERE artifact_id = ?", 'SELECT * FROM knowledge_artifacts WHERE artifact_id = %s'),
                (artifact_id,),
            ).fetchone()
            return dict(row) if row else None

    def _build_authority_source_manifest(self, active_artifacts: list[str]) -> list[dict]:
        if not active_artifacts:
            return []
        placeholders = ", ".join(sql_query("?", '%s') for _ in active_artifacts)
        rows = self._conn.execute(
            f"""
            SELECT artifact_id, source_type, authority_domain, tag_authority_level,
                   semantic_tier, authority_origin, source_ref, source_hash, jurisdiction,
                   effective_from, effective_to, source_version, ingestion_status
            FROM knowledge_artifacts
            WHERE artifact_id IN ({placeholders})
            ORDER BY artifact_id
            """,
            active_artifacts,
        ).fetchall()
        if len(rows) != len(set(active_artifacts)):
            found_ids = {str(row["artifact_id"]) for row in rows}
            missing = sorted(set(active_artifacts) - found_ids)
            raise KeyError(f"Unknown artifact ids in KGL version manifest: {','.join(missing)}")
        return [
            {
                "artifact_id": str(row["artifact_id"]),
                "source_type": str(row["source_type"]),
                "authority_domain": str(row["authority_domain"]),
                "tag_authority_level": str(row["tag_authority_level"]),
                "semantic_tier": str(row["semantic_tier"]),
                "authority_origin": str(row["authority_origin"]),
                "source_ref": str(row["source_ref"]),
                "source_hash": str(row["source_hash"]),
                "jurisdiction": row["jurisdiction"],
                "effective_from": row["effective_from"],
                "effective_to": row["effective_to"],
                "source_version": row["source_version"],
                "ingestion_status": str(row["ingestion_status"]),
            }
            for row in rows
        ]

    def record_kgl_version(
        self,
        *,
        knowledge_version_id: str,
        active_artifacts: list[str],
        source_hashes: list[str],
        authority_source_manifest: list[dict] | None = None,
        extracted_claims: list[dict],
        extracted_constraints: list[dict],
        extracted_projection_rules: list[dict],
        content_hash: str,
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None,
    ) -> dict:
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM kgl_versions WHERE knowledge_version_id = ?", 'SELECT * FROM kgl_versions WHERE knowledge_version_id = %s'),
                (knowledge_version_id,),
            ).fetchone()
            if row:
                return self.get_kgl_version(knowledge_version_id)  # type: ignore[return-value]

            authority_source_manifest = (
                authority_source_manifest
                if authority_source_manifest is not None
                else self._build_authority_source_manifest(active_artifacts)
            )

            self._conn.execute(
                sql_query("""
                INSERT INTO kgl_versions (
                  knowledge_version_id, active_artifacts_json, source_hashes_json, authority_source_manifest_json,
                  extracted_claims_json, extracted_constraints_json, extracted_projection_rules_json,
                  effective_from, effective_to, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO kgl_versions (\n                  knowledge_version_id, active_artifacts_json, source_hashes_json, authority_source_manifest_json,\n                  extracted_claims_json, extracted_constraints_json, extracted_projection_rules_json,\n                  effective_from, effective_to, content_hash, created_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    knowledge_version_id,
                    json.dumps(active_artifacts, sort_keys=True),
                    json.dumps(source_hashes, sort_keys=True),
                    json.dumps(authority_source_manifest, sort_keys=True),
                    json.dumps(extracted_claims, sort_keys=True),
                    json.dumps(extracted_constraints, sort_keys=True),
                    json.dumps(extracted_projection_rules, sort_keys=True),
                    effective_from or _now_iso(),
                    effective_to,
                    content_hash,
                    _now_iso(),
                ),
            )
            self._conn.commit()
            return self.get_kgl_version(knowledge_version_id)  # type: ignore[return-value]

    def get_kgl_version(self, knowledge_version_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                sql_query("SELECT * FROM kgl_versions WHERE knowledge_version_id = ?", 'SELECT * FROM kgl_versions WHERE knowledge_version_id = %s'),
                (knowledge_version_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            for key in (
                "active_artifacts_json",
                "source_hashes_json",
                "authority_source_manifest_json",
                "extracted_claims_json",
                "extracted_constraints_json",
                "extracted_projection_rules_json",
            ):
                result[key] = json.loads(result[key])
            return result

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def get_kgl_store() -> KGLStore:
    return KGLStore()
