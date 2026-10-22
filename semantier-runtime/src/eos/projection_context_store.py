"""Immutable projection context snapshot store (v6 Slice 3 foundation)."""

# Patent evidence C005: immutable projection context snapshot that captures the
# selected evidence context and artifact version pins used as the replayable
# decision basis, referenced in the admissibility-governance method claims.

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


def _normalize_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    return value


def create_snapshot(
    *,
    projection_context_ref: str,
    org_id: str,
    retrieval_query_hash: str,
    provider_ref: str = "hermes-agent/plugins/memory/holographic",
    provider_version: str,
    adapter_version: str = "semantier_memory_adapter_v1",
    selected_context: Dict[str, Any],
    created_by: str,
    event_id: Optional[str] = None,
    knowledge_version: Optional[str] = None,
    tag_ontology_version: Optional[str] = None,
    projection_bundle_version: Optional[str] = None,
    constraint_bundle_version: Optional[str] = None,
    admissibility_policy_version: Optional[str] = None,
    rejected_items_summary: Optional[list[dict]] = None,
    retrieved_source_refs: Optional[list[dict]] = None,
    excluded_context_refs: Optional[list[dict]] = None,
    selection_reason: str = "",
) -> Dict[str, Any]:
    """Create an immutable projection context snapshot.

    Fails if the reference already exists.
    """
    snapshot_payload = {
        "projection_context_ref": projection_context_ref,
        "org_id": org_id,
        "event_id": event_id,
        "knowledge_version": knowledge_version,
        "tag_ontology_version": tag_ontology_version,
        "projection_bundle_version": projection_bundle_version,
        "constraint_bundle_version": constraint_bundle_version,
        "retrieval_query_hash": retrieval_query_hash,
        "provider_ref": provider_ref,
        "provider_version": provider_version,
        "adapter_version": adapter_version,
        "admissibility_policy_version": admissibility_policy_version,
        "rejected_items_summary": _normalize_json(rejected_items_summary, []),
        "retrieved_source_refs": _normalize_json(retrieved_source_refs, []),
        "selected_context": _normalize_json(selected_context, {}),
        "excluded_context_refs": _normalize_json(excluded_context_refs, []),
        "selection_reason": selection_reason,
        "created_by": created_by,
    }
    projection_context_hash = content_hash(snapshot_payload)

    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("SELECT projection_context_ref FROM projection_context_snapshots WHERE projection_context_ref = ?", 'SELECT projection_context_ref FROM projection_context_snapshots WHERE projection_context_ref = %s'),
            (projection_context_ref,),
        ).fetchone()
        if existing:
            raise ValueError(f"projection_context_ref already exists: {projection_context_ref}")

        conn.execute(
            hosted.sql_query("""
            INSERT INTO projection_context_snapshots (
              projection_context_ref, projection_context_hash, org_id, event_id,
              knowledge_version, tag_ontology_version, projection_bundle_version,
              constraint_bundle_version, retrieval_query_hash, provider_ref, provider_version,
              adapter_version, admissibility_policy_version, rejected_items_summary_json,
              retrieved_source_refs_json, selected_context_json, excluded_context_refs_json,
              selection_reason, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO projection_context_snapshots (
              projection_context_ref, projection_context_hash, org_id, event_id,
              knowledge_version, tag_ontology_version, projection_bundle_version,
              constraint_bundle_version, retrieval_query_hash, provider_ref, provider_version,
              adapter_version, admissibility_policy_version, rejected_items_summary_json,
              retrieved_source_refs_json, selected_context_json, excluded_context_refs_json,
              selection_reason, created_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                projection_context_ref,
                projection_context_hash,
                org_id,
                event_id,
                knowledge_version,
                tag_ontology_version,
                projection_bundle_version,
                constraint_bundle_version,
                retrieval_query_hash,
                provider_ref,
                provider_version,
                adapter_version,
                admissibility_policy_version,
                canonical_json(snapshot_payload["rejected_items_summary"]),
                canonical_json(snapshot_payload["retrieved_source_refs"]),
                canonical_json(snapshot_payload["selected_context"]),
                canonical_json(snapshot_payload["excluded_context_refs"]),
                selection_reason,
                now_iso(),
                created_by,
            ),
        )

    return {
        **snapshot_payload,
        "projection_context_hash": projection_context_hash,
    }


def load_snapshot(projection_context_ref: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM projection_context_snapshots WHERE projection_context_ref = ?", 'SELECT * FROM projection_context_snapshots WHERE projection_context_ref = %s'),
            (projection_context_ref,),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["rejected_items_summary"] = json.loads(result.pop("rejected_items_summary_json") or "[]")
    result["retrieved_source_refs"] = json.loads(result.pop("retrieved_source_refs_json") or "[]")
    result["selected_context"] = json.loads(result.pop("selected_context_json") or "{}")
    result["excluded_context_refs"] = json.loads(result.pop("excluded_context_refs_json") or "[]")
    return result


def verify_snapshot_hash(projection_context_ref: str) -> Dict[str, Any]:
    row = load_snapshot(projection_context_ref)
    if not row:
        return {"status": "NOT_FOUND", "reason": "projection_context_ref_not_found"}
    payload = {
        "projection_context_ref": row["projection_context_ref"],
        "org_id": row["org_id"],
        "event_id": row.get("event_id"),
        "knowledge_version": row.get("knowledge_version"),
        "tag_ontology_version": row.get("tag_ontology_version"),
        "projection_bundle_version": row.get("projection_bundle_version"),
        "constraint_bundle_version": row.get("constraint_bundle_version"),
        "retrieval_query_hash": row["retrieval_query_hash"],
        "provider_ref": row.get("provider_ref") or "hermes-agent/plugins/memory/holographic",
        "provider_version": row["provider_version"],
        "adapter_version": row.get("adapter_version") or "semantier_memory_adapter_v1",
        "admissibility_policy_version": row.get("admissibility_policy_version"),
        "rejected_items_summary": row.get("rejected_items_summary", []),
        "retrieved_source_refs": row.get("retrieved_source_refs", []),
        "selected_context": row.get("selected_context", {}),
        "excluded_context_refs": row.get("excluded_context_refs", []),
        "selection_reason": row.get("selection_reason") or "",
        "created_by": row["created_by"],
    }
    recalculated = content_hash(payload)
    return {
        "status": "OK" if recalculated == row["projection_context_hash"] else "HASH_MISMATCH",
        "stored_hash": row["projection_context_hash"],
        "recalculated_hash": recalculated,
        "match": recalculated == row["projection_context_hash"],
    }
