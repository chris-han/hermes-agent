"""Replay binding persistence and verification helpers (v6 Slice 4)."""

# Patent evidence C004: replay binding creation and verification that pin the
# decision basis (artifact versions, projection context hashes, resolution trace,
# effects, event payload) for reproducible governance decisions, referenced in
# the admissibility-governance method claims.

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional

from eos.artifact_hashing import content_hash, canonical_json
from eos.db import get_conn, now_iso
from eos.projection_context_store import load_snapshot, verify_snapshot_hash
from storage.kgl_store import KGLStore


def _resolve_authority_source_manifest(
    knowledge_version: Optional[str],
    explicit_manifest: Optional[list[dict]],
) -> list[dict]:
    if explicit_manifest is not None:
        return explicit_manifest
    if not knowledge_version:
        return []
    version = KGLStore().get_kgl_version(knowledge_version)
    if not version:
        return []
    manifest = version.get("authority_source_manifest_json")
    return manifest if isinstance(manifest, list) else []


def create_replay_binding(
    *,
    event_id: str,
    pins: Dict[str, Any],
    projection_context_ref: str,
    projection_context_hash: str,
    resolution_trace: Dict[str, Any],
    effects: Any,
    event_payload: Any,
    authority_source_manifest: Optional[list[dict]] = None,
) -> Dict[str, Any]:
    resolved_manifest = _resolve_authority_source_manifest(pins.get("K_v"), authority_source_manifest)
    row = {
        "event_id": event_id,
        "ontology_version": pins.get("O_v"),
        "contract_version": pins.get("C_v"),
        "tag_ontology_version": pins.get("O_tag_v"),
        "knowledge_version": pins.get("K_v"),
        "projection_bundle_version": pins.get("Pi_v"),
        "constraint_bundle_version": pins.get("ConstraintBundle_v"),
        "precedence_graph_version": pins.get("PrecedenceGraph_v"),
        "resolution_rules_version": pins.get("ResolutionRules_v"),
        "projection_context_ref": projection_context_ref,
        "projection_context_hash": projection_context_hash,
        "authority_source_manifest_json": resolved_manifest,
        "authority_source_manifest_hash": content_hash(resolved_manifest),
        "resolution_trace_hash": content_hash(resolution_trace),
        "effect_hash": content_hash(effects),
        "event_hash": content_hash(event_payload),
        "created_at": now_iso(),
    }
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("SELECT * FROM replay_bindings WHERE event_id = ?", 'SELECT * FROM replay_bindings WHERE event_id = %s'),
            (event_id,),
        ).fetchone()
        if existing:
            existing_row = dict(existing)
            comparable_fields = (
                "ontology_version",
                "contract_version",
                "tag_ontology_version",
                "knowledge_version",
                "projection_bundle_version",
                "constraint_bundle_version",
                "precedence_graph_version",
                "resolution_rules_version",
                "projection_context_ref",
                "projection_context_hash",
                "authority_source_manifest_hash",
                "resolution_trace_hash",
                "effect_hash",
                "event_hash",
            )
            if all(existing_row.get(field) == row.get(field) for field in comparable_fields):
                return get_replay_binding(event_id) or existing_row
            raise ValueError(
                f"replay_binding already exists for event_id={event_id} with different pinned artifacts"
            )
        conn.execute(
            hosted.sql_query("""
            INSERT INTO replay_bindings (
              event_id, ontology_version, contract_version, tag_ontology_version, knowledge_version,
              projection_bundle_version, constraint_bundle_version, precedence_graph_version,
              resolution_rules_version, projection_context_ref, projection_context_hash, authority_source_manifest_json,
              authority_source_manifest_hash,
              resolution_trace_hash, effect_hash, event_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO replay_bindings (
              event_id, ontology_version, contract_version, tag_ontology_version, knowledge_version,
              projection_bundle_version, constraint_bundle_version, precedence_graph_version,
              resolution_rules_version, projection_context_ref, projection_context_hash, authority_source_manifest_json,
              authority_source_manifest_hash,
              resolution_trace_hash, effect_hash, event_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["event_id"], row["ontology_version"], row["contract_version"],
                row["tag_ontology_version"], row["knowledge_version"], row["projection_bundle_version"],
                row["constraint_bundle_version"], row["precedence_graph_version"], row["resolution_rules_version"],
                row["projection_context_ref"], row["projection_context_hash"],
                canonical_json(row["authority_source_manifest_json"]), row["authority_source_manifest_hash"],
                row["resolution_trace_hash"], row["effect_hash"], row["event_hash"], row["created_at"],
            ),
        )
    return row


def get_replay_binding(event_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        r = conn.execute(hosted.sql_query("SELECT * FROM replay_bindings WHERE event_id = ?", 'SELECT * FROM replay_bindings WHERE event_id = %s'), (event_id,)).fetchone()
    if not r:
        return None
    result = dict(r)
    result["authority_source_manifest_json"] = json.loads(
        result.get("authority_source_manifest_json") or "[]"
    )
    return result


def verify_replay_binding(event_id: str) -> Dict[str, Any]:
    binding = get_replay_binding(event_id)
    if not binding:
        return {"status": "NOT_FOUND", "reason": "replay_binding_not_found"}

    ref = binding.get("projection_context_ref")
    if not ref:
        return {"status": "FAIL_CLOSED", "reason": "missing_projection_context_ref"}

    snapshot = load_snapshot(ref)
    if not snapshot:
        return {"status": "FAIL_CLOSED", "reason": "projection_context_snapshot_missing", "projection_context_ref": ref}

    hash_ver = verify_snapshot_hash(ref)
    if hash_ver.get("status") != "OK":
        return {"status": "FAIL_CLOSED", "reason": "projection_context_hash_mismatch", "projection_context_ref": ref, "hash_verification": hash_ver}

    if snapshot.get("projection_context_hash") != binding.get("projection_context_hash"):
        return {
            "status": "FAIL_CLOSED",
            "reason": "projection_context_hash_mismatch_with_binding",
            "projection_context_ref": ref,
            "snapshot_hash": snapshot.get("projection_context_hash"),
            "binding_hash": binding.get("projection_context_hash"),
        }

    required_pin_fields = [
        "knowledge_version",
        "projection_bundle_version",
        "resolution_rules_version",
        "precedence_graph_version",
    ]
    missing = [field for field in required_pin_fields if not binding.get(field)]
    if missing:
        return {"status": "FAIL_CLOSED", "reason": "missing_required_replay_pins", "missing": missing}

    manifest = binding.get("authority_source_manifest_json", [])
    stored_manifest_hash = binding.get("authority_source_manifest_hash") or ""
    recalculated_manifest_hash = content_hash(manifest)
    if stored_manifest_hash and stored_manifest_hash != recalculated_manifest_hash:
        return {
            "status": "FAIL_CLOSED",
            "reason": "authority_source_manifest_hash_mismatch",
            "stored_hash": stored_manifest_hash,
            "recalculated_hash": recalculated_manifest_hash,
        }

    return {
        "status": "OK",
        "event_id": event_id,
        "projection_context_ref": ref,
        "authority_source_manifest_hash": stored_manifest_hash or recalculated_manifest_hash,
        "authority_source_manifest_count": len(manifest),
    }
