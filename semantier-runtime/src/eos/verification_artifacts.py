"""Durable explainability, audit, export, and verification-contract artifacts."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso
from eos.replay_binding_store import get_replay_binding, verify_replay_binding


def _persist_json_artifact(
    *,
    table: str,
    id_column: str,
    id_prefix: str,
    payload_column: str,
    subject_ref: str,
    payload: Dict[str, Any],
    lineage: Optional[list[dict]] = None,
    status: Optional[str] = None,
    activated_at: Optional[str] = None,
) -> Dict[str, Any]:
    row_id = f"{id_prefix}_{uuid4().hex}"
    created_at = now_iso()
    payload_json = canonical_json(payload)
    hash_value = content_hash(payload)
    columns = [id_column, "subject_ref", payload_column, "content_hash", "created_at"]
    values = [row_id, subject_ref, payload_json, hash_value, created_at]
    if table == "external_audit_exports":
        columns.append("export_lineage_json")
        values.append(canonical_json(lineage or []))
    if status is not None:
        columns.append("status")
        values.append(status)
    if activated_at is not None:
        columns.append("activated_at")
        values.append(activated_at)
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(columns))
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(values),
        )
    result = {
        id_column: row_id,
        "subject_ref": subject_ref,
        "content_hash": hash_value,
        "created_at": created_at,
        "payload": payload,
    }
    if status is not None:
        result["status"] = status
    if activated_at is not None:
        result["activated_at"] = activated_at
    return result


def persist_explainability_surface(event_id: str, surface: Dict[str, Any]) -> Dict[str, Any]:
    explainability_surface_id = f"xsurf_{uuid4().hex}"
    created_at = now_iso()
    surface_json = canonical_json(surface)
    hash_value = content_hash(surface)
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO explainability_surfaces
              (explainability_surface_id, event_id, surface_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, """
            INSERT INTO explainability_surfaces
              (explainability_surface_id, event_id, surface_json, content_hash, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """),
            (
                explainability_surface_id,
                event_id,
                surface_json,
                hash_value,
                created_at,
            ),
        )
    return {
        "explainability_surface_id": explainability_surface_id,
        "event_id": event_id,
        "content_hash": hash_value,
        "created_at": created_at,
        "surface": surface,
    }


def get_latest_explainability_surface(event_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM explainability_surfaces
            WHERE event_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM explainability_surfaces
            WHERE event_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (event_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["surface"] = json.loads(result["surface_json"])
    return result


def create_audit_evidence_package(subject_ref: str, package: Dict[str, Any]) -> Dict[str, Any]:
    return _persist_json_artifact(
        table="audit_evidence_packages",
        id_column="audit_evidence_package_id",
        id_prefix="aep",
        payload_column="package_json",
        subject_ref=subject_ref,
        payload=package,
    )


def create_external_audit_export(subject_ref: str, export_payload: Dict[str, Any]) -> Dict[str, Any]:
    return _persist_json_artifact(
        table="external_audit_exports",
        id_column="external_audit_export_id",
        id_prefix="auditx",
        payload_column="export_json",
        subject_ref=subject_ref,
        payload=export_payload,
        lineage=export_payload.get("lineage_chain"),
    )


def create_replay_binding_audit_export(event_id: str) -> Dict[str, Any]:
    binding = get_replay_binding(event_id)
    if binding is None:
        raise KeyError(f"replay_binding_not_found:{event_id}")
    verification = verify_replay_binding(event_id)
    lineage_chain = [
        {
            "event_id": event_id,
            "projection_context_ref": binding.get("projection_context_ref"),
            "projection_context_hash": binding.get("projection_context_hash"),
            "authority_source_manifest_hash": binding.get("authority_source_manifest_hash"),
        }
    ]
    export_payload = {
        "schema_version": "1.0",
        "exported_at": now_iso(),
        "exported_by": "system",
        "event_id": event_id,
        "knowledge_version": binding.get("knowledge_version"),
        "projection_bundle_version": binding.get("projection_bundle_version"),
        "constraint_bundle_version": binding.get("constraint_bundle_version"),
        "projection_context_ref": binding.get("projection_context_ref"),
        "projection_context_hash": binding.get("projection_context_hash"),
        "authority_source_manifest": binding.get("authority_source_manifest_json", []),
        "authority_source_manifest_hash": binding.get("authority_source_manifest_hash"),
        "lineage_chain": lineage_chain,
        "replay_verification": verification,
    }
    export_payload["bundle_hash"] = content_hash(export_payload)
    return create_external_audit_export(event_id, export_payload)


def create_external_verification_manifest(
    subject_ref: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return _persist_json_artifact(
        table="external_verification_manifests",
        id_column="external_verification_manifest_id",
        id_prefix="vman",
        payload_column="manifest_json",
        subject_ref=subject_ref,
        payload=manifest,
    )


def create_verification_contract(
    subject_ref: str,
    contract: Dict[str, Any],
    *,
    status: str = "ACTIVE",
) -> Dict[str, Any]:
    activated_at = now_iso() if status == "ACTIVE" else None
    return _persist_json_artifact(
        table="verification_contracts",
        id_column="verification_contract_id",
        id_prefix="vcontract",
        payload_column="contract_json",
        subject_ref=subject_ref,
        payload=contract,
        status=status,
        activated_at=activated_at,
    )


def get_latest_verification_contract(subject_ref: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM verification_contracts
            WHERE subject_ref = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM verification_contracts
            WHERE subject_ref = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (subject_ref,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["contract"] = json.loads(result["contract_json"])
    return result
