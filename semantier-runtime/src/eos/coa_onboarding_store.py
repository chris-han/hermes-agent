"""v7.7 COA onboarding candidate artifacts and activation helpers.

This module introduces explicit onboarding candidates:
- COATaxonomyCandidate_t
- ProjectionBundleCandidate_t
- ConstraintBundleCandidate_t

Candidates move through CANDIDATE -> APPROVED -> ACTIVE.
Activation publishes versioned COA_v / Pi_v artifacts through the existing
proposal machinery in semantier.storage.coa_store.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso
from storage.coa_store import (
    activate_coa_change_proposal,
    activate_projection_rule_proposal,
    approve_coa_change_proposal,
    approve_projection_rule_proposal,
    create_coa_change_proposal,
    create_projection_rule_proposal,
)


def _insert_candidate(
    *,
    table: str,
    id_column: str,
    id_prefix: str,
    org_id: str,
    payload: Dict[str, Any],
    proposed_by: Optional[str],
) -> Dict[str, Any]:
    candidate_id = f"{id_prefix}_{uuid4().hex}"
    created_at = now_iso()
    payload_json = canonical_json(payload)
    hash_value = content_hash(payload)

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query(f"""
            INSERT INTO {table}
              ({id_column}, org_id, candidate_json, content_hash, status,
               proposed_by, approved_by, created_at, approved_at, activated_at)
            VALUES (?, ?, ?, ?, 'CANDIDATE', ?, NULL, ?, NULL, NULL)
            """, f"\n            INSERT INTO {table}\n              ({id_column}, org_id, candidate_json, content_hash, status,\n               proposed_by, approved_by, created_at, approved_at, activated_at)\n            VALUES (%s, %s, %s, %s, 'CANDIDATE', %s, NULL, %s, NULL, NULL)\n            "),
            (candidate_id, org_id, payload_json, hash_value, proposed_by, created_at),
        )

    return {
        id_column: candidate_id,
        "org_id": org_id,
        "status": "CANDIDATE",
        "content_hash": hash_value,
        "created_at": created_at,
        "candidate": payload,
    }


def _load_candidate(table: str, id_column: str, candidate_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query(f"SELECT * FROM {table} WHERE {id_column} = ?", f'SELECT * FROM {table} WHERE {id_column} = %s'),
            (candidate_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"onboarding_candidate_not_found:{table}:{candidate_id}")
    result = dict(row)
    result["candidate"] = json.loads(result["candidate_json"])
    return result


def _approve_candidate(
    *,
    table: str,
    id_column: str,
    candidate_id: str,
    approved_by: str,
) -> Dict[str, Any]:
    row = _load_candidate(table, id_column, candidate_id)
    if row["status"] != "CANDIDATE":
        raise ValueError(f"onboarding_candidate_not_pending:{candidate_id}:{row['status']}")

    approved_at = now_iso()
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query(f"""
            UPDATE {table}
            SET status = 'APPROVED', approved_by = ?, approved_at = ?
            WHERE {id_column} = ?
            """, f"\n            UPDATE {table}\n            SET status = 'APPROVED', approved_by = %s, approved_at = %s\n            WHERE {id_column} = %s\n            "),
            (approved_by, approved_at, candidate_id),
        )

    row["status"] = "APPROVED"
    row["approved_by"] = approved_by
    row["approved_at"] = approved_at
    return row


def _activate_candidate(
    *,
    table: str,
    id_column: str,
    candidate_id: str,
) -> Dict[str, Any]:
    row = _load_candidate(table, id_column, candidate_id)
    if row["status"] != "APPROVED":
        raise ValueError(f"onboarding_candidate_requires_approval:{candidate_id}:{row['status']}")

    activated_at = now_iso()
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query(f"""
            UPDATE {table}
            SET status = 'ACTIVE', activated_at = ?
            WHERE {id_column} = ?
            """, f"\n            UPDATE {table}\n            SET status = 'ACTIVE', activated_at = %s\n            WHERE {id_column} = %s\n            "),
            (activated_at, candidate_id),
        )

    row["status"] = "ACTIVE"
    row["activated_at"] = activated_at
    return row


def create_onboarding_candidates(
    *,
    org_id: str,
    coa_taxonomy: Dict[str, Any],
    projection_bundle: Dict[str, Any],
    constraint_bundle: Dict[str, Any],
    proposed_by: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Create COA/projection/constraint candidates for onboarding."""
    coa_payload = {
        "schema_version": "semantier.coa_taxonomy_candidate.v1",
        "org_id": org_id,
        **coa_taxonomy,
    }
    projection_payload = {
        "schema_version": "semantier.projection_bundle_candidate.v1",
        "org_id": org_id,
        **projection_bundle,
    }
    constraint_payload = {
        "schema_version": "semantier.constraint_bundle_candidate.v1",
        "org_id": org_id,
        **constraint_bundle,
    }

    coa = _insert_candidate(
        table="coa_taxonomy_candidates",
        id_column="coa_taxonomy_candidate_id",
        id_prefix="coacand",
        org_id=org_id,
        payload=coa_payload,
        proposed_by=proposed_by,
    )
    projection = _insert_candidate(
        table="projection_bundle_candidates",
        id_column="projection_bundle_candidate_id",
        id_prefix="pbcand",
        org_id=org_id,
        payload=projection_payload,
        proposed_by=proposed_by,
    )
    constraint = _insert_candidate(
        table="constraint_bundle_candidates",
        id_column="constraint_bundle_candidate_id",
        id_prefix="cbcand",
        org_id=org_id,
        payload=constraint_payload,
        proposed_by=proposed_by,
    )

    return {
        "coa_taxonomy_candidate": coa,
        "projection_bundle_candidate": projection,
        "constraint_bundle_candidate": constraint,
    }


def approve_onboarding_candidates(
    *,
    coa_taxonomy_candidate_id: str,
    projection_bundle_candidate_id: str,
    constraint_bundle_candidate_id: str,
    approved_by: str,
) -> Dict[str, Dict[str, Any]]:
    """Approve all onboarding candidates before activation."""
    coa = _approve_candidate(
        table="coa_taxonomy_candidates",
        id_column="coa_taxonomy_candidate_id",
        candidate_id=coa_taxonomy_candidate_id,
        approved_by=approved_by,
    )
    projection = _approve_candidate(
        table="projection_bundle_candidates",
        id_column="projection_bundle_candidate_id",
        candidate_id=projection_bundle_candidate_id,
        approved_by=approved_by,
    )
    constraint = _approve_candidate(
        table="constraint_bundle_candidates",
        id_column="constraint_bundle_candidate_id",
        candidate_id=constraint_bundle_candidate_id,
        approved_by=approved_by,
    )
    return {
        "coa_taxonomy_candidate": coa,
        "projection_bundle_candidate": projection,
        "constraint_bundle_candidate": constraint,
    }


def activate_onboarding_candidates(
    *,
    org_id: str,
    coa_taxonomy_candidate_id: str,
    projection_bundle_candidate_id: str,
    constraint_bundle_candidate_id: str,
    activated_by: str,
) -> Dict[str, Any]:
    """Activate approved onboarding candidates and publish COA_v / Pi_v versions."""
    coa_candidate = _load_candidate(
        "coa_taxonomy_candidates", "coa_taxonomy_candidate_id", coa_taxonomy_candidate_id
    )
    projection_candidate = _load_candidate(
        "projection_bundle_candidates", "projection_bundle_candidate_id", projection_bundle_candidate_id
    )
    constraint_candidate = _load_candidate(
        "constraint_bundle_candidates", "constraint_bundle_candidate_id", constraint_bundle_candidate_id
    )

    for row, row_id in (
        (coa_candidate, coa_taxonomy_candidate_id),
        (projection_candidate, projection_bundle_candidate_id),
        (constraint_candidate, constraint_bundle_candidate_id),
    ):
        if row["status"] != "APPROVED":
            raise ValueError(f"onboarding_candidate_requires_approval:{row_id}:{row['status']}")

    coa_accounts = coa_candidate["candidate"].get("custom_accounts", [])
    coa_justification = coa_candidate["candidate"].get("justification", "COA onboarding activation")

    coa_proposal = create_coa_change_proposal(
        base_coa_version_id=None,
        proposed_accounts=coa_accounts,
        justification=coa_justification,
        proposed_by=activated_by,
    )
    approve_coa_change_proposal(coa_proposal["coa_change_proposal_id"], approved_by=activated_by)
    coa_version = activate_coa_change_proposal(coa_proposal["coa_change_proposal_id"], org_id=org_id)

    projection_rules = projection_candidate["candidate"].get("rules", [])
    projection_justification = projection_candidate["candidate"].get(
        "justification", "Projection bundle onboarding activation"
    )
    projection_proposal = create_projection_rule_proposal(
        base_projection_bundle_id=None,
        proposed_rules=projection_rules,
        justification=projection_justification,
        proposed_by=activated_by,
    )
    approve_projection_rule_proposal(
        projection_proposal["projection_rule_proposal_id"], approved_by=activated_by
    )
    projection_version = activate_projection_rule_proposal(
        projection_proposal["projection_rule_proposal_id"], org_id=org_id
    )

    _activate_candidate(
        table="coa_taxonomy_candidates",
        id_column="coa_taxonomy_candidate_id",
        candidate_id=coa_taxonomy_candidate_id,
    )
    _activate_candidate(
        table="projection_bundle_candidates",
        id_column="projection_bundle_candidate_id",
        candidate_id=projection_bundle_candidate_id,
    )
    constraint_row = _activate_candidate(
        table="constraint_bundle_candidates",
        id_column="constraint_bundle_candidate_id",
        candidate_id=constraint_bundle_candidate_id,
    )

    constraint_bundle_version = constraint_row["candidate"].get(
        "constraint_bundle_version", f"{org_id}:constraint:1"
    )

    return {
        "org_id": org_id,
        "activated_by": activated_by,
        "coa_version": coa_version,
        "projection_bundle_version": projection_version,
        "constraint_bundle_version": constraint_bundle_version,
        "resulting_version_pins": {
            "COA_v": coa_version["coa_version_id"],
            "Pi_v": projection_version["projection_bundle_id"],
            "ConstraintBundle_v": constraint_bundle_version,
        },
    }
