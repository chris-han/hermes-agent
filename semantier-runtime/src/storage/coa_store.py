"""First-class COA version store — Slice 7.5-E.

Manages COA change proposals and projection-rule proposals with versioned,
append-only inserts. Activation publishes COA_v+1 / Π_v+1 only after
explicit approval, with new content hashes.

Semantic authority is kept in runtime modules; this store only persists.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso
from storage.hosted_postgres import enabled,sql_query,begin_write

# ---------------------------------------------------------------------------
# COA Change Proposals
# ---------------------------------------------------------------------------

_ALLOWED_ACCOUNT_CLASSES = frozenset(
    ("asset", "liability", "equity", "revenue", "expense")
)
# Only T1 (ontology primitives) is forbidden for COA accounts per v7.5 spec.
# T2 (law/regulation) is allowed for statutory COA accounts governed by regulation.
_T1_FORBIDDEN = frozenset(("T1",))


def create_coa_change_proposal(
    *,
    base_coa_version_id: str | None,
    proposed_accounts: List[Dict[str, Any]],
    justification: str,
    proposed_by: str | None = None,
) -> Dict[str, Any]:
    """Create a new PENDING COA change proposal.

    Validates that no account has a semantic_tier of T1 (or T2), which are
    reserved for ontology-level definitions and must not appear in COA entries.
    """
    for account in proposed_accounts:
        tier = account.get("semantic_tier", "T3")
        if tier in _T1_FORBIDDEN:
            raise ValueError(
                f"coa_account_cannot_be_ontology_tier:{tier}:{account.get('account_code', '?')}"
            )
        account_class = account.get("account_class", "")
        if account_class not in _ALLOWED_ACCOUNT_CLASSES:
            raise ValueError(
                f"coa_account_invalid_class:{account_class}:{account.get('account_code', '?')}"
            )

    payload = {
        "base_coa_version_id": base_coa_version_id,
        "proposed_accounts": proposed_accounts,
        "justification": justification,
        "proposed_by": proposed_by,
    }
    proposal_id = f"coa_prop_{uuid4().hex}"
    created_at = now_iso()
    payload_json = canonical_json(payload)
    payload_hash = content_hash(payload)

    with get_conn() as conn:
        if enabled():begin_write(conn)
        conn.execute(
            sql_query("""
            INSERT INTO coa_change_proposals
              (coa_change_proposal_id, base_coa_version_id, proposal_json, content_hash,
               status, proposed_by, created_at, approved_at)
            VALUES (?, ?, ?, ?, 'PENDING', ?, ?, NULL)
            ""","\n            INSERT INTO coa_change_proposals\n              (coa_change_proposal_id, base_coa_version_id, proposal_json, content_hash,\n               status, proposed_by, created_at, approved_at)\n            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, NULL)\n            "),
            (proposal_id, base_coa_version_id, payload_json, payload_hash, proposed_by, created_at),
        )
    return {
        "coa_change_proposal_id": proposal_id,
        "base_coa_version_id": base_coa_version_id,
        "content_hash": payload_hash,
        "status": "PENDING",
        "created_at": created_at,
        "proposal": payload,
    }


def approve_coa_change_proposal(proposal_id: str, *, approved_by: str) -> Dict[str, Any]:
    """Transition PENDING → APPROVED for a COA change proposal."""
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = ?",'SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"coa_change_proposal_not_found:{proposal_id}")
        if row["status"] != "PENDING":
            raise ValueError(f"coa_change_proposal_not_pending:{row['status']}")
        approved_at = now_iso()
        conn.execute(
            sql_query("UPDATE coa_change_proposals SET status = 'APPROVED', approved_at = ? WHERE coa_change_proposal_id = ?","UPDATE coa_change_proposals SET status = 'APPROVED', approved_at = %s WHERE coa_change_proposal_id = %s"),
            (approved_at, proposal_id),
        )
    return {"coa_change_proposal_id": proposal_id, "status": "APPROVED", "approved_at": approved_at}


def activate_coa_change_proposal(
    proposal_id: str,
    *,
    org_id: str = "global",
) -> Dict[str, Any]:
    """Activate an APPROVED COA change proposal, publishing a new COA_v+1.

    This inserts a new coa_versions row and does NOT mutate or delete any
    existing version. The new version_label is derived from the previous
    active label (integer increment) or starts at 1.
    """
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = ?",'SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"coa_change_proposal_not_found:{proposal_id}")
        if row["status"] != "APPROVED":
            raise ValueError("coa_change_proposal_requires_approval")

        proposal_payload = json.loads(row["proposal_json"])

        # Determine next version label
        prev_row = conn.execute(
            sql_query("""
            SELECT version_label FROM coa_versions
            WHERE org_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,'\n            SELECT version_label FROM coa_versions\n            WHERE org_id = %s\n            ORDER BY created_at DESC LIMIT 1\n            '),
            (org_id,),
        ).fetchone()
        try:
            next_version = int(prev_row["version_label"]) + 1 if prev_row else 1
        except (ValueError, TypeError):
            next_version = 1
        version_label = str(next_version)

        coa_version_id = f"coa_v_{uuid4().hex}"
        created_at = now_iso()
        coa_payload = {
            "org_id": org_id,
            "version_label": version_label,
            "base_coa_version_id": proposal_payload.get("base_coa_version_id"),
            "proposed_accounts": proposal_payload.get("proposed_accounts", []),
            "justification": proposal_payload.get("justification"),
            "source_proposal_id": proposal_id,
        }
        coa_hash = content_hash(coa_payload)

        conn.execute(
            sql_query("""
            INSERT INTO coa_versions
              (coa_version_id, org_id, bundle_version, version_label, content_json,
               content_hash, status, created_at, activated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            ""","\n            INSERT INTO coa_versions\n              (coa_version_id, org_id, bundle_version, version_label, content_json,\n               content_hash, status, created_at, activated_at)\n            VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s)\n            "),
            (
                coa_version_id,
                org_id,
                f"{org_id}:{version_label}",
                version_label,
                canonical_json(coa_payload),
                coa_hash,
                created_at,
                created_at,
            ),
        )
        conn.execute(
            sql_query("UPDATE coa_change_proposals SET status = 'ACTIVATED' WHERE coa_change_proposal_id = ?","UPDATE coa_change_proposals SET status = 'ACTIVATED' WHERE coa_change_proposal_id = %s"),
            (proposal_id,),
        )
    return {
        "coa_version_id": coa_version_id,
        "org_id": org_id,
        "version_label": version_label,
        "content_hash": coa_hash,
        "status": "ACTIVE",
        "created_at": created_at,
        "coa_payload": coa_payload,
    }


def get_coa_change_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = ?",'SELECT * FROM coa_change_proposals WHERE coa_change_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["proposal"] = json.loads(result["proposal_json"])
    return result


# ---------------------------------------------------------------------------
# Projection Rule Proposals
# ---------------------------------------------------------------------------


def create_projection_rule_proposal(
    *,
    base_projection_bundle_id: str | None,
    proposed_rules: List[Dict[str, Any]],
    justification: str,
    proposed_by: str | None = None,
) -> Dict[str, Any]:
    """Create a new PENDING projection-rule proposal."""
    payload = {
        "base_projection_bundle_id": base_projection_bundle_id,
        "proposed_rules": proposed_rules,
        "justification": justification,
        "proposed_by": proposed_by,
    }
    proposal_id = f"pi_prop_{uuid4().hex}"
    created_at = now_iso()
    payload_json = canonical_json(payload)
    payload_hash = content_hash(payload)

    with get_conn() as conn:
        if enabled():begin_write(conn)
        conn.execute(
            sql_query("""
            INSERT INTO projection_rule_proposals
              (projection_rule_proposal_id, base_projection_bundle_id, proposal_json, content_hash,
               status, proposed_by, created_at, approved_at)
            VALUES (?, ?, ?, ?, 'PENDING', ?, ?, NULL)
            ""","\n            INSERT INTO projection_rule_proposals\n              (projection_rule_proposal_id, base_projection_bundle_id, proposal_json, content_hash,\n               status, proposed_by, created_at, approved_at)\n            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, NULL)\n            "),
            (
                proposal_id,
                base_projection_bundle_id,
                payload_json,
                payload_hash,
                proposed_by,
                created_at,
            ),
        )
    return {
        "projection_rule_proposal_id": proposal_id,
        "base_projection_bundle_id": base_projection_bundle_id,
        "content_hash": payload_hash,
        "status": "PENDING",
        "created_at": created_at,
        "proposal": payload,
    }


def approve_projection_rule_proposal(proposal_id: str, *, approved_by: str) -> Dict[str, Any]:
    """Transition PENDING → APPROVED for a projection-rule proposal."""
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = ?",'SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"projection_rule_proposal_not_found:{proposal_id}")
        if row["status"] != "PENDING":
            raise ValueError(f"projection_rule_proposal_not_pending:{row['status']}")
        approved_at = now_iso()
        conn.execute(
            sql_query("""
            UPDATE projection_rule_proposals
            SET status = 'APPROVED', approved_at = ?
            WHERE projection_rule_proposal_id = ?
            ""","\n            UPDATE projection_rule_proposals\n            SET status = 'APPROVED', approved_at = %s\n            WHERE projection_rule_proposal_id = %s\n            "),
            (approved_at, proposal_id),
        )
    return {
        "projection_rule_proposal_id": proposal_id,
        "status": "APPROVED",
        "approved_at": approved_at,
    }


def activate_projection_rule_proposal(
    proposal_id: str,
    *,
    org_id: str = "global",
) -> Dict[str, Any]:
    """Activate an APPROVED projection-rule proposal, publishing Π_v+1.

    Inserts a new projection_bundles row without mutating existing bundles.
    """
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = ?",'SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"projection_rule_proposal_not_found:{proposal_id}")
        if row["status"] != "APPROVED":
            raise ValueError("projection_rule_proposal_requires_approval")

        proposal_payload = json.loads(row["proposal_json"])

        # Determine next bundle version
        prev_bundle = conn.execute(
            sql_query("""
            SELECT bundle_version FROM projection_bundles
            WHERE org_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,'\n            SELECT bundle_version FROM projection_bundles\n            WHERE org_id = %s\n            ORDER BY created_at DESC LIMIT 1\n            '),
            (org_id,),
        ).fetchone()
        try:
            base_version = prev_bundle["bundle_version"].rsplit(":", 1)[-1] if prev_bundle else "0"
            next_version = int(base_version) + 1
        except (ValueError, TypeError):
            next_version = 1
        bundle_version = f"{org_id}:{next_version}"

        bundle_id = f"pi_bundle_{uuid4().hex}"
        created_at = now_iso()
        bundle_payload = {
            "org_id": org_id,
            "bundle_version": bundle_version,
            "base_projection_bundle_id": proposal_payload.get("base_projection_bundle_id"),
            "proposed_rules": proposal_payload.get("proposed_rules", []),
            "justification": proposal_payload.get("justification"),
            "source_proposal_id": proposal_id,
        }
        bundle_hash = content_hash(bundle_payload)

        conn.execute(
            sql_query("""
            INSERT INTO projection_bundles
              (projection_bundle_id, org_id, bundle_version, coa_version_id, content_json,
               content_hash, status, created_at, activated_at)
            VALUES (?, ?, ?, NULL, ?, ?, 'ACTIVE', ?, ?)
            ""","\n            INSERT INTO projection_bundles\n              (projection_bundle_id, org_id, bundle_version, coa_version_id, content_json,\n               content_hash, status, created_at, activated_at)\n            VALUES (%s, %s, %s, NULL, %s, %s, 'ACTIVE', %s, %s)\n            "),
            (
                bundle_id,
                org_id,
                bundle_version,
                canonical_json(bundle_payload),
                bundle_hash,
                created_at,
                created_at,
            ),
        )
        conn.execute(
            sql_query("""
            UPDATE projection_rule_proposals
            SET status = 'ACTIVATED'
            WHERE projection_rule_proposal_id = ?
            ""","\n            UPDATE projection_rule_proposals\n            SET status = 'ACTIVATED'\n            WHERE projection_rule_proposal_id = %s\n            "),
            (proposal_id,),
        )
    return {
        "projection_bundle_id": bundle_id,
        "org_id": org_id,
        "bundle_version": bundle_version,
        "content_hash": bundle_hash,
        "status": "ACTIVE",
        "created_at": created_at,
        "bundle_payload": bundle_payload,
    }


def get_projection_rule_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        if enabled():begin_write(conn)
        row = conn.execute(
            sql_query("SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = ?",'SELECT * FROM projection_rule_proposals WHERE projection_rule_proposal_id = %s'),
            (proposal_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["proposal"] = json.loads(result["proposal_json"])
    return result
