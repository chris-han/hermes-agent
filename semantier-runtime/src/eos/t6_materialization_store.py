from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_hash
from eos.db import get_conn, init_db, now_iso
from eos.replay_binding_store import create_replay_binding


def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    for key in list(data.keys()):
        if key.endswith("_json") and data[key] is not None:
            try:
                data[key] = json.loads(data[key])
            except (TypeError, json.JSONDecodeError):
                pass
    return data


def _candidate_hash(candidate_json: Dict[str, Any], explicit_hash: Optional[str]) -> str:
    if explicit_hash:
        return explicit_hash
    return compute_hash(candidate_json)


def _compute_transition_hash(
    candidate_id: str,
    from_state: Optional[str],
    to_state: str,
    transition_reason: str,
    actor_type: str,
    actor_id: Optional[str],
    created_at: str,
    prior_event_hash: Optional[str],
) -> str:
    payload = {
        "candidate_id": candidate_id,
        "from_state": from_state,
        "to_state": to_state,
        "transition_reason": transition_reason,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "created_at": created_at,
        "prior_event_hash": prior_event_hash,
    }
    return compute_hash(payload)


def _latest_transition_event_hash(conn, candidate_id: str) -> Optional[str]:
    row = conn.execute(
        hosted.sql_query("""
        SELECT event_hash
        FROM t6_materialization_transitions
        WHERE candidate_id = ?
        ORDER BY created_at DESC, transition_id DESC
        LIMIT 1
        """, """
        SELECT event_hash
        FROM t6_materialization_transitions
        WHERE candidate_id = %s
        ORDER BY created_at DESC, transition_id DESC
        LIMIT 1
        """),
        (candidate_id,),
    ).fetchone()
    return row["event_hash"] if row is not None else None


def _append_transition(
    conn,
    *,
    candidate_id: str,
    from_state: Optional[str],
    to_state: str,
    transition_reason: str,
    actor_type: str,
    actor_id: Optional[str],
    created_at: str,
) -> Dict[str, str]:
    transition_id = f"t6t_{uuid4().hex}"
    prior_event_hash = _latest_transition_event_hash(conn, candidate_id)
    event_hash = _compute_transition_hash(
        candidate_id=candidate_id,
        from_state=from_state,
        to_state=to_state,
        transition_reason=transition_reason,
        actor_type=actor_type,
        actor_id=actor_id,
        created_at=created_at,
        prior_event_hash=prior_event_hash,
    )
    conn.execute(
        hosted.sql_query("""
        INSERT INTO t6_materialization_transitions (
          transition_id,
          candidate_id,
          from_state,
          to_state,
          transition_reason,
          actor_type,
          actor_id,
          prior_event_hash,
          event_hash,
          created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, """
        INSERT INTO t6_materialization_transitions (
          transition_id,
          candidate_id,
          from_state,
          to_state,
          transition_reason,
          actor_type,
          actor_id,
          prior_event_hash,
          event_hash,
          created_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """),
        (
            transition_id,
            candidate_id,
            from_state,
            to_state,
            transition_reason,
            actor_type,
            actor_id,
            prior_event_hash,
            event_hash,
            created_at,
        ),
    )
    return {"transition_id": transition_id, "event_hash": event_hash}


def _insert_decision(
    conn,
    *,
    candidate_id: str,
    decision_type: str,
    decision_scope: str,
    actor_type: str,
    actor_id: Optional[str],
    reason_json: Dict[str, Any],
    created_at: str,
) -> str:
    decision_id = f"t6d_{uuid4().hex}"
    conn.execute(
        hosted.sql_query("""
        INSERT INTO t6_materialization_decisions (
          decision_id,
          candidate_id,
          decision_type,
          decision_scope,
          actor_type,
          actor_id,
          reason_json,
          created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """, """
        INSERT INTO t6_materialization_decisions (
          decision_id,
          candidate_id,
          decision_type,
          decision_scope,
          actor_type,
          actor_id,
          reason_json,
          created_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """),
        (
            decision_id,
            candidate_id,
            decision_type,
            decision_scope,
            actor_type,
            actor_id,
            canonical_json(reason_json),
            created_at,
        ),
    )
    return decision_id


def _latest_outcome_row(conn, candidate_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        hosted.sql_query("""
        SELECT *
        FROM t6_materialization_outcomes
        WHERE candidate_id = ?
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT 1
        """, """
        SELECT *
        FROM t6_materialization_outcomes
        WHERE candidate_id = %s
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT 1
        """),
        (candidate_id,),
    ).fetchone()
    return _row_to_dict(row)


def list_outcomes(candidate_id: str) -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM t6_materialization_outcomes
            WHERE candidate_id = ?
            ORDER BY created_at ASC, outcome_id ASC
            """, """
            SELECT *
            FROM t6_materialization_outcomes
            WHERE candidate_id = %s
            ORDER BY created_at ASC, outcome_id ASC
            """),
            (candidate_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _replay_binding_key_hash(
    *,
    rea_id: str,
    projection_context_hash: str,
    projection_version: str,
    policy_bundle_hash: str,
) -> str:
    return compute_hash(
        {
            "rea_id": rea_id,
            "projection_context_hash": projection_context_hash,
            "projection_version": projection_version,
            "policy_bundle_hash": policy_bundle_hash,
        }
    )


def validate_outcome_chain(candidate_id: str) -> Dict[str, Any]:
    """Validate outcome chain and return deterministic canonical head resolution."""

    outcomes = list_outcomes(candidate_id)
    if not outcomes:
        return {"status": "EMPTY", "candidate_id": candidate_id}

    by_id = {row["outcome_id"]: row for row in outcomes}
    referenced = set()
    roots = []
    for row in outcomes:
        parent = row.get("supersedes_outcome_id")
        if parent is None:
            roots.append(row["outcome_id"])
        else:
            if parent not in by_id:
                return {
                    "status": "CHAIN_INTEGRITY_FAILURE",
                    "reason": "missing_parent",
                    "candidate_id": candidate_id,
                    "outcome_id": row["outcome_id"],
                    "missing_parent": parent,
                }
            referenced.add(parent)

    if len(roots) != 1:
        return {
            "status": "CHAIN_INTEGRITY_FAILURE",
            "reason": "invalid_root_count",
            "candidate_id": candidate_id,
            "root_count": len(roots),
        }

    # Cycle check via parent traversal.
    for outcome_id in by_id:
        seen = set()
        current = outcome_id
        while current is not None:
            if current in seen:
                return {
                    "status": "CHAIN_INTEGRITY_FAILURE",
                    "reason": "cycle_detected",
                    "candidate_id": candidate_id,
                    "outcome_id": outcome_id,
                }
            seen.add(current)
            current = by_id[current].get("supersedes_outcome_id")

    heads = [row for row in outcomes if row["outcome_id"] not in referenced]
    heads_sorted = sorted(
        heads,
        key=lambda row: (row["created_at"], row["outcome_id"]),
    )
    canonical_head = heads_sorted[-1]
    return {
        "status": "OK",
        "candidate_id": candidate_id,
        "root_outcome_id": roots[0],
        "head_count": len(heads_sorted),
        "head_outcome_ids": [row["outcome_id"] for row in heads_sorted],
        "canonical_head_outcome_id": canonical_head["outcome_id"],
    }


def resolve_failure_recipient(
    *,
    mode: str,
    rea_submitter_user_id: Optional[str],
    role_authorized_handler_user_id: Optional[str],
) -> Dict[str, Any]:
    """Resolve user-facing failure routing target per mode contract."""

    normalized = mode.strip().upper()
    if normalized == "AUTO":
        return {
            "mode": "AUTO",
            "recipient_user_id": rea_submitter_user_id,
            "recipient_role": "REA_SUBMITTER",
        }
    if normalized in {"H-I-L", "HIL", "HUMAN_IN_THE_LOOP"}:
        return {
            "mode": "H-I-L",
            "recipient_user_id": role_authorized_handler_user_id,
            "recipient_role": "ROLE_AUTHORIZED_HANDLER",
        }
    raise ValueError("mode must be AUTO or H-I-L")


def get_candidate(candidate_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
    return _row_to_dict(row)


def list_candidate_transitions(candidate_id: str) -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_transitions WHERE candidate_id = ? ORDER BY created_at ASC, transition_id ASC", 'SELECT * FROM t6_materialization_transitions WHERE candidate_id = %s ORDER BY created_at ASC, transition_id ASC'),
            (candidate_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def list_candidates_for_org(*, org_id: str) -> List[Dict[str, Any]]:
    """List candidates for a trusted organization without exposing raw SQL to plugins."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE org_id = ? ORDER BY created_at DESC", 'SELECT * FROM t6_materialization_candidates WHERE org_id = %s ORDER BY created_at DESC'),
            (org_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_candidate_by_source_identity(*, org_id: str, source_channel: str, source_message_id: str) -> Optional[Dict[str, Any]]:
    """Resolve a candidate identity without allowing plugin SQL access."""
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE org_id = ? AND source_channel = ? AND source_message_id = ? ORDER BY created_at DESC LIMIT 1", 'SELECT * FROM t6_materialization_candidates WHERE org_id = %s AND source_channel = %s AND source_message_id = %s ORDER BY created_at DESC LIMIT 1'),
            (org_id, source_channel, source_message_id),
        ).fetchone()
    return _row_to_dict(row)


def defer_candidate(*, candidate_id: str, actor_id: str, reason: str, trigger: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Append a deferred review outcome while retaining C1_VALIDATED maturity."""
    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'), (candidate_id,)).fetchone()
        if row is None: raise ValueError(f"candidate not found: {candidate_id}")
        if row["fact_maturity_stage"] != "C1_VALIDATED": raise ValueError("DEFERRED_REQUIRES_C1_VALIDATED")
        _insert_decision(conn, candidate_id=candidate_id, decision_type="DEFERRED", decision_scope="FACT_ADMISSION", actor_type="human", actor_id=actor_id, reason_json={"reason": reason, "trigger": trigger or {}}, created_at=now)
        transition = _append_transition(conn, candidate_id=candidate_id, from_state="C1_VALIDATED", to_state="C1_VALIDATED", transition_reason="DEFERRED", actor_type="human", actor_id=actor_id, created_at=now)
        conn.commit()
    return {"result": "DEFERRED", "candidate_id": candidate_id, "candidate": get_candidate(candidate_id), "transition": transition, "trigger": trigger or {}}


def record_radar_operator_outcome(
    *, candidate_id: str, organization_id: str, workspace_id: str | None,
    outcome_type: str, reason: str, evidence_refs: list[str], case_ref: str | None,
    actor_id: str,
) -> Dict[str, Any]:
    """Append a case-scoped operator outcome without changing T6 maturity."""
    allowed = {"qualified", "deferred", "rejected", "false_positive", "contacted", "diligence_opened"}
    if outcome_type not in allowed:
        raise ValueError("RADAR_OPERATOR_OUTCOME_INVALID")
    if not candidate_id or not organization_id or not actor_id or not reason:
        raise ValueError("RADAR_OPERATOR_OUTCOME_INCOMPLETE")
    init_db()
    now = now_iso()
    outcome_id = f"radar_outcome_{uuid4().hex}"
    payload = {
        "candidate_id": candidate_id, "organization_id": organization_id,
        "workspace_id": workspace_id, "outcome_type": outcome_type,
        "reason": reason, "evidence_refs": list(evidence_refs),
        "case_ref": case_ref, "actor_id": actor_id,
    }
    digest = compute_hash(payload)
    with get_conn() as conn:
        candidate = conn.execute(
            hosted.sql_query("SELECT org_id FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT org_id FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate is None or candidate["org_id"] != organization_id:
            raise KeyError("OPPORTUNITY_CANDIDATE_NOT_FOUND")
        conn.execute(
            hosted.sql_query("""INSERT INTO investment_radar_operator_outcomes
            (outcome_id, candidate_id, organization_id, workspace_id, outcome_type,
             reason, evidence_refs_json, case_ref, actor_id, created_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", """INSERT INTO investment_radar_operator_outcomes
            (outcome_id, candidate_id, organization_id, workspace_id, outcome_type,
             reason, evidence_refs_json, case_ref, actor_id, created_at, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""),
            (outcome_id, candidate_id, organization_id, workspace_id, outcome_type,
             reason, canonical_json(list(evidence_refs)), case_ref, actor_id, now, digest),
        )
        conn.commit()
    return {"outcome_id": outcome_id, **payload, "content_hash": digest, "created_at": now}


def list_radar_operator_outcomes(*, candidate_id: str, organization_id: str) -> list[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM investment_radar_operator_outcomes WHERE candidate_id = ? AND organization_id = ? ORDER BY created_at ASC, outcome_id ASC", 'SELECT * FROM investment_radar_operator_outcomes WHERE candidate_id = %s AND organization_id = %s ORDER BY created_at ASC, outcome_id ASC'),
            (candidate_id, organization_id),
        ).fetchall()
    result = []
    for row in rows:
        item = _row_to_dict(row)
        raw_refs = item.pop("evidence_refs_json", [])
        item["evidence_refs"] = raw_refs if isinstance(raw_refs, list) else json.loads(raw_refs or "[]")
        result.append(item)
    return result


def record_radar_notification(*, candidate_id: str, organization_id: str,
                              workspace_id: str | None, notification_type: str,
                              payload: Dict[str, Any]) -> Dict[str, Any]:
    if notification_type not in {"candidate_admitted", "candidate_materially_changed"}:
        raise ValueError("RADAR_NOTIFICATION_TYPE_INVALID")
    init_db()
    digest = compute_hash({"candidate_id": candidate_id, "organization_id": organization_id, "notification_type": notification_type, "payload": payload})
    notification_id = f"radar_notification_{uuid4().hex}"
    now = now_iso()
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""INSERT OR IGNORE INTO investment_radar_notifications
            (notification_id, candidate_id, organization_id, workspace_id,
             notification_type, content_hash, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", """INSERT INTO investment_radar_notifications
            (notification_id, candidate_id, organization_id, workspace_id,
             notification_type, content_hash, payload_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (notification_id, candidate_id, organization_id, workspace_id,
             notification_type, digest, canonical_json(payload), now),
        )
        conn.commit()
    return {"notification_id": notification_id, "candidate_id": candidate_id, "organization_id": organization_id, "notification_type": notification_type, "content_hash": digest, "created_at": now}


def submit_candidate(
    *,
    org_id: str,
    source_channel: str,
    source_message_id: str,
    candidate_json: Dict[str, Any],
    workspace_id: Optional[str] = None,
    source_session_id: Optional[str] = None,
    validation_status: Optional[str] = None,
    policy_bundle_version: Optional[str] = None,
    projection_context_ref: Optional[str] = None,
    projection_context_hash: Optional[str] = None,
    candidate_hash: Optional[str] = None,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Submit a T6 materialization candidate with dedupe + transition append.

    Dedupe key: (org_id, source_channel, source_message_id, candidate_hash).
    """

    if not source_message_id:
        raise ValueError("source_message_id is required for deterministic submit dedupe")

    init_db()
    now = now_iso()
    resolved_hash = _candidate_hash(candidate_json, candidate_hash)

    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM t6_materialization_candidates
            WHERE org_id = ?
              AND source_channel = ?
              AND source_message_id = ?
              AND candidate_hash = ?
            """, """
            SELECT *
            FROM t6_materialization_candidates
            WHERE org_id = %s
              AND source_channel = %s
              AND source_message_id = %s
              AND candidate_hash = %s
            """),
            (org_id, source_channel, source_message_id, resolved_hash),
        ).fetchone()
        if existing is not None:
            existing_candidate = _row_to_dict(existing)
            return {
                "result": "DUPLICATE",
                "candidate_id": existing_candidate["candidate_id"],
                "candidate": existing_candidate,
            }

        candidate_id = f"t6c_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO t6_materialization_candidates (
              candidate_id,
              org_id,
              workspace_id,
              source_channel,
              source_session_id,
              source_message_id,
              candidate_json,
              candidate_hash,
              validation_status,
              fact_maturity_stage,
              projection_status,
              policy_bundle_version,
              projection_context_ref,
              projection_context_hash,
              created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, """
            INSERT INTO t6_materialization_candidates (
              candidate_id,
              org_id,
              workspace_id,
              source_channel,
              source_session_id,
              source_message_id,
              candidate_json,
              candidate_hash,
              validation_status,
              fact_maturity_stage,
              projection_status,
              policy_bundle_version,
              projection_context_ref,
              projection_context_hash,
              created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """),
            (
                candidate_id,
                org_id,
                workspace_id,
                source_channel,
                source_session_id,
                source_message_id,
                canonical_json(candidate_json),
                resolved_hash,
                validation_status,
                "C0_CAPTURED",
                "PROJECTION_PENDING",
                policy_bundle_version,
                projection_context_ref,
                projection_context_hash,
                now,
            ),
        )

        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state=None,
            to_state="C0_CAPTURED",
            transition_reason="SUBMIT_CANDIDATE",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    created_candidate = get_candidate(candidate_id)
    return {
        "result": "SUBMITTED",
        "candidate_id": candidate_id,
        "candidate": created_candidate,
        "transition": transition,
    }


def validate_candidate(
    *,
    candidate_id: str,
    allow: bool,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Apply VALIDATE_ALLOW or VALIDATE_REJECT from C0_CAPTURED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        candidate_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"candidate not found: {candidate_id}")

        candidate = _row_to_dict(candidate_row)
        current_state = candidate["fact_maturity_stage"]
        if current_state != "C0_CAPTURED":
            raise ValueError(
                f"invalid transition: VALIDATE_* requires C0_CAPTURED, got {current_state}"
            )

        to_state = "C1_VALIDATED" if allow else "REJECTED"
        transition_reason = "VALIDATE_ALLOW" if allow else "VALIDATE_REJECT"
        validation_status = "ALLOW" if allow else "REJECT"

        conn.execute(
            hosted.sql_query("""
            UPDATE t6_materialization_candidates
            SET fact_maturity_stage = ?, validation_status = ?
            WHERE candidate_id = ?
            """, """
            UPDATE t6_materialization_candidates
            SET fact_maturity_stage = %s, validation_status = %s
            WHERE candidate_id = %s
            """),
            (to_state, validation_status, candidate_id),
        )

        if not allow:
            _insert_decision(
                conn,
                candidate_id=candidate_id,
                decision_type="SYSTEM_REJECT",
                decision_scope="FACT_ADMISSION",
                actor_type=actor_type,
                actor_id=actor_id,
                reason_json={"reason": "validation_reject"},
                created_at=now,
            )

        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state=current_state,
            to_state=to_state,
            transition_reason=transition_reason,
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": transition_reason,
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def auto_allow_candidate(
    *,
    candidate_id: str,
    decision_scope: str = "FACT_ADMISSION",
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Apply AUTO_ALLOW from C1_VALIDATED to C2_APPROVED."""

    if decision_scope not in {"FACT_ADMISSION", "PROJECTION_TRUST", "BOTH"}:
        raise ValueError("decision_scope must be FACT_ADMISSION, PROJECTION_TRUST, or BOTH")

    init_db()
    now = now_iso()
    with get_conn() as conn:
        candidate_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(candidate_row)
        current_state = candidate["fact_maturity_stage"]

        existing_decision = conn.execute(
            hosted.sql_query("""
            SELECT decision_id
            FROM t6_materialization_decisions
            WHERE candidate_id = ?
              AND decision_type = 'AUTO_ALLOW'
              AND decision_scope = ?
              AND COALESCE(actor_id, '') = COALESCE(?, '')
            ORDER BY created_at DESC, decision_id DESC
            LIMIT 1
            """, """
            SELECT decision_id
            FROM t6_materialization_decisions
            WHERE candidate_id = %s
              AND decision_type = 'AUTO_ALLOW'
              AND decision_scope = %s
              AND COALESCE(actor_id, '') = COALESCE(%s, '')
            ORDER BY created_at DESC, decision_id DESC
            LIMIT 1
            """),
            (candidate_id, decision_scope, actor_id),
        ).fetchone()
        if current_state == "C2_APPROVED" and existing_decision is not None:
            return {
                "result": "DUPLICATE",
                "candidate_id": candidate_id,
                "candidate": candidate,
                "decision_id": existing_decision["decision_id"],
            }
        if current_state != "C1_VALIDATED":
            raise ValueError(
                f"invalid transition: AUTO_ALLOW requires C1_VALIDATED, got {current_state}"
            )

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="AUTO_ALLOW",
            decision_scope=decision_scope,
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "auto_policy_allow"},
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("""
            UPDATE t6_materialization_candidates
            SET fact_maturity_stage = ?
            WHERE candidate_id = ?
            """, """
            UPDATE t6_materialization_candidates
            SET fact_maturity_stage = %s
            WHERE candidate_id = %s
            """),
            ("C2_APPROVED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state=current_state,
            to_state="C2_APPROVED",
            transition_reason="AUTO_ALLOW",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "AUTO_ALLOW",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def auto_deny_to_queue(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Auto policy deny path that falls back to approval queue from C1_VALIDATED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        if candidate["fact_maturity_stage"] != "C1_VALIDATED":
            raise ValueError("invalid transition: AUTO_DENY_TO_QUEUE requires C1_VALIDATED")

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="QUEUED_FOR_APPROVAL",
            decision_scope="FACT_ADMISSION",
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "auto_policy_deny_fallback_to_queue"},
            created_at=now,
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="C1_VALIDATED",
            transition_reason="AUTO_DENY_TO_QUEUE",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "AUTO_DENY_TO_QUEUE",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def queue_for_approval(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Append QUEUE_FOR_APPROVAL event while remaining in C1_VALIDATED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        candidate_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(candidate_row)
        current_state = candidate["fact_maturity_stage"]
        if current_state != "C1_VALIDATED":
            raise ValueError(
                f"invalid transition: QUEUE_FOR_APPROVAL requires C1_VALIDATED, got {current_state}"
            )

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="QUEUED_FOR_APPROVAL",
            decision_scope="FACT_ADMISSION",
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "policy_requires_human_approval"},
            created_at=now,
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="C1_VALIDATED",
            transition_reason="QUEUE_FOR_APPROVAL",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "QUEUED_FOR_APPROVAL",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def human_approve_candidate(
    *,
    candidate_id: str,
    decision_scope: str = "FACT_ADMISSION",
    actor_id: str,
    membership_valid: bool = True,
) -> Dict[str, Any]:
    """Apply HUMAN_APPROVE from C1_VALIDATED to C2_APPROVED."""

    if not actor_id:
        raise ValueError("actor_id is required for HUMAN_APPROVE")
    if decision_scope not in {"FACT_ADMISSION", "PROJECTION_TRUST", "BOTH"}:
        raise ValueError("decision_scope must be FACT_ADMISSION, PROJECTION_TRUST, or BOTH")

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        current_state = candidate["fact_maturity_stage"]
        if current_state != "C1_VALIDATED":
            raise ValueError(
                f"invalid transition: HUMAN_APPROVE requires C1_VALIDATED, got {current_state}"
            )

        if not membership_valid:
            _insert_decision(
                conn,
                candidate_id=candidate_id,
                decision_type="SYSTEM_REJECT",
                decision_scope=decision_scope,
                actor_type="system",
                actor_id="membership_resolver",
                reason_json={"reason": "stale_membership"},
                created_at=now,
            )
            conn.execute(
                hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s WHERE candidate_id = %s'),
                ("REJECTED", candidate_id),
            )
            transition = _append_transition(
                conn,
                candidate_id=candidate_id,
                from_state="C1_VALIDATED",
                to_state="REJECTED",
                transition_reason="HUMAN_APPROVE_STALE_MEMBERSHIP_REJECT",
                actor_type="system",
                actor_id="membership_resolver",
                created_at=now,
            )
            conn.commit()
            return {
                "result": "HUMAN_APPROVE_STALE_MEMBERSHIP_REJECT",
                "candidate_id": candidate_id,
                "candidate": get_candidate(candidate_id),
                "transition": transition,
            }

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="HUMAN_APPROVE",
            decision_scope=decision_scope,
            actor_type="human",
            actor_id=actor_id,
            reason_json={"reason": "human_approval"},
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s WHERE candidate_id = %s'),
            ("C2_APPROVED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="C2_APPROVED",
            transition_reason="HUMAN_APPROVE",
            actor_type="human",
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "HUMAN_APPROVE",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def human_reject_candidate(
    *,
    candidate_id: str,
    decision_scope: str = "FACT_ADMISSION",
    actor_id: str,
) -> Dict[str, Any]:
    """Apply HUMAN_REJECT from C1_VALIDATED to REJECTED."""

    if not actor_id:
        raise ValueError("actor_id is required for HUMAN_REJECT")
    if decision_scope not in {"FACT_ADMISSION", "PROJECTION_TRUST", "BOTH"}:
        raise ValueError("decision_scope must be FACT_ADMISSION, PROJECTION_TRUST, or BOTH")

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        current_state = candidate["fact_maturity_stage"]
        if current_state != "C1_VALIDATED":
            raise ValueError(
                f"invalid transition: HUMAN_REJECT requires C1_VALIDATED, got {current_state}"
            )

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="HUMAN_REJECT",
            decision_scope=decision_scope,
            actor_type="human",
            actor_id=actor_id,
            reason_json={"reason": "human_reject"},
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s WHERE candidate_id = %s'),
            ("REJECTED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="REJECTED",
            transition_reason="HUMAN_REJECT",
            actor_type="human",
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "HUMAN_REJECT",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def approval_timeout_escalate(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
    authority_valid: bool = True,
    quorum_satisfied: bool = True,
) -> Dict[str, Any]:
    """Record timeout escalation while remaining in C1_VALIDATED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        if candidate["fact_maturity_stage"] != "C1_VALIDATED":
            raise ValueError(
                "invalid transition: APPROVAL_TIMEOUT_ESCALATE requires C1_VALIDATED"
            )
        if not authority_valid:
            raise ValueError("APPROVAL_TIMEOUT_ESCALATE requires authority_valid=True")
        if not quorum_satisfied:
            raise ValueError("APPROVAL_TIMEOUT_ESCALATE requires quorum_satisfied=True")

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="QUEUED_FOR_APPROVAL",
            decision_scope="FACT_ADMISSION",
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "approval_timeout_escalate"},
            created_at=now,
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="C1_VALIDATED",
            transition_reason="APPROVAL_TIMEOUT_ESCALATE",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "APPROVAL_TIMEOUT_ESCALATE",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def approval_timeout_reject(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Apply timeout reject from C1_VALIDATED to REJECTED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        if candidate["fact_maturity_stage"] != "C1_VALIDATED":
            raise ValueError(
                "invalid transition: APPROVAL_TIMEOUT_REJECT requires C1_VALIDATED"
            )

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="SYSTEM_REJECT",
            decision_scope="FACT_ADMISSION",
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "approval_timeout_reject"},
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s WHERE candidate_id = %s'),
            ("REJECTED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C1_VALIDATED",
            to_state="REJECTED",
            transition_reason="APPROVAL_TIMEOUT_REJECT",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "APPROVAL_TIMEOUT_REJECT",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def fact_commit_success(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
    rea_id: Optional[str] = None,
    justification_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit fact from C2_APPROVED to C3_MATERIALIZED and create baseline outcome."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        state = candidate["fact_maturity_stage"]
        if state == "C3_MATERIALIZED":
            latest = _latest_outcome_row(conn, candidate_id)
            return {
                "result": "DUPLICATE",
                "candidate_id": candidate_id,
                "candidate": candidate,
                "outcome": latest,
            }
        if state != "C2_APPROVED":
            raise ValueError(f"invalid transition: FACT_COMMIT_SUCCESS requires C2_APPROVED, got {state}")

        resolved_rea_id = rea_id or f"t6rea_{uuid4().hex}"
        resolved_justification_id = justification_id or f"t6jus_{uuid4().hex}"

        # Minimal fact admission artifacts in existing EOS governed tables.
        workflow_id = f"t6wf_{uuid4().hex}"
        claim_payload = candidate.get("candidate_json") or {}
        claim_hash = candidate.get("candidate_hash")
        conn.execute(
            hosted.sql_query("INSERT INTO workflows (workflow_id, signal_id, workflow_type, workflow_json, status, created_at) VALUES (?,?,?,?,?,?)", 'INSERT INTO workflows (workflow_id, signal_id, workflow_type, workflow_json, status, created_at) VALUES (%s,%s,%s,%s,%s,%s)'),
            (
                workflow_id,
                candidate_id,
                "t6_materialization_fact_commit",
                canonical_json({"candidate_id": candidate_id}),
                "COMMITTED",
                now,
            ),
        )
        conn.execute(
            hosted.sql_query("INSERT INTO rea_claims (rea_id, workflow_id, claim_json, claim_hash, created_at) VALUES (?,?,?,?,?)", 'INSERT INTO rea_claims (rea_id, workflow_id, claim_json, claim_hash, created_at) VALUES (%s,%s,%s,%s,%s)'),
            (
                resolved_rea_id,
                workflow_id,
                canonical_json(claim_payload),
                claim_hash,
                now,
            ),
        )
        conn.execute(
            hosted.sql_query("INSERT INTO justifications (justification_id, workflow_id, rea_id, justification_json, content_hash, created_at) VALUES (?,?,?,?,?,?)", 'INSERT INTO justifications (justification_id, workflow_id, rea_id, justification_json, content_hash, created_at) VALUES (%s,%s,%s,%s,%s,%s)'),
            (
                resolved_justification_id,
                workflow_id,
                resolved_rea_id,
                canonical_json({"candidate_id": candidate_id, "reason": "fact_commit_success"}),
                claim_hash,
                now,
            ),
        )

        outcome_id = f"t6o_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO t6_materialization_outcomes (
              outcome_id,
              candidate_id,
              rea_id,
              justification_id,
              fact_commit_status,
              projection_result_id,
              projection_status,
              trust_state,
              replay_binding_ref,
              supersedes_outcome_id,
              content_hash,
              fact_maturity_stage,
              created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, """
            INSERT INTO t6_materialization_outcomes (
              outcome_id,
              candidate_id,
              rea_id,
              justification_id,
              fact_commit_status,
              projection_result_id,
              projection_status,
              trust_state,
              replay_binding_ref,
              supersedes_outcome_id,
              content_hash,
              fact_maturity_stage,
              created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """),
            (
                outcome_id,
                candidate_id,
                resolved_rea_id,
                resolved_justification_id,
                "COMMITTED",
                None,
                "PROJECTION_PENDING",
                "PENDING",
                None,
                None,
                compute_hash({"candidate_id": candidate_id, "outcome_id": outcome_id, "kind": "baseline"}),
                "C3_MATERIALIZED",
                now,
            ),
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ?, projection_status = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s, projection_status = %s WHERE candidate_id = %s'),
            ("C3_MATERIALIZED", "PROJECTION_PENDING", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C2_APPROVED",
            to_state="C3_MATERIALIZED",
            transition_reason="FACT_COMMIT_SUCCESS",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    with get_conn() as conn:
        outcome_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_outcomes WHERE outcome_id = ?", 'SELECT * FROM t6_materialization_outcomes WHERE outcome_id = %s'),
            (outcome_id,),
        ).fetchone()

    return {
        "result": "FACT_COMMIT_SUCCESS",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "outcome": _row_to_dict(outcome_row),
        "transition": transition,
    }


def fact_commit_reject(
    *,
    candidate_id: str,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Reject fact commit from C2_APPROVED to REJECTED."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)
        if candidate["fact_maturity_stage"] != "C2_APPROVED":
            raise ValueError("invalid transition: FACT_COMMIT_REJECT requires C2_APPROVED")

        _insert_decision(
            conn,
            candidate_id=candidate_id,
            decision_type="SYSTEM_REJECT",
            decision_scope="FACT_ADMISSION",
            actor_type=actor_type,
            actor_id=actor_id,
            reason_json={"reason": "fact_commit_reject"},
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET fact_maturity_stage = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET fact_maturity_stage = %s WHERE candidate_id = %s'),
            ("REJECTED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C2_APPROVED",
            to_state="REJECTED",
            transition_reason="FACT_COMMIT_REJECT",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "FACT_COMMIT_REJECT",
        "candidate_id": candidate_id,
        "candidate": get_candidate(candidate_id),
        "transition": transition,
    }


def _append_outcome(
    conn,
    *,
    candidate_id: str,
    rea_id: str,
    justification_id: str,
    fact_commit_status: str,
    projection_result_id: Optional[str],
    projection_status: str,
    trust_state: str,
    replay_binding_ref: Optional[str],
    fact_maturity_stage: str,
    created_at: str,
) -> Dict[str, Any]:
    latest = _latest_outcome_row(conn, candidate_id)
    supersedes = latest["outcome_id"] if latest is not None else None
    outcome_id = f"t6o_{uuid4().hex}"
    payload = {
        "candidate_id": candidate_id,
        "rea_id": rea_id,
        "justification_id": justification_id,
        "projection_result_id": projection_result_id,
        "projection_status": projection_status,
        "replay_binding_ref": replay_binding_ref,
        "supersedes_outcome_id": supersedes,
    }
    conn.execute(
        hosted.sql_query("""
        INSERT INTO t6_materialization_outcomes (
          outcome_id,
          candidate_id,
          rea_id,
          justification_id,
          fact_commit_status,
          projection_result_id,
          projection_status,
          trust_state,
          replay_binding_ref,
          supersedes_outcome_id,
          content_hash,
          fact_maturity_stage,
          created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, """
        INSERT INTO t6_materialization_outcomes (
          outcome_id,
          candidate_id,
          rea_id,
          justification_id,
          fact_commit_status,
          projection_result_id,
          projection_status,
          trust_state,
          replay_binding_ref,
          supersedes_outcome_id,
          content_hash,
          fact_maturity_stage,
          created_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """),
        (
            outcome_id,
            candidate_id,
            rea_id,
            justification_id,
            fact_commit_status,
            projection_result_id,
            projection_status,
            trust_state,
            replay_binding_ref,
            supersedes,
            compute_hash(payload),
            fact_maturity_stage,
            created_at,
        ),
    )
    return {
        "outcome_id": outcome_id,
        "supersedes_outcome_id": supersedes,
    }


def projection_run_success(
    *,
    candidate_id: str,
    projection_version: str,
    policy_bundle_hash: str,
    projection_context_ref: Optional[str] = None,
    projection_context_hash: Optional[str] = None,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Append projection success outcome from C3_MATERIALIZED only."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        candidate_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(candidate_row)
        if candidate["fact_maturity_stage"] != "C3_MATERIALIZED":
            raise ValueError("invalid transition: PROJECTION_RUN_SUCCESS requires C3_MATERIALIZED")

        latest = _latest_outcome_row(conn, candidate_id)
        if latest is None:
            raise ValueError("projection requires baseline fact-commit outcome")
        rea_id = latest.get("rea_id")
        justification_id = latest.get("justification_id")
        ctx_ref = projection_context_ref or candidate.get("projection_context_ref") or f"t6ctx_{candidate_id}"
        ctx_hash = projection_context_hash or candidate.get("projection_context_hash") or compute_hash(
            {"candidate_id": candidate_id, "projection_context_ref": ctx_ref}
        )

        key_hash = _replay_binding_key_hash(
            rea_id=rea_id,
            projection_context_hash=ctx_hash,
            projection_version=projection_version,
            policy_bundle_hash=policy_bundle_hash,
        )
        projection_result_id = f"prj_{key_hash[:24]}"

        existing = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM t6_materialization_outcomes
            WHERE candidate_id = ?
              AND projection_result_id = ?
              AND projection_status = 'PROJECTION_MATERIALIZED'
            ORDER BY created_at DESC, outcome_id DESC
            LIMIT 1
            """, """
            SELECT *
            FROM t6_materialization_outcomes
            WHERE candidate_id = %s
              AND projection_result_id = %s
              AND projection_status = 'PROJECTION_MATERIALIZED'
            ORDER BY created_at DESC, outcome_id DESC
            LIMIT 1
            """),
            (candidate_id, projection_result_id),
        ).fetchone()
        if existing is not None:
            existing_outcome = _row_to_dict(existing)
            return {
                "result": "DUPLICATE",
                "candidate_id": candidate_id,
                "outcome": existing_outcome,
                "replay_binding_ref": existing_outcome.get("replay_binding_ref"),
            }

        replay_event_id = f"t6rb_{key_hash[:32]}"
        create_replay_binding(
            event_id=replay_event_id,
            pins={
                "Pi_v": projection_version,
                "C_v": policy_bundle_hash,
            },
            projection_context_ref=ctx_ref,
            projection_context_hash=ctx_hash,
            resolution_trace={"candidate_id": candidate_id, "projection_result_id": projection_result_id},
            effects={"status": "PROJECTION_MATERIALIZED"},
            event_payload={"candidate_id": candidate_id, "rea_id": rea_id},
        )

        appended = _append_outcome(
            conn,
            candidate_id=candidate_id,
            rea_id=rea_id,
            justification_id=justification_id,
            fact_commit_status="COMMITTED",
            projection_result_id=projection_result_id,
            projection_status="PROJECTION_MATERIALIZED",
            trust_state="MATERIALIZED",
            replay_binding_ref=replay_event_id,
            fact_maturity_stage="C3_MATERIALIZED",
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET projection_status = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET projection_status = %s WHERE candidate_id = %s'),
            ("PROJECTION_MATERIALIZED", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C3_MATERIALIZED",
            to_state="C3_MATERIALIZED",
            transition_reason="PROJECTION_RUN_SUCCESS",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "PROJECTION_RUN_SUCCESS",
        "candidate_id": candidate_id,
        "projection_result_id": projection_result_id,
        "replay_binding_ref": replay_event_id,
        "outcome_id": appended["outcome_id"],
        "transition": transition,
    }


def projection_run_exception(
    *,
    candidate_id: str,
    exception_code: str,
    exception_details: Optional[Dict[str, Any]] = None,
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Append projection exception outcome from C3_MATERIALIZED only."""

    init_db()
    now = now_iso()
    with get_conn() as conn:
        candidate_row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(candidate_row)
        if candidate["fact_maturity_stage"] != "C3_MATERIALIZED":
            raise ValueError("invalid transition: PROJECTION_RUN_EXCEPTION requires C3_MATERIALIZED")

        latest = _latest_outcome_row(conn, candidate_id)
        if latest is None:
            raise ValueError("projection requires baseline fact-commit outcome")
        rea_id = latest.get("rea_id")
        justification_id = latest.get("justification_id")
        projection_result_id = f"prjexc_{uuid4().hex[:24]}"
        exception_id = f"t6e_{uuid4().hex}"

        conn.execute(
            hosted.sql_query("""
            INSERT INTO t6_projection_exceptions (
              exception_id,
              candidate_id,
              rea_id,
              projection_context_hash,
              exception_code,
              exception_details_json,
              projection_result_id,
              resolved_by_outcome_id,
              retry_after,
              created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, """
            INSERT INTO t6_projection_exceptions (
              exception_id,
              candidate_id,
              rea_id,
              projection_context_hash,
              exception_code,
              exception_details_json,
              projection_result_id,
              resolved_by_outcome_id,
              retry_after,
              created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """),
            (
                exception_id,
                candidate_id,
                rea_id,
                candidate.get("projection_context_hash"),
                exception_code,
                canonical_json(exception_details or {}),
                projection_result_id,
                None,
                None,
                now,
            ),
        )
        appended = _append_outcome(
            conn,
            candidate_id=candidate_id,
            rea_id=rea_id,
            justification_id=justification_id,
            fact_commit_status="COMMITTED",
            projection_result_id=projection_result_id,
            projection_status="PROJECTION_EXCEPTION",
            trust_state="EXCEPTION",
            replay_binding_ref=None,
            fact_maturity_stage="C3_MATERIALIZED",
            created_at=now,
        )
        conn.execute(
            hosted.sql_query("UPDATE t6_materialization_candidates SET projection_status = ? WHERE candidate_id = ?", 'UPDATE t6_materialization_candidates SET projection_status = %s WHERE candidate_id = %s'),
            ("PROJECTION_EXCEPTION", candidate_id),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state="C3_MATERIALIZED",
            to_state="C3_MATERIALIZED",
            transition_reason="PROJECTION_RUN_EXCEPTION",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "result": "PROJECTION_RUN_EXCEPTION",
        "candidate_id": candidate_id,
        "exception_id": exception_id,
        "outcome_id": appended["outcome_id"],
        "transition": transition,
    }


def enforce_chain_integrity_or_fail_closed(
    *,
    candidate_id: str,
    mode: str,
    rea_submitter_user_id: Optional[str],
    role_authorized_handler_user_id: Optional[str],
    actor_type: str = "system",
    actor_id: Optional[str] = "system",
) -> Dict[str, Any]:
    """Validate outcome chain and emit fail-closed escalation event when invalid."""

    integrity = validate_outcome_chain(candidate_id)
    if integrity.get("status") == "OK" or integrity.get("status") == "EMPTY":
        return {
            "status": "OK",
            "candidate_id": candidate_id,
            "integrity": integrity,
        }

    recipient = resolve_failure_recipient(
        mode=mode,
        rea_submitter_user_id=rea_submitter_user_id,
        role_authorized_handler_user_id=role_authorized_handler_user_id,
    )
    now = now_iso()
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM t6_materialization_candidates WHERE candidate_id = ?", 'SELECT * FROM t6_materialization_candidates WHERE candidate_id = %s'),
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = _row_to_dict(row)

        exception_id = f"t6e_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO t6_projection_exceptions (
              exception_id,
              candidate_id,
              rea_id,
              projection_context_hash,
              exception_code,
              exception_details_json,
              projection_result_id,
              resolved_by_outcome_id,
              retry_after,
              created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, """
            INSERT INTO t6_projection_exceptions (
              exception_id,
              candidate_id,
              rea_id,
              projection_context_hash,
              exception_code,
              exception_details_json,
              projection_result_id,
              resolved_by_outcome_id,
              retry_after,
              created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """),
            (
                exception_id,
                candidate_id,
                None,
                candidate.get("projection_context_hash"),
                "CHAIN_INTEGRITY_FAILURE",
                canonical_json({"integrity": integrity, "recipient": recipient}),
                None,
                None,
                None,
                now,
            ),
        )
        transition = _append_transition(
            conn,
            candidate_id=candidate_id,
            from_state=candidate.get("fact_maturity_stage"),
            to_state=candidate.get("fact_maturity_stage"),
            transition_reason="CHAIN_INTEGRITY_FAILURE",
            actor_type=actor_type,
            actor_id=actor_id,
            created_at=now,
        )
        conn.commit()

    return {
        "status": "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "integrity": integrity,
        "recipient": recipient,
        "transition": transition,
    }
