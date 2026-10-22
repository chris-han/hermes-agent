"""
Monthly Accounting Workflow Store — v8.1

Provides period state machine CRUD and governed state transitions for
MonthlyAccountingWorkflow_t.

State machine:
  open → validating → blocked (on blocker detected)
  validating → ready_to_close (on all checks passing)
  blocked → validating (after resolving blockers)
  ready_to_close → closed
  closed → reported
  reported → filed
  filed → archived

Invariant: close_state must NOT transition to 'closed' if any of the following
blockers are present:
  - projection_exception
  - reconciliation_failure
  - missing_evidence
  - override_unapproved
  - validation_failure
  - unresolved_compliance_drift
  - incomplete_profit_loss_close
"""
from storage import hosted_postgres as hosted

import json
from datetime import datetime, timezone
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_hash
from eos.db import get_conn, now_iso


# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------

VALID_STATES = {
    "open",
    "validating",
    "blocked",
    "ready_to_close",
    "closed",
    "reported",
    "filed",
    "archived",
}

# Transitions: current_state → set of allowed next states
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "open": {"validating"},
    "validating": {"blocked", "ready_to_close"},
    "blocked": {"validating"},
    "ready_to_close": {"closed"},
    "closed": {"reported"},
    "reported": {"filed"},
    "filed": {"archived"},
    "archived": set(),
}

# Blockers that prevent transitioning to 'closed'
HARD_BLOCKERS = {
    "projection_exception",
    "reconciliation_failure",
    "missing_evidence",
    "override_unapproved",
    "validation_failure",
    "unresolved_compliance_drift",
    "incomplete_profit_loss_close",
    "cross_domain_escalate",
    "cross_domain_block_action",
    "cross_domain_block_export",
}


def _assert_transition(current_state: str, next_state: str) -> None:
    allowed = ALLOWED_TRANSITIONS.get(current_state, set())
    if next_state not in allowed:
        raise ValueError(
            f"Invalid state transition: {current_state!r} → {next_state!r}. "
            f"Allowed transitions from {current_state!r}: {sorted(allowed)}"
        )


def _recompute_hash(row: dict) -> str:
    payload = {k: v for k, v in row.items() if k not in ("content_hash", "updated_at")}
    return compute_hash(canonical_json(payload))


def _update_workflow(
    workflow_id: str,
    close_state: str,
    blockers: list[str],
    timestamp_field: str | None = None,
) -> dict:
    ts = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM monthly_accounting_workflows WHERE workflow_id = ?", 'SELECT * FROM monthly_accounting_workflows WHERE workflow_id = %s'),
            (workflow_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Workflow not found: {workflow_id!r}")

        row_dict = dict(row)
        row_dict["close_state"] = close_state
        row_dict["blockers_json"] = json.dumps(blockers)
        row_dict["updated_at"] = ts
        if timestamp_field:
            row_dict[timestamp_field] = ts
        row_dict["content_hash"] = _recompute_hash(row_dict)

        conn.execute(
            hosted.sql_query("""
            UPDATE monthly_accounting_workflows
            SET close_state = ?,
                blockers_json = ?,
                updated_at = ?,
                content_hash = ?,
                closed_at = COALESCE(?, closed_at),
                reported_at = COALESCE(?, reported_at),
                filed_at = COALESCE(?, filed_at),
                archived_at = COALESCE(?, archived_at)
            WHERE workflow_id = ?
            """, """
            UPDATE monthly_accounting_workflows
            SET close_state = %s,
                blockers_json = %s,
                updated_at = %s,
                content_hash = %s,
                closed_at = COALESCE(%s, closed_at),
                reported_at = COALESCE(%s, reported_at),
                filed_at = COALESCE(%s, filed_at),
                archived_at = COALESCE(%s, archived_at)
            WHERE workflow_id = %s
            """),
            (
                row_dict["close_state"],
                row_dict["blockers_json"],
                row_dict["updated_at"],
                row_dict["content_hash"],
                ts if timestamp_field == "closed_at" else None,
                ts if timestamp_field == "reported_at" else None,
                ts if timestamp_field == "filed_at" else None,
                ts if timestamp_field == "archived_at" else None,
                workflow_id,
            ),
        )
    return get_workflow_by_id(workflow_id)


def _get_workflow_raw(workflow_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM monthly_accounting_workflows WHERE workflow_id = ?", 'SELECT * FROM monthly_accounting_workflows WHERE workflow_id = %s'),
            (workflow_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["blockers"] = json.loads(d.get("blockers_json") or "[]")
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_workflow(org_id: str, period_id: str) -> dict:
    """Create a new MonthlyAccountingWorkflow_t in 'open' state."""
    workflow_id = f"maw_{uuid4().hex}"
    ts = now_iso()
    row = {
        "workflow_id": workflow_id,
        "org_id": org_id,
        "period_id": period_id,
        "close_state": "open",
        "blockers_json": "[]",
        "opened_at": ts,
        "closed_at": None,
        "reported_at": None,
        "filed_at": None,
        "archived_at": None,
        "updated_at": ts,
    }
    payload = {k: v for k, v in row.items() if k not in ("content_hash", "updated_at")}
    row["content_hash"] = compute_hash(canonical_json(payload))

    keys = [k for k in row if row[k] is not None]
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO monthly_accounting_workflows ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return _get_workflow_raw(workflow_id)


def get_workflow(org_id: str, period_id: str) -> dict | None:
    """Get workflow by org + period (unique per period)."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM monthly_accounting_workflows WHERE org_id = ? AND period_id = ?", 'SELECT * FROM monthly_accounting_workflows WHERE org_id = %s AND period_id = %s'),
            (org_id, period_id),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["blockers"] = json.loads(d.get("blockers_json") or "[]")
    return d


def get_workflow_by_id(workflow_id: str) -> dict | None:
    return _get_workflow_raw(workflow_id)


def get_workflow_state(org_id: str, period_id: str) -> str | None:
    wf = get_workflow(org_id, period_id)
    return wf["close_state"] if wf else None


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def _load_workflow_for_transition(org_id: str, period_id: str) -> dict:
    wf = get_workflow(org_id, period_id)
    if wf is None:
        raise ValueError(f"No workflow for org={org_id!r} period={period_id!r}")
    return wf


def transition_to_validating(org_id: str, period_id: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "validating")
    return _update_workflow(wf["workflow_id"], "validating", wf["blockers"])


def transition_to_blocked(org_id: str, period_id: str, blockers: list[str]) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "blocked")
    if not blockers:
        raise ValueError("At least one blocker reason must be provided to transition to 'blocked'")
    merged = list(set(wf["blockers"]) | set(blockers))
    return _update_workflow(wf["workflow_id"], "blocked", merged)


def transition_to_ready_to_close(org_id: str, period_id: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "ready_to_close")
    remaining = wf["blockers"]
    if remaining:
        raise ValueError(
            f"Cannot transition to 'ready_to_close' with unresolved blockers: {remaining}"
        )
    return _update_workflow(wf["workflow_id"], "ready_to_close", [])


def transition_to_closed(org_id: str, period_id: str) -> dict:
    """Transition period to 'closed'.

    Invariant: blocked if any hard blocker is present in the blockers list.
    """
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "closed")
    active_hard_blockers = [b for b in wf["blockers"] if b in HARD_BLOCKERS]
    if active_hard_blockers:
        raise ValueError(
            f"Cannot close period: unresolved hard blockers present: {active_hard_blockers}"
        )
    return _update_workflow(wf["workflow_id"], "closed", [], "closed_at")


def transition_to_reported(org_id: str, period_id: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "reported")
    return _update_workflow(wf["workflow_id"], "reported", wf["blockers"], "reported_at")


def transition_to_filed(org_id: str, period_id: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "filed")
    return _update_workflow(wf["workflow_id"], "filed", wf["blockers"], "filed_at")


def transition_to_archived(org_id: str, period_id: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    _assert_transition(wf["close_state"], "archived")
    return _update_workflow(wf["workflow_id"], "archived", wf["blockers"], "archived_at")


# ---------------------------------------------------------------------------
# Blocker management
# ---------------------------------------------------------------------------

def add_blocker(org_id: str, period_id: str, blocker: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    new_blockers = list(set(wf["blockers"]) | {blocker})
    return _update_workflow(wf["workflow_id"], wf["close_state"], new_blockers)


def resolve_blocker(org_id: str, period_id: str, blocker: str) -> dict:
    wf = _load_workflow_for_transition(org_id, period_id)
    new_blockers = [b for b in wf["blockers"] if b != blocker]
    return _update_workflow(wf["workflow_id"], wf["close_state"], new_blockers)


def list_blockers(org_id: str, period_id: str) -> list[str]:
    wf = get_workflow(org_id, period_id)
    if wf is None:
        return []
    return wf.get("blockers", [])
