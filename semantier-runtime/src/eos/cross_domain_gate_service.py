"""Gate 3 integration service for full-cycle workflows (v8.2).

This service bridges:
  - deterministic Gate 3 evaluation (cross_domain_resolver)
  - append-only persistence (cross_domain_store)
  - monthly workflow state/blocker updates (monthly_workflow_store)
"""

from __future__ import annotations

from eos.cross_domain_resolver import NON_BLOCKING_OUTCOMES, resolve_cross_domain
from eos.cross_domain_store import insert_cross_domain_order_result
from eos.monthly_workflow_store import (
    add_blocker,
    get_workflow,
    list_blockers,
    resolve_blocker,
    transition_to_blocked,
    transition_to_filed,
    transition_to_ready_to_close,
    transition_to_reported,
)


_OUTCOME_TO_BLOCKER = {
    "ESCALATE": "cross_domain_escalate",
    "BLOCK_ACTION": "cross_domain_block_action",
    "BLOCK_EXPORT": "cross_domain_block_export",
}


def _clear_cross_domain_blockers(org_id: str, period_id: str) -> None:
    for blocker in (
        "cross_domain_escalate",
        "cross_domain_block_action",
        "cross_domain_block_export",
    ):
        if blocker in list_blockers(org_id, period_id):
            resolve_blocker(org_id, period_id, blocker)


def _apply_blocking_outcome(org_id: str, period_id: str, outcome: str) -> dict:
    blocker = _OUTCOME_TO_BLOCKER[outcome]
    wf = get_workflow(org_id, period_id)
    if wf is None:
        raise ValueError(f"No workflow for org={org_id!r} period={period_id!r}")

    if wf["close_state"] == "validating":
        return transition_to_blocked(org_id, period_id, [blocker])
    return add_blocker(org_id, period_id, blocker)


def evaluate_and_record_cross_domain_action(
    org_id: str,
    period_id: str | None,
    action_intent: str,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Run Gate 3 resolution and persist CrossDomainOrderResult_t."""
    resolved = resolve_cross_domain(
        projection_set=projection_set,
        action_intent=action_intent,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )

    row = {
        "org_id": org_id,
        "period_id": period_id,
        "action_intent": action_intent,
        "projection_set_hash": resolved["projection_set_hash"],
        "policy_version": resolved["policy_version"],
        "resolver_version": resolved["resolver_version"],
        "order_result": resolved["order_result"],
        "disclosure_required": resolved["disclosure_required"],
        "limits_json": resolved.get("limits"),
        "blocked_action": resolved["blocked_action"],
        "blocked_export": resolved["blocked_export"],
        "reason_codes_json": resolved.get("reason_codes", []),
    }
    persisted = insert_cross_domain_order_result(row)
    return persisted


def transition_to_ready_to_close_with_cross_domain(
    org_id: str,
    period_id: str,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Evaluate Gate 3 for `close` intent and transition workflow accordingly.

    Behavior:
      - non-blocking outcome: clear cross-domain blockers and move to ready_to_close
      - blocking/escalation outcome: add blocker and move/stay blocked
    """
    order_result = evaluate_and_record_cross_domain_action(
        org_id=org_id,
        period_id=period_id,
        action_intent="close",
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )

    outcome = order_result["order_result"]

    # Always clear stale cross-domain blockers before re-applying the current outcome.
    _clear_cross_domain_blockers(org_id, period_id)

    if outcome in NON_BLOCKING_OUTCOMES:
        workflow = transition_to_ready_to_close(org_id, period_id)
        return {"workflow": workflow, "cross_domain_order_result": order_result}

    workflow = _apply_blocking_outcome(org_id, period_id, outcome)

    return {"workflow": workflow, "cross_domain_order_result": order_result}


def transition_to_reported_with_cross_domain(
    org_id: str,
    period_id: str,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Evaluate Gate 3 for `report_export` intent before reported transition."""
    order_result = evaluate_and_record_cross_domain_action(
        org_id=org_id,
        period_id=period_id,
        action_intent="report_export",
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )

    outcome = order_result["order_result"]
    _clear_cross_domain_blockers(org_id, period_id)

    if outcome in NON_BLOCKING_OUTCOMES:
        workflow = transition_to_reported(org_id, period_id)
        return {"workflow": workflow, "cross_domain_order_result": order_result}

    workflow = _apply_blocking_outcome(org_id, period_id, outcome)

    return {"workflow": workflow, "cross_domain_order_result": order_result}


def transition_to_filed_with_cross_domain(
    org_id: str,
    period_id: str,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Evaluate Gate 3 for `tax_package` intent before filed transition."""
    order_result = evaluate_and_record_cross_domain_action(
        org_id=org_id,
        period_id=period_id,
        action_intent="tax_package",
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )

    outcome = order_result["order_result"]
    _clear_cross_domain_blockers(org_id, period_id)

    if outcome in NON_BLOCKING_OUTCOMES:
        workflow = transition_to_filed(org_id, period_id)
        return {"workflow": workflow, "cross_domain_order_result": order_result}

    workflow = _apply_blocking_outcome(org_id, period_id, outcome)

    return {"workflow": workflow, "cross_domain_order_result": order_result}


def _evaluate_non_stateful_action_with_cross_domain(
    org_id: str,
    period_id: str | None,
    action_intent: str,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    order_result = evaluate_and_record_cross_domain_action(
        org_id=org_id,
        period_id=period_id,
        action_intent=action_intent,
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )

    workflow = None
    if period_id is not None:
        outcome = order_result["order_result"]
        if outcome in NON_BLOCKING_OUTCOMES:
            _clear_cross_domain_blockers(org_id, period_id)
            workflow = get_workflow(org_id, period_id)
        else:
            workflow = _apply_blocking_outcome(org_id, period_id, outcome)

    return {"workflow": workflow, "cross_domain_order_result": order_result}


def evaluate_insurance_package_with_cross_domain(
    org_id: str,
    period_id: str | None,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Evaluate Gate 3 for `insurance_package` intent and persist result."""
    return _evaluate_non_stateful_action_with_cross_domain(
        org_id=org_id,
        period_id=period_id,
        action_intent="insurance_package",
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )


def evaluate_lender_package_with_cross_domain(
    org_id: str,
    period_id: str | None,
    projection_set: dict,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None = None,
) -> dict:
    """Evaluate Gate 3 for `lender_package` intent and persist result."""
    return _evaluate_non_stateful_action_with_cross_domain(
        org_id=org_id,
        period_id=period_id,
        action_intent="lender_package",
        projection_set=projection_set,
        xpolicy=xpolicy,
        resolver_contract=resolver_contract,
        tiered_verification_contract=tiered_verification_contract,
    )
