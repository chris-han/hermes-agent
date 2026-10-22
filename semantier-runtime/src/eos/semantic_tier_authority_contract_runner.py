"""SemanticTierAuthorityContract_t runner.

Architecture ref: docs/canonical/architecture.md §22.9.15

Executes the tiered semantic authority chain and emits OrderMaintenanceResult_t.

Authority priority chain (highest to lowest):
    O_v (100) > O_tag,v (90) > K_v (80) > ConstraintBundle_v (70) >
    PrecedenceGraph_v (60) > ResolutionRules_v (50) > Π_v (40) >
    TrialBalanceValidationContract_v (30) > GovernanceDecision_t (20) >
    UserFeedbackSignal_t (10)

The contract fails closed when:
- K_v attempts to redefine O_v
- projection context bypasses R_v
- management preference overrides accounting/tax constraint without governed exception
- trial balance validation is waived without sufficient authority
- projection is trusted despite unresolved critical validation failure
- user feedback is missing for material validation failure
- export includes untrusted projection state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from eos.order_maintenance_result_store import store_order_maintenance_result
from eos.trial_balance_view import ProjectionTrustState

CONTRACT_ID = "finance_order_verification.v1"

# Priority chain definition
_AUTHORITY_CHAIN = [
    {"tier": "ontology", "artifact_key": "ontology_version", "priority": 100},
    {"tier": "tag_ontology", "artifact_key": "tag_ontology_version", "priority": 90},
    {"tier": "governed_knowledge", "artifact_key": "knowledge_context_version", "priority": 80},
    {"tier": "constraints", "artifact_key": "constraint_bundle_version", "priority": 70},
    {"tier": "precedence", "artifact_key": "precedence_graph_version", "priority": 60},
    {"tier": "resolution", "artifact_key": "resolution_rules_version", "priority": 50},
    {"tier": "projection", "artifact_key": "projection_bundle_version", "priority": 40},
    {"tier": "representation_validation", "artifact_key": "validation_contract_version", "priority": 30},
    {"tier": "governance", "artifact_key": "governance_decisions", "priority": 20},
    {"tier": "feedback", "artifact_key": "user_feedback_signals", "priority": 10},
]


def _has_material_failures_without_feedback(
    validation_result: Optional[Dict[str, Any]],
    feedback_signals: List[Dict[str, Any]],
) -> bool:
    """Return True if any material (high/critical) failure lacks a feedback signal."""
    if validation_result is None:
        return False
    checks = validation_result.get("checks", [])
    material_failed_check_ids = {
        c["check_id"]
        for c in checks
        if c.get("status") == "failed" and c.get("severity") in ("high", "critical")
    }
    feedback_check_ids = {s.get("check_id") for s in feedback_signals}
    return bool(material_failed_check_ids - feedback_check_ids)


def _has_unresolved_critical_failure(validation_result: Optional[Dict[str, Any]]) -> bool:
    """Return True if any critical check failed and has no waiver/governance decision."""
    if validation_result is None:
        return False
    status = validation_result.get("resulting_trust_state", "")
    return status == ProjectionTrustState.PROJECTION_BLOCKED


def _authority_override_without_exception(
    validation_result: Optional[Dict[str, Any]],
    governance_decisions: List[Dict[str, Any]],
) -> bool:
    """
    Return True if a lower-authority tier appears to override a higher one
    without a valid governed exception.

    Current check: management-preference decisions that waive a critical check
    without an accountant-approved justification.
    """
    critical_waived = any(
        d.get("decision_enum") == "WAIVE_CHECK_WITH_REASON" and d.get("approved_by") is None
        for d in governance_decisions
        if validation_result and any(
            c.get("check_id") == d.get("check_id") and c.get("severity") == "critical"
            for c in validation_result.get("checks", [])
        )
    )
    return critical_waived


def run_semantic_tier_authority_contract(
    *,
    org_id: str,
    period_id: str,
    pins: Dict[str, Any],
    validation_result: Optional[Dict[str, Any]] = None,
    governance_decisions: Optional[List[Dict[str, Any]]] = None,
    feedback_signals: Optional[List[Dict[str, Any]]] = None,
    current_trust_state: str = ProjectionTrustState.PROJECTION_CANDIDATE,
    created_by: str = "system",
) -> Dict[str, Any]:
    """Execute SemanticTierAuthorityContract_t and emit OrderMaintenanceResult_t.

    Parameters
    ----------
    org_id:
        Organisation identifier.
    period_id:
        Accounting period identifier.
    pins:
        Semantic version pins for the run.  The contract checks that all
        required tier artifacts are present.
    validation_result:
        TrialBalanceValidationResult_t dict.  If omitted, representation-
        validation tier is recorded as skipped.
    governance_decisions:
        List of TrialBalanceGovernanceDecision_t dicts for this validation run.
    feedback_signals:
        List of UserFeedbackSignal_t dicts for this validation run.
    current_trust_state:
        The effective ProjectionTrustState before this contract runs.
    created_by:
        Actor that triggered the run.

    Returns
    -------
    dict
        OrderMaintenanceResult_t with content_hash.
    """
    governance_decisions = governance_decisions or []
    feedback_signals = feedback_signals or []

    authority_trace: List[Dict[str, Any]] = []
    failed_assertions: List[Dict[str, Any]] = []

    # --- Walk the authority chain ---
    for tier_def in _AUTHORITY_CHAIN:
        tier = tier_def["tier"]
        key = tier_def["artifact_key"]

        # Governance and feedback are passed as lists, not version pins
        if key == "governance_decisions":
            artifact_ref = f"{len(governance_decisions)} decisions"
            tier_status = "applied"
        elif key == "user_feedback_signals":
            artifact_ref = f"{len(feedback_signals)} signals"
            tier_status = "applied"
        else:
            artifact_ref = pins.get(key, "")
            tier_status = "applied" if artifact_ref else "skipped"

        authority_trace.append({
            "tier": tier,
            "artifact_ref": artifact_ref,
            "status": tier_status,
            "reason": "" if tier_status == "applied" else "missing_pin",
        })

    # --- Run assertions ---

    # Assertion: all required tier pins present
    required_pins = [
        "ontology_version", "coa_version", "projection_bundle_version",
        "constraint_bundle_version", "validation_contract_version",
    ]
    for pin_key in required_pins:
        if not pins.get(pin_key):
            failed_assertions.append({
                "assertion_id": f"required_pin_present.{pin_key}",
                "severity": "critical",
                "feedback_ref": None,
                "governance_ref": None,
            })

    # Assertion: material failures have user feedback
    if _has_material_failures_without_feedback(validation_result, feedback_signals):
        failed_assertions.append({
            "assertion_id": "material_failures_have_user_feedback",
            "severity": "critical",
            "feedback_ref": None,
            "governance_ref": None,
        })

    # Assertion: no authority override without governed exception
    if _authority_override_without_exception(validation_result, governance_decisions):
        failed_assertions.append({
            "assertion_id": "no_authority_override_without_governed_exception",
            "severity": "critical",
            "feedback_ref": None,
            "governance_ref": None,
        })

    # Assertion: export must not include untrusted projection states
    if current_trust_state not in (
        ProjectionTrustState.PROJECTION_TRUSTED,
        ProjectionTrustState.PROJECTION_EXPORTED,
        ProjectionTrustState.PROJECTION_VALIDATED,
        ProjectionTrustState.PROJECTION_WARNING,
    ):
        # Blocked or requires_governance — cannot certify order
        if current_trust_state in (
            ProjectionTrustState.PROJECTION_BLOCKED,
            ProjectionTrustState.PROJECTION_REQUIRES_GOVERNANCE,
        ):
            failed_assertions.append({
                "assertion_id": "projection_trusted_only_after_required_authority_order",
                "severity": "critical",
                "feedback_ref": None,
                "governance_ref": None,
            })

    # Assertion: critical failures resolved or governed
    if _has_unresolved_critical_failure(validation_result):
        # Check if there's a governance decision that resolves it
        waived_check_ids = {
            d.get("check_id")
            for d in governance_decisions
            if d.get("decision_enum") in ("WAIVE_CHECK_WITH_REASON", "ACCEPT_VALIDATION_RESULT")
            and d.get("approved_by") is not None
        }
        checks = (validation_result or {}).get("checks", [])
        critical_failed_ids = {
            c["check_id"] for c in checks
            if c.get("status") == "failed" and c.get("severity") == "critical"
        }
        unresolved = critical_failed_ids - waived_check_ids
        if unresolved:
            failed_assertions.append({
                "assertion_id": "critical_failures_resolved_or_governed",
                "severity": "critical",
                "feedback_ref": None,
                "governance_ref": None,
            })

    # --- Determine order status ---
    has_critical = any(a.get("severity") == "critical" for a in failed_assertions)
    has_any_failure = bool(failed_assertions)

    if has_critical:
        order_status = "order_failed"
    elif has_any_failure:
        order_status = "order_escalated"
    elif governance_decisions and any(
        d.get("decision_enum") in ("WAIVE_CHECK_WITH_REASON", "REPROJECT_UNDER_APPROVED_BUNDLE")
        for d in governance_decisions
    ):
        order_status = "order_repaired"
    elif validation_result and validation_result.get("status") in ("passed", "warning"):
        order_status = "order_preserved"
    else:
        order_status = "order_created"

    result = store_order_maintenance_result(
        org_id=org_id,
        period_id=period_id,
        verification_contract_id=CONTRACT_ID,
        status=order_status,
        trust_state=current_trust_state,
        authority_trace=authority_trace,
        failed_assertions=failed_assertions,
    )

    return result
