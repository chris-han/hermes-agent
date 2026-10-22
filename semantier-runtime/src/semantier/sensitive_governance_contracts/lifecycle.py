from __future__ import annotations

from semantier.sensitive_governance_contracts.enums import (
    AssessmentState,
    RevealLeaseState,
    RunState,
)

ASSESSMENT_TRANSITIONS = {
    AssessmentState.ESCALATED: {
        AssessmentState.PREPARED,
        AssessmentState.BLOCKED,
    },
    AssessmentState.PREPARED: {
        AssessmentState.CONSUMED,
        AssessmentState.EXPIRED,
        AssessmentState.BLOCKED,
    },
    AssessmentState.CONSUMED: set(),
    AssessmentState.EXPIRED: set(),
    AssessmentState.BLOCKED: set(),
}

RUN_TRANSITIONS = {
    RunState.PREPARED: {RunState.SUCCEEDED, RunState.FAILED},
    RunState.SUCCEEDED: set(),
    RunState.FAILED: set(),
}

REVEAL_LEASE_TRANSITIONS = {
    RevealLeaseState.ISSUED: {
        RevealLeaseState.EXPIRED,
        RevealLeaseState.REVOKED,
        RevealLeaseState.CONSUMED,
    },
    RevealLeaseState.EXPIRED: set(),
    RevealLeaseState.REVOKED: set(),
    RevealLeaseState.CONSUMED: set(),
}


def transition_allowed(current: str, next_state: str, transitions: dict) -> bool:
    current_state = next(iter(state for state in transitions if state.value == current))
    target_state = next(iter(state for state in transitions if state.value == next_state))
    return target_state in transitions[current_state]
