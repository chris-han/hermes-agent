"""Deterministic composition for factorized project-memory observations."""

from __future__ import annotations

from contracts.project_memory_factorized import (
    AuthorityObservationV1,
    ExecutionObservationV1,
)


def compose_authority_state(
    observation: AuthorityObservationV1,
    *,
    source_role: str,
    epistemic_state: str,
) -> str:
    if source_role != "authority_record" or epistemic_state == "CLAIMED":
        if observation.proposal_present == "YES":
            return "PROPOSED"
        return "UNKNOWN"
    if (
        observation.authorization_present == "YES"
        and observation.authorization_effective == "YES"
    ):
        return "ACTIVATED"
    if observation.authorization_present == "YES":
        return "APPROVED"
    if observation.required_authorization_absent == "YES":
        return "UNAPPROVED"
    if observation.proposal_present == "YES":
        return "PROPOSED"
    return "UNKNOWN"


def compose_execution_state(
    observation: ExecutionObservationV1,
    *,
    source_role: str,
    epistemic_state: str,
) -> str:
    if source_role != "execution_receipt" or epistemic_state == "CLAIMED":
        return "UNKNOWN"
    if observation.terminal_event == "FAILURE":
        return "FAILED"
    if observation.terminal_event == "SUCCESS":
        return "SUCCEEDED"
    if observation.run_started == "NO":
        return "NOT_RUN"
    if (
        observation.run_started == "YES"
        and observation.activity_current == "YES"
        and observation.terminal_event == "NONE"
    ):
        return "RUNNING"
    return "UNKNOWN"
