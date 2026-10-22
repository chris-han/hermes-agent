"""Deterministic lowering from structured execution features to typed observations."""

from __future__ import annotations

from contracts.project_memory_execution_receipt import StructuredEventFeatureV1
from contracts.project_memory_factorized import ExecutionObservationV1


def lower_execution_observation(
    feature: StructuredEventFeatureV1,
) -> ExecutionObservationV1:
    """Lower source-owned execution features without semantic inference.

    Logical implications are allowed only when they follow from the frozen
    execution-receipt contract. Missing/unknown feature values remain UNKNOWN.
    """

    if feature.explicit_no_start == "YES":
        run_started = "NO"
    elif (
        feature.run_identity_present == "YES"
        or feature.first_step_time_present == "YES"
        or feature.activity_signal_present == "YES"
        or feature.terminal_event_present == "YES"
    ):
        run_started = "YES"
    else:
        run_started = "UNKNOWN"

    if feature.activity_signal_freshness == "CURRENT":
        activity_current = "YES"
    elif feature.activity_signal_freshness == "STALE":
        activity_current = "NO"
    else:
        activity_current = "UNKNOWN"

    if feature.terminal_event_kind == "SUCCESS":
        terminal_event = "SUCCESS"
    elif feature.terminal_event_kind == "FAILURE":
        terminal_event = "FAILURE"
    else:
        terminal_event = "UNKNOWN"

    return ExecutionObservationV1(
        run_started=run_started,
        activity_current=activity_current,
        terminal_event=terminal_event,
    )
