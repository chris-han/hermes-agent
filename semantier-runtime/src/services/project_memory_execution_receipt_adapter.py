"""Deterministic adapter from a source-owned execution receipt to project-memory features."""

from __future__ import annotations

from datetime import timedelta
import json

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_execution_receipt import (
    ExecutionFeatureDerivationV1,
    ExecutionReceiptSourceV1,
    StructuredEventFeatureV1,
    bind_structured_event_feature_identity,
)
from contracts.runtime_context_reconstruction import MemoryCandidateV1


_RECEIPT_KEY = "execution_receipt_v1"


def _payload(candidate: MemoryCandidateV1) -> dict:
    value = json.loads(candidate.payload.canonical_json)
    return value if isinstance(value, dict) else {}


def _candidate_source_role(candidate: MemoryCandidateV1) -> str:
    payload = _payload(candidate)
    role = payload.get("source_role") or payload.get("producer_role")
    if isinstance(role, str) and role:
        return role
    if candidate.asserted_governance.role == "empirical":
        return "execution_receipt"
    if candidate.asserted_governance.role == "binding":
        return "authority_record"
    return "project_chat"


def _candidate_pin(candidate: MemoryCandidateV1) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind="memory_candidate",
        artifact_ref=candidate.candidate_ref,
        artifact_hash=candidate.candidate_hash,
    )


def _derivation(feature: str, method: str, locator: str) -> dict:
    return ExecutionFeatureDerivationV1(
        feature=feature,
        method=method,
        source_locator=locator,
    ).model_dump(mode="json")


class ProjectMemoryExecutionReceiptAdapter:
    """Compile only exact source-owned execution receipt structure.

    The adapter does not parse free text, infer absent fields, create Evidence, or
    grant execution authority. It returns None when the bounded receipt contract
    is absent and fails closed when a claimed receipt is malformed.
    """

    def extract(
        self, *, candidate: MemoryCandidateV1
    ) -> StructuredEventFeatureV1 | None:
        payload = _payload(candidate)
        structured = payload.get("structured_fields")
        if not isinstance(structured, dict) or _RECEIPT_KEY not in structured:
            return None

        if _candidate_source_role(candidate) != "execution_receipt":
            raise ValueError(
                "execution_receipt_v1 requires execution_receipt source role"
            )
        if candidate.asserted_governance.role != "empirical":
            raise ValueError(
                "execution_receipt_v1 requires empirical asserted governance"
            )

        receipt = ExecutionReceiptSourceV1.model_validate(structured[_RECEIPT_KEY])
        source = next(
            (
                item
                for item in candidate.source_objects
                if item.object_ref == receipt.source_object_ref
                and item.locator == receipt.source_locator
            ),
            None,
        )
        if source is None:
            raise ValueError(
                "execution receipt source object does not match candidate lineage"
            )
        if source.object_kind != "event":
            raise ValueError("execution receipt source object must be an event")

        no_start = (
            "YES"
            if receipt.no_start is True
            else "NO"
            if receipt.no_start is False
            else "UNKNOWN"
        )
        if receipt.no_start is True:
            run_identity = "NO"
            first_step = "NO"
            run_method = "EXPLICIT_NEGATIVE_FIELD"
            first_step_method = "EXPLICIT_NEGATIVE_FIELD"
        else:
            run_identity = "YES" if receipt.run_id is not None else "UNKNOWN"
            first_step = "YES" if receipt.started_at is not None else "UNKNOWN"
            run_method = (
                "EXACT_STRUCTURED_FIELD" if receipt.run_id is not None else "UNRESOLVED"
            )
            first_step_method = (
                "EXACT_STRUCTURED_FIELD"
                if receipt.started_at is not None
                else "UNRESOLVED"
            )

        worker_assignment = "YES" if receipt.worker_id is not None else "UNKNOWN"
        activity_present = (
            "YES" if receipt.activity_observed_at is not None else "UNKNOWN"
        )
        if receipt.activity_observed_at is None:
            freshness = "UNKNOWN"
            freshness_method = "UNRESOLVED"
        else:
            age = receipt.observed_at - receipt.activity_observed_at
            window = timedelta(seconds=receipt.activity_freshness_window_seconds)
            freshness = "CURRENT" if age <= window else "STALE"
            freshness_method = "SOURCE_TIME_COMPARISON"

        if receipt.terminal_event is None:
            terminal_present = "UNKNOWN"
            terminal_kind = "UNKNOWN"
            terminal_time_present = "UNKNOWN"
            terminal_method = "UNRESOLVED"
        else:
            terminal_present = "YES"
            terminal_kind = receipt.terminal_event.kind
            terminal_time_present = "YES"
            terminal_method = "EXACT_STRUCTURED_FIELD"

        no_start_method = (
            "EXPLICIT_NEGATIVE_FIELD"
            if receipt.no_start is True
            else "EXACT_STRUCTURED_FIELD"
            if receipt.no_start is False
            else "UNRESOLVED"
        )
        locator = receipt.source_locator
        derivations = [
            _derivation("run_identity_present", run_method, locator),
            _derivation(
                "worker_assignment_present",
                "EXACT_STRUCTURED_FIELD"
                if receipt.worker_id is not None
                else "UNRESOLVED",
                locator,
            ),
            _derivation("first_step_time_present", first_step_method, locator),
            _derivation(
                "activity_signal_present",
                "EXACT_STRUCTURED_FIELD"
                if receipt.activity_observed_at is not None
                else "UNRESOLVED",
                locator,
            ),
            _derivation("activity_signal_freshness", freshness_method, locator),
            _derivation("terminal_event_present", terminal_method, locator),
            _derivation("terminal_event_kind", terminal_method, locator),
            _derivation("terminal_event_time_present", terminal_method, locator),
            _derivation("explicit_no_start", no_start_method, locator),
        ]

        feature_payload = {
            "schema_version": "structured_event_feature.v1",
            "memory_candidate": _candidate_pin(candidate).model_dump(mode="json"),
            "producer_schema": receipt.producer_schema,
            "source_object_ref": receipt.source_object_ref,
            "source_locator": receipt.source_locator,
            "observed_at": receipt.observed_at,
            "run_identity_present": run_identity,
            "worker_assignment_present": worker_assignment,
            "first_step_time_present": first_step,
            "activity_signal_present": activity_present,
            "activity_signal_freshness": freshness,
            "terminal_event_present": terminal_present,
            "terminal_event_kind": terminal_kind,
            "terminal_event_time_present": terminal_time_present,
            "explicit_no_start": no_start,
            "derivations": derivations,
            "compiler_profile": "execution-receipt-feature.v1",
        }
        return StructuredEventFeatureV1.model_validate(
            bind_structured_event_feature_identity(feature_payload)
        )
