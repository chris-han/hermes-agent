"""Pure retained verification for advisory trial receipts and derived candidates."""

from __future__ import annotations

from datetime import timedelta
from typing import Mapping

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedSourceV1
from contracts.runtime_learning_trial import AdvisoryTrialVariantV1

from .common import (
    PinKey,
    _require_supplied_source,
    _validate_closure,
)
from .compiler import _candidate_value, _resolve, _trial_receipt_value


def _validate_receipt_relationship(
    receipt: AdvisoryTrialVariantV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> None:
    predecessor_source = _resolve(closure, receipt.source)
    source_candidate, source_context, _ = _candidate_value(predecessor_source, closure)
    if (
        receipt.source.artifact_ref != source_candidate.candidate_set_ref
        or receipt.source.artifact_hash != source_candidate.candidate_set_hash
        or receipt.inputs.artifact_ref != source_context.inputs.input_ref
        or receipt.inputs.artifact_hash != source_context.inputs.input_hash
        or receipt.organization_id != source_candidate.organization_id
        or receipt.workspace_id != source_candidate.workspace_id
    ):
        raise ValueError("trial receipt does not match its native source and input")
    if receipt.variant == "ADD_MISSING_FRAME":
        if receipt.recorded_at != source_context.inputs.recorded_at + timedelta(
            seconds=2
        ):
            raise ValueError("repair receipt has unsupported saved time")
        if (
            source_candidate.compiler_ref
            != source_context.inputs.resource("semantic_compiler").resource_ref
            + "#advisory-trial-variant.v1"
            or len(predecessor_source.dependencies) != 1
            or predecessor_source.dependencies[0].artifact_kind
            != "advisory_trial_variant"
        ):
            raise ValueError("repair receipt requires a native trial predecessor")
        prior_receipt, _ = _trial_receipt_value(
            _resolve(closure, predecessor_source.dependencies[0]), closure
        )
        if prior_receipt.variant != "OMIT_EXCEPTION":
            raise ValueError(
                "repair receipt requires the omitted-exception predecessor"
            )
        original, _context, _ = _candidate_value(
            _resolve(closure, prior_receipt.source), closure
        )
        exceptions = tuple(
            frame for frame in original.frames if frame.frame_kind == "exception"
        )
        if len(exceptions) != 1 or receipt.added_frame != exceptions[0]:
            raise ValueError("repair receipt added frame is not the original exception")
    else:
        if (
            source_candidate.compiler_ref
            != source_context.inputs.resource("semantic_compiler").resource_ref
            or source_candidate.change_cause != "initial_compilation"
            or source_candidate.predecessor_candidate_set_ref is not None
            or source_candidate.predecessor_candidate_set_hash is not None
            or source_candidate.triggering_ref is not None
        ):
            raise ValueError("trial receipt source is not an initial native candidate")
        if receipt.recorded_at != source_context.inputs.recorded_at + timedelta(
            seconds=1
        ):
            raise ValueError("trial receipt has unsupported saved time")


def verify_advisory_trial_variant_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one native trial receipt without fetching its consuming candidate."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    receipt, dependencies = _trial_receipt_value(source, closure)
    _validate_receipt_relationship(receipt, closure)
    return dependencies


__all__ = ["verify_advisory_trial_variant_v1"]
