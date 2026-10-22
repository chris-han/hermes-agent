"""Provider-neutral bounded discriminator for residual project-memory state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_state import (
    ProjectStateDerivationV1,
    TypedStateDecisionV1,
    bind_typed_state_decision_identity,
)
from eos.artifact_hashing import content_hash


FRONTIERS: dict[str, tuple[str, ...]] = {
    "epistemic": ("OBSERVED", "CLAIMED", "INFERRED"),
    "authority": ("PROPOSED", "UNAPPROVED", "APPROVED", "ACTIVATED", "UNKNOWN"),
    "execution": ("NOT_RUN", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"),
    "verification": ("NOT_RUN", "PASSED", "FAILED", "UNKNOWN"),
}


class CandidateScorer(Protocol):
    provider_ref: str
    model_ref: str
    profile_ref: str

    def score(
        self,
        *,
        text: str,
        source_role: str,
        epistemic_state: str,
        dimension: str,
        frontier: Sequence[str],
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class DiscriminatorResult:
    decision: TypedStateDecisionV1
    derivation: ProjectStateDerivationV1


def _sha(value: object) -> str:
    return "sha256:" + content_hash(value)


def _eligible_frontier(
    *,
    dimension: str,
    frontier: tuple[str, ...],
    source_role: str,
    epistemic_state: str,
) -> tuple[str, ...]:
    values = list(frontier)
    if dimension == "authority" and source_role != "authority_record":
        if epistemic_state == "CLAIMED":
            values = ["UNKNOWN"]
        else:
            values = [item for item in values if item not in {"APPROVED", "ACTIVATED"}]
    if dimension in {"execution", "verification"} and (
        source_role != "execution_receipt" or epistemic_state == "CLAIMED"
    ):
        values = ["UNKNOWN"]
    if "UNKNOWN" in frontier and "UNKNOWN" not in values:
        values.append("UNKNOWN")
    return tuple(values)


class ProjectStateDiscriminator:
    def __init__(self, scorer: CandidateScorer) -> None:
        self._scorer = scorer

    def decide(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        text: str,
        source_role: str,
        epistemic_state: str,
        dimension: str,
        source_locator_refs: tuple[str, ...] = (),
    ) -> DiscriminatorResult:
        if dimension not in FRONTIERS:
            raise ValueError(f"unsupported project-state dimension {dimension!r}")
        frontier = _eligible_frontier(
            dimension=dimension,
            frontier=FRONTIERS[dimension],
            source_role=source_role,
            epistemic_state=epistemic_state,
        )
        input_payload = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "text": text,
            "source_role": source_role,
            "epistemic_state": epistemic_state,
            "dimension": dimension,
        }
        input_hash = _sha(input_payload)
        frontier_hash = _sha({"dimension": dimension, "frontier": frontier})

        scores = dict(
            self._scorer.score(
                text=text,
                source_role=source_role,
                epistemic_state=epistemic_state,
                dimension=dimension,
                frontier=frontier,
            )
        )
        if set(scores) != set(frontier):
            raise ValueError("candidate scorer must return exactly the eligible frontier")
        if any(not isinstance(value, (float, int)) for value in scores.values()):
            raise ValueError("candidate scores must be numeric")

        selected = sorted(frontier, key=lambda item: (-float(scores[item]), frontier.index(item)))[0]
        abstained = selected == "UNKNOWN"
        # This is a relative candidate score, not calibrated confidence. Only emit
        # ppm when the scorer explicitly returns normalized probabilities.
        confidence_ppm = None
        total = sum(float(value) for value in scores.values())
        if all(float(value) >= 0 for value in scores.values()) and abs(total - 1.0) <= 1e-6:
            confidence_ppm = round(float(scores[selected]) * 1_000_000)

        payload = bind_typed_state_decision_identity(
            {
                "schema_version": "typed_state_decision.v1",
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "dimension": dimension,
                "selected_state": selected,
                "abstained": abstained,
                "confidence_ppm": confidence_ppm,
                "provider_ref": self._scorer.provider_ref,
                "model_ref": self._scorer.model_ref,
                "profile_ref": self._scorer.profile_ref,
                "input_hash": input_hash,
                "frontier_hash": frontier_hash,
            }
        )
        decision = TypedStateDecisionV1.model_validate(payload)
        pin = ArtifactPinV1(
            artifact_kind="typed_state_decision",
            artifact_ref=decision.decision_ref,
            artifact_hash=decision.decision_hash,
        )
        derivation = ProjectStateDerivationV1(
            dimension=dimension,
            method="BOUNDED_TYPED_DISCRIMINATOR",
            source_locator_refs=source_locator_refs,
            decision_receipt=pin,
            confidence_ppm=decision.confidence_ppm,
        )
        return DiscriminatorResult(decision=decision, derivation=derivation)
