"""Bounded Loop-1 candidate generation over reconstructed epistemic history.

This module deliberately owns no persistence, evaluation, admission, activation, or
self-modifying search policy. It accepts ephemeral ideation drafts, filters them
against a frozen search/evaluation boundary, and returns object-level proposal
specifications that can be passed to the existing ChangeCandidate lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ChangeCandidate,
    ChangeDelta,
    ChangeOrigin,
    ContextGraphOrigin,
    EvaluationBasis,
    PreContextOrigin,
    WorkflowEditOrigin,
)
from eos.artifact_hashing import content_hash
from services.change_candidate_service import EpistemicLearningProjectionV1


ObjectChangeSurface = Literal[
    "OBJECT_ONTOLOGY",
    "OBJECT_POLICY",
    "OBJECT_CONTEXT_FORMATION",
    "OBJECT_REASONING",
]

LearningSystemChangeSurface = Literal[
    "LEARNING_CANDIDATE_GENERATION",
    "LEARNING_SEARCH_POLICY",
    "LEARNING_ROUTING",
    "LEARNING_EPISTEMIC_RECONSTRUCTION",
    "LEARNING_EVALUATION_POLICY",
]

ChangeSurface = ObjectChangeSurface | LearningSystemChangeSurface
SearchLane = Literal["REFINE", "ALTERNATIVE"]
LearningScope = Literal["OBJECT", "LEARNING_SYSTEM"]

_OBJECT_SURFACE_TARGET = {
    "OBJECT_ONTOLOGY": "ONTOLOGY",
    "OBJECT_POLICY": "POLICY",
    "OBJECT_CONTEXT_FORMATION": "CONTEXT_FORMATION",
    "OBJECT_REASONING": "REASONING",
}
_LEARNING_SYSTEM_SURFACES = {
    "LEARNING_CANDIDATE_GENERATION",
    "LEARNING_SEARCH_POLICY",
    "LEARNING_ROUTING",
    "LEARNING_EPISTEMIC_RECONSTRUCTION",
    "LEARNING_EVALUATION_POLICY",
}


class CandidateGenerationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CandidateDraft:
    """Ephemeral ideation output before Semantier admits it into Loop 1."""

    change_surface: ChangeSurface
    search_lane: SearchLane
    target: Literal["ONTOLOGY", "POLICY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW"]
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    hypothesis: str
    delta: ChangeDelta
    source_candidate_refs: tuple[ArtifactPinV1, ...] = ()


@dataclass(frozen=True)
class CandidateSearchPolicyV1:
    """Frozen per-campaign bounded search policy.

    The pin must already be present in EvaluationBasis.config_pins. The policy
    cannot update or replace itself during a Loop-1 campaign.
    """

    policy_pin: ArtifactPinV1
    max_candidates_per_round: int = 2
    max_rounds: int = 2
    max_candidates_total: int = 4
    lanes: tuple[SearchLane, ...] = ("REFINE", "ALTERNATIVE")

    def __post_init__(self) -> None:
        if self.policy_pin.artifact_kind != "candidate_search_policy":
            raise CandidateGenerationError("INVALID_SEARCH_POLICY_PIN")
        if self.max_candidates_per_round < 1:
            raise CandidateGenerationError("INVALID_MAX_CANDIDATES_PER_ROUND")
        if self.max_rounds < 1:
            raise CandidateGenerationError("INVALID_MAX_ROUNDS")
        if self.max_candidates_total < 1:
            raise CandidateGenerationError("INVALID_MAX_CANDIDATES_TOTAL")
        if len(self.lanes) != len(set(self.lanes)):
            raise CandidateGenerationError("DUPLICATE_SEARCH_LANE")


@dataclass(frozen=True)
class CandidateProposalSpec:
    """Validated object-level proposal ready for the existing ChangeCandidate seam."""

    change_surface: ObjectChangeSurface
    search_lane: SearchLane
    origin: ChangeOrigin
    target: Literal["ONTOLOGY", "POLICY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW"]
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    hypothesis: str
    delta: ChangeDelta
    evaluation_basis: EvaluationBasis
    search_policy_pin: ArtifactPinV1
    source_candidate_refs: tuple[ArtifactPinV1, ...]
    unresolved_question: str
    experiment_fingerprint: str

    def change_candidate_kwargs(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        matched_cases: tuple[ArtifactPinV1, ...],
        independent_holdout_cases: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
    ) -> dict[str, object]:
        return {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "origin": self.origin,
            "failure_class": self.failure_class,
            "target": self.target,
            "hypothesis": self.hypothesis,
            "evaluation_basis": self.evaluation_basis,
            "delta": self.delta,
            "matched_cases": matched_cases,
            "independent_holdout_cases": independent_holdout_cases,
            "recorded_at": recorded_at,
        }


@dataclass(frozen=True)
class CandidateGenerationResult:
    selected: tuple[CandidateProposalSpec, ...]
    deferred_meta: tuple[CandidateDraft, ...]
    suppressed_fingerprints: tuple[str, ...]
    stop_reason: str | None = None


class CandidateScopeClassifier:
    """Classify by change ownership, not by the existing ChangeCandidate target enum."""

    @staticmethod
    def classify(draft: CandidateDraft) -> LearningScope:
        if draft.change_surface in _LEARNING_SYSTEM_SURFACES:
            return "LEARNING_SYSTEM"
        expected_target = _OBJECT_SURFACE_TARGET.get(draft.change_surface)
        if expected_target is None:
            raise CandidateGenerationError("UNKNOWN_CHANGE_SURFACE")
        if draft.target != expected_target:
            raise CandidateGenerationError("SURFACE_TARGET_MISMATCH")
        if draft.target == "ONTOLOGY" and draft.failure_class != "STRUCTURE":
            raise CandidateGenerationError("ONTOLOGY_CHANGE_REQUIRES_STRUCTURE")
        if draft.target == "POLICY" and draft.failure_class != "STRUCTURE":
            raise CandidateGenerationError("POLICY_CHANGE_REQUIRES_STRUCTURE")
        if draft.delta.claims_semantic_graph_change and draft.failure_class != "STRUCTURE":
            raise CandidateGenerationError("SEMANTIC_GRAPH_CHANGE_REQUIRES_STRUCTURE")
        if not draft.hypothesis.strip():
            raise CandidateGenerationError("EMPTY_HYPOTHESIS")
        return "OBJECT"


class BoundedCandidateGenerationService:
    """Stateless Loop-1 gate over agent/LLM candidate drafts."""

    def __init__(self) -> None:
        self._classifier = CandidateScopeClassifier()

    @staticmethod
    def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
        return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash

    @staticmethod
    def _origin_pin(origin: ChangeOrigin) -> ArtifactPinV1:
        if isinstance(origin, ContextGraphOrigin):
            return origin.context
        if isinstance(origin, PreContextOrigin):
            return origin.formation_input.pin()
        if isinstance(origin, WorkflowEditOrigin):
            return origin.workflow
        raise CandidateGenerationError("INVALID_ORIGIN")

    @classmethod
    def _fingerprint_values(
        cls,
        *,
        origin: ChangeOrigin,
        target: str,
        failure_class: str,
        delta: ChangeDelta,
        evaluation_basis: EvaluationBasis,
    ) -> str:
        payload = {
            "origin": cls._origin_pin(origin).model_dump(mode="json"),
            "target": target,
            "failure_class": failure_class,
            "delta": delta.model_dump(mode="json"),
            "evaluation_basis": evaluation_basis.model_dump(mode="json"),
        }
        return "sha256:" + content_hash(payload)

    @classmethod
    def _candidate_fingerprint(cls, candidate: ChangeCandidate) -> str:
        return cls._fingerprint_values(
            origin=candidate.origin,
            target=candidate.target,
            failure_class=candidate.failure_class,
            delta=candidate.delta,
            evaluation_basis=candidate.evaluation_basis,
        )

    @staticmethod
    def _projection_candidate_pins(
        projection: EpistemicLearningProjectionV1,
    ) -> set[tuple[str, str, str]]:
        return {
            (candidate.pin().artifact_kind, candidate.pin().artifact_ref, candidate.pin().artifact_hash)
            for candidate in (
                *projection.accepted_candidates,
                *projection.rejected_candidates,
                *projection.pending_candidates,
            )
        }

    def generate(
        self,
        *,
        projection: EpistemicLearningProjectionV1,
        origin: ChangeOrigin,
        evaluation_basis: EvaluationBasis,
        policy: CandidateSearchPolicyV1,
        drafts: tuple[CandidateDraft, ...],
        round_index: int,
        tested_candidate_count: int,
    ) -> CandidateGenerationResult:
        if round_index < 1:
            raise CandidateGenerationError("INVALID_ROUND_INDEX")
        if tested_candidate_count < 0:
            raise CandidateGenerationError("INVALID_TESTED_CANDIDATE_COUNT")
        if round_index > policy.max_rounds:
            return CandidateGenerationResult((), (), (), "MAX_ROUNDS_REACHED")
        if tested_candidate_count >= policy.max_candidates_total:
            return CandidateGenerationResult((), (), (), "MAX_CANDIDATES_REACHED")
        if policy.policy_pin not in evaluation_basis.config_pins:
            raise CandidateGenerationError("SEARCH_POLICY_NOT_PINNED")

        origin_pin = self._origin_pin(origin)
        if projection.current_context is not None and origin_pin != projection.current_context:
            raise CandidateGenerationError("ORIGIN_CONTEXT_MISMATCH")
        if not projection.unresolved_questions:
            return CandidateGenerationResult((), (), (), "NO_UNRESOLVED_QUESTION")
        unresolved_question = projection.unresolved_questions[0]

        visible_candidate_pins = self._projection_candidate_pins(projection)
        failed_fingerprints = {
            self._candidate_fingerprint(candidate)
            for candidate in projection.rejected_candidates
        }

        meta: list[CandidateDraft] = []
        suppressed: list[str] = []
        by_lane: dict[str, list[CandidateProposalSpec]] = {lane: [] for lane in policy.lanes}

        for draft in drafts:
            scope = self._classifier.classify(draft)
            if scope == "LEARNING_SYSTEM":
                meta.append(draft)
                continue
            if draft.search_lane not in policy.lanes:
                raise CandidateGenerationError("SEARCH_LANE_NOT_ALLOWED")
            for ref in draft.source_candidate_refs:
                if self._pin_key(ref) not in visible_candidate_pins:
                    raise CandidateGenerationError("SOURCE_CANDIDATE_OUTSIDE_PROJECTION")

            fingerprint = self._fingerprint_values(
                origin=origin,
                target=draft.target,
                failure_class=draft.failure_class,
                delta=draft.delta,
                evaluation_basis=evaluation_basis,
            )
            if fingerprint in failed_fingerprints:
                suppressed.append(fingerprint)
                continue

            proposal = CandidateProposalSpec(
                change_surface=draft.change_surface,
                search_lane=draft.search_lane,
                origin=origin,
                target=draft.target,
                failure_class=draft.failure_class,
                hypothesis=draft.hypothesis,
                delta=draft.delta,
                evaluation_basis=evaluation_basis,
                search_policy_pin=policy.policy_pin,
                source_candidate_refs=draft.source_candidate_refs,
                unresolved_question=unresolved_question,
                experiment_fingerprint=fingerprint,
            )
            by_lane[draft.search_lane].append(proposal)

        selected: list[CandidateProposalSpec] = []
        for lane in policy.lanes:
            candidates = sorted(
                by_lane[lane],
                key=lambda item: (item.experiment_fingerprint, item.hypothesis),
            )
            if candidates:
                selected.append(candidates[0])

        remaining_budget = max(0, policy.max_candidates_total - tested_candidate_count)
        limit = min(policy.max_candidates_per_round, remaining_budget)
        selected = selected[:limit]

        stop_reason = None
        if not selected:
            stop_reason = "NO_ADMISSIBLE_CANDIDATE"

        return CandidateGenerationResult(
            selected=tuple(selected),
            deferred_meta=tuple(meta),
            suppressed_fingerprints=tuple(sorted(set(suppressed))),
            stop_reason=stop_reason,
        )


__all__ = [
    "BoundedCandidateGenerationService",
    "CandidateDraft",
    "CandidateGenerationError",
    "CandidateGenerationResult",
    "CandidateProposalSpec",
    "CandidateScopeClassifier",
    "CandidateSearchPolicyV1",
]
