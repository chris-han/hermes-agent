"""Deterministic W2 acquisition policy over governed W1 episodes."""

from __future__ import annotations

from hashlib import sha256

from contracts.episode_acquisition import AcquisitionDecisionV1, AcquisitionPolicyV1, AcquisitionSignalSetV1
from contracts.governed_decision_episode import GovernedDecisionEpisodeV1
from semantier.cognitive_context.store import CognitiveContextStore


class EpisodeAcquisitionService:
    """Select episodes for downstream learning; never creates labels or ChangeCandidates."""

    def __init__(self, *, store: CognitiveContextStore, policy: AcquisitionPolicyV1) -> None:
        if not isinstance(store, CognitiveContextStore):
            raise TypeError("acquisition requires CognitiveContextStore")
        if not isinstance(policy, AcquisitionPolicyV1):
            raise TypeError("acquisition requires AcquisitionPolicyV1")
        self._store = store
        self._policy = policy

    def _audit_selected(self, *, episode: GovernedDecisionEpisodeV1, audit_index: int) -> tuple[bool, int]:
        if audit_index < 0:
            raise ValueError("audit_index must be nonnegative")
        digest = sha256(
            f"{self._policy.seed}|{episode.episode_ref}|{episode.episode_hash}".encode("utf-8")
        ).digest()
        rank = int.from_bytes(digest[:8], "big")
        threshold = int(self._policy.random_audit_rate * (2**64 - 1))
        selected = (
            self._policy.random_audit_budget > 0
            and audit_index < self._policy.random_audit_budget
            and rank <= threshold
        )
        return selected, rank

    def evaluate(
        self,
        *,
        episode: GovernedDecisionEpisodeV1,
        audit_index: int,
        typed_failure: bool = False,
        conflict: bool = False,
        projection_insufficient: bool = False,
        representation_insufficient: bool = False,
        model_envelope_insufficient: bool = False,
        frontier_insufficient: bool = False,
        novelty: bool = False,
        uncertainty: float | None = None,
        prospective_impact: bool = False,
        commitment_divergence: bool = False,
        protected_holdout: bool = False,
        already_labeled: bool = False,
    ) -> AcquisitionDecisionV1:
        if not isinstance(episode, GovernedDecisionEpisodeV1):
            raise TypeError("acquisition requires GovernedDecisionEpisodeV1")
        signals = AcquisitionSignalSetV1.create(
            typed_failure=typed_failure,
            conflict=conflict,
            projection_insufficient=projection_insufficient,
            representation_insufficient=representation_insufficient,
            model_envelope_insufficient=model_envelope_insufficient,
            frontier_insufficient=frontier_insufficient,
            novelty=novelty,
            uncertainty=uncertainty,
            prospective_impact=prospective_impact,
            commitment_divergence=commitment_divergence,
            protected_holdout=protected_holdout,
            already_labeled=already_labeled,
        )

        excluded_reason = None
        if signals.protected_holdout:
            excluded_reason = "PROTECTED_HOLDOUT"
        elif signals.already_labeled:
            excluded_reason = "ALREADY_LABELED"

        reasons: list[str] = []
        mapping = (
            (signals.typed_failure, "TYPED_FAILURE"),
            (signals.conflict, "CONFLICT"),
            (signals.projection_insufficient, "PROJECTION_INSUFFICIENT"),
            (signals.representation_insufficient, "REPRESENTATION_INSUFFICIENT"),
            (signals.model_envelope_insufficient, "MODEL_ENVELOPE_INSUFFICIENT"),
            (signals.frontier_insufficient, "FRONTIER_INSUFFICIENT"),
            (signals.novelty, "NOVELTY"),
            (signals.prospective_impact, "PROSPECTIVE_IMPACT"),
            (signals.commitment_divergence, "COMMITMENT_DIVERGENCE"),
        )
        reasons.extend(reason for enabled, reason in mapping if enabled)
        if (
            signals.uncertainty is not None
            and signals.uncertainty >= self._policy.uncertainty_threshold
        ):
            reasons.append("UNCERTAINTY")

        audit_selected, audit_rank = self._audit_selected(
            episode=episode,
            audit_index=audit_index,
        )
        if audit_selected:
            reasons.append("RANDOM_AUDIT")

        if excluded_reason is not None:
            reasons = []

        decision = AcquisitionDecisionV1.create(
            organization_id=episode.organization_id,
            workspace_id=episode.workspace_id,
            episode_ref=episode.episode_ref,
            episode_hash=episode.episode_hash,
            policy_ref=self._policy.policy_ref,
            policy_hash=self._policy.policy_hash,
            signal_ref=signals.signal_ref,
            signal_hash=signals.signal_hash,
            audit_index=audit_index,
            eligible=bool(reasons),
            reasons=tuple(reasons),
            excluded_reason=excluded_reason,
            random_audit_rank=audit_rank,
            random_audit_selected=("RANDOM_AUDIT" in reasons),
        )
        return self._store.save_or_get_derived_projection(decision)

    def replay(
        self,
        *,
        receipt_ref: str,
        organization_id: str,
        workspace_id: str,
    ) -> AcquisitionDecisionV1:
        return self._store.get_derived_projection(
            receipt_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=AcquisitionDecisionV1,
        )
