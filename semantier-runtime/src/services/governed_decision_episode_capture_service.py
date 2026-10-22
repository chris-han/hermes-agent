"""Governed W1 episode capture over existing derived-projection persistence."""

from __future__ import annotations

from datetime import datetime

from contracts.artifact_pin import ArtifactPinV1
from contracts.commitment_transition import CommitmentTransitionV1
from contracts.decision_graph_projection import (
    DecisionEvidenceViewV1,
    DecisionGraphProjectionV1,
    ProjectionSufficiencyProofV1,
)
from contracts.decision_verification import DecisionVerificationV1
from contracts.governed_decision_episode import GovernedDecisionEpisodeV1
from contracts.semantic_kernel import DecisionRecord, HumanResponsibilityContext
from contracts.typed_decision_model import (
    DecisionDistributionV1,
    TypedDecisionFrontierV1,
    TypedDecisionModelInputV1,
)
from semantier.cognitive_context.store import CognitiveContextStore


def _pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
    return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)


def _ordered_pins(values: tuple[ArtifactPinV1, ...]) -> tuple[ArtifactPinV1, ...]:
    return tuple(sorted(
        values,
        key=lambda pin: (
            pin.artifact_kind.encode("utf-8"),
            pin.artifact_ref.encode("utf-8"),
            pin.artifact_hash.encode("utf-8"),
        ),
    ))


class GovernedDecisionEpisodeCaptureService:
    """Capture replay evidence without acquiring label, decision, or commitment authority."""

    def __init__(self, *, store: CognitiveContextStore) -> None:
        if not isinstance(store, CognitiveContextStore):
            raise TypeError("episode capture requires CognitiveContextStore")
        self._store = store

    @staticmethod
    def _validate_projection_closure(
        *,
        dgp: DecisionGraphProjectionV1,
        sufficiency: ProjectionSufficiencyProofV1,
        evidence_view: DecisionEvidenceViewV1,
        frontier: TypedDecisionFrontierV1 | None,
    ) -> None:
        if sufficiency.status != "SUFFICIENT":
            raise ValueError("PROJECTION_INSUFFICIENT")
        if dgp.sufficiency_proof != sufficiency.pin():
            raise ValueError("EPISODE_DGP_SUFFICIENCY_BINDING_MISMATCH")
        if (
            sufficiency.frontier_ref,
            sufficiency.frontier_hash,
            sufficiency.dlp_ref,
            sufficiency.dlp_hash,
        ) != (
            dgp.frontier_ref,
            dgp.frontier_hash,
            dgp.source_dlp_ref,
            dgp.source_dlp_hash,
        ):
            raise ValueError("EPISODE_DGP_SUFFICIENCY_BASIS_MISMATCH")
        if evidence_view.decision_graph_projection != dgp.pin():
            raise ValueError("EPISODE_DGP_EVIDENCE_VIEW_BINDING_MISMATCH")
        if (
            evidence_view.source_dlp_ref,
            evidence_view.source_dlp_hash,
            evidence_view.model_input_hash,
        ) != (
            dgp.source_dlp_ref,
            dgp.source_dlp_hash,
            dgp.source_model_input_hash,
        ):
            raise ValueError("EPISODE_DLP_EVIDENCE_VIEW_BINDING_MISMATCH")
        if frontier is not None:
            if (
                frontier.frontier_ref,
                frontier.frontier_hash,
                frontier.dlp_ref,
                frontier.dlp_hash,
            ) != (
                dgp.frontier_ref,
                dgp.frontier_hash,
                dgp.source_dlp_ref,
                dgp.source_dlp_hash,
            ):
                raise ValueError("EPISODE_FRONTIER_DGP_BINDING_MISMATCH")

    @staticmethod
    def _validate_model_capture(
        *,
        dgp: DecisionGraphProjectionV1,
        frontier: TypedDecisionFrontierV1 | None,
        model_input: TypedDecisionModelInputV1 | None,
        distribution: DecisionDistributionV1 | None,
        verification: DecisionVerificationV1,
    ) -> tuple[
        ArtifactPinV1 | None,
        ArtifactPinV1 | None,
        ArtifactPinV1 | None,
        ArtifactPinV1 | None,
    ]:
        if distribution is None and model_input is None:
            if verification.scope == "INTERPRETED_PREDICATE":
                raise ValueError("INCOMPLETE_MODEL_CAPTURE_BINDINGS")
            return None, None, None, None
        if distribution is None or model_input is None or frontier is None:
            raise ValueError("INCOMPLETE_MODEL_CAPTURE_BINDINGS")
        if (
            model_input.dlp_ref,
            model_input.dlp_hash,
            model_input.dlp_model_input_hash,
            model_input.typed_decision_frontier.frontier_ref,
            model_input.typed_decision_frontier.frontier_hash,
        ) != (
            dgp.source_dlp_ref,
            dgp.source_dlp_hash,
            dgp.source_model_input_hash,
            frontier.frontier_ref,
            frontier.frontier_hash,
        ):
            raise ValueError("EPISODE_MODEL_INPUT_DGP_BINDING_MISMATCH")
        distribution.verify_against(model_input)
        if verification.status != "VERIFIED":
            raise ValueError("EPISODE_VERIFIED_DECISION_REQUIRED")
        if verification.scope == "INTERPRETED_PREDICATE":
            expected_projection = _pin(
                "decision_local_projection",
                dgp.source_dlp_ref,
                dgp.source_dlp_hash,
            )
            expected_frontier = _pin(
                "typed_decision_frontier",
                frontier.frontier_ref,
                frontier.frontier_hash,
            )
            expected_input = _pin(
                "typed_decision_model_input",
                model_input.input_ref,
                model_input.input_hash,
            )
            expected_distribution = _pin(
                "decision_distribution",
                distribution.distribution_ref,
                distribution.distribution_hash,
            )
            if (
                verification.projection,
                verification.frontier,
                verification.model_input,
                verification.distribution,
            ) != (
                expected_projection,
                expected_frontier,
                expected_input,
                expected_distribution,
            ):
                raise ValueError("EPISODE_VERIFICATION_MODEL_BINDING_MISMATCH")
        return (
            _pin("model_provider", distribution.producer_ref, distribution.producer_hash),
            _pin(
                "model_config",
                distribution.producer_config_ref,
                distribution.producer_config_hash,
            ),
            _pin(
                "decision_distribution",
                distribution.distribution_ref,
                distribution.distribution_hash,
            ),
            _pin("typed_decision_model_input", model_input.input_ref, model_input.input_hash),
        )

    @staticmethod
    def _commitment_bindings(
        *,
        decision_record: DecisionRecord,
        commitment_transition: CommitmentTransitionV1 | None,
        responsibility: HumanResponsibilityContext | None,
    ) -> dict:
        if commitment_transition is None:
            if responsibility is not None:
                raise ValueError("COMMITMENT_BINDINGS_WITHOUT_TRANSITION")
            return {
                "commitment_transition": None,
                "responsibility_anchor": None,
                "commitment_authority_basis_hash": None,
                "commitment_scope_ref": None,
                "commitment_effective_from": None,
                "commitment_effective_until": None,
            }
        if responsibility is None:
            raise ValueError("INCOMPLETE_COMMITMENT_BINDINGS")
        if commitment_transition.decision != decision_record.pin():
            raise ValueError("EPISODE_COMMITMENT_DECISION_MISMATCH")
        if commitment_transition.responsibility_anchor != responsibility.pin():
            raise ValueError("EPISODE_COMMITMENT_RESPONSIBILITY_MISMATCH")
        return {
            "commitment_transition": commitment_transition.pin(),
            "responsibility_anchor": responsibility.pin(),
            "commitment_authority_basis_hash": commitment_transition.authority_basis_hash,
            "commitment_scope_ref": commitment_transition.scope_ref,
            "commitment_effective_from": commitment_transition.effective_from,
            "commitment_effective_until": commitment_transition.effective_until,
        }

    def capture(
        self,
        *,
        descriptive_basis: tuple[ArtifactPinV1, ...],
        normative_basis: tuple[ArtifactPinV1, ...],
        epistemic_basis: tuple[ArtifactPinV1, ...],
        dgp: DecisionGraphProjectionV1,
        sufficiency: ProjectionSufficiencyProofV1,
        evidence_view: DecisionEvidenceViewV1,
        verification: DecisionVerificationV1,
        decision_record: DecisionRecord,
        captured_at: datetime,
        frontier: TypedDecisionFrontierV1 | None = None,
        model_input: TypedDecisionModelInputV1 | None = None,
        distribution: DecisionDistributionV1 | None = None,
        deterministic_receipts: tuple[ArtifactPinV1, ...] = (),
        commitment_transition: CommitmentTransitionV1 | None = None,
        responsibility: HumanResponsibilityContext | None = None,
        observed_outcome: ArtifactPinV1 | None = None,
        acquisition_reasons: tuple[str, ...] = (),
        qualified_representation: ArtifactPinV1 | None = None,
    ) -> GovernedDecisionEpisodeV1:
        self._validate_projection_closure(
            dgp=dgp,
            sufficiency=sufficiency,
            evidence_view=evidence_view,
            frontier=frontier,
        )
        if verification.status != "VERIFIED":
            raise ValueError("EPISODE_VERIFIED_DECISION_REQUIRED")
        if (decision_record.organization_id, decision_record.workspace_id) != (
            dgp.organization_id,
            dgp.workspace_id,
        ):
            raise ValueError("EPISODE_SCOPE_MISMATCH")
        if decision_record.context != dgp.canonical_context or decision_record.ontology != dgp.ontology:
            raise ValueError("EPISODE_DECISION_BASIS_MISMATCH")
        if verification.semantic_result is not None and decision_record.result_code != verification.semantic_result:
            raise ValueError("EPISODE_DECISION_VERIFICATION_RESULT_MISMATCH")

        provider, config, prediction, serialized_input = self._validate_model_capture(
            dgp=dgp,
            frontier=frontier,
            model_input=model_input,
            distribution=distribution,
            verification=verification,
        )
        commitment = self._commitment_bindings(
            decision_record=decision_record,
            commitment_transition=commitment_transition,
            responsibility=responsibility,
        )

        episode = GovernedDecisionEpisodeV1.create(
            organization_id=dgp.organization_id,
            workspace_id=dgp.workspace_id,
            descriptive_basis=_ordered_pins(descriptive_basis),
            normative_basis=_ordered_pins(normative_basis),
            epistemic_basis=_ordered_pins(epistemic_basis),
            decision_graph_projection=dgp.pin(),
            projection_sufficiency_proof=sufficiency.pin(),
            decision_evidence_view=evidence_view.pin(),
            typed_decision_frontier=_pin(
                "typed_decision_frontier",
                dgp.frontier_ref,
                dgp.frontier_hash,
            ),
            qualified_representation=qualified_representation,
            serialized_model_input=serialized_input,
            model_provider=provider,
            model_config_pin=config,
            model_prediction=prediction,
            model_evidence=((prediction,) if prediction is not None else ()),
            deterministic_receipts=_ordered_pins(deterministic_receipts),
            decision_verification=verification.pin(),
            decision_record=decision_record.pin(),
            acquisition_reasons=tuple(
                sorted(set(acquisition_reasons), key=lambda value: value.encode("utf-8"))
            ),
            observed_outcome=observed_outcome,
            captured_at=captured_at,
            **commitment,
        )
        return self._store.save_or_get_derived_projection(episode)

    def replay(self, *, episode_ref: str, organization_id: str, workspace_id: str) -> GovernedDecisionEpisodeV1:
        return self._store.get_derived_projection(
            episode_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=GovernedDecisionEpisodeV1,
        )
