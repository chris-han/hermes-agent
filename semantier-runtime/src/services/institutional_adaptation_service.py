"""Deterministic pre-learning triage and cross-context promotion gates.

The service consumes already-attributed observations and produces derived evidence.
It never admits or activates policy/workflow changes.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

from contracts.artifact_pin import ArtifactPinV1
from contracts.institutional_adaptation import (
    AdaptationIndependenceClusterV1,
    AdaptationObservationV1,
    AdaptationTriageReceiptV1,
    CrossContextAdaptationHypothesisV1,
    DesignPressureOpportunityV1,
    InstitutionalAdaptationIngressV1,
    InstitutionalDivergenceDetectionV1,
    SemanticAdaptationClusterV1,
    TriageDisposition,
)


_POSITIVE = {
    "MANDATED_LOCALIZATION",
    "AUTHORITY_LOCALIZATION",
    "BENEFICIAL_LOCAL_OPTIMIZATION",
}


class InstitutionalAdaptationService:
    """Owns no authority; enforces Design-Law-18 evidence semantics."""

    @staticmethod
    def detect_divergence(
        *,
        organization_id: str,
        workspace_id: str,
        global_baseline: ArtifactPinV1,
        local_successor: ArtifactPinV1,
        semantic_delta: ArtifactPinV1,
        changed_dimensions: tuple[str, ...] = (),
        presentation_equivalent: bool = False,
        detected_at: datetime,
    ) -> InstitutionalDivergenceDetectionV1:
        if global_baseline == local_successor:
            classification = "NO_DIVERGENCE"
            reasons = ("IDENTICAL_BASELINE_SUCCESSOR",)
            presentation_equivalent = True
            changed_dimensions = ()
        elif changed_dimensions:
            classification = "MATERIAL_INSTITUTIONAL_DIVERGENCE"
            reasons = tuple(f"MATERIAL_{item}" for item in sorted(set(changed_dimensions)))
            presentation_equivalent = False
        elif presentation_equivalent:
            classification = "NON_MATERIAL_DIVERGENCE"
            reasons = ("PRESENTATION_EQUIVALENT_ONLY",)
        else:
            classification = "UNRESOLVED_MATERIALITY"
            reasons = ("UNCLASSIFIED_SEMANTIC_DELTA",)

        return InstitutionalDivergenceDetectionV1.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            global_baseline=global_baseline,
            local_successor=local_successor,
            semantic_delta=semantic_delta,
            changed_dimensions=changed_dimensions,
            presentation_equivalent=presentation_equivalent,
            classification=classification,
            reason_codes=reasons,
            detected_at=detected_at,
        )

    @staticmethod
    def triage(
        *,
        observation: AdaptationObservationV1,
        disposition: TriageDisposition,
        reason_codes: tuple[str, ...],
        evidence_refs: tuple[ArtifactPinV1, ...] = (),
        protected_invariant_refs: tuple[ArtifactPinV1, ...] = (),
        recorded_at: datetime,
    ) -> AdaptationTriageReceiptV1:
        if disposition == "INSUFFICIENT_ATTRIBUTION":
            learning_eligible = False
            eligible_scope = "NONE"
            review_required = True
        elif disposition == "CONTEXTUAL_PREFERENCE":
            learning_eligible = True
            eligible_scope = "LOCAL"
            review_required = False
        elif disposition == "POLICY_EROSION":
            learning_eligible = True
            eligible_scope = "CROSS_CONTEXT_INPUT"
            review_required = True
        else:
            learning_eligible = True
            eligible_scope = "CROSS_CONTEXT_INPUT"
            review_required = disposition == "AUTHORITY_LOCALIZATION"

        return AdaptationTriageReceiptV1.create(
            adaptation_observation=observation.pin(),
            disposition=disposition,
            reason_codes=reason_codes,
            evidence_refs=evidence_refs,
            protected_invariant_refs=protected_invariant_refs,
            learning_eligible=learning_eligible,
            eligible_scope=eligible_scope,
            review_required=review_required,
            recorded_at=recorded_at,
        )

    @staticmethod
    def bind_ingress(
        *,
        acquisition: ArtifactPinV1,
        divergence_detection: InstitutionalDivergenceDetectionV1,
        observation: AdaptationObservationV1,
        triage_receipt: AdaptationTriageReceiptV1,
        recorded_at: datetime,
    ) -> InstitutionalAdaptationIngressV1:
        if acquisition.artifact_kind != "acquisition_decision":
            raise ValueError("ADAPTATION_INGRESS_REQUIRES_ACQUISITION")
        if divergence_detection.classification != "MATERIAL_INSTITUTIONAL_DIVERGENCE":
            raise ValueError("ADAPTATION_INGRESS_REQUIRES_MATERIAL_DIVERGENCE")
        if (
            divergence_detection.organization_id,
            divergence_detection.workspace_id,
            divergence_detection.global_baseline,
            divergence_detection.local_successor,
            divergence_detection.semantic_delta,
        ) != (
            observation.organization_id,
            observation.workspace_id,
            observation.global_baseline,
            observation.local_successor,
            observation.semantic_delta,
        ):
            raise ValueError("ADAPTATION_INGRESS_DETECTION_OBSERVATION_MISMATCH")
        if triage_receipt.adaptation_observation != observation.pin():
            raise ValueError("ADAPTATION_INGRESS_TRIAGE_OBSERVATION_MISMATCH")

        route = {
            "MANDATED_LOCALIZATION": "ATTRIBUTE",
            "AUTHORITY_LOCALIZATION": "ATTRIBUTE",
            "BENEFICIAL_LOCAL_OPTIMIZATION": "ATTRIBUTE",
            "CONTEXTUAL_PREFERENCE": "LOCAL_ONLY",
            "POLICY_EROSION": "DESIGN_PRESSURE_ONLY",
            "INSUFFICIENT_ATTRIBUTION": "HOLD",
        }[triage_receipt.disposition]

        return InstitutionalAdaptationIngressV1.create(
            organization_id=observation.organization_id,
            workspace_id=observation.workspace_id,
            acquisition=acquisition,
            divergence_detection=divergence_detection.pin(),
            adaptation_observation=observation.pin(),
            triage_receipt=triage_receipt.pin(),
            disposition=triage_receipt.disposition,
            route=route,
            recorded_at=recorded_at,
        )

    @staticmethod
    def semantic_cluster(
        *,
        global_baseline: ArtifactPinV1,
        receipts: Iterable[AdaptationTriageReceiptV1],
        canonical_adaptation_key: str,
        normalized_effect_signature: str,
        applicability_signature: str,
        cluster_method_version: str,
    ) -> SemanticAdaptationClusterV1:
        members = tuple(receipts)
        if not members:
            raise ValueError("SEMANTIC_CLUSTER_REQUIRES_RECEIPTS")
        if any(not receipt.learning_eligible for receipt in members):
            raise ValueError("SEMANTIC_CLUSTER_REJECTS_INELIGIBLE_RECEIPT")
        return SemanticAdaptationClusterV1.create(
            global_baseline=global_baseline,
            canonical_adaptation_key=canonical_adaptation_key,
            member_triage_receipt_refs=tuple(receipt.pin() for receipt in members),
            normalized_effect_signature=normalized_effect_signature,
            applicability_signature=applicability_signature,
            cluster_method_version=cluster_method_version,
        )

    @staticmethod
    def design_pressure_opportunity(
        *,
        cluster: SemanticAdaptationClusterV1,
        receipts: Iterable[AdaptationTriageReceiptV1],
        independence_clusters: Iterable[AdaptationIndependenceClusterV1],
        distinct_context_family_refs: tuple[ArtifactPinV1, ...],
        hypothesis: str,
    ) -> DesignPressureOpportunityV1 | None:
        receipt_by_observation = {
            receipt.adaptation_observation.artifact_ref: receipt for receipt in receipts
        }
        erosion_clusters: list[AdaptationIndependenceClusterV1] = []
        protected: dict[tuple[str, str, str], ArtifactPinV1] = {}

        for group in independence_clusters:
            if group.semantic_cluster != cluster.pin() or group.independence_status != "INDEPENDENT":
                continue
            if group.representative_support_ref is None:
                continue
            receipt = receipt_by_observation.get(group.representative_support_ref.artifact_ref)
            if receipt is None or receipt.disposition != "POLICY_EROSION":
                continue
            erosion_clusters.append(group)
            for pin in receipt.protected_invariant_refs:
                protected[(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)] = pin

        # Same trigger geometry as Stage A, but negative evidence remains zero
        # positive promotion support and cannot directly form a ChangeCandidate.
        if len(erosion_clusters) < 3 or len(distinct_context_family_refs) < 2 or not protected:
            return None

        return DesignPressureOpportunityV1.create(
            global_baseline=cluster.global_baseline,
            semantic_cluster=cluster.pin(),
            erosion_independence_cluster_refs=tuple(group.pin() for group in erosion_clusters),
            distinct_context_family_refs=distinct_context_family_refs,
            protected_invariant_refs=tuple(protected.values()),
            hypothesis=hypothesis,
        )

    @staticmethod
    def cross_context_hypothesis(
        *,
        cluster: SemanticAdaptationClusterV1,
        receipts: Iterable[AdaptationTriageReceiptV1],
        independence_clusters: Iterable[AdaptationIndependenceClusterV1],
        distinct_context_family_refs: tuple[ArtifactPinV1, ...],
        counterexample_refs: tuple[ArtifactPinV1, ...],
        outcome_witness_refs: tuple[ArtifactPinV1, ...],
        erosion_screen: ArtifactPinV1,
        erosion_screen_pass: bool,
        common_cause_screen: ArtifactPinV1,
        common_cause_screen_pass: bool,
        protected_invariant_screen: ArtifactPinV1,
        protected_invariant_screen_pass: bool,
        counterexample_review_completed: bool,
        unresolved_authority_contradiction: bool,
        applicability_hypothesis: str,
        target_kind: str,
    ) -> CrossContextAdaptationHypothesisV1 | None:
        receipts = tuple(receipts)
        independence_clusters = tuple(independence_clusters)
        receipt_by_observation = {
            receipt.adaptation_observation.artifact_ref: receipt for receipt in receipts
        }
        distribution = Counter(receipt.disposition for receipt in receipts)

        # Erosion is evidence of design pressure, never direct positive precedent.
        if distribution["POLICY_EROSION"]:
            return None
        if unresolved_authority_contradiction:
            return None

        independent_positive: list[AdaptationIndependenceClusterV1] = []
        unresolved_common_cause = False
        for group in independence_clusters:
            if group.semantic_cluster != cluster.pin():
                raise ValueError("INDEPENDENCE_CLUSTER_SEMANTIC_CLUSTER_MISMATCH")
            if group.independence_status == "UNRESOLVED":
                unresolved_common_cause = True
                continue
            if group.independence_status != "INDEPENDENT" or group.representative_support_ref is None:
                continue
            receipt = receipt_by_observation.get(group.representative_support_ref.artifact_ref)
            if receipt is not None and receipt.disposition in _POSITIVE:
                independent_positive.append(group)

        if unresolved_common_cause:
            return None

        support_count = len(independent_positive)
        context_count = len(
            {(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in distinct_context_family_refs}
        )

        stage_a = (
            support_count >= 3
            and context_count >= 2
            and common_cause_screen_pass
        )
        if not stage_a:
            return None

        stage_b = (
            support_count >= 5
            and context_count >= 3
            and len(outcome_witness_refs) >= 2
            and counterexample_review_completed
            and erosion_screen_pass
            and common_cause_screen_pass
            and protected_invariant_screen_pass
        )
        stage = "B" if stage_b else "A"

        return CrossContextAdaptationHypothesisV1.create(
            global_baseline=cluster.global_baseline,
            semantic_cluster=cluster.pin(),
            supporting_independence_cluster_refs=tuple(
                group.pin() for group in independent_positive
            ),
            distinct_context_family_refs=distinct_context_family_refs,
            counterexample_refs=counterexample_refs,
            triage_distribution=dict(distribution),
            outcome_witness_refs=outcome_witness_refs,
            erosion_screen=erosion_screen,
            common_cause_screen=common_cause_screen,
            protected_invariant_screen=protected_invariant_screen,
            applicability_hypothesis=applicability_hypothesis,
            threshold_stage=stage,
            target_kind=target_kind,
        )


__all__ = ["InstitutionalAdaptationService"]
