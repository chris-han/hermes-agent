"""Typed, append-only learning projection from governed runtime dispositions."""

from __future__ import annotations

from datetime import datetime

from contracts.ontology_admission import (
    FailureAttributionV1,
    HumanDispositionV1,
    LearningEvidenceV1,
)
from eos.artifact_hashing import content_hash


class OntologyLearningService:
    """Create learning artifacts without mutating admitted runtime authority."""

    _FIXTURE_KIND = {
        "CONFIRMED": "positive",
        "FALSE_POSITIVE": "hard_negative",
        "FALSE_NEGATIVE": "hard_negative",
        "DISPUTED": "disputed",
    }

    def materialize(
        self,
        *,
        disposition: HumanDispositionV1,
        evidence_refs: tuple[str, ...],
        owning_layer: str,
        owner_artifact_ref: str,
        created_at: datetime,
    ) -> tuple[LearningEvidenceV1, FailureAttributionV1]:
        if disposition.disposition == "NEEDS_MORE_EVIDENCE":
            raise ValueError("NEEDS_MORE_EVIDENCE is non-learning until a pinned successor exists")
        if not evidence_refs:
            raise ValueError("learning evidence requires exact evidence refs")

        learning_basis = {
            "organization_id": disposition.organization_id,
            "workspace_id": disposition.workspace_id,
            "evaluation_bundle_ref": disposition.evaluation_bundle_ref,
            "evaluation_bundle_hash": disposition.evaluation_bundle_hash,
            "disposition_ref": disposition.disposition_ref,
            "disposition_hash": disposition.disposition_hash,
            "fixture_candidate_kind": self._FIXTURE_KIND[disposition.disposition],
            "evidence_refs": tuple(sorted(set(evidence_refs))),
            "admissible_reuse_scope": ("ontology_build_fixture_candidate",),
            "mutation_forbidden": True,
        }
        learning_digest = content_hash(learning_basis)
        learning = LearningEvidenceV1(
            learning_evidence_ref=f"learning-evidence:{learning_digest}",
            learning_evidence_hash=f"sha256:{learning_digest}",
            created_at=created_at,
            **learning_basis,
        )

        attribution_basis = {
            "organization_id": disposition.organization_id,
            "workspace_id": disposition.workspace_id,
            "learning_evidence_ref": learning.learning_evidence_ref,
            "learning_evidence_hash": learning.learning_evidence_hash,
            "owning_layer": owning_layer,
            "owner_artifact_ref": owner_artifact_ref,
            "proposed_successor_kind": "bounded_repair",
        }
        attribution_digest = content_hash(attribution_basis)
        attribution = FailureAttributionV1(
            attribution_ref=f"failure-attribution:{attribution_digest}",
            attribution_hash=f"sha256:{attribution_digest}",
            created_at=created_at,
            **attribution_basis,
        )
        return learning, attribution


__all__ = ["OntologyLearningService"]
