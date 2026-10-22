"""W4 frozen comparison basis and intervention/comparability gate."""

from __future__ import annotations

from contracts.artifact_pin import ArtifactPinV1
from contracts.intervention_comparison import (
    ComparisonCompatibilityV1,
    ComparisonManifestV1,
)
from semantier.cognitive_context.store import CognitiveContextStore


class InterventionComparabilityService:
    """Freeze exact comparison basis; does not perform semantic evaluation or admission."""

    _DIMENSIONS = (
        "owning_stage",
        "evaluator",
        "obligation_profile",
        "comparison_policy",
        "case_panel",
        "protected_holdout",
        "dgp",
        "representation_contract",
        "serializer_policy",
        "context_envelope",
        "runtime_pins",
        "provider_pins",
        "config_pins",
    )

    def __init__(self, *, store: CognitiveContextStore) -> None:
        if not isinstance(store, CognitiveContextStore):
            raise TypeError("comparability requires CognitiveContextStore")
        self._store = store

    def freeze_manifest(self, **kwargs) -> ComparisonManifestV1:
        manifest = ComparisonManifestV1.create(**kwargs)
        return self._store.save_or_get_derived_projection(manifest)

    @staticmethod
    def _different_dimensions(
        left: ComparisonManifestV1,
        right: ComparisonManifestV1,
    ) -> tuple[str, ...]:
        return tuple(
            dimension
            for dimension in InterventionComparabilityService._DIMENSIONS
            if getattr(left, dimension) != getattr(right, dimension)
        )

    def compare_basis(
        self,
        *,
        left: ComparisonManifestV1,
        right: ComparisonManifestV1,
        declared_intervention_dimension: str | None = None,
    ) -> ComparisonCompatibilityV1:
        if (
            left.organization_id,
            left.workspace_id,
        ) != (
            right.organization_id,
            right.workspace_id,
        ):
            raise ValueError("COMPARISON_SCOPE_MISMATCH")
        if left.candidate != right.candidate:
            raise ValueError("COMPARISON_CANDIDATE_MISMATCH")

        differences = self._different_dimensions(left, right)

        if declared_intervention_dimension is not None:
            if declared_intervention_dimension not in self._DIMENSIONS:
                raise ValueError("UNKNOWN_INTERVENTION_DIMENSION")
            if (
                left.mode != "INTERVENTION"
                or right.mode != "INTERVENTION"
                or left.declared_intervention is None
                or right.declared_intervention is None
                or left.declared_intervention != right.declared_intervention
            ):
                raise ValueError("INTERVENTION_BASIS_NOT_EXPLICIT")
            comparable = differences == (declared_intervention_dimension,)
            causal = comparable
        else:
            comparable = not differences
            causal = False

        receipt = ComparisonCompatibilityV1.create(
            organization_id=left.organization_id,
            workspace_id=left.workspace_id,
            left_manifest=left.pin(),
            right_manifest=right.pin(),
            status="COMPARABLE" if comparable else "INCOMPARABLE",
            causal_claim_permitted=causal,
            differing_dimensions=differences,
            declared_intervention_dimension=(
                declared_intervention_dimension if comparable else None
            ),
        )
        return self._store.save_or_get_derived_projection(receipt)

    @staticmethod
    def assert_protected_holdout_not_in_generation_inputs(
        manifest: ComparisonManifestV1,
        *,
        generation_inputs: tuple[ArtifactPinV1, ...],
    ) -> None:
        protected = set(manifest.protected_holdout)
        leaked = tuple(pin for pin in generation_inputs if pin in protected)
        if leaked:
            raise ValueError("PROTECTED_HOLDOUT_LEAK")

    @staticmethod
    def assert_causal_claim_allowed(receipt: ComparisonCompatibilityV1) -> None:
        if not receipt.causal_claim_permitted:
            raise ValueError("CAUSAL_CLAIM_NOT_AUTHORIZED")
