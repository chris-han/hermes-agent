"""Tender adapter for the generic decision-local projection compiler."""

from __future__ import annotations

import re
from typing import Callable, Mapping

from contracts.decision_local_projection import (
    DecisionContractV1,
    DecisionLocalProjectionV1,
    MaterialDependencyRequirementV1,
    SemanticPredicateSpecV1,
)
from contracts.observed_constraint_hypothesis import (
    EffectiveRestrictionRuntimeContextV1,
    RuntimeSemanticRetrievalSnapshotV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from contracts.semantic_kernel import WorkingContext
from services.decision_local_projection_service import DecisionLocalProjectionService
from services.governed_rule_version_integrity import (
    governed_rule_version_hash,
    verify_governed_rule_version_bytes,
)

_ASCII_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_RUNTIME_AUTHORITY_FIELDS = (
    "organization_id",
    "workspace_id",
    "accepted_graph_release_ref",
    "accepted_graph_release_hash",
    "governed_rule_projection_ref",
    "governed_rule_projection_hash",
    "ordered_governed_rule_version_refs",
    "ordered_governed_rule_version_set_hash",
    "activation_set_snapshot_ref",
    "activation_set_snapshot_hash",
    "activation_resolver_policy_version",
    "resolved_activation_set_hash",
    "effective_context_resolver_hash",
    "effective_at",
)


class TenderDecisionLocalProjectionService:
    """Bind Tender semantics to exact pre-model runtime authority without owning findings."""

    def __init__(
        self,
        *,
        projection_service: DecisionLocalProjectionService,
        governed_rule_resolver: Callable[[str], Mapping | None] | None = None,
    ) -> None:
        if not isinstance(projection_service, DecisionLocalProjectionService):
            raise TypeError("Tender projection adapter requires DecisionLocalProjectionService")
        if governed_rule_resolver is not None and not callable(governed_rule_resolver):
            raise TypeError("governed_rule_resolver must be callable")
        self._projection_service = projection_service
        self._governed_rule_resolver = governed_rule_resolver

    @staticmethod
    def decision_contract(
        *,
        issue_type: str,
        family_id: int,
        deterministic_constraints: tuple[ArtifactPinV1, ...] = (),
        material_requirements: tuple[MaterialDependencyRequirementV1, ...] = (),
    ) -> DecisionContractV1:
        if _ASCII_ID.fullmatch(issue_type) is None:
            raise ValueError("issue_type must be an ASCII-stable identifier")
        if not 1 <= int(family_id) <= 15:
            raise ValueError("family_id must be in the canonical Tender family space 1..15")
        return DecisionContractV1.create(
            purpose="tender_restriction_semantic_predicate",
            semantic_predicates=(
                SemanticPredicateSpecV1(
                    predicate_id="tender.restriction_present",
                    semantic_type_ref=issue_type,
                    family_ref=f"tender.family.{int(family_id)}",
                    material_requirements=tuple(sorted(
                        material_requirements, key=lambda item: item.dependency_id.encode("ascii")
                    )),
                ),
            ),
            deterministic_constraints=deterministic_constraints,
        )

    @staticmethod
    def _validate_snapshot_authority(
        *,
        runtime_context: EffectiveRestrictionRuntimeContextV1,
        retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
    ) -> None:
        if tuple(getattr(runtime_context, field) for field in _RUNTIME_AUTHORITY_FIELDS) != tuple(
            getattr(retrieval_snapshot, field) for field in _RUNTIME_AUTHORITY_FIELDS
        ):
            raise ValueError("TENDER_RUNTIME_AUTHORITY_MISMATCH")

    def _runtime_binding(
        self,
        *,
        runtime_context: EffectiveRestrictionRuntimeContextV1,
        retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
        canonical_pattern_ref: str,
    ) -> tuple[tuple[ArtifactPinV1, ...], str, int]:
        return self.validate_runtime_binding(runtime_context=runtime_context,
            retrieval_snapshot=retrieval_snapshot, canonical_pattern_ref=canonical_pattern_ref,
            governed_rule_resolver=self._governed_rule_resolver)

    @staticmethod
    def validate_runtime_binding(
        *, runtime_context: EffectiveRestrictionRuntimeContextV1,
        retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
        canonical_pattern_ref: str,
        governed_rule_resolver: Callable[[str], Mapping | None] | None,
    ) -> tuple[tuple[ArtifactPinV1, ...], str, int]:
        TenderDecisionLocalProjectionService._validate_snapshot_authority(
            runtime_context=runtime_context,
            retrieval_snapshot=retrieval_snapshot,
        )
        pattern_material = tuple(
            item
            for item in retrieval_snapshot.ordered_material
            if item.kind == "canonical_pattern" and item.ref == canonical_pattern_ref
        )
        if len(pattern_material) != 1:
            raise ValueError("TENDER_CANONICAL_PATTERN_NOT_PINNED")
        pattern = pattern_material[0]

        if governed_rule_resolver is None:
            raise ValueError("TENDER_GOVERNED_RULE_RESOLVER_REQUIRED")
        matching_rules: list[Mapping] = []
        for rule_ref in retrieval_snapshot.ordered_governed_rule_version_refs:
            rule = governed_rule_resolver(rule_ref)
            if rule is None:
                raise ValueError("TENDER_GOVERNED_RULE_UNRESOLVABLE")
            verify_governed_rule_version_bytes(rule)
            if str(rule.get("rule_version_id") or "") != rule_ref:
                raise ValueError("TENDER_GOVERNED_RULE_IDENTITY_MISMATCH")
            if str(rule.get("organization_id") or "") != runtime_context.organization_id:
                raise ValueError("TENDER_RUNTIME_AUTHORITY_MISMATCH")
            scope = rule.get("applicability_scope") or {}
            if (
                scope.get("source_graph_pattern_id"),
                scope.get("source_graph_pattern_hash"),
            ) == (pattern.ref, pattern.hash):
                matching_rules.append(rule)
        if len(matching_rules) != 1:
            raise ValueError("TENDER_CANONICAL_PATTERN_RULE_BINDING_AMBIGUOUS")
        rule = matching_rules[0]
        if (
            str(rule.get("source_graph_version") or "")
            != runtime_context.accepted_graph_release_ref
            or str(rule.get("source_graph_release_hash") or "")
            != runtime_context.accepted_graph_release_hash
        ):
            raise ValueError("TENDER_RUNTIME_AUTHORITY_MISMATCH")
        scope = rule.get("applicability_scope") or {}
        canonical = scope.get("canonical_restriction_pattern")
        if not isinstance(canonical, Mapping) or (
            str(canonical.get("pattern_id") or ""),
            str(canonical.get("pattern_hash") or ""),
        ) != (pattern.ref, pattern.hash):
            raise ValueError("TENDER_DECISION_READINESS_PATTERN_BINDING_MISSING")
        semantic_dimensions = canonical.get("semantic_dimensions")
        pattern_scope = (
            semantic_dimensions.get("scope")
            if isinstance(semantic_dimensions, Mapping)
            else None
        )
        family_ref = (
            pattern_scope.get("tender_family_ref")
            if isinstance(pattern_scope, Mapping)
            else None
        )
        family_match = (
            re.fullmatch(r"tender\.family\.([1-9]|1[0-5])", family_ref)
            if isinstance(family_ref, str)
            else None
        )
        family_id = int(family_match.group(1)) if family_match is not None else None
        semantic_type_ref = (
            semantic_dimensions.get("constraint_dimension")
            if isinstance(semantic_dimensions, Mapping)
            else None
        )
        if (
            not isinstance(family_id, int)
            or isinstance(family_id, bool)
            or not 1 <= family_id <= 15
            or not isinstance(semantic_type_ref, str)
            or _ASCII_ID.fullmatch(semantic_type_ref) is None
        ):
            raise ValueError("TENDER_DECISION_READINESS_SEMANTICS_MISSING")

        constraints = (
            ArtifactPinV1(
                artifact_kind="accepted_graph_release",
                artifact_ref=runtime_context.accepted_graph_release_ref,
                artifact_hash=runtime_context.accepted_graph_release_hash,
            ),
            ArtifactPinV1(
                artifact_kind="governed_rule_projection",
                artifact_ref=runtime_context.governed_rule_projection_ref,
                artifact_hash=runtime_context.governed_rule_projection_hash,
            ),
            ArtifactPinV1(
                artifact_kind="canonical_pattern",
                artifact_ref=pattern.ref,
                artifact_hash=pattern.hash,
            ),
            ArtifactPinV1(
                artifact_kind="governed_rule_version",
                artifact_ref=str(rule["rule_version_id"]),
                artifact_hash=governed_rule_version_hash(rule),
            ),
            ArtifactPinV1(
                artifact_kind="activation_set_snapshot",
                artifact_ref=runtime_context.activation_set_snapshot_ref,
                artifact_hash=runtime_context.activation_set_snapshot_hash,
            ),
            ArtifactPinV1(
                artifact_kind="resolved_activation_set",
                artifact_ref=(
                    "resolved-activation-set:"
                    + runtime_context.resolved_activation_set_hash[7:]
                ),
                artifact_hash=runtime_context.resolved_activation_set_hash,
            ),
            ArtifactPinV1(
                artifact_kind="effective_restriction_runtime_context",
                artifact_ref=(
                    "effective-restriction-runtime-context:"
                    + runtime_context.effective_context_resolver_hash[7:]
                ),
                artifact_hash=runtime_context.effective_context_resolver_hash,
            ),
        )
        return constraints, semantic_type_ref, family_id

    def compile(
        self,
        *,
        working_context: WorkingContext,
        decision_subject_ref: str,
        issue_type: str,
        family_id: int,
    ) -> DecisionLocalProjectionV1:
        return self._projection_service.compile(
            working_context=working_context,
            decision_subject_ref=decision_subject_ref,
            decision_contract=self.decision_contract(issue_type=issue_type, family_id=family_id),
        )

    def compile_runtime_bound(
        self,
        *,
        working_context: WorkingContext,
        decision_subject_ref: str,
        runtime_context: EffectiveRestrictionRuntimeContextV1,
        retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
        canonical_pattern_ref: str,
        source_region: RuntimeSemanticSourceRegionV1 | None = None,
    ) -> DecisionLocalProjectionV1:
        if (
            working_context.organization_id,
            working_context.workspace_id,
        ) != (
            runtime_context.organization_id,
            runtime_context.workspace_id,
        ):
            raise ValueError("TENDER_RUNTIME_AUTHORITY_MISMATCH")
        constraints, semantic_type_ref, family_id = self._runtime_binding(
            runtime_context=runtime_context,
            retrieval_snapshot=retrieval_snapshot,
            canonical_pattern_ref=canonical_pattern_ref,
        )
        source_dependencies = () if source_region is None else source_region.material_dependencies
        material_requirements = tuple(
            MaterialDependencyRequirementV1(
                dependency_id=item.dependency_id,
                required_evidence_role=item.required_evidence_role,
                min_count=item.min_count,
            )
            for item in source_dependencies
        )
        if source_dependencies:
            if (source_region.evidence[0].evidence_ref.organization_id,
                source_region.evidence[0].evidence_ref.workspace_id) != (
                working_context.organization_id, working_context.workspace_id
            ):
                raise ValueError("TENDER_MATERIAL_DEPENDENCY_SCOPE_MISMATCH")
            closure = self._projection_service.evidence_closure(working_context)
            source_basis = {
                (anchor.evidence_ref.source_identity_ref,
                 anchor.evidence_ref.selector_hash,
                 anchor.evidence_ref.evidence_role)
                for anchor in source_region.evidence
            }
            context_basis = {
                (evidence.source_ref, evidence.selector_hash, evidence.evidence_role)
                for evidence in closure
            }
            if source_basis != context_basis:
                raise ValueError("TENDER_MATERIAL_DEPENDENCY_SOURCE_CLOSURE_MISMATCH")
            region_evidence_refs = {
                anchor.evidence_ref.evidence_ref for anchor in source_region.evidence
            }
            if any(not set(item.basis_evidence_refs) <= region_evidence_refs
                   for item in source_dependencies):
                raise ValueError("TENDER_MATERIAL_DEPENDENCY_BASIS_UNGROUNDED")
        release_pin = next(
            pin for pin in constraints if pin.artifact_kind == "accepted_graph_release"
        )
        return self._projection_service.compile(
            working_context=working_context,
            decision_subject_ref=decision_subject_ref,
            decision_contract=self.decision_contract(
                issue_type=semantic_type_ref,
                family_id=family_id,
                deterministic_constraints=constraints,
                material_requirements=material_requirements,
            ),
            required_source_graph=release_pin,
        )


__all__ = ["TenderDecisionLocalProjectionService"]
