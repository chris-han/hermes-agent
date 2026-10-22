"""Tender-scoped deterministic comparison using governed stored authority."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from contracts.frame_match_result_v2 import (
    FrameMatchResultDTOV2,
    create_frame_match_result_v2,
)
from contracts.observed_constraint_hypothesis import (
    ObservedConstraintHypothesisV1,
    ObservedConstraintHypothesisV2,
)
from contracts.observed_constraint_hypothesis import RuntimeSemanticRetrievalSnapshotV1
from contracts.semantic_constraint_frame import (
    ConstraintFrameV01,
    ComposedFramePatternV01,
    FramePatternV01,
    FrameValueConstraintV01,
    NormalizedQuantityV01,
)
from eos.artifact_hashing import canonical_json, content_hash
from plugins.tender_sensitive_term_detection_modeling.frame_semantics import (
    accepted_composed_frame_patterns_from_rules,
    match_constraint_frame,
    match_composed_frames,
)
from contracts.accepted_graph_release_v2 import parse_accepted_graph_release_unverified
from services.accepted_graph_rule_projection_service import AcceptedGraphRuleProjectionService
from services.effective_restriction_runtime_context_service import EffectiveRestrictionRuntimeContextService
from services.runtime_semantic_hypothesis_generation_service import _validate_snapshot_authority
from services.runtime_observation_correction_service import RuntimeObservationCorrectionService
from storage.governed_rule_store import GovernedRuleStore
from storage.human_grounding_store import HumanGroundingStore
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
    _FRAME_MATCH_WRITE_CAPABILITY,
)


def _quantity(value: Mapping[str, Any]) -> NormalizedQuantityV01:
    return NormalizedQuantityV01(**dict(value))


def _value_constraint(value: Mapping[str, Any]) -> FrameValueConstraintV01:
    payload = dict(value)
    if isinstance(payload.get("normalized_quantity"), Mapping):
        payload["normalized_quantity"] = _quantity(payload["normalized_quantity"])
    return FrameValueConstraintV01(**payload)


def _frame(value: Mapping[str, Any]) -> ConstraintFrameV01:
    payload = dict(value)
    payload.pop("frame_hash", None)
    payload["value_constraint"] = _value_constraint(payload["value_constraint"])
    if isinstance(payload.get("slot_evidence_refs"), Mapping):
        payload["slot_evidence_refs"] = {
            str(key): tuple(items) for key, items in payload["slot_evidence_refs"].items()
        }
    if "inferred_slots" in payload:
        payload["inferred_slots"] = tuple(payload["inferred_slots"])
    return ConstraintFrameV01(**payload)


def _slot_diagnostics(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "slot_name": item.slot_name,
            "state": "matched" if item.matched else "not_matched",
            "observed_value": asdict(item.observed_value)
            if hasattr(item.observed_value, "__dataclass_fields__")
            else item.observed_value,
            "canonical_value": asdict(item.canonical_value)
            if hasattr(item.canonical_value, "__dataclass_fields__")
            else item.canonical_value,
            "rule_refs": list(item.rule_refs),
        }
        for item in result.slot_diagnostics
    ]


def _match_composed_observation(
    observed_frames: tuple[ConstraintFrameV01, ...],
    pattern: ComposedFramePatternV01,
):
    ordered = tuple(
        sorted(observed_frames, key=lambda frame: canonical_json(asdict(frame)))
    )
    return match_composed_frames(ordered, pattern)


def _rule_hash(rule: Mapping[str, Any]) -> str:
    value = str(rule.get("content_hash") or "")
    return value if value.startswith("sha256:") else "sha256:" + value


def _verify_governed_rule_bytes(rule: Mapping[str, Any]) -> None:
    fields = (
        "rule_version_id", "rule_candidate_id", "organization_id",
        "authority_domain", "semantic_tier", "jurisdiction", "rule_family",
        "rule_state", "source_graph_version", "source_graph_release_hash",
        "source_graph_rule_id", "source_anchor_refs", "applicability_scope",
        "governed_rule_text", "extracted_rationale", "severity", "confidence",
        "uncertainty_notes", "reviewer_edits", "approval_evidence",
        "test_evidence", "activation_refs", "supersedes_rule_version_id",
    )
    payload = {field: rule.get(field) for field in fields}
    if str(rule.get("content_hash") or "") != content_hash(canonical_json(payload)):
        raise ValueError("governed rule version content hash mismatch")


def _evaluate_resolved_frame_match_v2(
    *,
    hypothesis: ObservedConstraintHypothesisV1,
    retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
    pattern: FramePatternV01 | ComposedFramePatternV01,
    governed_rule_version: Mapping[str, Any],
) -> FrameMatchResultDTOV2:
    """Evaluate one exact hypothesis/pattern authority binding without retrieval semantics."""

    if (
        retrieval_snapshot.snapshot_id,
        retrieval_snapshot.snapshot_hash,
        retrieval_snapshot.run_id,
    ) != (
        hypothesis.retrieval_snapshot_ref,
        hypothesis.retrieval_snapshot_hash,
        hypothesis.run_id,
    ):
        raise ValueError("retrieval snapshot identity/run differs from hypothesis pins")
    authority_fields = (
        "organization_id",
        "workspace_id",
        "accepted_graph_release_ref",
        "accepted_graph_release_hash",
        "governed_rule_projection_ref",
        "governed_rule_projection_hash",
        "activation_set_snapshot_ref",
        "activation_set_snapshot_hash",
        "effective_context_resolver_hash",
    )
    if tuple(getattr(retrieval_snapshot, field) for field in authority_fields) != tuple(
        getattr(hypothesis, field) for field in authority_fields
    ):
        raise ValueError("retrieval snapshot authority differs from hypothesis pins")
    material_pairs = {
        (item.ref, item.hash)
        for item in retrieval_snapshot.ordered_material
        if item.kind == "canonical_pattern"
    }
    if (pattern.pattern_id, pattern.pattern_hash) not in material_pairs or (
        pattern.pattern_id,
        pattern.pattern_hash,
    ) not in set(
        zip(hypothesis.retrieved_pattern_refs, hypothesis.retrieved_pattern_hashes)
    ):
        raise ValueError("canonical pattern is outside the exact retrieval pins")
    if pattern.lifecycle_state != "accepted":
        raise ValueError("runtime matching requires an accepted canonical pattern")
    if pattern.graph_release_ref != hypothesis.accepted_graph_release_ref:
        raise ValueError("canonical pattern release ref differs from hypothesis authority")
    scope = governed_rule_version.get("applicability_scope") or {}
    _verify_governed_rule_bytes(governed_rule_version)
    rule_ref = str(governed_rule_version.get("rule_version_id") or "")
    if rule_ref not in retrieval_snapshot.ordered_governed_rule_version_refs:
        raise ValueError("governed rule version is outside the retrieval authority")
    if (
        scope.get("source_graph_pattern_id"),
        scope.get("source_graph_pattern_hash"),
    ) != (pattern.pattern_id, pattern.pattern_hash):
        raise ValueError("governed rule version does not bind the canonical pattern")

    unresolved = hypothesis.uncertainty.get("unresolved_slots", ())
    if not isinstance(unresolved, (list, tuple)):
        raise ValueError("hypothesis unresolved_slots must be an array")
    uncertainty_keys = set(hypothesis.uncertainty)
    has_material_uncertainty = (
        hypothesis.uncertainty.get("state") != "bounded"
        or bool(unresolved)
        or bool(uncertainty_keys - {"state", "unresolved_slots"})
    )
    if has_material_uncertainty:
        relation = "UNRESOLVED"
        diagnostics: list[dict[str, Any]] = []
    else:
        if isinstance(pattern, ComposedFramePatternV01):
            if not isinstance(hypothesis, ObservedConstraintHypothesisV2) or (
                hypothesis.observation_shape != "composed_frame_set"
            ):
                raise ValueError(
                    "composed canonical pattern requires explicit composed observation shape"
                )
            try:
                composed_match = _match_composed_observation(
                    tuple(_frame(candidate) for candidate in hypothesis.candidate_frames),
                    pattern,
                )
            except (KeyError, TypeError, ValueError):
                relation, diagnostics = "UNRESOLVED", []
            else:
                relation, diagnostics = composed_match.semantic_relation, []
        else:
            try:
                matches = [
                    match_constraint_frame(_frame(candidate), pattern)
                    for candidate in hypothesis.candidate_frames
                ]
            except (KeyError, TypeError, ValueError):
                matches = []
            outcomes = {
                (item.semantic_relation, canonical_json(_slot_diagnostics(item)))
                for item in matches
            }
            if len(outcomes) != 1:
                relation, diagnostics = "UNRESOLVED", []
            else:
                selected = matches[0]
                relation = selected.semantic_relation
                diagnostics = _slot_diagnostics(selected)

    state = {
        "FRAME_EQUIVALENT": "MATCH",
        "FRAME_ENTAILS": "MATCH",
        "NO_MATCH": "NO_MATCH",
        "UNRESOLVED": "UNKNOWN",
    }[relation]
    hard_negative_material = [
        item for item in retrieval_snapshot.ordered_material if item.kind == "hard_negative"
    ]
    negative_diagnostics = [
        {
            "material_ref": str(item.ref),
            "material_hash": str(item.hash),
            "state": "retrieval_evidence_only",
        }
        for item in sorted(
            hard_negative_material,
            key=lambda item: (str(item.ref).encode("utf-8"), str(item.hash)),
        )
    ]
    return create_frame_match_result_v2(
        organization_id=hypothesis.organization_id,
        workspace_id=hypothesis.workspace_id,
        semantic_relation=relation,
        frame_match_state=state,
        observed_hypothesis_ref=hypothesis.hypothesis_id,
        observed_hypothesis_hash=hypothesis.hypothesis_hash,
        source_identity_ref=hypothesis.source_binding.source_identity_ref,
        source_hash=hypothesis.source_binding.source_hash,
        evidence_refs=hypothesis.evidence_refs,
        source_anchor_refs=hypothesis.source_anchor_refs,
        canonical_pattern_ref=pattern.pattern_id,
        canonical_pattern_hash=pattern.pattern_hash,
        accepted_graph_release_ref=hypothesis.accepted_graph_release_ref,
        accepted_graph_release_hash=hypothesis.accepted_graph_release_hash,
        governed_rule_version_ref=rule_ref,
        governed_rule_version_hash=_rule_hash(governed_rule_version),
        activation_set_snapshot_ref=hypothesis.activation_set_snapshot_ref,
        activation_set_snapshot_hash=hypothesis.activation_set_snapshot_hash,
        activation_resolver_policy_version=retrieval_snapshot.activation_resolver_policy_version,
        effective_context_resolver_hash=hypothesis.effective_context_resolver_hash,
        resolved_activation_set_hash=retrieval_snapshot.resolved_activation_set_hash,
        slot_diagnostics=diagnostics,
        hard_negative_diagnostics=negative_diagnostics,
        exception_diagnostics=[],
    )


class TenderRuntimeFrameMatchService:
    """Resolve exact accepted Tender authority before invoking the frozen kernel."""

    def __init__(self, *, runtime_root: str | Path, db_path: str | Path) -> None:
        self._runtime_root = Path(runtime_root)
        self._db_path = Path(db_path)

    def match(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        hypothesis_ref: str,
        hypothesis_hash: str,
        canonical_pattern_ref: str,
    ) -> FrameMatchResultDTOV2:
        artifact_store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        hypothesis = artifact_store.get_hypothesis(
            hypothesis_ref=hypothesis_ref, hypothesis_hash=hypothesis_hash
        )
        if hypothesis.predecessor_hypothesis_ref is not None:
            hypothesis = RuntimeObservationCorrectionService(
                runtime_root=self._runtime_root,
                grounding_store=HumanGroundingStore(self._db_path, graph_backend=None),
            ).get_verified_successor(
                organization_id=organization_id,
                workspace_id=workspace_id,
                successor_hypothesis_ref=hypothesis_ref,
                successor_hypothesis_hash=hypothesis_hash,
            )
        snapshot = artifact_store.get_retrieval_snapshot(
            snapshot_ref=hypothesis.retrieval_snapshot_ref,
            snapshot_hash=hypothesis.retrieval_snapshot_hash,
        )
        current = EffectiveRestrictionRuntimeContextService(self._db_path).resolve_current(
            organization_id=organization_id, workspace_id=workspace_id
        )
        _validate_snapshot_authority(snapshot, current)
        grounding = HumanGroundingStore(self._db_path, graph_backend=None)
        release = parse_accepted_graph_release_unverified(
            grounding.get_graph_release(
                snapshot.accepted_graph_release_ref,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
        )
        patterns = AcceptedGraphRuleProjectionService._load_v2_patterns(
            grounding,
            release,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        pattern_by_ref = {item.pattern_id: item for item, _binding in patterns}
        canonical = pattern_by_ref.get(canonical_pattern_ref)
        if canonical is None:
            raise ValueError("Tender runtime requires an authoritative canonical pattern")
        rule_store = GovernedRuleStore(self._db_path)
        rules = []
        for rule_ref in snapshot.ordered_governed_rule_version_refs:
            rule = rule_store.get_rule_version(rule_ref)
            if rule is None:
                raise ValueError("effective governed rule version is missing")
            scope = rule.get("applicability_scope") or {}
            if scope.get("source_graph_pattern_id") == canonical_pattern_ref:
                rules.append(rule)
        if len(rules) != 1:
            raise ValueError("canonical pattern must resolve to one active governed rule")
        if canonical.pattern_kind == "atomic" and len(canonical.frames) == 1:
            projection = AcceptedGraphRuleProjectionService._frame_pattern_projection(
                canonical, release.graph_version
            )
            projection["frame"] = _frame(projection["frame"])
            pattern: FramePatternV01 | ComposedFramePatternV01 = FramePatternV01(
                **projection
            )
        elif canonical.pattern_kind == "composed":
            composed = accepted_composed_frame_patterns_from_rules(rules)
            if len(composed) != 1 or (
                composed[0].pattern_id,
                composed[0].pattern_hash,
            ) != (canonical.pattern_id, canonical.pattern_hash):
                raise ValueError(
                    "active governed rule does not preserve the composed canonical pattern"
                )
            pattern = composed[0]
        else:
            raise ValueError("Tender runtime canonical pattern shape is unsupported")
        return artifact_store._put_frame_match_result(
            _evaluate_resolved_frame_match_v2(
                hypothesis=hypothesis,
                retrieval_snapshot=snapshot,
                pattern=pattern,
                governed_rule_version=rules[0],
            ),
            capability=_FRAME_MATCH_WRITE_CAPABILITY,
        )


__all__ = ["TenderRuntimeFrameMatchService"]
