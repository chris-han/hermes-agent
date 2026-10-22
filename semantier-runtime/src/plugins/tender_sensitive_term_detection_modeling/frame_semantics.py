"""Constraint-frame compiler, extractor, and deterministic matcher v0.1."""

from __future__ import annotations

import json
import pkgutil
import re
from dataclasses import asdict, replace
from decimal import Decimal
from typing import Any

from contracts.tender_sensitive_term_detection_modeling_core import (
    ComposedFramePatternV01,
    ConstraintFrameV01,
    FrameMatchResultDTO,
    FramePatternV01,
    FrameSlotDiagnosticV01,
    FrameValueConstraintV01,
    NormalizedQuantityV01,
)
from eos.artifact_hashing import content_hash


SEMANTIC_PROFILE_ID = "tender_semantics.v0_1"
PACKAGE_NAME = "plugins.tender_sensitive_term_detection_modeling"


def _load_package_json(relative_path: str) -> dict[str, Any]:
    raw = pkgutil.get_data(PACKAGE_NAME, relative_path)
    if raw is None:
        raise ValueError(f"missing packaged semantic asset: {relative_path}")
    return json.loads(raw.decode("utf-8"))


def load_operator_semantics() -> dict[str, Any]:
    payload = _load_package_json("assets/tender_operator_semantics.v0_1.json")
    if payload.get("profile_id") != "tender_operator_semantics.v0_1":
        raise ValueError("unsupported tender operator semantics profile")
    return payload


def load_semantic_profile() -> dict[str, Any]:
    payload = _load_package_json("assets/tender_semantics.v0_1.json")
    if payload.get("semantic_profile_id") != SEMANTIC_PROFILE_ID:
        raise ValueError("unsupported tender semantic profile")
    required = {
        "frame_schema",
        "graph_release_binding",
        "operator_rules",
        "matcher",
        "composition_rules",
    }
    if not required.issubset(payload):
        raise ValueError("tender semantic profile is incomplete")
    return payload


def normalize_cny_quantity(
    text: str, *, semantics: dict[str, Any] | None = None
) -> NormalizedQuantityV01 | None:
    semantics = semantics or load_operator_semantics()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(亿元|万元|元)", text)
    if match is None:
        return None
    amount, unit = match.groups()
    unit_rule = semantics["units"][unit]
    normalized = Decimal(amount) * Decimal(str(unit_rule["multiplier"]))
    return NormalizedQuantityV01(
        amount=amount,
        unit_ref=str(unit_rule["unit_ref"]),
        normalized_amount=format(normalized.normalize(), "f"),
        normalized_unit_ref="CNY_YUAN",
        currency_ref="CNY",
    )


def _first_ref(text: str, definitions: list[dict[str, Any]], key: str) -> str | None:
    for definition in definitions:
        if any(marker in text for marker in definition.get("markers") or []):
            return str(definition[key])
    return None


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def _concept(text: str, semantics: dict[str, Any]) -> str | None:
    for concept_ref, aliases in semantics["concept_aliases"].items():
        if _contains_any(text, aliases):
            return str(concept_ref)
    return None


def extract_constraint_frame(
    text: str,
    *,
    evidence_ref: str,
    semantics: dict[str, Any] | None = None,
) -> ConstraintFrameV01 | None:
    """Extract only the bounded proving-set semantics from one clause."""

    semantics = semantics or load_operator_semantics()
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return None
    evidence = (evidence_ref,)
    scope_guard = (
        normalized.startswith("不得以")
        or "为由拒绝" in normalized
        or ("不得" in normalized and "限制" in normalized)
    )

    attribute_ref = _first_ref(normalized, semantics["attributes"], "attribute_ref")
    if attribute_ref is not None:
        operator_ref = _first_ref(normalized, semantics["operators"], "operator_ref")
        quantity = normalize_cny_quantity(normalized, semantics=semantics)
        if operator_ref is None:
            operator_ref = "GE" if "XX" in normalized and "以上" in normalized else "ASSERTS"
        value = (
            FrameValueConstraintV01(
                value_kind="normalized_quantity", normalized_quantity=quantity
            )
            if quantity is not None
            else FrameValueConstraintV01(
                value_kind="context_ref", context_ref="source_threshold"
            )
        )
        mandatory = _contains_any(normalized, semantics["modality"]["mandatory"])
        descriptive = _contains_any(normalized, semantics["modality"]["description"])
        modality_ref = "description" if descriptive and not mandatory else "mandatory"
        consequence_ref = (
            "invalid_bid"
            if _contains_any(normalized, semantics["consequence_markers"]["invalid_bid"])
            else "eligibility_restriction"
        )
        return ConstraintFrameV01(
            target_type_ref="supplier",
            attribute_ref=attribute_ref,
            operator_ref=operator_ref,
            value_constraint=value,
            modality_ref=modality_ref,
            stage_ref="eligibility",
            consequence_ref=consequence_ref,
            slot_evidence_refs={name: evidence for name in (
                "attribute_ref", "operator_ref", "value_constraint", "modality_ref"
            )},
            inferred_slots=("target_type_ref", "stage_ref", "consequence_ref"),
        )

    concept_ref = _concept(normalized, semantics)
    if concept_ref == "state_owned_enterprise":
        return ConstraintFrameV01(
            target_type_ref="supplier",
            attribute_ref="ownership_type",
            operator_ref="ALLOW" if scope_guard else "RESTRICT_TO",
            value_constraint=FrameValueConstraintV01(
                value_kind="concept_ref", concept_ref=concept_ref
            ),
            modality_ref="prohibition_of_restriction" if scope_guard else "mandatory",
            stage_ref="eligibility",
            consequence_ref="participation_allowed" if scope_guard else "ineligible",
            slot_evidence_refs={name: evidence for name in (
                "attribute_ref", "operator_ref", "value_constraint", "modality_ref"
            )},
            inferred_slots=("target_type_ref", "stage_ref", "consequence_ref"),
        )
    if concept_ref == "project_local_supplier" or (
        "注册地" in normalized and "项目所在地" in normalized
    ):
        return ConstraintFrameV01(
            target_type_ref="supplier",
            attribute_ref="supplier_location_relation",
            operator_ref="EQ",
            value_constraint=FrameValueConstraintV01(
                value_kind="concept_ref", concept_ref="project_local_supplier"
            ),
            modality_ref="scoring",
            stage_ref="evaluation",
            consequence_ref=(
                "score_bonus"
                if _contains_any(normalized, semantics["consequence_markers"]["score_bonus"])
                else "description"
            ),
            slot_evidence_refs={name: evidence for name in (
                "attribute_ref", "value_constraint", "consequence_ref"
            )},
            inferred_slots=("target_type_ref", "operator_ref", "modality_ref", "stage_ref"),
        )
    if concept_ref == "consortium":
        return ConstraintFrameV01(
            target_type_ref="supplier",
            attribute_ref="participation_form",
            operator_ref="ALLOW" if scope_guard else "PROHIBITS",
            value_constraint=FrameValueConstraintV01(
                value_kind="concept_ref", concept_ref="consortium"
            ),
            modality_ref="prohibition_of_restriction" if scope_guard else "mandatory",
            stage_ref="participation",
            consequence_ref="participation_allowed" if scope_guard else "submission_rejected",
            slot_evidence_refs={name: evidence for name in (
                "attribute_ref", "operator_ref", "value_constraint", "modality_ref"
            )},
            inferred_slots=("target_type_ref", "stage_ref", "consequence_ref"),
        )
    if concept_ref == "cmmi_level_5":
        return ConstraintFrameV01(
            target_type_ref="supplier",
            attribute_ref="certification",
            operator_ref="REQUIRES",
            value_constraint=FrameValueConstraintV01(
                value_kind="concept_ref", concept_ref="cmmi_level_5"
            ),
            modality_ref="mandatory",
            stage_ref="eligibility",
            consequence_ref=(
                "invalid_bid"
                if _contains_any(normalized, semantics["consequence_markers"]["invalid_bid"])
                else "ineligible"
            ),
            slot_evidence_refs={name: evidence for name in (
                "attribute_ref", "operator_ref", "value_constraint", "modality_ref"
            )},
            inferred_slots=("target_type_ref", "stage_ref", "consequence_ref"),
        )
    return None


def compile_frame_pattern_candidate(
    *,
    source_term: str,
    source_example: str | None,
    source_evidence_refs: tuple[str, ...],
    accepted_schema_projection: dict[str, Any],
    graph_release_ref: str,
    validated_relation_evidence: tuple[dict[str, Any], ...] = (),
) -> FramePatternV01:
    """Compile a candidate only through an accepted induced-schema projection."""

    if accepted_schema_projection.get("review_state") not in {"accepted", "edited"}:
        raise ValueError("frame compilation requires an accepted corpus schema")
    if accepted_schema_projection.get("representation_type") != "ConstraintFrameV01":
        raise ValueError("accepted corpus schema does not select ConstraintFrameV01")
    frame = extract_constraint_frame(source_term, evidence_ref=source_evidence_refs[0])
    if frame is None and source_example:
        frame = extract_constraint_frame(
            source_example,
            evidence_ref=source_evidence_refs[0],
        )
    if frame is None:
        raise ValueError("source term/example did not produce a constraint frame")

    relation_slot_map = {
        "modifier_of": ("modality_ref", "operator_ref"),
        "has_threshold": ("operator_ref", "value_constraint"),
        "requires": ("operator_ref", "consequence_ref"),
        "prohibits": ("operator_ref", "consequence_ref"),
        "applies_to": ("target_type_ref", "stage_ref"),
    }
    slot_evidence = dict(frame.slot_evidence_refs)
    for relation in validated_relation_evidence:
        if relation.get("review_state") not in {"accepted", "edited"}:
            continue
        relation_type = str(relation.get("relation_type") or "")
        slots = relation_slot_map.get(relation_type, ())
        refs = tuple(str(ref) for ref in relation.get("evidence_refs") or () if ref)
        for slot in slots:
            slot_evidence[slot] = tuple(dict.fromkeys((*slot_evidence.get(slot, ()), *refs)))
    if slot_evidence != frame.slot_evidence_refs:
        frame = replace(frame, slot_evidence_refs=slot_evidence)
    payload = {
        "frame": asdict(frame),
        "graph_release_ref": graph_release_ref,
        "source_evidence_refs": source_evidence_refs,
        "corpus_schema_candidate_ref": accepted_schema_projection["schema_candidate_id"],
    }
    digest = content_hash(payload)
    return FramePatternV01(
        pattern_id=f"frame-pattern.{digest[:24]}",
        frame=frame,
        graph_release_ref=graph_release_ref,
        source_evidence_refs=source_evidence_refs,
        lifecycle_state="candidate",
        pattern_hash=f"sha256:{digest}",
        corpus_schema_candidate_ref=str(
            accepted_schema_projection["schema_candidate_id"]
        ),
    )


def _quantity_from_payload(payload: dict[str, Any]) -> NormalizedQuantityV01:
    return NormalizedQuantityV01(
        amount=str(payload["amount"]),
        unit_ref=str(payload["unit_ref"]),
        normalized_amount=str(payload["normalized_amount"]),
        normalized_unit_ref=str(payload["normalized_unit_ref"]),
        currency_ref=payload.get("currency_ref"),
    )


def _value_from_payload(payload: dict[str, Any]) -> FrameValueConstraintV01:
    quantity = payload.get("normalized_quantity")
    return FrameValueConstraintV01(
        value_kind=str(payload["value_kind"]),
        concept_ref=payload.get("concept_ref"),
        context_ref=payload.get("context_ref"),
        string_value=payload.get("string_value"),
        normalized_quantity=(
            _quantity_from_payload(quantity) if isinstance(quantity, dict) else None
        ),
    )


def _frame_from_payload(payload: dict[str, Any]) -> ConstraintFrameV01:
    return ConstraintFrameV01(
        target_type_ref=str(payload["target_type_ref"]),
        attribute_ref=str(payload["attribute_ref"]),
        operator_ref=str(payload["operator_ref"]),
        value_constraint=_value_from_payload(dict(payload["value_constraint"])),
        modality_ref=str(payload["modality_ref"]),
        stage_ref=str(payload["stage_ref"]),
        consequence_ref=str(payload["consequence_ref"]),
        slot_evidence_refs={
            str(key): tuple(str(ref) for ref in refs)
            for key, refs in (payload.get("slot_evidence_refs") or {}).items()
        },
        inferred_slots=tuple(str(item) for item in payload.get("inferred_slots") or ()),
    )


def accepted_frame_patterns_from_rules(
    active_rules: list[dict[str, Any]],
) -> tuple[FramePatternV01, ...]:
    """Load frame patterns only from already-active governed rule projections."""

    patterns: list[FramePatternV01] = []
    for rule in active_rules:
        scope = rule.get("applicability_scope") or {}
        payload = scope.get("frame_pattern") if isinstance(scope, dict) else None
        if not isinstance(payload, dict):
            continue
        graph_release_ref = str(
            rule.get("source_graph_release_ref")
            or payload.get("graph_release_ref")
            or "active_governed_rule_projection"
        )
        patterns.append(
            FramePatternV01(
                pattern_id=str(payload["pattern_id"]),
                frame=_frame_from_payload(dict(payload["frame"])),
                graph_release_ref=graph_release_ref,
                source_evidence_refs=tuple(
                    str(ref) for ref in payload.get("source_evidence_refs") or ()
                ),
                lifecycle_state="accepted",
                pattern_hash=str(payload["pattern_hash"]),
                corpus_schema_candidate_ref=str(
                    payload["corpus_schema_candidate_ref"]
                ),
            )
        )
    return tuple(patterns)


def accepted_composed_frame_patterns_from_rules(
    active_rules: list[dict[str, Any]],
) -> tuple[ComposedFramePatternV01, ...]:
    """Load composed patterns only from active governed rule projections."""

    patterns: list[ComposedFramePatternV01] = []
    for rule in active_rules:
        scope = rule.get("applicability_scope") or {}
        payload = (
            scope.get("composed_frame_pattern") if isinstance(scope, dict) else None
        )
        if not isinstance(payload, dict):
            continue
        graph_release_ref = str(
            rule.get("source_graph_release_ref")
            or payload.get("graph_release_ref")
            or "active_governed_rule_projection"
        )
        constituents = tuple(
            FramePatternV01(
                pattern_id=str(item["pattern_id"]),
                frame=_frame_from_payload(dict(item["frame"])),
                graph_release_ref=graph_release_ref,
                source_evidence_refs=tuple(
                    str(ref) for ref in item.get("source_evidence_refs") or ()
                ),
                lifecycle_state="accepted",
                pattern_hash=str(item["pattern_hash"]),
                corpus_schema_candidate_ref=str(
                    item["corpus_schema_candidate_ref"]
                ),
            )
            for item in payload.get("constituents") or ()
        )
        patterns.append(
            ComposedFramePatternV01(
                pattern_id=str(payload["pattern_id"]),
                composition_operator=str(payload["composition_operator"]),
                constituents=constituents,
                graph_release_ref=graph_release_ref,
                source_evidence_refs=tuple(
                    str(ref) for ref in payload.get("source_evidence_refs") or ()
                ),
                lifecycle_state="accepted",
                pattern_hash=str(payload["pattern_hash"]),
            )
        )
    return tuple(patterns)


def extract_composed_constraint_frames(
    text: str,
    *,
    evidence_ref: str,
) -> tuple[ConstraintFrameV01, ...]:
    """Extract the one bounded requirement + absence-consequence composition."""

    requirement = extract_constraint_frame(text, evidence_ref=evidence_ref)
    if requirement is None or requirement.attribute_ref != "certification":
        return ()
    semantics = load_operator_semantics()
    if not _contains_any(text, semantics["consequence_markers"]["invalid_bid"]):
        return (requirement,)
    requirement = ConstraintFrameV01(
        target_type_ref=requirement.target_type_ref,
        attribute_ref=requirement.attribute_ref,
        operator_ref=requirement.operator_ref,
        value_constraint=requirement.value_constraint,
        modality_ref=requirement.modality_ref,
        stage_ref=requirement.stage_ref,
        consequence_ref="ineligible",
        slot_evidence_refs=requirement.slot_evidence_refs,
        inferred_slots=requirement.inferred_slots,
    )
    invalidation = ConstraintFrameV01(
        target_type_ref="bid_submission",
        attribute_ref="required_certification_presence",
        operator_ref="IF_ABSENT",
        value_constraint=FrameValueConstraintV01(
            value_kind="concept_ref", concept_ref="cmmi_level_5"
        ),
        modality_ref="mandatory",
        stage_ref="submission_review",
        consequence_ref="invalid_bid",
        slot_evidence_refs={
            name: (evidence_ref,)
            for name in (
                "attribute_ref",
                "operator_ref",
                "value_constraint",
                "consequence_ref",
            )
        },
        inferred_slots=("target_type_ref", "modality_ref", "stage_ref"),
    )
    return (requirement, invalidation)


def compile_composed_frame_pattern_candidate(
    *,
    source_text: str,
    source_evidence_refs: tuple[str, ...],
    accepted_schema_projection: dict[str, Any],
    graph_release_ref: str,
) -> ComposedFramePatternV01:
    if accepted_schema_projection.get("review_state") not in {"accepted", "edited"}:
        raise ValueError("frame compilation requires an accepted corpus schema")
    frames = extract_composed_constraint_frames(
        source_text, evidence_ref=source_evidence_refs[0]
    )
    if len(frames) != 2:
        raise ValueError("source text did not produce the bounded ALL_OF composition")
    constituents: list[FramePatternV01] = []
    for index, frame in enumerate(frames):
        digest = content_hash({"index": index, "frame": asdict(frame)})
        constituents.append(
            FramePatternV01(
                pattern_id=f"frame-pattern.{digest[:24]}",
                frame=frame,
                graph_release_ref=graph_release_ref,
                source_evidence_refs=source_evidence_refs,
                lifecycle_state="candidate",
                pattern_hash=f"sha256:{digest}",
                corpus_schema_candidate_ref=str(
                    accepted_schema_projection["schema_candidate_id"]
                ),
            )
        )
    digest = content_hash(
        {"composition_operator": "ALL_OF", "constituents": [item.pattern_id for item in constituents]}
    )
    return ComposedFramePatternV01(
        pattern_id=f"composed-frame-pattern.{digest[:24]}",
        composition_operator="ALL_OF",
        constituents=tuple(constituents),
        graph_release_ref=graph_release_ref,
        source_evidence_refs=source_evidence_refs,
        lifecycle_state="candidate",
        pattern_hash=f"sha256:{digest}",
    )


def _value_equal(
    observed: FrameValueConstraintV01, canonical: FrameValueConstraintV01
) -> tuple[bool, bool]:
    if canonical.value_kind == "context_ref" and canonical.context_ref == "source_threshold":
        return observed.value_kind in {"normalized_quantity", "context_ref"}, False
    if observed.value_kind != canonical.value_kind:
        return False, False
    if observed.value_kind == "concept_ref":
        return observed.concept_ref == canonical.concept_ref, False
    if observed.value_kind == "normalized_quantity":
        observed_amount = observed.normalized_quantity.normalized_amount
        canonical_amount = canonical.normalized_quantity.normalized_amount
        return observed_amount == canonical_amount, False
    if observed.value_kind == "context_ref":
        return observed.context_ref == canonical.context_ref, False
    return observed.string_value == canonical.string_value, False


def match_constraint_frame(
    observed: ConstraintFrameV01,
    pattern: FramePatternV01,
    *,
    semantic_profile_id: str = SEMANTIC_PROFILE_ID,
) -> FrameMatchResultDTO:
    """Perform deterministic bounded slot unification."""

    canonical = pattern.frame
    diagnostics: list[FrameSlotDiagnosticV01] = []
    entails = False
    for slot in (
        "target_type_ref",
        "attribute_ref",
        "operator_ref",
        "value_constraint",
        "modality_ref",
        "stage_ref",
        "consequence_ref",
    ):
        observed_value = getattr(observed, slot)
        canonical_value = getattr(canonical, slot)
        rule_refs: tuple[str, ...] = ()
        if slot == "value_constraint":
            matched, slot_entails = _value_equal(observed_value, canonical_value)
            entails = entails or slot_entails
            rule_refs = ("tender_operator_semantics.v0_1:value",)
        elif slot == "operator_ref" and canonical_value == "GE" and observed_value == "GT":
            matched = True
            entails = True
            rule_refs = ("tender_operator_semantics.v0_1:GT_entails_GE",)
        elif slot == "consequence_ref" and canonical_value == "eligibility_restriction":
            matched = observed_value in {"eligibility_restriction", "invalid_bid"}
            entails = entails or observed_value == "invalid_bid"
            rule_refs = ("tender_operator_semantics.v0_1:consequence_strength",)
        else:
            matched = observed_value == canonical_value
        diagnostics.append(
            FrameSlotDiagnosticV01(
                slot_name=slot,
                matched=matched,
                observed_value=observed_value,
                canonical_value=canonical_value,
                rule_refs=rule_refs,
            )
        )
    success = all(item.matched for item in diagnostics)
    return FrameMatchResultDTO(
        semantic_relation=(
            "FRAME_ENTAILS" if success and entails else "FRAME_EQUIVALENT" if success else "NO_MATCH"
        ),
        derivation_shape="single_frame",
        matched_pattern_ref=pattern.pattern_id if success else None,
        constituent_matches=(),
        slot_diagnostics=tuple(diagnostics),
        semantic_profile_id=semantic_profile_id,
    )


def match_composed_frames(
    observed_frames: tuple[ConstraintFrameV01, ...],
    pattern: ComposedFramePatternV01,
    *,
    semantic_profile_id: str = SEMANTIC_PROFILE_ID,
) -> FrameMatchResultDTO:
    """Match the bounded ALL_OF composition with weakest-constituent rollup."""

    if pattern.composition_operator != "ALL_OF":
        raise ValueError("only ALL_OF composition is supported")
    ordered_observed = tuple(
        sorted(observed_frames, key=lambda frame: content_hash(asdict(frame)))
    )
    edges: list[list[tuple[int, FrameMatchResultDTO]]] = []
    for constituent in pattern.constituents:
        matches = []
        for index, observed in enumerate(ordered_observed):
            result = match_constraint_frame(
                observed, constituent, semantic_profile_id=semantic_profile_id
            )
            if result.semantic_relation != "NO_MATCH":
                matches.append((index, result))
        matches.sort(
            key=lambda item: (
                0 if item[1].semantic_relation == "FRAME_EQUIVALENT" else 1,
                content_hash(asdict(ordered_observed[item[0]])),
            )
        )
        edges.append(matches)

    assigned_observation: dict[int, int] = {}
    selected: dict[int, FrameMatchResultDTO] = {}

    def assign(constituent_index: int, visited: set[int]) -> bool:
        for observed_index, result in edges[constituent_index]:
            if observed_index in visited:
                continue
            visited.add(observed_index)
            prior = assigned_observation.get(observed_index)
            if prior is None or assign(prior, visited):
                assigned_observation[observed_index] = constituent_index
                selected[constituent_index] = result
                return True
        return False

    for constituent_index in range(len(pattern.constituents)):
        if not assign(constituent_index, set()):
            return FrameMatchResultDTO(
                semantic_relation="NO_MATCH",
                derivation_shape="composed_frame_set",
                matched_pattern_ref=None,
                constituent_matches=tuple(
                    selected[index] for index in sorted(selected)
                ),
                slot_diagnostics=(),
                semantic_profile_id=semantic_profile_id,
            )
    constituent_matches = [selected[index] for index in range(len(pattern.constituents))]
    rolled_up = (
        "FRAME_ENTAILS"
        if any(item.semantic_relation == "FRAME_ENTAILS" for item in constituent_matches)
        else "FRAME_EQUIVALENT"
    )
    return FrameMatchResultDTO(
        semantic_relation=rolled_up,
        derivation_shape="composed_frame_set",
        matched_pattern_ref=pattern.pattern_id,
        constituent_matches=tuple(constituent_matches),
        slot_diagnostics=(),
        semantic_profile_id=semantic_profile_id,
    )
