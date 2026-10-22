"""Deterministic SemanticOccurrence to DocumentCandidateAssertion grouping."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from contracts.knowledge_feature_pack import MachineProposal, content_hash
from contracts.knowledge_reasoning_phase1 import (
    DocumentCandidateAssertion,
    GroundingState,
    NormalizedAssertion,
    SemanticOccurrence,
)


def group_occurrences(
    occurrences: Iterable[SemanticOccurrence],
    *,
    candidate_graph_id: str,
    source_document_hash: str,
    machine_proposal: MachineProposal,
) -> tuple[DocumentCandidateAssertion, ...]:
    items = tuple(occurrences)
    grouped: dict[tuple[str, ...], list[SemanticOccurrence]] = defaultdict(list)
    for occurrence in items:
        attrs = occurrence.extraction_attributes
        signature = _normalized_signature(occurrence)
        if occurrence.group_ref:
            key = (occurrence.document_id, "group", occurrence.group_ref, *signature)
        else:
            # Without provider grouping, only overlapping/nested evidence with
            # the same semantic structure may collapse. Separate clauses stay separate.
            overlap_group = _overlap_group(occurrence, grouped, signature)
            key = overlap_group or (
                occurrence.document_id,
                "anchor",
                occurrence.source_anchor.anchor_id,
                *signature,
            )
        grouped[key].append(occurrence)

    assertions = []
    for key, members in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(
            members,
            key=lambda item: (
                item.source_anchor.start_offset,
                item.source_anchor.end_offset,
                item.occurrence_id,
            ),
        )
        normalized = _normalized_assertion(ordered)
        identity = {
            "candidate_graph_id": candidate_graph_id,
            "occurrence_refs": [item.occurrence_id for item in ordered],
            "normalized": normalized.model_dump(mode="json"),
        }
        assertions.append(
            DocumentCandidateAssertion(
                assertion_id=f"assertion_{content_hash(identity)[7:31]}",
                candidate_graph_id=candidate_graph_id,
                source_document_hash=source_document_hash,
                source_anchors=tuple(item.source_anchor for item in ordered),
                occurrence_count=len(ordered),
                occurrence_refs=tuple(item.occurrence_id for item in ordered),
                proposal_source="llm_proposal",
                machine_proposal=machine_proposal,
                confidence=_mean_confidence(ordered),
                grounding_state=GroundingState.UNRESOLVED,
                normalized_assertion=normalized,
            )
        )
    return tuple(assertions)


def _normalized_signature(occurrence: SemanticOccurrence) -> tuple[str, ...]:
    attrs = occurrence.extraction_attributes
    return tuple(
        str(attrs.get(field) or "").strip().casefold()
        for field in ("assertion_type", "subject_text", "predicate_text", "object_text")
    )


def _overlap_group(
    occurrence: SemanticOccurrence,
    grouped: dict[tuple[str, ...], list[SemanticOccurrence]],
    signature: tuple[str, ...],
) -> tuple[str, ...] | None:
    start = occurrence.source_anchor.start_offset
    end = occurrence.source_anchor.end_offset
    for key, members in grouped.items():
        if key[0] != occurrence.document_id or key[-len(signature):] != signature:
            continue
        if any(start < item.source_anchor.end_offset and end > item.source_anchor.start_offset for item in members):
            return key
    return None


def _normalized_assertion(occurrences: list[SemanticOccurrence]) -> NormalizedAssertion:
    def first(field: str):
        return next(
            (item.extraction_attributes[field] for item in occurrences if item.extraction_attributes.get(field) not in (None, "")),
            None,
        )

    relation = next((item for item in occurrences if item.extraction_class == "relationship"), occurrences[0])
    attrs = relation.extraction_attributes
    candidate_relation_id = None
    if any(attrs.get(field) for field in ("subject_text", "predicate_text", "object_text")):
        relation_basis = {
            "occurrences": [item.occurrence_id for item in occurrences],
            "subject": attrs.get("subject_text"),
            "predicate": attrs.get("predicate_text"),
            "object": attrs.get("object_text"),
        }
        candidate_relation_id = f"candidate_relation_{content_hash(relation_basis)[7:31]}"
    return NormalizedAssertion(
        assertion_type=first("assertion_type") or relation.extraction_class,
        subject={"text": first("subject_text")} if first("subject_text") else None,
        predicate=first("predicate_text"),
        object={"text": first("object_text")} if first("object_text") else None,
        modality=first("modality"),
        quantifier=first("quantifier"),
        condition=first("condition"),
        exception=first("exception"),
        consequence=first("consequence"),
        document_role=first("document_role"),
        candidate_family_id=first("candidate_family_id"),
        candidate_concept_ref=first("candidate_concept_ref"),
        candidate_relation_id=candidate_relation_id,
    )


def _mean_confidence(occurrences: list[SemanticOccurrence]) -> float:
    values = [item.machine_confidence for item in occurrences if item.machine_confidence is not None]
    return sum(values) / len(values) if values else 0.0
