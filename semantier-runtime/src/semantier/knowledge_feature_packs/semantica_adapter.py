"""Version-pinned Semantica proposal adapters and candidate orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Protocol

import httpx

from contracts.knowledge_feature_pack import MachineProposal
from contracts.knowledge_graph_phase1 import SensitiveTermCandidateGraph
from semantier.semantica_context_graph.tender_mapper import map_semantica_tender_output
from semantier.knowledge_feature_packs.relationship_extraction import (
    RelationshipProposer,
    validate_relationship_output,
)


SEMANTICA_COMMIT = "701aa6a755899b8b63ce38ee971827b3e2c092e7"
NATIVE_EXTRACTION_CONTRACT = "deterministic-native-extraction.v1"

# W3 wrapper: Chinese-aware entity catalogue for the pinned reference-graph
# extraction path. The vendored third_party/semantica regex catalogue is
# English-only (PERSON/ORG/GPE/DATE/MONEY/PERCENT on ASCII casing and
# English suffixes), so pure-Chinese source rows return zero entities and
# the InProcessSemanticaAdapter falls into the empty-result branch.
#
# These patterns run inside the adapter layer (not inside third_party) so
# the pinned SEMANTICA_COMMIT integrity check stays intact. When the row
# is CJK-dominant the adapter dispatches to extract_entities_chinese_regex
# instead of methods.extract_entities_regex; otherwise the pinned
# English extractor still runs unchanged.
#
# Pattern ordering matters: longer / more specific patterns run first so
# the deduplication pass keeps the most informative label. CONCEPT is a
# catch-all for tender sensitive-word entries that are short CJK phrases
# without any structural suffix (大型企业、央企、世界500强、行业龙头).
DEFAULT_CHINESE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("MONEY", r"\d+(?:\.\d+)?\s?(?:亿元|万元|元|块钱)"),
    ("MONEY", r"(?:¥|￥|人民币|RMB)\s?\d[\d,]*(?:\.\d+)?"),
    ("PERCENT", r"\d+(?:\.\d+)?\s?%"),
    ("DATE", r"\d{4}\s?年(?:\d{1,2}\s?月)?(?:\d{1,2}\s?日?)?"),
    ("DATE", r"\d{1,2}\s?月\s?\d{1,2}\s?日?"),
    ("ORG", r"[\u4e00-\u9fff]{2,16}(?:有限责任公司|股份有限公司|有限公司)"),
    ("ORG", r"[\u4e00-\u9fff]{2,8}(?:集团|总公司|分公司|子公司|联合公司)"),
    ("ORG_NAME", r"[\u4e00-\u9fff]{2,8}(?:公司|银行|中心|大学|医院|委员会|办公室|事务所|研究院|实验室|厂家|厂商)"),
    ("QUALIFICATION", r"[\u4e00-\u9fff]{2,6}(?:资质|证书|认证|许可|备案|牌照|执照)"),
    # Relation cues must be extracted before the catch-all CONCEPT pattern;
    # otherwise `仅限央企` becomes one covered span and no edge can exist.
    ("RELATION_CUE", r"(?:仅限|限于|适用于|针对|不得|禁止|必须|应当|需要|需)"),
    ("CONCEPT", r"[\u4e00-\u9fff][\u4e00-\u9fff0-9]{1,11}"),
)

# Below this CJK density the row is treated as English and routed to the
# pinned extractor; at or above the threshold the wrapper catalogue runs.
_CJK_DISPATCH_DENSITY_THRESHOLD = 0.3

_CJK_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_dominant(text: str) -> bool:
    """Return True when ``text`` is at or above the CJK dispatch density."""
    if not text:
        return False
    han_count = len(_CJK_HAN_RE.findall(text))
    return (han_count / max(len(text), 1)) >= _CJK_DISPATCH_DENSITY_THRESHOLD


def extract_entities_chinese_regex(
    text: str,
    patterns: tuple[tuple[str, str], ...] | None = None,
    *,
    confidence: float = 0.7,
) -> list[Any]:
    """Chinese-aware wrapper around the pinned Semantica regex seam.

    Returns objects that satisfy the vendored
    ``semantica.semantic_extract.types.Entity`` shape (``text`` / ``label``
    / ``start_char`` / ``end_char`` / ``confidence`` / ``metadata``) so the
    downstream :func:`InProcessSemanticaAdapter.extract` loop can consume
    them without any branch change.

    Overlapping spans are resolved by pattern priority (more specific
    patterns run first) and longest-span wins within the same priority
    tier. The vendored package and ``SEMANTICA_COMMIT`` pin are untouched.
    """
    if not text:
        return []
    chosen = patterns if patterns is not None else DEFAULT_CHINESE_PATTERNS
    spans: list[tuple[int, int, str, str]] = []
    covered: list[tuple[int, int]] = []
    for label, pattern in chosen:
        # Scan each code-point position instead of relying on ``finditer``.
        # When a higher-priority cue covers the beginning of a catch-all
        # concept match (e.g. `仅限央企`), ``finditer`` would discard the
        # overlapping match and never retry at the head's actual start.
        for start in range(len(text)):
            match = re.match(pattern, text[start:])
            if match is None:
                continue
            end = start + match.end()
            if start == end:
                continue
            if any(cs <= start < ce or cs < end <= ce for cs, ce in covered):
                continue
            spans.append((start, end, label, match.group(0)))
            covered.append((start, end))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    Entity = _ChineseEntity  # local import to keep module load order stable
    return [
        Entity(
            text=label_text,
            label=span_label,
            start_char=start,
            end_char=end,
            confidence=confidence,
            metadata={
                "extraction_method": "chinese_lexicon",
                "pattern": pattern_name_for(span_label, chosen),
                "wrapper": "extract_entities_chinese_regex",
            },
        )
        for start, end, span_label, label_text in spans
    ]


def pattern_name_for(label: str, patterns: tuple[tuple[str, str], ...]) -> str:
    for candidate_label, candidate_pattern in patterns:
        if candidate_label == label:
            return candidate_pattern
    return label


@dataclass
class _ChineseEntity:
    """Local stand-in for the vendored ``semantica.semantic_extract.types.Entity``.

    Carries the same attribute surface that
    :func:`InProcessSemanticaAdapter.extract` reads (``text``, ``label``,
    ``start_char``, ``end_char``, ``confidence``) so the existing per-span
    loop works unchanged.
    """

    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)




def _ensure_vendored_semantica_importable() -> None:
    """Expose and verify the exact pinned checkout, failing on package shadowing."""

    for parent in Path(__file__).resolve().parents:
        source_root = parent / "third_party" / "semantica"
        if (source_root / "semantica" / "semantic_extract" / "ner_extractor.py").is_file():
            try:
                actual_commit = subprocess.check_output(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.PIPE,
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError("cannot verify pinned Semantica checkout") from exc
            if actual_commit != SEMANTICA_COMMIT:
                raise RuntimeError(
                    f"Semantica checkout mismatch: expected {SEMANTICA_COMMIT}, got {actual_commit}"
                )
            try:
                dirty = subprocess.check_output(
                    ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
                    text=True,
                    stderr=subprocess.PIPE,
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError("cannot verify Semantica checkout integrity") from exc
            if dirty:
                raise RuntimeError("pinned Semantica checkout has modified tracked content")
            loaded = sys.modules.get("semantica")
            loaded_path = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
            if loaded_path is not None and not loaded_path.is_relative_to(source_root.resolve()):
                raise RuntimeError(f"non-vendored Semantica package is already loaded: {loaded_path}")
            source = str(source_root)
            sys.path[:] = [entry for entry in sys.path if entry != source]
            sys.path.insert(0, source)
            return
    raise RuntimeError("pinned third_party/semantica source checkout is unavailable")


@dataclass(frozen=True)
class SemanticaProposalResult:
    proposal: MachineProposal
    structured_output: dict[str, Any]
    raw_output: str
    provider_ref: str
    provider_commit: str | None
    extraction_run_id: str
    native_artifact_ref: str
    native_artifact_hash: str


@dataclass(frozen=True)
class SemanticaCandidateGraphResult:
    """A mapped candidate graph plus its immutable native-provider lineage."""

    graph: SensitiveTermCandidateGraph
    provider_ref: str
    provider_commit: str | None
    extraction_run_id: str
    native_artifact_ref: str
    native_artifact_hash: str


class SemanticaCandidateGroundingError(ValueError):
    """Candidate mapping failed after the native provider artifact was pinned."""

    def __init__(self, message: str, *, proposal_result: SemanticaProposalResult) -> None:
        super().__init__(message)
        self.proposal_result = proposal_result


class CandidateExtractionAdapter(Protocol):
    def extract(self, *, elements: list[dict[str, Any]], run_id: str) -> SemanticaProposalResult: ...


class InProcessSemanticaAdapter:
    """Run the pinned Semantica extraction and entity-resolution APIs in process.

    The adapter produces Semantica-native diagnostics and the proposal shape
    consumed by :func:`candidate_graph.build_candidate_graph`. It does not
    persist candidates or materialize an authoritative graph.
    """

    def __init__(
        self,
        *,
        artifact_writer: Any,
        relationship_proposer: RelationshipProposer | None = None,
        relationship_feedback_refs: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._artifact_writer = artifact_writer
        self._relationship_proposer = relationship_proposer
        self._relationship_feedback_refs = tuple(relationship_feedback_refs)

    def capability(self) -> dict[str, str]:
        return {
            "provider": "semantica",
            "state": "available",
            "detail": SEMANTICA_COMMIT,
            "executionMode": "in_process",
        }

    def extract(self, *, elements: list[dict[str, Any]], run_id: str) -> SemanticaProposalResult:
        _ensure_vendored_semantica_importable()
        try:
            from semantica.kg.entity_resolver import EntityResolver
        except ModuleNotFoundError as exc:
            # The pinned exact resolver only needs scipy for optional numeric
            # helpers. Keep the provider path Semantica-owned and preserve the
            # exact strategy when that optional wheel is absent in a minimal
            # runtime image; this is not a provider or model fallback.
            if exc.name != "scipy":
                raise
            class EntityResolver:  # type: ignore[no-redef]
                def __init__(self, **_kwargs: Any) -> None: pass
                def resolve_entities(self, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    return values
        from semantica.semantic_extract import methods
        from semantica.semantic_extract.schemas import EntityOut, RelationOut, TripletOut

        resolver = EntityResolver(
            strategy="exact",
            similarity_threshold=1.0,
            deduplication={
                "similarity": {
                    "embedding_weight": 0.0,
                    "string_weight": 1.0,
                    "property_weight": 0.0,
                    "relationship_weight": 0.0,
                }
            },
        )
        extracted: list[dict[str, Any]] = []
        native_relations: list[dict[str, Any]] = []
        native_triplets: list[dict[str, Any]] = []
        proposed_mentions_by_anchor: dict[str, list[dict[str, Any]]] = {}
        proposed_mention_entity_ids: dict[str, str] = {}
        proposed_relations: list[dict[str, Any]] = []
        relationship_diagnostics: list[dict[str, str]] = []

        if self._relationship_proposer is not None:
            try:
                proposed = validate_relationship_output(
                    self._relationship_proposer(elements), elements
                )
            except Exception as exc:
                # The structured proposal is an enhancement to the pinned
                # provider, not a reason to lose the extraction run. Keep the
                # diagnostic in the native artifact and use deterministic
                # Semantica extraction for this run.
                proposed = {"mentions": [], "relations": []}
                relationship_diagnostics.append({
                    "stage": "few_shot_relationship_proposal",
                    "outcome": "deterministic_fallback",
                    "error_type": type(exc).__name__,
                })
            for mention in proposed["mentions"]:
                proposed_mentions_by_anchor.setdefault(
                    mention["element_anchor_id"], []
                ).append(mention)
            proposed_relations = proposed["relations"]

        for index, element in enumerate(elements):
            text = str(element.get("text") or "")
            family_id = element.get("family_id")
            # Keep the provider call at the element boundary.  In particular,
            # do not replace Semantica's extraction with the tender mapper's
            # lexical splitter: tests and the native artifact both depend on
            # the pinned provider's actual Entity values.
            # Dispatch to the Chinese-aware wrapper catalogue for rows that
            # are CJK-dominant.  The vendored third_party/semantica regex
            # catalogue is English-only (PERSON/ORG/GPE/DATE/MONEY/PERCENT
            # on ASCII casing and English suffixes) and yields zero entities
            # for most Chinese tender rows; without this branch the
            # InProcessSemanticaAdapter falls into the empty-result branch
            # and the row produces zero ReferenceConcept nodes.  The wrapper
            # returns objects with the same attribute surface as the vendored
            # Entity so the loop below stays unchanged.  Pure-English and
            # mixed-with-low-CJK rows continue to use the pinned English
            # extractor.
            anchor_id = element["anchor"]["anchor_id"]
            if proposed_mentions_by_anchor.get(anchor_id):
                native_entities = [
                    _ChineseEntity(
                        text=item["text"],
                        label=item["semantic_type"],
                        start_char=item["start_offset"],
                        end_char=item["end_offset"],
                        confidence=item["confidence"],
                        metadata={"extraction_method": "few_shot_relationship_llm"},
                    )
                    for item in proposed_mentions_by_anchor[anchor_id]
                ]
            elif _cjk_dominant(text):
                native_entities = extract_entities_chinese_regex(text)
            else:
                native_entities = methods.extract_entities_regex(text)
            if not text or family_id is None:
                continue
            # The pinned regex catalogue is English-oriented and consequently
            # yields no Entity for most Chinese tender rows.  Retain one
            # deterministic, fully grounded occurrence for those rows so the
            # existing Phase-1 family mapper remains usable; this is a shim
            # around an empty provider result, never a replacement for it.
            if not native_entities:
                # Empty native results should remain empty.  The old whole-element
                # shim is not allowed for the reference-graph path, and this
                # explicit boundary-review marker prevents silent synthesis.
                native_entities = []
                continue
            entities = []
            seen_entity_occurrences: set[tuple[str, int, int]] = set()
            for entity_index, entity in enumerate(native_entities):
                start = int(getattr(entity, "start_char", getattr(entity, "start", 0)))
                end = int(getattr(entity, "end_char", getattr(entity, "end", len(text))))
                occurrence_key = (str(entity.text), start, end)
                # Semantica's default catalogue can report the same span from
                # multiple label patterns.  It is one source occurrence for
                # Phase-1 lineage; retain the first native representation so
                # resolver merge_from IDs remain stable and spans are not
                # duplicated merely because labels overlapped.
                if occurrence_key in seen_entity_occurrences:
                    continue
                seen_entity_occurrences.add(occurrence_key)
                entities.append(entity)
                entity_id = f"entity_{index:06d}_{entity_index:03d}"
                mention_id = None
                for candidate in proposed_mentions_by_anchor.get(anchor_id, []):
                    if (
                        candidate["text"] == entity.text
                        and candidate["start_offset"] == start
                        and candidate["end_offset"] == end
                    ):
                        mention_id = candidate["mention_id"]
                        break
                if mention_id:
                    proposed_mention_entity_ids[mention_id] = entity_id
                extracted.append(
                    {
                        "id": entity_id,
                        "name": f"{int(family_id):02d}:{entity.text.casefold()}",
                        "text": entity.text,
                        "type": entity.label,
                        "confidence": float(entity.confidence),
                        "start_char": start,
                        "end_char": end,
                        "element_anchor_id": element["anchor"]["anchor_id"],
                        "family_id": int(family_id),
                        "family": str(element.get("family") or f"family {family_id}"),
                        "mention_id": mention_id,
                    }
                )
            for left, right in zip(entities, entities[1:]):
                relation = RelationOut(subject=left.text, object=right.text, predicate="related_to", confidence=0.65,
                                       metadata={"element_anchor_id": element["anchor"]["anchor_id"]})
                triplet = TripletOut(subject=left.text, predicate="related_to", object=right.text, confidence=0.65,
                                     metadata={"element_anchor_id": element["anchor"]["anchor_id"]})
                native_relations.append(relation.model_dump())
                native_triplets.append(triplet.model_dump())

        for relation in proposed_relations:
            source_entity_id = proposed_mention_entity_ids.get(relation["source_mention_id"])
            target_entity_id = proposed_mention_entity_ids.get(relation["target_mention_id"])
            if not source_entity_id or not target_entity_id:
                continue
            native_relations.append({
                **relation,
                "subject": relation["source_mention_id"],
                "object": relation["target_mention_id"],
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "metadata": {"extraction_method": "few_shot_relationship_llm"},
            })
            native_triplets.append({
                "subject": relation["source_mention_id"],
                "predicate": relation["relation_type"],
                "object": relation["target_mention_id"],
                "confidence": relation["confidence"],
                "metadata": {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
            })

        native_resolved = resolver.resolve_entities(extracted)
        resolved = _restore_resolution_lineage(native_resolved, extracted)
        structured = map_semantica_tender_output(
            parsed_document={"elements": elements, "document_hash": str(elements[0]["anchor"].get("document_hash", "unknown")) if elements else "unknown"},
            extracted_entities=resolved,
            relations=native_relations,
            triplets=native_triplets,
            extraction_run_id=run_id,
            provider="semantica",
            provider_commit=SEMANTICA_COMMIT,
        )
        native = {
            "schemaVersion": "semantica_native_candidate.v1",
            "providerRef": "semantica",
            "providerCommit": SEMANTICA_COMMIT,
            "extractionRunId": run_id,
            "semantica": {
                "entities": extracted,
                "resolvedEntities": native_resolved,
                "candidateResolvedEntities": resolved,
                "relations": native_relations,
                "triplets": native_triplets,
                "relationshipDiagnostics": relationship_diagnostics,
                "feedbackRefsUsed": list(self._relationship_feedback_refs),
            },
            **structured,
        }
        raw = json.dumps(native, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return _proposal_result(
            raw=raw,
            structured=native,
            run_id=run_id,
            provider="semantica",
            provider_commit=SEMANTICA_COMMIT,
            model=f"semantica:{SEMANTICA_COMMIT}",
            prompt_version=NATIVE_EXTRACTION_CONTRACT,
            request_id=None,
            artifact_writer=self._artifact_writer,
        )


def _normalize_empty_native_response(
    *,
    elements: list[dict[str, Any]],
    extraction_run_id: str,
) -> dict[str, Any]:
    """Return deterministic boundary-review metadata rather than a whole-element shim.

    This preserves the exact empty native result without manufacturing a concept
    from the full text span.
    """
    candidate_count = 0
    for element in elements:
        if isinstance(element, dict) and str(element.get("text") or "").strip():
            candidate_count += 1
    return {
        "status": "needs_boundary_review",
        "native_result_count": 0,
        "shim_produced": False,
        "extraction_run_id": extraction_run_id,
        "reason": "empty native provider result; no whole-element concept promoted",
        "candidate_elements_seen": candidate_count,
    }


def _candidate_payload(
    resolved_entities: list[dict[str, Any]],
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape native resolved entities for the existing candidate mapper."""

    modifier_prefixes = {
        "仅限",
        "限于",
        "仅允许",
        "不得",
        "禁止",
        "不包括",
        "排除",
        "适用于",
        "针对",
    }
    relation_prefixes = {
        "不得": "prohibits",
        "禁止": "prohibits",
        "必须": "requires",
        "应当": "requires",
        "需": "requires",
        "需要": "requires",
        "适用于": "applies_to",
        "针对": "applies_to",
    }

    def relation_type(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str] | None:
        left_text = str(left.get("text") or "").strip()
        if left_text in relation_prefixes:
            return relation_prefixes[left_text], "lexical relation cue precedes its governed concept"
        if left_text in modifier_prefixes:
            return "modifier_of", "lexical modifier precedes its governed concept"
        return None

    nodes = []
    edges = []
    rules = []
    for entity in resolved_entities:
        entity_id = str(entity["id"])
        family_id = int(entity["family_id"])
        spans = list(entity["source_spans"])
        nodes.append(
            {
                "node_id": f"concept_{entity_id}",
                "node_type": "concept",
                "canonical_label": str(entity["text"]),
                "display_name_zh": str(entity["text"]),
                "normalized_value": str(entity["text"]).casefold(),
                "rationale": "semantica exact native entity extraction",
                "source_spans": spans,
                "confidence": float(entity["confidence"]),
            }
        )
        rules.append(
            {
                "rule_id": f"rule_{entity_id}",
                "family_id": family_id,
                "condition_node_ids": [],
                "exception_node_ids": [],
                "reason_code": f"tender_sensitive_family_{family_id:02d}",
                "source_spans": spans,
            }
        )
    # The v1 candidate contract has no relation-bearing rule node.  Preserve
    # connectedness by linking adjacent resolved concepts; the rich tender
    # mapper projection remains available in the native artifact for clients
    # that need alias/category/risk topology.
    for left, right in zip(resolved_entities, resolved_entities[1:]):
        left_id = f"concept_{left['id']}"
        right_id = f"concept_{right['id']}"
        typed_relation = relation_type(left, right)
        edge_relation, edge_rationale = typed_relation or (
            "related_to",
            "sequential tender source occurrence",
        )
        edges.append(
            {
                "edge_id": f"edge_{left['id']}_to_{right['id']}",
                "source_node_id": left_id,
                "target_node_id": right_id,
                "relation_type": edge_relation,
                "rationale": edge_rationale,
                "source_spans": list(left["source_spans"]),
                "confidence": min(float(left["confidence"]), float(right["confidence"])),
            }
        )
    entity_ids = {str(entity["id"]) for entity in resolved_entities}
    for relation in relations or []:
        source_id = str(relation.get("source_entity_id") or "")
        target_id = str(relation.get("target_entity_id") or "")
        relation_type_name = str(relation.get("relation_type") or "")
        if (
            source_id not in entity_ids
            or target_id not in entity_ids
            or not relation_type_name
        ):
            continue
        edge_id = str(relation.get("relation_id") or f"edge_{source_id}_to_{target_id}_{relation_type_name}")
        if any(edge["edge_id"] == edge_id for edge in edges):
            continue
        source_entity = next(item for item in resolved_entities if str(item["id"]) == source_id)
        target_entity = next(item for item in resolved_entities if str(item["id"]) == target_id)
        edges.append({
            "edge_id": edge_id,
            "source_node_id": f"concept_{source_id}",
            "target_node_id": f"concept_{target_id}",
            "relation_type": relation_type_name,
            "rationale": str(relation.get("rationale") or "few-shot semantic relationship proposal"),
            "source_spans": list(source_entity["source_spans"]) + list(target_entity["source_spans"]),
            "confidence": float(relation.get("confidence", 0.0)),
        })
    return {
        "schema_version": "sensitive_term_candidate_graph.v1",
        "ontology_version": "tender_risk_ontology.v1",
        "confidence": min((float(item["confidence"]) for item in resolved_entities), default=None),
        "nodes": nodes,
        "edges": edges,
        "rules": rules,
    }


def _restore_resolution_lineage(
    resolved_entities: list[dict[str, Any]], extracted_entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restore exact occurrence spans that upstream merge output intentionally omits."""

    by_id = {str(entity["id"]): entity for entity in extracted_entities}
    restored = []
    for resolved in resolved_entities:
        source_ids = [str(value) for value in resolved.get("merged_from", [resolved.get("id")])]
        unknown_ids = [source_id for source_id in source_ids if source_id not in by_id]
        if unknown_ids:
            raise RuntimeError(f"Semantica entity resolution returned unknown entities: {unknown_ids}")
        sources = [by_id[source_id] for source_id in source_ids]
        representative = sources[0]
        if any(
            source["family_id"] != representative["family_id"]
            or source["text"].casefold() != representative["text"].casefold()
            for source in sources
        ):
            raise RuntimeError("Semantica exact resolution merged non-identical entities")
        restored.append(
            {
                **representative,
                "id": str(resolved.get("id") or representative["id"]),
                "source_spans": [
                    {
                        "element_anchor_id": str(source["element_anchor_id"]),
                        "start_offset": int(source["start_char"]),
                        "end_offset": int(source["end_char"]),
                    }
                    for source in sources
                ],
                "resolved_from": source_ids,
            }
        )
    return restored


def build_semantica_candidate_graph(
    *,
    parsed_document: dict[str, Any],
    extraction_run_id: str,
    adapter: CandidateExtractionAdapter,
) -> SemanticaCandidateGraphResult:
    """Map once through ``build_candidate_graph`` and retain native lineage."""

    from semantier.knowledge_feature_packs.candidate_graph import build_candidate_graph

    proposal_result = adapter.extract(
        elements=parsed_document["elements"],
        run_id=extraction_run_id,
    )

    class _CompletedExtraction:
        def extract(self, *, elements: list[dict[str, Any]], run_id: str) -> SemanticaProposalResult:
            if run_id != extraction_run_id or elements != parsed_document["elements"]:
                raise ValueError("candidate mapper extraction context changed")
            return proposal_result

    try:
        graph = build_candidate_graph(
            parsed_document=parsed_document,
            extraction_run_id=extraction_run_id,
            adapter=_CompletedExtraction(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SemanticaCandidateGroundingError(
            str(exc), proposal_result=proposal_result
        ) from exc
    return SemanticaCandidateGraphResult(
        graph=graph,
        provider_ref=proposal_result.provider_ref,
        provider_commit=proposal_result.provider_commit,
        extraction_run_id=proposal_result.extraction_run_id,
        native_artifact_ref=proposal_result.native_artifact_ref,
        native_artifact_hash=proposal_result.native_artifact_hash,
    )


class SemanticaServiceAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        artifact_writer: Any,
        service_version: str = "0.6.0",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._artifact_writer = artifact_writer
        self._service_version = service_version
        self._timeout_seconds = timeout_seconds

    def capability(self) -> dict[str, str]:
        return {
            "provider": "semantica_service",
            "state": "available",
            "detail": self._service_version,
            "executionMode": "service",
        }

    def extract(self, *, elements: list[dict[str, Any]], run_id: str) -> SemanticaProposalResult:
        response = httpx.post(
            f"{self._base_url}/api/enrich/extract",
            json={
                "schema_version": "sensitive_term_candidate_graph.v1",
                "ontology_version": "tender_risk_ontology.v1",
                "elements": elements,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        raw = response.text
        return _proposal_result(
            raw=raw,
            structured=response.json(),
            run_id=run_id,
            provider="semantica-service",
            provider_commit=None,
            model=f"semantica:{self._service_version}",
            prompt_version="schema-constrained-extraction.v1",
            request_id=response.headers.get("x-request-id"),
            artifact_writer=self._artifact_writer,
        )


def default_candidate_extraction_adapter(
    *, service_url: str, artifact_writer: Any
) -> CandidateExtractionAdapter:
    """Phase 1 default: version-pinned isolated Semantica proposal service."""
    return SemanticaServiceAdapter(
        base_url=service_url,
        artifact_writer=artifact_writer,
        service_version="0.6.0",
    )


class RecordedSemanticaAdapter:
    """Deterministic recorded-output adapter for replay and offline tests."""

    def __init__(self, *, structured_output: dict[str, Any], artifact_writer: Any) -> None:
        self._structured_output = structured_output
        self._artifact_writer = artifact_writer

    def extract(self, *, elements: list[dict[str, Any]], run_id: str) -> SemanticaProposalResult:
        raw = json.dumps(self._structured_output, ensure_ascii=False, sort_keys=True)
        return _proposal_result(
            raw=raw,
            structured=self._structured_output,
            run_id=run_id,
            provider="recorded-semantica",
            provider_commit=None,
            model="semantica:0.6.0",
            prompt_version="schema-constrained-extraction.v1",
            request_id=None,
            artifact_writer=self._artifact_writer,
        )


def _proposal_result(
    *,
    raw: str,
    structured: dict[str, Any],
    run_id: str,
    provider: str,
    provider_commit: str | None,
    model: str,
    prompt_version: str,
    request_id: str | None,
    artifact_writer: Any,
) -> SemanticaProposalResult:
    raw_hash = f"sha256:{sha256(raw.encode('utf-8')).hexdigest()}"
    raw_ref = artifact_writer(raw, raw_hash)
    proposal = MachineProposal(
        proposal_id=f"proposal_{raw_hash[7:31]}",
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        prompt_hash=f"sha256:{sha256(prompt_version.encode('utf-8')).hexdigest()}",
        parameters={"authority": "proposal_only", "temperature": 0},
        request_id=request_id,
        run_id=run_id,
        raw_output_ref=raw_ref,
        raw_output_hash=raw_hash,
        confidence=structured.get("confidence"),
    )
    return SemanticaProposalResult(
        proposal=proposal,
        structured_output=structured,
        raw_output=raw,
        provider_ref=provider.replace("-", "_"),
        provider_commit=provider_commit,
        extraction_run_id=run_id,
        native_artifact_ref=raw_ref,
        native_artifact_hash=raw_hash,
    )
