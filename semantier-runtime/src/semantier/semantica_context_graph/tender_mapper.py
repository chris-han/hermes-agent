"""Deterministic Semantica-to-Semantier tender mapper."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .tender_schema import TenderExtractionResult


MAPPER_VERSION = "semantier-tender-mapper.v1"
_SEPARATORS = re.compile(r"[、，,；;。/|\n]+")
_NON_MENTION = {"场景分类", "违规敏感词", "违规示例", "通用全场景"}


def _mentions(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield bounded lexical mentions, never the complete cell by default."""
    cursor = 0
    for part in _SEPARATORS.split(text):
        value = part.strip()
        start = text.find(value, cursor) if value else -1
        if value and start >= 0:
            cursor = start + len(value)
            if value not in _NON_MENTION and len(value) >= 2:
                yield value, start, start + len(value)


def map_semantica_tender_output(
    *,
    parsed_document: dict[str, Any],
    extracted_entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    triplets: list[dict[str, Any]],
    extraction_run_id: str,
    provider: str,
    provider_commit: str,
) -> dict[str, Any]:
    """Return the existing candidate payload plus a governed extraction view.

    Canonical concepts are keyed by the document's tender family. Mentions are
    aliases attached to those concepts; clauses and risk links are represented
    as connected candidate nodes, preserving the provider's exact spans.
    """
    del relations, triplets  # retained in the native artifact; mapping is deterministic.
    by_anchor = {e["anchor"]["anchor_id"]: e for e in parsed_document["elements"]}
    entities: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    family_ids: dict[int, str] = {}
    seen_aliases: set[str] = set()
    family_evidence: dict[int, dict[str, Any]] = {}

    for index, source in enumerate(extracted_entities):
        anchor_id = str(source["element_anchor_id"])
        element = by_anchor[anchor_id]
        family_id = int(element.get("family_id") or 1)
        family_label = str(element.get("family") or f"tender_family_{family_id:02d}")
        span = {
            "element_anchor_id": anchor_id,
            "start_offset": int(source["start_char"]),
            "end_offset": int(source["end_char"]),
        }
        entity_id = f"alias_{family_id:02d}_{index:05d}"
        concept_id = f"concept_{family_id:02d}"
        category_id = f"category_tender_family_{family_id:02d}"
        risk_id = f"risk_discriminatory_requirement_{family_id:02d}"
        family_ids.setdefault(family_id, concept_id)
        family_evidence.setdefault(family_id, {"anchor_id": anchor_id, "source_hash": parsed_document["document_hash"],
                                               "locator": anchor_id, "quote": str(source["text"]),
                                               "start_offset": int(source["start_char"]), "end_offset": int(source["end_char"])})
        if entity_id in seen_aliases:
            continue
        seen_aliases.add(entity_id)
        normalized = str(source["text"]).casefold()
        entities.append({
            "item_id": entity_id,
            "machine_identifier": entity_id,
            "display_label": str(source["text"]),
            "layer": "alias",
            "normalized_value": normalized,
            "confidence": float(source.get("confidence", 0.8)),
            "extraction_run_id": extraction_run_id,
            "mapper_version": MAPPER_VERSION,
            "evidence": [{"anchor_id": anchor_id, "source_hash": parsed_document["document_hash"],
                          "locator": anchor_id, "quote": str(source["text"]),
                          "start_offset": int(source["start_char"]), "end_offset": int(source["end_char"])}],
        })
        nodes.extend([
            {"node_id": entity_id, "node_type": "alias", "canonical_label": str(source["text"]),
             "display_name_zh": str(source["text"]), "normalized_value": normalized,
             "rationale": "Semantica entity mention mapped to tender family", "source_spans": [span],
             "confidence": float(source.get("confidence", 0.8))},
            {"node_id": concept_id, "node_type": "concept", "canonical_label": family_label,
             "display_name_zh": family_label, "normalized_value": f"tender_family_{family_id:02d}",
             "rationale": "governed tender family canonical concept", "source_spans": [span], "confidence": 0.9},
            {"node_id": category_id, "node_type": "category", "canonical_label": "tender requirement category",
             "display_name_zh": "招标要求类别", "normalized_value": "tender_requirement_category",
             "rationale": "governed tender taxonomy", "source_spans": [span], "confidence": 0.9},
            {"node_id": risk_id, "node_type": "risk", "canonical_label": "discriminatory requirement risk",
             "display_name_zh": "可能构成排他性要求", "normalized_value": "discriminatory_requirement",
             "rationale": "candidate risk link for reviewer inspection", "source_spans": [span], "confidence": 0.75},
        ])
        base = {"source_spans": [span], "confidence": 0.8, "rationale": "Semantier tender mapper v1"}
        edges.extend([
            {"edge_id": f"edge_{entity_id}_alias_of", "source_node_id": entity_id, "target_node_id": concept_id,
             "relation_type": "alias_of", **base},
            {"edge_id": f"edge_{concept_id}_in_category", "source_node_id": concept_id, "target_node_id": category_id,
             "relation_type": "in_category", **base},
            {"edge_id": f"edge_{concept_id}_creates_risk", "source_node_id": concept_id, "target_node_id": risk_id,
             "relation_type": "creates_risk", **base},
        ])
        rules.append({"rule_id": f"rule_{family_id:02d}", "family_id": family_id,
                      "condition_node_ids": (concept_id,), "exception_node_ids": (),
                      "reason_code": "tender_sensitive_family", "source_spans": [span]})

    for family_id in sorted(family_ids):
        evidence = (family_evidence[family_id],)
        label = next(node["canonical_label"] for node in nodes if node["node_id"] == family_ids[family_id])
        for item_id, layer, display, normalized in (
            (family_ids[family_id], "concept", label, f"tender_family_{family_id:02d}"),
            (f"category_tender_family_{family_id:02d}", "category", "tender requirement category", "tender_requirement_category"),
            (f"risk_discriminatory_requirement_{family_id:02d}", "risk", "discriminatory requirement risk", "discriminatory_requirement"),
        ):
            entities.append({"item_id": item_id, "machine_identifier": item_id, "display_label": display,
                             "layer": layer, "normalized_value": normalized, "confidence": 0.8,
                             "extraction_run_id": extraction_run_id, "mapper_version": MAPPER_VERSION,
                             "evidence": list(evidence)})

    extraction = TenderExtractionResult.model_validate({
        "extraction_run_id": extraction_run_id, "mapper_version": MAPPER_VERSION,
        "provider": provider, "provider_commit": provider_commit,
        "source_hash": parsed_document["document_hash"], "entities": entities,
        "relations": [{"relation_id": edge["edge_id"], "source_item_id": edge["source_node_id"],
                        "target_item_id": edge["target_node_id"], "relation_type": edge["relation_type"],
                        "confidence": edge["confidence"], "extraction_run_id": extraction_run_id,
                        "mapper_version": MAPPER_VERSION, "evidence": entities[0]["evidence"]}
                       for edge in edges if edge["source_node_id"].startswith("alias_") and edge["target_node_id"].startswith("concept_")],
        "constraints": [],
    })
    return {"schema_version": "sensitive_term_candidate_graph.v2", "ontology_version": "tender_risk_ontology.v1",
            "confidence": min((item["confidence"] for item in nodes), default=None), "nodes": _unique(nodes),
            "edges": _unique_edges(edges), "rules": rules, "tender_extraction": extraction.model_dump(mode="json")}


def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({item["node_id"]: item for item in items}.values())


def _unique_edges(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({item["edge_id"]: item for item in items}.values())


__all__ = ["MAPPER_VERSION", "map_semantica_tender_output"]
