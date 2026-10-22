"""Few-shot relationship proposal contract and deterministic validation."""

from __future__ import annotations

import json
from typing import Any, Callable


ALLOWED_RELATION_TYPES = frozenset({
    "alias_of", "related_to", "modifier_of", "broader_than", "narrower_than",
    "in_category", "conditioned_by", "exception_to", "supports_rule",
    "imposes_requirement", "applies_to", "has_attribute", "has_threshold",
    "creates_risk", "supported_by", "requires", "prohibits", "greater_than",
    "greater_than_or_equal", "less_than", "less_than_or_equal",
})

RELATION_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mentions", "relations"],
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mention_id", "element_anchor_id", "text", "start_offset", "end_offset", "semantic_type", "confidence"],
                "properties": {
                    "mention_id": {"type": "string"},
                    "element_anchor_id": {"type": "string"},
                    "text": {"type": "string"},
                    "start_offset": {"type": "integer", "minimum": 0},
                    "end_offset": {"type": "integer", "minimum": 1},
                    "semantic_type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["relation_id", "source_mention_id", "target_mention_id", "relation_type", "confidence", "rationale"],
                "properties": {
                    "relation_id": {"type": "string"},
                    "source_mention_id": {"type": "string"},
                    "target_mention_id": {"type": "string"},
                    "relation_type": {"type": "string", "enum": sorted(ALLOWED_RELATION_TYPES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 240},
                },
            },
        },
    },
}


def build_relationship_extraction_input(
    elements: list[dict[str, Any]],
    *,
    negative_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize only canonical paragraph identity and text for one run."""
    payload: dict[str, Any] = {
        "paragraphs": [
            {
                "element_anchor_id": str(item["anchor"]["anchor_id"]),
                "text": str(item.get("text") or ""),
            }
            for item in elements
        ]
    }
    if negative_examples:
        allowed = (
            "feedback_id", "event_hash", "reason_code", "concept_label",
            "matched_text", "context_text",
        )
        payload["negative_evidence_examples"] = [
            {key: str(item.get(key) or "") for key in allowed}
            for item in negative_examples[:50]
        ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def validate_relationship_output(
    payload: Any,
    elements: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Reject proposals that cannot be grounded exactly in canonical text."""
    if not isinstance(payload, dict):
        raise ValueError("relationship proposal must be a JSON object")
    by_anchor = {
        str(item["anchor"]["anchor_id"]): str(item.get("text") or "")
        for item in elements
    }
    mentions = payload.get("mentions")
    relations = payload.get("relations")
    if not isinstance(mentions, list) or not isinstance(relations, list):
        raise ValueError("relationship proposal requires mentions and relations arrays")

    valid_mentions: list[dict[str, Any]] = []
    mention_ids: set[str] = set()
    for item in mentions:
        if not isinstance(item, dict):
            raise ValueError("mention must be an object")
        mention_id = str(item.get("mention_id") or "")
        anchor_id = str(item.get("element_anchor_id") or "")
        text = str(item.get("text") or "")
        start = item.get("start_offset")
        end = item.get("end_offset")
        if not mention_id or mention_id in mention_ids:
            raise ValueError("mention IDs must be non-empty and unique")
        if anchor_id not in by_anchor or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("mention references an invalid source span")
        source_text = by_anchor[anchor_id]
        if start < 0 or end <= start or end > len(source_text) or source_text[start:end] != text:
            raise ValueError("mention text does not exactly match its source span")
        confidence = float(item.get("confidence", -1))
        if not 0 <= confidence <= 1:
            raise ValueError("mention confidence must be between 0 and 1")
        mention_ids.add(mention_id)
        valid_mentions.append({
            "mention_id": mention_id,
            "element_anchor_id": anchor_id,
            "text": text,
            "start_offset": start,
            "end_offset": end,
            "semantic_type": str(item.get("semantic_type") or "concept"),
            "confidence": confidence,
        })

    valid_relations: list[dict[str, Any]] = []
    relation_ids: set[str] = set()
    for item in relations:
        if not isinstance(item, dict):
            raise ValueError("relation must be an object")
        relation_id = str(item.get("relation_id") or "")
        source_id = str(item.get("source_mention_id") or "")
        target_id = str(item.get("target_mention_id") or "")
        relation_type = str(item.get("relation_type") or "")
        confidence = float(item.get("confidence", -1))
        if not relation_id or relation_id in relation_ids:
            raise ValueError("relation IDs must be non-empty and unique")
        if source_id not in mention_ids or target_id not in mention_ids or source_id == target_id:
            raise ValueError("relation references invalid mention IDs")
        if relation_type not in ALLOWED_RELATION_TYPES:
            raise ValueError(f"unsupported relation type: {relation_type}")
        if not 0 <= confidence <= 1:
            raise ValueError("relation confidence must be between 0 and 1")
        relation_ids.add(relation_id)
        valid_relations.append({
            "relation_id": relation_id,
            "source_mention_id": source_id,
            "target_mention_id": target_id,
            "relation_type": relation_type,
            "confidence": confidence,
            "rationale": str(item.get("rationale") or "")[:240],
        })
    return {"mentions": valid_mentions, "relations": valid_relations}


RelationshipProposer = Callable[[list[dict[str, Any]]], dict[str, Any]]
