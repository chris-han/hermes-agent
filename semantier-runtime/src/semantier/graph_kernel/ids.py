from __future__ import annotations

from typing import Any

from eos.artifact_hashing import content_hash


def node_id(*, organization_id: str, node_type: str, canonical_key: str) -> str:
    return "graph_node_" + content_hash(
        {
            "organization_id": organization_id,
            "node_type": node_type,
            "canonical_key": canonical_key,
        }
    )[:32]


def assertion_id(
    *,
    organization_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
    source_ref: str,
    source_anchor: str | None,
) -> str:
    return "graph_assertion_" + content_hash(
        {
            "organization_id": organization_id,
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "source_ref": source_ref,
            "source_anchor": source_anchor,
        }
    )[:32]


def stable_hash(payload: dict[str, Any]) -> str:
    return content_hash(payload)

