"""Deterministic KGL source registration helpers."""

from __future__ import annotations

from typing import Any, Dict

from eos.authority_validation import (
    validate_anchor_metadata,
    validate_effectivity_window,
    validate_source_provenance,
)


_REQUIRED_FIELDS = (
    "source_uri",
    "jurisdiction",
    "source_version",
    "extraction_method",
    "curator",
)


def normalize_source_registration(payload: dict | None, *, strict_authority: bool = False) -> Dict[str, Any]:
    source = dict(payload or {})
    missing = [field for field in _REQUIRED_FIELDS if not source.get(field)]
    if missing:
        raise ValueError(",".join(missing))

    anchors = source.get("anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("anchors")
    if strict_authority:
        effective_from, effective_to = validate_effectivity_window(
            source,
            prefix="legal source registration",
        )
        validate_source_provenance(source, prefix="legal source registration")
        normalized_anchors = validate_anchor_metadata(
            anchors,
            prefix="legal source registration",
        )
    else:
        effective_from = source.get("effective_from")
        effective_to = source.get("effective_to")
        normalized_anchors = []
        for anchor in anchors:
            if not isinstance(anchor, dict) or not anchor.get("anchor_id"):
                raise ValueError("anchor_id")
            normalized_anchors.append(dict(anchor))

    ambiguities = source.get("ambiguities", [])
    if not isinstance(ambiguities, list):
        raise ValueError("ambiguities")

    return {
        "source_uri": str(source["source_uri"]),
        "jurisdiction": str(source["jurisdiction"]),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "source_version": str(source["source_version"]),
        "anchors": sorted(normalized_anchors, key=lambda item: str(item["anchor_id"])),
        "extraction_method": str(source["extraction_method"]),
        "curator": str(source["curator"]),
        "ambiguities": list(ambiguities),
    }
