"""Shared fail-closed validators for governed authority inputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_aware_iso(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def validate_effectivity_window(
    payload: dict[str, Any],
    *,
    prefix: str,
) -> tuple[str, str | None]:
    effective_from = str(payload.get("effective_from") or "").strip()
    if not effective_from:
        raise ValueError(f"{prefix} effective_from is required")
    from_dt = parse_aware_iso(effective_from, field=f"{prefix} effective_from")
    raw_effective_to = payload.get("effective_to")
    effective_to = None if raw_effective_to is None else str(raw_effective_to).strip()
    if effective_to:
        to_dt = parse_aware_iso(effective_to, field=f"{prefix} effective_to")
        if to_dt <= from_dt:
            raise ValueError(f"{prefix} effective_to must be after effective_from")
    return effective_from, effective_to


def validate_source_provenance(
    payload: dict[str, Any],
    *,
    prefix: str,
    require_dual_control: bool = True,
) -> dict[str, Any]:
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{prefix} source_provenance is required")
    required = [
        "source_type",
        "source_ref",
        "prepared_by",
        "approved_by",
        "approved_at",
    ]
    missing = [key for key in required if not str(provenance.get(key) or "").strip()]
    if missing:
        raise ValueError(f"{prefix} source_provenance missing: {', '.join(missing)}")
    parse_aware_iso(str(provenance["approved_at"]), field=f"{prefix} approved_at")
    if require_dual_control and provenance["prepared_by"] == provenance["approved_by"]:
        raise ValueError(f"{prefix} source_provenance requires separate approval")
    return provenance


def validate_anchor_metadata(anchors: Any, *, prefix: str) -> list[dict[str, Any]]:
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"{prefix} anchors are required")
    normalized: list[dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError(f"{prefix} anchor must be an object")
        if not str(anchor.get("anchor_id") or "").strip():
            raise ValueError(f"{prefix} anchor_id is required")
        if not str(anchor.get("local_hash") or "").strip():
            raise ValueError(f"{prefix} anchor local_hash is required")
        normalized.append(dict(anchor))
    return normalized
