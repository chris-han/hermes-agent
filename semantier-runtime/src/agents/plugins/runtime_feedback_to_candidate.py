"""Runtime feedback → candidate delta bridge (R7-01, R7-02).

Tenant/workspace-scoped bridge that converts a runtime finding
disposition / false positive / missed finding / ambiguity into a
candidate delta that the Human Grounding workbench can review.

The bridge is append-oriented: historical run/release/activation/source
bindings remain replayable and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class CandidateDelta:
    candidate_delta_ref: str
    organization_id: str
    workspace_id: str
    tender_run_id: str
    finding_id: str
    feedback_kind: str  # 'accept' | 'reject' | 'false_positive' | 'missed_finding' | 'ambiguity'
    source_finding_ref: str
    created_at: str


def convert_tender_feedback_to_candidate_delta(
    *,
    organization_id: str,
    workspace_id: str,
    tender_run_id: str,
    finding_id: str,
    feedback_kind: str,
    source_finding_ref: str,
    lineage: dict[str, Any] | None = None,
) -> CandidateDelta:
    """Convert one runtime finding disposition into a candidate delta ref.

    Returns a deterministic, tenant-scoped reference that the Ground
    workbench can consume. The original finding binding is preserved
    in the returned payload so audit / replay remain possible.
    """
    if not organization_id:
        raise ValueError("organization_id is required")
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if feedback_kind not in {
        "accept", "reject", "false_positive", "missed_finding", "ambiguity",
    }:
        raise ValueError(f"unsupported feedback_kind: {feedback_kind}")

    payload = {
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "tender_run_id": tender_run_id,
        "finding_id": finding_id,
        "feedback_kind": feedback_kind,
        "source_finding_ref": source_finding_ref,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()

    return CandidateDelta(
        candidate_delta_ref=f"cd/{organization_id}/{workspace_id}/{digest[:24]}",
        organization_id=organization_id,
        workspace_id=workspace_id,
        tender_run_id=tender_run_id,
        finding_id=finding_id,
        feedback_kind=feedback_kind,
        source_finding_ref=source_finding_ref,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = ["CandidateDelta", "convert_tender_feedback_to_candidate_delta"]