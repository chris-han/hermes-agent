"""Append-only UX evidence for the tender MVL review journey."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import current_workspace_runs_dir


class TenderMVLObservationStore:
    schema = "tender_mvl_review_observation.v1"

    def append(self, observation: dict[str, Any]) -> dict[str, Any]:
        required = {
            "run_id", "finding_id", "started_at", "completed_at", "disposition",
            "interaction_count", "feedback_type", "explanation_sufficient",
            "reviewer_note", "activation_set_snapshot_id", "resolved_activation_set_hash",
        }
        missing = sorted(required - set(observation))
        if missing:
            raise ValueError(f"review observation missing fields: {missing}")
        root = current_workspace_runs_dir()
        if not root:
            raise ValueError("active session runs root is required for MVL observations")
        path = Path(root).resolve() / "tender-mvl" / "review-observations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": self.schema, **observation}
        payload["observation_hash"] = content_hash(payload)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
        return {"artifact_ref": str(path), "artifact_hash": content_hash(payload), "observation": payload}

    def append_review(
        self,
        *,
        run_id: str,
        finding_id: str | None,
        started_at: datetime,
        completed_at: datetime,
        disposition: str,
        interaction_count: int,
        feedback_type: str | None,
        explanation_sufficient: bool,
        reviewer_note: str | None,
        activation_set_snapshot_id: str,
        resolved_activation_set_hash: str,
    ) -> dict[str, Any]:
        def iso(value: datetime) -> str:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("observation timestamps must be timezone-aware")
            return value.astimezone(timezone.utc).isoformat()
        return self.append({
            "run_id": run_id,
            "finding_id": finding_id,
            "started_at": iso(started_at),
            "completed_at": iso(completed_at),
            "disposition": disposition,
            "interaction_count": interaction_count,
            "feedback_type": feedback_type,
            "explanation_sufficient": explanation_sufficient,
            "reviewer_note": reviewer_note,
            "activation_set_snapshot_id": activation_set_snapshot_id,
            "resolved_activation_set_hash": resolved_activation_set_hash,
        })
