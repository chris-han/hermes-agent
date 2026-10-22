"""Append-only workspace artifact store for SchemaLearningFeedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts.schema_learning_feedback import SchemaLearningFeedback
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2


class SchemaLearningFeedbackStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.root = (
            Path(workspace_root).resolve()
            / "workspaces"
            / workspace_id
            / "artifacts"
            / "schema-learning-feedback"
        )

    def put(self, feedback: SchemaLearningFeedback | SchemaLearningFeedbackV2) -> dict:
        target = self.root / feedback.feedback_id / f"{feedback.schema_version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            feedback.as_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise ValueError("SCHEMA_LEARNING_FEEDBACK_ARTIFACT_CONFLICT")
        target.write_text(encoded, encoding="utf-8")
        return feedback.as_dict()

    def get(self, feedback_id: str) -> dict[str, Any]:
        if not feedback_id.startswith("schema-feedback-"):
            raise ValueError("invalid schema learning feedback id")
        matches = sorted(
            (self.root / feedback_id).glob("schema_learning_feedback.v*.json")
        )
        if len(matches) != 1:
            raise ValueError(f"schema learning feedback not found: {feedback_id}")
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        if payload.get("schema_version") == "schema_learning_feedback.v2":
            return SchemaLearningFeedbackV2.from_dict(payload).as_dict()
        return SchemaLearningFeedback.from_dict(payload).as_dict()


__all__ = ["SchemaLearningFeedbackStore"]
