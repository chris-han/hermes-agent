from __future__ import annotations

from typing import Any

from agents.corpus_graph_schema_capability_factory import (
    build_corpus_graph_schema_capability,
)
from agents.governed_tool_context import resolve_authenticated_tool_context


class CorpusGraphSchemaToolAPI:
    """Authenticated Chat adapter over the Web-shared capability facade."""

    def __init__(self, *, context: Any) -> None:
        self._context = context
        authority = {
            "workspace_id": getattr(context, "workspace_id", None),
            "user_id": getattr(context, "user_id", None),
            "organization_id": getattr(context, "organization_id", None),
            "workspace_root": getattr(context, "workspace_root", None),
        }
        if not getattr(context, "authenticated", False) or any(
            value is None or not str(value).strip() for value in authority.values()
        ):
            raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
        self._capability = build_corpus_graph_schema_capability(
            workspace_root=authority["workspace_root"],
            workspace_id=str(authority["workspace_id"]),
            user_id=str(authority["user_id"]),
            organization_id=str(authority["organization_id"]),
        )

    def invoke(self, *, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "profiles":
            return self._capability.profiles()
        if operation == "induce":
            return self._capability.induce(
                lineage_id=payload["lineage_id"],
                relative_path=payload["source"],
                profile_id=payload["profile_id"],
                intended_use=payload["intended_use"],
                competency_questions=payload["competency_questions"],
            )
        if operation == "list":
            return self._capability.list(limit=payload.get("limit", 20))
        if operation == "get":
            return self._capability.get(payload["decision_id"])
        if operation == "intent_get":
            return self._capability.get_graph_intent(
                payload["graph_intent_ref"],
                graph_intent_hash=payload["graph_intent_hash"],
            )
        if operation == "intent_review":
            return self._capability.review_graph_intent(
                payload["graph_intent_ref"],
                graph_intent_hash=payload["graph_intent_hash"],
                channel="chat",
                state=payload["state"],
                operating_mode="assisted",
                relation_edit=payload.get("relation_edit"),
                selected_question_ids=payload["selected_question_ids"],
            )
        if operation == "review":
            return self._capability.review(
                payload["decision_id"],
                state=payload["state"],
                note=payload.get("note", ""),
                edited_candidate=payload.get("edited_candidate"),
            )
        if operation == "sample":
            if payload["action"] == "execute":
                return self._capability.execute_sample(payload["decision_id"])
            return self._capability.replay_sample(payload["sample_id"])
        if operation == "feedback_retest":
            return self._capability.feedback_retest(
                payload["correction_decision_id"],
                baseline_decision_id=payload["baseline_decision_id"],
                requested_relation_type=payload["requested_relation_type"],
            )
        raise ValueError(f"CORPUS_GRAPH_SCHEMA_OPERATION_UNSUPPORTED:{operation}")


def authenticated_tool_api() -> CorpusGraphSchemaToolAPI:
    context = resolve_authenticated_tool_context()
    if context is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    return CorpusGraphSchemaToolAPI(context=context)
