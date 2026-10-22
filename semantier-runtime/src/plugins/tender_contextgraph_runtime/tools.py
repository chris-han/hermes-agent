from __future__ import annotations

import json
from typing import Any


def _invoke(operation: str, args: dict[str, Any], **kwargs: Any) -> str:
    binding = kwargs.get("tender_contextgraph_runtime_api")
    if binding is None:
        return json.dumps({"ok": False, "error": {"message": "AUTHENTICATED_RUNTIME_CONTEXT_REQUIRED"}}, sort_keys=True)
    try:
        actor_id = str(kwargs.get("actor_id") or "")
        if operation == "emit_feedback":
            result = binding.emit_feedback(args, actor_id=actor_id)
        else:
            result = getattr(binding, operation)(args)
        return json.dumps({"ok": True, "result": result}, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return json.dumps({"ok": False, "error": {"message": str(exc) or exc.__class__.__name__}}, ensure_ascii=False, sort_keys=True)


def tender_contextgraph_search(args, **kwargs):
    return _invoke("search", args, **kwargs)


def tender_contextgraph_get_node(args, **kwargs):
    return _invoke("get_node", args, **kwargs)


def tender_contextgraph_expand_neighborhood(args, **kwargs):
    return _invoke("expand_neighborhood", args, **kwargs)


def tender_contextgraph_get_path_evidence(args, **kwargs):
    return _invoke("get_path_evidence", args, **kwargs)


def tender_contextgraph_review(args, **kwargs):
    return _invoke("review", args, **kwargs)


def tender_contextgraph_emit_feedback(args, **kwargs):
    return _invoke("emit_feedback", args, **kwargs)
