from __future__ import annotations

import json
import logging
from typing import Any

from agents.corpus_graph_schema_tool_api import authenticated_tool_api

from .schemas import validate


logger = logging.getLogger(__name__)


def _error(code: str, message: str) -> str:
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _invoke(operation: str, args: dict[str, Any]) -> str:
    try:
        validate(operation, args)
    except ValueError as exc:
        # These messages are produced by our closed schema validator and name
        # fields/constraints only; they never interpolate caller values.
        return _error("INVALID_REQUEST", str(exc))
    try:
        result = authenticated_tool_api().invoke(operation=operation, payload=args)
    except ValueError as exc:
        if str(exc) == "AUTHENTICATED_REQUEST_CONTEXT_REQUIRED":
            return _error(
                "AUTHENTICATED_REQUEST_CONTEXT_REQUIRED",
                "AUTHENTICATED_REQUEST_CONTEXT_REQUIRED",
            )
        # Core services may carry parser, database, or filesystem details in
        # exception text. The Chat boundary deliberately does not expose it.
        return _error(
            "REQUEST_REJECTED",
            "The corpus schema request could not be completed.",
        )
    except Exception as exc:
        # Do not attach exc_info or exception text: either can contain source
        # paths, SQL, parser excerpts, or other tenant-sensitive details.
        logger.error(
            "Corpus schema tool operation failed: operation=%s exception_type=%s",
            operation,
            type(exc).__name__,
        )
        return _error("INTERNAL_ERROR", "The corpus schema tool failed internally.")
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False, sort_keys=True)


def corpus_graph_schema_profiles(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("profiles", args)


def corpus_graph_schema_induce(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("induce", args)


def corpus_graph_schema_list(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("list", args)


def corpus_graph_schema_get(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("get", args)


def corpus_graph_schema_intent_get(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("intent_get", args)


def corpus_graph_schema_intent_review(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("intent_review", args)


def corpus_graph_schema_review(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("review", args)


def corpus_graph_schema_sample(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("sample", args)


def corpus_graph_schema_feedback_retest(args: dict[str, Any], **_kwargs: Any) -> str:
    return _invoke("feedback_retest", args)
