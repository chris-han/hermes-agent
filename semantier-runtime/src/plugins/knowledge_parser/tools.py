from __future__ import annotations

import json
from typing import Any

from . import service


def _json_response(fn: Any, args: dict[str, Any], kwargs: dict[str, Any]) -> str:
    try:
        return json.dumps(fn(args=args, **kwargs), ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return json.dumps(
            {"status": "error", "error_code": str(exc) or exc.__class__.__name__},
            ensure_ascii=False,
            sort_keys=True,
        )


def parse_knowledge_source(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.parse_knowledge_source, args, kwargs)


def get_canonical_classification_input(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.get_canonical_classification_input, args, kwargs)


def get_knowledge_parse_run(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.get_knowledge_parse_run, args, kwargs)


def verify_knowledge_parse_replay(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.verify_knowledge_parse_replay, args, kwargs)
