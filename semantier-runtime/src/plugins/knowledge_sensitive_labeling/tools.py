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


def classify_knowledge_input(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.classify_knowledge_input, args, kwargs)


def get_classification_evidence(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.get_classification_evidence, args, kwargs)


def generate_knowledge_labeling_report(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.generate_knowledge_labeling_report, args, kwargs)


def generate_knowledge_labeling_derivative(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.generate_knowledge_labeling_derivative, args, kwargs)


def verify_knowledge_labeling_replay(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.verify_knowledge_labeling_replay, args, kwargs)
