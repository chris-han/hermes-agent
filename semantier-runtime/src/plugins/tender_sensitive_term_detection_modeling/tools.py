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


def compile_sensitive_lexicon(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.compile_sensitive_lexicon, args, kwargs)


def review_sensitive_lexicon_candidates(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.review_sensitive_lexicon_candidates, args, kwargs)


def label_tender_document(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.label_tender_document, args, kwargs)


def generate_sensitive_labeling_report(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.generate_sensitive_labeling_report, args, kwargs)


def generate_labeled_docx(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.generate_labeled_docx, args, kwargs)


def verify_sensitive_labeling_replay(args: dict[str, Any], **kwargs: Any) -> str:
    return _json_response(service.verify_sensitive_labeling_replay, args, kwargs)
