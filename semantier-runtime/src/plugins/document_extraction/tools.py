from __future__ import annotations

import json
from typing import Any


def _error(exc: Exception) -> str:
    return json.dumps(
        {"status": "error", "error_code": str(exc) or exc.__class__.__name__},
        ensure_ascii=False,
        sort_keys=True,
    )


def extract_document_content(args: dict[str, Any], **_kw: Any) -> str:
    try:
        from .service import extract_live, extract_replay

        mode = str(args.get("mode") or "live")
        result = extract_replay(args) if mode == "replay" else extract_live(args)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return _error(exc)


def document_text_view(result: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Page text for the model; complete geometry stays in the pinned artifact."""
    offset = max(0, int(args.get('offset', 0)))
    limit = min(6000, max(1, int(args.get('limit', 6000))))
    groups: dict[Any, list[str]] = {}
    document = result.get('document') or {}
    for block in document.get('blocks', []):
        if block.get('text'):
            groups.setdefault(block.get('page', block.get('page_number')), []).append(block['text'])
    for table in document.get('tables', []):
        for row in table.get('rows', []):
            text = ' | '.join(str(cell.get('text', '')) for cell in row.get('cells', []))
            if text:
                groups.setdefault(table.get('page'), []).append(text)
    cursor = 0
    pages = []
    for page, lines in groups.items():
        text = '\n'.join(lines)
        if cursor:
            text = '\n' + text
        start, end = max(0, offset-cursor), min(len(text), offset+limit-cursor)
        if end > start:
            pages.append({'page': page, 'text': text[start:end]})
        cursor += len(text)
    return {key: value for key, value in result.items() if key != 'document'} | {
        'response_view': 'text',
        'document_view': {'schema': 'document_text_view.v1', 'pages': pages,
                          'offset': offset, 'total_characters': cursor,
                          'next_offset': offset+limit if offset+limit < cursor else None},
    }


def extract_document_content_tool(args: dict[str, Any], **kwargs: Any) -> str:
    """Registered model surface defaults to bounded text; Python API stays full."""
    result = json.loads(extract_document_content(args, **kwargs))
    if result.get('status') == 'completed':
        try:
            result = document_text_view(result, args)
        except (TypeError, ValueError) as exc:
            return _error(exc)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
