"""Teaching intent shape; validation grants no execution or identity authority."""
from __future__ import annotations

import json
from typing import Any

_FIELDS = frozenset({'intent', 'participants', 'time_window', 'duration_minutes',
                     'constraints', 'missing_information', 'risk_level'})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def parse_meeting_intent(raw: str) -> dict[str, Any]:
    """Reject invalid model output before a caller can dispatch downstream tools."""
    if not isinstance(raw, str) or len(raw) > 65_536:
        raise ValueError('INVALID_MEETING_INTENT')
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError('INVALID_MEETING_JSON') from exc
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError('INVALID_MEETING_FIELDS')
    if value['intent'] not in ('create_meeting', 'update_meeting', 'clarify'):
        raise ValueError('INVALID_MEETING_ACTION')
    if value['risk_level'] not in ('low', 'medium', 'high'):
        raise ValueError('INVALID_MEETING_RISK')
    for key in ('participants', 'constraints', 'missing_information'):
        items = value[key]
        if not isinstance(items, list) or len(items) > 100 or any(
            not isinstance(item, str) or not item.strip() or len(item) > 1024
            for item in items
        ):
            raise ValueError('INVALID_MEETING_LIST')
    if not isinstance(value['time_window'], str) or len(value['time_window']) > 1024:
        raise ValueError('INVALID_MEETING_WINDOW')
    duration = value['duration_minutes']
    if type(duration) is not int or not 1 <= duration <= 480:
        raise ValueError('INVALID_MEETING_DURATION')
    if value['intent'] == 'create_meeting' and (
        not value['participants'] or not value['time_window'].strip()
        or value['missing_information']
    ):
        raise ValueError('INCOMPLETE_MEETING_REQUIRES_CLARIFICATION')
    if value['intent'] == 'clarify' and not value['missing_information']:
        raise ValueError('CLARIFICATION_REASON_REQUIRED')
    return value
