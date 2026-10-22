"""Lossless research representation: share exact text, never infer materiality."""
from collections import Counter
from hashlib import sha256
import json
import re


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(value):
    return 'sha256:' + sha256(canonical(value).encode()).hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def compact(payload):
    counts = Counter(_strings(payload))
    # Exact source excerpts are reusable inside denormalized assertion labels.
    excerpts = payload.get('dlp_model_input', payload).get('source_excerpts', [])
    selected = {r['text'] for r in excerpts if len(r.get('text', '')) >= 128}
    selected.update(s for s, count in counts.items() if len(s) >= 64 and count > 1)
    strings = sorted(selected, key=lambda s: (-len(s), s))
    indices = {s: i for i, s in enumerate(strings)}
    pattern = re.compile('|'.join(re.escape(s) for s in strings)) if strings else None

    def encode(value):
        if isinstance(value, str) and pattern:
            if value in indices:
                return {'$text_ref': indices[value]}
            parts, start = [], 0
            for match in pattern.finditer(value):
                if match.start() > start:
                    parts.append(value[start:match.start()])
                parts.append({'$text_ref': indices[match.group()]})
                start = match.end()
            if not parts:
                return value
            if start < len(value):
                parts.append(value[start:])
            return {'$concat': parts}
        if isinstance(value, dict):
            if any(k in value for k in ('$text_ref', '$concat', '$object')):
                return {'$object': [[k, encode(v)] for k, v in value.items()]}
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value

    result = {'strings': strings, 'root': encode(payload)}
    if expand(result) != payload:
        raise ValueError('LOSSLESS_RECONSTRUCTION_FAILED')
    return result


def expand(compacted):
    strings = compacted['strings']

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'$text_ref'}:
                index = value['$text_ref']
                if type(index) is not int or not 0 <= index < len(strings):
                    raise ValueError('INVALID_TEXT_REFERENCE')
                return strings[index]
            if set(value) == {'$concat'}:
                return ''.join(decode(v) for v in value['$concat'])
            if set(value) == {'$object'}:
                return {k: decode(v) for k, v in value['$object']}
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(compacted['root'])


def envelope(payload, *, budget, count_tokens):
    packed = compact(payload)
    serialized = canonical(packed)
    count = count_tokens(serialized)
    return {'payload': packed, 'token_count': count,
            'disposition': 'SUFFICIENT' if count <= budget else 'CONTEXT_ENVELOPE_INSUFFICIENT',
            'source_payload_hash': content_hash(payload), 'content_hash': content_hash(packed)}
