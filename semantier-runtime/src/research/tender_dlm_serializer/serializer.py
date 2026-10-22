"""Exact, self-contained decision text sharing; no semantic inference or omission."""
from collections import Counter
from copy import deepcopy
from itertools import groupby
import json
import re

from src.research.tender_small_dlm.context_envelope import canonical, content_hash, expand

VERSION = 'decision_optimized_dlm_view.v2'


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _encode(value, indices, pattern):
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
        if any(k in value for k in ('$text_ref', '$concat', '$object', '$raw_text', '+', '!')):
            return {'$object': [[k, _encode(v, indices, pattern)] for k, v in sorted(value.items())]}
        return {k: _encode(v, indices, pattern) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_encode(v, indices, pattern) for v in value]
    return value


def _pattern(strings):
    return re.compile('|'.join(re.escape(s) for s in sorted(strings, key=lambda s: (-len(s), s)))) if strings else None


def _prepare(payload):
    root = deepcopy(payload)
    table, joined = None, []
    shared_contract_fields = []
    shared_frontier_questions = False
    if isinstance(root, dict) and isinstance(root.get('dlp_model_input'), dict):
        dlp = root['dlp_model_input']
        contract = root.get('decision_contract', {})
        if isinstance(contract, dict):
            for key in sorted(set(dlp) & set(contract)):
                if isinstance(dlp[key], (dict, list)) and canonical(dlp[key]) == canonical(contract[key]):
                    shared_contract_fields.append(key)
                    del dlp[key]
        frontier = root.get('typed_decision_frontier', {})
        if (isinstance(frontier, dict) and 'questions' in frontier
                and 'semantic_predicates' in dlp
                and canonical(dlp['semantic_predicates']) == canonical(frontier['questions'])):
            shared_frontier_questions = True
            del dlp['semantic_predicates']
        rows = dlp.get('source_excerpts')
        if isinstance(rows, list) and rows and all(set(r) == {'ref', 'role', 'text'} for r in rows):
            texts = [r['text'] for r in rows]
            label = '\n'.join(t.strip() for t in texts)
            for i, assertion in enumerate(dlp.get('assertions', [])):
                for field in ('subject_label', 'object_label'):
                    if assertion.get(field) == label:
                        joined.append([i, field])
                        del assertion[field]
            refs = [r['ref'] for r in rows]
            table = {'texts': texts,
                     'role_runs': [[len(list(group)), role] for role, group in groupby(r['role'] for r in rows)]}
            if refs == ['e' + str(i) for i in range(len(rows))]:
                table['ref_sequence'] = ['e', len(rows)]
            else:
                table['refs'] = refs
            del dlp['source_excerpts']
    return {'inherited_root': root, 'evidence_table': table, 'joined_assertion_labels': joined,
            'shared_contract_fields': shared_contract_fields,
            'shared_frontier_questions': shared_frontier_questions}


def _restore(prepared):
    root, table = prepared['inherited_root'], prepared['evidence_table']
    if table is not None:
        texts = table['texts']
        roles = [role for count, role in table['role_runs'] for _ in range(count)]
        refs = table.get('refs')
        if refs is None:
            prefix, count = table['ref_sequence']
            refs = [prefix + str(i) for i in range(count)]
        if not len(texts) == len(roles) == len(refs):
            raise ValueError('EVIDENCE_TABLE_LENGTH_MISMATCH')
        dlp = root['dlp_model_input']
        dlp['source_excerpts'] = [{'ref': ref, 'role': role, 'text': text}
                                 for ref, role, text in zip(refs, roles, texts)]
        label = '\n'.join(t.strip() for t in texts)
        for i, field in prepared['joined_assertion_labels']:
            if field in dlp['assertions'][i]:
                raise ValueError('DUPLICATE_ASSERTION_LABEL')
            dlp['assertions'][i][field] = label
    elif prepared['joined_assertion_labels']:
        raise ValueError('MISSING_EVIDENCE_TABLE')
    for key in prepared['shared_contract_fields']:
        if key in root['dlp_model_input']:
            raise ValueError('DUPLICATE_SHARED_CONTRACT_FIELD')
        root['dlp_model_input'][key] = deepcopy(root['decision_contract'][key])
    if prepared['shared_frontier_questions']:
        if 'semantic_predicates' in root['dlp_model_input']:
            raise ValueError('DUPLICATE_SHARED_FRONTIER')
        root['dlp_model_input']['semantic_predicates'] = deepcopy(root['typed_decision_frontier']['questions'])
    return root


def _compact(value):
    if isinstance(value, dict):
        if set(value) == {'$text_ref'}:
            return '~' + str(value['$text_ref'])
        if set(value) == {'$raw_text'}:
            return '@' + str(value['$raw_text'])
        return {{'$concat': '+', '$object': '!'}.get(k, k): _compact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    if isinstance(value, str) and value.startswith(('~', '@', '!')):
        return '!' + value
    return value


def _uncompact(value):
    if isinstance(value, str):
        if value.startswith('!'):
            return value[1:]
        if value.startswith(('~', '@')):
            number = value[1:]
            if not number.isascii() or not number.isdigit() or str(int(number)) != number:
                raise ValueError('INVALID_COMPACT_REFERENCE')
            return {'$text_ref' if value[0] == '~' else '$raw_text': int(number)}
        return value
    if isinstance(value, dict):
        return {{'+': '$concat', '!': '$object'}.get(k, k): _uncompact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_uncompact(v) for v in value]
    return value


def serialize(payload):
    # Validate the JSON domain and preserve numeric/string types exactly.
    canonical(payload)
    payload = _prepare(payload)
    counts = Counter(_strings(payload))
    selected = {s for s, count in counts.items() if len(s) >= 24 and count > 1}
    hashes = Counter(fragment for value in _strings(payload)
                     for fragment in re.findall(r'[0-9a-f]{64}', value))
    selected.update(value for value, count in hashes.items() if count > 1)
    strings = sorted(selected, key=lambda s: (len(s), s))
    indices = {s: i for i, s in enumerate(strings)}
    records = []
    for i, value in enumerate(strings):
        # Earlier records are strictly bounded; there are no cycles or forward refs.
        prior = strings[:i]
        encoded = _encode(value, {s: indices[s] for s in prior}, _pattern(prior))
        records.append(encoded)
    root = _encode(payload, indices, _pattern(strings))
    # Raw records avoid repeated JSON escapes in exact source text. Length framing
    # makes embedded delimiters harmless and is deterministic over Unicode codepoints.
    literal_texts = []
    def literals(value):
        if isinstance(value, str) and len(value) >= 64:
            index = len(literal_texts)
            literal_texts.append(value)
            return {'$raw_text': index}
        if isinstance(value, list):
            return [literals(v) for v in value]
        if isinstance(value, dict):
            # $raw_text occurs only in this transport tree; input reserved keys
            # are escaped by _encode as ordinary $object pairs.
            return {k: literals(v) for k, v in value.items()}
        return value
    # Escape $raw_text in input before codec to avoid collisions at this layer.
    header = {'view_version': VERSION, 'strings': literals(records), 'root': literals(root)}
    return canonical(_compact(header)) + '\n' + ''.join(str(len(s)) + '\n' + s for s in literal_texts)


def deserialize(body):
    try:
        header_text, rest = body.split('\n', 1)
        header = _uncompact(json.loads(header_text))
        if set(header) != {'view_version', 'strings', 'root'} or header['view_version'] != VERSION:
            raise ValueError('INVALID_VIEW_HEADER')
        raw = []
        while rest:
            size_text, rest = rest.split('\n', 1)
            if not size_text.isdigit() or str(int(size_text)) != size_text:
                raise ValueError('INVALID_TEXT_LENGTH')
            size = int(size_text)
            if len(rest) < size:
                raise ValueError('TRUNCATED_TEXT')
            raw.append(rest[:size])
            rest = rest[size:]
        used = []
        def restore(value):
            if isinstance(value, dict):
                if set(value) == {'$raw_text'}:
                    index = value['$raw_text']
                    if type(index) is not int or index != len(used) or index >= len(raw):
                        raise ValueError('INVALID_RAW_REFERENCE')
                    used.append(index)
                    return raw[index]
                return {k: restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v) for v in value]
            return value
        # Match allocation order before canonical key sorting.
        records = restore(header['strings'])
        root = restore(header['root'])
        if len(used) != len(raw):
            raise ValueError('UNUSED_RAW_TEXT')
        strings = []
        for record in records:
            value = expand({'strings': strings, 'root': record})
            if not isinstance(value, str):
                raise ValueError('NON_TEXT_RECORD')
            strings.append(value)
        result = _restore(expand({'strings': strings, 'root': root}))
        if serialize(result) != body:
            raise ValueError('NONCANONICAL_VIEW')
        return result
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError('INVALID_VIEW') from exc


def semantic_items(value, path=''):
    """Path identity preserves all bindings, multiplicity, order and empty containers."""
    rows = {path: {'type': type(value).__name__, 'value': value if not isinstance(value, (dict, list)) else len(value)}}
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            rows.update(semantic_items(child, path + '/' + key.replace('~', '~0').replace('/', '~1')))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            rows.update(semantic_items(child, path + '/' + str(i)))
    return rows


def verify_closure(source, body):
    original = semantic_items(source)
    reconstructed = semantic_items(deserialize(body))
    if canonical(original) != canonical(reconstructed):
        raise ValueError('SEMANTIC_CLOSURE_LOSS')
    hashes = {k: content_hash(v) for k, v in original.items()}
    return {'disposition': 'PASS', 'canonical_semantic_items': hashes,
            'source_semantic_hash': content_hash(original),
            'reconstructed_semantic_hash': content_hash(reconstructed),
            'item_count': len(original), 'omitted_items': []}
