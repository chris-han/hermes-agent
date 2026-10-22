"""Deterministic provider view; full source/evidence remain in proposal receipts.

Only repeated location/selector envelopes and parser/source-map metadata leave
this view. All canonical blocks, text, structure, citations and schema semantics
remain. This view has no source authority and is never used as CanonicalSourceIR.
"""
import json
from hashlib import sha256
from pathlib import Path
from eos.artifact_hashing import canonical_json


def _evidence_view(value):
    if isinstance(value, list):
        return [_evidence_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get('schemaVersion') == 'semantier.evidence_ref.v1':
        return {**{key: value[key] for key in ('schemaVersion', 'evidenceRef', 'sourceIdentityRef',
                                              'selectorHash', 'evidenceRole')},
                'selector': {'exactQuote': value.get('selector', {}).get('exactQuote')},
                'provenance': {'source_block_id': value['provenance']['source_block_id']}}
    return {key: _evidence_view(item) for key, item in value.items()}


def provider_input_text(full_input_text: str) -> str:
    request = json.loads(full_input_text)
    source = request.get('source_context', request)
    document = source.get('canonical_source')
    if document is not None:
        # These are provenance envelopes, never the original text or source blocks.
        document.pop('sourceMap', None)
        document.pop('parserTrace', None)
    result = _evidence_view(request)
    return canonical_json(result)


def verify_provider_input(receipt) -> None:
    actual = receipt.provider_input_text
    if actual is not None and actual != provider_input_text(receipt.input_text):
        raise ValueError('normative provider input differs from retained full source view')


PROMPT_VIEW_HASH = sha256(Path(__file__).read_bytes()).hexdigest()
