"""Check complete serialized Messages requests at the existing SDK boundary.

This transport neither selects authority nor routes models. Its caller supplies
an authorized profile, the required material, and the existing artifact writer.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Callable

import httpx

from contracts.runtime_judgment_execution import PROFILE_FIELDS, profile_resolved, messages_texts as _texts


class CompatibilityFailure(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)




class CompleteRequestCompatibilityTransport(httpx.BaseTransport):
    """Count every physical request, without editing or truncating its body."""

    def __init__(self, *, transport: httpx.BaseTransport, profile: dict,
                 required_texts: tuple[str, ...], record: Callable[[dict], None]):
        self._transport = transport
        self._profile = deepcopy(profile)
        self._required_texts = Counter(required_texts)
        self._record = record

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raw = request.read()
        profile = {key: deepcopy(value) for key, value in self._profile.items() if key in PROFILE_FIELDS}
        receipt = dict(schema_version='complete_request_compatibility.v1', profile=profile,
            checked_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            request_url=str(request.url), request_body=raw.decode('utf-8'),
            request_hash='sha256:' + sha256(raw).hexdigest(), input_tokens=None,
            outcome='NO_COMPATIBLE_MODEL', reason=None)

        def fail(reason):
            receipt['reason'] = reason
            self._record(deepcopy(receipt))
            raise CompatibilityFailure(reason)

        if not profile_resolved(self._profile):
            fail('PROFILE_UNRESOLVED')
        endpoint = httpx.URL(profile['base_url'].rstrip('/') + '/v1/messages')
        if endpoint.scheme != 'https' or endpoint.query or request.method != 'POST' or request.url != endpoint:
            fail('UNAUTHORIZED_ENDPOINT')
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            fail('REQUEST_UNRESOLVED')
        if not isinstance(body, dict) or body.get('model') != profile['model']:
            fail('MODEL_BINDING_MISMATCH')
        if type(body.get('max_tokens')) is not int or body['max_tokens'] != profile['required_output_reserve']:
            fail('OUTPUT_RESERVE_MISMATCH')
        if not self._required_texts or not self._required_texts <= _texts(body):
            fail('MATERIAL_BINDING_MISMATCH')
        if profile['required_output_reserve'] > profile['output_limit']:
            fail('OUTPUT_OVERFLOW')
        count_url = endpoint.copy_with(path=endpoint.path + '/count_tokens')
        # The same body, auth and provider headers are counted. No JSON
        # reconstruction, prompt shortening or alternate template is allowed.
        count_request = httpx.Request('POST', count_url, headers=request.headers,
                                     content=raw, extensions=request.extensions)
        try:
            counted = self._transport.handle_request(count_request)
            try:
                count_raw = counted.read()
                count_data = json.loads(count_raw)
                tokens = count_data.get('input_tokens')
                if counted.status_code != 200 or type(tokens) is not int or tokens < 0:
                    raise ValueError('unsupported count')
            finally:
                counted.close()
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            fail('COUNT_UNRESOLVED')
        receipt.update(input_tokens=tokens, count_endpoint=str(count_url),
            counting_basis='provider_count_api', count_response=count_data,
            count_response_body=count_raw.decode('utf-8'),
            count_response_hash='sha256:' + sha256(count_raw).hexdigest())
        if tokens > profile['input_limit']:
            fail('INPUT_OVERFLOW')
        if tokens + profile['required_output_reserve'] > profile['effective_context_limit']:
            fail('CONTEXT_OVERFLOW')
        receipt.update(outcome='COMPATIBLE', reason=None)
        # A failed durable write prevents forwarding, rather than losing lineage.
        self._record(deepcopy(receipt))
        if request.content != raw:
            raise CompatibilityFailure('REQUEST_CHANGED_AFTER_COUNT')
        return self._transport.handle_request(request)

    def close(self):
        self._transport.close()
