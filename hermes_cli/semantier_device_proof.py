"""Managed-device client wire encoding; never enrollment/seat/attestation authority.

Runtime owns validation and authorization. Server/client conformance tests pin
these versioned domain messages; this client has no Runtime package dependency.
"""
from datetime import datetime
import json

_FIELDS = frozenset({'challenge_id', 'nonce', 'user_id', 'organization_id',
                     'workspace_id', 'account_id', 'currency', 'intent',
                     'replaces_device_id', 'expires_at'})


def enrollment_message(context):
    if not isinstance(context, dict) or set(context) != _FIELDS:
        raise ValueError('invalid_enrollment_context')
    for name in _FIELDS - {'replaces_device_id'}:
        if not isinstance(context[name], str) or not 1 <= len(context[name]) <= 256:
            raise ValueError('invalid_enrollment_context')
    replacement = context['replaces_device_id']
    if replacement is not None and (not isinstance(replacement, str) or not 1 <= len(replacement) <= 128):
        raise ValueError('invalid_replacement')
    if context['intent'] not in {'register', 'replace', 'reconnect'}:
        raise ValueError('invalid_enrollment_intent')
    if (context['intent'] == 'replace') != (replacement is not None):
        raise ValueError('invalid_replacement')
    try:
        expiry = datetime.fromisoformat(context['expires_at'])
    except ValueError:
        raise ValueError('invalid_expiry') from None
    if expiry.tzinfo is None or expiry.utcoffset().total_seconds() != 0:
        raise ValueError('utc_expiry_required')
    return b'semantier.managed-device.enrollment.v1\x00' + json.dumps(
        context, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


MANAGED_PERMISSIONS = frozenset({'skills:sync','knowledge:sync','plugin:execute'})


_OPERATION_FIELDS = frozenset({'request_ref','nonce','device_id','user_id','organization_id',
    'workspace_id','account_id','currency','permission','method','path','payload_hash','expires_at'})


def operation_message(context):
    if not isinstance(context,dict) or set(context) != _OPERATION_FIELDS:
        raise ValueError('invalid_operation_context')
    if any(not isinstance(value,str) or not 1 <= len(value) <= 512 for value in context.values()):
        raise ValueError('invalid_operation_context')
    if context['permission'] not in MANAGED_PERMISSIONS or context['method'] not in {'GET','POST'}:
        raise ValueError('invalid_operation_scope')
    if not context['path'].startswith('/api/personal-capabilities/') or '?' in context['path'] or '#' in context['path']:
        raise ValueError('invalid_operation_path')
    import re
    if not re.fullmatch('[a-f0-9]{64}',context['payload_hash']):
        raise ValueError('invalid_payload_hash')
    expiry=datetime.fromisoformat(context['expires_at'])
    if expiry.tzinfo is None or expiry.utcoffset().total_seconds() != 0:
        raise ValueError('utc_expiry_required')
    return b'semantier.managed-device.operation.v1\x00'+json.dumps(
        context,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
