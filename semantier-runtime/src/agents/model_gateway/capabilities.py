"""Sanitized Semantier Cloud Provider profiles. Metadata never grants authority."""
from datetime import datetime, timezone
import json
from fastapi import HTTPException
from contracts.model_gateway import canonical_hash

CAPABILITIES = {'tools':'supports_function_calling', 'vision':'supports_vision',
    'reasoning':'supports_reasoning', 'stream':'supports_streaming',
    'structured_output':'supports_response_schema'}
LIMITS = ('context_window', 'max_output_tokens')
OVERRIDES = frozenset((*LIMITS, *CAPABILITIES, 'reasoning_controls'))


def validate_overrides(value):
    if value is None:
        return
    if not isinstance(value, dict) or set(value)-OVERRIDES:
        raise ValueError('unsupported capability profile fields')
    for key, item in value.items():
        if item is None:
            continue
        if key == 'reasoning_controls':
            if not isinstance(item, list) or any(v != 'enable_thinking' for v in item) or len(item) > 1:
                raise ValueError('unsupported declared reasoning controls')
        elif key in LIMITS:
            if type(item) is not int or not 1 <= item <= 10000000:
                raise ValueError('positive bounded capability limit required')
        elif type(item) is not bool:
            raise ValueError('capabilities must be boolean or null')


def _positive(value):
    return value if type(value) is int and value > 0 else None


def project_profile(deployment, native):
    info = native.get('model_info') or {}
    override = deployment.capability_profile or {}
    sources = {}
    limits = {}
    for key, native_keys in {'context_window':('context_window','max_input_tokens'),
                             'max_output_tokens':('max_output_tokens',)}.items():
        candidates = [_positive(info.get(k)) for k in native_keys]
        candidates.append(_positive(override.get(key)))
        limits[key] = min((v for v in candidates if v is not None), default=None)
        sources[key] = ('deployment+verified_native' if _positive(override.get(key)) and any(_positive(info.get(k)) for k in native_keys)
                        else 'deployment' if _positive(override.get(key)) else 'verified_native' if limits[key] else 'unknown')
    if limits['max_output_tokens'] is not None:
        limits['max_output_tokens'] = min(limits['max_output_tokens'],32768)
    caps = {}
    for key, native_key in CAPABILITIES.items():
        value = info.get(native_key)
        if key == 'stream' and value is None and 'stream' in (info.get('supported_openai_params') or []):
            value = True
        value = value if type(value) is bool else None
        approved = override.get(key)
        # The existing deployment capability tuple is an upper bound, never a guessed native ability.
        if key not in deployment.capabilities or approved is False or value is False:
            caps[key] = False
        else:
            caps[key] = approved if type(approved) is bool else value
        sources[key] = 'deployment' if key not in deployment.capabilities or type(approved) is bool else 'verified_native' if value is not None else 'unknown'
    # Optional capabilities not represented by legacy tuples remain unknown, rather than false.
    for key in ('vision','reasoning','structured_output'):
        if key not in deployment.capabilities and key not in override:
            caps[key] = None
            sources[key] = 'unknown'
    core = {'id':deployment.alias, 'label':deployment.alias, 'provider_display':deployment.provider,
        'transport':'semantier-cloud', 'protocol':deployment.protocol,
        'deployment_revision':deployment.revision, **limits, 'capabilities':caps,
        'request_parameters':{'output':'max_tokens', 'reasoning_controls':list(override.get('reasoning_controls') or [])}, 'sources':sources}
    revision = canonical_hash({'profile':core,'binding':deployment.config_fingerprint})
    return {**core, 'account_id':deployment.account_id,'currency':deployment.currency,'funding':deployment.funding, 'revision':revision, 'resolved_at':datetime.now(timezone.utc).isoformat(),
        'access':'authorized', 'configured':True, 'readiness':'checked_at_dispatch',
        'unknown_reason':'metadata_not_declared' if any(v is None for v in (*limits.values(),*caps.values())) else None}


def validate_profile_request(body, profile, expected_revision=None):
    if expected_revision is not None and expected_revision != profile['revision']:
        raise HTTPException(409, 'cloud_profile_revision_changed; resolve before a new request')
    strict = expected_revision is not None
    cap = body.get('max_completion_tokens', body.get('max_tokens'))
    output = profile['max_output_tokens']
    if output and cap > output:
        raise HTTPException(422, 'output exceeds deployment profile')
    if strict and (output is None or profile['context_window'] is None):
        raise HTTPException(422, 'cloud context/output limits unknown; approved profile required')
    requested = {'tools':bool(body.get('tools') or body.get('tool_choice') or body.get('parallel_tool_calls')),
        'stream':body.get('stream',False), 'reasoning':any(k in body for k in ('reasoning_effort','enable_thinking')),
        'structured_output':'response_format' in body,
        'vision':any(isinstance(m.get('content'),list) and any(isinstance(c,dict) and c.get('type') in {'image_url','input_image'} for c in m['content']) for m in body['messages'] if isinstance(m,dict))}
    for key, used in requested.items():
        state = profile['capabilities'][key]
        if used and (state is False or (strict and state is not True)):
            raise HTTPException(422, 'unsupported or unknown capability: '+key)
    if requested['reasoning'] and strict:
        declared = profile['request_parameters']['reasoning_controls']
        if any(k in body and k not in declared for k in ('reasoning_effort','enable_thinking')):
            raise HTTPException(422, 'reasoning controls not declared for this profile')
    context = profile['context_window']
    if context:
        # Conservative UTF-8 byte upper bound, including tools, framing and output reserve.
        prompt = {k:body[k] for k in ('messages','tools','response_format') if k in body}
        upper_bound = len(json.dumps(prompt,ensure_ascii=False).encode()) + 64*len(body['messages']) + cap
        if upper_bound > context:
            raise HTTPException(422, 'request exceeds conservative context budget')


def task_token_reservation_bound(body,profile):
    """Reuse the existing conservative text framing bound; refuse unknown hard cases."""
    if not profile.get('context_window') or not profile.get('max_output_tokens'):
        return None
    if any(k in body for k in ('enable_thinking','reasoning_effort')):
        return None
    if any(not isinstance(m,dict) or m.get('content') is not None and not isinstance(m.get('content'),str) for m in body['messages']):
        return None
    cap=body.get('max_completion_tokens',body.get('max_tokens'))
    bound=len(json.dumps(body,ensure_ascii=False).encode())+64*len(body['messages'])+cap
    return bound if bound<=profile['context_window'] else None
