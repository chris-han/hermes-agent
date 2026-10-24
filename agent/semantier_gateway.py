"""Opt-in, profile-owned Semantier endpoint and read-only billing bridge.

The token supplies identity. These headers carry correlation only. Existing custom-provider
configuration owns model selection; this extension never substitutes routes or credentials.
"""
import logging
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit
from uuid import uuid4, uuid5, NAMESPACE_URL

SCHEMA_VERSION = 'semantier.billing.v1'


@dataclass(frozen=True, repr=False)
class GatewaySettings:
    base_url: str
    api_key: str


def billing_backend() -> str:
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly().get('billing') or {}
    return str(config.get('backend') or 'nous') if isinstance(config, dict) else 'unsupported'


def cloud_connection_config():
    from hermes_cli.runtime_provider import load_config
    cfg = load_config()
    legacy = cfg.get('semantier_gateway') or {}
    providers = cfg.get('providers') or {}
    named = providers.get('semantier-cloud') if isinstance(providers,dict) else None
    model = cfg.get('model') or {}
    current = model if isinstance(model,dict) else {}
    protected = isinstance(named,dict) and named.get('credential_ref') and (current.get('provider') == 'semantier-cloud' or current.get('base_url') == named.get('base_url'))
    if isinstance(legacy,dict) and legacy.get('enabled') is True and not protected:
        return legacy
    if not isinstance(named,dict):
        return None
    base = str(named.get('base_url') or '')
    if current.get('provider') != 'semantier-cloud' and not (base and current.get('base_url') == base):
        return None
    return {'enabled':True,'base_url':base,'api_key_env':named.get('key_env') or current.get('key_env'),
            'api_key':named.get('api_key') or current.get('api_key'), 'credential_ref':named.get('credential_ref')}


def gateway_settings() -> GatewaySettings | None:
    from hermes_cli.config import get_env_value
    config = cloud_connection_config()
    if config is None:
        return None
    base = str(config.get('base_url') or '').rstrip('/')
    parsed = urlsplit(base)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != '/v1' or not parsed.hostname
            or (parsed.scheme != 'https' and not (
                parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}))):
        raise ValueError('Semantier Cloud Provider requires a configured public HTTPS /v1 endpoint (HTTP loopback is allowed).')
    if config.get('credential_ref'):
        from hermes_cli.cloud_device_client import credential
        return GatewaySettings(base, credential(config['credential_ref'], base_url=base))
    key_name = str(config.get('api_key_env') or '')
    key = get_env_value(key_name) if key_name else config.get('api_key')
    if isinstance(key,str) and key.startswith('${') and key.endswith('}'):
        key = get_env_value(key[2:-1])
    if not key:
        raise ValueError('Semantier Cloud Provider scoped credential is unavailable for this profile.')
    return GatewaySettings(base, key)


def is_gateway_route(base_url: str) -> bool:
    config = cloud_connection_config()
    return bool(config and config.get('base_url') and str(base_url).rstrip('/') == str(config['base_url']).rstrip('/'))


def attribute_request(kwargs: dict, *, base_url: str, purpose: str,
                      session_id: str = '', run_id: str = '', request_id: str | None = None) -> dict:
    if not is_gateway_route(base_url):
        return kwargs
    gateway_settings()  # Invalid configured endpoint/credential fails closed only on this route.
    if purpose not in {'chat', 'title_generation'}:
        raise ValueError('This Semantier hosted profile supports primary chat and title generation only.')
    headers = dict(kwargs.get('extra_headers') or {})
    headers['Idempotency-Key'] = (str(uuid5(NAMESPACE_URL, f'hermes:{session_id}:{request_id}'))
                                  if request_id else headers.get('Idempotency-Key') or str(uuid4()))
    for name, value in {'Session-Id': session_id, 'Run-Id': run_id, 'Purpose': purpose}.items():
        value = str(value or '')
        if len(value) > 200 or any(ord(char) < 32 or ord(char) > 126 for char in value):
            raise ValueError('Invalid Semantier correlation value.')
        if value:
            headers[f'X-Semantier-{name}'] = value
    return {**kwargs, 'extra_headers': headers}


def refusal(code: str, message: str, *, request_id: str | None = None) -> dict:
    return {'ok': False, 'backend': 'semantier', 'schema_version': SCHEMA_VERSION,
            'error': code, 'message': message, 'request_id': request_id}


def billing_request(resource: str, *, schema_version: str | None = None) -> dict:
    """Only two read operations are allowed; redirects cannot forward profile tokens."""
    if schema_version not in {SCHEMA_VERSION, 'semantier.billing.v2'}:
        return refusal('unsupported_version', 'This client does not support Semantier billing v1.')
    if resource not in {'state', 'subscription'}:
        return refusal('unsupported_capability', 'Commercial billing actions are unavailable.')
    try:
        settings = gateway_settings()
        if not settings:
            return refusal('provider_not_configured', 'Semantier Cloud Provider is disabled for this profile.')
    except ValueError:
        return refusal('provider_not_configured', 'Semantier endpoint or credential is unavailable for this profile.')
    import requests
    from pydantic import ValidationError
    from tui_gateway.contracts.semantier_billing import (
        SemantierBillingState, SemantierSubscriptionState, SemantierBillingV2)
    account_id = None
    path = '/api/billing/' + resource
    model = SemantierBillingState if resource == 'state' else SemantierSubscriptionState
    if resource == 'state' and schema_version == 'semantier.billing.v2':
        from hermes_cli.config import load_config_readonly
        billing = load_config_readonly().get('billing') or {}
        account_id = billing.get('account_id') if isinstance(billing, dict) else None
        path = '/api/billing/commerce/state'
        if account_id:
            path += '?' + urlencode({'account_id': str(account_id)})
        model = SemantierBillingV2
    try:
        response = requests.get(settings.base_url[:-3] + path,
                                headers={'Authorization': 'Bearer ' + settings.api_key},
                                timeout=15, allow_redirects=False)
        request_id = response.headers.get('x-request-id')
        if response.status_code != 200:
            codes = {401: 'unauthenticated', 403: 'insufficient_scope', 429: 'rate_limited'}
            # Backend error codes are data; upstream messages/URLs can contain secrets and never cross.
            code = codes.get(response.status_code, 'billing_unavailable')
            try:
                error = response.json().get('error', {})
                candidate = error.get('code') if isinstance(error, dict) else error
                if candidate in {'membership_revoked', 'insufficient_scope', 'billing_unavailable',
                                  'unauthenticated', 'unsupported_capability'}:
                    code = candidate
            except (ValueError, AttributeError):
                pass
            return refusal(code, 'Semantier billing is unavailable. Check the active connection and access.',
                           request_id=request_id)
        payload = model.model_validate(response.json()).model_dump(mode='json', exclude_none=model is not SemantierBillingV2)
        if account_id and payload['billing_account_id'] != str(account_id):
            return refusal('billing_unavailable', 'The billing response does not match this profile account.')
        return {'ok': True, **payload}
    except (requests.RequestException, ValueError, ValidationError):
        return refusal('billing_unavailable', 'Semantier billing could not be loaded. Retry this connection.')


def read_billing(resource: str, *, schema_version: str | None = None) -> dict:
    """Shared transport adapter, preserving the existing Nous wire without legacy dispatch."""
    backend = billing_backend()
    if backend == 'semantier':
        return billing_request(resource, schema_version=schema_version)
    if backend != 'nous':
        return refusal('provider_not_configured', 'The selected billing backend is unsupported.')
    try:
        from tui_gateway.billing_view import _serialize_billing_state, _serialize_subscription_state
        if resource == 'state':
            from agent.billing_view import BillingState, build_billing_state
            from hermes_cli.anon_auth import has_free_tier_account
            if has_free_tier_account():
                return _serialize_billing_state(BillingState(logged_in=False), free_tier_account=True)
            return _serialize_billing_state(build_billing_state())
        from agent.subscription_view import build_subscription_state
        return _serialize_subscription_state(build_subscription_state())
    except Exception:
        logging.getLogger(__name__).exception('Could not load legacy Nous billing view')
        return {'ok': True, 'logged_in': False, 'error': f'could not load {resource} state'}


# Memory-only advisory cache. A rotated token or another profile/origin never reuses entries.
# Dispatch authority remains remote; no credential material or catalog is persisted here.
from threading import RLock
from time import monotonic
from hashlib import sha256
from copy import deepcopy
_catalog_lock = RLock()
_catalog_cache = {}
_catalog_pending = {}
_catalog_generation = 0


def clear_cloud_catalog():
    global _catalog_generation
    with _catalog_lock:
        _catalog_generation += 1
        _catalog_cache.clear()
        _catalog_pending.clear()


def cloud_catalog(*, refresh=False):
    from hermes_constants import get_hermes_home
    from concurrent.futures import Future, TimeoutError
    import requests
    try:
        settings = gateway_settings()
    except ValueError:
        clear_cloud_catalog()
        raise
    if settings is None:
        clear_cloud_catalog()
        raise ValueError('Semantier Cloud Provider is not configured for this profile.')
    key = (str(get_hermes_home()), settings.base_url, sha256(settings.api_key.encode()).hexdigest())
    with _catalog_lock:
        generation = _catalog_generation
        cached = _catalog_cache.get(key)
        if not refresh and cached and monotonic()-cached[0] < 30:
            return deepcopy(cached[1])
        pending = _catalog_pending.get(key)
        leader = pending is None
        if leader:
            pending = Future()
            _catalog_pending[key] = pending
    if not leader:
        try:
            return deepcopy(pending.result(timeout=5))
        except TimeoutError:
            raise ValueError('Semantier Cloud Provider catalog timed out; refresh.') from None
    try:
        with requests.Session() as client:
            client.trust_env = False
            response = client.get(settings.base_url[:-3]+'/api/cloud-provider/catalog',
                headers={'Authorization':'Bearer '+settings.api_key}, timeout=5, allow_redirects=False)
        if response.status_code != 200:
            logging.getLogger(__name__).warning('Cloud catalog rejected status=%s',response.status_code)
            raise ValueError('Semantier Cloud Provider catalog unavailable; refresh or reconnect.')
        payload = response.json()
        if payload.get('object') != 'list' or not isinstance(payload.get('data'),list):
            raise ValueError('Invalid Semantier Cloud Provider catalog.')
        with _catalog_lock:
            if len(_catalog_cache) >= 16:
                _catalog_cache.clear()
            if generation == _catalog_generation:
                _catalog_cache[key] = (monotonic(),payload)
        pending.set_result(payload)
        return deepcopy(payload)
    except (requests.RequestException,ValueError):
        with _catalog_lock:
            _catalog_cache.pop(key,None)
        error = ValueError('Semantier Cloud Provider catalog unavailable; refresh or reconnect.')
        pending.set_exception(error)
        raise error from None
    finally:
        with _catalog_lock:
            if _catalog_pending.get(key) is pending:
                _catalog_pending.pop(key,None)


def cloud_model_profile(model, *, refresh=False):
    rows = [row for row in cloud_catalog(refresh=refresh)['data'] if row.get('id') == model]
    if len(rows) != 1:
        raise ValueError('Selected Semantier Cloud Provider model is unavailable; refresh models.')
    return rows[0]


def configure_cloud_request(kwargs, *, base_url):
    """Compile protocol fields for the chosen alias; never substitute supplier transport."""
    if not is_gateway_route(base_url):
        return kwargs
    profile = cloud_model_profile(kwargs.get('model'))
    context, output = profile.get('context_window'), profile.get('max_output_tokens')
    if type(context) is not int or context <= 0 or type(output) is not int or output <= 0:
        raise ValueError('Semantier Cloud Provider context/output limits unknown; deployment configuration required.')
    result = dict(kwargs)
    extra = result.pop('extra_body',{}) or {}
    if not isinstance(extra,dict) or set(extra)-{'enable_thinking'}:
        raise ValueError('Unsupported Semantier Cloud Provider request overrides.')
    result.update(extra)
    caps = profile['capabilities']
    for capability, fields in {'tools':('tools','tool_choice','parallel_tool_calls'),
        'reasoning':('reasoning_effort','enable_thinking'), 'structured_output':('response_format',)}.items():
        if caps.get(capability) is not True or capability == 'reasoning' and not profile['request_parameters']['reasoning_controls']:
            for field in fields:
                result.pop(field,None)
    messages = result.get('messages',[])
    if caps.get('vision') is not True and any(isinstance(m.get('content'),list) and any(
            isinstance(c,dict) and c.get('type') in {'image_url','input_image'} for c in m['content']) for m in messages):
        raise ValueError('Selected Semantier Cloud Provider model has no approved vision capability.')
    import json
    prompt = {k:result[k] for k in ('messages','tools','response_format') if k in result}
    reserve = len(json.dumps(prompt,ensure_ascii=False).encode()) + 64*len(messages)
    room = min(output,32768,context-reserve)
    if room < 1:
        raise ValueError('Semantier Cloud Provider context budget exceeded; compact or start a new session.')
    cap = result.pop('max_completion_tokens',result.pop('max_tokens',None))
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError('Invalid Semantier Cloud Provider output cap.')
    result[profile['request_parameters']['output']] = min(cap or room,room)
    if caps.get('stream') is not True:
        result['stream'] = False
        result.pop('stream_options',None)
    headers = dict(result.get('extra_headers') or {})
    headers['X-Semantier-Capability-Revision'] = profile['revision']
    result['extra_headers'] = headers
    return result


def cloud_switch_result(model, current_provider, *, is_global=False):
    from hermes_cli.model_switch import ModelSwitchResult
    try:
        settings = gateway_settings()
        if settings is None:
            raise ValueError('Semantier Cloud Provider connection required.')
        profile = cloud_model_profile(model,refresh=True)
        if profile['protocol'] not in {'chat','chat_completions'}:
            raise ValueError('Unsupported Semantier Cloud Provider protocol.')
        return ModelSwitchResult(success=True,new_model=model,target_provider='custom',
            provider_changed=current_provider!='custom',api_key=settings.api_key,base_url=settings.base_url,
            api_mode='chat_completions',provider_label='Semantier Cloud Provider',is_global=is_global,
            runtime_capabilities={'supports_'+k:v is True for k,v in profile['capabilities'].items()})
    except ValueError as error:
        return ModelSwitchResult(success=False,error_message=str(error))
