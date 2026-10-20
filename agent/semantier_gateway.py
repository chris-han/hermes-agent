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


def gateway_settings() -> GatewaySettings | None:
    from hermes_cli.config import get_env_value, load_config_readonly
    config = load_config_readonly().get('semantier_gateway') or {}
    if not isinstance(config, dict):
        raise ValueError('Invalid Semantier gateway configuration.')
    if config.get('enabled') is not True:
        return None
    base = str(config.get('base_url') or '').rstrip('/')
    parsed = urlsplit(base)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != '/v1' or not parsed.hostname
            or (parsed.scheme != 'https' and not (
                parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}))):
        raise ValueError('Semantier gateway requires an HTTPS /v1 endpoint (HTTP loopback is allowed).')
    key_name = str(config.get('api_key_env') or '')
    key = get_env_value(key_name) if key_name else None
    if not key:
        raise ValueError('Semantier gateway credential is unavailable for this profile.')
    return GatewaySettings(base, key)


def is_gateway_route(base_url: str) -> bool:
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly().get('semantier_gateway') or {}
    return (isinstance(config, dict) and config.get('enabled') is True and bool(config.get('base_url'))
            and str(base_url).rstrip('/') == str(config['base_url']).rstrip('/'))


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
            return refusal('provider_not_configured', 'Semantier gateway is disabled for this profile.')
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
