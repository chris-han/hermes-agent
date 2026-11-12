"""Windows-native Cloud device connection v1.0.1; no secrets cross the local API.

The existing profile scope/config writer own selection. This owner persists the
private protocol journal with DPAPI; network replies never become config authority
until their exact original profile generation is committed.
"""
from contextlib import contextmanager
from datetime import datetime, timezone, UTC
from hashlib import sha256
import json
import os
import secrets
import socket
import ssl
import sys
import threading
from urllib.parse import parse_qs, urlsplit

from hermes_constants import get_hermes_home

CONTRACT = 'semantier-cloud-device-connection/v1'
_TERMINAL = {'denied', 'expired', 'cancelled', 'revoked'}
_THREAD_LOCK = threading.RLock()
_LOCK_HOLDER = threading.local()


# Canonical public service, also linked by the provider catalog. An explicit
# profile deployment/portal always wins; selecting a model still waits for approval.
DEFAULT_CLOUD_ORIGIN = 'https://cloud.semantier.cn'


class CloudConnectionError(ValueError):
    """Public, secret-free recovery information; never constructed from server prose."""
    def __init__(self, code, message, status=502, *, retryable=False, recovery='retry'):
        self.code, self.status = code, status
        self.retryable, self.recovery = retryable, recovery
        self.diagnostics = {'error_source': 'client', 'http_status': None}
        super().__init__(message)

    def detail(self):
        return {'code': self.code, 'message': str(self),
                'retryable': self.retryable, 'recovery': self.recovery, **self.diagnostics}


class CloudDeviceError(CloudConnectionError):
    def __init__(self, status, reason=None):
        self.reason = reason if reason == 'start_conflict' else None
        code, message, recovery = {
            401: ('cloud_authorization_required', 'Semantier Cloud authorization is invalid or expired. Sign in again.', 'reauthorize'),
            403: ('cloud_authorization_denied', 'Semantier Cloud refused this authorization. Check the account and permissions.', 'reauthorize'),
            404: ('cloud_session_unavailable', 'This Cloud authorization session is unavailable. Start sign-in again.', 'reauthorize'),
            409: ('cloud_server_conflict', 'Semantier Cloud reported a sign-in conflict. Retry the same connection.', 'retry'),
            410: ('cloud_session_expired', 'This Cloud authorization has expired. Start sign-in again.', 'reauthorize'),
            422: ('cloud_protocol_unsupported', 'Cloud protocol unsupported by this Hermes version. Update the app or check the configured service.', 'upgrade_client'),
            429: ('cloud_rate_limited', 'Semantier Cloud is busy. Wait briefly, then retry.', 'retry'),
        }.get(status, ('cloud_server_error', 'Semantier Cloud could not complete sign-in (HTTP '+str(status)+'). Retry later.', 'retry'))
        if 300 <= status < 400:
            code, message, recovery = ('cloud_origin_redirected', 'Semantier Cloud redirected this request. Check the configured service origin.', 'provider_settings')
        super().__init__(code, message, 502 if 300 <= status < 400 else status, retryable=status >= 500 or status in {409, 429}, recovery=recovery)
        self.diagnostics = {'error_source': 'server', 'http_status': status}


class CloudCredentialError(CloudConnectionError):
    """A definite native credential refusal, not a network or entitlement verdict."""
    def __init__(self, message):
        super().__init__('cloud_credential_expired', message, 401, recovery='reauthorize')


def _transport_error(error, *, transport=None):
    """Use exception types/libcurl codes, never echo or classify private URL prose."""
    pending, seen = [error], set()
    codes, names, dns, tls = set(), set(), False, False
    while pending and len(seen) < 20:
        item = pending.pop()
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        names.add(type(item).__name__.lower())
        dns |= isinstance(item, socket.gaierror)
        tls |= isinstance(item, ssl.SSLError)
        code = getattr(item, 'code', None)
        if isinstance(code, int):
            codes.add(code)
        pending.extend([item.__cause__, item.__context__, *item.args])
        pending.extend([getattr(item, 'reason', None)])
    def public(code, message, status, **kwargs):
        result = CloudConnectionError(code, message, status, **kwargs)
        result.diagnostics = {'error_source': 'transport', 'http_status': None}
        if codes:
            result.diagnostics['transport_code'] = min(codes)
        return result

    if dns or codes & {5, 6} or 'nameresolutionerror' in names:
        return public('cloud_dns_failed', 'Semantier Cloud DNS lookup failed. Check DNS or the configured provider.', 503, retryable=True, recovery='network')
    if tls or codes & {35, 51, 58, 60, 77, 83, 90} or any('ssl' in name or 'certificate' in name for name in names):
        return public('cloud_tls_failed', 'Semantier Cloud TLS certificate validation failed. Check the trusted certificate and provider settings.', 502, recovery='tls_configuration')
    if codes & {1, 4, 95, 96} or transport == 'http3':
        return public('cloud_http3_unavailable', 'Semantier Cloud HTTP/3 transport is unavailable. Check HTTP/3 support and UDP connectivity.', 503, retryable=True, recovery='http3')
    return public('cloud_network_unavailable', 'Semantier Cloud could not be reached. Check the network and retry the same connection.', 503, retryable=True, recovery='network')


def _digest(value):
    return sha256(value).hexdigest()


def _dpapi(value, decrypt=False):
    if os.name != 'nt':
        raise CloudConnectionError('cloud_storage_unavailable', 'Protected Windows credential storage is unavailable; connect on Windows.', 503, recovery='windows_storage')
    import ctypes
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    raw = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise CloudConnectionError('cloud_storage_unavailable', 'Protected Windows credential storage is unavailable; reconnect.', 503, recovery='windows_storage')
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


def _protect(value):
    return _dpapi(value)


def _unprotect(value):
    return _dpapi(value, decrypt=True)


def _path():
    home = get_hermes_home()
    if not home.is_dir() or home.is_symlink() or getattr(home, 'is_junction', lambda: False)():
        raise CloudConnectionError('cloud_profile_unavailable', 'Original profile is unavailable.', 409, recovery='original_profile')
    path = home / '.cloud-device.bin'
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise CloudConnectionError('cloud_profile_mismatch', 'Protected device store must not be linked.', 409, recovery='original_profile')
    return path


@contextmanager
def _locked():
    from hermes_cli.auth import _file_lock
    from hermes_cli.profiles import _profile_lifecycle_lock
    from hermes_constants import get_default_hermes_root
    path = _path().with_suffix('.lock')
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise CloudConnectionError('cloud_profile_mismatch', 'Protected device lock must not be linked.', 409, recovery='original_profile')
    with _THREAD_LOCK, _profile_lifecycle_lock(get_default_hermes_root()), _file_lock(
            path, _LOCK_HOLDER, 10, 'Device connection is busy; retry.'):
        _path()  # lifecycle fence must still contain the same live home
        yield


def _read():
    path = _path()
    if not path.exists():
        return {}
    if path.stat().st_size > 1024*1024:
        raise CloudConnectionError('cloud_storage_invalid', 'Protected device store is invalid; reconnect.', 409, recovery='reauthorize')
    try:
        state = json.loads(_unprotect(path.read_bytes()))
        if not isinstance(state, dict) or state.get('home') != str(get_hermes_home().resolve()):
            raise CloudConnectionError('cloud_profile_mismatch', 'Protected Cloud credentials belong to another profile. Return to the original profile.', 409, recovery='original_profile')
        return state
    except CloudConnectionError:
        raise
    except (OSError, ValueError):
        raise CloudConnectionError('cloud_storage_invalid', 'Protected device store is unavailable; reconnect.', 409, recovery='reauthorize') from None


def _write(state):
    from utils import atomic_write_bytes
    state['home'] = str(get_hermes_home().resolve())
    atomic_write_bytes(_path(), _protect(json.dumps(state).encode()), mode=0o600, fsync_dir=True)


def _time(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise CloudConnectionError('cloud_response_invalid', 'Cloud timestamp is invalid; retry or reauthorize.', 502) from None
    if parsed.tzinfo is None:
        raise CloudConnectionError('cloud_response_invalid', 'Cloud timestamp must be UTC-aware.', 502, recovery='retry')
    return parsed.astimezone(UTC).timestamp()


def _now():
    return datetime.now(UTC).timestamp()


def _utc(value):
    return datetime.fromtimestamp(_time(value),UTC).isoformat()


def _after(seconds):
    return datetime.fromtimestamp(_now()+seconds,UTC).isoformat()


def _origin(value):
    try:
        p = urlsplit(value)
        p.port  # Reject malformed/out-of-range ports before issuing a request.
    except (TypeError, ValueError):
        raise CloudConnectionError('cloud_config_invalid', 'Semantier Cloud provider origin is invalid.', 422, recovery='provider_settings') from None
    if (p.username or p.password or p.query or p.fragment or p.path not in {'', '/', '/v1'}
            or not p.hostname or p.scheme not in {'http', 'https'}
            or p.scheme == 'http' and p.hostname not in {'localhost', '127.0.0.1', '::1'}):
        raise CloudConnectionError('cloud_config_invalid', 'Cloud requires HTTPS or explicit DEVELOPMENT loopback.', 422, recovery='provider_settings')
    return f'{p.scheme}://{p.netloc}'


def _config():
    from hermes_cli.config import read_user_config_raw
    cfg = read_user_config_raw()
    if (not isinstance(cfg, dict)
            or any(cfg.get(key) is not None and not isinstance(cfg[key], dict) for key in ('providers', 'billing', 'semantier_gateway'))
            or ((cfg.get('providers') or {}).get('semantier-cloud') is not None
                and not isinstance((cfg.get('providers') or {}).get('semantier-cloud'), dict))):
        raise CloudConnectionError('cloud_config_invalid', 'Semantier Cloud provider configuration is invalid. Check this profile settings.', 422, recovery='provider_settings')
    return cfg


def _config_hash():
    path = get_hermes_home()/'config.yaml'
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
        raise CloudConnectionError('cloud_profile_mismatch', 'Original profile configuration must not be linked.', 409, recovery='original_profile')
    return _digest(path.read_bytes())


def _cloud_http(origin):
    settings = (_config().get('providers') or {}).get('semantier-cloud') or {}
    version = settings.get('http_version')
    if version is None and sys.platform == 'win32' and urlsplit(origin).scheme == 'https':
        version = '3'
    if version not in (None, '3'):
        raise CloudConnectionError('cloud_config_invalid', 'Unsupported Cloud HTTP version.', 422, recovery='provider_settings')
    if version == '3':
        if urlsplit(origin).scheme != 'https':
            raise CloudConnectionError('cloud_config_invalid', 'HTTP/3 requires HTTPS.', 422, recovery='provider_settings')
        try:
            from curl_cffi import requests
        except ImportError:
            raise CloudConnectionError('cloud_http3_unavailable', 'HTTP/3 transport unavailable.', 503, recovery='http3') from None
        options = {'http_version': 'v3only', 'trust_env': False}
        ca_file = settings.get('ssl_ca_cert')
        if ca_file is not None:
            if not isinstance(ca_file, str) or not ca_file.strip():
                raise CloudConnectionError('cloud_config_invalid', 'Invalid Cloud CA certificate path.', 422, recovery='tls_configuration')
            options['verify'] = ca_file.strip()
        if sys.platform == 'win32' and ca_file is None:
            from curl_cffi import CurlOpt
            # libcurl CURLSSLOPT_NATIVE_CA: include the Windows ROOT and CA stores.
            options['curl_options'] = {CurlOpt.SSL_OPTIONS: 1 << 4}
        return requests, options
    import requests
    return requests, {}


def _wire(origin, operation, body=None, bearer=None):
    requests, session_options = _cloud_http(origin)
    headers = {'Accept':'application/json'}
    if bearer:
        headers['Authorization'] = 'Bearer '+bearer
    try:
        with requests.Session(**session_options) as client:
            client.trust_env = False
            kwargs = dict(headers=headers, timeout=15, allow_redirects=False)
            if operation == 'readiness':
                response = client.get(origin+'/api/cloud-provider/device/readiness', **kwargs)
            else:
                response = client.post(origin+'/api/cloud-provider/device/'+operation,
                    json={'contract_version':CONTRACT, **(body or {})}, **kwargs)
        if len(response.content) > 1024*1024:
            raise CloudConnectionError('cloud_response_invalid', 'Semantier Cloud returned an oversized response. Retry the same connection.', 502)
        if response.status_code != 200:
            reason = None
            if response.status_code == 409:
                try:
                    detail = response.json().get('detail')
                    if detail == 'start_conflict':
                        reason = detail
                except (ValueError, AttributeError):
                    pass
            raise CloudDeviceError(response.status_code, reason)
        try:
            result = response.json()
        except ValueError:
            raise CloudConnectionError('cloud_response_invalid', 'Semantier Cloud returned an invalid response. Retry the same connection.', 502) from None
        if not isinstance(result, dict) or result.get('contract_version') != CONTRACT:
            raise CloudConnectionError('cloud_response_invalid', 'Cloud connection contract mismatch.', 502, recovery='retry')
        return result
    except requests.exceptions.RequestException as error:
        public = _transport_error(error, transport='http3' if session_options.get('http_version') == 'v3only' else 'http')
        public.diagnostics.update(operation=operation.rsplit('/', 1)[-1],
                                  transport='http3' if session_options.get('http_version') == 'v3only' else 'http')
        raise public from None


def _public(state):
    return {'session_id':state['session_id'], 'status':state.get('status', 'pending'),
            'expires_at':state.get('expires_at'), 'retry_after':state.get('retry_after',5),
            'error_message':state.get('error_message'), 'account_id':state.get('account_id'),
            'model':state.get('model'), 'readiness':state.get('readiness')}


def _started(state):
    return {'session_id':state['session_id'], 'flow':'device_code', 'user_code':'',
            'verification_url':state['verification_uri'],
            'expires_in':max(0, int(_time(state['expires_at'])-_now())), 'poll_interval':5}


def _new_state(previous):
    cfg = _config()
    provider = (cfg.get('providers') or {}).get('semantier-cloud') or {}
    model = cfg.get('model') or {}
    legacy = cfg.get('semantier_gateway') or {}
    base = (provider.get('base_url') or (legacy.get('base_url') if legacy.get('enabled') is True else None)
            or (model.get('base_url') if isinstance(model,dict) and model.get('provider') == 'semantier-cloud' else None))
    if not base:
        base = DEFAULT_CLOUD_ORIGIN
    origin = _origin(str(base))
    from hermes_constants import get_default_hermes_root
    home = str(get_hermes_home().resolve())
    return {'home':home, 'origin':origin, 'session_id':'cloud-'+secrets.token_urlsafe(24),
            'proof':secrets.token_urlsafe(32), 'claim_id':secrets.token_hex(16),
            'portal_origin':_origin(str(provider.get('portal_origin') or (cfg.get('billing') or {}).get('portal_origin') or origin)),
            'original_config':cfg,
            'runtime_ref':_digest(str(get_default_hermes_root().resolve()).encode()),
            'profile_ref':_digest(home.encode()), 'generation':previous.get('generation',0)+1,
            'config_hash':_config_hash(), 'status':'starting',
            'credentials':previous.get('credentials',{})}


def start():
    with _locked():
        state = _read()
        if state and state.get('status') not in _TERMINAL | {'approved'}:
            if state.get('expires_at') and _time(state['expires_at']) <= _now():
                state['status'] = 'expired'
            elif state.get('verification_uri') and (state.get('config_hash') == _config_hash()
                    or state.get('committed_hash') == _config_hash()
                    or state.get('staged_config') == _config()):
                return _started(state)
        if not state or state.get('status') != 'starting':
            state = _new_state(state)
            _write(state)  # persist idempotency/proof BEFORE a possibly lost start reply
        for attempt in range(2):
            body = {k:state[k] for k in ('runtime_ref','profile_ref','generation')}
            body.update(client_label='Semantier Agent Windows', possession_challenge=_digest(state['proof'].encode()),
                        idempotency_key=state['session_id'])
            try:
                reply = _wire(state['origin'], 'start', body)
                break
            except CloudDeviceError as error:
                if attempt or error.status != 409 or error.reason != 'start_conflict':
                    raise
                # An older client may have persisted the proof before losing its
                # reply, then upgraded its start payload. Only that exact conflict
                # permits a new intent; retain all already-installed credentials.
                state = _new_state(state)
                _write(state)
        uri = reply['data']['verification_uri']
        if _origin(f'{urlsplit(uri).scheme}://{urlsplit(uri).netloc}') != state['portal_origin']:
            raise CloudConnectionError('cloud_response_invalid', 'Cloud verification origin mismatch.', 502, recovery='provider_settings')
        parsed = urlsplit(uri)
        if (parsed.path != '/billing/connect' or parsed.fragment or parsed.username or parsed.password
                or parse_qs(parsed.query) != {'intent_ref':[reply['intent_ref']]}):
            raise CloudConnectionError('cloud_response_invalid', 'Cloud verification URL is invalid.', 502, recovery='retry')
        state.update(intent_ref=reply['intent_ref'], expires_at=_utc(reply['expires_at']),
                     verification_uri=uri, status='pending', retry_after=5)
        _time(state['expires_at'])
        _write(state)
        return _started(state)


def _session(session_id):
    state = _read()
    if not state or not secrets.compare_digest(state.get('session_id',''), session_id):
        raise CloudConnectionError('cloud_session_unavailable', 'Cloud session is unavailable for this profile.', 404, recovery='original_profile')
    return state


def cancel(session_id):
    with _locked():
        state = _session(session_id)
        state['status'] = 'cancelled'
        if state.get('original_config') and (state.get('committed_hash') == _config_hash()
                or state.get('staged_config') == _config()):
            from hermes_cli.config import atomic_config_replace
            atomic_config_replace(get_hermes_home()/'config.yaml',state['original_config'])
        if state.get('credential_ref'):
            state['credentials'].pop(state['credential_ref'],None)
        _write(state)  # local fence survives network failure/restart
        remote = False
        try:
            if state.get('intent_ref'):
                _wire(state['origin'], state['intent_ref']+'/cancel', {'proof':state['proof']})
                remote = True
        except ValueError:
            pass
        return {'ok':True, 'remote_cancelled':remote}


def poll(session_id):
    with _locked():
        state = _session(session_id)
        try:
            if state['status'] in _TERMINAL | {'approved'}:
                return _public(state)
            if state.get('staged_config') and _config() == state['staged_config']:
                state['committed_hash'] = _config_hash()
                _write(state)
            if state.get('committed_hash') and _config_hash() == state['committed_hash']:
                return _ack(state)
            if state['config_hash'] != _config_hash():
                cancel(session_id)
                return _public(_read())
            if _time(state['expires_at']) <= _now():
                state['status'] = 'expired'
                _write(state)
                return _public(state)
            if state.get('next_poll') and _now() < _time(state['next_poll']):
                return _public(state)
            reply = _wire(state['origin'], state['intent_ref']+'/poll', {'proof':state['proof']})
            if reply.get('intent_ref') != state['intent_ref']:
                raise CloudConnectionError('cloud_profile_mismatch', 'Cloud intent mismatch.', 409, recovery='original_profile')
            status = reply['status']
            state['retry_after'] = min(60,max(5,int(reply.get('retry_after',5))))
            state['next_poll'] = _after(state['retry_after'])
            if status in _TERMINAL:
                state['status'] = status
            elif status in {'approved','claimed'}:
                _install(state, reply['data'])
                return _ack(state)
            elif status not in {'pending','slow_down'}:
                raise CloudConnectionError('cloud_response_invalid', 'Cloud connection status is invalid; reconnect.', 502, recovery='reauthorize')
            _write(state)
            return _public(state)
        except CloudConnectionError as error:
            # Only a definite authorization/session refusal retires this intent.
            # The owning session was validated before this boundary. Preserve all
            # installed credentials; network/409 failures keep the same proof.
            if error.code in {'cloud_authorization_required', 'cloud_authorization_denied',
                              'cloud_session_unavailable', 'cloud_session_expired', 'cloud_credential_expired'}:
                state['status'] = 'denied' if error.code in {
                    'cloud_authorization_required', 'cloud_authorization_denied'} else 'expired'
                _write(state)
            raise


def _binding(state, data):
    for key in ('runtime_ref','profile_ref','generation'):
        if data.get(key) != state[key]:
            raise CloudConnectionError('cloud_profile_mismatch', 'Cloud original profile generation mismatch.', 409, recovery='original_profile')
    if not data.get('account_id') or data.get('currency') not in {'CNY','USD'}:
        raise CloudConnectionError('cloud_response_invalid', 'Cloud account binding is unavailable.', 502, recovery='reauthorize')
    return {k:data[k] for k in ('runtime_ref','profile_ref','generation','account_id')}


def _install(state, binding):
    from hermes_cli import config as config_mod
    command = {**_binding(state,binding), 'proof':state['proof'], 'claim_id':state['claim_id']}
    reply = _wire(state['origin'], state['intent_ref']+'/claim', command)
    data = reply['data']
    if reply.get('intent_ref') != state['intent_ref'] or data.get('claim_id') != state['claim_id']:
        raise CloudConnectionError('cloud_profile_mismatch', 'Cloud claim binding mismatch.', 409, recovery='original_profile')
    if _binding(state,data) != _binding(state,binding) or data.get('currency') != binding['currency']:
        raise CloudConnectionError('cloud_profile_mismatch', 'Cloud account binding mismatch.', 409, recovery='original_profile')
    if _origin(data['inference_origin']) != state['origin'] or _origin(data['billing_origin']) != state['origin']:
        raise CloudConnectionError('cloud_profile_mismatch', 'Cloud approved origin mismatch.', 409, recovery='original_profile')
    if _time(data['installation_expires_at']) <= _now() or _time(data['credential']['expires_at']) <= _now():
        raise CloudConnectionError('cloud_credential_expired', 'Cloud credential expired; reauthorize.', 401, recovery='reauthorize')
    # Credential journal precedes config publication: a crash can recover the same claim.
    ref = data['credential']['token_id']
    data['credential']['expires_at'] = _utc(data['credential']['expires_at'])
    data['installation_expires_at'] = _utc(data['installation_expires_at'])
    state['credentials'][ref] = data
    state['credential_ref'] = ref
    state['account_id'] = data['account_id']
    state['currency'] = data['currency']
    _write(state)
    with config_mod._CONFIG_LOCK:
        if _config_hash() != state['config_hash']:
            raise CloudConnectionError('cloud_profile_changed', 'Original profile was reconfigured; reconnect.', 409, recovery='original_profile')
        cfg = _config()
        entry = {'base_url':state['origin']+'/v1', 'credential_ref':ref, 'portal_origin':state['portal_origin']}
        previous_cloud = ((cfg.get('providers') or {}).get('semantier-cloud') or {})
        if previous_cloud.get('http_version') == '3':
            entry['http_version'] = '3'
            if previous_cloud.get('ssl_ca_cert'):
                entry['ssl_ca_cert'] = previous_cloud['ssl_ca_cert']
        cfg['providers'] = cfg.get('providers') or {}
        cfg['providers']['semantier-cloud'] = entry
        if not isinstance(cfg.get('model'),dict):
            cfg['model'] = {'default':str(cfg.get('model') or '')}
        previous = cfg['model']
        if previous.get('provider') != 'semantier-cloud' and (previous.get('api_key') or previous.get('key_env') or previous.get('key_cmd')):
            from hermes_cli.config_providers import get_compatible_custom_providers
            saved = {key:previous[key] for key in ('base_url','api_key','key_env','key_cmd','api_mode') if previous.get(key)}
            if not saved.get('base_url'):
                raise CloudConnectionError('cloud_config_invalid', 'Previous embedded provider credential needs its configured endpoint before switching.', 422, recovery='provider_settings')
            saved['default_model'] = previous.get('default','')
            existing = get_compatible_custom_providers(cfg)
            if not any(all(row.get(key) == value for key,value in saved.items() if key != 'default_model') for row in existing):
                key = 'saved-'+_digest(saved['base_url'].encode())[:12]
                name = key
                suffix = 1
                while name in cfg['providers']:
                    suffix += 1
                    name = key+'-'+str(suffix)
                saved['name'] = urlsplit(saved['base_url']).hostname or str(previous.get('provider') or 'Saved provider')
                cfg['providers'][name] = saved
        cfg['model'].update(provider='semantier-cloud', base_url=entry['base_url'],api_mode='chat_completions')
        cfg['model'].pop('api_key',None)
        cfg['model'].pop('key_env',None)
        cfg['model'].pop('key_cmd',None)
        rows = _catalog(state['origin'], data['credential']['secret'])
        allowed = sorted({row['id'] for row in rows if isinstance(row.get('id'),str) and row['id']})
        selected = cfg['model'].get('default')
        if selected not in allowed:
            selected = allowed[0] if allowed else ''
        cfg['model']['default'] = selected
        cfg['model'].pop('capability_revision',None)
        if selected:
            revision = next(row.get('revision') for row in rows if row.get('id') == selected)
            if not isinstance(revision,str) or not revision:
                raise CloudConnectionError('cloud_response_invalid', 'Cloud model capability revision unavailable; retry the same connection.', 502, recovery='retry')
            cfg['model']['capability_revision'] = revision
        state['model'] = selected
        cfg['billing'] = cfg.get('billing') or {}
        cfg['billing'].update(backend='semantier', account_id=data['account_id'], currency=data['currency'], portal_origin=state['portal_origin'])
        # Persist the complete intended config inside the protected journal before its atomic replace.
        state['staged_config'] = cfg
        state['status'] = 'installing'
        _write(state)
        config_mod.atomic_config_replace(get_hermes_home()/'config.yaml',cfg)
        state['committed_hash'] = _config_hash()
        _write(state)


def _catalog(origin, bearer):
    requests, session_options = _cloud_http(origin)
    try:
        with requests.Session(**session_options) as client:
            client.trust_env = False
            response = client.get(origin+'/api/cloud-provider/catalog',
                headers={'Authorization':'Bearer '+bearer}, timeout=5, allow_redirects=False)
        if response.status_code != 200:
            raise CloudDeviceError(response.status_code)
        if len(response.content) > 1024*1024:
            raise CloudConnectionError('cloud_response_invalid', 'Cloud model directory unavailable; retry the same connection.', 502)
        try:
            result = response.json()
        except ValueError:
            raise CloudConnectionError('cloud_response_invalid', 'Cloud model directory invalid.', 502) from None
        if not isinstance(result, dict) or result.get('object') != 'list' or not isinstance(result.get('data'),list):
            raise CloudConnectionError('cloud_response_invalid', 'Cloud model directory invalid.', 502, recovery='retry')
        return result['data']
    except requests.exceptions.RequestException as error:
        public = _transport_error(error, transport='http3' if session_options.get('http_version') == 'v3only' else 'http')
        public.diagnostics.update(operation='catalog', transport='http3' if session_options.get('http_version') == 'v3only' else 'http')
        raise public from None


def _ack(state):
    data = state['credentials'][state['credential_ref']]
    body = {**_binding(state,data), 'proof':state['proof'], 'claim_id':state['claim_id'],
            'installed_generation':state['generation']}
    reply = _wire(state['origin'], state['intent_ref']+'/ack', body)
    if (reply.get('status') != 'installed' or reply.get('intent_ref') != state['intent_ref']
            or reply.get('data',{}).get('claim_id') != state['claim_id']
            or reply.get('data',{}).get('installed_generation') != state['generation']):
        raise CloudConnectionError('cloud_response_invalid', 'Cloud installation acknowledgment unavailable; retry the same connection.', 502, recovery='retry')
    state['status'] = 'approved'
    state.pop('staged_config',None)
    _refresh_readiness(state)
    _write(state)
    return _public(state)


def _refresh_readiness(state):
    if state.get('next_readiness') and _now() < _time(state['next_readiness']):
        return
    state['next_readiness'] = _after(30)
    try:
        data = state['credentials'][state['credential_ref']]
        readiness = _wire(state['origin'],'readiness',bearer=data['credential']['secret'])
        if readiness.get('account_id') != state['account_id'] or readiness.get('currency') != state['currency']:
            raise CloudConnectionError('cloud_profile_mismatch', 'Cloud readiness account mismatch.', 409, recovery='original_profile')
        state['readiness'] = {key:readiness.get(key) for key in (
            'account_id','currency','payment_entitlement_allowance','account_provisioning','catalog','device','as_of','renewal')}
    except CloudDeviceError as error:
        if error.status in {401,403}:
            state['status'] = 'revoked'
            state['credentials'].pop(state['credential_ref'],None)
        state['readiness'] = None
    except ValueError:
        state['readiness'] = None


def credential(ref, *, base_url):
    with _locked():
        state = _read()
        data = state.get('credentials',{}).get(ref)
        if not data or _time(data['credential']['expires_at']) <= _now():
            raise CloudCredentialError('Cloud credential unavailable or expired; reauthorize.')
        if _origin(base_url) != _origin(data['inference_origin']) or urlsplit(base_url).path != '/v1':
            raise CloudConnectionError('cloud_profile_mismatch', 'Protected Cloud credential cannot be sent to another endpoint.', 409, recovery='original_profile')
        return data['credential']['secret']


def status():
    with _locked():
        state = _read()
        if state.get('status') == 'approved':
            _refresh_readiness(state)
            _write(state)
        ref = state.get('credential_ref')
        installed = state.get('credentials',{}).get(ref)
        usable = bool(installed and _time(installed['credential']['expires_at']) > _now())
        return {'logged_in':bool(state.get('status') == 'approved' and usable), 'source':'semantier-cloud',
                'account_id':state.get('account_id'), 'expires_at':installed['credential']['expires_at'] if installed else None,
                'readiness':state.get('readiness'), 'reauthorization_required':not usable or state.get('status') == 'revoked'}


def disconnect():
    with _locked():
        state = _read()
        remote = False
        ref = state.get('credential_ref')
        if ref:
            try:
                _wire(state['origin'],'revoke',bearer=credential(ref,base_url=state['origin']+'/v1'))
                remote = True
            except ValueError:
                pass
            state['credentials'].pop(ref,None)
        state['status'] = 'revoked'
        _write(state)
        from hermes_cli import config as config_mod
        with config_mod._CONFIG_LOCK:
            cfg = _config()
            entry = (cfg.get('providers') or {}).get('semantier-cloud') or {}
            if ref and entry.get('credential_ref') == ref:
                cfg['providers'].pop('semantier-cloud', None)
                if (cfg.get('billing') or {}).get('account_id') == state.get('account_id'):
                    cfg['billing'].pop('account_id', None)
                config_mod.atomic_config_replace(get_hermes_home()/'config.yaml', cfg)
        return {'ok':True,'provider':'semantier-cloud','remote_revoked':remote,
                'reauthorization_required':True}
