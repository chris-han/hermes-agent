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
import threading
from urllib.parse import parse_qs, urlsplit

from hermes_constants import get_hermes_home

CONTRACT = 'semantier-cloud-device-connection/v1'
_TERMINAL = {'denied', 'expired', 'cancelled', 'revoked'}
_THREAD_LOCK = threading.RLock()
_LOCK_HOLDER = threading.local()


class CloudDeviceError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__('Cloud connection unavailable (HTTP '+str(status)+'); retry or reauthorize.')


def _digest(value):
    return sha256(value).hexdigest()


def _dpapi(value, decrypt=False):
    if os.name != 'nt':
        raise ValueError('Protected Windows credential storage is unavailable; connect on Windows.')
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
        raise ValueError('Protected Windows credential storage is unavailable; reconnect.')
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
        raise ValueError('Original profile is unavailable.')
    path = home / '.cloud-device.bin'
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('Protected device store must not be linked.')
    return path


@contextmanager
def _locked():
    from hermes_cli.auth import _file_lock
    from hermes_cli.profiles import _profile_lifecycle_lock
    from hermes_constants import get_default_hermes_root
    path = _path().with_suffix('.lock')
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('Protected device lock must not be linked.')
    with _THREAD_LOCK, _profile_lifecycle_lock(get_default_hermes_root()), _file_lock(
            path, _LOCK_HOLDER, 10, 'Device connection is busy; retry.'):
        _path()  # lifecycle fence must still contain the same live home
        yield


def _read():
    path = _path()
    if not path.exists():
        return {}
    if path.stat().st_size > 1024*1024:
        raise ValueError('Protected device store is invalid; reconnect.')
    try:
        state = json.loads(_unprotect(path.read_bytes()))
        if not isinstance(state, dict) or state.get('home') != str(get_hermes_home().resolve()):
            raise ValueError()
        return state
    except (OSError, ValueError):
        raise ValueError('Protected device store is unavailable; reconnect.') from None


def _write(state):
    from utils import atomic_write_bytes
    state['home'] = str(get_hermes_home().resolve())
    atomic_write_bytes(_path(), _protect(json.dumps(state).encode()), mode=0o600, fsync_dir=True)


def _time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Cloud timestamp must be UTC-aware.')
    return parsed.astimezone(UTC).timestamp()


def _now():
    return datetime.now(UTC).timestamp()


def _utc(value):
    return datetime.fromtimestamp(_time(value),UTC).isoformat()


def _after(seconds):
    return datetime.fromtimestamp(_now()+seconds,UTC).isoformat()


def _origin(value):
    p = urlsplit(value)
    if (p.username or p.password or p.query or p.fragment or p.path not in {'', '/', '/v1'}
            or not p.hostname or p.scheme not in {'http', 'https'}
            or p.scheme == 'http' and p.hostname not in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('Cloud requires HTTPS or explicit DEVELOPMENT loopback.')
    return f'{p.scheme}://{p.netloc}'


def _config():
    from hermes_cli.config import read_user_config_raw
    return read_user_config_raw()


def _config_hash():
    path = get_hermes_home()/'config.yaml'
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
        raise ValueError('Original profile configuration must not be linked.')
    return _digest(path.read_bytes())


def _wire(origin, operation, body=None, bearer=None):
    import requests
    headers = {'Accept':'application/json'}
    if bearer:
        headers['Authorization'] = 'Bearer '+bearer
    try:
        with requests.Session() as client:
            client.trust_env = False
            kwargs = dict(headers=headers, timeout=15, allow_redirects=False)
            if operation == 'readiness':
                response = client.get(origin+'/api/cloud-provider/device/readiness', **kwargs)
            else:
                response = client.post(origin+'/api/cloud-provider/device/'+operation,
                    json={'contract_version':CONTRACT, **(body or {})}, **kwargs)
        if response.status_code != 200 or len(response.content) > 1024*1024:
            raise CloudDeviceError(response.status_code)
        result = response.json()
        if not isinstance(result, dict) or result.get('contract_version') != CONTRACT:
            raise ValueError('Cloud connection contract mismatch.')
        return result
    except requests.RequestException:
        raise ValueError('Cloud connection unavailable; retry the same connection.') from None


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
    base = provider.get('base_url') or (cfg.get('model') or {}).get('base_url')
    origin = _origin(str(base or ''))
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
        body = {k:state[k] for k in ('runtime_ref','profile_ref','generation')}
        body.update(client_label='Semantier Agent Windows', possession_challenge=_digest(state['proof'].encode()),
                    idempotency_key=state['session_id'])
        reply = _wire(state['origin'], 'start', body)
        uri = reply['data']['verification_uri']
        if _origin(f'{urlsplit(uri).scheme}://{urlsplit(uri).netloc}') != state['portal_origin']:
            raise ValueError('Cloud verification origin mismatch.')
        parsed = urlsplit(uri)
        if (parsed.path != '/billing/connect' or parsed.fragment or parsed.username or parsed.password
                or parse_qs(parsed.query) != {'intent_ref':[reply['intent_ref']]}):
            raise ValueError('Cloud verification URL is invalid.')
        state.update(intent_ref=reply['intent_ref'], expires_at=_utc(reply['expires_at']),
                     verification_uri=uri, status='pending', retry_after=5)
        _time(state['expires_at'])
        _write(state)
        return _started(state)


def _session(session_id):
    state = _read()
    if not state or not secrets.compare_digest(state.get('session_id',''), session_id):
        raise ValueError('Cloud session is unavailable for this profile.')
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
            raise ValueError('Cloud intent mismatch.')
        status = reply['status']
        state['retry_after'] = min(60,max(5,int(reply.get('retry_after',5))))
        state['next_poll'] = _after(state['retry_after'])
        if status in _TERMINAL:
            state['status'] = status
        elif status in {'approved','claimed'}:
            _install(state, reply['data'])
            return _ack(state)
        elif status not in {'pending','slow_down'}:
            raise ValueError('Cloud connection status is invalid; reconnect.')
        _write(state)
        return _public(state)


def _binding(state, data):
    for key in ('runtime_ref','profile_ref','generation'):
        if data.get(key) != state[key]:
            raise ValueError('Cloud original profile generation mismatch.')
    if not data.get('account_id') or data.get('currency') not in {'CNY','USD'}:
        raise ValueError('Cloud account binding is unavailable.')
    return {k:data[k] for k in ('runtime_ref','profile_ref','generation','account_id')}


def _install(state, binding):
    from hermes_cli import config as config_mod
    command = {**_binding(state,binding), 'proof':state['proof'], 'claim_id':state['claim_id']}
    reply = _wire(state['origin'], state['intent_ref']+'/claim', command)
    data = reply['data']
    if reply.get('intent_ref') != state['intent_ref'] or data.get('claim_id') != state['claim_id']:
        raise ValueError('Cloud claim binding mismatch.')
    if _binding(state,data) != _binding(state,binding) or data.get('currency') != binding['currency']:
        raise ValueError('Cloud account binding mismatch.')
    if _origin(data['inference_origin']) != state['origin'] or _origin(data['billing_origin']) != state['origin']:
        raise ValueError('Cloud approved origin mismatch.')
    if _time(data['installation_expires_at']) <= _now() or _time(data['credential']['expires_at']) <= _now():
        raise ValueError('Cloud credential expired; reauthorize.')
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
            raise ValueError('Original profile was reconfigured; reconnect.')
        cfg = _config()
        entry = {'base_url':state['origin']+'/v1', 'credential_ref':ref, 'portal_origin':state['portal_origin']}
        cfg.setdefault('providers',{})['semantier-cloud'] = entry
        if not isinstance(cfg.get('model'),dict):
            cfg['model'] = {'default':str(cfg.get('model') or '')}
        cfg['model'].update(provider='semantier-cloud', base_url=entry['base_url'])
        cfg['model'].pop('api_key',None)
        cfg['model'].pop('key_env',None)
        rows = _catalog(state['origin'], data['credential']['secret'])
        allowed = sorted({row['id'] for row in rows if isinstance(row.get('id'),str) and row['id']})
        selected = cfg['model'].get('default')
        if selected not in allowed:
            selected = allowed[0] if allowed else ''
        cfg['model']['default'] = selected
        state['model'] = selected
        cfg.setdefault('billing',{}).update(backend='semantier', account_id=data['account_id'], currency=data['currency'], portal_origin=state['portal_origin'])
        # Persist the complete intended config inside the protected journal before its atomic replace.
        state['staged_config'] = cfg
        state['status'] = 'installing'
        _write(state)
        config_mod.atomic_config_replace(get_hermes_home()/'config.yaml',cfg)
        state['committed_hash'] = _config_hash()
        _write(state)


def _catalog(origin, bearer):
    import requests
    try:
        with requests.Session() as client:
            client.trust_env = False
            response = client.get(origin+'/api/cloud-provider/catalog',
                headers={'Authorization':'Bearer '+bearer}, timeout=5, allow_redirects=False)
        if response.status_code != 200 or len(response.content) > 1024*1024:
            raise ValueError('Cloud model directory unavailable; retry the same connection.')
        result = response.json()
        if result.get('object') != 'list' or not isinstance(result.get('data'),list):
            raise ValueError('Cloud model directory invalid.')
        return result['data']
    except requests.RequestException:
        raise ValueError('Cloud model directory unavailable; retry the same connection.') from None


def _ack(state):
    data = state['credentials'][state['credential_ref']]
    body = {**_binding(state,data), 'proof':state['proof'], 'claim_id':state['claim_id'],
            'installed_generation':state['generation']}
    reply = _wire(state['origin'], state['intent_ref']+'/ack', body)
    if (reply.get('status') != 'installed' or reply.get('intent_ref') != state['intent_ref']
            or reply.get('data',{}).get('claim_id') != state['claim_id']
            or reply.get('data',{}).get('installed_generation') != state['generation']):
        raise ValueError('Cloud installation acknowledgment unavailable; retry the same connection.')
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
            raise ValueError('Cloud readiness account mismatch.')
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
            raise ValueError('Cloud credential unavailable or expired; reauthorize.')
        if _origin(base_url) != _origin(data['inference_origin']) or urlsplit(base_url).path != '/v1':
            raise ValueError('Protected Cloud credential cannot be sent to another endpoint.')
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
