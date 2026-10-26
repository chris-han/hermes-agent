"""Profile-bound Semantier feedback custody. Independent of Nous telemetry.

A dedicated possession-bound feedback authorization authenticates submission.
The reporting capability remains DPAPI-protected and is never returned locally.
"""
from datetime import datetime,timedelta,timezone, UTC
import json
from functools import cache
import importlib.util
import os
from uuid import uuid4
from hermes_constants import get_hermes_home
from hermes_cli import cloud_device_client as device

CONSENT='semantier-operational-reporting/1'
MAX_QUEUE=1000
MAX_BYTES=1024*1024


@cache
def _contract():
    """Load the exact shared contract shipped with this Hermes composition."""
    from pm.paths import repo_root
    path = repo_root() / 'semantier-runtime/src/contracts/client_operations.py'
    spec = importlib.util.spec_from_file_location('semantier_client_operations_contract', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Shipped Semantier feedback contract is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path():
    device._path()  # Existing profile/link lifecycle checks.
    path=get_hermes_home()/'.semantier-feedback.bin'
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('Protected feedback store must not be linked')
    return path


def _read():
    path=_path()
    if not path.exists():return {'enabled':False,'queue':[],'dropped':0}
    if path.stat().st_size>MAX_BYTES+65536:raise ValueError('Feedback store exceeds limit')
    value=json.loads(device._unprotect(path.read_bytes()))
    if value.get('home')!=str(get_hermes_home().resolve()):raise ValueError('Feedback profile mismatch')
    return value


def _write(value):
    from utils import atomic_write_bytes
    value['home']=str(get_hermes_home().resolve())
    atomic_write_bytes(_path(),device._protect(json.dumps(value).encode()),mode=0o600,fsync_dir=True)


def _service():
    cfg=device._config();settings=cfg.get('client_feedback') or {}
    existing=device._read()
    origin=settings.get('service_url') or existing.get('origin')
    portal=settings.get('portal_origin') or existing.get('portal_origin')
    if not origin or not portal:raise ValueError('Semantier feedback service is not configured for this profile')
    return device._origin(origin),device._origin(portal)


def _request(origin,secret,method,path,body=None):
    import httpx
    with httpx.Client(timeout=15,follow_redirects=False,trust_env=False) as client:
        r=client.request(method,origin+'/api/client-operations'+path,json=body,headers={'authorization':'Bearer '+secret} if secret else {})
    if r.status_code not in {200,201}:
        error=device.CloudDeviceError(r.status_code)
        if r.status_code==429:
            try:delay=int(r.headers.get('retry-after','30'))
            except ValueError:delay=30
            error.retry_after_seconds=max(1,min(delay,900))
        raise error
    return r.json()


def status():
    with device._locked():
        state=_read()
        diagnostic=state.get('diagnostic_authorization')
        signed_in=bool(diagnostic and datetime.fromisoformat(diagnostic['expires_at'])>datetime.now(UTC))
        return {'enabled':state['enabled'],'purpose':'update_results_only','destination':'Semantier Cloud','consent_version':CONSENT,'queued':len(state['queue']),'dropped':state['dropped'],'revocation_pending':bool(state.get('revocation_pending')),'authorization_pending':(state.get('authorization') or {}).get('purpose'), 'diagnostic_signed_in':signed_in}


def consent(enabled):
    if type(enabled) is not bool:raise ValueError('Explicit boolean consent required')
    if enabled:raise ValueError('Enable reporting through explicit feedback authorization')
    with device._locked():
        state=_read();state['enabled']=False;state['queue']=[]
        state['revocation_pending']=bool(state.get('capability'));_write(state)
        _revoke(state);return status()


def authorization_start(purpose):
    if purpose not in {'diagnostic','reporting'}:raise ValueError('Invalid feedback purpose')
    from hashlib import sha256
    import secrets
    with device._locked():
        state=_read();origin,portal=_service()
        previous=state.get('authorization')
        reuse=previous and previous['purpose']==purpose and datetime.fromisoformat(previous['expires_at'])>datetime.now(UTC)
        if reuse and previous.get('authorization_url'):
            return {k:previous[k] for k in ('purpose','authorization_url','expires_at')}
        if previous and not reuse:authorization_cancel()
        ref=state.setdefault('profile_ref',uuid4().hex)
        if reuse:
            pending=previous;intent=pending['intent_ref'];proof=pending['proof']
        else:
            intent=uuid4().hex;proof=secrets.token_urlsafe(32)
            pending={'intent_ref':intent,'proof':proof,'purpose':purpose,'origin':origin,'portal':portal,
                     'expires_at':(datetime.now(UTC)+timedelta(minutes=15)).isoformat()}
        state['authorization']=pending;_write(state)
        reply=_request(origin,'','POST','/authorizations',{'intent_ref':intent,'proof_challenge':sha256(proof.encode()).hexdigest(),'purpose':purpose,'profile_ref':ref})
        from urllib.parse import urlsplit,parse_qs
        url=urlsplit(reply['authorization_url'])
        if device._origin(f'{url.scheme}://{url.netloc}')!=portal or url.path!='/client-feedback/authorize' or parse_qs(url.query)!={'intent_ref':[intent]}:
            raise ValueError('Feedback authorization origin mismatch')
        pending.update(authorization_url=reply['authorization_url'],expires_at=reply['expires_at']);state['authorization']=pending;_write(state)
        return {k:pending[k] for k in ('purpose','authorization_url','expires_at')}


def authorization_poll():
    with device._locked():
        state=_read();pending=state.get('authorization')
        if not pending:return {'status':'unavailable'}
        reply=_request(pending['origin'],'','POST','/authorizations/'+pending['intent_ref']+'/poll',{'proof':pending['proof']})
        if reply['status']=='approved':
            if reply['purpose']!=pending['purpose'] or reply['profile_ref']!=state['profile_ref']:raise ValueError('Feedback authorization scope mismatch')
            if pending['purpose']=='reporting':state.update(enabled=True,installation_ref=reply['installation_ref'],capability=reply['capability'],origin=pending['origin'],queue=[],revocation_pending=False)
            else:state['diagnostic_authorization']={'origin':pending['origin'],'capability':reply['capability'],'expires_at':reply['capability_expires_at']}
            state.pop('authorization',None);_write(state)
        elif reply['status'] in {'expired','denied','cancelled'}:
            state.pop('authorization',None);_write(state)
        return {'status':reply['status'],'purpose':pending['purpose']}


def authorization_cancel():
    with device._locked():
        state=_read();pending=state.get('authorization')
        if pending:
            _request(pending['origin'],'','POST','/authorizations/'+pending['intent_ref']+'/cancel',{'proof':pending['proof']})
            state.pop('authorization',None);_write(state)
        return {'status':'cancelled'}


def _diagnostic_identity():
    item=_read().get('diagnostic_authorization')
    if not item or datetime.fromisoformat(item['expires_at'])<=datetime.now(UTC):raise device.CloudDeviceError(401)
    return item['origin'],item['capability']


def _revoke(state):
    if not state.get('revocation_pending'):return
    try:_request(state['origin'],state['capability'],'DELETE','/installations/'+state['installation_ref']+'/reporting')
    except device.CloudDeviceError as e:
        if e.status not in {401,404}:return
    except (OSError,ValueError):return
    state.pop('capability',None);state['revocation_pending']=False;_write(state)


def record(events):
    UpdateEvent = _contract().UpdateEvent
    with device._locked():
        state=_read()
        if not state['enabled']:return status()
        existing={e['event_id']:e for e in state['queue']}
        for raw in events:
            event=UpdateEvent.model_validate(raw).model_dump()
            event['installation_ref']=state['installation_ref']
            if event['event_id'] in existing:
                if existing[event['event_id']]!=event:raise ValueError('Changed event identity')
                continue
            state['queue'].append(event);existing[event['event_id']]=event
        cutoff=datetime.now(UTC)-timedelta(days=7)
        retained=[e for e in state['queue'] if datetime.fromisoformat(e['occurred_at'])>=cutoff]
        state['dropped']+=len(state['queue'])-len(retained);state['queue']=retained
        while len(state['queue'])>MAX_QUEUE or len(json.dumps(state['queue']).encode())>MAX_BYTES:
            state['queue'].pop(0);state['dropped']+=1
        _write(state)
    return flush()


def flush():
    with device._locked():
        state=_read()
        if not state['enabled']:
            _revoke(state);return status()
        if not state['queue']:return status()
        now=datetime.now(UTC)
        if state.get('retry_after') and datetime.fromisoformat(state['retry_after'])>now:return status()
        cutoff=now-timedelta(days=7)
        retained=[e for e in state['queue'] if datetime.fromisoformat(e['occurred_at'])>=cutoff]
        state['dropped']+=len(state['queue'])-len(retained);state['queue']=retained
        batch=state['queue'][:100]
        import httpx
        try:reply=_request(state['origin'],state['capability'],'POST','/events',{'events':batch})
        except (httpx.HTTPError,device.CloudDeviceError) as error:
            if isinstance(error,device.CloudDeviceError) and error.status in {401,403}:
                state['enabled']=False;state['queue']=[];state.pop('capability',None)
            else:
                failures=min(state.get('failures',0)+1,8);state['failures']=failures
                state['retry_after']=(now+timedelta(seconds=min(15*2**failures,900))).isoformat()
            _write(state);return status()
        done={x['event_id'] for x in reply['items'] if x['result'] in {'accepted','duplicate','rejected'}}
        state['queue']=[x for x in state['queue'] if x['event_id'] not in done]
        state['failures']=0;state.pop('retry_after',None);_write(state)
        return status()


def preview(body):
    contract = _contract()
    sanitize, MAX_DIAGNOSTIC_BYTES = contract.sanitize, contract.MAX_DIAGNOSTIC_BYTES
    note,n=sanitize(str(body.get('note',''))[:4000]);context,m=sanitize(str(body.get('context',''))[:32768])
    payload={'schema_version':'1.0.0','submission_key':body.get('submission_key') or uuid4().hex,
             'app_version':body['app_version'],'platform':'win32','release_id':body.get('release_id'),
             'update_id':body.get('update_id'),'error_code':body.get('error_code'),
             'note':note,'sections':{'update':context} if body.get('include_update',True) else {}}
    Diagnostic = _contract().Diagnostic
    payload=Diagnostic.model_validate(payload).model_dump()
    size=len(json.dumps(payload,ensure_ascii=False).encode())
    if size>MAX_DIAGNOSTIC_BYTES:raise ValueError('Diagnostic too large')
    return {'payload':payload,'bytes':size,'redactions':n+m,'retention_days':30,'destination':'Semantier Cloud'}


def submit(body):
    Diagnostic = _contract().Diagnostic
    data=Diagnostic.model_validate(body).model_dump()
    with device._locked():
        origin,secret=_diagnostic_identity()
        try:
            # A lost POST response can be recovered without another submission.
            try:return _request(origin,secret,'GET','/diagnostics/receipts?submission_key='+data['submission_key'])
            except device.CloudDeviceError as error:
                if error.status!=404:raise
            return _request(origin,secret,'POST','/diagnostics',data)
        except device.CloudDeviceError as error:
            if error.status==401:
                state=_read();state.pop('diagnostic_authorization',None);_write(state)
            raise


def erase(diagnostic_id):
    if not isinstance(diagnostic_id,str) or not diagnostic_id.isascii() or not diagnostic_id.isalnum():raise ValueError('Invalid diagnostic ID')
    with device._locked():
        origin,secret=_diagnostic_identity()
        return _request(origin,secret,'DELETE','/diagnostics/'+diagnostic_id+'/payload')
