"""Managed native Python child preflight, installed as a read-only sitecustomize.

This is a route safety boundary, not an observation/assessment hook. It runs
before upstream module execution, and deliberately exits on missing or changed
configuration. It contains no credentials or Runtime authority.
"""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def _verify():
    home = Path(os.environ['HERMES_HOME'])
    if str(home.resolve()) != os.environ['SEMANTIER_MANAGED_EXPECTED_HOME']:
        raise ValueError('home')
    url = os.environ['SEMANTIER_MANAGED_EXPECTED_BASE_URL']
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or
        parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or
        os.environ.get('OPENAI_BASE_URL') != url):
        raise ValueError('route')
    config = home / 'config.yaml'
    if config.is_symlink() or hashlib.sha256(config.read_bytes()).hexdigest() != os.environ['SEMANTIER_MANAGED_CONFIG_SHA256']:
        raise ValueError('config')
    # The profile is a unique managed execution capability. Record only the
    # process identity/role and configuration hash, never arguments or secrets.
    cmdline = Path('/proc/self/cmdline').read_bytes().split(b'\0')
    if b'hermes_cli.main' in cmdline:
        role = 'gateway' if b'gateway' in cmdline else 'cli'
        target = home / 'managed-child-preflight.jsonl'
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'role': role, 'config_sha256': os.environ['SEMANTIER_MANAGED_CONFIG_SHA256']}, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


try:
    _verify()
except BaseException:
    os._exit(78)


def _qualified_http_boundary():
    """Optional finite loopback qualification; never enabled by an upstream patch."""
    raw = os.environ.get('SEMANTIER_MANAGED_HTTP_LIMITS')
    if not raw:
        return
    import fcntl
    import ipaddress
    import socket
    import sys
    limits = json.loads(raw)
    if limits != {'discovery': 4, 'model': 4}:
        raise ValueError('invalid qualification request limits')
    home = Path(os.environ['HERMES_HOME'])
    ledger = home / 'http-reservations.jsonl'
    expected = urlsplit(os.environ['SEMANTIER_MANAGED_EXPECTED_BASE_URL'])
    def stop(reason):
        target=home/'http-stop.json'
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600) if not target.exists() else None
        if fd is not None:
            with os.fdopen(fd,'w') as stream:
                stream.write(json.dumps({'reason':reason,'pid':os.getpid()}));stream.flush();os.fsync(stream.fileno())
        raise RuntimeError(reason)
    def audit(event, args):
        if event == 'socket.connect' and isinstance(args[1],tuple):
            if not ipaddress.ip_address(args[1][0]).is_loopback:
                stop('nonloopback socket refused')
        if event == 'socket.getaddrinfo':
            host=args[0].decode('ascii') if isinstance(args[0],bytes) else args[0]
            if host not in ('127.0.0.1','localhost','::1',None):
                import traceback
                with (home/'dns-refused.json').open('w') as stream:
                    stream.write(json.dumps({'host_sha256':hashlib.sha256(str(host).encode()).hexdigest(),'callers':[{'file':frame.filename,'function':frame.name,'line':frame.lineno} for frame in traceback.extract_stack()[-8:]]}));stream.flush();os.fsync(stream.fileno())
                stop('external DNS refused')
    sys.addaudithook(audit)
    def reserve(method, url, body):
        if (home/'http-stop.json').exists():stop('qualification transport already stopped')
        target = urlsplit(str(url))
        if target.scheme != 'http' or target.hostname != expected.hostname or target.port != expected.port:
            stop('unexpected HTTP destination')
        category = 'model' if target.path == '/v1/chat/completions' and method == 'POST' else 'discovery' if (target.path == '/api/show' and method == 'POST') or (target.path in ('/v1/models','/models','/api/tags','/api/v1/models','/models-dev.json') and method == 'GET') else None
        if category is None:
            with (home/'http-refused.jsonl').open('a') as stream:
                stream.write(json.dumps({'method':method,'path_sha256':hashlib.sha256(target.path.encode()).hexdigest()})+'\n');stream.flush();os.fsync(stream.fileno())
            stop('unexpected HTTP operation')
        body = body.encode() if isinstance(body,str) else body or b''
        with ledger.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            prior=[json.loads(line) for line in stream]
            if sum(row['category']==category for row in prior)>=limits[category]:
                stop('physical request limit reached')
            stream.write(json.dumps({'recorded_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'sequence':len(prior)+1,'pid':os.getpid(),'category':category,'method':method,'path':target.path,'body_sha256':hashlib.sha256(body).hexdigest()},sort_keys=True)+'\n')
            stream.flush();os.fsync(stream.fileno())
            return home.parent.name+':'+str(len(prior)+1)
    import httpx
    original = httpx.Client.send
    original_async = httpx.AsyncClient.send
    def send(self,request,**kwargs):
        kwargs['follow_redirects']=False
        request.headers['X-Semantier-Qualification-Request']=reserve(request.method,request.url,request.read())
        try:response=original(self,request,**kwargs)
        except Exception:stop('HTTP transport failed')
        if response.status_code>=300:stop('HTTP response refused')
        return response
    async def asend(self,request,**kwargs):
        kwargs['follow_redirects']=False
        request.headers['X-Semantier-Qualification-Request']=reserve(request.method,request.url,await request.aread())
        try:response=await original_async(self,request,**kwargs)
        except Exception:stop('HTTP transport failed')
        if response.status_code>=300:stop('HTTP response refused')
        return response
    httpx.Client.send=send;httpx.AsyncClient.send=asend
    # OpenAI retries would be additional physical requests, so disable them at construction.
    import openai
    for cls in (openai.OpenAI,openai.AsyncOpenAI):
        original_init=cls.__init__
        def init(self,*args,_original=original_init,**kwargs):
            kwargs['max_retries']=0
            kwargs.setdefault('http_client',httpx.AsyncClient(trust_env=False) if isinstance(self,openai.AsyncOpenAI) else httpx.Client(trust_env=False))
            _original(self,*args,**kwargs)
        cls.__init__=init
    import requests
    original_request=requests.Session.request
    original_send=requests.Session.send
    def request(self,*args,**kwargs):
        self.trust_env=False;kwargs['allow_redirects']=False
        for adapter in self.adapters.values():
            if adapter.max_retries.total != 0:stop('transport retries refused')
        return original_request(self,*args,**kwargs)
    def request_send(self,request,**kwargs):
        request.headers['X-Semantier-Qualification-Request']=reserve(request.method,request.url,request.body)
        try:response=original_send(self,request,**kwargs)
        except Exception:stop('HTTP transport failed')
        if response.status_code>=300:stop('HTTP response refused')
        return response
    requests.Session.request=request;requests.Session.send=request_send


try:
    _qualified_http_boundary()
except BaseException:
    os._exit(78)
