"""Core-owned hosted channel pairing; no file or environment allowlist writes."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time

from agents.auth_db_postgres import connection,write_transaction,_stamp


class PostgresPairingStore:
    def __init__(self, *, pairing_dir=None, scope=None):
        if scope is None and pairing_dir is None:
            raise ValueError('explicit governed pairing scope required')
        self.scope=str(scope) if scope is not None else str(Path(pairing_dir).resolve())
        self._dir=Path(pairing_dir) if pairing_dir is not None else None
        self._lock=threading.RLock()
        self.profile=None

    @staticmethod
    def looks_like_request_id(value):
        value=str(value or "").strip()
        return len(value)==16 and all(ch in "0123456789abcdefABCDEF" for ch in value)

    @contextmanager
    def _transaction(self):
        with self._lock,write_transaction(),connection() as c:
            yield c

    @staticmethod
    def _aliases(platform,user_id):
        from gateway.pairing import _user_id_aliases
        return _user_id_aliases(platform,user_id)

    @staticmethod
    def _normalized(platform,user_id):
        from gateway.pairing import _normalize_user_id
        return _normalize_user_id(platform,user_id)

    def is_approved(self,platform,user_id):
        with connection() as c:
            return bool(c.execute('SELECT 1 FROM semantier_identity.gateway_pairing_approved WHERE scope=%s AND platform=%s AND user_id=ANY(%s)',(self.scope,platform,list(self._aliases(platform,user_id)))).fetchone())

    def list_approved(self,platform=None):
        with connection() as c:
            rows=c.execute('SELECT platform,user_id,user_name,approved_at FROM semantier_identity.gateway_pairing_approved WHERE scope=%s AND (%s::text IS NULL OR platform=%s) ORDER BY platform,user_id',(self.scope,platform,platform)).fetchall()
            return [{**r,'approved_at':r['approved_at'].isoformat().replace('+00:00','Z')} for r in rows]

    def _approve_user(self,platform,user_id,user_name=''):
        with self._transaction() as c:
            c.execute('DELETE FROM semantier_identity.gateway_pairing_approved WHERE scope=%s AND platform=%s AND user_id=ANY(%s)',(self.scope,platform,list(self._aliases(platform,user_id))))
            c.execute('INSERT INTO semantier_identity.gateway_pairing_approved VALUES(%s,%s,%s,%s,now())',(self.scope,platform,self._normalized(platform,user_id),user_name))

    def revoke(self,platform,user_id):
        with self._transaction() as c:
            return c.execute('DELETE FROM semantier_identity.gateway_pairing_approved WHERE scope=%s AND platform=%s AND user_id=ANY(%s)',(self.scope,platform,list(self._aliases(platform,user_id)))).rowcount>0

    def _limit(self,c,key):
        row=c.execute('SELECT coalesce(extract(epoch FROM deadline_at),value) AS value FROM semantier_identity.gateway_pairing_limits WHERE scope=%s AND key=%s',(self.scope,key)).fetchone()
        return row['value'] if row else 0

    def _set_limit(self,c,key,value):
        counter = key.startswith('_failures:')
        c.execute('INSERT INTO semantier_identity.gateway_pairing_limits(scope,key,value,deadline_at) VALUES(%s,%s,%s,%s) ON CONFLICT(scope,key) DO UPDATE SET value=excluded.value,deadline_at=excluded.deadline_at',(self.scope,key,value if counter else 0,None if counter else _stamp(value)))

    def _cleanup(self,c,platform=None):
        c.execute("DELETE FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND (%s::text IS NULL OR platform=%s) AND created_at<now()-interval '1 hour'",(self.scope,platform,platform))

    def generate_code(self,platform,user_id,user_name=''):
        with self._transaction() as c:
            self._cleanup(c,platform)
            now=time.time();aliases=self._aliases(platform,user_id)
            if self._limit(c,'_lockout:'+platform)>now or any(now-self._limit(c,platform+':'+a)<600 for a in aliases): return None
            count=c.execute('SELECT count(*) AS count FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND platform=%s',(self.scope,platform)).fetchone()['count']
            if count>=3:return None
            code=''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
            salt=secrets.token_bytes(16)
            stored=json.dumps({'request_id':secrets.token_hex(8),'salt':salt.hex(),'hash':hashlib.sha256(salt+code.encode()).hexdigest()},sort_keys=True)
            c.execute('INSERT INTO semantier_identity.gateway_pairing_pending VALUES(%s,%s,%s,%s,%s,now())',(self.scope,platform,stored,self._normalized(platform,user_id),user_name))
            for alias in aliases:self._set_limit(c,platform+':'+alias,now)
            return code

    def _matching(self,c,platform,*,code=None,request_id=None):
        self._cleanup(c,platform)
        if code is not None and self._limit(c,'_lockout:'+platform)>time.time():return None
        rows=c.execute('SELECT * FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND platform=%s',(self.scope,platform)).fetchall()
        for row in rows:
            try:
                proof=json.loads(row['code'])
                matched=secrets.compare_digest(proof['request_id'],request_id) if request_id is not None else secrets.compare_digest(proof['hash'],hashlib.sha256(bytes.fromhex(proof['salt'])+''.join(str(code).upper().split()).encode()).hexdigest())
            except (ValueError,KeyError,TypeError):continue
            if matched:return row
        if code is not None:
            failures=self._limit(c,'_failures:'+platform)+1
            self._set_limit(c,'_failures:'+platform,0 if failures>=5 else failures)
            if failures>=5:self._set_limit(c,'_lockout:'+platform,time.time()+3600)
        return None

    def peek_code(self,platform,code):
        with self._transaction() as c:
            row=self._matching(c,platform,code=code)
            return {'user_id':row['user_id'],'user_name':row['user_name']} if row else None

    def _approve(self,platform,*,code=None,request_id=None):
        with self._transaction() as c:
            row=self._matching(c,platform,code=code,request_id=request_id)
            if not row:return None
            c.execute('DELETE FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND platform=%s AND code=%s',(self.scope,platform,row['code']))
            self._approve_user(platform,row['user_id'],row['user_name'])
            self._set_limit(c,'_failures:'+platform,0)
            return {'user_id':row['user_id'],'user_name':row['user_name']}

    def approve_code(self,platform,code):return self._approve(platform,code=code)
    def approve_request(self,platform,request_id):return self._approve(platform,request_id=request_id.strip().lower())

    def list_pending(self,platform=None):
        with self._transaction() as c:
            self._cleanup(c,platform)
            rows=c.execute('SELECT platform,code,user_id,user_name,created_at FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND (%s::text IS NULL OR platform=%s) ORDER BY platform,created_at',(self.scope,platform,platform)).fetchall()
            out=[]
            for row in rows:
                try:request_id=json.loads(row['code'])['request_id']
                except (ValueError,KeyError,TypeError):request_id=''
                out.append({'platform':row['platform'],'request_id':request_id,'user_id':row['user_id'],'user_name':row['user_name'],'age_minutes':int((time.time()-row['created_at'].timestamp())/60)})
            return out

    def clear_pending(self,platform=None):
        with self._transaction() as c:
            return c.execute('DELETE FROM semantier_identity.gateway_pairing_pending WHERE scope=%s AND (%s::text IS NULL OR platform=%s)',(self.scope,platform,platform)).rowcount

    def _is_locked_out(self,platform):
        with connection() as c:return self._limit(c,'_lockout:'+platform)>time.time()

    def has_recent_decline(self,platform,user_id):
        with connection() as c:return any(time.time()-self._limit(c,'decline:'+platform+':'+a)<86400 for a in self._aliases(platform,user_id))

    def record_decline(self,platform,user_id):
        with self._transaction() as c:
            for alias in self._aliases(platform,user_id):self._set_limit(c,'decline:'+platform+':'+alias,time.time())


def install_hosted_channel_storage():
    """Bind the optional adopted channel runner to core persistence before startup.

    The wrappers retain explicitly local desktop behavior. Hosted lookups never
    fall back to files, and native file errors cannot mint pairing authority.
    """
    from agents.auth_db import is_postgresql
    if not is_postgresql():return
    import gateway.pairing as pairing
    import gateway.platforms.weixin as weixin
    if getattr(pairing,'_semantier_pg_storage_installed',False):return
    native_pairing=pairing.PairingStore
    native_tokens=weixin.ContextTokenStore
    native_load_sync,native_save_sync=weixin._load_sync_buf,weixin._save_sync_buf
    native_load_account,native_save_account=weixin.load_weixin_account,weixin.save_weixin_account

    def pairing_store(profile=None,*,pairing_dir=None,auth_db_path=None):
        if not is_postgresql():return native_pairing(profile=profile)
        if pairing_dir is None:
            if profile:
                from hermes_constants import get_default_hermes_root,get_hermes_dir
                root=get_default_hermes_root()
                home=root if profile=='default' else root/'profiles'/profile
                pairing_dir=get_hermes_dir('platforms/pairing','pairing',home=home)
            else:pairing_dir=pairing._default_pairing_dir()
        return PostgresPairingStore(pairing_dir=pairing_dir)

    class ContextTokens:
        def __new__(cls,home):
            if not is_postgresql():return native_tokens(home)
            return super().__new__(cls)
        def __init__(self,home):self.home=home
        def restore(self,account_id):load_account(self.home,account_id)
        def get(self,account_id,user_id):
            from agents.auth_db_postgres import get_weixin_context_token
            if not load_account(self.home,account_id):raise RuntimeError("governed Weixin account required")
            return get_weixin_context_token(account_id,user_id)
        async def set(self,account_id,user_id,token):
            import asyncio
            from agents.auth_db_postgres import save_weixin_context_token
            if not load_account(self.home,account_id):raise RuntimeError("governed Weixin account required")
            await asyncio.to_thread(save_weixin_context_token,account_id,user_id,token)

    def load_sync(home,account_id):
        if not is_postgresql():return native_load_sync(home,account_id)
        from agents.auth_db_postgres import get_weixin_sync_buf
        if not load_account(home,account_id):raise RuntimeError("governed Weixin account required")
        return get_weixin_sync_buf(account_id)
    def save_sync(home,account_id,sync_buf):
        if not is_postgresql():return native_save_sync(home,account_id,sync_buf)
        from agents.auth_db_postgres import save_weixin_sync_buf
        if not load_account(home,account_id):raise RuntimeError("governed Weixin account required")
        return save_weixin_sync_buf(account_id,sync_buf)
    def load_account(home,account_id):
        if not is_postgresql():return native_load_account(home,account_id)
        from agents.auth_db_postgres import get_weixin_runtime_account
        from agents.gateway_identity import resolve_workspace_paths
        account=get_weixin_runtime_account(account_id)
        if not account:return None
        workspace=account.get('owner_workspace_id')
        if not workspace or resolve_workspace_paths(workspace)[1].resolve()!=Path(home).resolve():
            raise PermissionError('Weixin account belongs to another governed workspace')
        return account
    def save_account(home,*,account_id,token,base_url,user_id=''):
        if not is_postgresql():return native_save_account(home,account_id=account_id,token=token,base_url=base_url,user_id=user_id)
        from agents.auth_db_postgres import save_weixin_runtime_account
        account=load_account(home,account_id)
        if not account:raise RuntimeError('governed Weixin account must be provisioned before adapter persistence')
        save_weixin_runtime_account({**account,'token':token,'base_url':base_url,'external_user_id':user_id or account.get('external_user_id')})
    class HostedPairingFactory(native_pairing):
        def __new__(cls,profile=None,**kwargs):
            if is_postgresql():return pairing_store(profile=profile,**kwargs)
            return super().__new__(cls)

    pairing.PairingStore=HostedPairingFactory
    weixin.ContextTokenStore=ContextTokens
    weixin._load_sync_buf,weixin._save_sync_buf=load_sync,save_sync
    weixin.load_weixin_account,weixin.save_weixin_account=load_account,save_account
    pairing._semantier_pg_storage_installed=True
