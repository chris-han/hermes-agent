"""Finite pinned LiteLLM API projection; native objects never confer local authority."""
from __future__ import annotations

from agents.model_gateway.credential_store import CredentialDependencyError


class NativeAccountError(ValueError):
    def __init__(self,code,*,retryable=False,retry_after=None):
        self.code,self.retryable,self.retry_after=code,retryable,retry_after
        super().__init__(code)


class NativeAccountAdapter:
    def __init__(self,proxy):
        self.proxy=proxy

    async def _call(self,method,path,body=None,params=None):
        return await self.proxy.native_operation(method,path,body,params)

    @staticmethod
    def _owned(row,metadata):
        if not isinstance(row,dict) or not isinstance(row.get('metadata'),dict) or any(row['metadata'].get(k)!=v for k,v in metadata.items()):
            raise NativeAccountError('OWNERSHIP_CONFLICT')
        return row

    async def _read(self,kind,identity):
        data=await self._call('GET','/'+kind+'/info',params={kind+'_id' if kind!='key' else 'key':identity})
        if data is None:return None
        if not isinstance(data,dict):raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        return data.get('info' if kind=='key' else kind+'_info')

    async def _create(self,path,body,kind,identity):
        # Caller-supplied identities make a committed-but-lost reply recoverable.
        try:await self._call('POST',path,body)
        except NativeAccountError as error:
            if error.code not in {'NATIVE_TIMEOUT','NATIVE_UNAVAILABLE','INVALID_POLICY'}:raise
            row=await self._read(kind,identity)
            if row is None:raise error
            return row
        row=await self._read(kind,identity)
        if row is None:raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        return row

    async def _user(self,user_id,base):
        metadata={**base,'user_id':user_id,'identity_revision':1}
        row=await self._read('user',user_id)
        if row is None:
            row=await self._create('/user/new',{'user_id':user_id,'auto_create_key':False,'send_invite_email':False,
                'user_role':'internal_user_viewer','models':['no-default-models'],'metadata':metadata},'user',user_id)
        self._owned(row,metadata)
        if row.get('user_id')!=user_id or row.get('user_role')!='internal_user_viewer' or row.get('models')!=['no-default-models']:
            raise NativeAccountError('OWNERSHIP_CONFLICT')

    async def _team(self,p,base):
        policy=p['desired_policy'];team=p['native_team_id']
        metadata={**base,**{k:p[k] for k in ('organization_id','currency')},'billing_account_id':p['account_id'],'funding':policy['funding']}
        row=await self._read('team',team)
        models=policy['models'] or ['no-default-models']
        owned={**metadata,'desired_revision':p['desired_revision'],'policy_hash':p['desired_policy_hash']}
        fields={'team_id':team,'models':models,**{k:v for k,v in policy['limits'].items() if k in {'rpm_limit','tpm_limit'}},'metadata':owned}
        if row is None:
            row=await self._create('/team/new',{**fields,'team_alias':'semantier-'+team,'blocked':True},'team',team)
            self._owned(row,metadata)
            # This pinned API automatically adds its management caller as team
            # admin. A newly created owned team must remove that implicit member;
            # subsequent reconciliations reject any unexpected membership.
            requested={m['user_id'] for m in policy['members'] if m['active']}
            for member in row.get('members_with_roles') or []:
                if member.get('user_id') not in requested and member.get('role')=='admin':
                    await self._call('POST','/team/member_delete',{'team_id':team,'user_id':member['user_id']})
        self._owned(row,metadata)
        if row.get('team_id')!=team:raise NativeAccountError('OWNERSHIP_CONFLICT')
        # Only declared operational policy fields are updated; spend/budget and
        # unrelated metadata remain owned by the native service/operator.
        await self._call('POST','/team/update',{**fields,'metadata':{**row['metadata'],**owned}})
        row=await self._read('team',team);self._owned(row,owned)
        if sorted(row.get('models') or [])!=sorted(models) or any(row.get(k)!=v for k,v in policy['limits'].items() if k in {'rpm_limit','tpm_limit'}):
            raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        return row,metadata

    async def _members(self,p,row,base):
        members=p['desired_policy']['members'];known={m['user_id'] for m in members}
        existing=row.get('members_with_roles') or []
        if row.get('blocked') is True and row.get('metadata',{}).get('membership_initialized') is not True:
            # The pinned team-info API omits created_by. Its sole implicit
            # creator admin is removable only on an exact owned, still-closed
            # team before first membership verification. This also recovers a
            # create that committed after both its reply and read-back timed out.
            implicit=[m for m in existing if m.get('user_id') not in known]
            if len(implicit)==1 and implicit[0].get('role')=='admin':
                await self._call('POST','/team/member_delete',{'team_id':p['native_team_id'],'user_id':implicit[0]['user_id']})
            row=await self._read('team',p['native_team_id']);existing=row.get('members_with_roles') or []
        if any(m.get('user_id') not in known or m.get('role')!='user' for m in existing):
            raise NativeAccountError('OWNERSHIP_CONFLICT')
        current={m['user_id'] for m in existing};active={m['user_id'] for m in members if m['active']}
        for uid in sorted(active):
            await self._user(uid,base)
            if uid not in current:await self._call('POST','/team/member_add',{'team_id':p['native_team_id'],'member':{'user_id':uid,'role':'user'}})
        for uid in sorted(current-active):
            await self._call('POST','/team/member_delete',{'team_id':p['native_team_id'],'user_id':uid})
        row=await self._read('team',p['native_team_id'])
        if {(m.get('user_id'),m.get('role')) for m in row.get('members_with_roles',[])}!={(uid,'user') for uid in active}:
            raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        await self._call('POST','/team/update',{'team_id':p['native_team_id'],'metadata':{**row['metadata'],'membership_initialized':True}})

    @staticmethod
    def _key_scope(p,row):
        return {'owner':'semantier','schema_version':1,'target_instance_id':p['target_instance_id'],
                'environment_namespace':p['environment_namespace'],'organization_id':p['organization_id'],
                'billing_account_id':p['account_id'],'currency':p['currency'],'funding':p['desired_policy']['funding'],
                'user_id':row['user_id'],'generation':row['generation'],'policy_hash':row['policy_hash'],
                'desired_revision':row.get('policy_revision',p['desired_revision'])}

    async def _block_key(self,p,credential):
        row=await self._read('key',credential['native_key_hash'])
        if row is None:return
        self._owned(row,self._key_scope(p,credential))
        if row.get('user_id')!=credential['user_id'] or row.get('team_id')!=p['native_team_id']:
            raise NativeAccountError('OWNERSHIP_CONFLICT')
        if row.get('status') in {'deleted','revoked'} or row.get('deleted_at'):return
        if not row.get('blocked'):await self._call('POST','/key/block',{'key':credential['native_key_hash']})
        row=await self._read('key',credential['native_key_hash'])
        if not row or row.get('blocked') is not True:raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)

    async def _key(self,p,credential):
        metadata=self._key_scope(p,dict(user_id=credential.user_id,generation=credential.generation,policy_hash=credential.policy_hash,policy_revision=credential.desired_revision))
        listing=await self._call('GET','/key/list',params={'key_alias':credential.key_alias,'return_full_object':'true','size':2})
        if not isinstance(listing,dict) or not isinstance(listing.get('keys'),list):raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        keys=listing['keys']
        if len(keys)>1 or listing.get('total_count',len(keys))>1:raise NativeAccountError('OWNERSHIP_CONFLICT')
        if keys and keys[0].get('token')!=credential.native_key_hash:raise NativeAccountError('OWNERSHIP_CONFLICT')
        row=await self._read('key',credential.native_key_hash)
        if row is None:
            row=await self._create('/key/generate',{'key':credential.secret,'key_alias':credential.key_alias,'user_id':credential.user_id,
                'team_id':p['native_team_id'],'models':p['desired_policy']['models'],'key_type':'llm_api','metadata':metadata,
                **p['desired_policy']['limits']},'key',credential.native_key_hash)
        self._owned(row,metadata)
        if (row.get('user_id')!=credential.user_id or row.get('team_id')!=p['native_team_id'] or row.get('key_alias')!=credential.key_alias
            or sorted(row.get('models') or [])!=sorted(p['desired_policy']['models']) or row.get('blocked') is True or row.get('status')!='active'
            or any(row.get(k)!=v for k,v in p['desired_policy']['limits'].items())):
            raise NativeAccountError('OWNERSHIP_CONFLICT')
        return credential.native_key_hash

    async def reconcile(self,projection,*,prepare_credential,credentials):
        p=projection;policy=p['desired_policy']
        base={'owner':'semantier','schema_version':1,'target_instance_id':p['target_instance_id'],'environment_namespace':p['environment_namespace']}
        row,metadata=await self._team(p,base)
        eligible={m['user_id'] for m in policy['members'] if m['eligible']}
        # Whole-account suspension closes the team first. Member changes never
        # suspend another eligible member of an already active account.
        if not policy['account_eligible'] and row.get('blocked') is not True:
            await self._call('POST','/team/update',{'team_id':p['native_team_id'],'blocked':True})
        await self._members(p,row,base)
        selected={}
        for uid in sorted(eligible):
            try:credential=await prepare_credential(uid)
            except CredentialDependencyError:raise NativeAccountError('BLOCKED_DEPENDENCY') from None
            selected[uid]=await self._key(p,credential)
        for old in credentials:
            if old['native_key_hash']!=selected.get(old['user_id']) and old['status']!='BLOCKED':await self._block_key(p,old)
        if policy['account_eligible']:
            # The pinned block/unblock endpoints leave the auth cache stale;
            # team/update writes the same field and refreshes cached policy.
            await self._call('POST','/team/update',{'team_id':p['native_team_id'],'blocked':False})
        row=await self._read('team',p['native_team_id']);self._owned(row,metadata)
        blocked=not policy['account_eligible']
        if row.get('blocked') is not blocked:raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True)
        return {'team_blocked':blocked,'keys':selected,'native_team_id':p['native_team_id'],'policy_hash':p['desired_policy_hash'],'desired_revision':p['desired_revision']}
