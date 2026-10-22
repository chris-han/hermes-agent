"""Narrow AEAD custody for server-held, account-scoped native key generations."""
from __future__ import annotations

import base64
from dataclasses import dataclass,field
import json
import os
import re


class CredentialDependencyError(ValueError):
    """Only a bounded dependency code may leave this owner."""


def _aad(scope,user_id,generation,policy_hash):
    return json.dumps(['semantier-native-credential-v1',*scope,user_id,int(generation),policy_hash],separators=(',',':'),ensure_ascii=True).encode()


class AEADKeyring:
    def __init__(self,keys,active_key_id):
        if not isinstance(keys,dict) or not 1<=len(keys)<=16 or not isinstance(active_key_id,str) or active_key_id not in keys:
            raise CredentialDependencyError('CREDENTIAL_KEY_UNAVAILABLE')
        if any(not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',k) or not isinstance(v,bytes) or len(v)!=32 for k,v in keys.items()):
            raise CredentialDependencyError('CREDENTIAL_KEY_INVALID')
        if len(set(keys.values()))!=len(keys):raise CredentialDependencyError('CREDENTIAL_KEY_INVALID')
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.exceptions import InvalidTag
        except ImportError:raise CredentialDependencyError('CREDENTIAL_DEPENDENCY_UNAVAILABLE') from None
        self._aes_type,self._invalid_tag=AESGCM,InvalidTag
        self._keys=dict(keys);self.active_key_id=active_key_id

    def __repr__(self):return 'AEADKeyring(<server keys redacted>)'

    def encrypt(self,plaintext,*,scope,user_id,generation,policy_hash):
        nonce=os.urandom(12)
        ciphertext=self._aes_type(self._keys[self.active_key_id]).encrypt(nonce,plaintext.encode(),_aad(scope,user_id,generation,policy_hash))
        return {'encryption_key_id':self.active_key_id,'nonce':nonce,'ciphertext':ciphertext}

    def decrypt(self,row):
        key=self._keys.get(row['encryption_key_id'])
        if key is None:raise CredentialDependencyError('CREDENTIAL_KEY_UNAVAILABLE')
        scope=tuple(row[k] for k in ('target_instance_id','environment_namespace','organization_id','account_id','currency'))
        try:return self._aes_type(key).decrypt(bytes(row['nonce']),bytes(row['ciphertext']),_aad(scope,row['user_id'],row['generation'],row['policy_hash'])).decode()
        except (self._invalid_tag,UnicodeError,ValueError):raise CredentialDependencyError('CREDENTIAL_AUTHENTICATION_FAILED') from None


def credential_keyring(value,*,key_id=None):
    """Bare base64 initial key or explicit historical JSON keyring; no session key."""
    try:
        if not isinstance(value,str) or not 1<=len(value)<=65536:raise CredentialDependencyError('CREDENTIAL_KEY_INVALID')
        reference=key_id or os.environ.get('SEMANTIER_NATIVE_CREDENTIAL_KEY_REF','')
        if reference and 'SESSION' in reference.upper():raise CredentialDependencyError('CREDENTIAL_KEY_REFERENCE_INVALID')
        if value.lstrip().startswith('{'):
            data=json.loads(value)
            if not isinstance(data,dict) or set(data)!={'active','keys'} or not isinstance(data['keys'],dict) or not 1<=len(data['keys'])<=16:raise CredentialDependencyError('CREDENTIAL_KEY_INVALID')
            keys={k:base64.b64decode(v,validate=True) for k,v in data['keys'].items()}
            return AEADKeyring(keys,data['active'])
        reference=key_id or os.environ.get('SEMANTIER_NATIVE_CREDENTIAL_KEY_REF','')
        if not reference or 'SESSION' in reference.upper():raise CredentialDependencyError('CREDENTIAL_KEY_REFERENCE_INVALID')
        return AEADKeyring({reference:base64.b64decode(value,validate=True)},reference)
    except (KeyError,TypeError,ValueError) as exc:
        if isinstance(exc,CredentialDependencyError):raise
        raise CredentialDependencyError('CREDENTIAL_KEY_INVALID') from None


@dataclass(frozen=True)
class ScopedCredential:
    secret: str=field(repr=False)
    target_instance_id: str
    environment_namespace: str
    organization_id: str
    account_id: str
    currency: str
    user_id: str
    generation: int
    desired_revision: int
    native_key_hash: str
    native_team_id: str=''
    key_alias: str=''
    policy_hash: str=''
    models: tuple[str,...]=()

    @property
    def metadata(self):
        return {'semantier_user_id':self.user_id,'semantier_organization_id':self.organization_id,
                'semantier_billing_account_id':self.account_id,'semantier_currency':self.currency,
                'semantier_credential_generation':self.generation,'semantier_desired_revision':self.desired_revision,
                'semantier_native_target_instance_id':self.target_instance_id,'semantier_native_environment_namespace':self.environment_namespace,
                'semantier_native_key_hash':self.native_key_hash}


def dependency_keyring():
    """Resolve local server custody and management presence without HTTP."""
    from agents.tenant_secret_policy import resolve_secret
    keyref=os.environ.get('SEMANTIER_NATIVE_CREDENTIAL_KEY_REF','')
    managementref=os.environ.get('SEMANTIER_MODEL_GATEWAY_PROXY_SECRET_REF','SEMANTIER_MODEL_GATEWAY_PROXY_KEY')
    if not keyref or 'SESSION' in keyref.upper() or not os.environ.get('SEMANTIER_MODEL_GATEWAY_PROXY_URL'):
        return None
    try:
        native=resolve_secret(keyref,workspace_env=None,platform_env=os.environ,auth_cache=None)
        management=resolve_secret(managementref,workspace_env=None,platform_env=os.environ,auth_cache=None)
    except (ValueError,TypeError):return None
    if not native or not management:return None
    try:return credential_keyring(native.value,key_id=keyref)
    except CredentialDependencyError:return None


def dependency_status():
    """Local configuration only, for read-only readiness projections."""
    return None if dependency_keyring() is not None else 'BLOCKED_DEPENDENCY'
