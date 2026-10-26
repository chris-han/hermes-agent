"""Server-bound model gateway contracts; no provider secrets cross this boundary."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any

PERMISSIONS = frozenset({'model:invoke', 'usage:read_self', 'usage:read_org', 'billing:read',
                         'billing:manage', 'provider_credentials:manage', 'budgets:manage'})
CHAT_FIELDS = frozenset({'model', 'messages', 'max_tokens', 'max_completion_tokens', 'stream',
                         'temperature', 'top_p', 'stop', 'tools', 'tool_choice', 'parallel_tool_calls',
                         'response_format', 'reasoning_effort', 'enable_thinking', 'presence_penalty', 'frequency_penalty', 'seed', 'stream_options'})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def native_fingerprint(row: dict) -> str:
    """Pin the registry view; masked credentials cannot prove independent secret drift."""
    info=row.get('model_info') or {}
    return canonical_hash({'model_name':row.get('model_name'),'litellm_params':row.get('litellm_params'),
                           'model_info':{k:info[k] for k in ('id','revision') if k in info}})


def validate_chat(body: dict) -> dict:
    if not isinstance(body, dict):
        raise ValueError('request must be an object')
    unsupported = set(body) - CHAT_FIELDS
    if unsupported:
        raise ValueError('unsupported parameters: ' + ', '.join(sorted(unsupported)))
    if not isinstance(body.get('model'), str) or not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('model and nonempty messages required')
    cap = body.get('max_completion_tokens', body.get('max_tokens'))
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 32768:
        raise ValueError('explicit output cap required (1..32768)')
    if 'max_tokens' in body and 'max_completion_tokens' in body:
        raise ValueError('choose one output cap')
    if not isinstance(body.get('stream', False), bool):
        raise ValueError('stream must be boolean')
    if 'reasoning_effort' in body and (not isinstance(body['reasoning_effort'],str) or body['reasoning_effort'] not in {'none','minimal','low','medium','high','xhigh'}):
        raise ValueError('unsupported reasoning_effort')
    if 'enable_thinking' in body and type(body['enable_thinking']) is not bool:
        raise ValueError('enable_thinking must be boolean')
    canonical_hash(body)
    return dict(body)


@dataclass(frozen=True)
class Deployment:
    deployment_id: str
    organization_id: str
    alias: str
    proxy_model: str
    revision: str
    config_fingerprint: str
    account_id: str
    funding: str
    currency: str
    max_cost: Decimal
    provider: str
    region: str
    protocol: str
    credential_binding_id: str
    native_model_id: str
    customer_price_version: str | None = None
    supplier_price_version: str | None = None
    fx_version: str | None = None
    provider_model_id: str | None = None
    model_version: str | None = None
    version_status: str = 'UNRESOLVED'
    capabilities: tuple[str, ...] = ('chat', 'stream', 'tools')
    enabled: bool = True
    qualified: bool = False
    tokens_per_minute: int | None = None
    capability_profile: dict | None = None
    normalizer_version: str | None = None
    inclusion_rules_version: str | None = None

    def __post_init__(self):
        from agents.model_gateway.capabilities import validate_overrides
        validate_overrides(self.capability_profile)
        if (self.capability_profile or {}).get('reasoning_controls') and (self.provider != 'dashscope' or self.protocol != 'openai_chat_completions'):
            raise ValueError('declared thinking control requires DashScope OpenAI chat')
        if (self.normalizer_version, self.inclusion_rules_version) != (None, None):
            if (self.provider != 'dashscope' or
                    (self.normalizer_version, self.inclusion_rules_version) !=
                    ('native-usage.v2', 'provider-inclusion.v2')):
                raise ValueError('unsupported deployment normalization pair')
        if self.tokens_per_minute is not None and (type(self.tokens_per_minute) is not int or not 1 <= self.tokens_per_minute <= 1000000):
            raise ValueError('tokens_per_minute must be an integer in 1..1000000')
        for identifier in (self.deployment_id,self.alias,self.revision,self.account_id,self.provider,self.protocol,self.native_model_id):
            if not identifier or not identifier.isascii() or any(c.isspace() for c in identifier):
                raise ValueError('ASCII-stable machine identifiers required')
        if self.version_status not in {'UNRESOLVED','PINNED'} or (self.version_status=='PINNED' and not self.model_version):
            raise ValueError('explicit model version or unresolved status required')
        if self.qualified:
            raise ValueError('qualified compositions are excluded from common gateway')
        if self.funding not in {'CUSTOMER_CLOUD', 'SEMANTIER_CREDITS'}:
            raise ValueError('unsupported funding')
        if self.currency not in {'CNY', 'USD'} or not self.max_cost.is_finite() or self.max_cost <= 0:
            raise ValueError('known positive cost bound and native currency required')
        if self.funding == 'SEMANTIER_CREDITS' and not self.customer_price_version:
            raise ValueError('platform funding requires pinned customer tariff')
        if not all((self.revision, self.config_fingerprint, self.credential_binding_id, self.region, self.protocol)):
            raise ValueError('exact deployment binding required')


from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer


class ModelReexecutionCommandV1(BaseModel):
    """Explicit control-plane intent; never part of the model request."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['model_reexecution_command.v1'] = 'model_reexecution_command.v1'
    expected_predecessor_request_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    reason_code: Literal['EXPLICIT_REPEAT', 'POST_REPAIR_VALIDATION', 'REVISED_REQUEST']
    reason: str = Field(min_length=1, max_length=1000)
    request: dict
    expected_profile_revision: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        # Preserve historical v1 records/hashes when no revision was frozen.
        value = handler(self)
        if self.expected_profile_revision is None:
            value.pop('expected_profile_revision', None)
        return value

    @model_validator(mode='after')
    def bounded_model_only(self):
        validate_chat(self.request)
        if not self.reason.strip():
            raise ValueError('nonempty reason required')
        if self.request.get('stream') or set(self.request) & {'tools','tool_choice','parallel_tool_calls'}:
            raise ValueError('reexecution requires nonstreaming model-only request')
        if len(json.dumps(self.request, ensure_ascii=False).encode()) > 1048576:
            raise ValueError('request exceeds 1 MiB')
        return self

    def command_hash(self, predecessor_id, actor_user_id, workspace_id):
        return canonical_hash({'schema_version': self.schema_version,
            'predecessor_id': predecessor_id, 'expected_predecessor_request_hash': self.expected_predecessor_request_hash,
            'actor_user_id': actor_user_id, 'workspace_id': workspace_id,
            'reason_code': self.reason_code, 'reason': self.reason, 'request_hash': canonical_hash(self.request),
            **({'expected_profile_revision': self.expected_profile_revision} if self.expected_profile_revision is not None else {})})


class ModelReexecutionControlV1(BaseModel):
    """Trusted producer argument; body is constructed from exact structured material."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    command_id: str = Field(min_length=1, max_length=200, pattern=r'^[\x21-\x7e]+$')
    predecessor_id: str = Field(min_length=1, max_length=200)
    expected_predecessor_request_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    reason_code: Literal['EXPLICIT_REPEAT', 'POST_REPAIR_VALIDATION', 'REVISED_REQUEST']
    reason: str = Field(min_length=1, max_length=1000)
    expected_profile_revision: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        # Preserve historical v1 records/hashes when no revision was frozen.
        value = handler(self)
        if self.expected_profile_revision is None:
            value.pop('expected_profile_revision', None)
        return value

    def with_request(self, body):
        return ModelReexecutionCommandV1(**self.model_dump(exclude={'command_id','predecessor_id'}), request=body)


from datetime import datetime, timedelta
from eos.artifact_hashing import content_hash
from contracts.ontology_building import ArtifactPinV1


class ModelReexecutionRecordV1(BaseModel):
    """Existing workspace artifact owner retains intent and one-hop owner evidence."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['model_reexecution_record.v1'] = 'model_reexecution_record.v1'
    artifact_ref: str
    artifact_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    organization_id: str
    workspace_id: str
    actor_ref: str
    control: ModelReexecutionControlV1
    command_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    request_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    material_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    schema_pin: ArtifactPinV1
    lineage: dict | None = None
    created_at: datetime

    @model_validator(mode='after')
    def verify(self):
        if self.created_at.utcoffset()!=timedelta(0): raise ValueError('aware UTC required')
        expected_command=canonical_hash({'schema_version':'model_reexecution_command.v1',
            'predecessor_id':self.control.predecessor_id,
            'expected_predecessor_request_hash':self.control.expected_predecessor_request_hash,
            'actor_user_id':self.actor_ref.removeprefix('user:'),'workspace_id':self.workspace_id,
            'reason_code':self.control.reason_code,'reason':self.control.reason,'request_hash':self.request_hash,
            **({'expected_profile_revision':self.control.expected_profile_revision} if self.control.expected_profile_revision is not None else {})})
        if self.command_hash!=expected_command: raise ValueError('reexecution command hash mismatch')
        changed=self.request_hash!=self.control.expected_predecessor_request_hash
        if changed!=(self.control.reason_code=='REVISED_REQUEST'): raise ValueError('reexecution request relation mismatch')
        if self.lineage is not None:
            link=self.lineage
            row=link.get('relation',{})
            if link.get('schema_version')!='model_reexecution_lineage.v1' or link.get('sha256')!=canonical_hash(row):
                raise ValueError('lineage hash mismatch')
            if (row.get('reason_code'),row.get('reason'),row.get('request_relation'))!=(self.control.reason_code,self.control.reason,'CHANGED_REQUEST' if changed else 'SAME_REQUEST'):
                raise ValueError('lineage reason/relation mismatch')
            if (row.get('organization_id'),row.get('workspace_id'),row.get('actor_user_id'),row.get('command_id'),row.get('command_hash'),row.get('request_hash'),row.get('predecessor_invocation_id'),row.get('predecessor_request_hash'))!=(self.organization_id,self.workspace_id,self.actor_ref.removeprefix('user:'),self.control.command_id,self.command_hash,self.request_hash,self.control.predecessor_id,self.control.expected_predecessor_request_hash):
                raise ValueError('lineage scope mismatch')
        digest=content_hash(self.model_dump(mode='json',exclude={'artifact_ref','artifact_hash'}))
        if (self.artifact_ref,self.artifact_hash)!=('model-reexecution:'+digest,'sha256:'+digest):
            raise ValueError('reexecution record hash mismatch')
        return self


# FinOps selectors are correlation only. Financial/identity/physical fields are
# never accepted here. Allocation comes from the billing policy owner.
TASK_HEADERS = {
    'task_id':'x-semantier-task-id', 'root_run_id':'x-semantier-root-run-id',
    'parent_run_id':'x-semantier-parent-run-id', 'run_id':'x-semantier-run-id',
    'session_id':'x-semantier-session-id', 'purpose':'x-semantier-purpose',
    'business_object_ref':'x-semantier-business-object-ref',
}


def task_context(actor, account_id, currency, correlation):
    import re
    if not isinstance(correlation,dict) or set(correlation)-set(TASK_HEADERS):
        raise ValueError('unsupported task correlation fields')
    if not correlation:return {}
    if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,199}',v) for v in correlation.values()):
        raise ValueError('bounded opaque ASCII task references required')
    context=dict(correlation)
    root=context.get('root_run_id') or context.get('parent_run_id') or context.get('run_id') or context.get('task_id') or context.get('session_id')
    namespace=canonical_hash([actor.organization_id,account_id,currency,actor.user_id,actor.workspace_id,getattr(actor,'client_namespace',None)])
    return {**context,'schema_version':'semantier.task-consumption.v1','root_run_id':root,
            'actor_user_id':actor.user_id,'workspace_id':actor.workspace_id,
            'namespace':namespace,'scope_id':canonical_hash([namespace,root]) if root else None,
            'provenance':'CLIENT_CORRELATION','allocation_status':'UNALLOCATED'}
