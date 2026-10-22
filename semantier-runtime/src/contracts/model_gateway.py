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
                         'response_format', 'reasoning_effort', 'presence_penalty', 'frequency_penalty', 'seed', 'stream_options'})


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

    def __post_init__(self):
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
