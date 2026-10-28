"""Semantier release/feedback v1 untrusted boundary. No financial authority."""
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = '1.2.0'
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
WindowsVersion = Annotated[str, Field(pattern=r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')]
GRANTS = frozenset({'release_read','release_manage','release_ingest','diagnostic_read','diagnostic_triage'})
MAX_DIAGNOSTIC_BYTES = 5 * 1024 * 1024


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)

    @model_validator(mode='before')
    @classmethod
    def text_only(cls,value):
        def inspect(item):
            if isinstance(item,str) and any(ord(c)<32 and c not in '\n\r\t' for c in item):raise ValueError('unsupported control character')
            if isinstance(item,dict):
                for k,v in item.items():inspect(k);inspect(v)
            elif isinstance(item,list):
                for v in item:inspect(v)
        inspect(value)
        return value


class UpdateEvent(Strict):
    schema_version: Literal['1.0.0']
    event_id: Identifier
    update_id: Identifier
    installation_ref: Identifier
    release_id: Identifier
    target_sha256: Digest
    from_version: WindowsVersion
    target_version: WindowsVersion
    package_family: Annotated[str,Field(min_length=1,max_length=160)]
    stage: Literal['check','download','verify','handoff','restart_confirmation']
    outcome: Literal['pending','unknown','failed','target_confirmed']
    error_code: Identifier | None = None
    occurred_at: str
    observed_version: WindowsVersion | None = None
    observed_family: str | None = None
    observed_sha256: Digest | None = None

    @field_validator('occurred_at')
    @classmethod
    def aware(cls,value):
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError('aware UTC timestamp required')
        return parsed.astimezone(timezone.utc).isoformat()

    @model_validator(mode='after')
    def target_binding(self):
        if self.outcome=='target_confirmed' and (self.stage!='restart_confirmation' or
            self.observed_version!=self.target_version or self.observed_family!=self.package_family or
            self.observed_sha256!=self.target_sha256):
            raise ValueError('exact verified target confirmation required')
        return self


class EventBatch(Strict):
    events: Annotated[list[UpdateEvent],Field(min_length=1,max_length=100)]


class Promotion(Strict):
    release_id: Identifier
    expected_revision: Annotated[int,Field(ge=0)]
    idempotency_key: Identifier


class ChannelControl(Strict):
    expected_revision: Annotated[int,Field(ge=0)]
    idempotency_key: Identifier


class Diagnostic(Strict):
    schema_version: Literal['1.0.0']
    submission_key: Identifier
    app_version: WindowsVersion
    release_id: Identifier | None = None
    update_id: Identifier | None = None
    error_code: Identifier | None = None
    platform: Literal['win32']
    note: Annotated[str,Field(max_length=4000)] = ''
    sections: Annotated[dict[str,str],Field(max_length=8)]

    @field_validator('sections')
    @classmethod
    def allowed(cls,value):
        if not set(value)<={'app','runtime','update'}:raise ValueError('unsupported diagnostic section')
        return value


class Classification(Strict):
    expected_revision: Annotated[int,Field(ge=0)]
    category: Literal['unclassified','startup','update','provider','ui','other']
    severity: Literal['unassessed','low','medium','high']
    status: Literal['new','triaged','resolved']
    fix_release_id: Identifier | None = None
    operator_note: Annotated[str,Field(max_length=4000)] = ''


# Deliberately removes whole sensitive lines/content sections rather than trying
# to infer which fragment of a conversation or tool result might be harmless.
_REDACTIONS = (
    (r'(?im)^.*(?:authorization|cookie|password|api[_-]?key|access[_-]?token|refresh[_-]?token|secret)\s*[:=].*$', '[redacted credential line]'),
    (r'(?im)^.*(?:prompt|conversation|tool[_ ]?(?:result|output)|document[_ ]?content|user[_ ]?(?:id|email))\s*[:=].*$', '[excluded content line]'),
    (r'(?i)\b(?:sk-|smg_|sco_)[A-Za-z0-9_-]{8,}', '[redacted credential]'),
    (r'(?i)[A-Z]:\\Users\\[^\s]+', '[redacted user path]'),
    (r'/(?:home|Users)/[^\s]+', '[redacted user path]'),
    (r'(?i)(https?://[^\s?#]+)\?[^\s]*', r'\1?[redacted]'),
    (r'\b[^\s@]+@[^\s@]+\.[A-Za-z]{2,}\b', '[redacted email]'),
)


def sanitize(value):
    count=0
    for pattern,replacement in _REDACTIONS:
        value,n=re.subn(pattern,replacement,value);count+=n
    return value,count


def summarize_attempts(outcomes):
    values=list(outcomes);success=values.count('target_confirmed');failed=values.count('failed');finalized=success+failed
    return dict(observed=len(values),target_confirmed=success,failed=failed,
                pending_unknown=len(values)-finalized,finalized=finalized,
                success_percent=100*success/finalized if finalized else None)


class ReleaseCandidate(Strict):
    release_id: Identifier
    manifest: dict
    notes: Annotated[dict[str,str],Field(min_length=1,max_length=9)]
    native_receipt: dict
    receipt_hmac: Digest


class FeedbackAuthorization(Strict):
    intent_ref: Identifier
    proof_challenge: Digest
    purpose: Literal['diagnostic','reporting']
    profile_ref: Identifier


class FeedbackProof(Strict):
    proof: Annotated[str,Field(min_length=32,max_length=128,pattern=r'^[A-Za-z0-9_-]+$')]


class FeedbackDecision(Strict):
    decision: Literal['approve','deny']


DeliveryTargetId = Literal['channel1', 'channel2', 'channel3']
DELIVERY_TARGET_IDS = ('channel1', 'channel2', 'channel3')
DELIVERY_ACTIONS = frozenset({'read', 'publish', 'ingest'})


class PublicationIntent(Strict):
    release_id: Identifier
    channel: Identifier
    delivery_target_ids: Annotated[list[DeliveryTargetId], Field(min_length=1, max_length=3)]
    target_bindings: dict[DeliveryTargetId, Digest]
    expected_revisions: dict[DeliveryTargetId, Annotated[int, Field(ge=1)]]
    preview_plan_digests: dict[DeliveryTargetId, Digest]
    idempotency_key: Identifier

    @field_validator('delivery_target_ids')
    @classmethod
    def normalized_targets(cls, value):
        if len(value) != len(set(value)):
            raise ValueError('duplicate delivery target')
        return sorted(value)

    @model_validator(mode='after')
    def captured_maps(self):
        selected = set(self.delivery_target_ids)
        if any(set(value) != selected for value in
               (self.target_bindings, self.expected_revisions, self.preview_plan_digests)):
            raise ValueError('target maps must exactly match captured selection')
        return self


class PublicationRetry(Strict):
    delivery_target_ids: Annotated[list[DeliveryTargetId], Field(min_length=1, max_length=3)]
    expected_batch_revision: Annotated[int, Field(ge=1)]
    expected_revisions: dict[DeliveryTargetId, Annotated[int, Field(ge=1)]]
    preview_plan_digests: dict[DeliveryTargetId, Digest]
    idempotency_key: Identifier

    @model_validator(mode='after')
    def exact_subset(self):
        if len(set(self.delivery_target_ids)) != len(self.delivery_target_ids):
            raise ValueError('duplicate retry target')
        if any(set(value) != set(self.delivery_target_ids) for value in (self.preview_plan_digests, self.expected_revisions)):
            raise ValueError('retry preview must match exact target subset')
        self.delivery_target_ids = sorted(self.delivery_target_ids)
        return self


class TargetBindingAdmission(Strict):
    delivery_target_id: DeliveryTargetId
    candidate: ReleaseCandidate
    descriptor_receipt: dict
    descriptor_receipt_hmac: Digest
