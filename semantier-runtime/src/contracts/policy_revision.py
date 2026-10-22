"""Immutable DEVELOPMENT publication/build/eval observations, never authority."""
from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator
from eos.artifact_hashing import content_hash


class PolicyRevisionRecordV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['policy_revision_record.v1'] = 'policy_revision_record.v1'
    organization_id: str
    workspace_id: str
    artifact_ref: str
    artifact_hash: str
    kind: Literal['source', 'draft', 'publication', 'build', 'eval_run', 'eval_report', 'native_extraction', 'consumer_candidate', 'consumer_run']
    actor_ref: str
    produced_at: datetime
    development_only: Literal[True] = True
    authority_granted: Literal[False] = False
    data: dict

    @model_validator(mode='after')
    def immutable_identity(self):
        if self.produced_at.utcoffset() != timedelta(0):
            raise ValueError('POLICY_UTC_REQUIRED')
        digest = content_hash(self.model_dump(mode='json', exclude={'artifact_ref', 'artifact_hash'}))
        if self.artifact_ref != 'policy-revision:' + digest or self.artifact_hash != 'sha256:' + digest:
            raise ValueError('POLICY_RECORD_HASH_MISMATCH')
        if self.kind in ('source', 'draft'):
            from hashlib import sha256
            if self.data['source_hash'] != 'sha256:' + sha256(self.data['markdown'].encode()).hexdigest():
                raise ValueError('POLICY_SOURCE_HASH_MISMATCH')
        return self

    @classmethod
    def create(cls, **basis):
        basis = dict(schema_version='policy_revision_record.v1', development_only=True, authority_granted=False, **basis)
        # Canonical hash uses serialized timestamps, matching persistence validation.
        basis['produced_at'] = basis['produced_at'].isoformat().replace('+00:00', 'Z')
        digest = content_hash(basis)
        return cls(artifact_ref='policy-revision:' + digest, artifact_hash='sha256:' + digest, **basis)
