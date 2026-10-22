"""Versioned read projections; these types grant no evaluation authority."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from contracts.artifact_pin import ArtifactPinV1

class ProgressObjectV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    object_id: str
    kind: Literal['output', 'revision', 'path', 'run', 'contribution']
    availability: Literal['available', 'unassessed', 'unavailable']
    native_verdict: str | None = None
    reason: str
    source_pins: list[ArtifactPinV1] = Field(default_factory=list)

class ProgressRelationV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source: str
    target: str
    kind: Literal['contains', 'revises', 'supports', 'contributes', 'precedes']
    source_pin: ArtifactPinV1

class ArtifactExcerptV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_pin: ArtifactPinV1
    json_pointer: str = Field(pattern=r'^/')
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)
    exact_text: str
    excerpt_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')

    @model_validator(mode='after')
    def exact_bytes(self):
        from hashlib import sha256
        raw = self.exact_text.encode('utf-8')
        if self.end_byte - self.start_byte != len(raw) or self.excerpt_hash != 'sha256:'+sha256(raw).hexdigest():
            raise ValueError('excerpt byte identity mismatch')
        return self


class EvaluationProgressV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['evaluation_progress.v1'] = 'evaluation_progress.v1'
    source_id: str
    source_owner: Literal['canonical', 'diagnostic_archive']
    organization_id: str | None = None
    workspace_id: str | None = None
    produced_at: str | None = None
    source_pins: list[ArtifactPinV1]
    objects: list[ProgressObjectV1]
    relations: list[ProgressRelationV1] = Field(default_factory=list)
    source_excerpts: list[ArtifactExcerptV1] = Field(default_factory=list)
    source_fields: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int | None]
    metric_coverage: list[dict[str, Any]]
    source_readiness: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str]
    content_hash: str = ''

    @model_validator(mode='after')
    def identities(self):
        from datetime import datetime
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)): raise ValueError('duplicate object IDs')
        pins = {(p.artifact_ref, p.artifact_hash) for p in self.source_pins}
        for excerpt in self.source_excerpts:
            if (excerpt.source_pin.artifact_ref, excerpt.source_pin.artifact_hash) not in pins: raise ValueError('excerpt source pin mismatch')
        for rel in self.relations:
            if rel.source not in ids or rel.target not in ids: raise ValueError('partial relation endpoint')
        if self.produced_at:
            dt = datetime.fromisoformat(self.produced_at.replace('Z','+00:00'))
            if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0: raise ValueError('produced_at requires UTC')
        return self
