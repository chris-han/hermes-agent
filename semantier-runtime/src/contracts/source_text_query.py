"""Task-local DEVELOPMENT source-text request; never semantic authority."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from services.source_text_query_engine import digest

SAFE = r'^[A-Za-z0-9][A-Za-z0-9._:-]*$'
SHA = r'^[0-9a-f]{64}$'

class SourceTextPin(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    logical_source_id: str = Field(min_length=1)
    normalized_document_ref: str = Field(min_length=1)
    source_identity_ref: str = Field(min_length=1)
    canonical_sha256: str = Field(pattern=SHA)
    text_sha256: str = Field(pattern=SHA)
    raw_sha256: str = Field(pattern=SHA)
    edition: str = Field(min_length=1)

class SourceTextQueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    classification: Literal['DEVELOPMENT'] = 'DEVELOPMENT'
    organization_id: str = Field(pattern=SAFE)
    workspace_id: str = Field(pattern=SAFE)
    request_id: str = Field(pattern=SAFE, max_length=128)
    sources: tuple[SourceTextPin, ...] = Field(min_length=1, max_length=16)
    manifest_sha256: str = Field(pattern=SHA)
    spec: dict
    spec_sha256: str = Field(pattern=SHA)

    @model_validator(mode='after')
    def pins(self):
        if len({s.logical_source_id for s in self.sources}) != len(self.sources):
            raise ValueError('duplicate source mapping')
        if digest([s.model_dump(mode='json') for s in self.sources]) != self.manifest_sha256:
            raise ValueError('manifest hash mismatch')
        if digest(self.spec) != self.spec_sha256:
            raise ValueError('spec hash mismatch')
        return self
