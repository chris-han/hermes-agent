"""Append-only contextual memory changes; these artifacts grant no authority."""
from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator
from contracts.semantic_kernel import _ContentAddressed


class RetainedContextItemChangeV1(_ContentAddressed):
    ARTIFACT_KIND = 'retained_context_item_change'
    schema_version: Literal['retained_context_item_change.v1'] = 'retained_context_item_change.v1'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    item_ref: str = Field(min_length=1)
    source_file: Literal['MEMORY.md', 'USER.md']
    revision: int = Field(ge=1)
    expected_revision: str
    prior_content: str
    content: str | None
    action: Literal['correct', 'suppress']
    request_id: str = Field(min_length=1)
    recorded_at: datetime
    origin: dict[str, str] | None = None

    @model_validator(mode='after')
    def action_binds_content(self):
        if (self.action == 'suppress') != (self.content is None):
            raise ValueError('action must bind exact correction/suppression content')
        return self


RETAINED_CONTEXT_ARTIFACT_MODELS = {
    'retained_context_item_change.v1': RetainedContextItemChangeV1,
}
