"""Companion observations in the existing artifact owner, never semantic authority."""
from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator
from contracts.semantic_kernel import _ContentAddressed, _Contract
from eos.artifact_hashing import content_hash


class LearningTraceObservationV1(_ContentAddressed):
    ARTIFACT_KIND = 'learning_trace_observation'
    schema_version: Literal['learning_trace_observation.v1'] = 'learning_trace_observation.v1'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    backend: Literal['sqlite', 'postgresql']
    source_store_id: str = Field(min_length=1)
    source_epoch: str | None
    profile_id: str = Field(min_length=1)
    adapter_revision: Literal['hermes-learning-trace.v1'] = 'hermes-learning-trace.v1'
    session_id: str = Field(min_length=1)
    collection_method: Literal['hook', 'poll', 'native_receipt']
    source_locator: str = Field(min_length=1, max_length=500)
    event_kind: Literal['tool', 'request', 'response', 'message', 'approval']
    phase: str = Field(min_length=1, max_length=100)
    native_occurrence_id: str | None
    occurrence_id: str | None
    payload: dict
    source_occurred_at: datetime | None
    captured_at: datetime

    @model_validator(mode='after')
    def bound_occurrence(self):
        key = None
        if self.source_epoch and self.native_occurrence_id:
            key = 'sha256:' + content_hash([self.organization_id, self.workspace_id,
                self.owner_ref, self.backend, self.source_store_id, self.source_epoch,
                self.profile_id, self.session_id, self.event_kind, self.phase,
                self.native_occurrence_id])
        if self.occurrence_id != key:
            raise ValueError('TRACE_OCCURRENCE_ID_MISMATCH')
        return self


class ActionableLearningLessonV1(_Contract):
    """Structured draft metadata for existing Memory/Skill writers; not execution."""
    schema_version: Literal['actionable_learning_lesson.v1'] = 'actionable_learning_lesson.v1'
    target_ref: str = Field(min_length=1)
    target_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    content_role: Literal['memory', 'knowledge']
    capabilities: tuple[Literal['extraction', 'organization', 'update_invalidation',
                                'retrieval_assembly', 'utility_verification'], ...]
    applicability: str = Field(min_length=1)
    exclusions: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    verification_method: str = Field(min_length=1)
    rollback: str = Field(min_length=1)
    evidence_refs: tuple[dict, ...] = Field(min_length=1)
    counter_evidence_refs: tuple[dict, ...] = ()
    change_kind: Literal['knowledge_content', 'maintenance_policy', 'retrieval_assembly_policy', 'unspecified'] = 'unspecified'
    utility: Literal['unmeasured'] = 'unmeasured'


LEARNING_TRACE_ARTIFACT_MODELS = {'learning_trace_observation.v1': LearningTraceObservationV1}
