"""Exact retained assessment and materialization evidence; neither grants authority."""
from datetime import datetime
from typing import Literal
from contracts.ontology_building import ArtifactPinV1
from contracts.candidate_admission_classification import CandidateAdmissionClassificationV1
from contracts.semantic_kernel import _ContentAddressed, SHA256_PATTERN
from pydantic import Field, JsonValue, model_validator


class LearnedSkillAssessmentV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_assessment'
    schema_version: Literal['learned_skill_assessment.v1'] = 'learned_skill_assessment.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    version: ArtifactPinV1
    predecessor: ArtifactPinV1
    classification: CandidateAdmissionClassificationV1
    eligible: bool
    counterevidence: tuple[str, ...]
    recorded_at: datetime

    @model_validator(mode='after')
    def binds_routing(self):
        if self.classification.candidate != self.candidate:
            raise ValueError('assessment must bind exact routing candidate')
        if self.eligible != (self.classification.disposition == 'PROCEDURAL_SAFE_TO_PROJECT'):
            raise ValueError('assessment eligibility must preserve routing boundary')
        if self.version.artifact_kind != 'skill_artifact_version' or self.predecessor.artifact_kind != 'skill_artifact_version':
            raise ValueError('assessment must bind exact skill versions')
        return self


class LearnedSkillPublicationReceiptV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_publication_receipt'
    schema_version: Literal['learned_skill_publication_receipt.v1'] = 'learned_skill_publication_receipt.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    version: ArtifactPinV1
    activation_ref: str
    runtime_path: str
    recorded_at: datetime


LEARNED_SKILL_ARTIFACT_MODELS = {
    'learned_skill_assessment.v1': LearnedSkillAssessmentV1,
    'learned_skill_publication_receipt.v1': LearnedSkillPublicationReceiptV1,
}


class LearnedSkillProposalV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_proposal'
    schema_version: Literal['learned_skill_proposal.v1'] = 'learned_skill_proposal.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    version: ArtifactPinV1
    predecessor: ArtifactPinV1
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: dict[str, JsonValue] | None = None
    recorded_at: datetime


LEARNED_SKILL_ARTIFACT_MODELS['learned_skill_proposal.v1'] = LearnedSkillProposalV1


class LearnedSkillSetAssessmentV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_set_assessment'
    schema_version: Literal['learned_skill_set_assessment.v1'] = 'learned_skill_set_assessment.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    member_assessments: tuple[LearnedSkillAssessmentV1, ...]
    eligible: bool
    recorded_at: datetime

    @model_validator(mode='after')
    def exact_members(self):
        if len(self.member_assessments) < 2 or any(a.candidate != self.candidate for a in self.member_assessments) or self.eligible != all(a.eligible for a in self.member_assessments):
            raise ValueError('set assessment requires all exact member assessments')
        return self


class LearnedSkillSetReceiptV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_set_receipt'
    schema_version: Literal['learned_skill_set_receipt.v1'] = 'learned_skill_set_receipt.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    versions: tuple[ArtifactPinV1, ...]
    activation_ref: str
    generation_hash: str
    recorded_at: datetime


LEARNED_SKILL_ARTIFACT_MODELS.update({'learned_skill_set_assessment.v1':LearnedSkillSetAssessmentV1,'learned_skill_set_receipt.v1':LearnedSkillSetReceiptV1})


class LearnedSkillSetProposalV1(_ContentAddressed):
    ARTIFACT_KIND = 'learned_skill_set_proposal'
    schema_version: Literal['learned_skill_set_proposal.v1'] = 'learned_skill_set_proposal.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    request_id: str
    request_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    origin: dict[str, JsonValue]
    recorded_at: datetime


LEARNED_SKILL_ARTIFACT_MODELS['learned_skill_set_proposal.v1'] = LearnedSkillSetProposalV1
