"""Canonical, non-compensatory ontology V&V evidence, never admission authority."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import content_hash
from contracts.artifact_pin import ArtifactPinV1
from contracts.validation_contract import SourceRequirementAssessmentV1
from contracts.source_consumer_verification import ConsumerVerificationV1
from contracts.normative_conflict import (
    NormativeSetV1, NormativeCompatibilityResultV1, ConflictCoreV1, ConflictResolutionV1,
)

NormativeEvidence = Annotated[
    NormativeSetV1 | NormativeCompatibilityResultV1 | ConflictCoreV1 | ConflictResolutionV1,
    Field(discriminator='schema_version'),
]


def normative_evidence_pin(artifact):
    kind = artifact.schema_version.split('.')[0]
    ref, hash_ = {
        'normative_set': ('normative_set_ref', 'normative_set_hash'),
        'normative_compatibility_result': ('result_ref', 'result_hash'),
        'conflict_core': ('conflict_core_ref', 'conflict_core_hash'),
        'conflict_resolution': ('resolution_ref', 'resolution_hash'),
    }[kind]
    return ArtifactPinV1(artifact_kind=kind, artifact_ref=getattr(artifact, ref), artifact_hash=getattr(artifact, hash_))



SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

QualityGate = Literal['LOGIC', 'CONSTRAINT', 'MODELING', 'FITNESS', 'GROUNDING', 'REPLAY']
QualitySeverity = Literal['BLOCK', 'WARN', 'INFO']
QualityRequirementApplicability = Literal['REQUIRED', 'NOT_APPLICABLE', 'UNRESOLVED_APPLICABILITY']
QualityRequirementStatus = Literal['PASS', 'WARN', 'BLOCK', 'NOT_EVALUATED', 'NOT_APPLICABLE']
QualityDisposition = Literal['QUALIFIED', 'QUALIFIED_WITH_WARNINGS', 'NOT_QUALIFIED', 'NOT_EVALUABLE']
_ID = r'^[A-Za-z][A-Za-z0-9_.:-]*$'


class _Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, revalidate_instances='always')


class _Addressed(_Contract):
    REF: ClassVar[str]
    HASH: ClassVar[str]
    KIND: ClassVar[str]

    @model_validator(mode='after')
    def exact_identity(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError(f'{name} must be timezone-aware UTC')
        digest = content_hash(self.model_dump(mode='json', exclude={self.REF, self.HASH}))
        if (getattr(self, self.REF), getattr(self, self.HASH)) != (
            self.KIND.replace('_', '-') + ':' + digest, 'sha256:' + digest,
        ):
            raise ValueError('quality artifact identity must bind exact content')
        return self

    @classmethod
    def create(cls, **values):
        probe = cls.model_construct(**{**values, cls.REF: 'pending', cls.HASH: 'sha256:' + '0' * 64})
        digest = content_hash(probe.model_dump(mode='json', exclude={cls.REF, cls.HASH}, warnings=False))
        return cls(**{**values, cls.REF: cls.KIND.replace('_', '-') + ':' + digest, cls.HASH: 'sha256:' + digest})

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind=self.KIND, artifact_ref=getattr(self, self.REF),
                             artifact_hash=getattr(self, self.HASH))


def _ordered_unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f'{label} must be unique')
    if tuple(values) != tuple(sorted(values)):
        raise ValueError(f'{label} must use deterministic order')


class QualityRequirementV1(_Contract):
    requirement_id: str = Field(pattern=_ID)
    gate: QualityGate
    required: bool = True
    # Applicability is independent of mandatory policy. Historical profiles omit
    # the default field byte-for-byte; governed successors may explicitly record N/A.
    applicability: QualityRequirementApplicability = Field(default='REQUIRED', exclude_if=lambda value: value == 'REQUIRED')
    applicability_rationale: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    applicability_evidence_refs: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    # An unavailable evaluator is never silently successful. A frozen profile
    # may deliberately classify its absence as a blocking policy failure.
    unavailable_status: Literal['NOT_EVALUATED', 'BLOCK'] = 'NOT_EVALUATED'

    @model_validator(mode='after')
    def applicability_is_explicit(self):
        if self.applicability == 'REQUIRED':
            if self.applicability_rationale is not None or self.applicability_evidence_refs:
                raise ValueError('required quality requirement cannot carry exclusion evidence')
        elif not (self.applicability_rationale or '').strip() or not self.applicability_evidence_refs:
            raise ValueError('non-required applicability state requires rationale and evidence')
        if any(not ref.strip() for ref in self.applicability_evidence_refs):
            raise ValueError('quality applicability evidence refs must not be blank')
        _ordered_unique(self.applicability_evidence_refs, 'quality applicability evidence refs')
        return self


class QualityRequirementApplicabilityEntryV1(_Contract):
    requirement_id: str = Field(pattern=_ID)
    state: QualityRequirementApplicability
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def deterministic_evidence(self):
        if not self.rationale.strip() or any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError('quality applicability review requires nonblank rationale and evidence')
        _ordered_unique(self.evidence_refs, 'quality applicability entry evidence refs')
        return self


class QualityPrecedenceExpectationV1(_Contract):
    """Source-reviewed branch expectation; no added observations or runtime oracle."""
    test_id: str = Field(min_length=1)
    rule_ref: str = Field(min_length=1)
    relation_ref: str = Field(min_length=1)
    relation_kind: Literal['exception', 'precedence']
    controlling_ref: str = Field(min_length=1)
    expected_branch_outcome: Literal['MATCH', 'NO_MATCH', 'UNKNOWN']
    expected_rule_outcome: Literal['MATCH', 'NO_MATCH', 'UNKNOWN']
    expected_predicate_results: dict[str, Literal['SATISFIED', 'FAILED', 'UNKNOWN', 'UNSUPPORTED']]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode='after')
    def explicit_review(self):
        if not self.rationale.strip() or any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError('precedence expectation requires source review')
        _ordered_unique(self.evidence_refs, 'precedence expectation evidence')
        return self


class QualityProfileApplicabilityReviewV1(_Addressed):
    REF = 'review_ref'
    HASH = 'review_hash'
    KIND = 'quality_profile_applicability_review'
    schema_version: Literal['quality_profile_applicability_review.v1'] = 'quality_profile_applicability_review.v1'
    review_ref: str
    review_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)
    request_id: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    reviewer_kind: Literal['human', 'delegated_ai'] | None = Field(default=None, exclude_if=lambda value: value is None)
    predecessor_profile: ArtifactPinV1
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    competency: ArtifactPinV1
    candidate: ArtifactPinV1
    validation: ArtifactPinV1
    entries: tuple[QualityRequirementApplicabilityEntryV1, ...] = Field(min_length=1)
    conflict_audit_hash: str = Field(pattern=SHA256_PATTERN)
    conflict_audit: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    precedence_expectations: tuple[QualityPrecedenceExpectationV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    reviewed_at: datetime

    @model_validator(mode='after')
    def exact_review_basis(self):
        _ordered_unique(tuple(item.requirement_id for item in self.entries), 'quality applicability requirement IDs')
        _ordered_unique(tuple((e.test_id, e.rule_ref, e.relation_ref) for e in self.precedence_expectations),
                        'quality precedence expectation keys')
        if self.predecessor_profile.artifact_kind != 'quality_profile':
            raise ValueError('quality applicability predecessor kind mismatch')
        if self.competency.artifact_kind != 'competency_contract' or self.candidate.artifact_kind != 'normative_semantic_candidate_set' or self.validation.artifact_kind != 'validation_contract':
            raise ValueError('quality applicability basis pin kind mismatch')
        return self


class QualityProfileV1(_Addressed):
    REF = 'profile_ref'
    HASH = 'profile_hash'
    KIND = 'quality_profile'
    schema_version: Literal['quality_profile.v1'] = 'quality_profile.v1'
    profile_ref: str
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    name: str = Field(pattern=_ID)
    requirements: tuple[QualityRequirementV1, ...] = Field(min_length=1)
    allow_warnings: bool = False
    predecessor_profile: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    applicability_review: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode='after')
    def exact_requirements(self):
        _ordered_unique(tuple(item.requirement_id for item in self.requirements), 'profile requirement IDs')
        if (self.predecessor_profile is None) != (self.applicability_review is None):
            raise ValueError('quality profile successor pins must be paired')
        if self.predecessor_profile is None and any(
            item.applicability != 'REQUIRED' for item in self.requirements
        ):
            raise ValueError('quality applicability exclusions require governed successor pins')
        if self.predecessor_profile is not None:
            if self.predecessor_profile.artifact_kind != 'quality_profile' or self.applicability_review.artifact_kind != 'quality_profile_applicability_review':
                raise ValueError('quality profile successor pin kind mismatch')
        return self


class QualityFindingV1(_Addressed):
    REF = 'finding_ref'
    HASH = 'finding_hash'
    KIND = 'quality_finding'
    schema_version: Literal['quality_finding.v1'] = 'quality_finding.v1'
    finding_ref: str
    finding_hash: str = Field(pattern=SHA256_PATTERN)
    requirement_id: str = Field(pattern=_ID)
    gate: QualityGate
    severity: QualitySeverity
    code: str = Field(pattern=_ID)
    message: str = Field(min_length=1)
    evidence_refs: tuple[str, ...]
    subject_refs: tuple[str, ...] = ()

    @model_validator(mode='after')
    def exact_references(self):
        for name in ('evidence_refs', 'subject_refs'):
            values = getattr(self, name)
            if any(not value for value in values):
                raise ValueError('finding references must not be empty strings')
            _ordered_unique(values, name)
        return self


class QualityRequirementResultV1(_Addressed):
    REF = 'result_ref'
    HASH = 'result_hash'
    KIND = 'quality_requirement_result'
    schema_version: Literal['quality_requirement_result.v1'] = 'quality_requirement_result.v1'
    result_ref: str
    result_hash: str = Field(pattern=SHA256_PATTERN)
    requirement_id: str = Field(pattern=_ID)
    gate: QualityGate
    quality_profile: ArtifactPinV1
    subject: ArtifactPinV1
    evaluator: ArtifactPinV1
    status: QualityRequirementStatus
    findings: tuple[QualityFindingV1, ...] = ()
    unavailable_reason: str | None = Field(default=None, min_length=1)
    evidence: tuple[ArtifactPinV1, ...] = ()
    source_requirement_assessments: tuple[SourceRequirementAssessmentV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    consumer_verification: ConsumerVerificationV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    # Retained existing engine artifacts, owned by the FITNESS conflict result.
    normative_evidence: tuple[NormativeEvidence, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode='after')
    def attributable_status(self):
        if self.consumer_verification is not None:
            v = self.consumer_verification
            if (self.gate, self.requirement_id) != ('FITNESS', 'source_consumer_binding') or self.subject != v.plan:
                raise ValueError('consumer evidence belongs to its exact plan verification result')
            full = all(a.coverage == 'FULL' for a in v.requirement_assessments)
            expected = 'NOT_EVALUATED' if v.eligibility == 'UNRESOLVED' else 'PASS' if full else 'BLOCK'
            if self.status != expected:
                raise ValueError('consumer result must preserve typed coverage')
        if self.quality_profile.artifact_kind != 'quality_profile':
            raise ValueError('result profile pin kind mismatch')
        _ordered_unique(tuple(item.finding_ref for item in self.findings), 'finding refs')
        _ordered_unique(tuple((item.artifact_kind, item.artifact_ref, item.artifact_hash)
                              for item in self.evidence), 'result evidence pins')
        if any((finding.requirement_id, finding.gate) != (self.requirement_id, self.gate)
               for finding in self.findings):
            raise ValueError('finding must bind the exact requirement and gate')
        if self.normative_evidence:
            if (self.gate, self.requirement_id) != ('FITNESS', 'conflict_detection'):
                raise ValueError('normative evidence belongs to the conflict detection result')
            pins = tuple(normative_evidence_pin(item) for item in self.normative_evidence)
            _ordered_unique(tuple((p.artifact_kind, p.artifact_ref, p.artifact_hash) for p in pins), 'normative evidence')
            if any(pin not in self.evidence for pin in pins):
                raise ValueError('retained normative evidence requires exact evidence pins')
        severities = {finding.severity for finding in self.findings}
        if self.status == 'NOT_APPLICABLE':
            if self.findings or self.unavailable_reason or not self.evidence:
                raise ValueError('not-applicable quality result requires retained applicability evidence only')
        elif self.status == 'NOT_EVALUATED':
            if not self.unavailable_reason:
                raise ValueError('unavailable evaluator requires an explicit reason')
            if 'BLOCK' in severities:
                raise ValueError('blocking findings require BLOCK status')
        else:
            expected = 'BLOCK' if 'BLOCK' in severities else 'WARN' if 'WARN' in severities else 'PASS'
            if self.status != expected:
                raise ValueError('result status must preserve finding severity')
            if self.unavailable_reason and self.status != 'BLOCK':
                raise ValueError('unavailable evaluator cannot produce a successful status')
        return self


class QualityEvaluationRunV1(_Addressed):
    REF = 'run_ref'
    HASH = 'run_hash'
    KIND = 'quality_evaluation_run'
    schema_version: Literal['quality_evaluation_run.v1'] = 'quality_evaluation_run.v1'
    run_ref: str
    run_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    quality_profile: ArtifactPinV1
    # Retain the pinned definition to validate coverage during offline reads.
    profile_definition: QualityProfileV1
    subject: ArtifactPinV1
    competency: ArtifactPinV1
    validation: ArtifactPinV1
    results: tuple[QualityRequirementResultV1, ...]
    disposition: QualityDisposition
    executed_at: datetime

    @staticmethod
    def disposition_for(profile, results) -> QualityDisposition:
        if any(item.applicability == 'UNRESOLVED_APPLICABILITY' for item in profile.requirements):
            return 'NOT_EVALUABLE'
        if any(item.status == 'BLOCK' for item in results):
            return 'NOT_QUALIFIED'
        mandatory = {item.requirement_id for item in profile.requirements if item.required}
        if any(item.requirement_id in mandatory and item.status == 'NOT_EVALUATED' for item in results):
            return 'NOT_EVALUABLE'
        if any(item.status == 'WARN' for item in results):
            return 'QUALIFIED_WITH_WARNINGS'
        return 'QUALIFIED'

    @classmethod
    def create(cls, **values):
        profile = QualityProfileV1.model_validate(values['profile_definition'])
        results = tuple(QualityRequirementResultV1.model_validate(item) for item in values['results'])
        values = {**values, 'profile_definition': profile, 'results': results}
        values.setdefault('disposition', cls.disposition_for(profile, results))
        return super().create(**values)

    @model_validator(mode='after')
    def exact_coverage_and_disposition(self):
        if (self.organization_id, self.workspace_id) != (
            self.profile_definition.organization_id, self.profile_definition.workspace_id,
        ):
            raise ValueError('run profile tenant scope mismatch')
        if self.quality_profile != self.profile_definition.pin():
            raise ValueError('run profile pin mismatch')
        expected = tuple((item.requirement_id, item.gate) for item in self.profile_definition.requirements)
        actual = tuple((item.requirement_id, item.gate) for item in self.results)
        if actual != expected:
            raise ValueError('result coverage must exactly match ordered profile requirements')
        for requirement, result in zip(self.profile_definition.requirements, self.results, strict=True):
            if result.consumer_verification is not None and (result.consumer_verification.organization_id,
                result.consumer_verification.workspace_id) != (self.organization_id, self.workspace_id):
                raise ValueError('consumer evidence tenant scope mismatch')
            for artifact in result.normative_evidence:
                if isinstance(artifact, NormativeSetV1) and (artifact.organization_id, artifact.workspace_id) != (self.organization_id, self.workspace_id):
                    raise ValueError('normative evidence tenant scope mismatch')
            if result.subject != self.subject or result.quality_profile != self.quality_profile:
                raise ValueError('result subject/profile pin mismatch')
            if requirement.applicability == 'NOT_APPLICABLE':
                if result.status not in {'NOT_APPLICABLE', 'BLOCK', 'NOT_EVALUATED'}:
                    raise ValueError('not-applicable requirement requires explicit not-applicable result')
            elif result.status == 'NOT_APPLICABLE':
                raise ValueError('required or unresolved applicability cannot be reported as not applicable')
            if result.unavailable_reason and result.status != requirement.unavailable_status:
                raise ValueError('unavailable result must follow the frozen profile policy')
        if self.disposition != self.disposition_for(self.profile_definition, self.results):
            raise ValueError('disposition cannot compensate for blocked or missing evaluation')
        return self

    def require_basis(self, *, quality_profile, subject, competency, validation):
        """Consumers verify the exact independently selected basis; never reevaluate."""
        if (self.quality_profile, self.subject, self.competency, self.validation) != (
            quality_profile, subject, competency, validation,
        ):
            raise ValueError('quality evaluation basis pin mismatch')


class QualityRequirementDeltaV1(_Addressed):
    REF = 'delta_ref'
    HASH = 'delta_hash'
    KIND = 'quality_requirement_delta'
    schema_version: Literal['quality_requirement_delta.v1'] = 'quality_requirement_delta.v1'
    delta_ref: str
    delta_hash: str = Field(pattern=SHA256_PATTERN)
    requirement_id: str = Field(pattern=_ID)
    gate: QualityGate
    before_run: ArtifactPinV1 | None
    after_run: ArtifactPinV1 | None
    before_result: ArtifactPinV1 | None
    after_result: ArtifactPinV1 | None
    before_status: QualityRequirementStatus
    after_status: QualityRequirementStatus

    @model_validator(mode='after')
    def attributable_transition(self):
        for side in ('before', 'after'):
            run = getattr(self, side + '_run')
            result = getattr(self, side + '_result')
            status = getattr(self, side + '_status')
            if result is None:
                if status != 'NOT_EVALUATED':
                    raise ValueError('absent result must be represented as NOT_EVALUATED')
            elif run is None:
                raise ValueError('result requires its exact evaluation run')
            if run is not None and run.artifact_kind != 'quality_evaluation_run':
                raise ValueError('delta evaluation run pin kind mismatch')
            if result is not None and result.artifact_kind != 'quality_requirement_result':
                raise ValueError('delta result pin kind mismatch')
        return self
