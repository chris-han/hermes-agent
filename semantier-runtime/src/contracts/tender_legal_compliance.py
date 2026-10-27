"""Frozen derived Tender compliance inputs, separate from screening and opinions.

Structural binding is not legal qualification. The production verifier must resolve
retained owner records before publishing a scoped EVALUATION DecisionRecord.
"""
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from contracts.evidence_location import EvidenceRef, SourceIdentity
from contracts.governed_frame_finding_v1 import GovernedFrameFindingV1
from contracts.normative_schema import NormativeSchemaContractV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import DecisionRecord, _ContentAddressed
from contracts.ontology_certification import NormativeAuthorityDependencySetV1
from contracts.ontology_paired_evaluation import ObservedFactsV1
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.tender_observed_facts_projection import TenderObservedFactsProjectionReceiptV1
from eos.artifact_hashing import content_hash
from contracts.tender_owner_adoption import TenderOwnerAdoptionProposalV1, TenderOwnerAdoptionV1, validate_frozen

SHA = r'^sha256:[0-9a-f]{64}$'


class TenderNormativeProvisionV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    version_label: str = Field(min_length=1)
    article_locator: str = Field(min_length=1)
    identity: SourceIdentity
    canonical_source_ir: ArtifactPinV1
    evidence: EvidenceRef

    @model_validator(mode='after')
    def exact_source(self):
        if (self.identity.organization_id, self.identity.workspace_id, self.identity.source_identity_ref) != (
            self.evidence.organization_id, self.evidence.workspace_id, self.evidence.source_identity_ref
        ):
            raise ValueError('NORMATIVE_SOURCE_IDENTITY_MISMATCH')
        if self.canonical_source_ir.artifact_kind != 'canonical_source_ir':
            raise ValueError('NORMATIVE_SOURCE_IR_KIND_MISMATCH')
        if not self.evidence.selector.exact_quote:
            raise ValueError('NORMATIVE_EXACT_QUOTE_REQUIRED')
        return self


class TenderNormativeInputsV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    normative_schema: NormativeSchemaContractV1
    candidate: NormativeSemanticCandidateSetV1
    plan: CompiledRulePlanV1
    rule_id: str = Field(min_length=1, pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$')
    authority: NormativeAuthorityDependencySetV1
    provisions: tuple[TenderNormativeProvisionV1, ...] = Field(min_length=1)
    facts: ObservedFactsV1
    projection: TenderObservedFactsProjectionReceiptV1
    binding_review: ArtifactPinV1

    @model_validator(mode='after')
    def paired_inputs(self):
        scope = (self.plan.organization_id, self.plan.workspace_id)
        for value in (self.normative_schema, self.candidate, self.authority, self.facts, self.projection):
            if (value.organization_id, value.workspace_id) != scope:
                raise ValueError('NORMATIVE_INPUT_SCOPE_MISMATCH')
        if (self.plan.candidate_set_ref, self.plan.candidate_set_hash) != (
            self.candidate.candidate_set_ref, self.candidate.candidate_set_hash
        ) or (self.candidate.normative_schema_ref, self.candidate.normative_schema_hash) != (
            self.normative_schema.schema_ref, self.normative_schema.schema_hash
        ):
            raise ValueError('NORMATIVE_CANDIDATE_PLAN_MISMATCH')
        if len([r for r in self.plan.rules if r.rule_id == self.rule_id]) != 1:
            raise ValueError('EXACT_NORMATIVE_RULE_ID_REQUIRED')
        expected = (
            ('compiled_plan', 'compiled_rule_plan', self.plan.plan_ref, self.plan.plan_hash),
            ('normative_schema', 'normative_schema_contract', self.normative_schema.schema_ref, self.normative_schema.schema_hash),
            ('observed_facts', 'paired_observed_facts', self.facts.facts_ref, self.facts.facts_hash),
        )
        for field, kind, ref, digest in expected:
            pin = getattr(self.projection, field)
            if (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) != (kind, ref, digest):
                raise ValueError('FACT_PROJECTION_BINDING_MISMATCH')
        if self.projection.subject_binding is None:
            raise ValueError('EXACT_CASE_SUBJECT_REQUIRED')
        if self.binding_review.artifact_kind != 'decision_record':
            raise ValueError('BINDING_REVIEW_RECORD_REQUIRED')
        dependencies = {(d.source_id, d.version_id): d for d in self.authority.dependencies}
        for provision in self.provisions:
            dependency = dependencies.get((provision.source_id, provision.version_id))
            if dependency is None or provision.section_id not in {s.section_id for s in dependency.sections}:
                raise ValueError('NORMATIVE_PROVISION_VERSION_MISMATCH')
            if dependency.source_content_hash != provision.identity.source_hash:
                raise ValueError('NORMATIVE_ORIGINAL_CONTENT_HASH_MISMATCH')
            if (provision.identity.organization_id, provision.identity.workspace_id) != scope:
                raise ValueError('NORMATIVE_SOURCE_SCOPE_MISMATCH')
        return self


class TenderOwnerAdvisoryInputsV1(BaseModel):
    """Versioned internal interpretation, with frozen owner commitments.

    Kept separate from independently governed legal authority dependencies.
    """
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['owner_adopted_internal_advisory.v1'] = 'owner_adopted_internal_advisory.v1'
    basis_kind: Literal['OWNER_ADOPTED_INTERNAL_ADVISORY'] = 'OWNER_ADOPTED_INTERNAL_ADVISORY'
    independent_legal_validation: Literal['NOT_ESTABLISHED'] = 'NOT_ESTABLISHED'
    proposal: TenderOwnerAdoptionProposalV1
    adoption: TenderOwnerAdoptionV1
    source_decision: DecisionRecord
    interpretation_decision: DecisionRecord
    normative_schema: NormativeSchemaContractV1
    candidate: NormativeSemanticCandidateSetV1
    plan: CompiledRulePlanV1
    facts: ObservedFactsV1 | None = None
    projection: TenderObservedFactsProjectionReceiptV1 | None = None

    @model_validator(mode='after')
    def exact_adopted_inputs(self):
        validate_frozen(adoption=self.adoption, proposal=self.proposal,
            source_decision=self.source_decision, interpretation_decision=self.interpretation_decision)
        p = self.proposal
        for pin, value, ref, digest in (
            (p.schema_, self.normative_schema, self.normative_schema.schema_ref, self.normative_schema.schema_hash),
            (p.candidate, self.candidate, self.candidate.candidate_set_ref, self.candidate.candidate_set_hash),
            (p.plan, self.plan, self.plan.plan_ref, self.plan.plan_hash)):
            if ((pin.artifact_ref,pin.artifact_hash) != (ref,digest) or
                (value.organization_id,value.workspace_id) != (p.organization_id,p.workspace_id)):
                raise ValueError('ADOPTED_INPUT_PIN_MISMATCH')
        if ((self.plan.candidate_set_ref,self.plan.candidate_set_hash) != (p.candidate.artifact_ref,p.candidate.artifact_hash)
            or (self.candidate.normative_schema_ref,self.candidate.normative_schema_hash) != (p.schema_.artifact_ref,p.schema_.artifact_hash)
            or (self.candidate.source_identity_ref,self.candidate.source_hash) != (p.source_identity.source_identity_ref,p.source_identity.source_hash)
            or (self.candidate.canonical_source_ir_ref,self.candidate.canonical_source_ir_hash) != (p.canonical_source_ir.artifact_ref,p.canonical_source_ir.artifact_hash)
            or len([r for r in self.plan.rules if r.rule_id == p.rule_ref]) != 1):
            raise ValueError('ADOPTED_RULE_SOURCE_MISMATCH')
        if (self.facts is None) != (self.projection is None):
            raise ValueError('ADVISORY_FACT_PROJECTION_REQUIRED_TOGETHER')
        if self.projection is not None:
            if (self.projection.compiled_plan != p.plan or self.projection.normative_schema != p.schema_
                or (self.projection.observed_facts.artifact_ref,self.projection.observed_facts.artifact_hash) != (self.facts.facts_ref,self.facts.facts_hash)
                or (self.facts.organization_id,self.facts.workspace_id) != (p.organization_id,p.workspace_id)):
                raise ValueError('ADVISORY_FACT_PROJECTION_MISMATCH')
        return self


class TenderSourceSubclaimV1(BaseModel):
    """Descriptive source support; this carrier cannot grant legal qualification."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    subclaim_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$')
    evidence: tuple[EvidenceRef, ...]
    source_traceability: Literal['AVAILABLE', 'UNKNOWN']
    rule_temporal_validity: Literal['UNKNOWN'] = 'UNKNOWN'
    case_applicability: Literal['UNKNOWN'] = 'UNKNOWN'
    fact_sufficiency: Literal['UNKNOWN'] = 'UNKNOWN'
    missing_premises: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def support_is_not_truth(self):
        if self.source_traceability != ('AVAILABLE' if self.evidence else 'UNKNOWN'):
            raise ValueError('SOURCE_SUPPORT_EVIDENCE_REQUIRED')
        refs = [e.evidence_ref for e in self.evidence]
        if len(refs) != len(set(refs)):
            raise ValueError('DUPLICATE_SOURCE_OCCURRENCE')
        return self


class TenderSourceDocumentV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    role: Literal['ORIGINAL', 'CORRECTION', 'REPLACEMENT', 'ANNOUNCEMENT', 'RULE']
    identity: SourceIdentity
    canonical_source_ir: ArtifactPinV1
    normalized_document_ref: str
    derivation_input: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)




class TenderSourceCaseBindingV1(_ContentAddressed):
    """Nested frozen source expression, not a standalone authority/store family."""
    ARTIFACT_KIND = 'tender_source_case_binding'
    schema_version: Literal['tender_source_case_binding.v1'] = 'tender_source_case_binding.v1'
    organization_id: str
    workspace_id: str
    prepared_by: str
    case_id: str = Field(min_length=1)
    package_id: int = Field(gt=0)
    event_date: date
    date_precision: Literal['DAY'] = 'DAY'
    ordering: Literal['EXPLICIT_VERSION', 'UNRESOLVED_SAME_DAY']
    recovery_run_id: str
    recovery_hash: str = Field(pattern=SHA)
    documents: tuple[TenderSourceDocumentV1, ...] = Field(min_length=1)
    original_ref: str
    correction_refs: tuple[str, ...]
    replacement_ref: str | None
    operative: Literal['ORIGINAL', 'ORIGINAL_WITH_CORRECTIONS', 'REPLACEMENT']
    precedence_evidence: tuple[EvidenceRef, ...]
    subclaims: tuple[TenderSourceSubclaimV1, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def expression(self):
        docs = {d.identity.source_identity_ref: d for d in self.documents}
        if len(docs) != len(self.documents) or len(set(self.correction_refs)) != len(self.correction_refs):
            raise ValueError('AMBIGUOUS_SOURCE_EXPRESSION')
        expected = [(self.original_ref, 'ORIGINAL'), *((r, 'CORRECTION') for r in self.correction_refs)]
        if self.replacement_ref is not None:
            expected.append((self.replacement_ref, 'REPLACEMENT'))
        for ref, role in expected:
            if ref not in docs or docs[ref].role != role:
                raise ValueError('SOURCE_EXPRESSION_ROLE_MISMATCH')
        if self.operative == 'REPLACEMENT' and self.replacement_ref is None:
            raise ValueError('EXPLICIT_REPLACEMENT_REQUIRED')
        if self.operative == 'ORIGINAL_WITH_CORRECTIONS' and not self.correction_refs:
            raise ValueError('EXPLICIT_CORRECTION_REQUIRED')
        if self.correction_refs and not self.precedence_evidence:
            raise ValueError('AMENDMENT_PRECEDENCE_EVIDENCE_REQUIRED')
        operative_sources = ({self.replacement_ref} if self.operative == 'REPLACEMENT' else
            {self.original_ref, *self.correction_refs} if self.operative == 'ORIGINAL_WITH_CORRECTIONS' else {self.original_ref})
        if any(e.source_identity_ref not in operative_sources for c in self.subclaims for e in c.evidence):
            raise ValueError('SOURCE_CASE_OPERATIVE_VERSION_MISMATCH')
        if len({c.subclaim_id for c in self.subclaims}) != len(self.subclaims):
            raise ValueError('DUPLICATE_SUBCLAIM')
        scope = (self.organization_id, self.workspace_id)
        for d in self.documents:
            if (d.identity.organization_id, d.identity.workspace_id) != scope:
                raise ValueError('SOURCE_CASE_SCOPE_MISMATCH')
            if not d.identity.source_version or d.canonical_source_ir.artifact_kind != 'canonical_source_ir':
                raise ValueError('EXACT_SOURCE_VERSION_REQUIRED')
        for e in (*self.precedence_evidence, *(e for c in self.subclaims for e in c.evidence)):
            if (e.organization_id, e.workspace_id) != scope or e.source_identity_ref not in docs:
                raise ValueError('SOURCE_CASE_SCOPE_MISMATCH')
        from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_pin
        for e in (*self.precedence_evidence, *(e for c in self.subclaims for e in c.evidence)):
            evidence_ref_pin(e)
            if e.provenance.get('source_hash') != docs[e.source_identity_ref].identity.source_hash:
                raise ValueError('SOURCE_CASE_HASH_MISMATCH')
        return self


class TenderLegalComplianceBasisV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['tender_legal_compliance_basis.v1'] = 'tender_legal_compliance_basis.v1'
    basis_ref: str
    basis_hash: str = Field(pattern=SHA)
    organization_id: str
    workspace_id: str
    episode: ArtifactPinV1
    inspection_result: ArtifactPinV1
    finding: GovernedFrameFindingV1
    target_ir: ArtifactPinV1
    normative: TenderNormativeInputsV1 | None
    advisory: TenderOwnerAdvisoryInputsV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    supersedes_assessment: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    gaps: tuple[str, ...]
    source_case_binding: TenderSourceCaseBindingV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    frozen_at: datetime

    @property
    def binding_status(self):
        return 'UNRESOLVED' if (self.normative is None and self.advisory is None) or self.gaps else 'BOUND_INPUTS'

    @model_validator(mode='after')
    def exact_binding(self, info: ValidationInfo):
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() != timedelta(0):
            raise ValueError('UTC_REQUIRED')
        scope = (self.organization_id, self.workspace_id)
        if self.source_case_binding is not None and (self.source_case_binding.organization_id, self.source_case_binding.workspace_id) != scope:
            raise ValueError('SOURCE_CASE_SCOPE_MISMATCH')
        if (self.finding.organization_id, self.finding.workspace_id) != scope:
            raise ValueError('FINDING_SCOPE_MISMATCH')
        for pin, kind in ((self.episode, 'tender_journey_inspection_episode'),
                          (self.inspection_result, 'runtime_semantic_inspection_result.v1'),
                          (self.target_ir, 'canonical_source_ir')):
            if pin.artifact_kind != kind:
                raise ValueError('BASIS_PIN_KIND_MISMATCH')
        if self.supersedes_assessment is not None and self.supersedes_assessment.artifact_kind != "decision_record":
            raise ValueError("ASSESSMENT_SUCCESSOR_PIN_KIND_MISMATCH")
        if self.advisory is not None:
            a = self.advisory
            if self.normative is not None:
                raise ValueError('INDEPENDENT_AND_INTERNAL_BASIS_CANNOT_MIX')
            if ((a.proposal.organization_id,a.proposal.workspace_id) != scope
                or a.proposal.canonical_pattern_ref != self.finding.canonical_pattern_ref
                or a.proposal.canonical_pattern_hash != self.finding.canonical_pattern_hash
                or a.adoption.adopted_at > self.frozen_at):
                raise ValueError('ADVISORY_FINDING_BINDING_MISMATCH')
            if a.projection is not None and a.projection.subject_binding is not None:
                subject = a.projection.subject_binding
                if (subject.source_ref,subject.source_hash,subject.canonical_source_ir) != (self.finding.source_identity_ref,self.finding.source_hash,self.target_ir):
                    raise ValueError('ADVISORY_CASE_SOURCE_MISMATCH')
                if subject.selector_hash not in {e['selectorHash'] for e in self.finding.evidence_refs}:
                    raise ValueError('ADVISORY_CASE_OCCURRENCE_MISMATCH')
        if self.normative is None and self.advisory is None and not self.gaps:
            raise ValueError('UNRESOLVED_BINDING_REQUIRES_NAMED_GAPS')
        if len(self.gaps) != len(set(self.gaps)):
            raise ValueError('DUPLICATE_BINDING_GAP')
        if self.normative is not None:
            n = self.normative
            if (n.plan.organization_id, n.plan.workspace_id) != scope:
                raise ValueError('NORMATIVE_INPUT_SCOPE_MISMATCH')
            subject = n.projection.subject_binding
            if (subject.source_ref, subject.source_hash, subject.canonical_source_ir) != (
                self.finding.source_identity_ref, self.finding.source_hash, self.target_ir
            ):
                raise ValueError('CASE_SUBJECT_SOURCE_MISMATCH')
            if subject.selector_hash not in {e['selectorHash'] for e in self.finding.evidence_refs}:
                raise ValueError('CASE_SUBJECT_OCCURRENCE_MISMATCH')
        digest = content_hash(self.model_dump(mode='json', exclude={'basis_ref', 'basis_hash'}))
        if not (info.context and info.context.get('compliance_draft')) and (self.basis_ref, self.basis_hash) != ('tender-legal-compliance-basis:' + digest, 'sha256:' + digest):
            raise ValueError('COMPLIANCE_BASIS_HASH_MISMATCH')
        return self

    def original_provisions(self):
        """Common frozen original projection without inventing registered law IDs."""
        if self.normative is not None:
            return tuple((p.identity, p.canonical_source_ir, p.evidence,
                dict(title=p.source_title, version=p.version_label, article=p.article_locator))
                for p in self.normative.provisions)
        if self.advisory is not None:
            p = self.advisory.proposal
            return tuple((p.source_identity, p.canonical_source_ir, evidence,
                dict(title=p.source_identity.original_name,
                    version=p.legal_edition or 'LEGAL_EDITION_NOT_ESTABLISHED',
                    article=evidence.selector.source_element_ref or p.rule_ref))
                for evidence in p.source_evidence)
        return ()

    def pin(self):
        return ArtifactPinV1(artifact_kind='tender_legal_compliance_basis', artifact_ref=self.basis_ref, artifact_hash=self.basis_hash)

    @classmethod
    def create(cls, **values):
        probe = cls.model_validate({**values, 'basis_ref':'pending', 'basis_hash':'sha256:'+'0'*64}, context={'compliance_draft':True})
        payload = probe.model_dump(mode='json', exclude={'basis_ref','basis_hash'})
        digest = content_hash(payload)
        return cls.model_validate({**payload, 'basis_ref':'tender-legal-compliance-basis:'+digest, 'basis_hash':'sha256:'+digest})


class TenderLegalComplianceVerificationV1(BaseModel):
    """Derived frozen verification embedded in the EVALUATION reasoning trace."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['tender_legal_compliance_verification.v1'] = 'tender_legal_compliance_verification.v1'
    verification_ref: str
    verification_hash: str = Field(pattern=SHA)
    basis: ArtifactPinV1
    policy: ArtifactPinV1
    projection: ArtifactPinV1
    hypothesis: ArtifactPinV1
    status: Literal['VERIFIED', 'INVALID_BINDING']
    outcome: Literal['VIOLATION', 'NO_VIOLATION_IN_SCOPE', 'NOT_APPLICABLE', 'UNKNOWN', 'CONFLICT']
    diagnostics: tuple[str, ...]
    evaluation: dict | None

    @model_validator(mode='after')
    def exact_verification(self):
        digest = content_hash(self.model_dump(mode='json', exclude={'verification_ref','verification_hash'}))
        if (self.verification_ref, self.verification_hash) != ('tender-legal-verification:'+digest, 'sha256:'+digest):
            raise ValueError('LEGAL_VERIFICATION_HASH_MISMATCH')
        if self.status == 'INVALID_BINDING' and self.outcome != 'UNKNOWN':
            raise ValueError('INVALID_BINDING_HAS_NO_LEGAL_OUTCOME')
        return self

    @classmethod
    def create(cls, **values):
        payload = {'schema_version':'tender_legal_compliance_verification.v1', **values}
        from pydantic import TypeAdapter
        payload = TypeAdapter(dict).dump_python(payload, mode='json')
        digest = content_hash(payload)
        return cls.model_validate({**payload, 'verification_ref':'tender-legal-verification:'+digest, 'verification_hash':'sha256:'+digest})

    def pin(self):
        return ArtifactPinV1(artifact_kind='tender_legal_compliance_verification',
                            artifact_ref=self.verification_ref, artifact_hash=self.verification_hash)


class TenderFrozenComplianceDecisionV1(BaseModel):
    """Report attachment containing the exact retained record and its basis."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    decision: DecisionRecord
    basis: TenderLegalComplianceBasisV1
    verification: TenderLegalComplianceVerificationV1

    @model_validator(mode='after')
    def retained_binding(self):
        d, b, v = self.decision, self.basis, self.verification
        expected_subject = ArtifactPinV1(artifact_kind='governed_frame_finding',
            artifact_ref=b.finding.finding_id, artifact_hash=b.finding.finding_hash)
        if (d.purpose != 'EVALUATION' or d.authorization is not None
            or (d.organization_id, d.workspace_id) != (b.organization_id, b.workspace_id)
            or d.subject != expected_subject or v.basis != b.pin()
            or v.status != 'VERIFIED' or d.result_code != v.outcome
            or d.reasoning.strategy != 'tender_scoped_compliance_evaluation'
            or len(d.reasoning.invocations) != 1):
            raise ValueError('RETAINED_COMPLIANCE_DECISION_MISMATCH')
        invocation = d.reasoning.invocations[0]
        if (set(invocation.inputs) != {b.pin(), b.episode, b.inspection_result}
            or invocation.executable_engine != v.policy or invocation.rules != (v.policy,)
            or invocation.outputs != (v.outcome,) or len(invocation.proof_steps) != 1
            or TenderLegalComplianceVerificationV1.model_validate_json(invocation.proof_steps[0]) != v
            or invocation.evidence != d.evidence
            or v.hypothesis.artifact_ref != b.finding.observed_hypothesis_ref
            or v.hypothesis.artifact_hash != b.finding.observed_hypothesis_hash):
            raise ValueError('RETAINED_COMPLIANCE_PROOF_MISMATCH')
        return self


class TenderAssessmentSelectionV1(BaseModel):
    """Frozen selection, explicitly unresolved when retained assessments compete.

    Artifact ordering is presentation only. No historical UNKNOWN is superseded
    merely because another assessment was appended later.
    """
    model_config = ConfigDict(extra='forbid', frozen=True)
    finding: ArtifactPinV1
    candidates: tuple[ArtifactPinV1, ...]
    selected: ArtifactPinV1 | None
    superseded: tuple[ArtifactPinV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    status: Literal['UNAVAILABLE', 'UNIQUE', 'UNRESOLVED_MULTIPLE']

    @model_validator(mode='after')
    def exact_selection(self):
        if self.finding.artifact_kind != 'governed_frame_finding' or any(
            pin.artifact_kind != 'decision_record' for pin in self.candidates
        ):
            raise ValueError('ASSESSMENT_SELECTION_KIND_MISMATCH')
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError('DUPLICATE_ASSESSMENT_CANDIDATE')
        if len(set(self.superseded)) != len(self.superseded) or not set(self.superseded).issubset(set(self.candidates)):
            raise ValueError('ASSESSMENT_SUCCESSOR_INVENTORY_MISMATCH')
        active = tuple(pin for pin in self.candidates if pin not in self.superseded)
        if self.candidates and not active:
            raise ValueError('ASSESSMENT_SUCCESSOR_CYCLE')
        expected = 'UNAVAILABLE' if not active else 'UNIQUE' if len(active) == 1 else 'UNRESOLVED_MULTIPLE'
        selected = active[0] if expected == 'UNIQUE' else None
        if self.status != expected or self.selected != selected:
            raise ValueError('ASSESSMENT_SELECTION_MISMATCH')
        return self

    @classmethod
    def from_candidates(cls, *, finding, candidates, superseded=()):
        candidates = tuple(sorted(candidates, key=lambda pin: (pin.artifact_ref, pin.artifact_hash)))
        superseded = tuple(sorted(superseded, key=lambda pin: (pin.artifact_ref,pin.artifact_hash)))
        active = tuple(pin for pin in candidates if pin not in superseded)
        return cls(finding=finding, candidates=candidates, superseded=superseded,
            selected=active[0] if len(active) == 1 else None,
            status='UNAVAILABLE' if not active else 'UNIQUE' if len(active) == 1 else 'UNRESOLVED_MULTIPLE')


def freeze_assessment_selection(*, inventory, entries):
    """One selection record for each original Finding, including missing ones."""
    entries = tuple(TenderFrozenComplianceDecisionV1.model_validate(entry) for entry in entries)
    if len({entry.decision.pin() for entry in entries}) != len(entries):
        raise ValueError('DUPLICATE_ASSESSMENT_CANDIDATE')
    by_pin = {entry.decision.pin(): entry for entry in entries}
    superseded = set()
    for entry in entries:
        predecessor = entry.basis.supersedes_assessment
        if predecessor is None:
            continue
        old = by_pin.get(predecessor)
        if (old is None or old.decision.subject != entry.decision.subject
            or old.basis.episode != entry.basis.episode or old.basis.frozen_at >= entry.basis.frozen_at):
            raise ValueError('ASSESSMENT_SUCCESSOR_BINDING_MISMATCH')
        superseded.add(predecessor)
    return tuple(TenderAssessmentSelectionV1.from_candidates(finding=ArtifactPinV1.model_validate(pin),
        candidates=tuple(entry.decision.pin() for entry in entries if entry.decision.subject == ArtifactPinV1.model_validate(pin)),
        superseded=tuple(entry.decision.pin() for entry in entries if entry.decision.subject == ArtifactPinV1.model_validate(pin) and entry.decision.pin() in superseded))
        for pin in inventory)
