"""Embedded source-evolution qualification; existing policy/store owners persist it.

Separate from historical v1 to avoid quality -> normative -> quality cycles.
"""
from typing import Literal
from pydantic import Field, model_validator, StrictBool
from contracts.source_consumer_verification import Embedded, H, ID
from contracts.artifact_pin import ArtifactPinV1

# Embedded non-authoritative source-evolution evidence. The fixed-source v1
# contract above remains byte/meaning compatible with historical retained runs.
from datetime import datetime
from contracts.policy_revision import PolicyRevisionRecordV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.expense_reimbursement import ExpenseTeachingPolicyV1
from contracts.governed_delta import DeltaEnvelopeV1, ImpactProjectionV1
from contracts.canonical_source_ir import CanonicalSourceDocument


class SourceRequirementInstanceV1(Embedded):
    logical_id: ID
    instance_ref: ID
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    state: Literal['ACTIVE', 'UNRESOLVED'] = 'ACTIVE'
    target_ids: tuple[ID, ...] = Field(min_length=1)
    depends_on: tuple[ID, ...] = ()


class SourceConsumerDependencyV1(Embedded):
    requirement_id: ID
    target_id: ID
    consumer: Literal['input_prerequisite', 'input_requirement', 'set_output', 'calendar']
    selector: str
    fact_ref: str | None = None
    operand: int | tuple[str, ...] | None = None
    case_ids: tuple[ID, ...] = Field(min_length=1)


class SourceNativeInputV1(Embedded):
    case_id: ID
    invoice_date: str | None
    submitted_on: str
    applicant_fields: dict[str, str]
    invoice_attached: bool = True


class SourceConsumerSupportV1(Embedded):
    binding_id: ID
    instance_ref: ID
    target_id: ID
    source_hash: str = Field(pattern=H)
    consumer_hash: str = Field(pattern=H)
    # Exact typed dependency and expected operand, independent of candidate bindings.
    dependency_hash: str = Field(pattern=H)


class SourceChangeBasisV1(Embedded):
    source: PolicyRevisionRecordV1
    canonical_source: CanonicalSourceDocument
    session_ref: str
    requirements: tuple[SourceRequirementInstanceV1, ...] = Field(min_length=1)
    dependencies: tuple[SourceConsumerDependencyV1, ...] = Field(min_length=1)
    dependency_inventory_complete: bool
    inputs: tuple[SourceNativeInputV1, ...] = Field(min_length=1)
    candidate: NormativeSemanticCandidateSetV1
    plan: CompiledRulePlanV1
    # Read-only field/effect adapter from the historical teaching bundle. It does
    # not attest qualification of the new candidate or become an active policy.
    adapter_policy: ExpenseTeachingPolicyV1
    supports: tuple[SourceConsumerSupportV1, ...]
    consumer_assets: dict[str, str]


class SourceNativeObservationV1(Embedded):
    case_id: ID
    input_hash: str = Field(pattern=H)
    plan_hash: str = Field(pattern=H)
    response: dict
    response_hash: str = Field(pattern=H)
    observed_at: datetime


class SourceImpactFindingV1(Embedded):
    requirement_id: ID
    instance_ref: ID
    target_id: ID
    status: Literal['SUPPORTED', 'STALE', 'MISSING', 'UNKNOWN', 'CONFLICT', 'WITHDRAWN']
    reason: str
    binding_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()


class SourceChangeAssessmentV1(Embedded):
    schema_version: Literal['source_change_consumer_assessment.v1'] = 'source_change_consumer_assessment.v1'
    exposure: Literal['DEVELOPMENT'] = 'DEVELOPMENT'
    authorship: Literal['ASSISTANT_AUTHORED'] = 'ASSISTANT_AUTHORED'
    completeness: Literal['NOT_QUALIFIED'] = 'NOT_QUALIFIED'
    predecessor: SourceChangeBasisV1
    successor: SourceChangeBasisV1
    equivalences: dict[str, str]
    correction_parent: ArtifactPinV1 | None = None
    delta: DeltaEnvelopeV1
    impact: ImpactProjectionV1
    selected_case_ids: tuple[str, ...]
    observations: tuple[SourceNativeObservationV1, ...]
    findings: tuple[SourceImpactFindingV1, ...]
    produced_at: datetime

    @model_validator(mode='after')
    def retained_source_change(self):
        from services.source_consumer_verification import verify_source_change
        verify_source_change(self)
        return self


class SourceRequiredInventoryV1(Embedded):
    """Frozen authored input evidence, never a second admission/review authority."""
    schema_version: Literal['source_required_inventory.v1'] = 'source_required_inventory.v1'
    exposure: Literal['DEVELOPMENT'] = 'DEVELOPMENT'
    authorship: Literal['ASSISTANT_AUTHORED'] = 'ASSISTANT_AUTHORED'
    completeness: Literal['NOT_QUALIFIED'] = 'NOT_QUALIFIED'
    source: ArtifactPinV1
    source_hash: str = Field(pattern=H)
    organization_id: ID
    workspace_id: ID
    session_ref: ID
    effective_on: str
    validation_basis: ArtifactPinV1
    requirements: tuple[SourceRequirementInstanceV1, ...] = Field(min_length=1)
    dependencies: tuple[SourceConsumerDependencyV1, ...] = Field(min_length=1)
    inputs: tuple[SourceNativeInputV1, ...] = Field(min_length=1)
    predecessor_pin: ArtifactPinV1 | None = None
    # Withdrawn target -> exact successor source instance witnessing the authored
    # change. A changed hash alone is not a withdrawal declaration.
    withdrawals: dict[str, str] = Field(default_factory=dict)


class SourceChangeBasisV2(SourceChangeBasisV1):
    schema_version: Literal['source_change_basis.v2'] = 'source_change_basis.v2'
    required_inventory_pin: ArtifactPinV1 | None = None
    required_inventory: PolicyRevisionRecordV1 | None = None


class SourceChangeAssessmentV2(SourceChangeAssessmentV1):
    schema_version: Literal['source_change_consumer_assessment.v2'] = 'source_change_consumer_assessment.v2'
    predecessor: SourceChangeBasisV2
    successor: SourceChangeBasisV2
    coverage_status: Literal['SUPPORTED', 'DEFICIENT', 'UNKNOWN']


# Explicit new dispatch: legacy expense schemas above keep their byte meaning.
from typing import Annotated
from pydantic import ConfigDict, StrictInt, FiniteFloat, field_validator


class ProcurementFactsV1(Embedded):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, validate_by_name=True, serialize_by_alias=True)
    department: str | None = Field(default=None, alias='procurement.department')
    software_subscription: bool | None = Field(default=None, alias='procurement.software_subscription')
    budget: str | None = Field(default=None, alias='procurement.budget')
    commitment_date: str | None = Field(default=None, alias='procurement.commitment_date')
    first_year_non_cancellable_usd: Annotated[StrictInt | FiniteFloat, Field(ge=0)] | None = Field(default=None, alias='procurement.first_year_non_cancellable_usd')
    cancellable_usage_usd: Annotated[StrictInt | FiniteFloat, Field(ge=0)] | None = Field(default=None, alias='procurement.cancellable_usage_usd')
    emergency_security_remediation: bool | None = Field(default=None, alias='procurement.emergency_security_remediation')
    incident_ref: str | None = Field(default=None, alias='procurement.incident_ref')
    incident_recorded_by: str | None = Field(default=None, alias='procurement.incident_recorded_by')
    current_commitment_finance_approved: bool | None = Field(default=None, alias='procurement.current_commitment_finance_approved')
    entity_kind: str | None = Field(default=None, alias='procurement.entity_kind')
    prior_request_approved: bool | None = Field(default=None, alias='procurement.prior_request_approved')
    prior_request_usd: Annotated[StrictInt | FiniteFloat, Field(ge=0)] | None = Field(default=None, alias='procurement.prior_request_usd')

    @field_validator('commitment_date')
    @classmethod
    def iso_date(cls, value):
        from datetime import date
        if value is not None and date.fromisoformat(value).isoformat()!=value:
            raise ValueError('PROCUREMENT_DATE_INVALID')
        return value


class ProcurementInputV1(Embedded):
    case_id: ID
    facts: ProcurementFactsV1

    def native_facts(self):
        return self.facts.model_dump(mode='json', by_alias=True, exclude_none=True)


class ProcurementDependencyV1(Embedded):
    requirement_id: ID
    target_id: ID
    consumer: Literal['predicate'] = 'predicate'
    rule_id: ID
    role: Literal['condition','input_requirement','exception','precedence']
    selector: ID
    fact_ref: str
    operator: Literal['EQ','IN','GE','GT','LE','LT','DATE_ON_OR_AFTER','DATE_ON_OR_BEFORE','EXISTS']
    operand: str | StrictInt | FiniteFloat | bool | tuple[str,...] | None = None
    frame_ref: ID
    precedence_ref: ID | None = None
    case_ids: tuple[ID,...] = Field(min_length=1)

    @model_validator(mode='after')
    def role_shape(self):
        if (self.role=='precedence')!=(self.precedence_ref is not None):
            raise ValueError('PROCUREMENT_PRECEDENCE_ROLE_INVALID')
        return self


class ProcurementRequiredInventoryV1(SourceRequiredInventoryV1):
    schema_version: Literal['source_required_inventory.v2'] = 'source_required_inventory.v2'
    domain: Literal['procurement']
    source_document_hash: str = Field(pattern=H)
    canonical_hash: str = Field(pattern=H)
    dependencies: tuple[ProcurementDependencyV1,...] = Field(min_length=1)
    inputs: tuple[ProcurementInputV1,...] = Field(min_length=1)


class SourceChangeBasisV3(Embedded):
    schema_version: Literal['source_change_basis.v3'] = 'source_change_basis.v3'
    domain: Literal['procurement']
    source: PolicyRevisionRecordV1
    canonical_source: CanonicalSourceDocument
    session_ref: str
    requirements: tuple[SourceRequirementInstanceV1,...] = Field(min_length=1)
    dependencies: tuple[ProcurementDependencyV1,...]
    dependency_inventory_complete: bool
    inputs: tuple[ProcurementInputV1,...] = Field(min_length=1)
    candidate: NormativeSemanticCandidateSetV1
    plan: CompiledRulePlanV1
    context: PolicyRevisionRecordV1
    supports: tuple[SourceConsumerSupportV1,...]
    consumer_assets: dict[str,str]
    required_inventory_pin: ArtifactPinV1 | None = None
    required_inventory: PolicyRevisionRecordV1 | None = None


class SourceChangeAssessmentV3(SourceChangeAssessmentV1):
    schema_version: Literal['source_change_consumer_assessment.v3'] = 'source_change_consumer_assessment.v3'
    domain: Literal['procurement']
    predecessor: SourceChangeBasisV3
    successor: SourceChangeBasisV3
    coverage_status: Literal['SUPPORTED','DEFICIENT','UNKNOWN']
    verifier_hash: str = Field(pattern=H)


# Explicit DEVELOPMENT v1.1 successor. Raw form membership is durable data, not
# reconstructed from Pydantic defaults or a caller-selected evidence truth state.
from pydantic import JsonValue

INCIDENT_PROFILE = 'controlled-procurement.incident-evidence.v1.1'


class IncidentObservationClaimV1(Embedded):
    value: str | None = None
    state: Literal['KNOWN_ABSENT'] | None = None
    recorded_by: str
    observed_at: datetime


class IncidentClaimV1(Embedded):
    state: Literal['KNOWN_ABSENT', 'CONFLICT']
    recorded_by: str | None = None
    observed_at: datetime | None = None
    observations: tuple[IncidentObservationClaimV1, ...] = ()


class ProcurementSubmittedFactsV2(ProcurementFactsV1):
    incident_ref: str | IncidentClaimV1 | None = Field(default=None, alias='procurement.incident_ref')


class ProcurementInputV2(Embedded):
    case_id: ID
    form: dict[str, JsonValue]
    subject_ref: ID
    commitment_ref: ID
    scope_ref: ID
    as_of: datetime
    evidence_pins: tuple[ArtifactPinV1, ...] = ()

    @model_validator(mode='after')
    def submitted_shape(self):
        # JSON-mode strict parsing permits canonical date strings in assertions;
        # the original submitted JSON is kept verbatim in form.
        import json
        ProcurementSubmittedFactsV2.model_validate_json(json.dumps(self.form))
        if self.as_of.utcoffset() is None or self.as_of.utcoffset().total_seconds()!=0:
            raise ValueError('INCIDENT_AS_OF_UTC_REQUIRED')
        return self


class SyntheticIncidentWitnessV1(Embedded):
    profile_id: str
    source_pin: ArtifactPinV1
    kind: Literal['recorder_basis', 'recording']
    actor_ref: ID
    role: str | None = None
    basis_pin: ArtifactPinV1 | None = None
    proposition: Literal['incident_reference_recorded_by_required_actor'] = 'incident_reference_recorded_by_required_actor'
    polarity: Literal['POSITIVE', 'NEGATIVE'] = 'POSITIVE'
    incident_ref: str = 'INC-1'
    subject_ref: ID = 'software-security'
    commitment_ref: ID = 'commitment-1'
    scope_ref: ID = 'incident-ledger-2026'
    recorded_at: datetime
    observed_at: datetime
    valid_from: datetime
    valid_until: datetime

    @model_validator(mode='after')
    def utc(self):
        for key in ('recorded_at','observed_at','valid_from','valid_until'):
            v=getattr(self,key)
            if v.utcoffset() is None or v.utcoffset().total_seconds()!=0:
                raise ValueError('INCIDENT_WITNESS_UTC_REQUIRED')
        return self


class ProcurementDependencyV2(ProcurementDependencyV1):
    consumer: Literal['predicate','normalization','qualification','projection'] = 'predicate'
    role: Literal['condition','input_requirement','exception','precedence','normalization','qualification','projection']


class ProcurementRequiredInventoryV2(ProcurementRequiredInventoryV1):
    schema_version: Literal['source_required_inventory.v3'] = 'source_required_inventory.v3'
    profile_id: Literal['controlled-procurement.incident-evidence.v1.1']
    dependencies: tuple[ProcurementDependencyV2,...] = Field(min_length=1)
    inputs: tuple[ProcurementInputV2,...] = Field(min_length=1)


class SourceChangeBasisV4(SourceChangeBasisV3):
    schema_version: Literal['source_change_basis.v4'] = 'source_change_basis.v4'
    profile_id: Literal['controlled-procurement.incident-evidence.v1.1']
    dependencies: tuple[ProcurementDependencyV2,...]
    inputs: tuple[ProcurementInputV2,...] = Field(min_length=1)


class SourceChangeAssessmentV4(SourceChangeAssessmentV3):
    schema_version: Literal['source_change_consumer_assessment.v4'] = 'source_change_consumer_assessment.v4'
    predecessor: SourceChangeBasisV4
    successor: SourceChangeBasisV4


# Bounded component producer successor; legacy scalar/incident contracts above
# are intentionally unchanged. Raw form preserves suppliedness after persistence.
from pydantic import StrictBool

AMOUNT_PROFILE = 'controlled-procurement.component-amount.v1'


class ProcurementFeePeriodV1(Embedded):
    first_twelve_months: StrictBool | Literal['UNKNOWN'] | None = None
    start: str | None = None
    end_exclusive: str | None = None


class ProcurementFeeComponentV1(Embedded):
    component_id: ID
    currency: Literal['USD']
    amount_usd: Annotated[StrictInt, Field(ge=0)] | Literal['UNKNOWN'] | None = None
    fee_kind: Literal['subscription','usage','UNKNOWN'] | None = None
    separately_cancellable: StrictBool | Literal['UNKNOWN'] | None = None
    due_period: ProcurementFeePeriodV1 | Literal['UNKNOWN'] | None = None
    evidence: dict[str, str]


class ProcurementFeeFormV1(Embedded):
    facts: ProcurementFactsV1
    fee_components: tuple[ProcurementFeeComponentV1,...] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def bounded(self):
        if self.facts.first_year_non_cancellable_usd is not None and type(self.facts.first_year_non_cancellable_usd) is not int:raise ValueError('AMOUNT_CALLER_WHOLE_USD_REQUIRED')
        if self.facts.emergency_security_remediation is not False:
            raise ValueError('AMOUNT_ORDINARY_PROFILE_REQUIRED')
        if len({c.component_id for c in self.fee_components})!=len(self.fee_components):
            raise ValueError('AMOUNT_DUPLICATE_COMPONENT')
        if any(c.component_id=='native_amount' for c in self.fee_components):
            raise ValueError('AMOUNT_RESERVED_COMPONENT_ID')
        return self


class ProcurementInputV3(Embedded):
    case_id: ID
    form: dict[str, JsonValue]
    form_pin: ArtifactPinV1

    @model_validator(mode='after')
    def typed_form(self):
        ProcurementFeeFormV1.model_validate(self.form)
        caller=self.form.get('facts',{}).get('procurement.first_year_non_cancellable_usd')
        if caller is not None and type(caller) is not int:raise ValueError('AMOUNT_CALLER_WHOLE_USD_REQUIRED')
        return self


class ProcurementComponentReceiptV1(Embedded):
    component_id: ID
    component_hash: str = Field(pattern=H)
    disposition: Literal['INCLUDED','EXCLUDED','UNRESOLVED']
    reason: ID
    amount: StrictInt | None = Field(default=None, ge=0)
    supplied_keys: tuple[str,...]
    evidence_refs: tuple[str,...] = Field(min_length=1)


class ProcurementFormationV1(Embedded):
    profile_id: Literal['controlled-procurement.component-amount.v1'] = AMOUNT_PROFILE
    source_pin: ArtifactPinV1
    form_pin: ArtifactPinV1
    input_hash: str = Field(pattern=H)
    roster_hash: str = Field(pattern=H)
    producer_hash: str = Field(pattern=H)
    status: Literal['COMPLETE','UNKNOWN']
    amount: StrictInt | None = Field(default=None, ge=0)
    caller_supplied: bool
    caller_assertion: StrictInt | None = Field(default=None, ge=0)
    dispatch: Literal['READY','AGGREGATE_MISMATCH']
    receipts: tuple[ProcurementComponentReceiptV1,...] = Field(min_length=1)

    @model_validator(mode='after')
    def complete_shape(self):
        if (self.status=='COMPLETE')!=(self.amount is not None):raise ValueError('AMOUNT_FORMATION_STATE_INVALID')
        if self.status=='COMPLETE' and any(r.disposition=='UNRESOLVED' for r in self.receipts):raise ValueError('AMOUNT_UNRESOLVED_COMPLETE')
        if self.status=='UNKNOWN' and not any(r.disposition=='UNRESOLVED' for r in self.receipts):raise ValueError('AMOUNT_UNKNOWN_WITHOUT_BASIS')
        mismatch=self.amount is not None and self.caller_assertion is not None and self.amount!=self.caller_assertion
        if (self.dispatch=='AGGREGATE_MISMATCH')!=mismatch:raise ValueError('AMOUNT_DISPATCH_STATE_INVALID')
        return self


class ProcurementDependencyV3(ProcurementDependencyV1):
    consumer: Literal['predicate','component_receipt','formation_handoff'] = 'predicate'
    role: Literal['condition','input_requirement','exception','precedence','formation']


class ProcurementRequiredInventoryV3(ProcurementRequiredInventoryV1):
    schema_version: Literal['source_required_inventory.v4'] = 'source_required_inventory.v4'
    profile_id: Literal['controlled-procurement.component-amount.v1'] = AMOUNT_PROFILE
    dependencies: tuple[ProcurementDependencyV3,...] = Field(min_length=1)
    inputs: tuple[ProcurementInputV3,...] = Field(min_length=1)


class SourceChangeBasisV5(SourceChangeBasisV3):
    schema_version: Literal['source_change_basis.v5'] = 'source_change_basis.v5'
    profile_id: Literal['controlled-procurement.component-amount.v1'] = AMOUNT_PROFILE
    dependencies: tuple[ProcurementDependencyV3,...]
    inputs: tuple[ProcurementInputV3,...] = Field(min_length=1)
    forms: tuple[PolicyRevisionRecordV1,...]
    formations: tuple[PolicyRevisionRecordV1,...]
    # Candidate carriage is independently compared against the frozen inventory.
    receipt_bindings: dict[str, str]


class SourceChangeAssessmentV5(SourceChangeAssessmentV3):
    schema_version: Literal['source_change_consumer_assessment.v5'] = 'source_change_consumer_assessment.v5'
    predecessor: SourceChangeBasisV5
    successor: SourceChangeBasisV5

# Additive factual lane. These are non-authoritative source assertions, not policy.
FEE_FACTS_PROFILE = 'procurement.fee-facts.v1'


class FeeAssertionV1(Embedded):
    state: Literal['MISSING', 'NULL', 'UNKNOWN', 'KNOWN', 'CONFLICT']
    raw: JsonValue = None
    evidence_refs: tuple[str, ...] = ()


class FeeComponentFactsV1(Embedded):
    component_id: ID
    source_row_id: ID
    currency: Literal['CNY', 'USD']
    raw_amount: str | None = None
    source_unit: Literal['yuan', '万元', 'dollars']
    fee_kind: Literal['maintenance', 'subscription', 'usage']
    source_label: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    service_period: FeeAssertionV1
    cancellation: FeeAssertionV1
    due_condition: FeeAssertionV1
    amount_basis: Literal['tax_inclusive_stated_price', 'tax_exclusive_stated_price'] = 'tax_inclusive_stated_price'

    @model_validator(mode='after')
    def compatible_unit(self):
        if (self.currency == 'USD') != (self.source_unit == 'dollars'):
            raise ValueError('FEE_CURRENCY_UNIT_CONFLICT')
        return self


class FeeFactsFormV1(Embedded):
    source_kind: Literal['SOURCE_TRANSCRIPTION', 'SYNTHETIC_METAMORPHIC'] = 'SOURCE_TRANSCRIPTION'
    case_id: ID
    components: tuple[FeeComponentFactsV1, ...] = Field(min_length=1, max_length=100)
    amount_basis: Literal['tax_inclusive_stated_price', 'tax_exclusive_stated_price']
    source_context: dict[str, JsonValue]
    declared_total_minor: Annotated[StrictInt, Field(ge=0)] | None = None

    @model_validator(mode='after')
    def unique_components(self):
        if len({c.component_id for c in self.components}) != len(self.components):
            raise ValueError('FEE_DUPLICATE_COMPONENT')
        return self


class FeeMoneyV1(Embedded):
    currency: Literal['CNY', 'USD']
    scale: Annotated[StrictInt, Field(ge=2, le=2)] = 2
    amount_minor: Annotated[StrictInt, Field(ge=0)] | None
    status: Literal['COMPLETE', 'UNKNOWN']

    @model_validator(mode='after')
    def complete_money(self):
        if (self.status == 'COMPLETE') != (self.amount_minor is not None):
            raise ValueError('FEE_AMOUNT_STATUS_MISMATCH')
        return self


class FeeNormalizationReceiptV1(Embedded):
    component_id: ID
    source_row_id: ID
    component_hash: str = Field(pattern=H)
    source_hash: str = Field(pattern=H)
    locator: str
    raw_amount: str | None
    source_unit: Literal['yuan', '万元', 'dollars']
    money: FeeMoneyV1


class FeeFactsFormationV1(Embedded):
    producer_hash: str = Field(pattern=H)
    source_kind: Literal['SOURCE_TRANSCRIPTION', 'SYNTHETIC_METAMORPHIC']
    source_context: dict[str, JsonValue]
    component_assertions: dict[str, dict[str, FeeAssertionV1]]
    schema_version: Literal['fee_facts_formation.v1'] = 'fee_facts_formation.v1'
    total_basis: Literal['STATED_COMPONENT_TOTAL'] = 'STATED_COMPONENT_TOTAL'
    amount_basis: str
    receipts: tuple[FeeNormalizationReceiptV1, ...]
    total: FeeMoneyV1
    qualified_commitment: Literal['UNRESOLVED'] = 'UNRESOLVED'
    finance: Literal['NOT_EVALUATED'] = 'NOT_EVALUATED'
    finance_reason: Literal['POLICY_BASIS_ABSENT'] = 'POLICY_BASIS_ABSENT'
    caller_assertion: Annotated[StrictInt, Field(ge=0)] | None = None
    assertion_status: Literal['ABSENT', 'MATCH', 'MISMATCH', 'UNKNOWN']


class FeeFactsDependencyV1(Embedded):
    requirement_id: ID
    target_id: ID
    consumer: Literal['fee_component', 'fee_total']
    selector: ID
    case_ids: tuple[ID, ...] = ('factual-consumer',)


class FeeFactsCoverageBasisV1(Embedded):
    domain: Literal['fee_facts'] = 'fee_facts'
    requirements: tuple[SourceRequirementInstanceV1, ...]
    dependencies: tuple[FeeFactsDependencyV1, ...]
    bindings: dict[str, str]
    formation: FeeFactsFormationV1

# Explicit hypothetical lane; never changes factual fee/Finance meanings.
DEVELOPMENT_MEASURE_PROFILE = 'procurement.dev.measure.v1'
DEVELOPMENT_MEASURE_PROFILE_HASH = 'sha256:72290bc06b697b3b5bba873dfc4fd8836285d9b654da1f807562777b9db9393c'


class DevelopmentMeasureFixtureV1(Embedded):
    schema_version: Literal['development_measure_fixture.v1'] = Field(alias='schema')
    profile: dict[str, JsonValue]
    inputs: dict[str, dict[str, JsonValue]] = Field(min_length=1, max_length=100)
    reference_definitions: dict[str, dict[str, JsonValue]]

    @model_validator(mode='after')
    def input_only(self):
        def check(value):
            if isinstance(value, dict):
                if any(k.startswith('expected') or k in ('assessment','coverage_status','consumer_output') for k in value):
                    raise ValueError('DEVELOPMENT_EXPECTATIONS_FORBIDDEN')
                for v in value.values():check(v)
            elif isinstance(value,list):
                for v in value:check(v)
        check(self.model_dump(mode='json'))
        return self


class DevelopmentMeasureResultV1(Embedded):
    status: Literal['HYPOTHETICAL_EXPOSURE','UNRESOLVED','REFUSED']
    hypothetical_exposure_fen: Annotated[StrictInt, Field(ge=0,le=2**63-1)] | None
    # Exact promoted reference shape, including raw missing/null observations.
    details: dict[str, JsonValue]
    real_contract: Literal['NOT_APPLIED'] = 'NOT_APPLIED'
    qualified_commitment: Literal['UNRESOLVED'] = 'UNRESOLVED'
    finance: Literal['NOT_EVALUATED'] = 'NOT_EVALUATED'
    finance_reason: Literal['POLICY_BASIS_ABSENT'] = 'POLICY_BASIS_ABSENT'

    @model_validator(mode='after')
    def consistent(self):
        d=self.details
        if (d.get('status'),d.get('exposure_fen'))!=(self.status,self.hypothetical_exposure_fen):
            raise ValueError('DEVELOPMENT_RESULT_PROJECTION_MISMATCH')
        if ((self.status=='HYPOTHETICAL_EXPOSURE')!=(self.hypothetical_exposure_fen is not None)
                or d.get('provenance')!=['DEVELOPMENT','ASSISTANT_AUTHORED','NOT_QUALIFIED']
                or d.get('owner_decisions')!={f'Q{i}':'PENDING' for i in range(1,7)}
                or d.get('classification')!='ASSISTANT_SELECTED_DEVELOPMENT_ASSUMPTIONS'
                or d.get('domain_review')!='NOT_PERFORMED' or type(d.get('independent_reviews')) is not int
                or d['independent_reviews']!=0):raise ValueError('DEVELOPMENT_FALSE_AUTHORITY_OR_RESULT')
        return self


class DevelopmentFieldReceiptV1(Embedded):
    receipt_id: ID
    fixture_hash: str = Field(pattern=H)
    fixture_key: ID
    pointer: str
    supplied: StrictBool
    raw: JsonValue
    reference: str
    reference_pointer: str


class DevelopmentMeasureFormationV1(Embedded):
    input_hash: str = Field(pattern=H)
    profile_hash: str = Field(pattern=H)
    core_hash: str = Field(pattern=H)
    result: DevelopmentMeasureResultV1
    receipts: tuple[DevelopmentFieldReceiptV1,...]


class DevelopmentMeasureDependencyV1(Embedded):
    requirement_id: ID
    target_id: ID
    consumer: Literal['development_field','development_handoff']
    selector: ID
    case_ids: tuple[ID,...] = ('hypothetical-measure',)


class DevelopmentMeasureCoverageBasisV1(Embedded):
    domain: Literal['development_measure'] = 'development_measure'
    requirements: tuple[SourceRequirementInstanceV1,...]
    dependencies: tuple[DevelopmentMeasureDependencyV1,...]
    bindings: dict[str,str]
    formation: DevelopmentMeasureFormationV1
