"""Expense task inputs; authentication and policy authority are never tool arguments."""
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApplicantFields(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = ''
    department: str = ''
    purpose: str = ''


class ExpenseReimbursementRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['evaluate', 'replay', 'read_current', 'save_draft']
    session_id: str | None = None
    workspace_id: str | None = None
    financial_ref: str | None = None
    submitted_on: str | None = None
    applicant_fields: ApplicantFields = Field(default_factory=ApplicantFields)
    result_ref: str | None = None
    result_hash: str | None = None

    @model_validator(mode='after')
    def required_inputs(self):
        if self.action == 'evaluate':
            if not self.financial_ref or not self.submitted_on:
                raise ValueError('invoice artifact and explicit submission date required')
            if date.fromisoformat(self.submitted_on).isoformat() != self.submitted_on:
                raise ValueError('canonical calendar submission date required')
        elif self.action != 'read_current' and (not self.result_ref or not self.result_hash):
            raise ValueError('exact result reference and hash required')
        return self


class ExpenseTeachingPolicyV1(BaseModel):
    """A qualified teaching input, never an organizational policy admission."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['expense_teaching_policy.v1'] = 'expense_teaching_policy.v1'
    policy_ref: str
    policy_hash: str
    organization_id: str
    workspace_id: str
    teaching_candidate: Literal[True] = True
    schema_pin: dict
    candidate_pin: dict
    plan_pin: dict
    source_identity_ref: str
    source_hash: str
    field_bindings: dict[str, Literal['invoice_date', 'submitted_on', 'name', 'department', 'purpose', 'is_draft']]
    disposition_effects: dict[str, Literal['exception_review_required']]
    qualification: dict
    created_at: str

    @model_validator(mode='after')
    def exact_identity(self):
        from datetime import datetime, timedelta
        from eos.artifact_hashing import content_hash
        from contracts.ontology_building import ArtifactPinV1
        for key, kind in [('schema_pin', 'normative_schema_contract'), ('candidate_pin', 'normative_semantic_candidate_set'), ('plan_pin', 'compiled_rule_plan')]:
            if ArtifactPinV1.model_validate(getattr(self, key)).artifact_kind != kind:
                raise ValueError('EXPENSE_POLICY_INPUT_KIND_MISMATCH')
        if datetime.fromisoformat(self.created_at).utcoffset() != timedelta(0):
            raise ValueError('EXPENSE_POLICY_UTC_REQUIRED')
        report = self.qualification
        if (report.get('schema_version') != 'expense_teaching_qualification.v1'
            or report.get('plan_hash') != self.plan_pin['artifact_hash']
            or report.get('source_hash') != self.source_hash
            or report.get('report_hash') != 'sha256:' + content_hash({k: v for k, v in report.items() if k != 'report_hash'})
            or not report.get('cases')
            or any(item.get('status') != 'PASS' for item in report['cases'])):
            raise ValueError('EXPENSE_POLICY_QUALIFICATION_MISMATCH')
        if datetime.fromisoformat(report['produced_at']).utcoffset() != timedelta(0):
            raise ValueError('EXPENSE_POLICY_QUALIFICATION_UTC_REQUIRED')
        digest = content_hash(self.model_dump(mode='json', exclude={'policy_ref', 'policy_hash'}))
        if self.policy_ref != 'expense-teaching-policy:' + digest or self.policy_hash != 'sha256:' + digest:
            raise ValueError('EXPENSE_POLICY_HASH_MISMATCH')
        return self
