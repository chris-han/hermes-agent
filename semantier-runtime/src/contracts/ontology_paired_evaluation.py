"""Immutable contracts for same-obligation paired ontology evaluation."""
from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash
from contracts.ontology_validation import RuntimeSemanticEvaluationBundleV1

HASH = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError("paired evaluation timestamps must be timezone-aware UTC")
        return self


class _Addressed(_Contract):
    REF: ClassVar[str]
    HASH_FIELD: ClassVar[str]
    PREFIX: ClassVar[str]

    @model_validator(mode="after")
    def bound(self):
        digest = content_hash(self.model_dump(mode="json", exclude={self.REF, self.HASH_FIELD}))
        if getattr(self, self.REF) != f"{self.PREFIX}:{digest}" or getattr(self, self.HASH_FIELD) != f"sha256:{digest}":
            raise ValueError("paired evaluation ref/hash does not bind exact content")
        return self

    @classmethod
    def create(cls, **values):
        probe = cls.model_construct(**{**values, cls.REF: "pending", cls.HASH_FIELD: "sha256:" + "0" * 64})
        digest = content_hash(probe.model_dump(mode="json", exclude={cls.REF, cls.HASH_FIELD}))
        return cls(**{**values, cls.REF: f"{cls.PREFIX}:{digest}", cls.HASH_FIELD: f"sha256:{digest}"})


class ArtifactPayloadPinV1(_Contract):
    artifact_ref: str
    artifact_hash: str = Field(pattern=HASH)
    payload_hash: str = Field(pattern=HASH)

    @classmethod
    def of(cls, artifact: BaseModel, ref_field: str, hash_field: str):
        return cls(artifact_ref=getattr(artifact, ref_field), artifact_hash=getattr(artifact, hash_field),
                   payload_hash="sha256:" + content_hash(artifact.model_dump(mode="json")))


class PairedEvaluationSubjectV1(_Contract):
    baseline_manifest: ArtifactPayloadPinV1
    successor_manifest: ArtifactPayloadPinV1
    baseline_snapshot: ArtifactPayloadPinV1
    successor_snapshot: ArtifactPayloadPinV1
    alignment_delta: ArtifactPayloadPinV1 | None = None


class PairedEvaluationPolicyV1(_Contract):
    policy_version: Literal["ontology_paired_evaluation_policy.v1"] = "ontology_paired_evaluation_policy.v1"
    required_case_ids: tuple[str, ...]
    critical_regression_blocks: Literal[True] = True
    require_all_required_measured: Literal[True] = True

    @model_validator(mode="after")
    def unique(self):
        if len(self.required_case_ids) != len(set(self.required_case_ids)):
            raise ValueError("required case ids must be unique")
        return self

    @property
    def policy_hash(self):
        return "sha256:" + content_hash(self.model_dump(mode="json"))


class EvidenceTraceObligationV1(_Contract):
    execution_kind: Literal["evidence_trace"] = "evidence_trace"
    case_id: str
    obligation_id: str
    competency_id: str | None = None
    case_kind: Literal["positive", "hard_negative", "boundary", "ambiguity"]
    critical: bool
    relation_type: str
    source_node_id: str
    target_node_id: str
    expected_present: bool
    expected_evidence_refs: tuple[str, ...]

    @model_validator(mode="after")
    def positive_trace_has_reviewed_evidence(self):
        if self.expected_present and not self.expected_evidence_refs:
            raise ValueError("positive evidence trace requires reviewed evidence refs")
        return self


class DeterministicRuleObligationV1(_Contract):
    execution_kind: Literal["deterministic_rule"] = "deterministic_rule"
    case_id: str
    obligation_id: str
    competency_id: str
    case_kind: Literal["positive", "hard_negative", "boundary", "ambiguity"]
    critical: bool
    expected_outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    observed_facts_ref: str
    observed_facts_hash: str = Field(pattern=HASH)
    baseline_plan_ref: str | None = None
    baseline_plan_hash: str | None = Field(default=None, pattern=HASH)
    successor_plan_ref: str | None = None
    successor_plan_hash: str | None = Field(default=None, pattern=HASH)
    rule_id: str
    expected_evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def paired_plans(self):
        if (self.baseline_plan_ref is None) != (self.baseline_plan_hash is None) or (
            (self.successor_plan_ref is None) != (self.successor_plan_hash is None)
        ):
            raise ValueError("each rule plan ref/hash must be paired")
        return self


PairedObligationV1 = Annotated[Union[EvidenceTraceObligationV1, DeterministicRuleObligationV1],
                               Field(discriminator="execution_kind")]


class OntologyPairedObligationSuiteV1(_Addressed):
    REF = "suite_ref"; HASH_FIELD = "suite_hash"; PREFIX = "ontology-paired-obligation-suite"
    schema_version: Literal["ontology_paired_obligation_suite.v1"] = "ontology_paired_obligation_suite.v1"
    suite_ref: str
    suite_hash: str = Field(pattern=HASH)
    organization_id: str
    workspace_id: str
    source_identity_ref: str
    source_hash: str = Field(pattern=HASH)
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str = Field(pattern=HASH)
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=HASH)
    competency_contract_ref: str | None = None
    competency_contract_hash: str | None = Field(default=None, pattern=HASH)
    competency_contract_content_hash: str | None = Field(default=None, pattern=HASH)
    reviewer_ref: str
    review_state: Literal["accepted"] = "accepted"
    decision_ref: str
    decision_hash: str = Field(pattern=HASH)
    reviewed_at: datetime
    cases: tuple[PairedObligationV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reviewed_unique_cases(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation case ids must be unique")
        present = (self.competency_contract_ref is not None, self.competency_contract_hash is not None,
                   self.competency_contract_content_hash is not None)
        if len(set(present)) != 1:
            raise ValueError("competency contract ref/hash/content hash must be supplied together")
        return self


class ObservedFactsV1(_Addressed):
    REF = "facts_ref"; HASH_FIELD = "facts_hash"; PREFIX = "paired-observed-facts"
    schema_version: Literal["paired_observed_facts.v1"] = "paired_observed_facts.v1"
    facts_ref: str
    facts_hash: str = Field(pattern=HASH)
    organization_id: str
    workspace_id: str
    facts: dict[str, Any]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    frozen_at: datetime


class EvaluatorCodeUnitV1(_Contract):
    code_ref: str
    code_hash: str = Field(pattern=HASH)
    code_bytes: bytes

    @model_validator(mode="after")
    def exact_bytes(self):
        if self.code_hash != "sha256:" + sha256(self.code_bytes).hexdigest():
            raise ValueError("evaluator code unit hash mismatch")
        return self


class PairedEvaluatorBundleV1(_Addressed):
    REF = "evaluator_ref"; HASH_FIELD = "evaluator_hash"; PREFIX = "paired-evaluator-bundle"
    schema_version: Literal["paired_evaluator_bundle.v1"] = "paired_evaluator_bundle.v1"
    evaluator_ref: str
    evaluator_hash: str = Field(pattern=HASH)
    organization_id: str
    workspace_id: str
    code_units: tuple[EvaluatorCodeUnitV1, ...] = Field(min_length=2)
    policy: PairedEvaluationPolicyV1
    policy_hash: str = Field(pattern=HASH)
    supported_execution_kinds: tuple[Literal["evidence_trace", "deterministic_rule"], ...] = (
        "evidence_trace", "deterministic_rule")
    frozen_at: datetime

    @model_validator(mode="after")
    def exact_bundle(self):
        if self.policy_hash != self.policy.policy_hash:
            raise ValueError("evaluator policy hash mismatch")
        refs = [item.code_ref for item in self.code_units]
        if len(refs) != len(set(refs)):
            raise ValueError("evaluator code unit refs must be unique")
        if len(self.supported_execution_kinds) != len(set(self.supported_execution_kinds)):
            raise ValueError("supported execution kinds must be unique")
        return self

    @classmethod
    def create(cls, **values):
        policy = values["policy"]
        values["policy_hash"] = policy.policy_hash
        return super().create(**values)


class PairedCaseResultV1(_Contract):
    case_id: str
    execution_kind: Literal["evidence_trace", "deterministic_rule"]
    measurement_scope: Literal["evidence_navigation", "behavioral"]
    critical: bool
    status: Literal["MEASURED", "UNMEASURED", "ERROR"]
    expected_result: Any = None
    actual_result: Any = None
    passed: bool | None = None
    reason_code: str
    witness_refs: tuple[str, ...] = ()
    runtime_trace_ref: str | None = None
    runtime_trace_hash: str | None = Field(default=None, pattern=HASH)
    runtime_trace_payload_hash: str | None = Field(default=None, pattern=HASH)

    @model_validator(mode="after")
    def measured_state(self):
        if self.status == "MEASURED" and self.passed is None:
            raise ValueError("measured case requires pass/fail")
        if self.status != "MEASURED" and self.passed is not None:
            raise ValueError("unmeasured/error case cannot claim pass/fail")
        present = (self.runtime_trace_ref is not None, self.runtime_trace_hash is not None,
                   self.runtime_trace_payload_hash is not None)
        if len(set(present)) != 1:
            raise ValueError("runtime trace ref/hash/payload hash must be supplied together")
        return self


class OntologyPairedEvaluationRunV1(_Addressed):
    REF = "run_ref"; HASH_FIELD = "run_hash"; PREFIX = "ontology-paired-evaluation-run"
    schema_version: Literal["ontology_paired_evaluation_run.v1"] = "ontology_paired_evaluation_run.v1"
    run_ref: str
    run_hash: str = Field(pattern=HASH)
    organization_id: str
    workspace_id: str
    side: Literal["baseline", "successor"]
    manifest: ArtifactPayloadPinV1
    snapshot: ArtifactPayloadPinV1
    suite_ref: str
    suite_hash: str = Field(pattern=HASH)
    evaluator_ref: str
    evaluator_hash: str = Field(pattern=HASH)
    cases: tuple[PairedCaseResultV1, ...]
    evaluated_at: datetime


class OntologyPairedEvaluationReportV1(_Addressed):
    REF = "report_ref"; HASH_FIELD = "report_hash"; PREFIX = "ontology-paired-evaluation-report"
    schema_version: Literal["ontology_paired_evaluation_report.v1"] = "ontology_paired_evaluation_report.v1"
    report_ref: str
    report_hash: str = Field(pattern=HASH)
    organization_id: str
    workspace_id: str
    subject: PairedEvaluationSubjectV1
    suite_ref: str | None
    suite_hash: str | None = Field(default=None, pattern=HASH)
    evaluator_ref: str
    evaluator_hash: str = Field(pattern=HASH)
    baseline_run_ref: str | None = None
    baseline_run_hash: str | None = Field(default=None, pattern=HASH)
    successor_run_ref: str | None = None
    successor_run_hash: str | None = Field(default=None, pattern=HASH)
    verdict: Literal["INCONCLUSIVE", "REGRESSED", "IMPROVED", "CHANGED"]
    reason_code: str
    behavioral_measured_count: int = Field(ge=0)
    behavioral_total_count: int = Field(ge=0)
    evidence_measured_count: int = Field(ge=0)
    evidence_total_count: int = Field(ge=0)
    authority_state: Literal["provisional"] = "provisional"
    evaluated_at: datetime

    @model_validator(mode="after")
    def complete_pins(self):
        for ref, digest in ((self.suite_ref, self.suite_hash),
                            (self.baseline_run_ref, self.baseline_run_hash),
                            (self.successor_run_ref, self.successor_run_hash)):
            if (ref is None) != (digest is None):
                raise ValueError("paired report refs/hashes must be paired")
        if (self.baseline_run_ref is None) != (self.successor_run_ref is None):
            raise ValueError("paired report must pin both runs or neither")
        return self


class OntologyPairedEvaluationResultV1(_Contract):
    baseline_run: OntologyPairedEvaluationRunV1 | None
    successor_run: OntologyPairedEvaluationRunV1 | None
    report: OntologyPairedEvaluationReportV1
    runtime_traces: tuple[RuntimeSemanticEvaluationBundleV1, ...] = ()

    @model_validator(mode="after")
    def exact_runtime_traces(self):
        if (self.baseline_run is None) != (self.successor_run is None):
            raise ValueError("paired result must contain both run payloads or neither")
        if self.baseline_run is None:
            if self.report.baseline_run_ref is not None or self.report.successor_run_ref is not None:
                raise ValueError("paired report run pins require exact run payloads")
        else:
            if (self.report.baseline_run_ref, self.report.baseline_run_hash,
                    self.report.successor_run_ref, self.report.successor_run_hash) != (
                    self.baseline_run.run_ref, self.baseline_run.run_hash,
                    self.successor_run.run_ref, self.successor_run.run_hash):
                raise ValueError("paired report does not pin exact returned runs")
            tenant = (self.report.organization_id, self.report.workspace_id)
            if any((run.organization_id, run.workspace_id) != tenant for run in (
                    self.baseline_run, self.successor_run)):
                raise ValueError("paired run payload crosses report tenant")
            if (self.baseline_run.side, self.successor_run.side) != ("baseline", "successor"):
                raise ValueError("paired run payload sides are reversed")
            if (self.baseline_run.manifest, self.baseline_run.snapshot,
                    self.successor_run.manifest, self.successor_run.snapshot) != (
                    self.report.subject.baseline_manifest, self.report.subject.baseline_snapshot,
                    self.report.subject.successor_manifest, self.report.subject.successor_snapshot):
                raise ValueError("paired runs do not bind report subject")
        expected = {
            (case.runtime_trace_ref, case.runtime_trace_hash, case.runtime_trace_payload_hash)
            for run in (self.baseline_run, self.successor_run) if run is not None
            for case in run.cases if case.runtime_trace_ref is not None
        }
        actual = {(trace.bundle_ref, trace.bundle_hash,
                   "sha256:" + content_hash(trace.model_dump(mode="json")))
                  for trace in self.runtime_traces}
        if len(actual) != len(self.runtime_traces) or actual != expected:
            raise ValueError("runtime trace payloads must exactly resolve all run trace pins")
        if any((trace.organization_id, trace.workspace_id) != (
                self.report.organization_id, self.report.workspace_id) for trace in self.runtime_traces):
            raise ValueError("runtime trace payload crosses paired report tenant")
        return self


__all__ = [name for name in tuple(globals()) if name.endswith("V1")]
