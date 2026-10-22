"""Immutable research bindings, persisted as ordinary derived projections.

These records confer no production, CAL, admission, or activation authority.
PILOT membership remains owned by ExperimentIntegritySnapshotV1.
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash


class _ResearchProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    REF_PREFIX: ClassVar[str]
    ARTIFACT_KIND: ClassVar[str]
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"})

    @model_validator(mode="after")
    def exact_identity(self):
        digest = content_hash(self.hash_payload())
        if (self.projection_ref, self.projection_hash) != (f"{self.REF_PREFIX}:{digest}", f"sha256:{digest}"):
            raise ValueError("RESEARCH_CONTENT_IDENTITY_MISMATCH")
        return self

    @classmethod
    def create(cls, **values):
        provisional = cls.model_construct(**values)
        digest = content_hash(provisional.hash_payload())
        return cls.model_validate({**values, "projection_ref": f"{cls.REF_PREFIX}:{digest}", "projection_hash": f"sha256:{digest}"})

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind=self.ARTIFACT_KIND, artifact_ref=self.projection_ref, artifact_hash=self.projection_hash)


class ResearchCaseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    source: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    language: Literal["EN", "ZH"]
    gold_candidate: Literal["A", "B", "C"]
    categories: tuple[Literal["hard_negative", "qualification", "exception"], ...]
    decision_local_projection: ArtifactPinV1
    decision_contract: ArtifactPinV1
    model_input_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def category_semantics(self):
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("RESEARCH_MEMBERSHIP_DUPLICATE_CATEGORY")
        if "hard_negative" in self.categories and self.gold_candidate != "B":
            raise ValueError("RESEARCH_MEMBERSHIP_HARD_NEGATIVE_REQUIRES_GOLD_B")
        if len({pin.artifact_ref for pin in self.evidence}) != len(self.evidence):
            raise ValueError("RESEARCH_MEMBERSHIP_DUPLICATE_EVIDENCE")
        return self


class ResearchMembershipV1(_ResearchProjection):
    REF_PREFIX = "research-confirmatory-membership"
    ARTIFACT_KIND = "research_confirmatory_membership"
    schema_version: Literal["research_confirmatory_membership.v1"] = "research_confirmatory_membership.v1"
    cases: tuple[ResearchCaseV1, ...] = Field(min_length=72)
    excluded_source_groups: tuple[str, ...]
    reference_labels: ArtifactPinV1
    exclusion_evidence: ArtifactPinV1

    def denominators(self) -> dict[str, dict[str, int]]:
        result = {}
        for language in ("EN", "ZH"):
            cases = [case for case in self.cases if case.language == language]
            result[language] = {
                "total": len(cases), "binding": len(cases),
                **{f"class_{label}": sum(case.gold_candidate == label for case in cases) for label in "ABC"},
                **{category: sum(category in case.categories for case in cases)
                   for category in ("hard_negative", "qualification", "exception")},
            }
        return result

    @model_validator(mode="after")
    def independent_complete_populations(self):
        groups = [case.source_group_id for case in self.cases]
        if len(set(groups)) != len(groups) or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("RESEARCH_MEMBERSHIP_REQUIRES_INDEPENDENT_SOURCE_GROUPS")
        if set(groups) & set(self.excluded_source_groups):
            raise ValueError("RESEARCH_MEMBERSHIP_EXCLUDED_SOURCE_GROUP")
        for language, counts in self.denominators().items():
            if counts["total"] < 36 or any(counts[f"class_{label}"] < 12 for label in "ABC"):
                raise ValueError("RESEARCH_MEMBERSHIP_LANGUAGE_CLASS_MINIMUM")
            if any(counts[key] < 8 for key in ("hard_negative", "qualification", "exception")):
                raise ValueError("RESEARCH_MEMBERSHIP_CATEGORY_MINIMUM")
            for category in ("qualification", "exception"):
                for label in "ABC":
                    count = sum(case.language == language and case.gold_candidate == label and category in case.categories for case in self.cases)
                    if count < 2:
                        raise ValueError("RESEARCH_MEMBERSHIP_CATEGORY_CLASS_MINIMUM")
        return self


class ResearchComparisonBasisV1(_ResearchProjection):
    REF_PREFIX = "research-comparison-basis"
    ARTIFACT_KIND = "research_comparison_basis"
    schema_version: Literal["research_comparison_basis.v1"] = "research_comparison_basis.v1"
    pre_pilot_snapshot: ArtifactPinV1
    trajectory_green_receipt: ArtifactPinV1
    mechanics_pilot_snapshot: ArtifactPinV1
    mechanics_pass_evidence: ArtifactPinV1
    confirmatory_membership: ArtifactPinV1
    ordered_baseline_configs: tuple[ArtifactPinV1, ArtifactPinV1, ArtifactPinV1]
    a4_producer_identity: ArtifactPinV1
    a4_model_tokenizer_runtime_basis: ArtifactPinV1
    evaluator: ArtifactPinV1
    obligation_suite: ArtifactPinV1
    integrity_policy: ArtifactPinV1
    runtime_contract: ArtifactPinV1
    scorer_contract: ArtifactPinV1
    collector_contract: ArtifactPinV1
    comparison_protocol: ArtifactPinV1
    finding_composition_contract: ArtifactPinV1 | None = None

    @model_validator(mode="after")
    def exact_child_roles(self):
        expected = {
            "pre_pilot_snapshot": "experiment_integrity_snapshot",
            "mechanics_pilot_snapshot": "experiment_integrity_snapshot",
            "trajectory_green_receipt": "trajectory_instrumentation_gate_receipt",
            "mechanics_pass_evidence": "pilot_release_gate_receipt",
            "confirmatory_membership": "research_confirmatory_membership",
            "a4_producer_identity": "producer_identity",
            "a4_model_tokenizer_runtime_basis": "producer_config",
            "evaluator": "pilot_comparison_evaluator",
            "obligation_suite": "tender_decision_local_obligation_suite",
            "integrity_policy": "trajectory_integrity_policy",
            "runtime_contract": "tender_decision_runtime_contract",
            "scorer_contract": "tender_decision_scorer_contract",
            "collector_contract": "trajectory_collector_contract",
            "comparison_protocol": "research_comparison_protocol",
        }
        if any(getattr(self, field).artifact_kind != kind for field, kind in expected.items()):
            raise ValueError("RESEARCH_BASIS_CHILD_KIND_MISMATCH")
        if any(pin.artifact_kind != "research_baseline_config" for pin in self.ordered_baseline_configs):
            raise ValueError("RESEARCH_BASIS_BASELINE_KIND_MISMATCH")
        if len({pin.artifact_ref for pin in self.ordered_baseline_configs}) != 3:
            raise ValueError("RESEARCH_BASIS_BASELINES_MUST_BE_DISTINCT")
        if self.pre_pilot_snapshot == self.mechanics_pilot_snapshot:
            raise ValueError("RESEARCH_BASIS_MECHANICS_SNAPSHOT_REQUIRED")
        return self


class ResearchBaselineConfigV1(_ResearchProjection):
    REF_PREFIX = "research-baseline-config"
    ARTIFACT_KIND = "research_baseline_config"
    schema_version: Literal["research_baseline_config.v1"] = "research_baseline_config.v1"
    baseline_id: Literal["B1", "B2", "B5"]
    protocol: Literal["FULL_AUTOREGRESSIVE", "DIRECT_VERDICT", "BOUNDED_S1"]
    a4_config: ArtifactPinV1
    configuration: dict[str, Any]
    prompt_templates: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    obligation_suite: ArtifactPinV1

    @model_validator(mode="after")
    def exact_treatment(self):
        if self.protocol != {"B1": "FULL_AUTOREGRESSIVE", "B2": "DIRECT_VERDICT", "B5": "BOUNDED_S1"}[self.baseline_id]:
            raise ValueError("RESEARCH_BASELINE_PROTOCOL_MISMATCH")
        if self.a4_config.artifact_kind != "producer_config" or not self.configuration:
            raise ValueError("RESEARCH_BASELINE_A4_CONFIG_REQUIRED")
        return self


class ResearchComparisonProtocolV1(_ResearchProjection):
    REF_PREFIX = "research-comparison-protocol"
    ARTIFACT_KIND = "research_comparison_protocol"
    schema_version: Literal["research_comparison_protocol.v1"] = "research_comparison_protocol.v1"
    membership: ArtifactPinV1
    denominators: dict[str, dict[str, int]]
    guard_rates: dict[str, float]
    c1_median_latency_ratio_max: Literal[0.67] = .67
    c1_p95_latency_ratio_max: Literal[0.80] = .80
    c1_generated_token_ratio_max: Literal[0.25] = .25
    c2a_macro_f1_gain_min: Literal[0.10] = .10
    c2a_hard_negative_reduction_min: Literal[0.30] = .30
    guard_error_count_max: Literal[1] = 1
    binding_violations_max: Literal[0] = 0
    development_corrections_max: Literal[1] = 1
    confirmatory_runs_max: Literal[1] = 1
    retry_attempts_max: Literal[0] = 0
    invalid_output_treatment: Literal["COUNT_AS_ERROR_IN_FROZEN_POPULATION"] = "COUNT_AS_ERROR_IN_FROZEN_POPULATION"
    hard_negative_error_definition: Literal["ANY_NON_B_OUTPUT_IN_GOLD_B_HARD_NEGATIVE_POPULATION"] = "ANY_NON_B_OUTPUT_IN_GOLD_B_HARD_NEGATIVE_POPULATION"
    semantic_comparator: Literal["PREDICATE_STATUS_QUERY_AND_SORTED_EVIDENCE_PINS"] = "PREDICATE_STATUS_QUERY_AND_SORTED_EVIDENCE_PINS"
    required_semantic_fields: tuple[Literal["candidate", "query_ref", "evidence"], ...] = ("candidate", "query_ref", "evidence")
    uncertainty_method: Literal["PAIRED_SOURCE_GROUP_PERCENTILE_BOOTSTRAP"] = "PAIRED_SOURCE_GROUP_PERCENTILE_BOOTSTRAP"
    uncertainty_repetitions: Literal[2000] = 2000
    uncertainty_seed: Literal[20260917] = 20260917
    uncertainty_confidence: Literal[0.95] = .95
    uncertainty_decision_rule: Literal["TWO_SIDED_95_PERCENT_BOUNDS_MUST_CLEAR_THRESHOLDS"] = "TWO_SIDED_95_PERCENT_BOUNDS_MUST_CLEAR_THRESHOLDS"
    timing_warmup_repetitions: Literal[2] = 2
    timing_measured_repetitions: Literal[5] = 5
    timing_arm_order: tuple[Literal["B1", "B2", "B5"], ...] = ("B1", "B2", "B5")
    timing_order_rule: Literal["ROTATE_ARM_ORDER_BY_CASE_AND_REPETITION"] = "ROTATE_ARM_ORDER_BY_CASE_AND_REPETITION"
    timing_boundary: Literal["SYNCHRONIZED_MODEL_PLUS_DETERMINISTIC_ASSEMBLY"] = "SYNCHRONIZED_MODEL_PLUS_DETERMINISTIC_ASSEMBLY"
    percentile_method: Literal["NEAREST_RANK"] = "NEAREST_RANK"

    @staticmethod
    def expected_guard_rates(denominators):
        rates = {"binding": 0.0}
        for language in ("EN", "ZH"):
            counts = denominators[language]
            rates[f"{language}:overall"] = 1 / counts["total"]
            for label in "ABC":
                rates[f"{language}:false_negative:{label}"] = 1 / counts[f"class_{label}"]
            for category in ("hard_negative", "qualification", "exception"):
                rates[f"{language}:{category}"] = 1 / counts[category]
        for category in ("hard_negative", "qualification", "exception"):
            rates[f"pooled:{category}"] = 1 / sum(denominators[lang][category] for lang in ("EN", "ZH"))
        return rates

    @classmethod
    def create(cls, **values):
        values.setdefault("guard_rates", cls.expected_guard_rates(values["denominators"]))
        return super().create(**values)

    @model_validator(mode="after")
    def populations_and_schedule(self):
        try:
            rates = self.expected_guard_rates(self.denominators)
        except (KeyError, TypeError, ZeroDivisionError):
            raise ValueError("RESEARCH_PROTOCOL_DENOMINATORS_INVALID") from None
        if set(self.denominators) != {"EN", "ZH"} or self.guard_rates != rates:
            raise ValueError("RESEARCH_PROTOCOL_GUARD_RATES_MISMATCH")
        if self.required_semantic_fields != ("candidate", "query_ref", "evidence") or self.timing_arm_order != ("B1", "B2", "B5"):
            raise ValueError("RESEARCH_PROTOCOL_OBLIGATIONS_OR_ORDER_MISMATCH")
        return self


class ResearchInputEvidenceV1(_ResearchProjection):
    """Frozen independent reference annotations or source-group exclusions."""
    REF_PREFIX = "research-input-evidence"
    ARTIFACT_KIND = "research_input_evidence"
    schema_version: Literal["research_input_evidence.v1"] = "research_input_evidence.v1"
    purpose: Literal["REFERENCE_LABELS", "SOURCE_GROUP_EXCLUSIONS", "PROMPT_TEMPLATE"]
    source_ref: str = Field(min_length=1)
    source_payload: dict[str, Any]
    source_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    producer_outputs_used: Literal[False] = False

    @model_validator(mode="after")
    def source_identity(self):
        if self.source_hash != "sha256:" + content_hash(self.source_payload):
            raise ValueError("RESEARCH_INPUT_SOURCE_HASH_MISMATCH")
        return self
