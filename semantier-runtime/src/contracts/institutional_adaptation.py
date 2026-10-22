"""Derived evidence contracts for pre-learning institutional adaptation triage.

These records are content-addressed analytical evidence only. They do not extend
Semantier's five-object semantic kernel and carry no admission or activation
authority.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA256 = r"^sha256:[0-9a-f]{64}$"


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


def _ordered_unique_pins(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    pins = tuple(
        item if isinstance(item, ArtifactPinV1) else ArtifactPinV1.model_validate(item)
        for item in value
    )
    keys = tuple(_pin_key(pin) for pin in pins)
    if len(keys) != len(set(keys)):
        raise ValueError("pin collection contains duplicate identities")
    return tuple(sorted(pins, key=_pin_key))


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def timestamps_are_aware_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class _ContentAddressedEvidence(_Frozen):
    ARTIFACT_KIND: ClassVar[str]
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=_SHA256)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "artifact_ref": "pending",
                "artifact_hash": "sha256:" + "0" * 64,
                **values,
            },
            context={"content_addressed_draft": True},
        )
        normalized = draft.model_dump(
            mode="python", exclude={"artifact_ref", "artifact_hash"}
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        return cls.model_validate(
            {
                "artifact_ref": f"{cls.ARTIFACT_KIND}:{digest}",
                "artifact_hash": f"sha256:{digest}",
                **normalized,
            }
        )

    @model_validator(mode="after")
    def identity_binds_content(self, info: ValidationInfo):
        if info.context and info.context.get("content_addressed_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        if self.artifact_ref != f"{type(self).ARTIFACT_KIND}:{digest}":
            raise ValueError("artifact ref does not bind exact content")
        if self.artifact_hash != f"sha256:{digest}":
            raise ValueError("artifact hash does not bind exact content")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=type(self).ARTIFACT_KIND,
            artifact_ref=self.artifact_ref,
            artifact_hash=self.artifact_hash,
        )



InstitutionalDimension = Literal[
    "OBLIGATION",
    "AUTHORITY",
    "EVIDENCE",
    "THRESHOLD",
    "EXCEPTION",
    "PRECEDENCE",
    "TEMPORAL_EFFECT",
    "APPLICABILITY",
    "DECISION_FRONTIER",
    "COMMITMENT_AUTHORITY",
]


class InstitutionalDivergenceDetectionV1(_ContentAddressedEvidence):
    """Deterministic materiality result over one pinned baseline/local delta."""

    ARTIFACT_KIND = "institutional_divergence_detection"
    schema_version: Literal["semantier.institutional_divergence_detection.v1"] = (
        "semantier.institutional_divergence_detection.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    global_baseline: ArtifactPinV1
    local_successor: ArtifactPinV1
    semantic_delta: ArtifactPinV1
    changed_dimensions: tuple[InstitutionalDimension, ...] = ()
    presentation_equivalent: bool = False
    classification: Literal[
        "NO_DIVERGENCE",
        "NON_MATERIAL_DIVERGENCE",
        "MATERIAL_INSTITUTIONAL_DIVERGENCE",
        "UNRESOLVED_MATERIALITY",
    ]
    reason_codes: tuple[str, ...] = Field(min_length=1)
    detected_at: datetime

    @field_validator("changed_dimensions", "reason_codes", mode="before")
    @classmethod
    def canonicalize_values(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(sorted(set(str(item) for item in value)))

    @model_validator(mode="after")
    def validate_detection(self):
        same_artifact = self.global_baseline == self.local_successor
        if same_artifact and self.classification != "NO_DIVERGENCE":
            raise ValueError("identical baseline/successor must be NO_DIVERGENCE")
        if self.classification == "NO_DIVERGENCE":
            if not same_artifact or self.changed_dimensions or not self.presentation_equivalent:
                raise ValueError("NO_DIVERGENCE requires identical presentation-equivalent artifacts")
        elif self.classification == "NON_MATERIAL_DIVERGENCE":
            if same_artifact or self.changed_dimensions or not self.presentation_equivalent:
                raise ValueError("NON_MATERIAL_DIVERGENCE must be presentation-equivalent only")
        elif self.classification == "MATERIAL_INSTITUTIONAL_DIVERGENCE":
            if same_artifact or not self.changed_dimensions or self.presentation_equivalent:
                raise ValueError("material institutional divergence requires semantic dimensions")
        elif self.classification == "UNRESOLVED_MATERIALITY":
            if same_artifact or self.changed_dimensions or self.presentation_equivalent:
                raise ValueError("unresolved materiality is a changed unclassified delta")
        return self


class AdaptationObservationV1(_ContentAddressedEvidence):
    ARTIFACT_KIND = "adaptation_observation"
    schema_version: Literal["semantier.adaptation_observation.v1"] = (
        "semantier.adaptation_observation.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    global_baseline: ArtifactPinV1
    local_successor: ArtifactPinV1
    semantic_delta: ArtifactPinV1
    local_context: ArtifactPinV1
    jurisdiction: ArtifactPinV1 | None = None
    authority_basis_refs: tuple[ArtifactPinV1, ...] = ()
    law_contract_policy_refs: tuple[ArtifactPinV1, ...] = ()
    outcome_or_simulation_refs: tuple[ArtifactPinV1, ...] = ()
    actor_lineage_refs: tuple[ArtifactPinV1, ...] = ()
    effective_from: datetime
    effective_to: datetime | None = None
    recorded_at: datetime

    @field_validator(
        "authority_basis_refs",
        "law_contract_policy_refs",
        "outcome_or_simulation_refs",
        "actor_lineage_refs",
        mode="before",
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_window(self):
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to must not precede effective_from")
        return self


TriageDisposition = Literal[
    "MANDATED_LOCALIZATION",
    "AUTHORITY_LOCALIZATION",
    "BENEFICIAL_LOCAL_OPTIMIZATION",
    "CONTEXTUAL_PREFERENCE",
    "POLICY_EROSION",
    "INSUFFICIENT_ATTRIBUTION",
]


class AdaptationTriageReceiptV1(_ContentAddressedEvidence):
    ARTIFACT_KIND = "adaptation_triage_receipt"
    schema_version: Literal["semantier.adaptation_triage_receipt.v1"] = (
        "semantier.adaptation_triage_receipt.v1"
    )
    adaptation_observation: ArtifactPinV1
    disposition: TriageDisposition
    reason_codes: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[ArtifactPinV1, ...] = ()
    protected_invariant_refs: tuple[ArtifactPinV1, ...] = ()
    learning_eligible: bool
    eligible_scope: Literal["NONE", "LOCAL", "CROSS_CONTEXT_INPUT"]
    review_required: bool
    recorded_at: datetime

    @field_validator("evidence_refs", "protected_invariant_refs", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("reason_codes", mode="before")
    @classmethod
    def canonicalize_reason_codes(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        values = tuple(sorted(set(str(item) for item in value)))
        if not all(values):
            raise ValueError("reason codes must be non-empty")
        return values

    @model_validator(mode="after")
    def validate_triage_semantics(self):
        if self.adaptation_observation.artifact_kind != "adaptation_observation":
            raise ValueError("triage receipt must pin adaptation observation")
        if self.disposition == "INSUFFICIENT_ATTRIBUTION":
            if self.learning_eligible or self.eligible_scope != "NONE":
                raise ValueError("insufficient attribution cannot enter learning")
        elif not self.learning_eligible:
            raise ValueError("attributed adaptation dispositions remain learning opportunities")
        if self.disposition == "POLICY_EROSION":
            if self.eligible_scope != "CROSS_CONTEXT_INPUT" or not self.review_required:
                raise ValueError("policy erosion must remain reviewable cross-context evidence")
        return self



class InstitutionalAdaptationIngressV1(_ContentAddressedEvidence):
    """Conditional bridge from pre-learning triage into the ordinary learning loop."""

    ARTIFACT_KIND = "institutional_adaptation_ingress"
    schema_version: Literal["semantier.institutional_adaptation_ingress.v1"] = (
        "semantier.institutional_adaptation_ingress.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    acquisition: ArtifactPinV1
    divergence_detection: ArtifactPinV1
    adaptation_observation: ArtifactPinV1
    triage_receipt: ArtifactPinV1
    disposition: TriageDisposition
    route: Literal["ATTRIBUTE", "LOCAL_ONLY", "DESIGN_PRESSURE_ONLY", "HOLD"]
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_ingress(self):
        if self.acquisition.artifact_kind != "acquisition_decision":
            raise ValueError("adaptation ingress requires acquisition decision")
        if self.divergence_detection.artifact_kind != "institutional_divergence_detection":
            raise ValueError("adaptation ingress requires divergence detection")
        if self.adaptation_observation.artifact_kind != "adaptation_observation":
            raise ValueError("adaptation ingress requires adaptation observation")
        if self.triage_receipt.artifact_kind != "adaptation_triage_receipt":
            raise ValueError("adaptation ingress requires triage receipt")
        expected_route = {
            "MANDATED_LOCALIZATION": "ATTRIBUTE",
            "AUTHORITY_LOCALIZATION": "ATTRIBUTE",
            "BENEFICIAL_LOCAL_OPTIMIZATION": "ATTRIBUTE",
            "CONTEXTUAL_PREFERENCE": "LOCAL_ONLY",
            "POLICY_EROSION": "DESIGN_PRESSURE_ONLY",
            "INSUFFICIENT_ATTRIBUTION": "HOLD",
        }[self.disposition]
        if self.route != expected_route:
            raise ValueError("adaptation ingress route/disposition mismatch")
        return self


class SemanticAdaptationClusterV1(_ContentAddressedEvidence):
    ARTIFACT_KIND = "semantic_adaptation_cluster"
    schema_version: Literal["semantier.semantic_adaptation_cluster.v1"] = (
        "semantier.semantic_adaptation_cluster.v1"
    )
    global_baseline: ArtifactPinV1
    canonical_adaptation_key: str = Field(min_length=1)
    member_triage_receipt_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    normalized_effect_signature: str = Field(min_length=1)
    applicability_signature: str = Field(min_length=1)
    cluster_method_version: str = Field(min_length=1)

    @field_validator("member_triage_receipt_refs", mode="before")
    @classmethod
    def canonicalize_members(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_members(self):
        if any(
            pin.artifact_kind != "adaptation_triage_receipt"
            for pin in self.member_triage_receipt_refs
        ):
            raise ValueError("semantic cluster members must be triage receipts")
        return self


class AdaptationIndependenceClusterV1(_ContentAddressedEvidence):
    ARTIFACT_KIND = "adaptation_independence_cluster"
    schema_version: Literal["semantier.adaptation_independence_cluster.v1"] = (
        "semantier.adaptation_independence_cluster.v1"
    )
    semantic_cluster: ArtifactPinV1
    member_adaptation_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    causal_basis_refs: tuple[ArtifactPinV1, ...] = ()
    dependency_reason_codes: tuple[str, ...] = ()
    independence_status: Literal["INDEPENDENT", "DEPENDENT", "UNRESOLVED"]
    representative_support_ref: ArtifactPinV1 | None = None

    @field_validator(
        "member_adaptation_refs", "causal_basis_refs", mode="before"
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("dependency_reason_codes", mode="before")
    @classmethod
    def canonicalize_reasons(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(sorted(set(str(item) for item in value)))

    @model_validator(mode="after")
    def validate_independence(self):
        if self.semantic_cluster.artifact_kind != "semantic_adaptation_cluster":
            raise ValueError("independence cluster must pin semantic cluster")
        if any(
            pin.artifact_kind != "adaptation_observation"
            for pin in self.member_adaptation_refs
        ):
            raise ValueError("independence members must be adaptation observations")
        if self.independence_status != "INDEPENDENT" and self.representative_support_ref is not None:
            raise ValueError("only independent clusters may name representative support")
        return self


class DesignPressureOpportunityV1(_ContentAddressedEvidence):
    """Negative-evidence learning opportunity created from repeated policy erosion."""

    ARTIFACT_KIND = "design_pressure_opportunity"
    schema_version: Literal["semantier.design_pressure_opportunity.v1"] = (
        "semantier.design_pressure_opportunity.v1"
    )
    global_baseline: ArtifactPinV1
    semantic_cluster: ArtifactPinV1
    erosion_independence_cluster_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    distinct_context_family_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    protected_invariant_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    authority_state: Literal["EVIDENCE_ONLY"] = "EVIDENCE_ONLY"
    positive_support_units: Literal[0] = 0

    @field_validator(
        "erosion_independence_cluster_refs",
        "distinct_context_family_refs",
        "protected_invariant_refs",
        mode="before",
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_design_pressure(self):
        if self.semantic_cluster.artifact_kind != "semantic_adaptation_cluster":
            raise ValueError("design pressure must pin semantic adaptation cluster")
        if any(
            pin.artifact_kind != "adaptation_independence_cluster"
            for pin in self.erosion_independence_cluster_refs
        ):
            raise ValueError("design pressure support must pin independence clusters")
        return self


class CrossContextAdaptationHypothesisV1(_ContentAddressedEvidence):
    ARTIFACT_KIND = "cross_context_adaptation_hypothesis"
    schema_version: Literal["semantier.cross_context_adaptation_hypothesis.v1"] = (
        "semantier.cross_context_adaptation_hypothesis.v1"
    )
    global_baseline: ArtifactPinV1
    semantic_cluster: ArtifactPinV1
    supporting_independence_cluster_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    distinct_context_family_refs: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    counterexample_refs: tuple[ArtifactPinV1, ...] = ()
    triage_distribution: dict[TriageDisposition, int]
    outcome_witness_refs: tuple[ArtifactPinV1, ...] = ()
    erosion_screen: ArtifactPinV1
    common_cause_screen: ArtifactPinV1
    protected_invariant_screen: ArtifactPinV1
    applicability_hypothesis: str = Field(min_length=1)
    threshold_stage: Literal["A", "B"]
    authority_state: Literal["CANDIDATE_ONLY"] = "CANDIDATE_ONLY"
    target_kind: Literal["WORKFLOW", "POLICY", "POLICY_FORMATION"]

    @field_validator(
        "supporting_independence_cluster_refs",
        "distinct_context_family_refs",
        "counterexample_refs",
        "outcome_witness_refs",
        mode="before",
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_hypothesis(self):
        if self.semantic_cluster.artifact_kind != "semantic_adaptation_cluster":
            raise ValueError("hypothesis must pin semantic adaptation cluster")
        if any(
            pin.artifact_kind != "adaptation_independence_cluster"
            for pin in self.supporting_independence_cluster_refs
        ):
            raise ValueError("hypothesis support must pin independence clusters")
        if self.triage_distribution.get("POLICY_EROSION", 0) > 0:
            raise ValueError("policy erosion cannot be positive promotion precedent")
        return self


__all__ = [
    "AdaptationIndependenceClusterV1",
    "AdaptationObservationV1",
    "AdaptationTriageReceiptV1",
    "CrossContextAdaptationHypothesisV1",
    "DesignPressureOpportunityV1",
    "InstitutionalAdaptationIngressV1",
    "InstitutionalDivergenceDetectionV1",
    "InstitutionalDimension",
    "SemanticAdaptationClusterV1",
    "TriageDisposition",
]
