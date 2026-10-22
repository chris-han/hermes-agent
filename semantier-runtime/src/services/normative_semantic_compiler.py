"""Deterministic bounded compiler for the first A1 normative profile."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Mapping, TypeVar

from pydantic import BaseModel

from contracts.normative_semantics import (
    EntityDistinctionFrameV1,
    EntityDistinctionRuleEffectV1,
    ExceptionFrameV1,
    NormativeSemanticCandidateSetV1,
    PrecedenceFrameV1,
    QualificationFrameV1,
    RuleFrameV1,
    ScopeFrameV1,
    SemanticPredicateV1,
    TemporalConditionFrameV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_learning_trial import (
    AdvisoryCompilerInputsV1,
    AdvisoryTrialVariantV1,
)
from eos.artifact_hashing import content_hash


FrameT = TypeVar("FrameT", bound=BaseModel)

_FRAME_NATIVE_KEYS: dict[str, tuple[str, ...]] = {
    "entity_distinction": (
        "left_entity_ref",
        "right_entity_ref",
        "outcome_effect",
    ),
    "exception": ("applies_to_rule_ref", "conditions"),
    "precedence": (
        "higher_priority_ref",
        "lower_priority_ref",
        "condition_refs",
    ),
    "qualification": ("qualified_term", "definition"),
    "rule": (
        "modality",
        "actor_ref",
        "object_ref",
        "conditions",
        "consequence",
        "exception_refs",
        "scope_refs",
        "qualification_refs",
        "temporal_condition_refs",
        "precedence_refs",
    ),
    "scope": ("applies_to_rule_ref", "actor_scope", "object_scope"),
    "temporal_condition": (
        "effective_from",
        "effective_until",
        "event_time_ref",
    ),
}
_FRAME_COMMON_NATIVE_KEYS = (
    "frame_id",
    "display_name",
    "evidence_refs",
    "source_region_refs",
)
_INITIAL_FRAME_KINDS = (
    "entity_distinction",
    "exception",
    "precedence",
    "qualification",
    "rule",
    "scope",
    "temporal_condition",
)
_CANDIDATE_NATIVE_KEYS = (
    "organization_id",
    "workspace_id",
    "source_identity_ref",
    "source_hash",
    "canonical_source_ir_ref",
    "canonical_source_ir_hash",
    "competency_contract_ref",
    "competency_contract_hash",
    "normative_schema_ref",
    "normative_schema_hash",
    "profile_ref",
    "profile_hash",
    "compiler_ref",
    "compiler_hash",
    "prompt_hash",
    "provider_ref",
    "model_ref",
    "config_hash",
)
_TRIAL_COMPILER_REF_SUFFIX = "#advisory-trial-variant.v1"


class NormativeSemanticCompiler:
    def __init__(
        self,
        *,
        compiler_ref: str,
        compiler_hash: str,
        prompt_hash: str,
        provider_ref: str,
        model_ref: str,
        config_hash: str,
        compiled_at: datetime,
    ) -> None:
        self._pins = {
            "compiler_ref": compiler_ref,
            "compiler_hash": compiler_hash,
            "prompt_hash": prompt_hash,
            "provider_ref": provider_ref,
            "model_ref": model_ref,
            "config_hash": config_hash,
        }
        self._compiled_at = compiled_at

    @staticmethod
    def _frame(contract: type[FrameT], **payload: Any) -> FrameT:
        def jsonable(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json")
            if isinstance(value, tuple):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            return value

        digest = content_hash(jsonable(payload))
        return contract(frame_hash=f"sha256:{digest}", **payload)

    @staticmethod
    def _candidate_native_basis(
        source: NormativeSemanticCandidateSetV1 | Mapping[str, Any],
        *,
        frames: tuple[BaseModel, ...],
        overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        values = (
            source.model_dump(mode="json")
            if isinstance(source, NormativeSemanticCandidateSetV1)
            else dict(source)
        )
        if overrides:
            values.update(overrides)
        return {
            **{key: values[key] for key in _CANDIDATE_NATIVE_KEYS},
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }

    @staticmethod
    def _frame_native_basis(frame: BaseModel) -> dict[str, Any]:
        payload = frame.model_dump(mode="json")
        kind = str(payload.get("frame_kind", ""))
        try:
            keys = (*_FRAME_COMMON_NATIVE_KEYS, *_FRAME_NATIVE_KEYS[kind])
        except KeyError as exc:
            raise ValueError(f"unsupported native frame kind: {kind}") from exc
        if kind in {"scope", "qualification"} and "conditions" in payload:
            keys = (*keys, "conditions")
        if kind == "entity_distinction" and "rule_effects" in payload:
            keys = (*keys, "rule_effects")
        return {key: payload[key] for key in keys}

    @classmethod
    def _verify_initial_native_source(
        cls, source: NormativeSemanticCandidateSetV1
    ) -> None:
        if (
            source.change_cause != "initial_compilation"
            or source.predecessor_candidate_set_ref is not None
            or source.predecessor_candidate_set_hash is not None
            or source.triggering_ref is not None
        ):
            raise ValueError("trial source must be an initial compiler output")
        if (
            source.completeness_state != "COMPLETE"
            or source.unresolved_slot_refs != ()
        ):
            raise ValueError("trial source must retain initial completeness diagnostics")
        if tuple(frame.frame_kind for frame in source.frames) != _INITIAL_FRAME_KINDS:
            raise ValueError("trial source has an incompatible native compiler frame shape")
        for frame in source.frames:
            if (
                frame.display_name_zh is not None
                or frame.epistemic_status != "FOUND"
                or frame.frame_kind not in _FRAME_NATIVE_KEYS
            ):
                raise ValueError("trial source frame has incompatible top-level defaults")
            expected_hash = "sha256:" + content_hash(cls._frame_native_basis(frame))
            if frame.frame_hash != expected_hash:
                raise ValueError("trial source frame does not match native compiler identity")
        digest = content_hash(
            cls._candidate_native_basis(source, frames=tuple(source.frames))
        )
        if (
            source.candidate_set_ref != f"normative-candidate-set:{digest}"
            or source.candidate_set_hash != f"sha256:{digest}"
        ):
            raise ValueError("trial source does not match native compiler identity")

    def _verify_trial_inputs(
        self,
        *,
        source: NormativeSemanticCandidateSetV1,
        inputs: AdvisoryCompilerInputsV1,
    ) -> None:
        if (
            source.organization_id != inputs.organization_id
            or source.workspace_id != inputs.workspace_id
        ):
            raise ValueError("trial compiler source/input scope mismatch")
        source_producer = {
            "compiler_ref": source.compiler_ref,
            "compiler_hash": source.compiler_hash,
            "prompt_hash": source.prompt_hash,
            "provider_ref": source.provider_ref,
            "model_ref": source.model_ref,
            "config_hash": source.config_hash,
        }
        if self._pins != source_producer:
            raise ValueError("trial source producer does not match compiler instance")
        semantic = inputs.resource("semantic_compiler")
        prompt = inputs.resource("prompt")
        configuration = inputs.resource("configuration")
        frame_schema = inputs.resource("frame_schema")
        profile = inputs.resource("profile")
        if any(
            (
                semantic.resource_ref != source.compiler_ref,
                semantic.bytes_hash != source.compiler_hash,
                prompt.bytes_hash != source.prompt_hash,
                configuration.bytes_hash != source.config_hash,
                frame_schema.resource_ref != source.normative_schema_ref,
                frame_schema.bytes_hash != source.normative_schema_hash,
                profile.resource_ref != source.profile_ref,
                profile.bytes_hash != source.profile_hash,
                inputs.provider_ref != source.provider_ref,
                inputs.model_ref != source.model_ref,
                inputs.competency.artifact_ref != source.competency_contract_ref,
                inputs.competency.artifact_hash != source.competency_contract_hash,
            )
        ):
            raise ValueError("trial compiler producer inputs do not match source")

    @staticmethod
    def _candidate_pin(candidate: NormativeSemanticCandidateSetV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="normative_semantic_candidate_set",
            artifact_ref=candidate.candidate_set_ref,
            artifact_hash=candidate.candidate_set_hash,
        )

    def compile_trial_variant(
        self,
        *,
        source: NormativeSemanticCandidateSetV1,
        inputs: AdvisoryCompilerInputsV1,
        variant: Literal[
            "OMIT_EXCEPTION", "PROXY_THRESHOLD_20", "PROXY_THRESHOLD_10"
        ],
        recorded_at: datetime,
    ) -> tuple[NormativeSemanticCandidateSetV1, AdvisoryTrialVariantV1]:
        if variant not in {
            "OMIT_EXCEPTION",
            "PROXY_THRESHOLD_20",
            "PROXY_THRESHOLD_10",
        }:
            raise ValueError(f"unsupported trial compiler variant: {variant}")
        self._verify_initial_native_source(source)
        self._verify_trial_inputs(source=source, inputs=inputs)

        if variant == "OMIT_EXCEPTION":
            exceptions = tuple(
                frame for frame in source.frames if frame.frame_kind == "exception"
            )
            if len(exceptions) != 1:
                raise ValueError("OMIT_EXCEPTION requires exactly one exception frame")
            frames = tuple(frame for frame in source.frames if frame is not exceptions[0])
        else:
            matches = tuple(
                (frame, index)
                for frame in source.frames
                if isinstance(frame, RuleFrameV1)
                for index, predicate in enumerate(frame.conditions)
                if predicate.operator == "GE"
            )
            if len(matches) != 1:
                raise ValueError("threshold variant requires exactly one GE predicate")
            target_rule, predicate_index = matches[0]
            threshold = 20 if variant == "PROXY_THRESHOLD_20" else 10
            conditions = tuple(
                predicate.model_copy(update={"expected_value": threshold})
                if index == predicate_index
                else predicate
                for index, predicate in enumerate(target_rule.conditions)
            )
            rule_basis = self._frame_native_basis(target_rule)
            rule_basis["conditions"] = conditions
            changed_rule = self._frame(RuleFrameV1, **rule_basis)
            frames = tuple(
                changed_rule if frame.frame_id == target_rule.frame_id else frame
                for frame in source.frames
            )

        semantic = inputs.resource("semantic_compiler")
        trial_pins = {
            "competency_contract_ref": inputs.competency.artifact_ref,
            "competency_contract_hash": inputs.competency.artifact_hash,
            "normative_schema_ref": inputs.resource("frame_schema").resource_ref,
            "normative_schema_hash": inputs.resource("frame_schema").bytes_hash,
            "profile_ref": inputs.resource("profile").resource_ref,
            "profile_hash": inputs.resource("profile").bytes_hash,
            "compiler_ref": semantic.resource_ref + _TRIAL_COMPILER_REF_SUFFIX,
            "compiler_hash": semantic.bytes_hash,
            "prompt_hash": inputs.resource("prompt").bytes_hash,
            "provider_ref": inputs.provider_ref,
            "model_ref": inputs.model_ref,
            "config_hash": inputs.resource("configuration").bytes_hash,
        }
        basis = self._candidate_native_basis(
            source, frames=frames, overrides=trial_pins
        )
        digest = content_hash(basis)
        payload = source.model_dump(mode="json")
        payload.update(
            {
                **trial_pins,
                "candidate_set_ref": f"normative-candidate-set:{digest}",
                "candidate_set_hash": f"sha256:{digest}",
                "frames": [frame.model_dump(mode="json") for frame in frames],
                "predecessor_candidate_set_ref": None,
                "predecessor_candidate_set_hash": None,
                "change_cause": "initial_compilation",
                "triggering_ref": None,
                "compiled_at": recorded_at,
            }
        )
        candidate = NormativeSemanticCandidateSetV1.model_validate(payload)
        derivation = AdvisoryTrialVariantV1.create(
            organization_id=source.organization_id,
            workspace_id=source.workspace_id,
            source=self._candidate_pin(source),
            inputs=ArtifactPinV1(
                artifact_kind="advisory_compiler_inputs",
                artifact_ref=inputs.input_ref,
                artifact_hash=inputs.input_hash,
            ),
            variant=variant,
            candidate=self._candidate_pin(candidate),
            added_frame=None,
            recorded_at=recorded_at,
        )
        return candidate, derivation

    def compile(
        self,
        *,
        source: dict[str, Any],
        organization_id: str,
        workspace_id: str,
        source_identity_ref: str,
        source_hash: str,
        canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
        competency_contract_ref: str,
        competency_contract_hash: str,
        normative_schema_ref: str,
        normative_schema_hash: str,
        profile_ref: str,
        profile_hash: str,
    ) -> NormativeSemanticCandidateSetV1:
        sections = {
            str(item["section_id"]): item
            for item in source.get("source_sections", [])
            if isinstance(item, dict) and item.get("section_id")
        }
        required = {
            "policy.scope",
            "policy.rule",
            "policy.qualification",
            "policy.exception",
            "policy.precedence",
            "policy.entity_distinction",
            "policy.effectivity",
        }
        missing = sorted(required - sections.keys())
        if missing:
            raise ValueError(f"required canonical source regions are missing: {', '.join(missing)}")

        def evidence(section_id: str) -> tuple[str, ...]:
            return (str(sections[section_id]["evidence_ref"]),)

        rule_id = "rule.finance_approval"
        exception_id = "exception.emergency_security"
        scope_id = "scope.operations_software"
        qualification_id = "qualification.annual_committed_value"
        temporal_id = "temporal.policy_2026"
        precedence_id = "precedence.emergency_over_general"
        frames = [
            self._frame(
                EntityDistinctionFrameV1,
                frame_id="entity_distinction.request_vs_order",
                display_name="Purchase request versus purchase order",
                evidence_refs=evidence("policy.entity_distinction"),
                source_region_refs=("policy.entity_distinction",),
                left_entity_ref="entity.purchase_request",
                right_entity_ref="entity.purchase_order",
                outcome_effect="request approval does not transfer to a higher-value order",
                rule_effects=(
                    EntityDistinctionRuleEffectV1(rule_ref=rule_id, effect="applicability", conditions=(
                        SemanticPredicateV1(predicate_id="predicate.executed_purchase_order", operator="EQ",
                            fact_ref="fact.executed_purchase_order", expected_value=True,
                            evidence_refs=evidence("policy.entity_distinction")),)),
                    EntityDistinctionRuleEffectV1(rule_ref=rule_id, effect="input_requirement", conditions=(
                        SemanticPredicateV1(predicate_id="predicate.current_order_value", operator="EQ",
                            fact_ref="fact.committed_value_subject", expected_value="current_purchase_order",
                            evidence_refs=evidence("policy.entity_distinction")),)),
                ),
            ),
            self._frame(
                ExceptionFrameV1,
                frame_id=exception_id,
                display_name="Emergency security remediation exemption",
                evidence_refs=evidence("policy.exception"),
                source_region_refs=("policy.exception",),
                applies_to_rule_ref=rule_id,
                conditions=(
                    SemanticPredicateV1(
                        predicate_id="predicate.incident_ref_exists",
                        operator="EXISTS",
                        fact_ref="fact.incident_ref",
                        evidence_refs=evidence("policy.exception"),
                    ),
                    SemanticPredicateV1(
                        predicate_id="predicate.commitment_le_25000",
                        operator="LE",
                        fact_ref="fact.annual_committed_value_usd",
                        expected_value=25000,
                        evidence_refs=evidence("policy.exception"),
                    ),
                ),
            ),
            self._frame(
                PrecedenceFrameV1,
                frame_id=precedence_id,
                display_name="Emergency exemption precedence",
                evidence_refs=evidence("policy.precedence"),
                source_region_refs=("policy.precedence",),
                higher_priority_ref=exception_id,
                lower_priority_ref=rule_id,
                condition_refs=("predicate.commitment_le_25000",),
            ),
            self._frame(
                QualificationFrameV1,
                frame_id=qualification_id,
                display_name="Annual committed value definition",
                evidence_refs=evidence("policy.qualification"),
                source_region_refs=("policy.qualification",),
                qualified_term="annual_committed_value",
                definition=str(sections["policy.qualification"]["text"]),
                conditions=tuple(SemanticPredicateV1(
                    predicate_id=f"predicate.qualification.{name}", operator="EQ",
                    fact_ref=f"fact.{name}", expected_value=value,
                    evidence_refs=evidence("policy.qualification")) for name, value in (
                        ("value_period_months", 12),
                        ("value_includes_only_non_cancellable_subscription_fees", True),
                        ("value_excludes_separately_cancellable_usage", True))),
            ),
            self._frame(
                RuleFrameV1,
                frame_id=rule_id,
                display_name="Finance approval requirement",
                evidence_refs=evidence("policy.rule"),
                source_region_refs=("policy.rule",),
                modality="obligation",
                actor_ref="actor.operations_department",
                object_ref="object.software_subscription_commitment",
                conditions=(
                    SemanticPredicateV1(
                        predicate_id="predicate.value_ge_10000",
                        operator="GE",
                        fact_ref="fact.annual_committed_value_usd",
                        expected_value=10000,
                        evidence_refs=evidence("policy.rule"),
                    ),
                ),
                consequence="finance_approval_required_before_commitment",
                exception_refs=(exception_id,),
                scope_refs=(scope_id,),
                qualification_refs=(qualification_id,),
                temporal_condition_refs=(temporal_id,),
                precedence_refs=(precedence_id,),
            ),
            self._frame(
                ScopeFrameV1,
                frame_id=scope_id,
                display_name="Operations software operating-budget scope",
                evidence_refs=evidence("policy.scope"),
                source_region_refs=("policy.scope",),
                applies_to_rule_ref=rule_id,
                actor_scope=("actor.operations_department",),
                object_scope=("object.software_subscription", "fund.operating_budget"),
                conditions=tuple(SemanticPredicateV1(
                    predicate_id="predicate.scope." + name, operator="EQ", fact_ref="fact." + name,
                    expected_value=value, evidence_refs=evidence("policy.scope"))
                    for name, value in (("department", "operations"), ("item_category", "software_subscription"),
                                        ("funding_source", "operating_budget"))),
            ),
            self._frame(
                TemporalConditionFrameV1,
                frame_id=temporal_id,
                display_name="2026 policy effectivity",
                evidence_refs=evidence("policy.effectivity"),
                source_region_refs=("policy.effectivity",),
                effective_from="2026-01-01",
                effective_until="2026-12-31",
                event_time_ref="fact.commitment_date",
            ),
        ]
        frames.sort(key=lambda frame: frame.frame_id.encode("utf-8"))
        candidate_values = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_identity_ref": source_identity_ref,
            "source_hash": source_hash,
            "canonical_source_ir_ref": canonical_source_ir_ref,
            "canonical_source_ir_hash": canonical_source_ir_hash,
            "competency_contract_ref": competency_contract_ref,
            "competency_contract_hash": competency_contract_hash,
            "normative_schema_ref": normative_schema_ref,
            "normative_schema_hash": normative_schema_hash,
            "profile_ref": profile_ref,
            "profile_hash": profile_hash,
            **self._pins,
        }
        basis = self._candidate_native_basis(
            candidate_values, frames=tuple(frames)
        )
        digest = content_hash(basis)
        return NormativeSemanticCandidateSetV1(
            candidate_set_ref=f"normative-candidate-set:{digest}",
            candidate_set_hash=f"sha256:{digest}",
            frames=tuple(frames),
            completeness_state="COMPLETE",
            unresolved_slot_refs=(),
            compiled_at=self._compiled_at,
            **{key: value for key, value in basis.items() if key != "frames"},
        )


__all__ = ["NormativeSemanticCompiler"]
