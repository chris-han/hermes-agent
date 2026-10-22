"""Governed projection from definition-bound Tender observations to ObservedFactsV1."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path

from contracts.artifact_pin import ArtifactPinV1
from contracts.normative_schema import NormativeSchemaContractV1
from contracts.ontology_paired_evaluation import ObservedFactsV1
from contracts.ontology_validation import CompiledPredicateV1, CompiledRulePlanV1
from contracts.tender_observed_facts_projection import (
    TenderFactSubjectBindingV1,
    TenderCategoricalFactSensorContractV1,
    TenderCategoricalFactObservationReceiptV1,
    TenderFactObservationReceiptV1,
    TenderFactSensorContractV1,
    TenderObservedFactsProjectionReceiptV1,
)
from contracts.tender_document_date_observation import (
    TenderDocumentDateObservationV1,
)
from contracts.semantic_kernel import ContextGraph, Evidence
from eos.artifact_hashing import content_hash
from storage.ontology_build_store import OntologyBuildStore


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


class TenderObservedFactsProjectionService:
    """Materialize only facts already established by governed producers.

    This service is deliberately not a semantic interpreter.  It checks that
    each interpreted Boolean observation is bound to the exact source-derived
    normative fact definition, maps TRUE/FALSE to Boolean values, preserves
    UNKNOWN/missing facts as absence, and emits replayable provenance.
    """


    @staticmethod
    def resolve_subject(*, store, subject_binding):
        """Resolve one exact SOURCE assertion and located kernel Evidence occurrence."""
        binding = TenderFactSubjectBindingV1.model_validate(subject_binding.model_dump(mode="json"))
        scope = dict(organization_id=binding.organization_id, workspace_id=binding.workspace_id)
        context = store.get(**scope, artifact_ref=binding.context_graph.artifact_ref, contract_type=ContextGraph)
        if context is None or context.pin() != binding.context_graph:
            raise ValueError("SUBJECT_CONTEXT_PIN_MISMATCH")
        nodes = {n.node_id for n in context.nodes}
        if not {binding.actor_node_ref, binding.condition_node_ref} <= nodes:
            raise ValueError("SUBJECT_NODE_MISSING")
        assertion = next((a for a in context.assertions if a.assertion_id == binding.association_assertion_ref), None)
        if (assertion is None or assertion.origin != "SOURCE"
            or assertion.subject_ref != binding.actor_node_ref or assertion.object_ref != binding.condition_node_ref
            or "sha256:" + content_hash(assertion.model_dump(mode="json")) != binding.association_assertion_hash
            or binding.occurrence_evidence not in assertion.evidence):
            raise ValueError("SUBJECT_ASSOCIATION_MISMATCH")
        evidence = store.get(**scope, artifact_ref=binding.occurrence_evidence.artifact_ref, contract_type=Evidence)
        if evidence is None or evidence.pin() != binding.occurrence_evidence:
            raise ValueError("SUBJECT_OCCURRENCE_PIN_MISMATCH")
        if ((evidence.source_ref, evidence.source_hash, evidence.selector_hash)
            != (binding.source_ref, binding.source_hash, binding.selector_hash)
            or binding.canonical_source_ir not in evidence.provenance):
            raise ValueError("SUBJECT_SOURCE_PIN_MISMATCH")
        if evidence.evidence_role != "target":
            raise ValueError("CASE_TARGET_EVIDENCE_REQUIRED")
        if not evidence.selector.source_element_ref or not evidence.selector.exact_quote:
            raise ValueError("EXACT_LOCATED_OCCURRENCE_REQUIRED")
        return context, assertion, evidence

    @staticmethod
    def _projector_pin() -> ArtifactPinV1:
        digest = sha256(Path(__file__).read_bytes()).hexdigest()
        return ArtifactPinV1(
            artifact_kind="tender_observed_facts_projector",
            artifact_ref=f"tender-observed-facts-projector:v1:{digest}",
            artifact_hash=f"sha256:{digest}",
        )

    @staticmethod
    def _plan_pin(plan: CompiledRulePlanV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="compiled_rule_plan",
            artifact_ref=plan.plan_ref,
            artifact_hash=plan.plan_hash,
        )

    @staticmethod
    def _schema_pin(schema: NormativeSchemaContractV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="normative_schema_contract",
            artifact_ref=schema.schema_ref,
            artifact_hash=schema.schema_hash,
        )

    @staticmethod
    def _facts_pin(facts: ObservedFactsV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="paired_observed_facts",
            artifact_ref=facts.facts_ref,
            artifact_hash=facts.facts_hash,
        )

    @staticmethod
    def _predicate_fact_refs(predicate: CompiledPredicateV1) -> tuple[str, ...]:
        refs = [predicate.fact_ref]
        if predicate.comparison_fact_ref is not None:
            refs.append(predicate.comparison_fact_ref)
        return tuple(refs)

    @classmethod
    def required_fact_refs(cls, plan: CompiledRulePlanV1) -> tuple[str, ...]:
        refs: set[str] = set()
        for rule in plan.rules:
            for predicate in (
                *rule.predicates,
                *rule.exception_predicates,
                *(rule.input_requirements or ()),
                *(rule.disposition_effects or ()),
            ):
                refs.update(cls._predicate_fact_refs(predicate))
        for precedence in plan.precedences or ():
            for predicate in precedence.conditions:
                refs.update(cls._predicate_fact_refs(predicate))
        if not refs:
            raise ValueError("COMPILED_PLAN_REQUIRES_NO_FACTS")
        return tuple(sorted(refs))

    def project(
        self,
        *,
        store: OntologyBuildStore,
        plan: CompiledRulePlanV1,
        schema: NormativeSchemaContractV1,
        target_evidence: tuple[ArtifactPinV1, ...],
        sensor_contracts: tuple[TenderFactSensorContractV1, ...],
        observations: tuple[TenderFactObservationReceiptV1, ...],
        frozen_at: datetime,
        document_date_observations: tuple[TenderDocumentDateObservationV1, ...] = (),
        subject_binding: TenderFactSubjectBindingV1 | None = None,
        categorical_observations: tuple[TenderCategoricalFactObservationReceiptV1, ...] = (),
    ) -> tuple[ObservedFactsV1, TenderObservedFactsProjectionReceiptV1]:
        plan = CompiledRulePlanV1.model_validate(plan.model_dump(mode="json"))
        schema = NormativeSchemaContractV1.model_validate(schema.model_dump(mode="json"))

        scope = (plan.organization_id, plan.workspace_id)
        if (schema.organization_id, schema.workspace_id) != scope:
            raise ValueError("FOREIGN_SCOPE")

        target_evidence = tuple(sorted(target_evidence, key=_pin_key))
        if not target_evidence or any(
            pin.artifact_kind != "evidence" for pin in target_evidence
        ):
            raise ValueError("TARGET_EVIDENCE_REQUIRED")
        if len({_pin_key(pin) for pin in target_evidence}) != len(target_evidence):
            raise ValueError("TARGET_EVIDENCE_MUST_BE_UNIQUE")
        target_pins = {_pin_key(pin) for pin in target_evidence}
        if categorical_observations and subject_binding is None:
            raise ValueError("CATEGORICAL_SUBJECT_BINDING_REQUIRED")
        if subject_binding is not None:
            if (subject_binding.organization_id, subject_binding.workspace_id) != scope:
                raise ValueError("FOREIGN_SCOPE")
            self.resolve_subject(store=store, subject_binding=subject_binding)
            if _pin_key(subject_binding.occurrence_evidence) not in target_pins:
                raise ValueError("SUBJECT_OCCURRENCE_OUTSIDE_TARGET")

        required = self.required_fact_refs(plan)
        definitions = {
            definition.fact_ref: definition
            for definition in (schema.fact_definitions or ())
        }
        missing_definitions = [ref for ref in required if ref not in definitions]
        if missing_definitions:
            raise ValueError(
                "FACT_DEFINITION_MISSING:" + ",".join(missing_definitions)
            )

        sensors_by_fact: dict[str, TenderFactSensorContractV1] = {}
        for supplied in sensor_contracts:
            if (
                supplied.organization_id,
                supplied.workspace_id,
            ) != scope:
                raise ValueError("FOREIGN_SCOPE")
            if supplied.fact_ref not in required:
                raise ValueError("SENSOR_FACT_NOT_REQUIRED")
            expected = TenderFactSensorContractV1.from_schema(
                schema=schema, fact_ref=supplied.fact_ref
            )
            if supplied != expected:
                raise ValueError("SENSOR_DEFINITION_MISMATCH")
            if supplied.fact_ref in sensors_by_fact:
                raise ValueError("DUPLICATE_SENSOR_CONTRACT")
            sensors_by_fact[supplied.fact_ref] = expected

        observations_by_fact: dict[str, TenderFactObservationReceiptV1] = {}
        for supplied in observations:
            observation = TenderFactObservationReceiptV1.model_validate(
                supplied.model_dump(mode="json")
            )
            if (
                observation.organization_id,
                observation.workspace_id,
            ) != scope:
                raise ValueError("FOREIGN_SCOPE")
            if observation.fact_ref not in required:
                raise ValueError("OBSERVATION_FACT_NOT_REQUIRED")
            sensor = sensors_by_fact.get(observation.fact_ref)
            if sensor is None or observation.sensor != sensor.pin():
                raise ValueError("OBSERVATION_SENSOR_MISMATCH")
            if any(
                _pin_key(pin) not in target_pins
                for pin in observation.target_evidence
            ):
                raise ValueError("OBSERVATION_EVIDENCE_OUTSIDE_TARGET")
            if observation.fact_ref in observations_by_fact:
                raise ValueError("DUPLICATE_FACT_OBSERVATION")
            if observation.subject_binding != subject_binding:
                raise ValueError("BOOLEAN_SUBJECT_MISMATCH")
            from services.tender_fact_observation_service import TenderFactObservationService
            TenderFactObservationService.verify_receipt(store=store, receipt=observation, sensor=sensor)
            observations_by_fact[observation.fact_ref] = observation

        date_observations_by_fact: dict[str, TenderDocumentDateObservationV1] = {}
        for supplied in document_date_observations:
            observation = TenderDocumentDateObservationV1.model_validate(
                supplied.model_dump(mode="json")
            )
            if (
                observation.organization_id,
                observation.workspace_id,
            ) != scope:
                raise ValueError("FOREIGN_SCOPE")
            if observation.fact_ref not in required:
                raise ValueError("OBSERVATION_FACT_NOT_REQUIRED")
            definition = definitions[observation.fact_ref]
            if definition.value_type != "date":
                raise ValueError("DATE_OBSERVATION_REQUIRES_DATE_FACT")
            expected_schema = self._schema_pin(schema)
            if observation.normative_schema != expected_schema:
                raise ValueError("DATE_OBSERVATION_SCHEMA_MISMATCH")
            expected_definition_hash = "sha256:" + content_hash(
                definition.model_dump(mode="json", by_alias=True, exclude_none=False)
            )
            expected_definition_pin = ArtifactPinV1(
                artifact_kind="normative_fact_definition",
                artifact_ref=f"normative-fact-definition:{definition.fact_ref}",
                artifact_hash=expected_definition_hash,
            )
            if (
                observation.fact_definition != expected_definition_pin
                or observation.fact_definition_hash != expected_definition_hash
            ):
                raise ValueError("DATE_OBSERVATION_DEFINITION_MISMATCH")
            if any(
                _pin_key(pin) not in target_pins
                for pin in observation.target_evidence
            ):
                raise ValueError("OBSERVATION_EVIDENCE_OUTSIDE_TARGET")
            if observation.fact_ref in date_observations_by_fact:
                raise ValueError("DUPLICATE_FACT_OBSERVATION")
            date_observations_by_fact[observation.fact_ref] = observation


        categorical_by_fact = {}
        for supplied in categorical_observations:
            observation = TenderCategoricalFactObservationReceiptV1.model_validate(supplied.model_dump(mode="json"))
            if observation.subject_binding != subject_binding:
                raise ValueError("CATEGORICAL_SUBJECT_MISMATCH")
            if observation.fact_ref not in required:
                raise ValueError("OBSERVATION_FACT_NOT_REQUIRED")
            if observation.fact_ref in categorical_by_fact:
                raise ValueError("DUPLICATE_FACT_OBSERVATION")
            if any(_pin_key(pin) not in target_pins for pin in observation.target_evidence):
                raise ValueError("OBSERVATION_EVIDENCE_OUTSIDE_TARGET")
            sensor = TenderCategoricalFactSensorContractV1.from_schema(schema=schema, fact_ref=observation.fact_ref)
            if observation.sensor != sensor.pin():
                raise ValueError("OBSERVATION_SENSOR_MISMATCH")
            if observation.semantic_result == "KNOWN" and not definitions[observation.fact_ref].accepts_value(observation.value):
                raise ValueError("FACT_VALUE_TYPE_MISMATCH")
            from services.tender_fact_observation_service import TenderFactObservationService
            TenderFactObservationService.verify_categorical_receipt(store=store, receipt=observation, sensor=sensor)
            categorical_by_fact[observation.fact_ref] = observation

        materialized: dict[str, object] = {}
        unresolved: list[str] = []
        for fact_ref in required:
            definition = definitions[fact_ref]
            if definition.value_type == "string" and definition.allowed_values:
                observation = categorical_by_fact.get(fact_ref)
                if observation is None or observation.semantic_result == "UNKNOWN":
                    unresolved.append(fact_ref)
                else:
                    materialized[fact_ref] = observation.value
                continue
            if definition.value_type == "date":
                date_observation = date_observations_by_fact.get(fact_ref)
                if (
                    date_observation is None
                    or date_observation.status == "UNKNOWN"
                ):
                    unresolved.append(fact_ref)
                    continue
                value = date_observation.normalized_value
                if not definition.accepts_value(value):
                    raise ValueError("FACT_VALUE_TYPE_MISMATCH")
                materialized[fact_ref] = value
                continue
            observation = observations_by_fact.get(fact_ref)
            if observation is None or observation.semantic_result == "UNKNOWN":
                unresolved.append(fact_ref)
                continue
            if definition.value_type != "boolean":
                # Non-Boolean facts require a separate exact observation owner;
                # this projector never coerces semantic sensor state into them.
                unresolved.append(fact_ref)
                continue
            value = observation.semantic_result == "TRUE"
            if not definition.accepts_value(value):
                raise ValueError("FACT_VALUE_TYPE_MISMATCH")
            materialized[fact_ref] = value

        facts = ObservedFactsV1.create(
            organization_id=plan.organization_id,
            workspace_id=plan.workspace_id,
            facts=materialized,
            evidence_refs=tuple(pin.artifact_ref for pin in target_evidence),
            frozen_at=frozen_at,
        )
        receipt = TenderObservedFactsProjectionReceiptV1.create(
            organization_id=plan.organization_id,
            workspace_id=plan.workspace_id,
            compiled_plan=self._plan_pin(plan),
            normative_schema=self._schema_pin(schema),
            target_evidence=target_evidence,
            sensor_contracts=tuple(
                sensor.pin() for sensor in sensors_by_fact.values()
            ),
            observation_receipts=tuple(
                pin
                for pin in (
                    *(observation.pin() for observation in observations_by_fact.values()),
                    *(observation.pin() for observation in date_observations_by_fact.values()),
                    *(observation.pin() for observation in categorical_by_fact.values()),
                )
            ),
            required_fact_refs=required,
            unresolved_fact_refs=tuple(unresolved),
            observed_facts=self._facts_pin(facts),
            projector=self._projector_pin(),
            subject_binding=subject_binding,
            frozen_at=frozen_at,
        )
        store.put_many(
            (
                *date_observations_by_fact.values(),
                facts,
                receipt,
            )
        )
        return facts, receipt


__all__ = ["TenderObservedFactsProjectionService"]
