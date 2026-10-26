"""Definition-bound, replayable observations for Tender Boolean facts."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from contracts.artifact_pin import ArtifactPinV1
from contracts.ontology_certification import NormativeAuthorityDependencySetV1
from contracts.runtime_judgment_execution import (
    RuntimeJudgmentAttemptV1,
    RuntimeJudgmentRequestV1,
    RuntimeJudgmentResponseV1,
    verify_complete_request_compatibility,
)
from contracts.tender_observed_facts_projection import (
    TenderFactSubjectBindingV1,
    TenderCategoricalFactSensorContractV1,
    TenderCategoricalFactObservationReceiptV1,
    TenderFactObservationReceiptV1,
    TenderFactSensorContractV1,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_result: str = Field(pattern="^(TRUE|FALSE|UNKNOWN)$")
    evidence_keys: tuple[str, ...] = ()


class _CategoricalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_result: str = Field(pattern="^(KNOWN|UNKNOWN)$")
    value: str | None
    reason: str | None
    evidence_keys: tuple[str, ...] = ()


class TenderFactObservationService:
    """Produce non-authoritative observations tied to a current actor and exact evidence."""

    def __init__(self, *, generator, workspace_root):
        self.generator = generator
        self.workspace_root = Path(workspace_root).resolve()
        self.instructions = (
            Path(__file__).resolve().parents[1]
            / "prompts/tender_fact_observation_v1.txt"
        ).read_text(encoding="utf-8")

    @staticmethod
    def pin(artifact) -> ArtifactPinV1:
        if isinstance(artifact, (TenderFactObservationReceiptV1, TenderCategoricalFactObservationReceiptV1)):
            return artifact.pin()
        kind = artifact.schema_version.rsplit(".v", 1)[0]
        return ArtifactPinV1(
            artifact_kind=kind,
            artifact_ref=artifact.artifact_ref,
            artifact_hash=artifact.artifact_hash,
        )

    @staticmethod
    def _source_payload(region) -> dict:
        return {
            "source_region_ref": region.source_region_ref,
            "source_binding": region.source_binding.model_dump(mode="json"),
            "text": region.text,
            "evidence": [
                {
                    "key": f"e{index}",
                    "evidence_ref": anchor.evidence_ref.evidence_ref,
                    "evidence_hash": "sha256:"
                    + content_hash(anchor.evidence_ref.model_dump(mode="json")),
                    "text": anchor.evidence_ref.selector.exact_quote,
                    "locator": anchor.locator.model_dump(mode="json"),
                }
                for index, anchor in enumerate(region.evidence)
            ],
            "material_dependencies": [
                item.model_dump(mode="json") for item in region.material_dependencies
            ],
        }


    @staticmethod
    def _categorical_source(*, store, binding):
        from contracts.semantic_kernel import Evidence
        from services.tender_observed_facts_projection_service import TenderObservedFactsProjectionService
        context, association, occurrence = TenderObservedFactsProjectionService.resolve_subject(
            store=store, subject_binding=binding)
        pins = {p.artifact_ref: p for a in context.assertions
                if a.origin == "SOURCE" and a.subject_ref == binding.actor_node_ref for p in a.evidence}
        records = []
        scope = dict(organization_id=binding.organization_id, workspace_id=binding.workspace_id)
        for pin in sorted(pins.values(), key=lambda p: p.artifact_ref):
            evidence = store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=Evidence)
            if evidence is None or evidence.pin() != pin:
                raise ValueError("CATEGORICAL_CASE_EVIDENCE_PIN_MISMATCH")
            if evidence.evidence_role != "target":
                continue
            if (evidence.source_ref, evidence.source_hash) != (binding.source_ref, binding.source_hash):
                continue
            if not evidence.selector.exact_quote or not evidence.selector.source_element_ref:
                raise ValueError("EXACT_LOCATED_CASE_EVIDENCE_REQUIRED")
            records.append((pin, evidence))
        records.sort(key=lambda item: (item[0] != binding.occurrence_evidence, item[0].artifact_ref))
        payload = {
            "subject_binding": binding.model_dump(mode="json"),
            "actor_node": next(n.model_dump(mode="json") for n in context.nodes if n.node_id == binding.actor_node_ref),
            "association_assertion": association.model_dump(mode="json"),
            "condition_node_ref": binding.condition_node_ref,
            "text": occurrence.selector.exact_quote,
            "evidence": [{"key": f"e{i}", "evidence_ref": p.artifact_ref,
                          "evidence_hash": p.artifact_hash, "text": e.selector.exact_quote,
                          "selector": e.selector.model_dump(mode="json", by_alias=True),
                          "source_ref": e.source_ref, "source_hash": e.source_hash}
                         for i, (p, e) in enumerate(records)],
        }
        return payload, tuple(p for p, _ in records)

    @staticmethod
    def verify_receipt(*, store, receipt, sensor, support_rule_store=None):
        """Re-read producer lineage without a provider call."""
        import json
        from contracts.tender_observed_facts_projection import TenderFactObservationReceiptV2
        if isinstance(receipt, TenderFactObservationReceiptV2):
            return TenderFactObservationService.verify_support_receipt(
                store=store, receipt=receipt, sensor=sensor, rule_store=support_rule_store)

        if not isinstance(
            receipt,
            (TenderFactObservationReceiptV1, TenderCategoricalFactObservationReceiptV1),
        ):
            raise ValueError("UNSUPPORTED_FACT_OBSERVATION_PROTOCOL")
        prefix = (
            "CATEGORICAL"
            if isinstance(receipt, TenderCategoricalFactObservationReceiptV1)
            else "FACT_OBSERVATION"
        )
        receipt = type(receipt).model_validate(receipt.model_dump(mode="json"))
        sensor = type(sensor).model_validate(sensor.model_dump(mode="json"))
        if (sensor.organization_id, sensor.workspace_id) != (
            receipt.organization_id,
            receipt.workspace_id,
        ):
            raise ValueError("OBSERVATION_REPLAY_SCOPE_MISMATCH")
        if (
            receipt.producer.artifact_kind != "tender_fact_observation_producer"
            or receipt.producer.artifact_ref
            != "tender-fact-observation-producer:"
            + receipt.producer.artifact_hash.removeprefix("sha256:")
        ):
            raise ValueError("OBSERVATION_REPLAY_PRODUCER_MISMATCH")
        scope = dict(
            organization_id=receipt.organization_id, workspace_id=receipt.workspace_id
        )
        persisted = store.get(
            **scope, artifact_ref=receipt.receipt_ref, contract_type=type(receipt)
        )
        if (
            persisted != receipt
            or receipt.sensor != sensor.pin()
            or receipt.fact_ref != sensor.fact_ref
        ):
            raise ValueError(f"{prefix}_RETAINED_RECEIPT_MISMATCH")
        categorical = isinstance(receipt, TenderCategoricalFactObservationReceiptV1)
        scoped = receipt.subject_binding is not None
        source, pins = (
            TenderFactObservationService._categorical_source(
                store=store, binding=receipt.subject_binding
            )
            if scoped
            else (None, receipt.target_evidence)
        )

        def key(pin):
            return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)

        if receipt.target_evidence != tuple(sorted(pins, key=key)):
            raise ValueError(f"{prefix}_CASE_EVIDENCE_MISMATCH")
        requests = [
            p for p in receipt.basis if p.artifact_kind == "runtime_judgment_request"
        ]
        responses = [
            p for p in receipt.basis if p.artifact_kind == "runtime_judgment_response"
        ]
        attempts = [
            p for p in receipt.basis if p.artifact_kind == "runtime_judgment_attempt"
        ]
        if (
            len(requests) != 1
            or len(responses) != 1
            or not attempts
            or len(receipt.basis) != 2 + len(attempts)
            or len(set(receipt.basis)) != len(receipt.basis)
        ):
            raise ValueError(f"{prefix}_REPLAY_BASIS_MISMATCH")
        request = store.get(
            **scope,
            artifact_ref=requests[0].artifact_ref,
            contract_type=RuntimeJudgmentRequestV1,
        )
        response = store.get(
            **scope,
            artifact_ref=responses[0].artifact_ref,
            contract_type=RuntimeJudgmentResponseV1,
        )
        if (
            request is None
            or response is None
            or TenderFactObservationService.pin(request) != requests[0]
            or TenderFactObservationService.pin(response) != responses[0]
        ):
            raise ValueError(f"{prefix}_REPLAY_PIN_MISMATCH")
        try:
            payload = json.loads(request.judgment_request["input_text"])
            if (
                not isinstance(payload, dict)
                or not isinstance(request.judgment_request.get("instructions"), str)
                or not isinstance(request.judgment_request.get("json_schema"), dict)
            ):
                raise ValueError("INVALID_REQUEST_SHAPE")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("FACT_OBSERVATION_REPLAY_REQUEST_SHAPE_MISMATCH") from exc
        if not scoped:
            source = payload.get("target", {})
            retained_pins = tuple(
                sorted(
                    (
                        ArtifactPinV1(
                            artifact_kind="evidence",
                            artifact_ref=item["evidence_ref"],
                            artifact_hash=item["evidence_hash"],
                        )
                        for item in source.get("evidence", [])
                    ),
                    key=key,
                )
            )
            if retained_pins != receipt.target_evidence or not retained_pins:
                raise ValueError("FACT_OBSERVATION_REPLAY_EVIDENCE_MISMATCH")
        if (
            payload.get("authority_dependency_set_pin", {}).get("artifact_kind")
            != "normative_authority_dependency_set"
        ):
            raise ValueError("FACT_OBSERVATION_REPLAY_AUTHORITY_MISMATCH")
        config = {
            "prompt_hash": "sha256:"
            + sha256(request.judgment_request["instructions"].encode()).hexdigest(),
            "adapter_hash": request.execution_profile.get("adapter_hash"),
            "schema_hash": sensor.normative_schema_hash,
            "sensor_hash": sensor.sensor_hash,
            "authority_dependency_set_hash": payload.get(
                "authority_dependency_set_pin", {}
            ).get("artifact_hash"),
            "output_schema_hash": "sha256:"
            + content_hash(request.judgment_request["json_schema"]),
        }
        config_hash = "sha256:" + content_hash(config)
        if receipt.producer_config != ArtifactPinV1(
            artifact_kind="tender_fact_observation_config",
            artifact_ref="tender-fact-observation-config:"
            + config_hash.removeprefix("sha256:"),
            artifact_hash=config_hash,
        ):
            raise ValueError("FACT_OBSERVATION_REPLAY_CONFIG_MISMATCH")
        authority_pin = ArtifactPinV1.model_validate(
            payload["authority_dependency_set_pin"]
        )
        authority = store.get(
            **scope,
            artifact_ref=authority_pin.artifact_ref,
            contract_type=NormativeAuthorityDependencySetV1,
        )
        if authority is None or (authority.set_ref, authority.set_hash) != (
            authority_pin.artifact_ref,
            authority_pin.artifact_hash,
        ):
            raise ValueError("FACT_OBSERVATION_REPLAY_AUTHORITY_MISMATCH")
        if (
            payload.get("fact_ref") != sensor.fact_ref
            or payload.get("source_meaning") != sensor.source_meaning
            or payload.get("definition_evidence_refs")
            != list(sensor.definition_evidence_refs)
            or payload.get("frontier")
            != list(sensor.allowed_values if categorical else sensor.frontier)
            or payload.get("target") != source
            or payload.get("sensor_pin") != sensor.pin().model_dump(mode="json")
            or request.judgment_request.get("producer_pin")
            != receipt.producer.model_dump(mode="json")
            or request.judgment_request.get("producer_config_pin")
            != receipt.producer_config.model_dump(mode="json")
            or response.execution_ref != request.artifact_ref
            or response.execution_hash != request.artifact_hash
            or response.run_id != request.run_id
        ):
            raise ValueError(f"{prefix}_REPLAY_INPUT_MISMATCH")
        final = None
        for pin in attempts:
            attempt = store.get(
                **scope,
                artifact_ref=pin.artifact_ref,
                contract_type=RuntimeJudgmentAttemptV1,
            )
            if (
                attempt is None
                or TenderFactObservationService.pin(attempt) != pin
                or attempt.execution_ref != request.artifact_ref
                or attempt.execution_hash != request.artifact_hash
                or attempt.run_id != request.run_id
            ):
                raise ValueError(f"{prefix}_REPLAY_ATTEMPT_MISMATCH")
            verify_complete_request_compatibility(
                execution_profile=request.execution_profile,
                request={
                    k: request.judgment_request[k]
                    for k in ("instructions", "input_text", "json_schema")
                },
                receipt=attempt.compatibility,
            )
            if (attempt.artifact_ref, attempt.artifact_hash) == (
                response.predecessor_attempt_ref,
                response.predecessor_attempt_hash,
            ):
                final = attempt
        parsed = (_CategoricalResult if categorical else _Result).model_validate(
            response.result.get("parsed")
        )
        if (
            final is None
            or final.compatibility.get("outcome") != "COMPATIBLE"
            or parsed.semantic_result != receipt.semantic_result
            or (
                categorical
                and (parsed.value, parsed.reason) != (receipt.value, receipt.reason)
            )
            or response.result.get("audit", {}).get("compatibility")
            != final.compatibility
            or (response.result.get("provider"), response.result.get("model"))
            != (
                final.compatibility["profile"]["provider"],
                final.compatibility["profile"]["model"],
            )
        ):
            raise ValueError(f"{prefix}_REPLAY_RESULT_MISMATCH")
        key_pins = {
            item["key"]: ArtifactPinV1(
                artifact_kind="evidence",
                artifact_ref=item["evidence_ref"],
                artifact_hash=item["evidence_hash"],
            )
            for item in source["evidence"]
        }
        if not set(parsed.evidence_keys) <= set(key_pins):
            raise ValueError("OBSERVATION_REPLAY_CITATION_MISMATCH")
        if (
            scoped
            and tuple(sorted((key_pins[k] for k in parsed.evidence_keys), key=key))
            != receipt.cited_evidence
        ):
            raise ValueError(f"{prefix}_REPLAY_CITATION_MISMATCH")
        if (
            categorical
            and receipt.semantic_result == "KNOWN"
            and receipt.value not in sensor.allowed_values
        ):
            raise ValueError("CATEGORICAL_VALUE_OUTSIDE_DOMAIN")

    @staticmethod
    def verify_categorical_receipt(*, store, receipt, sensor):
        return TenderFactObservationService.verify_receipt(store=store, receipt=receipt, sensor=sensor)

    async def observe(
        self,
        *,
        store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        actor_context,
        schema,
        authority_dependency_set: NormativeAuthorityDependencySetV1,
        source_region,
        fact_ref: str,
        run_id: str,
        request_id: str,
        sensor_contract: TenderFactSensorContractV1 | TenderCategoricalFactSensorContractV1 | None = None,
        subject_binding: TenderFactSubjectBindingV1 | None = None,
    ) -> TenderFactObservationReceiptV1 | TenderCategoricalFactObservationReceiptV1:
        if not isinstance(store, OntologyBuildStore) or not isinstance(
            authority_resolver, RuntimeAuthorityResolver
        ):
            raise TypeError("FACT_OBSERVATION_GOVERNED_OWNERS_REQUIRED")
        if not isinstance(
            authority_dependency_set, NormativeAuthorityDependencySetV1
        ):
            raise TypeError("TENDER_AUTHORITY_DEPENDENCY_SET_REQUIRED")
        authority_resolver.require_current_actor(actor_context=actor_context)
        if (
            store._path.resolve()
            != ontology_build_store_path(self.workspace_root).resolve()
        ):
            raise ValueError("FACT_OBSERVATION_WORKSPACE_OWNER_MISMATCH")
        scope = {
            "organization_id": actor_context.organization_id,
            "workspace_id": actor_context.workspace_id,
        }
        if (schema.organization_id, schema.workspace_id) != (
            scope["organization_id"],
            scope["workspace_id"],
        ):
            raise ValueError("FACT_OBSERVATION_SCHEMA_SCOPE_MISMATCH")
        if (
            authority_dependency_set.organization_id,
            authority_dependency_set.workspace_id,
            authority_dependency_set.boundary_ref,
            authority_dependency_set.boundary_hash,
        ) != (
            scope["organization_id"],
            scope["workspace_id"],
            schema.boundary_ref,
            schema.boundary_hash,
        ):
            raise ValueError("AUTHORITY_SCOPE_MISMATCH")
        authority_pin = ArtifactPinV1(
            artifact_kind="normative_authority_dependency_set",
            artifact_ref=authority_dependency_set.set_ref,
            artifact_hash=authority_dependency_set.set_hash,
        )
        persisted_authority = store.get(
            **scope,
            artifact_ref=authority_dependency_set.set_ref,
            contract_type=NormativeAuthorityDependencySetV1,
        )
        if persisted_authority != authority_dependency_set:
            raise ValueError("GOVERNED_AUTHORITY_DEPENDENCY_SET_REQUIRED")
        definition = next((d for d in schema.fact_definitions or () if d.fact_ref == fact_ref), None)
        if definition is None:
            raise ValueError("FACT_DEFINITION_MISSING")
        categorical = definition.value_type == "string" and bool(definition.allowed_values)
        if categorical and subject_binding is None:
            raise ValueError("CATEGORICAL_SUBJECT_BINDING_REQUIRED")
        receipt_type = TenderCategoricalFactObservationReceiptV1 if categorical else TenderFactObservationReceiptV1
        result_type = _CategoricalResult if categorical else _Result
        instructions = self.instructions
        if subject_binding is not None:
            if (subject_binding.organization_id, subject_binding.workspace_id) != (scope["organization_id"], scope["workspace_id"]):
                raise ValueError("FACT_OBSERVATION_SUBJECT_SCOPE_MISMATCH")
            sensor = (TenderCategoricalFactSensorContractV1 if categorical else TenderFactSensorContractV1).from_schema(schema=schema, fact_ref=fact_ref)
            source, evidence_pins = self._categorical_source(store=store, binding=subject_binding)
            prompt = "tender_categorical_fact_observation_v1.txt" if categorical else "tender_scoped_boolean_fact_observation_v1.txt"
            instructions = (Path(__file__).resolve().parents[1] / "prompts" / prompt).read_text(encoding="utf-8")
        else:
            if not source_region.evidence:
                raise ValueError("FACT_OBSERVATION_TARGET_EVIDENCE_REQUIRED")
            if any((a.evidence_ref.organization_id, a.evidence_ref.workspace_id) !=
                   (scope["organization_id"], scope["workspace_id"]) for a in source_region.evidence):
                raise ValueError("FACT_OBSERVATION_SOURCE_SCOPE_MISMATCH")
            sensor = TenderFactSensorContractV1.from_schema(schema=schema, fact_ref=fact_ref)
            source = self._source_payload(source_region)
            evidence_pins = tuple(ArtifactPinV1(artifact_kind="evidence",
                artifact_ref=a.evidence_ref.evidence_ref,
                artifact_hash="sha256:" + content_hash(a.evidence_ref.model_dump(mode="json"))) for a in source_region.evidence)
        if sensor_contract is not None and sensor_contract != sensor:
            raise ValueError("SENSOR_DEFINITION_MISMATCH")
        input_payload = {
            "fact_ref": sensor.fact_ref,
            "source_meaning": sensor.source_meaning,
            "definition_evidence_refs": sensor.definition_evidence_refs,
            "frontier": (sensor.allowed_values if categorical else sensor.frontier),
            "sensor_pin": sensor.pin().model_dump(mode="json"),
            "authority_dependency_set_pin": authority_pin.model_dump(mode="json"),
            "target": source,
        }
        input_text = canonical_json(input_payload)
        output_schema = result_type.model_json_schema()
        if categorical:
            output_schema['properties']['value'] = {'anyOf': [{'type': 'string', 'enum': list(sensor.allowed_values)}, {'type': 'null'}]}
        profile = getattr(self.generator, "compatibility_execution_identity", None)
        generate = getattr(self.generator, "generate_compatible_structured", None)
        if not isinstance(profile, dict) or not callable(generate):
            raise ValueError("COMPATIBLE_FACT_OBSERVATION_ADAPTER_REQUIRED")
        producer_hash = "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
        producer_pin = ArtifactPinV1(
            artifact_kind="tender_fact_observation_producer",
            artifact_ref="tender-fact-observation-producer:"
            + producer_hash.removeprefix("sha256:"),
            artifact_hash=producer_hash,
        )
        config = {
            "prompt_hash": "sha256:" + sha256(instructions.encode()).hexdigest(),
            "adapter_hash": profile.get("adapter_hash"),
            "schema_hash": schema.schema_hash,
            "sensor_hash": sensor.sensor_hash,
            "authority_dependency_set_hash": authority_pin.artifact_hash,
            "output_schema_hash": "sha256:" + content_hash(output_schema),
        }
        config_hash = "sha256:" + content_hash(config)
        config_pin = ArtifactPinV1(
            artifact_kind="tender_fact_observation_config",
            artifact_ref="tender-fact-observation-config:"
            + config_hash.removeprefix("sha256:"),
            artifact_hash=config_hash,
        )
        request = RuntimeJudgmentRequestV1.create(
            **scope,
            run_id=run_id,
            request_id=request_id,
            judgment_request={
                "actor_ref": "user:" + actor_context.actor_id,
                "instructions": instructions,
                "input_text": input_text,
                "json_schema": output_schema,
                "producer_pin": producer_pin.model_dump(mode="json"),
                "producer_config_pin": config_pin.model_dump(mode="json"),
            },
            execution_profile=profile,
        )
        operation = {
            **scope,
            "actor_ref": "user:" + actor_context.actor_id,
            "operation": "OBSERVE_TENDER_FACT",
            "request_id": request_id,
            "request_hash": request.artifact_hash,
        }
        prior = store.get_operation_result(
            **operation, artifact_kind=("tender_categorical_fact_observation_receipt" if categorical else "tender_fact_observation_receipt")
        )
        if prior is not None:
            receipt = store.get(
                **scope,
                artifact_ref=prior.artifact_ref,
                contract_type=receipt_type,
            )
            if receipt is None or self.pin(receipt) != prior:
                raise ValueError("FACT_OBSERVATION_REPLAY_PIN_MISMATCH")
            if receipt.sensor != sensor.pin() or receipt.target_evidence != tuple(
                sorted(
                    evidence_pins,
                    key=lambda item: (
                        item.artifact_kind,
                        item.artifact_ref,
                        item.artifact_hash,
                    ),
                )
            ):
                raise ValueError("FACT_OBSERVATION_REPLAY_INPUT_MISMATCH")
            if (
                receipt.producer != producer_pin
                or receipt.producer_config != config_pin
            ):
                raise ValueError("FACT_OBSERVATION_REPLAY_PRODUCER_MISMATCH")
            request_pin = self.pin(request)
            request_artifact = store.get(
                **scope,
                artifact_ref=request_pin.artifact_ref,
                contract_type=RuntimeJudgmentRequestV1,
            )
            if request_artifact != request:
                raise ValueError("FACT_OBSERVATION_REPLAY_REQUEST_MISMATCH")
            if request_pin not in receipt.basis:
                raise ValueError("FACT_OBSERVATION_REPLAY_BASIS_MISMATCH")
            self.verify_receipt(store=store, receipt=receipt, sensor=sensor)
            return receipt

        request_pin = store.put(request)
        if request_pin is None:
            request_pin = self.pin(request)
        attempts: list[ArtifactPinV1] = []

        def record(compatibility):
            authority_resolver.require_current_actor(actor_context=actor_context)
            verify_complete_request_compatibility(
                execution_profile=profile,
                request={
                    "instructions": instructions,
                    "input_text": input_text,
                    "json_schema": output_schema,
                },
                receipt=compatibility,
            )
            attempt = RuntimeJudgmentAttemptV1.create(
                **scope,
                run_id=run_id,
                execution_ref=request.artifact_ref,
                execution_hash=request.artifact_hash,
                compatibility=compatibility,
            )
            store.put(attempt)
            attempts.append(self.pin(attempt))

        try:
            result = await generate(
                instructions=instructions,
                input_text=input_text,
                json_schema=output_schema,
                schema_name="tender_fact_observation_v1",
                purpose="tender_fact_observation",
                record=record,
            )
        except Exception as exc:
            raise ValueError("FACT_OBSERVATION_GENERATION_FAILED") from exc
        authority_resolver.require_current_actor(actor_context=actor_context)
        if not attempts:
            raise ValueError("FACT_OBSERVATION_COMPATIBILITY_RECEIPT_REQUIRED")
        compatibility = attempts[-1]
        attempt = store.get(
            **scope,
            artifact_ref=compatibility.artifact_ref,
            contract_type=RuntimeJudgmentAttemptV1,
        )
        if attempt is None:
            raise ValueError("FACT_OBSERVATION_ATTEMPT_MISSING")
        verify_complete_request_compatibility(
            execution_profile=profile,
            request={
                "instructions": instructions,
                "input_text": input_text,
                "json_schema": output_schema,
            },
            receipt=attempt.compatibility,
        )
        if attempt.compatibility.get("outcome") != "COMPATIBLE":
            raise ValueError("FACT_OBSERVATION_COMPATIBLE_FINAL_ATTEMPT_REQUIRED")
        if result.audit.get("compatibility") != attempt.compatibility:
            raise ValueError("FACT_OBSERVATION_TRANSPORT_BINDING_MISMATCH")
        parsed = result_type.model_validate(result.parsed)
        if categorical and parsed.semantic_result == "KNOWN" and parsed.value not in sensor.allowed_values:
            raise ValueError("CATEGORICAL_VALUE_OUTSIDE_DOMAIN")
        allowed_keys = {item["key"] for item in source["evidence"]}
        if not set(parsed.evidence_keys) <= allowed_keys:
            raise ValueError("FACT_OBSERVATION_EVIDENCE_KEY_MISMATCH")
        if (result.provider, result.model) != (
            attempt.compatibility.get("profile", {}).get("provider"),
            attempt.compatibility.get("profile", {}).get("model"),
        ):
            raise ValueError("FACT_OBSERVATION_PROVIDER_BINDING_MISMATCH")

        basis = (
            request_pin,
            *attempts,
        )
        response = RuntimeJudgmentResponseV1.create(
            **scope,
            run_id=run_id,
            execution_ref=request.artifact_ref,
            execution_hash=request.artifact_hash,
            predecessor_attempt_ref=compatibility.artifact_ref,
            predecessor_attempt_hash=compatibility.artifact_hash,
            result={
                "parsed": parsed.model_dump(mode="json"),
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "agent_id": result.agent_id,
                "usage": result.usage,
                "audit": result.audit,
            },
        )
        response_pin = store.put(response)
        if response_pin is None:
            response_pin = self.pin(response)
        receipt = receipt_type.create(
            **scope,
            sensor=sensor.pin(),
            fact_ref=fact_ref,
            semantic_result=parsed.semantic_result,
            target_evidence=evidence_pins,
            producer=producer_pin,
            producer_config=config_pin,
            basis=(*basis, response_pin),
            observed_at=datetime.now(timezone.utc),
            **(dict(subject_binding=subject_binding,
                    cited_evidence=tuple(evidence_pins[int(k[1:])] for k in parsed.evidence_keys)) if subject_binding is not None else {}),
            **(dict(value=parsed.value, reason=parsed.reason) if categorical else {}),
        )
        stored_pin = store.put_many_with_operation_result(
            **operation,
            artifacts=(receipt,),
            result_pin=self.pin(receipt),
        )
        if stored_pin != self.pin(receipt):
            raise ValueError("FACT_OBSERVATION_PERSISTENCE_MISMATCH")
        return receipt


    @staticmethod
    def resolve_support_profile(*, rule_store, version_pin, activation_id, organization_id,
                                workspace_id, at):
        """Resolve the explicit license through existing reviewed rule/activation owners."""
        import json
        from storage.governed_rule_store import GovernedRuleStore, _db_path
        from contracts.tender_observed_facts_projection import BooleanSupportProfileV1
        if not isinstance(rule_store, GovernedRuleStore):
            raise ValueError("SUPPORT_PROFILE_AUTHORITY_UNAVAILABLE")
        if rule_store._db_path.resolve()!=_db_path().resolve():
            raise ValueError("SUPPORT_PROFILE_OWNER_MISMATCH")
        if at.tzinfo is None or at.utcoffset() != timezone.utc.utcoffset(at):
            raise ValueError("SUPPORT_UTC_REQUIRED")
        version_pin=ArtifactPinV1.model_validate(version_pin)
        row=rule_store.get_rule_version(version_pin.artifact_ref)
        if (version_pin.artifact_kind!='governed_rule_version' or row is None
            or 'sha256:'+row['content_hash'] != version_pin.artifact_hash
            or row['rule_family']!='fact_support_boolean.v1'
            or row['authority_domain']!='fact_support' or row['semantic_tier']!='T4'
            or row['organization_id']!=organization_id):
            raise ValueError("SUPPORT_PROFILE_VERSION_MISMATCH")
        profile=BooleanSupportProfileV1.model_validate_json(row['governed_rule_text'])
        if canonical_json(json.loads(row['governed_rule_text']))!=canonical_json(profile.model_dump(mode='json')):
            raise ValueError('SUPPORT_EXPLICIT_PROFILE_FIELDS_REQUIRED')
        if (profile.organization_id,profile.workspace_id)!=(organization_id,workspace_id):
            raise ValueError("SUPPORT_PROFILE_SCOPE_MISMATCH")
        license={'profile_hash':profile.profile_hash,'licenses':
                 ['complete_premises','sufficient_bundles','origin_groups','bounded_coverage']}
        candidate=rule_store._candidate_store.get_candidate(row['rule_candidate_id'])
        if (candidate is None
            or candidate['draft_rule_text']!=row['governed_rule_text']
            or candidate['approval_evidence']!=license or row['approval_evidence']!=license
            or row['rule_state']!='APPROVED'):
            raise ValueError("SUPPORT_PROFILE_NOT_ADMITTED")
        # Follow and verify the append-only candidate chain, independent of same-time UUID ordering.
        events=rule_store._candidate_store.list_events(row['rule_candidate_id'])
        by_prior={}
        for event in events:
            prior=event['prior_event_hash']
            if prior in by_prior:
                raise ValueError("SUPPORT_APPROVAL_CHAIN_INVALID")
            payload={k:event[k] for k in ('rule_candidate_id','event_type','actor','actor_authority',
                'from_state','to_state','prior_event_hash','created_at')}
            payload['event_payload']=json.loads(event['event_json'])
            if content_hash(canonical_json(payload))!=event['event_hash']:
                raise ValueError("SUPPORT_APPROVAL_CHAIN_INVALID")
            by_prior[prior]=event
        prior=None;visited=[]
        while prior in by_prior:
            event=by_prior[prior];visited.append(event);prior=event['event_hash']
            if len(visited)>len(events):
                raise ValueError("SUPPORT_APPROVAL_CHAIN_INVALID")
        approved=[e for e in visited if e['from_state']=='NEEDS_REVIEW' and e['to_state']=='APPROVED'
                  and e['event_type']=='APPROVED' and e['actor_authority']=='policy_owner'
                  and json.loads(e['event_json']).get('approval_evidence')==license]
        if len(visited)!=len(events) or len(approved)!=1:
            raise ValueError("SUPPORT_PROFILE_NOT_ADMITTED")
        effective=[e for e in visited if datetime.fromisoformat(e['created_at'])<=at]
        if not effective or effective[-1]['to_state'] not in ('APPROVED','ACTIVATED'):
            raise ValueError('SUPPORT_PROFILE_NOT_EFFECTIVE')
        activation=rule_store._activation_store.resolve_executable_activations(
            activation_ids=[activation_id],organization_id=organization_id,
            workspace_id=workspace_id,effective_at=at.isoformat())[0]
        if (activation['rule_version_id']!=row['rule_version_id']
            or activation['approval_event_ref']!=approved[0]['event_hash']):
            raise ValueError("SUPPORT_ACTIVATION_APPROVAL_MISMATCH")
        return profile

    @staticmethod
    def _verify_support_origins(profile, target_evidence, source):
        for group in profile.groups:
            origin=group.origin
            evidence_origin=origin.artifact_kind=='evidence' and origin in target_evidence
            source_origin=(origin.artifact_kind=='source' and
                (origin.artifact_ref,origin.artifact_hash)==(source.get('source_identity_ref'),source.get('source_hash')))
            if not (evidence_origin or source_origin):
                raise ValueError('SUPPORT_ORIGIN_NOT_RETAINED')

    @staticmethod
    def verify_support_receipt(*, store, receipt, sensor, rule_store, current_at=None):
        from contracts.tender_observed_facts_projection import TenderFactObservationReceiptV2
        value=TenderFactObservationReceiptV2.model_validate(receipt.model_dump(mode='json'))
        d=value.support_declaration
        scope=dict(organization_id=value.organization_id,workspace_id=value.workspace_id)
        retained=store.get(**scope,artifact_ref=value.receipt_ref,contract_type=TenderFactObservationReceiptV2)
        original=store.get(**scope,artifact_ref=d.original_observation.artifact_ref,
                           contract_type=TenderFactObservationReceiptV1)
        if retained!=value or original is None or original.pin()!=d.original_observation:
            raise ValueError('SUPPORT_RETAINED_OBSERVATION_MISMATCH')
        TenderFactObservationService.verify_receipt(store=store,receipt=original,sensor=sensor)
        common=original.model_dump(mode='json',exclude={'schema_version','receipt_ref','receipt_hash'})
        if common!=value.model_dump(mode='json',exclude={'schema_version','receipt_ref','receipt_hash','support_declaration'}):
            raise ValueError('SUPPORT_ORIGINAL_OBSERVATION_MISMATCH')
        profile=TenderFactObservationService.resolve_support_profile(rule_store=rule_store,
            version_pin=d.rule_version,activation_id=d.activation_id,**scope,at=d.licensed_at)
        if profile!=d.profile or value.sensor!=sensor.pin():
            raise ValueError('SUPPORT_PROFILE_BINDING_MISMATCH')
        # Every origin unit must be an exact retained input, not an invented name/hash.
        import json
        request_pin=next(p for p in original.basis if p.artifact_kind=='runtime_judgment_request')
        request=store.get(**scope,artifact_ref=request_pin.artifact_ref,contract_type=RuntimeJudgmentRequestV1)
        target=json.loads(request.judgment_request['input_text'])['target']
        source=target.get('source_binding',{})
        TenderFactObservationService._verify_support_origins(profile,original.target_evidence,source)
        if current_at is not None:
            TenderFactObservationService.resolve_support_profile(rule_store=rule_store,
                version_pin=d.rule_version,activation_id=d.activation_id,**scope,at=current_at)
        return value

    async def observe_supported(self, *, rule_store=None, profile_version=None,
                                activation_id=None, **inputs):
        """Emit from a retained original observation plus an explicit governed finite license."""
        if profile_version is None:
            return await self.observe(**inputs)
        from contracts.tender_observed_facts_projection import (
            TenderFactObservationReceiptV2, FactSupportDeclarationV1)
        actor=inputs['actor_context'];inputs['authority_resolver'].require_current_actor(actor_context=actor)
        at=datetime.now(timezone.utc)
        profile=self.resolve_support_profile(rule_store=rule_store,version_pin=profile_version,
            activation_id=activation_id,organization_id=actor.organization_id,
            workspace_id=actor.workspace_id,at=at)
        sensor=TenderFactSensorContractV1.from_schema(schema=inputs['schema'],fact_ref=inputs['fact_ref'])
        if profile.sensor!=sensor.pin() or profile.subject_binding!=inputs.get('subject_binding'):
            raise ValueError('SUPPORT_PROFILE_FACT_SCHEMA_SUBJECT_MISMATCH')
        target=(self._categorical_source(store=inputs['store'],binding=profile.subject_binding)[1]
                if profile.subject_binding else tuple(ArtifactPinV1(artifact_kind='evidence',
                artifact_ref=e['evidence_ref'],artifact_hash=e['evidence_hash'])
                for e in self._source_payload(inputs['source_region'])['evidence']))
        if not set(profile.premises)<=set(target):
            raise ValueError('SUPPORT_PREMISES_NOT_RETAINED')
        source=(self._categorical_source(store=inputs['store'],binding=profile.subject_binding)[0]
                if profile.subject_binding else self._source_payload(inputs['source_region']))
        self._verify_support_origins(profile,target,source.get('source_binding',{}))
        original=await self.observe(**inputs)
        if original.semantic_result=='UNKNOWN':
            return original  # An unavailable observation never acquires a signed value.
        if profile.value!=(original.semantic_result=='TRUE'):
            raise ValueError('SUPPORT_CONFLICTING_VALUE')
        at=datetime.now(timezone.utc)
        if self.resolve_support_profile(rule_store=rule_store,version_pin=profile_version,
            activation_id=activation_id,organization_id=actor.organization_id,
            workspace_id=actor.workspace_id,at=at)!=profile:
            raise ValueError('SUPPORT_PROFILE_CHANGED_DURING_OBSERVATION')
        declaration=FactSupportDeclarationV1(profile=profile,rule_version=profile_version,
            activation_id=activation_id,original_observation=original.pin(),licensed_at=at)
        receipt=TenderFactObservationReceiptV2.create(**original.model_dump(
            exclude={'schema_version','receipt_ref','receipt_hash'}),support_declaration=declaration)
        inputs['store'].put(receipt)
        self.verify_support_receipt(store=inputs['store'],receipt=receipt,sensor=sensor,rule_store=rule_store)
        return receipt


__all__ = ["TenderFactObservationService"]
