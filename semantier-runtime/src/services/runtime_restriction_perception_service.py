"""Retain exact pre-retrieval perception requests, attempts and observations."""
from dataclasses import asdict
from contextlib import asynccontextmanager, ExitStack
import asyncio
from storage.governed_file_io import exclusive_file_lock
from datetime import datetime, timezone
from pathlib import Path
import json
import time

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_judgment_execution import verify_complete_request_compatibility
from contracts.runtime_restriction_perception import (
    PerceptionQuoteOccurrenceV1, RestrictionPerceptionProposalV1,
    RestrictionPerceptionOutputV1, RuntimeRestrictionPerceptionRequestV1,
    RuntimeRestrictionPerceptionAttemptV1, RuntimeRestrictionPerceptionResponseV1,
    RuntimeRestrictionPerceptionResponseV2,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


class RuntimeRestrictionPerceptionService:
    def __init__(self, *, generator, workspace_root):
        self.generator = generator
        self.workspace_root = Path(workspace_root).resolve()
        self.instructions = (Path(__file__).resolve().parents[1] / "prompts/runtime_restriction_perception_v2.txt").read_text()

    @staticmethod
    def pin(artifact):
        return ArtifactPinV1(artifact_kind=artifact.schema_version.rsplit(".v", 1)[0],
            artifact_ref=artifact.artifact_ref, artifact_hash=artifact.artifact_hash)

    @staticmethod
    def input_payload(region):
        return dict(source_region_ref=region.source_region_ref,
            source_binding=region.source_binding.model_dump(mode="json"),
            evidence=[dict(key=f"e{index}", evidence_ref=anchor.evidence_ref.evidence_ref,
                text=anchor.evidence_ref.selector.exact_quote) for index, anchor in enumerate(region.evidence)])

    @staticmethod
    def resolve_quote(value, evidence):
        quote = PerceptionQuoteOccurrenceV1.model_validate(value)
        text = evidence.get(quote.evidence_key)
        start = -1
        if text is None:
            raise ValueError("PERCEPTION_EVIDENCE_QUOTE_MISMATCH")
        for _ in range(quote.occurrence + 1):
            start = text.find(quote.text, start + 1)
            if start < 0:
                raise ValueError("PERCEPTION_EVIDENCE_QUOTE_MISMATCH")
        return dict(evidence_key=quote.evidence_key, text=quote.text, start=start, end=start + len(quote.text))

    @staticmethod
    def retrieval_query(output, region):
        return "\n".join(item.clause.text for item in output.observations
            if item.classification != "description") or region.text

    @asynccontextmanager
    async def _execution_lock(self, request_hash):
        directory = self.workspace_root / "runs" / "restriction-perception-locks"
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(self.workspace_root):
            raise ValueError("PERCEPTION_LOCK_OUTSIDE_WORKSPACE")
        lock_path = directory / (request_hash.removeprefix("sha256:") + ".lock")
        if not lock_path.resolve().is_relative_to(self.workspace_root):
            raise ValueError("PERCEPTION_LOCK_OUTSIDE_WORKSPACE")
        with ExitStack() as locks:
            while True:
                try:
                    locks.enter_context(exclusive_file_lock(lock_path, wait=False))
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            yield

    async def perceive_persisted(self, *, store, authority_resolver, actor_context, source_region, run_id, request_id):
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("perception requires production workspace owners")
        authority_resolver.require_current_actor(actor_context=actor_context)
        if store._path.resolve() != ontology_build_store_path(self.workspace_root).resolve():
            raise ValueError("PERCEPTION_WORKSPACE_OWNER_MISMATCH")
        scope = dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)
        if any((anchor.evidence_ref.organization_id, anchor.evidence_ref.workspace_id) !=
            (scope["organization_id"], scope["workspace_id"]) for anchor in source_region.evidence):
            raise ValueError("PERCEPTION_SOURCE_SCOPE_MISMATCH")
        profile = getattr(self.generator, "compatibility_execution_identity", None)
        generate = getattr(self.generator, "generate_compatible_structured", None)
        if not isinstance(profile, dict) or not callable(generate):
            raise ValueError("COMPATIBLE_PERCEPTION_ADAPTER_REQUIRED")
        basis = dict(**scope, run_id=run_id, actor_ref="user:" + actor_context.actor_id,
            source_region=source_region.model_dump(mode="json"), instructions=self.instructions,
            input_text=canonical_json(self.input_payload(source_region)),
            output_schema=RestrictionPerceptionProposalV1.model_json_schema(), execution_profile=profile)
        operation = dict(**scope, actor_ref=basis["actor_ref"], operation="PERCEIVE_RUNTIME_RESTRICTIONS",
            request_id=request_id, request_hash="sha256:" + content_hash(basis))
        async with self._execution_lock(operation["request_hash"]):
            authority_resolver.require_current_actor(actor_context=actor_context)
            prior = store.get_operation_result(**operation, artifact_kind="runtime_restriction_perception_response")
            if prior is not None:
                return self.read_persisted(store=store, **scope, response_pin=prior, source_region=source_region, run_id=run_id)
            # The request precedes the external call; each adapter attempt commits before invocation.
            request = RuntimeRestrictionPerceptionRequestV1.create(**basis, created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            preparation = {**operation, "operation": "PREPARE_RUNTIME_RESTRICTION_PERCEPTION"}
            request_pin = store.get_operation_result(**preparation, artifact_kind="runtime_restriction_perception_request")
            if request_pin is None:
                request_pin = store.put_many_with_operation_result(**preparation, artifacts=(request,), result_pin=self.pin(request))
            request = store.get(**scope, artifact_ref=request_pin.artifact_ref, contract_type=RuntimeRestrictionPerceptionRequestV1)
            if request is None or self.pin(request) != request_pin or request.model_dump(mode="json", exclude={"schema_version", "artifact_ref", "artifact_hash", "created_at"}) != basis:
                raise ValueError("PERCEPTION_REQUEST_RECOVERY_MISMATCH")
            attempt_pins = []

            def record(compatibility):
                authority_resolver.require_current_actor(actor_context=actor_context)
                verify_complete_request_compatibility(execution_profile=request.execution_profile,
                    request=dict(instructions=request.instructions, input_text=request.input_text, json_schema=request.output_schema),
                    receipt=compatibility)
                attempt = RuntimeRestrictionPerceptionAttemptV1.create(**scope, run_id=run_id,
                    request_pin=self.pin(request).model_dump(mode="json"), compatibility=compatibility,
                    created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
                store.put(attempt)
                attempt_pins.append(self.pin(attempt))

            result, error = None, None
            execution_started_ns = time.perf_counter_ns()
            try:
                response = await generate(instructions=request.instructions, input_text=request.input_text,
                    json_schema=request.output_schema, schema_name="runtime_restriction_perception_v2", purpose="runtime_restriction_perception", record=record)
                result = asdict(response)
            except Exception as exc:
                error = type(exc).__name__
                from contracts.ai_grounding import StructuredGenerationError
                raw = exc.raw_response if isinstance(exc, StructuredGenerationError) else None
                if raw is not None and attempt_pins:
                    attempt = store.get(**scope, artifact_ref=attempt_pins[-1].artifact_ref,
                        contract_type=RuntimeRestrictionPerceptionAttemptV1)
                    profile = attempt.compatibility.get('profile', {})
                    if attempt.compatibility.get('outcome') != 'COMPATIBLE' or (
                        raw['provider'], raw['model']
                    ) != (profile.get('provider'), profile.get('model')):
                        raise ValueError('PERCEPTION_REJECTED_RESPONSE_BINDING_MISMATCH') from exc
                    result = dict(raw)
            finally:
                execution_elapsed_ns = time.perf_counter_ns() - execution_started_ns
            authority_resolver.require_current_actor(actor_context=actor_context)
            response = RuntimeRestrictionPerceptionResponseV2.create(**scope, run_id=run_id,
                request_pin=self.pin(request).model_dump(mode="json"),
                attempt_pins=[pin.model_dump(mode="json") for pin in attempt_pins], result=result, error=error,
                execution_elapsed_ns=execution_elapsed_ns,
                created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            if error is not None and result is None:
                store.put(response)
                raise ValueError("PERCEPTION_GENERATION_FAILED: " + error)
            pin = store.put_many_with_operation_result(**operation, artifacts=(response,), result_pin=self.pin(response))
            return self.read_persisted(store=store, **scope, response_pin=pin, source_region=source_region, run_id=run_id)

    @classmethod
    def read_persisted(cls, *, store, organization_id, workspace_id, response_pin, source_region, run_id):
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)

        def read(pin, model):
            expected_kind = model.model_fields["schema_version"].default.rsplit(".v", 1)[0]
            if pin.artifact_kind != expected_kind:
                raise ValueError("PERCEPTION_INPUT_KIND_MISMATCH")
            artifact = store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=model)
            if artifact is None or artifact.artifact_hash != pin.artifact_hash or artifact.run_id != run_id:
                raise ValueError("PERCEPTION_INPUT_PIN_MISMATCH")
            return artifact

        raw_response = store.get_raw(**scope, artifact_ref=response_pin.artifact_ref)
        if raw_response is None or raw_response["artifact_hash"] != response_pin.artifact_hash:
            raise ValueError("PERCEPTION_INPUT_PIN_MISMATCH")
        response_model = {
            "runtime_restriction_perception_response.v1": RuntimeRestrictionPerceptionResponseV1,
            "runtime_restriction_perception_response.v2": RuntimeRestrictionPerceptionResponseV2,
        }.get(raw_response["schema_version"])
        if response_model is None:
            raise ValueError("PERCEPTION_INPUT_KIND_MISMATCH")
        response = read(response_pin, response_model)
        request = read(response.request_pin, RuntimeRestrictionPerceptionRequestV1)
        if request.source_region != source_region or json.loads(request.input_text) != cls.input_payload(source_region):
            raise ValueError("PERCEPTION_SOURCE_BINDING_MISMATCH")
        attempts = [read(pin, RuntimeRestrictionPerceptionAttemptV1) for pin in response.attempt_pins]
        for attempt in attempts:
            verify_complete_request_compatibility(execution_profile=request.execution_profile,
                request=dict(instructions=request.instructions, input_text=request.input_text, json_schema=request.output_schema),
                receipt=attempt.compatibility)
        if any(attempt.request_pin != response.request_pin for attempt in attempts):
            raise ValueError("PERCEPTION_ATTEMPT_BINDING_MISMATCH")
        if response.error is not None or response.result is None:
            raise ValueError("PERCEPTION_GENERATION_FAILED: " + str(response.error))
        if not attempts or attempts[-1].compatibility.get("outcome") != "COMPATIBLE" or response.result.get("audit", {}).get("compatibility") != attempts[-1].compatibility:
            raise ValueError("PERCEPTION_COMPATIBILITY_RECEIPT_REQUIRED")
        profile = attempts[-1].compatibility["profile"]
        if (response.result.get("provider"), response.result.get("model")) != (profile["provider"], profile["model"]):
            raise ValueError("PERCEPTION_PROVIDER_BINDING_MISMATCH")
        evidence = {item["key"]: item["text"] for item in cls.input_payload(source_region)["evidence"]}
        if request.output_schema == RestrictionPerceptionProposalV1.model_json_schema():
            proposal = RestrictionPerceptionProposalV1.model_validate(response.result["parsed"])
            observations = []
            for observation in proposal.observations:
                resolved = {"classification": observation.classification}
                for field in type(observation).model_fields:
                    if field == "classification":
                        continue
                    value = getattr(observation, field)
                    resolved[field] = (cls.resolve_quote(value, evidence) if field == "clause"
                        else [cls.resolve_quote(quote, evidence) for quote in value])
                observations.append(resolved)
            output = RestrictionPerceptionOutputV1.model_validate(dict(
                covered_evidence_keys=proposal.covered_evidence_keys, observations=observations))
        elif request.output_schema == RestrictionPerceptionOutputV1.model_json_schema():
            # Historical offset proposals retain their original strict validation.
            output = RestrictionPerceptionOutputV1.model_validate(response.result["parsed"])
        else:
            raise ValueError("PERCEPTION_OUTPUT_SCHEMA_UNSUPPORTED")
        if len(output.covered_evidence_keys) != len(set(output.covered_evidence_keys)) or set(output.covered_evidence_keys) != set(evidence):
            raise ValueError("PERCEPTION_SOURCE_COVERAGE_MISMATCH")
        occurrences = set()
        for observation in output.observations:
            occurrence = (observation.clause.evidence_key, observation.clause.start, observation.clause.end)
            if occurrence in occurrences:
                raise ValueError("PERCEPTION_DUPLICATE_OCCURRENCE")
            occurrences.add(occurrence)
            for field in type(observation).model_fields:
                spans = (observation.clause,) if field == "clause" else getattr(observation, field)
                if field == "classification":
                    continue
                for span in spans:
                    text = evidence.get(span.evidence_key)
                    if text is None or not (0 <= span.start < span.end <= len(text)) or text[span.start:span.end] != span.text:
                        raise ValueError("PERCEPTION_EVIDENCE_SPAN_MISMATCH")
        return response, output
