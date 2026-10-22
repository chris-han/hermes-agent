"""Provider-neutral generation of governed runtime semantic hypotheses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from contracts.ai_grounding import CompatibleStructuredGenerator, StructuredGenerationResult, StructuredGenerationError
from contracts.observed_constraint_hypothesis import (
    EffectiveRestrictionRuntimeContextV1,
    create_observed_constraint_hypothesis,
)
from eos.artifact_hashing import content_hash
from services.effective_restriction_runtime_context_service import (
    EffectiveRestrictionRuntimeContextService,
)
from services.runtime_semantic_hypothesis_prompt import (
    assemble_runtime_semantic_hypothesis_prompt,
    projection_schema_contract,
)
from services.runtime_semantic_retrieval_service import (
    RETRIEVAL_POLICY_HASH, PERCEPTION_RETRIEVAL_POLICY_REF, PERCEPTION_RETRIEVAL_POLICY_HASH,
    RETRIEVAL_POLICY_REF,
    _snake_case_json,
    _validate_source_region,
)
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
)


_FRAME_REF_FIELDS = (
    "target_type_ref",
    "attribute_ref",
    "operator_ref",
    "modality_ref",
    "stage_ref",
    "consequence_ref",
)
_OUTPUT_KEYS = {
    "candidate_frames",
    "assumptions",
    "supporting_evidence_refs",
    "counter_evidence_refs",
    "uncertainty",
}
_SEMANTIC_FRAME_FIELDS = set(_FRAME_REF_FIELDS) | {"value_constraint"}


def hypothesis_generation_output_schema() -> dict[str, Any]:
    frame_properties: dict[str, Any] = {
        field: {"type": "string"} for field in _FRAME_REF_FIELDS
    }
    frame_properties["value_constraint"] = {"type": "object"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OUTPUT_KEYS),
        "properties": {
            "candidate_frames": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(_FRAME_REF_FIELDS) + ["value_constraint"],
                    "properties": frame_properties,
                },
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "supporting_evidence_refs": {"type": "array", "items": {"type": "string"}},
            "counter_evidence_refs": {"type": "array", "items": {"type": "string"}},
            "uncertainty": {"type": "object"},
        },
    }


def _runtime_context(snapshot: Any) -> EffectiveRestrictionRuntimeContextV1:
    return EffectiveRestrictionRuntimeContextV1.model_validate(
        {
            field.alias: (
                "effective_restriction_runtime_context.v1"
                if name == "schema_version"
                else getattr(snapshot, name)
            )
            for name, field in EffectiveRestrictionRuntimeContextV1.model_fields.items()
        }
    )


def _validate_provider_output(
    parsed: Mapping[str, Any],
    *,
    allowed_refs: Mapping[str, set[str]],
    allowed_nested_vocabulary: Mapping[str, set[str]],
    evidence_refs: set[str],
) -> dict[str, Any]:
    if not isinstance(parsed, Mapping) or set(parsed) != _OUTPUT_KEYS:
        raise ValueError("provider output must match the bounded hypothesis schema exactly")
    frames = parsed.get("candidate_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("provider output requires at least one candidate frame")
    normalized_frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate frame must be an object")
        frame = dict(raw)
        if set(frame) != _SEMANTIC_FRAME_FIELDS:
            raise ValueError("provider candidate frame must contain semantic fields exactly")
        for field in _FRAME_REF_FIELDS:
            if frame.get(field) not in allowed_refs[field]:
                raise ValueError("candidate frame uses vocabulary outside active accepted pattern vocabulary")
        _validate_nested_vocabulary(
            frame["value_constraint"], allowed_nested_vocabulary=allowed_nested_vocabulary
        )
        identity = content_hash(
            {field: frame[field] for field in sorted(_SEMANTIC_FRAME_FIELDS)}
        )
        if identity in seen:
            raise ValueError("candidate frames must represent materially distinct interpretations")
        seen.add(identity)
        normalized_frames.append(frame)
    for field in ("assumptions", "supporting_evidence_refs", "counter_evidence_refs"):
        if not isinstance(parsed.get(field), list):
            raise ValueError(f"provider output {field} must be an array")
    supporting = tuple(parsed["supporting_evidence_refs"])
    counter = tuple(parsed["counter_evidence_refs"])
    if not set(supporting + counter).issubset(evidence_refs):
        raise ValueError("provider evidence refs must resolve within the source region EvidenceRefs")
    if not isinstance(parsed.get("uncertainty"), Mapping):
        raise ValueError("provider output uncertainty must be an object")
    return {**dict(parsed), "candidate_frames": normalized_frames}


def _collect_nested_vocabulary(value: Any, target: dict[str, set[str]]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                isinstance(nested, str)
                and (key.endswith("_ref") or key in {"schema_version", "value_kind"})
            ):
                target.setdefault(key, set()).add(nested)
            _collect_nested_vocabulary(nested, target)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_nested_vocabulary(nested, target)


def _validate_nested_vocabulary(
    value: Any, *, allowed_nested_vocabulary: Mapping[str, set[str]]
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            governed_key = key.endswith("_ref") or key in {"schema_version", "value_kind"}
            if governed_key:
                if nested is not None and not isinstance(nested, str):
                    raise ValueError("candidate frame nested governed vocabulary must be a string or null")
                if isinstance(nested, str) and nested not in allowed_nested_vocabulary.get(key, set()):
                    raise ValueError(
                        "candidate frame uses nested vocabulary outside active accepted pattern vocabulary"
                    )
            _validate_nested_vocabulary(
                nested, allowed_nested_vocabulary=allowed_nested_vocabulary
            )
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_nested_vocabulary(
                nested, allowed_nested_vocabulary=allowed_nested_vocabulary
            )


def _validate_snapshot_authority(snapshot: Any, current: EffectiveRestrictionRuntimeContextV1) -> None:
    if (
        (snapshot.retrieval_policy_ref, snapshot.retrieval_policy_hash) !=
        ((PERCEPTION_RETRIEVAL_POLICY_REF, PERCEPTION_RETRIEVAL_POLICY_HASH) if snapshot.query.get("perception_pin")
         else (RETRIEVAL_POLICY_REF, RETRIEVAL_POLICY_HASH))
    ):
        raise ValueError("retrieval snapshot policy pin mismatch")
    snapshot_context = _runtime_context(snapshot)
    payload = snapshot_context.model_dump(mode="json", by_alias=True)
    resolver_hash = payload.pop("effectiveContextResolverHash")
    if resolver_hash != "sha256:" + content_hash(
        {**payload, "resolver_version": "effective_restriction_runtime_context_resolver.v1"}
    ):
        raise ValueError("retrieval snapshot effective context hash mismatch")
    stable_fields = tuple(
        name
        for name in EffectiveRestrictionRuntimeContextV1.model_fields
        if name not in {"effective_at", "effective_context_resolver_hash"}
    )
    if tuple(getattr(snapshot_context, name) for name in stable_fields) != tuple(
        getattr(current, name) for name in stable_fields
    ):
        raise ValueError("retrieval snapshot authority is no longer currently executable")


class RuntimeSemanticHypothesisGenerationService:
    def __init__(
        self,
        *,
        runtime_root: str | Path,
        db_path: str | Path,
        generator: CompatibleStructuredGenerator,
    ) -> None:
        self._runtime_root = Path(runtime_root)
        self._db_path = Path(db_path)
        self._generator = generator

    async def generate_and_persist(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        run_id: str,
        decision_projection_ref: str,
        decision_projection_hash: str,
        request_id: str = "initial",
    ):
        if not decision_projection_ref.startswith("decision-local-projection:v2:"):
            raise ValueError("BOUNDED_DECISION_PROJECTION_REQUIRED")
        store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root, organization_id=organization_id, workspace_id=workspace_id)
        projection = store.get_bounded_projection(
            projection_ref=decision_projection_ref, projection_hash=decision_projection_hash)
        from contracts.runtime_semantic_source_region import source_region_to_legacy_dict
        source_region = source_region_to_legacy_dict(projection.source_region)
        region_ref, text, source_binding, evidence = _validate_source_region(
            organization_id=organization_id, workspace_id=workspace_id, source_region=source_region)
        snapshot = projection.retrieval_snapshot
        if snapshot.run_id != run_id:
            raise ValueError("retrieval snapshot run binding mismatch")
        current_context = EffectiveRestrictionRuntimeContextService(
            self._db_path
        ).resolve_current(
            organization_id=organization_id, workspace_id=workspace_id
        )
        _validate_snapshot_authority(snapshot, current_context)
        if snapshot.query.get("source_region_ref") != region_ref or snapshot.query.get(
            "text_hash"
        ) != "sha256:" + content_hash(text):
            raise ValueError("source region does not match the retrieval snapshot query")
        query_binding = snapshot.query.get("source_binding")
        if query_binding != source_binding.model_dump(mode="json"):
            raise ValueError("source binding does not match the retrieval snapshot query")
        if [
            _snake_case_json(item.model_dump(mode="json")) for item in evidence
        ] != snapshot.query.get("evidence_refs"):
            raise ValueError("source evidence does not match the retrieval snapshot query")

        selected_pairs = [(item.ref, item.hash) for item in snapshot.ordered_material
                          if item.kind == "canonical_pattern" and item.ref == projection.canonical_pattern_ref]
        schema = hypothesis_generation_output_schema()
        schema_contract = projection_schema_contract(projection, schema)
        allowed_refs = {field: set(values) for field, values in schema_contract["allowed_frame_refs"].items()}
        allowed_nested_vocabulary: dict[str, set[str]] = {}
        for frame in schema_contract["allowed_candidate_frames"]:
            _collect_nested_vocabulary(frame["value_constraint"], allowed_nested_vocabulary)
        prompt = assemble_runtime_semantic_hypothesis_prompt(
            projection=projection,
            output_schema=schema,
        )
        request = {
            "decision_projection_ref": projection.projection_ref,
            "decision_projection_hash": projection.projection_hash,
            "instructions": prompt.rendered_text, "input_text": prompt.input_text,
            "json_schema": schema,
        }
        identity = getattr(self._generator, "compatibility_execution_identity", None)
        generate = getattr(self._generator, "generate_compatible_structured", None)
        if not isinstance(identity, dict) or not callable(generate):
            raise StructuredGenerationError("COMPATIBLE_JUDGMENT_ADAPTER_REQUIRED")
        execution = store.prepare_judgment_execution(run_id=run_id, request=request,
            execution_profile=identity, request_id=request_id)
        with store.judgment_execution_lock(execution):
            response = store.get_judgment_response(execution=execution)
            if response is None:
                attempt = store.latest_judgment_attempt(execution=execution)
                initial_attempt = attempt

                def record(receipt):
                    nonlocal attempt
                    attempt = store.put_judgment_attempt(execution=execution,
                        compatibility=receipt, predecessor=attempt)
                    if receipt.get("outcome") == "COMPATIBLE":
                        _validate_snapshot_authority(snapshot, EffectiveRestrictionRuntimeContextService(
                            self._db_path).resolve_current(organization_id=organization_id, workspace_id=workspace_id))

                try:
                    result = await generate(instructions=prompt.rendered_text,
                        input_text=prompt.input_text, json_schema=schema, record=record)
                except Exception as exc:
                    raw = exc.raw_response if isinstance(exc, StructuredGenerationError) else None
                    if raw is not None and attempt is not None and attempt.compatibility.get('outcome') == 'COMPATIBLE':
                        rejected = StructuredGenerationResult(parsed={}, text=raw['text'],
                            provider=raw['provider'], model=raw['model'], agent_id='default', usage={},
                            audit=dict(compatibility=attempt.compatibility, structured_response_rejected=True))
                        store.put_judgment_response(execution=execution, attempt=attempt, result=rejected)
                    if attempt is initial_attempt:
                        record(dict(schema_version="complete_request_compatibility.v1", profile={},
                            outcome="NO_COMPATIBLE_MODEL", reason=getattr(exc, "code", "EXECUTION_UNRESOLVED")))
                    raise
                if attempt is None:
                    raise StructuredGenerationError("JUDGMENT_COMPATIBILITY_RECEIPT_REQUIRED")
                response = store.put_judgment_response(execution=execution, attempt=attempt, result=result)
            result = StructuredGenerationResult(**response.result)
            if result.audit.get('structured_response_rejected') is True:
                raise StructuredGenerationError('JUDGMENT_STRUCTURED_RESPONSE_REJECTED',
                    raw_response=dict(text=result.text, provider=result.provider, model=result.model))
        content = _validate_provider_output(
            result.parsed,
            allowed_refs=allowed_refs,
            allowed_nested_vocabulary=allowed_nested_vocabulary,
            evidence_refs={f"e{i}" for i in range(len(projection.evidence))},
        )
        anchors = {(item.evidence_ref.source_identity_ref, item.evidence_ref.selector_hash,
                    item.evidence_ref.evidence_role): item.evidence_ref.evidence_ref
                   for item in projection.source_region.evidence}
        evidence_handles = {f"e{i}": anchors[(item.source_ref, item.selector_hash, item.evidence_role)]
                            for i, item in enumerate(projection.ordered_evidence())}
        for field in ("supporting_evidence_refs", "counter_evidence_refs"):
            content[field] = [evidence_handles[ref] for ref in content[field]]
        context = _runtime_context(snapshot)
        request.update(execution_request_ref=execution.artifact_ref, execution_request_hash=execution.artifact_hash,
                       execution_response_ref=response.artifact_ref, execution_response_hash=response.artifact_hash)
        hypothesis = create_observed_constraint_hypothesis(
            organization_id=organization_id,
            workspace_id=workspace_id,
            run_id=run_id,
            source_binding=source_binding,
            evidence_refs=evidence,
            source_anchor_refs=[],
            candidate_frames=content["candidate_frames"],
            retrieved_pattern_refs=[ref for ref, _digest in selected_pairs],
            retrieved_pattern_hashes=[digest for _ref, digest in selected_pairs],
            assumptions=content["assumptions"],
            supporting_evidence_refs=content["supporting_evidence_refs"],
            counter_evidence_refs=content["counter_evidence_refs"],
            uncertainty=content["uncertainty"],
            provider_ref=result.provider,
            model_ref=result.model,
            prompt_asset_ref=prompt.prompt_asset_ref,
            prompt_asset_hash=prompt.prompt_asset_hash,
            schema_ref="observed_constraint_hypothesis.v3",
            judgment_request=request,
            retrieval_snapshot_ref=snapshot.snapshot_id,
            retrieval_snapshot_hash=snapshot.snapshot_hash,
            runtime_context=context,
            lifecycle_state="proposed",
        )
        return store.put_hypothesis(hypothesis, judgment_request=request)


__all__ = [
    "RuntimeSemanticHypothesisGenerationService",
    "hypothesis_generation_output_schema",
]
