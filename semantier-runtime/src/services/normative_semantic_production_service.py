"""Produce advisory semantic frames from replay-verified source/schema inputs."""
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from contracts.normative_schema import NormativeSchemaContractV1, NormativeSchemaProposalEvidenceV1
from contracts.normative_semantics import (
    ASCII_ID, NormativeFrameV1, NormativeSemanticCandidateSetV1,
    NormativeSemanticProposalEvidenceV1, NormativeSemanticCorrectionReviewV1,
)
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import canonical_json, content_hash
from services.normative_schema_production_service import NormativeSchemaProductionService
from services.normative_legal_source_context import LEGAL_SOURCE_CONTEXT_HASH
from services.normative_source_prompt_view import provider_input_text, verify_provider_input, PROMPT_VIEW_HASH
from storage.ontology_build_store import OntologyBuildStore


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: tuple[NormativeFrameV1, ...]
    unresolved_slot_refs: tuple[str, ...]


class NormativeSemanticProductionService:
    COMPILER_REF = "compiler:source-normative-semantics:v3"

    def __init__(self, *, llm, provider: str, model: str):
        self.llm, self.provider, self.model = llm, provider, model
        self.instructions = (Path(__file__).resolve().parents[1] /
            "prompts/normative_semantic_production_v19.txt").read_text(encoding="utf-8")
        self.compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes() + PROMPT_VIEW_HASH.encode('ascii') + LEGAL_SOURCE_CONTEXT_HASH.encode('ascii')).hexdigest()
        self.output_schema = _Proposal.model_json_schema()
        for definition in self.output_schema.get("$defs", {}).values():
            properties = definition.get("properties", {})
            if "frame_hash" in properties:
                del properties["frame_hash"]
                definition["required"] = sorted((set(definition["required"]) - {"frame_hash"})
                    | {"frame_kind", "epistemic_status"})
            if properties.get("frame_kind", {}).get("const") in {"scope", "qualification"}:
                definition["required"] = sorted(set(definition["required"]) | {"conditions"})
            if properties.get("frame_kind", {}).get("const") == "entity_distinction":
                definition["required"] = sorted(set(definition["required"]) | {"rule_effects"})

    @staticmethod
    def _pin(candidate):
        return ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
            artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash)

    async def produce_persisted(self, *, store, organization_id, workspace_id,
                                actor_ref, schema_pin, request_id, created_at, predecessor_proposal_pin=None, teaching_context=None):
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("semantic production requires the production artifact owner")
        if created_at.utcoffset() != timedelta(0):
            raise ValueError("semantic production requires UTC time")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        source_schema, source_report = NormativeSchemaProductionService.read_persisted(
            store=store, **scope, schema_pin=schema_pin)
        if source_schema.teaching_candidate:
            from services.teaching_candidate_scope import require_teaching_scope
            from storage.ontology_build_store import ontology_build_store_path
            require_teaching_scope(teaching_context, **scope)
            if (actor_ref != "user:" + teaching_context.user_id or
                store._path.resolve() != ontology_build_store_path(teaching_context.workspace_root).resolve()):
                raise PermissionError("NORMATIVE_SEMANTIC_TEACHING_SCOPE_MISMATCH")
        correction_pin = ({"predecessor_proposal": predecessor_proposal_pin.model_dump(mode="json")}
            if predecessor_proposal_pin is not None else {})
        request_hash = "sha256:" + content_hash(dict(**correction_pin, schema=schema_pin.model_dump(mode="json"),
            instructions=self.instructions, output_schema=self.output_schema,
            compiler_hash=self.compiler_hash, provider=self.provider, model=self.model))
        operation = dict(**scope, actor_ref=actor_ref, operation="PRODUCE_NORMATIVE_SEMANTICS",
                         request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="normative_semantic_candidate_set")
        if prior is not None:
            return self.read_persisted(store=store, **scope, candidate_pin=prior)
        generation_operation = {**operation, "operation": "PROPOSE_NORMATIVE_SEMANTICS"}
        generation_pin = store.get_operation_result(**generation_operation,
            artifact_kind="normative_semantic_proposal_evidence")
        if generation_pin is None:
            schema, report = source_schema, source_report
            if schema.fact_definitions is None:
                raise ValueError("NORMATIVE_SEMANTIC_FACT_DEFINITIONS_REQUIRED")
            source_receipt = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
                                       contract_type=NormativeSchemaProposalEvidenceV1)
            request = dict(source_context=json.loads(source_receipt.input_text),
                normative_schema=schema.model_dump(mode="json"), schema_adequacy=report.model_dump(mode="json"))
            if predecessor_proposal_pin is not None:
                request["zz_correction_context"] = self._correction_context(
                    store, scope, predecessor_proposal_pin, request)
            input_text = canonical_json(request)
            from contracts.ai_grounding import StructuredGenerationError
            rejected_text = None
            usage = None
            try:
                generated = await self.llm.generate_structured(instructions=self.instructions,
                    input_text=provider_input_text(input_text), json_schema=self.output_schema,
                    schema_name="normative_semantic_proposal_v1", purpose="normative_semantic_proposal")
                provider, model, output = generated.provider, generated.model, generated.parsed
                usage = dict(generated.usage)
            except StructuredGenerationError as exc:
                raw = exc.raw_response
                if raw is None:
                    raise
                provider, model, rejected_text = raw['provider'], raw['model'], raw['text']
                output = {}
                usage = raw.get('usage')
            if (provider, model) != (self.provider, self.model):
                raise ValueError("NORMATIVE_SEMANTIC_PROVIDER_ATTRIBUTION_MISMATCH")
            basis = dict(schema_version="normative_semantic_proposal_evidence.v1", **scope,
                instructions=self.instructions, input_text=input_text, provider_input_text=provider_input_text(input_text), output_schema=self.output_schema,
                output=output, provider=provider, model=model,
                compiler_ref=self.COMPILER_REF, compiler_hash=self.compiler_hash,
                created_at=created_at.isoformat().replace("+00:00", "Z"))
            if usage is not None:
                basis["usage"] = usage
            if rejected_text is not None:
                basis['rejected_response_text'] = rejected_text
            digest = content_hash(basis)
            receipt = NormativeSemanticProposalEvidenceV1(evidence_ref="normative-semantic-proposal:" + digest,
                evidence_hash="sha256:" + digest, **basis)
            generation_pin = store.put_many_with_operation_result(**generation_operation, artifacts=(receipt,),
                result_pin=ArtifactPinV1(artifact_kind="normative_semantic_proposal_evidence",
                    artifact_ref=receipt.evidence_ref, artifact_hash=receipt.evidence_hash))
        receipt = self._receipt(store, scope, generation_pin.artifact_ref, generation_pin.artifact_hash)
        candidate = self._compile(receipt)
        winner = store.put_many_with_operation_result(**operation, artifacts=(candidate,),
            result_pin=self._pin(candidate))
        return self.read_persisted(store=store, **scope, candidate_pin=winner)

    @classmethod
    def review_rejected_proposal_persisted(cls, *, ctx, authority_resolver, store,
        predecessor_pin, corrected_output, reviewer_kind, rationale, request_id, reviewed_at,
        predecessor_candidate_pin=None):
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import ontology_build_store_path
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("semantic correction requires production owners")
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("NORMATIVE_SEMANTIC_REVIEW_WORKSPACE_MISMATCH")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authorize = lambda: authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN")
        authorize()
        if reviewed_at.utcoffset() != timedelta(0) or not rationale.strip():
            raise ValueError("semantic review requires UTC time and a rationale")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if predecessor_pin.artifact_kind != "normative_semantic_proposal_evidence":
            raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_KIND_MISMATCH")
        prior = cls._receipt(store, scope, predecessor_pin.artifact_ref, predecessor_pin.artifact_hash)
        cls._correction_context(store, scope, predecessor_pin, json.loads(prior.input_text))
        lineage = {}
        if predecessor_candidate_pin is not None:
            predecessor = cls.read_persisted(store=store, **scope, candidate_pin=predecessor_candidate_pin)
            if predecessor.normative_schema_hash != json.loads(prior.input_text)['normative_schema']['schema_hash']:
                raise ValueError("NORMATIVE_SEMANTIC_REVIEW_SCHEMA_MISMATCH")
            if reviewed_at < predecessor.compiled_at:
                raise ValueError("semantic review cannot precede its predecessor")
            lineage = dict(predecessor_candidate_pin=predecessor_candidate_pin.model_dump(mode='json'))
        compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes() + PROMPT_VIEW_HASH.encode('ascii') + LEGAL_SOURCE_CONTEXT_HASH.encode('ascii')).hexdigest()
        request_hash = "sha256:" + content_hash(dict(predecessor=predecessor_pin.model_dump(mode="json"),
            corrected_output=corrected_output, reviewer_kind=reviewer_kind, rationale=rationale, compiler_hash=compiler_hash, **lineage))
        operation = dict(**scope, actor_ref="user:" + ctx.user_id, operation="REVIEW_REJECTED_NORMATIVE_SEMANTICS",
            request_id=request_id, request_hash=request_hash)
        winner = store.get_operation_result(**operation, artifact_kind="normative_semantic_candidate_set")
        if winner is None:
            basis = dict(schema_version="normative_semantic_correction_review.v1", **scope,
                actor_ref=operation["actor_ref"], reviewer_kind=reviewer_kind,
                predecessor_proposal_ref=prior.evidence_ref, predecessor_proposal_hash=prior.evidence_hash,
                corrected_output=corrected_output, rationale=rationale, compiler_hash=compiler_hash,
                reviewed_at=reviewed_at.isoformat().replace("+00:00", "Z"), **lineage)
            digest = content_hash(basis)
            review = NormativeSemanticCorrectionReviewV1(review_ref="normative-semantic-review:"+digest,
                review_hash="sha256:"+digest, **basis)
            candidate = cls._compile(prior, review=review)
            # Verify retained source/schema through the same owner before publication.
            schema, adequacy = NormativeSchemaProductionService.read_persisted(store=store, **scope,
                schema_pin=ArtifactPinV1(artifact_kind="normative_schema_contract",
                    artifact_ref=candidate.normative_schema_ref, artifact_hash=candidate.normative_schema_hash))
            source_receipt = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
                contract_type=NormativeSchemaProposalEvidenceV1)
            retained = json.loads(prior.input_text)
            if (retained["normative_schema"] != schema.model_dump(mode="json")
                or retained["schema_adequacy"] != adequacy.model_dump(mode="json")
                or source_receipt is None or retained["source_context"] != json.loads(source_receipt.input_text)):
                raise ValueError("NORMATIVE_SEMANTIC_REVIEW_SCHEMA_MISMATCH")
            winner = store.put_many_with_operation_result(**operation, artifacts=(review, candidate),
                result_pin=cls._pin(candidate), verify_before_commit=authorize)
        return cls.read_persisted(store=store, **scope, candidate_pin=winner)

    @classmethod
    def _correction_context(cls, store, scope, pin, request, *, retained_diagnostic=None):
        if pin.artifact_kind != "normative_semantic_proposal_evidence":
            raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_KIND_MISMATCH")
        prior = cls._receipt(store, scope, pin.artifact_ref, pin.artifact_hash)
        old = json.loads(prior.input_text)
        keys = ("source_context", "normative_schema", "schema_adequacy")
        if any(old.get(key) != request.get(key) for key in keys):
            raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_INPUT_MISMATCH")
        from contracts.ai_grounding import StructuredGenerationError
        try:
            cls._compile(prior)
        except (ValueError, StructuredGenerationError) as exc:
            error = str(exc)
        else:
            raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_REQUIRES_REJECTED_PROPOSAL")
        rejected = prior.output
        if prior.rejected_response_text is not None:
            rejected = prior.rejected_response_text
            # Diagnostic only: preserve the exact rejected bytes; never publish
            # a parsed or patched rejected proposal as a candidate.
            try:
                value = json.loads(rejected)
                _Proposal.model_validate({**value, "frames": [
                    {**frame, "frame_hash": "sha256:" + "0" * 64}
                    for frame in value.get("frames", [])]})
            except (ValueError, TypeError, AttributeError) as exc:
                error += ": " + str(exc)[:4000]
        if retained_diagnostic is not None:
            if not isinstance(retained_diagnostic, str) or not retained_diagnostic.strip():
                raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_DIAGNOSTIC_INVALID")
            # Diagnostics are retained provider input, not executable authority.
            # Replay binds their bytes through the receipt, not current-library
            # exception wording. Source, predecessor and rejection still verify.
            error = retained_diagnostic
        return dict(predecessor=pin.model_dump(mode="json"),
            rejected_output=rejected, rejection_error=error)

    @staticmethod
    def _receipt(store, scope, ref, digest):
        receipt = store.get(**scope, artifact_ref=ref, contract_type=NormativeSemanticProposalEvidenceV1)
        if receipt is None or receipt.evidence_hash != digest:
            raise ValueError("NORMATIVE_SEMANTIC_PROPOSAL_PIN_MISMATCH")
        return receipt

    @classmethod
    def _compile(cls, receipt, *, review=None):
        verify_provider_input(receipt)
        if review is not None:
            review = NormativeSemanticCorrectionReviewV1.model_validate(review)
            if (review.organization_id, review.workspace_id, review.predecessor_proposal_ref, review.predecessor_proposal_hash) != (
                receipt.organization_id, receipt.workspace_id, receipt.evidence_ref, receipt.evidence_hash):
                raise ValueError("NORMATIVE_SEMANTIC_REVIEW_PREDECESSOR_MISMATCH")
        if receipt.rejected_response_text is not None and review is None:
            from contracts.ai_grounding import StructuredGenerationError
            raise StructuredGenerationError('NORMATIVE_SEMANTIC_STRUCTURED_RESPONSE_REJECTED',
                raw_response=dict(text=receipt.rejected_response_text, provider=receipt.provider, model=receipt.model))
        request = json.loads(receipt.input_text)
        source = request["source_context"]
        schema = NormativeSchemaContractV1.model_validate(request["normative_schema"])
        if (schema.organization_id, schema.workspace_id) != (receipt.organization_id, receipt.workspace_id):
            raise ValueError("NORMATIVE_SEMANTIC_SCOPE_MISMATCH")
        output = NormativeSchemaProductionService.resolve_source_citations(
            review.corrected_output if review else receipt.output, source)
        if receipt.compiler_ref in {"compiler:source-normative-semantics:v2", "compiler:source-normative-semantics:v3"}:
            blocks = {item["evidenceRef"]: item["provenance"]["source_block_id"] for item in NormativeSchemaProductionService.source_evidence(source)}
            region_keys = {key: blocks[ref] for key, ref in source.get("source_citation_keys", {}).items()}
            output = {**output, "frames": [{**frame, "source_region_refs": [region_keys.get(ref, ref)
                for ref in frame.get("source_region_refs", ())]} for frame in output.get("frames", ())]}
        frames = []
        adapter = TypeAdapter(NormativeFrameV1)
        for raw in output.get("frames", []):
            if "frame_hash" in raw or "epistemic_status" not in raw:
                raise ValueError("semantic proposal must declare epistemic status, not frame hashes")
            frame = adapter.validate_python({**raw, "frame_hash": "sha256:" + "0" * 64})
            digest = content_hash(frame.model_dump(mode="json", exclude={"frame_hash"}))
            frames.append(frame.model_copy(update={"frame_hash": "sha256:" + digest}))
        parsed = _Proposal.model_validate({**output, "frames": frames})
        if not parsed.frames:
            raise ValueError("semantic proposal contains no supported frames")
        cls._validate_frames(parsed.frames, NormativeSchemaProductionService.source_evidence(source))
        if schema.fact_definitions is not None:
            facts = {fact.fact_ref: fact for fact in schema.fact_definitions}
            for frame in parsed.frames:
                for predicate in (getattr(frame, "conditions", ()) or ()):
                    fact = facts.get(predicate.fact_ref)
                    comparison = facts.get(predicate.comparison_fact_ref)
                    if predicate.comparison_fact_ref is not None:
                        allowed = ({"integer", "number"} if predicate.operator in {"RATIO_LE", "RATIO_GT"}
                            else {"date"} if predicate.operator == "CALENDAR_YEARS_AFTER" else {"string"})
                        valid = fact is not None and comparison is not None and fact.value_type in allowed and comparison.value_type in allowed
                    else:
                        valid = fact is not None and fact.accepts_predicate(predicate.operator, predicate.expected_value)
                    if not valid:
                        raise ValueError("NORMATIVE_SEMANTIC_FACT_BINDING_MISMATCH: "
                            + json.dumps(dict(frame_id=frame.frame_id, predicate_id=predicate.predicate_id,
                                fact_ref=predicate.fact_ref, declared_value_type=fact.value_type if fact else None,
                                operator=predicate.operator, expected_literal=predicate.expected_value), ensure_ascii=True))
                if frame.frame_kind == "rule":
                    for effect in frame.rule_effects or ():
                        fact = facts.get(effect.fact_ref)
                        if fact is None or not fact.accepts_predicate(
                            effect.operator, effect.expected_value
                        ):
                            raise ValueError("NORMATIVE_SEMANTIC_RULE_EFFECT_BINDING_MISMATCH")
                        if not set(effect.evidence_refs).issubset(frame.evidence_refs):
                            raise ValueError("NORMATIVE_SEMANTIC_RULE_EFFECT_EVIDENCE_MISMATCH")
                if frame.frame_kind == "temporal_condition":
                    fact = facts.get(frame.event_time_ref)
                    if fact is None or fact.value_type != "date":
                        raise ValueError("NORMATIVE_SEMANTIC_EVENT_TIME_FACT_MISMATCH")
        unresolved = set(parsed.unresolved_slot_refs)
        for kind, definition in (("scope", "ScopeFrameV1"), ("qualification", "QualificationFrameV1")):
            retained_schema = receipt.output_schema.get("$defs", {}).get(definition, {})
            if "conditions" in retained_schema.get("required", ()):
                unresolved.update(frame.frame_id for frame in frames if frame.frame_kind == kind and not frame.conditions)
        entity_schema = receipt.output_schema.get("$defs", {}).get("EntityDistinctionFrameV1", {})
        if "rule_effects" in entity_schema.get("required", ()):
            unresolved.update(frame.frame_id for frame in frames
                if frame.frame_kind == "entity_distinction" and not frame.rule_effects)
        unresolved.update(slot.dimension for slot in schema.slots if slot.epistemic_status != "FOUND")
        unresolved.update(frame.frame_id for frame in frames if frame.epistemic_status != "FOUND")
        kinds = {frame.frame_kind for frame in frames}
        if "relative_temporal_condition" in kinds:
            kinds.add("temporal_condition")
        # No absence is inferred. Completeness needs each DSL dimension represented.
        unresolved.update({"rule", "exception", "scope", "qualification", "precedence",
                           "temporal_condition", "entity_distinction"} - kinds)
        if receipt.compiler_ref == "compiler:source-normative-semantics:v3" and schema.source_review_ref is not None:
            reviewed_absence = {slot.dimension for slot in schema.slots if slot.epistemic_status == "NONE_EVIDENCED"}
            if reviewed_absence & kinds:
                raise ValueError("semantic proposal contradicts reviewed source absence")
            unresolved.difference_update(reviewed_absence)
        if any(ASCII_ID.fullmatch(ref) is None for ref in unresolved):
            raise ValueError("unresolved slot references must be ASCII-stable")
        boundary = source["boundary"]
        basis = dict(organization_id=receipt.organization_id, workspace_id=receipt.workspace_id,
            **{key: boundary[key] for key in ("source_identity_ref", "source_hash",
                "canonical_source_ir_ref", "canonical_source_ir_hash")},
            competency_contract_ref=schema.competency_contract_ref,
            competency_contract_hash=schema.competency_contract_hash,
            normative_schema_ref=schema.schema_ref, normative_schema_hash=schema.schema_hash,
            profile_ref=schema.profile_ref, profile_hash=schema.profile_hash,
            compiler_ref=receipt.compiler_ref, compiler_hash=receipt.compiler_hash,
            prompt_hash="sha256:" + sha256(receipt.instructions.encode()).hexdigest(),
            provider_ref=receipt.provider, model_ref=receipt.model,
            config_hash="sha256:" + content_hash(dict(provider=receipt.provider, model=receipt.model,
                                                      output_schema=receipt.output_schema)),
            proposal_evidence_ref=receipt.evidence_ref, proposal_evidence_hash=receipt.evidence_hash,
            frames=[f.model_dump(mode="json") for f in sorted(frames, key=lambda f: f.frame_id)],
            completeness_state="UNRESOLVED" if unresolved else "COMPLETE",
            unresolved_slot_refs=sorted(unresolved))
        if review is not None:
            basis.update(source_review_ref=review.review_ref, source_review_hash=review.review_hash,
                provider_ref="governed-correction-review:" + review.reviewer_kind,
                model_ref="not_applicable", compiler_hash=review.compiler_hash,
                config_hash="sha256:" + content_hash(dict(review=review.review_hash, output_schema=receipt.output_schema)))
            if review.predecessor_candidate_pin is not None:
                pin = ArtifactPinV1.model_validate(review.predecessor_candidate_pin)
                basis.update(predecessor_candidate_set_ref=pin.artifact_ref,
                    predecessor_candidate_set_hash=pin.artifact_hash, change_cause='bounded_repair',
                    triggering_ref=review.review_ref)
        if schema.teaching_candidate:
            basis["teaching_candidate"] = True
        digest = content_hash(basis)
        return NormativeSemanticCandidateSetV1(candidate_set_ref="normative-candidate:" + digest,
            candidate_set_hash="sha256:" + digest, compiled_at=review.reviewed_at if review else receipt.created_at, **basis)

    @staticmethod
    def _validate_frames(frames, source_evidence):
        evidence = {item["evidenceRef"]: item["provenance"]["source_block_id"] for item in source_evidence}
        by_id = {frame.frame_id: frame for frame in frames}
        if len(by_id) != len(frames):
            raise ValueError("duplicate frame IDs")
        predicate_ids = set()
        for frame in frames:
            if frame.epistemic_status == "NONE_EVIDENCED":
                raise ValueError("semantic proposal cannot invent audited absence")
            if not set(frame.evidence_refs).issubset(evidence):
                raise ValueError("frame evidence outside retained source")
            if set(frame.source_region_refs) != {evidence[ref] for ref in frame.evidence_refs}:
                raise ValueError("frame regions must bind the exact cited evidence")
            for key, value in frame.model_dump().items():
                if key.endswith("_ref") or key.endswith("_refs") or key in {"actor_scope", "object_scope"}:
                    refs = value if isinstance(value, tuple) else (value,)
                    if any(not isinstance(ref, str) or ASCII_ID.fullmatch(ref) is None for ref in refs):
                        raise ValueError("semantic machine references must be ASCII-stable")
            for predicate in (getattr(frame, "conditions", ()) or ()):
                if predicate.predicate_id in predicate_ids or any(ASCII_ID.fullmatch(ref) is None
                    for ref in (predicate.predicate_id, predicate.fact_ref)):
                    raise ValueError("predicate identity must be unique and ASCII-stable")
                predicate_ids.add(predicate.predicate_id)
                if not set(predicate.evidence_refs).issubset(frame.evidence_refs):
                    raise ValueError("predicate evidence must belong to its frame")
            if frame.frame_kind == "temporal_condition" and frame.epistemic_status == "FOUND":
                if date.fromisoformat(frame.effective_from) > date.fromisoformat(frame.effective_until):
                    raise ValueError("temporal window is reversed")
        for frame in frames:
            if frame.frame_kind == "rule":
                for field, kind in (("exception_refs", "exception"), ("scope_refs", "scope"),
                    ("qualification_refs", "qualification"), ("temporal_condition_refs", "temporal_condition"),
                    ("precedence_refs", "precedence")):
                    for ref in getattr(frame, field):
                        target = by_id.get(ref)
                        allowed_kinds = {kind, "relative_temporal_condition"} if kind == "temporal_condition" else {kind}
                        if target is None or target.frame_kind not in allowed_kinds:
                            raise ValueError("semantic reference has missing or wrong-kind target")
                        if kind in {"exception", "scope"} and target.applies_to_rule_ref != frame.frame_id:
                            raise ValueError("semantic applicability points to another rule")
            elif frame.frame_kind in {"exception", "scope"}:
                target = by_id.get(frame.applies_to_rule_ref)
                field = "exception_refs" if frame.frame_kind == "exception" else "scope_refs"
                if target is None or target.frame_kind != "rule" or frame.frame_id not in getattr(target, field):
                    raise ValueError("semantic applicability must be reciprocal")
            elif frame.frame_kind == "precedence":
                if (frame.higher_priority_ref == frame.lower_priority_ref or
                    any(ref not in by_id for ref in (frame.higher_priority_ref, frame.lower_priority_ref)) or
                    not set(frame.condition_refs).issubset(predicate_ids)):
                    raise ValueError("precedence references must resolve")
            elif frame.frame_kind == "entity_distinction":
                for effect in frame.rule_effects or ():
                    target = by_id.get(effect.rule_ref)
                    if target is None or target.frame_kind != "rule":
                        raise ValueError("entity distinction effects must bind exact rules")

    @classmethod
    def read_persisted(cls, *, store, organization_id, workspace_id, candidate_pin):
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if candidate_pin.artifact_kind != "normative_semantic_candidate_set":
            raise ValueError("NORMATIVE_SEMANTIC_INPUT_KIND_MISMATCH")
        candidate = store.get(**scope, artifact_ref=candidate_pin.artifact_ref,
                              contract_type=NormativeSemanticCandidateSetV1)
        if candidate is None or candidate.candidate_set_hash != candidate_pin.artifact_hash or not candidate.proposal_evidence_ref:
            raise ValueError("NORMATIVE_SEMANTIC_CANDIDATE_PIN_MISMATCH")
        receipt = cls._receipt(store, scope, candidate.proposal_evidence_ref, candidate.proposal_evidence_hash)
        schema, _ = NormativeSchemaProductionService.read_persisted(store=store, **scope,
            schema_pin=ArtifactPinV1(artifact_kind="normative_schema_contract",
                artifact_ref=candidate.normative_schema_ref, artifact_hash=candidate.normative_schema_hash))
        source_receipt = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
                                  contract_type=NormativeSchemaProposalEvidenceV1)
        review = None
        if candidate.source_review_ref is not None:
            review = store.get(**scope, artifact_ref=candidate.source_review_ref,
                contract_type=NormativeSemanticCorrectionReviewV1)
            if review is None or review.review_hash != candidate.source_review_hash:
                raise ValueError("NORMATIVE_SEMANTIC_REVIEW_PIN_MISMATCH")
            if review.predecessor_candidate_pin is not None:
                predecessor = cls.read_persisted(store=store, **scope,
                    candidate_pin=ArtifactPinV1.model_validate(review.predecessor_candidate_pin))
                if predecessor.normative_schema_hash != candidate.normative_schema_hash or predecessor.compiled_at > candidate.compiled_at:
                    raise ValueError("NORMATIVE_SEMANTIC_REVIEW_LINEAGE_MISMATCH")
        request = json.loads(receipt.input_text)
        context = request.get("zz_correction_context", request.get("correction_context"))
        if context is not None:
            pin = ArtifactPinV1.model_validate(context["predecessor"])
            if context != cls._correction_context(store, scope, pin, request,
                retained_diagnostic=context.get("rejection_error")):
                raise ValueError("NORMATIVE_SEMANTIC_CORRECTION_REPLAY_MISMATCH")
        if (request["normative_schema"] != schema.model_dump(mode="json") or
            request["source_context"] != json.loads(source_receipt.input_text) or cls._compile(receipt, review=review) != candidate):
            raise ValueError("NORMATIVE_SEMANTIC_REPLAY_MISMATCH")
        return candidate
