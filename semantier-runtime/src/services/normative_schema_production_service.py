"""Produce source-bound normative schema through existing governed input owners."""
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contracts.normative_schema import (
    NormativeSchemaContractV1, NormativeSchemaProposalEvidenceV1,
    NormativeSchemaSlotV1, SchemaAdequacyReportV1,
    NormativeFactDefinitionV1, SourceSearchCoverageManifestV1,
)
from contracts.ontology_building import (
    ArtifactPinV1, CanonicalEvidenceBoundaryV1, CompetencyContractV1, ConversationDecisionV1,
)
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
from services.corpus_graph_schema_capability_service import CorpusGraphSchemaCapabilityService
from services.graph_intent_discovery_service import GraphIntentDiscoveryService
from services.ontology_schema_adequacy_service import OntologySchemaAdequacyService
from services.normative_legal_source_context import LEGAL_SOURCE_CONTEXT_HASH
from services.normative_source_prompt_view import provider_input_text, verify_provider_input, PROMPT_VIEW_HASH
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slots: tuple[NormativeSchemaSlotV1, ...]
    fact_definitions: tuple[NormativeFactDefinitionV1, ...] | None = None


class NormativeSchemaProductionService:
    COMPILER_REF = "compiler:source-normative-schema:v1"

    def __init__(self, *, llm, provider: str, model: str):
        self.llm, self.provider, self.model = llm, provider, model
        self.instructions = (Path(__file__).resolve().parents[1] /
            "prompts/normative_schema_production_v13.txt").read_text(encoding="utf-8")
        self.compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes() + PROMPT_VIEW_HASH.encode('ascii') + LEGAL_SOURCE_CONTEXT_HASH.encode('ascii')).hexdigest()
        self.output_schema = _Proposal.model_json_schema()
        self.output_schema["required"].append("fact_definitions")

    @staticmethod
    def _pin(kind, artifact, ref_field, hash_field):
        return ArtifactPinV1(artifact_kind=kind, artifact_ref=getattr(artifact, ref_field),
                            artifact_hash=getattr(artifact, hash_field))

    async def produce_persisted(
        self, *, store: OntologyBuildStore, capability: CorpusGraphSchemaCapabilityService,
        competency_pin: ArtifactPinV1, schema_decision_pin: ArtifactPinV1 | None = None,
        teaching_candidate_pin: ArtifactPinV1 | None = None, teaching_context=None,
        request_id: str, created_at: datetime,
        authority_pin: ArtifactPinV1 | None = None, coverage_pin: ArtifactPinV1 | None = None,
        legal_store=None,
    ) -> tuple[NormativeSchemaContractV1, SchemaAdequacyReportV1]:
        if created_at.utcoffset() != timedelta(0):
            raise ValueError("normative schema production requires UTC time")
        if not isinstance(store, OntologyBuildStore) or not isinstance(capability, CorpusGraphSchemaCapabilityService):
            raise TypeError("normative schema production requires production input owners")
        intent_store = capability.discovery.graph_intent_store
        scope = dict(organization_id=capability.organization_id, workspace_id=capability.workspace_id)
        if ((intent_store.organization_id, intent_store.workspace_id) != (capability.organization_id, capability.workspace_id)
            or store._path.resolve() != ontology_build_store_path(intent_store.workspace_path).resolve()):
            raise ValueError("NORMATIVE_SCHEMA_WORKSPACE_OWNER_MISMATCH")
        teaching = teaching_candidate_pin is not None
        if teaching:
            from services.teaching_candidate_scope import require_teaching_scope
            require_teaching_scope(teaching_context, **scope)
            if schema_decision_pin is not None or teaching_candidate_pin.artifact_kind != "corpus_graph_schema_candidate":
                raise ValueError("NORMATIVE_SCHEMA_INPUT_KIND_MISMATCH")
            if (teaching_context.user_id != capability.user_id
                or Path(teaching_context.workspace_root).resolve() != intent_store.workspace_path.resolve()):
                raise PermissionError("NORMATIVE_SCHEMA_ACTOR_MISMATCH")
        elif schema_decision_pin is None or schema_decision_pin.artifact_kind != "schema_decision":
            raise ValueError("NORMATIVE_SCHEMA_INPUT_KIND_MISMATCH")
        if competency_pin.artifact_kind != "competency_contract":
            raise ValueError("NORMATIVE_SCHEMA_INPUT_KIND_MISMATCH")
        input_pin = teaching_candidate_pin if teaching else schema_decision_pin
        if (authority_pin is None) != (coverage_pin is None):
            raise ValueError("NORMATIVE_LEGAL_SOURCE_AUTHORITY_COVERAGE_PAIR_REQUIRED")
        source_pins = (dict(authority=authority_pin.model_dump(mode="json"),
            coverage=coverage_pin.model_dump(mode="json")) if authority_pin is not None else {})
        request_hash = "sha256:" + content_hash(dict(**source_pins,
            competency=competency_pin.model_dump(mode="json"),
            decision=input_pin.model_dump(mode="json"),
            instructions=self.instructions, output_schema=self.output_schema,
            compiler_hash=self.compiler_hash, provider=self.provider, model=self.model))
        operation = dict(**scope, actor_ref="user:" + capability.user_id,
            operation="PRODUCE_NORMATIVE_SCHEMA", request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="normative_schema_contract")
        if prior is not None:
            return self.read_persisted(store=store, **scope, schema_pin=prior)

        generation_operation = {**operation, "operation": "PROPOSE_NORMATIVE_SCHEMA"}
        prior_generation = store.get_operation_result(
            **generation_operation, artifact_kind="normative_schema_proposal_evidence")
        if prior_generation is not None:
            return self._publish_schema(store, scope, operation, prior_generation)

        competency = store.get(**scope, artifact_ref=competency_pin.artifact_ref, contract_type=CompetencyContractV1)
        if competency is None or competency.competency_contract_hash != competency_pin.artifact_hash:
            raise ValueError("NORMATIVE_SCHEMA_COMPETENCY_PIN_MISMATCH")
        from services.competency_contract_service import CompetencyContractService
        CompetencyContractService.verify_applicability_persisted(store=store, contract=competency)
        if not teaching and any(question.criticality == "proposed_critical" or (
                   question.criticality != "excluded" and question.review_state != "confirmed")
               for question in competency.questions):
            raise ValueError("NORMATIVE_SCHEMA_COMPETENCY_REVIEW_REQUIRED")
        boundary = store.get(**scope, artifact_ref=competency.boundary_ref, contract_type=CanonicalEvidenceBoundaryV1)
        if boundary is None or boundary.boundary_hash != competency.boundary_hash:
            raise ValueError("NORMATIVE_SCHEMA_BOUNDARY_PIN_MISMATCH")
        if teaching:
            candidate = capability.discovery.candidate_store.get_by_hash(input_pin.artifact_hash)
            if candidate["candidate_id"] != input_pin.artifact_ref or candidate["schema_version"] != "corpus_graph_schema_candidate.v3":
                raise ValueError("NORMATIVE_SCHEMA_CANDIDATE_PIN_MISMATCH")
            intent_binding = candidate["graph_intent_binding"]
            source_binding = candidate["source_binding"]
            schema_view = {kind: candidate["proposed_" + kind] for kind in
                           ("node_types", "relation_types", "frame_types", "event_types")}
            decision_fields = dict(teaching_candidate=True)
        else:
            decision = capability.discovery.get_schema_decision_contract(schema_decision_pin.artifact_ref)
            if decision.schema_version != "schema_decision.v3" or decision.schema_decision_hash != schema_decision_pin.artifact_hash:
                raise ValueError("NORMATIVE_SCHEMA_DECISION_PIN_MISMATCH")
            candidate = capability.discovery.candidate_store.get_by_hash(decision.candidate_binding["candidate_hash"])
            if candidate["candidate_id"] != decision.candidate_binding["candidate_id"]:
                raise ValueError("NORMATIVE_SCHEMA_CANDIDATE_PIN_MISMATCH")
            intent_binding, source_binding, schema_view = decision.graph_intent_binding, decision.source_binding, decision.approved_schema
            decision_fields = dict(schema_decision_ref=decision.schema_decision_id,
                schema_decision_hash=decision.schema_decision_hash,
                schema_decision_review=dict(candidate_binding=decision.candidate_binding, decided_at=decision.decided_at))
        brief = intent_store.get(competency.graph_intent_brief_ref,
                                 graph_intent_hash=competency.graph_intent_brief_hash)
        if not brief.compiler_eligible or intent_binding["graph_intent_ref"] != brief.graph_intent_id or intent_binding["graph_intent_hash"] != brief.graph_intent_hash:
            raise ValueError("NORMATIVE_SCHEMA_INTENT_MISMATCH")
        for key in ("source_identity_ref", "source_hash", "canonical_source_ir_ref", "canonical_source_ir_hash"):
            if not (getattr(boundary, key) == getattr(brief.source_binding, key)
                    == source_binding.get(key) == candidate["source_binding"].get(key)):
                raise ValueError("NORMATIVE_SCHEMA_SOURCE_PIN_MISMATCH:" + key)
        profile = capability.profile_registry.resolve(candidate["compiler_provenance"]["profile_id"])
        if profile.content_hash != candidate["compiler_provenance"]["profile_hash"]:
            raise ValueError("NORMATIVE_SCHEMA_PROFILE_PIN_MISMATCH")
        document = GraphIntentDiscoveryService(
            capability.workspace_root, intent_store=intent_store,
        )._load_bound_canonical_source_ir(brief)
        _, identity = intent_store.source_registry.resolve_source_identity_lineage(
            **scope, source_identity_ref=boundary.source_identity_ref)
        evidence = tuple(evidence_ref_for_block(document, block, source_identity=identity,
            created_at=boundary.created_at) for block in document.blocks)
        objects = sorted({str(item[key])
            for collection in ("node_types", "relation_types", "frame_types", "event_types")
            for item in schema_view.get(collection, [])
            for key in ("type_id", "relation_type_id") if item.get(key)})
        if not objects or not evidence:
            raise ValueError("NORMATIVE_SCHEMA_SOURCE_OR_SCHEMA_EMPTY")
        request = dict(**scope, boundary=boundary.model_dump(mode="json"),
            competency=competency.model_dump(mode="json"), graph_intent=brief.as_dict(),
            canonical_source=document.model_dump(mode="json", by_alias=True),
            source_evidence=[item.model_dump(mode="json", by_alias=True) for item in evidence],
            source_citation_keys={f"source-evidence:{index:04d}": item.evidence_ref
                for index, item in enumerate(evidence, start=1)},
            **decision_fields, **{("teaching_schema" if teaching else "approved_schema"): schema_view},
            schema_object_refs=objects,
            corpus_schema_candidate_ref=candidate["candidate_id"], corpus_schema_candidate_hash=candidate["candidate_hash"],
            profile_ref=profile.profile_id, profile_hash=profile.content_hash,
            required_dimensions=list(NormativeSchemaContractV1.REQUIRED_DIMENSIONS))
        if authority_pin is not None:
            from services.normative_legal_source_context import retain_legal_source_context
            request["legal_source_context"] = retain_legal_source_context(store=store, scope=scope,
                boundary=(boundary.boundary_ref, boundary.boundary_hash), authority_pin=authority_pin,
                coverage_pin=coverage_pin, legal_store=legal_store)
            request["source_citation_keys"] = {f"source-evidence:{index:04d}": item["evidenceRef"]
                for index, item in enumerate(self.source_evidence(request), start=1)}
        request["source_reading_index"] = self.source_reading_index(request)
        input_text = canonical_json(request)
        # The exact decision pin already binds these types in the operation hash.
        # Specialize a copy so this producer can serve different approved schemas.
        output_schema = deepcopy(self.output_schema)
        for definition in ("NormativeSchemaSlotV1", "NormativeFactDefinitionV1"):
            output_schema["$defs"][definition]["properties"]["schema_object_refs"]["items"]["enum"] = objects
        from contracts.ai_grounding import StructuredGenerationError
        rejected_text = None
        usage = None
        try:
            generated = await self.llm.generate_structured(instructions=self.instructions,
                input_text=provider_input_text(input_text), json_schema=output_schema,
                schema_name="normative_schema_proposal_v1", purpose="normative_schema_proposal")
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
            raise ValueError("NORMATIVE_SCHEMA_PROVIDER_ATTRIBUTION_MISMATCH")
        basis = dict(schema_version="normative_schema_proposal_evidence.v1", **scope,
            instructions=self.instructions, input_text=input_text, provider_input_text=provider_input_text(input_text), output_schema=output_schema,
            output=output, provider=provider, model=model,
            compiler_ref=self.COMPILER_REF, compiler_hash=self.compiler_hash,
            created_at=created_at.isoformat().replace("+00:00", "Z"))
        if usage is not None:
            basis["usage"] = usage
        if rejected_text is not None:
            basis['rejected_response_text'] = rejected_text
        digest = content_hash(basis)
        proposal = NormativeSchemaProposalEvidenceV1(
            evidence_ref="normative-schema-proposal:" + digest, evidence_hash="sha256:" + digest, **basis)
        generation_pin = store.put_many_with_operation_result(**generation_operation, artifacts=(proposal,),
            result_pin=self._pin("normative_schema_proposal_evidence", proposal, "evidence_ref", "evidence_hash"))
        return self._publish_schema(store, scope, operation, generation_pin)

    def _publish_schema(self, store, scope, operation, generation_pin):
        proposal = store.get(**scope, artifact_ref=generation_pin.artifact_ref,
                             contract_type=NormativeSchemaProposalEvidenceV1)
        if proposal is None or proposal.evidence_hash != generation_pin.artifact_hash:
            raise ValueError("NORMATIVE_SCHEMA_PROPOSAL_PIN_MISMATCH")
        schema, report = self._compile(proposal)
        winner = store.put_many_with_operation_result(**operation, artifacts=(schema, report),
            result_pin=self._pin("normative_schema_contract", schema, "schema_ref", "schema_hash"))
        return self.read_persisted(store=store, **scope, schema_pin=winner)

    @staticmethod
    def source_evidence(source):
        from services.normative_legal_source_context import legal_source_evidence
        return source["source_evidence"] + legal_source_evidence(source)

    @staticmethod
    def source_reading_index(source):
        """Co-locate exact citation, region and original text; no semantic inference."""
        blocks = {block["blockId"]: block for block in source["canonical_source"]["blocks"]}
        blocks.update({"legal-section:" + row["section_id"]: {"content": row["text"]}
            for row in source.get("legal_source_context", {}).get("sections", [])})
        evidence = {item["evidenceRef"]: item for item in NormativeSchemaProductionService.source_evidence(source)}
        return [dict(citation_key=key,
            source_region_ref=evidence[ref]["provenance"]["source_block_id"],
            text=blocks[evidence[ref]["provenance"]["source_block_id"]]["content"])
            for key, ref in source["source_citation_keys"].items()]

    @staticmethod
    def resolve_source_citations(output, source):
        """Decode retained citation keys without changing raw producer evidence."""
        citation_keys = source.get("source_citation_keys", {})
        if "source_citation_keys" in source:
            expected_keys = {f"source-evidence:{index:04d}": item["evidenceRef"]
                for index, item in enumerate(NormativeSchemaProductionService.source_evidence(source), start=1)}
            if citation_keys != expected_keys:
                raise ValueError("normative schema citation key mapping mismatch")

        if "source_reading_index" in source and ("source_citation_keys" not in source or
            source["source_reading_index"] != NormativeSchemaProductionService.source_reading_index(source)):
            raise ValueError("normative source reading index mismatch")

        def resolve(value):
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: [citation_keys.get(ref, ref) if isinstance(ref, str) else ref
                    for ref in item] if key == "evidence_refs" and isinstance(item, list)
                    else resolve(item) for key, item in value.items()}
            return value

        return resolve(output)

    @staticmethod
    def _compile(proposal):
        verify_provider_input(proposal)
        if proposal.rejected_response_text is not None:
            from contracts.ai_grounding import StructuredGenerationError
            raise StructuredGenerationError('NORMATIVE_SCHEMA_STRUCTURED_RESPONSE_REJECTED',
                raw_response=dict(text=proposal.rejected_response_text, provider=proposal.provider, model=proposal.model))
        request = json.loads(proposal.input_text)
        competency = CompetencyContractV1.model_validate(request["competency"])
        boundary = CanonicalEvidenceBoundaryV1.model_validate(request["boundary"])
        output = NormativeSchemaProductionService.resolve_source_citations(proposal.output, request)
        parsed = _Proposal.model_validate(output)
        declarations = [node["observed_fact_definitions"]
            for node in request.get("approved_schema", request.get("teaching_schema", {})).get("node_types", [])
            if "observed_fact_definitions" in node]
        if declarations:
            if any(not isinstance(items, list) for items in declarations):
                raise ValueError("approved observation definitions must be arrays")
            # The reviewed schema owns explicitly declared observations. Model
            # output remains advisory and cannot silently redefine that contract.
            parsed = parsed.model_copy(update={"fact_definitions": tuple(
                NormativeFactDefinitionV1.model_validate(item)
                for items in declarations for item in items)})
        if "fact_definitions" in proposal.output_schema.get("properties", {}) and parsed.fact_definitions is None:
            raise ValueError("source schema proposal must declare fact definitions")
        evidence_refs = {item["evidenceRef"] for item in NormativeSchemaProductionService.source_evidence(request)}
        # Reviewed schema facts may bind directly to retained canonical source
        # blocks. Accept that representation only when the exact block is part
        # of this proposal's pinned canonical source.
        evidence_refs.update(
            "evidence:canonical-block:" + block_ref
            for block in request["canonical_source"]["blocks"]
            for block_ref in {
                block["blockId"],
                block["blockId"].removeprefix("sha256:"),
            }
        )
        objects = set(request["schema_object_refs"])
        for slot in parsed.slots:
            if slot.epistemic_status == "NONE_EVIDENCED" or slot.search_coverage_ref is not None:
                raise ValueError("normative schema proposal cannot invent audited absence")
            if not set(slot.schema_object_refs).issubset(objects):
                raise ValueError("normative schema object is outside the approved schema")
            if not set(slot.evidence_refs).issubset(evidence_refs):
                raise ValueError("normative schema evidence is outside the source")
        for fact in parsed.fact_definitions or ():
            if not set(fact.schema_object_refs).issubset(objects) or not set(fact.evidence_refs).issubset(evidence_refs):
                raise ValueError("normative fact definition is outside the approved schema/source")
        basis = dict(organization_id=proposal.organization_id, workspace_id=proposal.workspace_id,
            boundary_ref=boundary.boundary_ref, boundary_hash=boundary.boundary_hash,
            competency_contract_ref=competency.competency_contract_ref,
            competency_contract_hash=competency.competency_contract_hash,
            graph_intent_brief_ref=competency.graph_intent_brief_ref,
            graph_intent_brief_hash=competency.graph_intent_brief_hash,
            **{key: request[key] for key in ("schema_decision_ref", "schema_decision_hash", "teaching_candidate",
                "corpus_schema_candidate_ref", "corpus_schema_candidate_hash", "profile_ref", "profile_hash") if key in request},
            compiler_ref=proposal.compiler_ref, compiler_hash=proposal.compiler_hash,
            proposal_evidence_ref=proposal.evidence_ref, proposal_evidence_hash=proposal.evidence_hash,
            slots=[slot.model_dump(mode="json") for slot in sorted(parsed.slots, key=lambda item: item.dimension)])
        if parsed.fact_definitions is not None:
            basis["fact_definitions"] = [fact.model_dump(mode="json") for fact in sorted(
                parsed.fact_definitions, key=lambda item: item.fact_ref)]
        digest = content_hash(basis)
        schema = NormativeSchemaContractV1(schema_ref="normative-schema:" + digest,
            schema_hash="sha256:" + digest, created_at=proposal.created_at, **basis)
        report = OntologySchemaAdequacyService().evaluate(schema=schema, questions=competency.questions,
            report_ref="schema-adequacy:" + digest, created_at=proposal.created_at)
        return schema, report

    @classmethod
    def review_source_absence_persisted(cls, *, store, authority_resolver, ctx,
        schema_pin, dimension, reviewed_region_refs, rationale, request_id, reviewed_at):
        """Record reviewed absence over the entire pinned source, without waiving competencies."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.competency_contract_service import CompetencyContractService
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("source absence review requires production owners")
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("NORMATIVE_SCHEMA_WORKSPACE_OWNER_MISMATCH")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if not rationale.strip():
            raise ValueError("source absence review requires a rationale")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        operation = dict(**scope, actor_ref="user:" + ctx.user_id,
            operation="REVIEW_NORMATIVE_SOURCE_ABSENCE", request_id=request_id,
            request_hash="sha256:" + content_hash(dict(schema=schema_pin.model_dump(mode="json"),
                dimension=dimension, regions=list(reviewed_region_refs), rationale=rationale)))
        prior = store.get_operation_result(**operation, artifact_kind="normative_schema_contract")
        if prior is not None:
            return cls.read_persisted(store=store, **scope, schema_pin=prior)
        schema, _ = cls.read_persisted(store=store, **scope, schema_pin=schema_pin)
        if schema.source_review_ref is not None:
            raise ValueError("source absence review requires an original generated schema")
        proposal = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
            contract_type=NormativeSchemaProposalEvidenceV1)
        if schema.teaching_candidate:
            from services.teaching_candidate_scope import require_teaching_scope
            require_teaching_scope(ctx, **scope)
        decision = CompetencyContractService().create_decision(**scope,
            actor_type="system" if schema.teaching_candidate else "human",
            governed_actor_ref=operation["actor_ref"], channel="chat", decision_kind="define_boundary",
            target_ref=schema.schema_ref, rationale=rationale, evidence_refs=tuple(reviewed_region_refs),
            decided_at=reviewed_at)
        coverage = cls._absence_coverage(schema, proposal, decision, dimension)
        successor, report = cls._absence_successor(schema, proposal, decision, coverage)
        pin = store.put_many_with_operation_result(**operation,
            artifacts=(decision, coverage, successor, report),
            result_pin=cls._pin("normative_schema_contract", successor, "schema_ref", "schema_hash"),
            verify_before_commit=lambda: authority_resolver.require_current_actor(actor_context=actor))
        return cls.read_persisted(store=store, **scope, schema_pin=pin)

    @staticmethod
    def _absence_coverage(schema, proposal, decision, dimension):
        from services.competency_contract_service import CompetencyContractService
        request = json.loads(proposal.input_text)
        regions = tuple(block["blockId"] for block in request["canonical_source"]["blocks"])
        if not regions or decision.evidence_refs != regions:
            raise ValueError("source absence requires exact complete ordered source coverage")
        slot = next((slot for slot in schema.slots if slot.dimension == dimension), None)
        if slot is None or slot.epistemic_status != "UNRESOLVED":
            raise ValueError("only an unresolved dimension can receive an absence review")
        expected = CompetencyContractService().create_decision(
            organization_id=schema.organization_id, workspace_id=schema.workspace_id,
            actor_type="system" if schema.teaching_candidate else "human", governed_actor_ref=decision.governed_actor_ref,
            channel="chat", decision_kind="define_boundary", target_ref=schema.schema_ref,
            rationale=decision.rationale, evidence_refs=regions, decided_at=decision.decided_at)
        if (decision != expected or not decision.rationale.strip()
            or not decision.governed_actor_ref.startswith("user:")):
            raise ValueError("source absence review identity mismatch")
        basis = dict(organization_id=schema.organization_id, workspace_id=schema.workspace_id,
            source_region_refs=regions, excluded_region_refs=(),
            search_methods=("complete_canonical_source_semantic_review",), pattern_classes=(dimension,),
            parser_ref="retained_canonical_source_ir.v1",
            provider_ref="development_teaching_system_review" if schema.teaching_candidate else "governed_human_review",
            config_hash="sha256:" + content_hash(dict(schema_ref=schema.schema_ref,
                schema_hash=schema.schema_hash, proposal_hash=proposal.evidence_hash, dimension=dimension)),
            dependency_traversal_refs=(request["boundary"]["canonical_source_ir_ref"],),
            result_refs=(decision.decision_ref,), governed_reviewer_ref=decision.governed_actor_ref,
            reviewed_at=decision.decided_at.isoformat())
        digest = content_hash(basis)
        return SourceSearchCoverageManifestV1(coverage_ref="source-search-coverage:" + digest,
            coverage_hash="sha256:" + digest, **basis)

    @classmethod
    def _absence_successor(cls, schema, proposal, decision, coverage):
        if len(coverage.pattern_classes) != 1 or coverage != cls._absence_coverage(
            schema, proposal, decision, coverage.pattern_classes[0]):
            raise ValueError("source absence coverage mismatch")
        basis = schema.model_dump(mode="json", exclude={"schema_ref", "schema_hash", "created_at"})
        basis.update(source_review_ref=decision.decision_ref, source_review_hash=decision.decision_hash)
        for slot in basis["slots"]:
            if slot["dimension"] == coverage.pattern_classes[0]:
                slot.update(epistemic_status="NONE_EVIDENCED", schema_object_refs=[],
                    search_coverage_ref=coverage.coverage_ref, search_coverage_hash=coverage.coverage_hash)
        digest = content_hash(basis)
        successor = NormativeSchemaContractV1(schema_ref="normative-schema:" + digest,
            schema_hash="sha256:" + digest, created_at=decision.decided_at, **basis)
        competency = CompetencyContractV1.model_validate(json.loads(proposal.input_text)["competency"])
        report = OntologySchemaAdequacyService().evaluate(schema=successor, questions=competency.questions,
            report_ref="schema-adequacy:" + digest, created_at=decision.decided_at)
        return successor, report

    @classmethod
    def review_precedence_mapping_persisted(cls, *, store, authority_resolver, ctx,
        schema_pin, schema_object_ref, binding_record_hash, source_evidence_ref,
        package_hash, rationale, request_id, reviewed_at):
        """Publish an explicit human review of one retained local conditional record."""
        from datetime import timedelta
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.normative_schema import NormativeSchemaMappingReviewV1
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("mapping review requires production owners")
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_WORKSPACE_MISMATCH")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        def authorize():
            return authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN")
        authorize()
        if reviewed_at.utcoffset() != timedelta(0) or not rationale.strip():
            raise ValueError("mapping review requires UTC time and a rationale")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        schema, _ = cls.read_persisted(store=store, **scope, schema_pin=schema_pin)
        if schema.mapping_review_ref is not None or schema.source_review_ref is not None:
            raise ValueError("mapping review requires the original generated schema")
        if reviewed_at < schema.created_at:
            raise ValueError("mapping review cannot precede its predecessor")
        compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
        basis = dict(schema_version="normative_schema_mapping_review.v1", **scope,
            actor_ref="user:"+ctx.user_id, reviewer_kind="human",
            predecessor_schema_ref=schema.schema_ref, predecessor_schema_hash=schema.schema_hash,
            proposal_ref=schema.proposal_evidence_ref, proposal_hash=schema.proposal_evidence_hash,
            schema_object_ref=schema_object_ref, binding_record_hash=binding_record_hash,
            source_evidence_ref=source_evidence_ref, package_hash=package_hash, rationale=rationale,
            compiler_ref="compiler:source-normative-schema-mapping-review:v1", compiler_hash=compiler_hash,
            reviewed_at=reviewed_at.isoformat().replace("+00:00", "Z"))
        operation = dict(**scope, actor_ref=basis["actor_ref"], operation="REVIEW_NORMATIVE_PRECEDENCE_MAPPING",
            request_id=request_id, request_hash="sha256:"+content_hash(basis))
        winner = store.get_operation_result(**operation, artifact_kind="normative_schema_contract")
        if winner is None:
            digest = content_hash(basis)
            review = NormativeSchemaMappingReviewV1(review_ref="normative-schema-mapping-review:"+digest,
                review_hash="sha256:"+digest, **basis)
            proposal = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
                contract_type=NormativeSchemaProposalEvidenceV1)
            successor, report = cls._mapping_successor(schema, proposal, review)
            winner = store.put_many_with_operation_result(**operation, artifacts=(review, successor, report),
                result_pin=cls._pin("normative_schema_contract", successor, "schema_ref", "schema_hash"),
                verify_before_commit=authorize)
        return cls.read_persisted(store=store, **scope, schema_pin=winner)

    @classmethod
    def _mapping_successor(cls, schema, proposal, review):
        from contracts.normative_schema import NormativeSchemaMappingReviewV1
        # Revalidate model_copy and retained review identity, not only the supplied hash pin.
        review = NormativeSchemaMappingReviewV1.model_validate(review.model_dump(mode="json"))
        if ((review.organization_id, review.workspace_id, review.predecessor_schema_ref,
             review.predecessor_schema_hash, review.proposal_ref, review.proposal_hash) !=
            (schema.organization_id, schema.workspace_id, schema.schema_ref, schema.schema_hash,
             proposal.evidence_ref, proposal.evidence_hash)
            or not review.rationale.strip() or review.reviewed_at < schema.created_at
            or schema.mapping_review_ref is not None or schema.source_review_ref is not None):
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_REVIEW_PREDECESSOR_MISMATCH")
        slot = next(s for s in schema.slots if s.dimension == "precedence")
        if slot.epistemic_status != "UNRESOLVED":
            raise ValueError("only unresolved precedence can receive a mapping review")
        source = json.loads(proposal.input_text)
        frames = [f for f in source["approved_schema"].get("frame_types", [])
            if f["type_id"] == review.schema_object_ref]
        if len(frames) != 1 or review.schema_object_ref not in source["schema_object_refs"]:
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_TYPE_MISMATCH")
        frame = frames[0]
        records = [r for r in frame.get("slot_definitions", {}).get("precedence_bindings", {}).get("records", [])
            if "sha256:"+content_hash(r) == review.binding_record_hash]
        if len(records) != 1:
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_RECORD_MISMATCH")
        record = records[0]
        rules = {r["rule_ref"] for r in frame.get("rule_templates", [])}
        if (not {record.get("higher_rule_ref"), record.get("lower_rule_ref")}.issubset(rules)
            or record.get("higher_rule_ref") == record.get("lower_rule_ref")
            or not isinstance(record.get("scope"), str) or not record["scope"].strip()
            or not record.get("conditions")):
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_RELATION_MISMATCH")
        evidence = [e for e in cls.source_evidence(source) if e["evidenceRef"] == review.source_evidence_ref]
        if len(evidence) != 1:
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_SOURCE_MISMATCH")
        region = evidence[0]["provenance"]["source_block_id"]
        blocks = {b["blockId"]:b for b in source["canonical_source"]["blocks"]}
        canonical_ref = "evidence:canonical-block:"+region.removeprefix("sha256:")
        if region not in blocks or canonical_ref not in record.get("evidence_refs", []):
            raise ValueError("NORMATIVE_SCHEMA_MAPPING_SOURCE_RECORD_MISMATCH")
        basis = schema.model_dump(mode="json", exclude={"schema_ref", "schema_hash", "created_at"})
        basis.update(mapping_review_ref=review.review_ref, mapping_review_hash=review.review_hash)
        for value in basis["slots"]:
            if value["dimension"] == "precedence":
                value.update(epistemic_status="FOUND", schema_object_refs=[review.schema_object_ref],
                    evidence_refs=[review.source_evidence_ref], search_coverage_ref=None, search_coverage_hash=None)
        digest = content_hash(basis)
        successor = NormativeSchemaContractV1(schema_ref="normative-schema:"+digest,
            schema_hash="sha256:"+digest, created_at=review.reviewed_at, **basis)
        competency = CompetencyContractV1.model_validate(source["competency"])
        report = OntologySchemaAdequacyService().evaluate(schema=successor, questions=competency.questions,
            report_ref="schema-adequacy:"+digest, created_at=review.reviewed_at)
        return successor, report

    @classmethod
    def read_persisted(cls, *, store, organization_id, workspace_id, schema_pin):
        if schema_pin.artifact_kind != "normative_schema_contract":
            raise ValueError("NORMATIVE_SCHEMA_INPUT_KIND_MISMATCH")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        schema = store.get(**scope, artifact_ref=schema_pin.artifact_ref, contract_type=NormativeSchemaContractV1)
        if schema is None or schema.schema_hash != schema_pin.artifact_hash or not schema.proposal_evidence_ref:
            raise ValueError("NORMATIVE_SCHEMA_PRODUCTION_RECEIPT_MISMATCH")
        proposal = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
                             contract_type=NormativeSchemaProposalEvidenceV1)
        if proposal is None or proposal.evidence_hash != schema.proposal_evidence_hash:
            raise ValueError("NORMATIVE_SCHEMA_PROPOSAL_PIN_MISMATCH")
        from services.competency_contract_service import CompetencyContractService
        competency = CompetencyContractV1.model_validate(json.loads(proposal.input_text)["competency"])
        CompetencyContractService.verify_applicability_persisted(store=store, contract=competency)
        expected, report = cls._compile(proposal)
        if schema.source_review_ref is not None:
            decision = store.get(**scope, artifact_ref=schema.source_review_ref, contract_type=ConversationDecisionV1)
            if decision is None or decision.decision_hash != schema.source_review_hash:
                raise ValueError("NORMATIVE_SCHEMA_SOURCE_REVIEW_PIN_MISMATCH")
            predecessor = store.get(**scope, artifact_ref=decision.target_ref, contract_type=NormativeSchemaContractV1)
            if predecessor != expected:
                raise ValueError("NORMATIVE_SCHEMA_REVIEW_PREDECESSOR_MISMATCH")
            slots = [slot for slot in schema.slots if slot.epistemic_status == "NONE_EVIDENCED"]
            if len(slots) != 1:
                raise ValueError("NORMATIVE_SCHEMA_SOURCE_REVIEW_SLOT_MISMATCH")
            coverage = store.get(**scope, artifact_ref=slots[0].search_coverage_ref,
                contract_type=SourceSearchCoverageManifestV1)
            if coverage is None or coverage.coverage_hash != slots[0].search_coverage_hash:
                raise ValueError("NORMATIVE_SCHEMA_SOURCE_COVERAGE_PIN_MISMATCH")
            expected, report = cls._absence_successor(expected, proposal, decision, coverage)
        if schema.mapping_review_ref is not None:
            from contracts.normative_schema import NormativeSchemaMappingReviewV1
            review = store.get(**scope, artifact_ref=schema.mapping_review_ref, contract_type=NormativeSchemaMappingReviewV1)
            if review is None or review.review_hash != schema.mapping_review_hash:
                raise ValueError("NORMATIVE_SCHEMA_MAPPING_REVIEW_PIN_MISMATCH")
            predecessor = store.get(**scope, artifact_ref=review.predecessor_schema_ref, contract_type=NormativeSchemaContractV1)
            if predecessor != expected:
                raise ValueError("NORMATIVE_SCHEMA_MAPPING_REVIEW_PREDECESSOR_MISMATCH")
            expected, report = cls._mapping_successor(expected, proposal, review)
        retained_report = store.get(**scope, artifact_ref=report.report_ref, contract_type=SchemaAdequacyReportV1)
        if expected != schema or retained_report != report:
            raise ValueError("NORMATIVE_SCHEMA_PRODUCTION_REPLAY_MISMATCH")
        return schema, report
