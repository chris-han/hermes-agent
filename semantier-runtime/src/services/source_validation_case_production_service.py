"""Independent advisory case production from replay-verified source/schema inputs."""
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path

from contracts.normative_schema import NormativeSchemaContractV1, NormativeSchemaProposalEvidenceV1
from contracts.ontology_building import ArtifactPinV1, CompetencyContractV1
from contracts.validation_contract import SourceValidationCasesV1, SourceValidationProposalEvidenceV1
from eos.artifact_hashing import canonical_json, content_hash
from services.normative_schema_production_service import NormativeSchemaProductionService
from services.normative_legal_source_context import LEGAL_SOURCE_CONTEXT_HASH
from services.normative_source_prompt_view import provider_input_text, verify_provider_input, PROMPT_VIEW_HASH
from storage.ontology_build_store import OntologyBuildStore


class SourceValidationCaseProductionService:
    COMPILER_REF = "compiler:source-validation-cases:v3"

    def __init__(self, *, llm, provider: str, model: str):
        self.llm, self.provider, self.model = llm, provider, model
        self.instructions = (Path(__file__).resolve().parents[1] /
            "prompts/source_validation_case_production_v4.txt").read_text(encoding="utf-8")
        self.compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes() + PROMPT_VIEW_HASH.encode('ascii') + LEGAL_SOURCE_CONTEXT_HASH.encode('ascii')).hexdigest()
        self.output_schema = SourceValidationCasesV1.model_json_schema()

    @staticmethod
    def _request(source, schema, compiler_ref):
        if compiler_ref == "compiler:source-validation-cases:v1":
            vocabulary = schema.model_dump(mode="json")
        elif compiler_ref in {"compiler:source-validation-cases:v2", "compiler:source-validation-cases:v3"}:
            # Share observable inputs, not implementation-adequacy conclusions.
            vocabulary = schema.model_dump(mode="json", include={"schema_ref", "schema_hash",
                "competency_contract_ref", "competency_contract_hash",
                "graph_intent_brief_ref", "graph_intent_brief_hash", "fact_definitions"})
        else:
            raise ValueError("SOURCE_VALIDATION_COMPILER_UNSUPPORTED")
        return dict(source_context=source, normative_schema=vocabulary)

    @staticmethod
    def _pin(receipt):
        return ArtifactPinV1(artifact_kind="source_validation_proposal_evidence",
            artifact_ref=receipt.evidence_ref, artifact_hash=receipt.evidence_hash)

    @classmethod
    def propose_reviewed_persisted(cls, *, ctx, authority_resolver, store, schema_pin,
        predecessor_proposal_pin, output, rationale, request_id, created_at):
        """Retain an explicit delegated-AI source proposal; acceptance is separate."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import ontology_build_store_path
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError('source review requires production owners')
        if not ctx.authenticated or store._path.resolve()!=ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError('SOURCE_VALIDATION_WORKSPACE_OWNER_MISMATCH')
        actor=RuntimeContextDTO(organization_id=ctx.organization_id,workspace_id=ctx.workspace_id,actor_id=ctx.user_id)
        authorize=lambda: authority_resolver.require_current_actor(actor_context=actor,authorization_class='TENANT_ADMIN')
        authorize()
        if created_at.utcoffset()!=timedelta(0) or not rationale.strip():
            raise ValueError('source review requires UTC and a rationale')
        scope=dict(organization_id=ctx.organization_id,workspace_id=ctx.workspace_id)
        schema,_=NormativeSchemaProductionService.read_persisted(store=store,**scope,schema_pin=schema_pin)
        source_receipt=store.get(**scope,artifact_ref=schema.proposal_evidence_ref,contract_type=NormativeSchemaProposalEvidenceV1)
        source=json.loads(source_receipt.input_text)
        cls._review_predecessor(store,scope,predecessor_proposal_pin,source)
        instructions=(Path(__file__).resolve().parents[1]/'prompts/source_validation_case_production_v4.txt').read_text()
        compiler_hash='sha256:'+sha256(Path(__file__).read_bytes()+PROMPT_VIEW_HASH.encode('ascii') + LEGAL_SOURCE_CONTEXT_HASH.encode('ascii')).hexdigest()
        basis=dict(schema_version='source_validation_proposal_evidence.v1',**scope,
            input_text=canonical_json(cls._request(source,schema,cls.COMPILER_REF)),
            instructions=instructions,output_schema=SourceValidationCasesV1.model_json_schema(),output=output,
            provider='governed-source-case-review:delegated_ai',model='not_applicable',
            compiler_ref=cls.COMPILER_REF,compiler_hash=compiler_hash,
            predecessor_proposal_pin=predecessor_proposal_pin.model_dump(mode='json'),
            review_actor_ref='user:'+ctx.user_id,review_rationale=rationale)
        operation=dict(**scope,actor_ref='user:'+ctx.user_id,operation='PROPOSE_REVIEWED_SOURCE_VALIDATION_CASES',
            request_id=request_id,request_hash='sha256:'+content_hash(basis))
        pin=store.get_operation_result(**operation,artifact_kind='source_validation_proposal_evidence')
        if pin is None:
            basis['created_at']=created_at.isoformat().replace('+00:00','Z')
            digest=content_hash(basis)
            receipt=SourceValidationProposalEvidenceV1(evidence_ref='source-validation-proposal:'+digest,
                evidence_hash='sha256:'+digest,**basis)
            cls._validate(receipt,schema,source)
            pin=store.put_many_with_operation_result(**operation,artifacts=(receipt,),result_pin=cls._pin(receipt),
                verify_before_commit=authorize)
        return cls.read_persisted(store=store,**scope,proposal_pin=pin)

    @staticmethod
    def _review_predecessor(store,scope,pin,source):
        if pin.artifact_kind!='source_validation_proposal_evidence':
            raise ValueError('SOURCE_VALIDATION_REVIEW_PREDECESSOR_KIND_MISMATCH')
        prior=store.get(**scope,artifact_ref=pin.artifact_ref,contract_type=SourceValidationProposalEvidenceV1)
        if prior is None or prior.evidence_hash!=pin.artifact_hash:
            raise ValueError('SOURCE_VALIDATION_REVIEW_PREDECESSOR_PIN_MISMATCH')
        old=json.loads(prior.input_text)['source_context']
        keys=('source_identity_ref','source_hash','canonical_source_ir_ref','canonical_source_ir_hash')
        if any(old['boundary'][key]!=source['boundary'][key] for key in keys):
            raise ValueError('SOURCE_VALIDATION_REVIEW_SOURCE_LINEAGE_MISMATCH')
        if old['competency'] != source['competency']:
            from services.competency_contract_service import CompetencyContractService
            successor=CompetencyContractV1.model_validate(source['competency'])
            if (successor.applicability_review_pin is None or
                successor.predecessor_hash != old['competency']['competency_contract_hash'] or
                successor.predecessor_ref != old['competency']['competency_contract_ref']):
                raise ValueError('SOURCE_VALIDATION_REVIEW_SOURCE_LINEAGE_MISMATCH')
            CompetencyContractService.verify_applicability_persisted(store=store, contract=successor)

    async def produce_persisted(self, *, store, organization_id, workspace_id,
                                actor_ref, schema_pin, request_id, created_at):
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("validation case production requires the production artifact owner")
        if created_at.utcoffset() != timedelta(0):
            raise ValueError("validation case production requires UTC time")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        request_hash = "sha256:" + content_hash(dict(schema=schema_pin.model_dump(mode="json"),
            instructions=self.instructions, output_schema=self.output_schema,
            compiler_hash=self.compiler_hash, provider=self.provider, model=self.model))
        operation = dict(**scope, actor_ref=actor_ref, operation="PROPOSE_SOURCE_VALIDATION_CASES",
            request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="source_validation_proposal_evidence")
        if prior is not None:
            return self.read_persisted(store=store, **scope, proposal_pin=prior)
        schema, _ = NormativeSchemaProductionService.read_persisted(store=store, **scope, schema_pin=schema_pin)
        if schema.fact_definitions is None:
            raise ValueError("SOURCE_VALIDATION_FACT_DEFINITIONS_REQUIRED")
        source_receipt = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
            contract_type=NormativeSchemaProposalEvidenceV1)
        input_text = canonical_json(self._request(json.loads(source_receipt.input_text), schema, self.COMPILER_REF))
        from contracts.ai_grounding import StructuredGenerationError
        rejected_text = None
        try:
            generated = await self.llm.generate_structured(instructions=self.instructions,
                input_text=provider_input_text(input_text), json_schema=self.output_schema,
                schema_name="source_validation_cases_v1", purpose="source_validation_cases")
            provider, model, output = generated.provider, generated.model, generated.parsed
        except StructuredGenerationError as exc:
            raw = exc.raw_response
            if raw is None:
                raise
            provider, model, rejected_text = raw['provider'], raw['model'], raw['text']
            output = {}
        if (provider, model) != (self.provider, self.model):
            raise ValueError("SOURCE_VALIDATION_PROVIDER_ATTRIBUTION_MISMATCH")
        basis = dict(schema_version="source_validation_proposal_evidence.v1", **scope,
            instructions=self.instructions, input_text=input_text, provider_input_text=provider_input_text(input_text), output_schema=self.output_schema,
            output=output, provider=provider, model=model,
            compiler_ref=self.COMPILER_REF, compiler_hash=self.compiler_hash,
            created_at=created_at.isoformat().replace("+00:00", "Z"))
        if rejected_text is not None:
            basis['rejected_response_text'] = rejected_text
        digest = content_hash(basis)
        receipt = SourceValidationProposalEvidenceV1(evidence_ref="source-validation-proposal:" + digest,
            evidence_hash="sha256:" + digest, **basis)
        winner = store.put_many_with_operation_result(**operation, artifacts=(receipt,),
            result_pin=self._pin(receipt))
        return self.read_persisted(store=store, **scope, proposal_pin=winner)

    @staticmethod
    def _validate(receipt, schema, source):
        verify_provider_input(receipt)
        if receipt.rejected_response_text is not None:
            from contracts.ai_grounding import StructuredGenerationError
            raise StructuredGenerationError('SOURCE_VALIDATION_STRUCTURED_RESPONSE_REJECTED',
                raw_response=dict(text=receipt.rejected_response_text, provider=receipt.provider, model=receipt.model))
        if schema.fact_definitions is None:
            raise ValueError("SOURCE_VALIDATION_FACT_DEFINITIONS_REQUIRED")
        proposal = SourceValidationCasesV1.model_validate(
            NormativeSchemaProductionService.resolve_source_citations(receipt.output, source))
        competency = CompetencyContractV1.model_validate(source["competency"])
        questions = {question.competency_id: question for question in competency.questions
            if question.review_state == "confirmed" and question.criticality in {"critical", "noncritical"}}
        unresolved = set(proposal.unresolved_competency_ids)
        if len(unresolved) != len(proposal.unresolved_competency_ids) or not unresolved.issubset(questions):
            raise ValueError("SOURCE_VALIDATION_UNRESOLVED_COMPETENCY_MISMATCH")
        facts = {fact.fact_ref: fact for fact in schema.fact_definitions}
        evidence = {item["evidenceRef"] for item in NormativeSchemaProductionService.source_evidence(source)}
        ids = set()
        for case in proposal.cases:
            if case.case_id in ids:
                raise ValueError("SOURCE_VALIDATION_DUPLICATE_CASE")
            ids.add(case.case_id)
            if receipt.compiler_ref == "compiler:source-validation-cases:v3" and (
                (case.case_kind == "positive" and case.expected_result != "MATCH")
                or (case.case_kind == "hard_negative" and case.expected_result != "NO_MATCH")):
                raise ValueError("SOURCE_VALIDATION_UNKNOWN_IS_NOT_CONTRAST")
            if case.competency_id not in questions:
                raise ValueError("SOURCE_VALIDATION_COMPETENCY_MISMATCH")
            if not set(case.required_semantic_dimensions).issubset(NormativeSchemaContractV1.REQUIRED_DIMENSIONS):
                raise ValueError("SOURCE_VALIDATION_DIMENSION_MISMATCH")
            question = questions[case.competency_id]
            if question.dimension_applicability is not None and not set(case.required_semantic_dimensions).issubset(
                question.required_semantic_dimensions):
                raise ValueError("SOURCE_VALIDATION_DIMENSION_NOT_REQUIRED")
            if not set(case.evidence_refs).issubset(evidence):
                raise ValueError("SOURCE_VALIDATION_EVIDENCE_MISMATCH")
            for ref, value in case.observed_facts.items():
                if ref not in facts or not facts[ref].accepts_value(value):
                    raise ValueError("SOURCE_VALIDATION_FACT_BINDING_MISMATCH")
        for ref, question in questions.items():
            if question.criticality != "critical" or ref in unresolved:
                continue
            cases = [case for case in proposal.cases if case.competency_id == ref]
            if question.dimension_applicability is not None:
                SourceValidationCaseProductionService.validate_applicability_coverage(question, cases)
            dimensions = {dimension for case in cases for dimension in case.required_semantic_dimensions}
            if (not {"positive", "hard_negative"}.issubset({case.case_kind for case in cases}) or
                not set(question.required_semantic_dimensions).issubset(dimensions)):
                raise ValueError("SOURCE_VALIDATION_CRITICAL_COVERAGE_UNRESOLVED")
        return proposal

    @staticmethod
    def validate_applicability_coverage(question, cases):
        if not question.applicability_resolved:
            raise ValueError("SOURCE_VALIDATION_UNRESOLVED_APPLICABILITY")
        required = set(question.required_semantic_dimensions)
        for kind, result in (("positive", "MATCH"), ("hard_negative", "NO_MATCH")):
            covered = {d for case in cases if case.case_kind == kind and case.expected_result == result
                for d in case.required_semantic_dimensions}
            if not required.issubset(covered):
                raise ValueError("SOURCE_VALIDATION_REQUIRED_DIMENSION_CONTRAST_MISSING")

    @classmethod
    def read_persisted(cls, *, store, organization_id, workspace_id, proposal_pin):
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if proposal_pin.artifact_kind != "source_validation_proposal_evidence":
            raise ValueError("SOURCE_VALIDATION_INPUT_KIND_MISMATCH")
        receipt = store.get(**scope, artifact_ref=proposal_pin.artifact_ref,
            contract_type=SourceValidationProposalEvidenceV1)
        if receipt is None or receipt.evidence_hash != proposal_pin.artifact_hash:
            raise ValueError("SOURCE_VALIDATION_PROPOSAL_PIN_MISMATCH")
        request = json.loads(receipt.input_text)
        if receipt.compiler_ref not in {"compiler:source-validation-cases:v1", "compiler:source-validation-cases:v2", "compiler:source-validation-cases:v3"}:
            raise ValueError("SOURCE_VALIDATION_COMPILER_UNSUPPORTED")
        retained_schema = request["normative_schema"]
        schema, _ = NormativeSchemaProductionService.read_persisted(store=store, **scope,
            schema_pin=ArtifactPinV1(artifact_kind="normative_schema_contract",
                artifact_ref=retained_schema["schema_ref"], artifact_hash=retained_schema["schema_hash"]))
        source_receipt = store.get(**scope, artifact_ref=schema.proposal_evidence_ref,
            contract_type=NormativeSchemaProposalEvidenceV1)
        source = json.loads(source_receipt.input_text)
        if receipt.predecessor_proposal_pin is not None:
            cls._review_predecessor(store,scope,receipt.predecessor_proposal_pin,source)
        if request != cls._request(source, schema, receipt.compiler_ref):
            raise ValueError("SOURCE_VALIDATION_REPLAY_INPUT_MISMATCH")
        return receipt, cls._validate(receipt, schema, source)
