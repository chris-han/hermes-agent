"""Owner-only internal advisory producer on the existing immutable kernel store."""
from contracts.tender_owner_adoption import (
    POLICY_REF, POLICY_HASH, TenderOwnerAdoptionProposalV1, TenderOwnerAdoptionV1,
    TenderOwnerRevocationV1, validate_frozen, validate_revocation, ontology_contract_hash,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import ContextGraph, Ontology, DecisionRecord, ReasoningTrace
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.normative_schema import NormativeSchemaProposalEvidenceV1
from contracts.knowledge_classification import RuntimeContextDTO
from eos.artifact_hashing import content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.normative_semantic_production_service import NormativeSemanticProductionService
from services.normative_schema_production_service import NormativeSchemaProductionService
from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler
from storage.ontology_build_store import OntologyBuildStore
from storage.graph_intent_brief_store import GraphIntentBriefStore
from contracts.evidence_location import EvidenceRef
import json


class TenderOwnerAdoptionService:
    def __init__(self, *, store, authority_resolver, intent_store):
        if (type(store) is not OntologyBuildStore or type(authority_resolver) is not RuntimeAuthorityResolver
            or type(intent_store) is not GraphIntentBriefStore):
            raise TypeError('owner adoption requires production owners')
        from storage.ontology_build_store import ontology_build_store_path
        if store._path.resolve() != ontology_build_store_path(intent_store.workspace_path).resolve():
            raise ValueError('OWNER_ADOPTION_WORKSPACE_STORE_MISMATCH')
        self.store, self.authority, self.intent_store = store, authority_resolver, intent_store

    def _scope(self, actor_context, *, owner=False):
        (self.authority.require_current_owner if owner else self.authority.require_current_actor)(actor_context=actor_context)
        if (actor_context.organization_id, actor_context.workspace_id) != (self.intent_store.organization_id, self.intent_store.workspace_id):
            raise ValueError('FORBIDDEN_SCOPE')
        return dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)

    def _get(self, scope, pin, model):
        value = self.store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=model)
        if value is None:
            raise ValueError('OWNER_ADOPTION_PIN_MISSING')
        actual = value.pin() if hasattr(value,'pin') else ArtifactPinV1(artifact_kind='compiled_rule_plan', artifact_ref=value.plan_ref, artifact_hash=value.plan_hash)
        if actual != pin:
            raise ValueError('OWNER_ADOPTION_PIN_MISMATCH')
        return value

    def propose(self, *, actor_context, candidate_pin, plan_pin, rule_ref, canonical_pattern_ref, context_pin, ontology_pin, _persist=True):
        scope = self._scope(actor_context)
        candidate = NormativeSemanticProductionService.read_persisted(store=self.store, **scope, candidate_pin=candidate_pin)
        plan = self._get(scope, plan_pin, CompiledRulePlanV1)
        expected = OntologyRulePlanCompiler(compiler_ref=plan.compiler_ref, compiler_hash=plan.compiler_hash).compile(
            candidate=candidate, validation_contract_ref=plan.validation_contract_ref, validation_contract_hash=plan.validation_contract_hash,
            competency_refs=plan.rules[0].competency_refs, compiled_at=plan.compiled_at, domain_basis=plan.domain_basis)
        if plan != expected or rule_ref not in {rule.rule_id for rule in plan.rules}:
            raise ValueError('OWNER_ADOPTION_PLAN_RULE_MISMATCH')
        context = self._get(scope,context_pin,ContextGraph)
        ontology = self._get(scope,ontology_pin,Ontology)
        if context.state != 'ADMITTED' or context.ontology != ontology_pin:
            raise ValueError('ADMITTED_CONTEXT_REQUIRED')
        admission=self._get(scope,context.admission_decision,DecisionRecord)
        predecessor=self._get(scope,context.predecessor,ContextGraph)
        if admission.purpose != 'CONTEXT_ADMISSION' or admission.subject != predecessor.pin() or admission.ontology != ontology_pin:
            raise ValueError('OWNER_ADOPTION_CONTEXT_ADMISSION_MISMATCH')
        if not any(n.node_type == 'CanonicalPattern' and n.node_id == 'canonical-pattern:' + canonical_pattern_ref and n.label == canonical_pattern_ref for n in context.nodes):
            raise ValueError('OWNER_ADOPTION_PATTERN_NOT_IN_CONTEXT')
        retrieval_pins={pin for invocation in admission.reasoning.invocations for pin in invocation.inputs
            if pin.artifact_kind=='runtime_semantic_retrieval_snapshot'}
        if len(retrieval_pins)!=1:
            raise ValueError('OWNER_ADOPTION_PATTERN_RETRIEVAL_REQUIRED')
        retrieval_pin=next(iter(retrieval_pins))
        from storage.runtime_semantic_hypothesis_artifact_store import RuntimeSemanticHypothesisArtifactStore
        snapshot=RuntimeSemanticHypothesisArtifactStore(self.intent_store.workspace_path,**scope).get_retrieval_snapshot(
            snapshot_ref=retrieval_pin.artifact_ref,snapshot_hash=retrieval_pin.artifact_hash)
        patterns=[m for m in snapshot.ordered_material if m.kind=='canonical_pattern' and m.ref==canonical_pattern_ref]
        if len(patterns)!=1:
            raise ValueError('OWNER_ADOPTION_PATTERN_RETRIEVAL_MISMATCH')
        identity, document = self.intent_store.resolve_evidence_source(source_identity_ref=candidate.source_identity_ref,
            canonical_source_ir_ref=candidate.canonical_source_ir_ref, canonical_source_ir_hash=candidate.canonical_source_ir_hash)
        if identity.source_hash != candidate.source_hash:
            raise ValueError('OWNER_ADOPTION_SOURCE_MISMATCH')
        self.intent_store.source_registry._verify_source_artifact(identity)
        schema_pin = ArtifactPinV1(artifact_kind='normative_schema_contract',artifact_ref=candidate.normative_schema_ref,artifact_hash=candidate.normative_schema_hash)
        schema,_ = NormativeSchemaProductionService.read_persisted(store=self.store, **scope,schema_pin=schema_pin)
        receipt = self.store.get(**scope,artifact_ref=schema.proposal_evidence_ref,contract_type=NormativeSchemaProposalEvidenceV1)
        source = json.loads(receipt.input_text)
        if source['canonical_source'] != document.model_dump(mode='json',by_alias=True):
            raise ValueError('OWNER_ADOPTION_SOURCE_IR_MISMATCH')
        evidence = tuple(EvidenceRef.model_validate(item) for item in NormativeSchemaProductionService.source_evidence(source))
        from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
        regenerated = {e.evidence_ref: e for e in (evidence_ref_for_block(document, block, source_identity=identity,created_at=evidence[0].created_at) for block in document.blocks)}
        if any(regenerated.get(e.evidence_ref) != e for e in evidence):
            raise ValueError('OWNER_ADOPTION_SOURCE_SELECTOR_MISMATCH')
        rule = next(rule for rule in plan.rules if rule.rule_id == rule_ref)
        if not set(rule.evidence_refs).issubset({item.evidence_ref for item in evidence}):
            raise ValueError('OWNER_ADOPTION_RULE_EVIDENCE_MISMATCH')
        proposal = TenderOwnerAdoptionProposalV1.create(**scope,source_identity=identity,
            canonical_source_ir=ArtifactPinV1(artifact_kind='canonical_source_ir',artifact_ref=candidate.canonical_source_ir_ref,artifact_hash=candidate.canonical_source_ir_hash),
            source_evidence=evidence,candidate=candidate_pin,schema=schema_pin,plan=plan_pin,rule_ref=rule_ref,
            canonical_pattern_ref=canonical_pattern_ref,canonical_pattern_hash=patterns[0].hash,
            pattern_retrieval_snapshot=retrieval_pin,ontology_contract_hash=ontology_contract_hash(ontology),context=context_pin,ontology=ontology_pin)
        if _persist:
            self.store.put(proposal)
        return proposal

    def _verify_proposal(self, actor_context, proposal):
        expected = self.propose(actor_context=actor_context,candidate_pin=proposal.candidate,plan_pin=proposal.plan,
            rule_ref=proposal.rule_ref,canonical_pattern_ref=proposal.canonical_pattern_ref,
            context_pin=proposal.context,ontology_pin=proposal.ontology,_persist=False)
        if expected != proposal:
            raise ValueError('OWNER_ADOPTION_PROPOSAL_CHANGED')

    def _decision(self, actor_context, proposal, operation, subject, recorded_at):
        authorization = self.authority.current_actor_authorization_basis(actor_context=actor_context,operation=operation,
            subject=subject,authorization_class='TENANT_ADMIN')
        return DecisionRecord.create(organization_id=proposal.organization_id,workspace_id=proposal.workspace_id,
            purpose='CONSEQUENTIAL_DECISION',subject=subject,context=proposal.context,ontology=proposal.ontology,
            result_code=operation,epistemic_status='KNOWN',evidence=(),
            reasoning=ReasoningTrace(strategy='tender_owner_internal_advisory',reasoning_version=POLICY_HASH,
                invocations=(),diagnostics=(proposal.model_dump_json(),)),actor_id=actor_context.actor_id,
            reasoner_identity=POLICY_REF,authorization=authorization,recorded_at=recorded_at)

    def adopt(self, *, actor_context, proposal_pin=None, proposal=None, request_id, recorded_at):
        scope = self._scope(actor_context,owner=True)
        proposal = self._get(scope,proposal_pin or proposal.pin(),TenderOwnerAdoptionProposalV1)
        self._verify_proposal(actor_context,proposal)
        return self.store._publish_tender_owner_adoption(service=self,actor_context=actor_context,
            proposal=proposal,adoption=None,expected_head_pin=None,request_id=request_id,recorded_at=recorded_at)

    def revoke(self, *, actor_context, adoption_pin, expected_head_pin, request_id, recorded_at):
        scope = self._scope(actor_context,owner=True)
        adoption = self.read(actor_context=actor_context,adoption_pin=adoption_pin)
        proposal = self._get(scope,adoption.proposal,TenderOwnerAdoptionProposalV1)
        return self.store._publish_tender_owner_adoption(service=self,actor_context=actor_context,
            proposal=proposal,adoption=adoption,expected_head_pin=expected_head_pin,request_id=request_id,recorded_at=recorded_at)

    def _history(self, scope, model):
        return tuple(self.store.get(**scope,artifact_ref=pin.artifact_ref,contract_type=model)
            for pin in self.store.list_artifact_pins(**scope,schema_version=model.model_fields['schema_version'].default))

    def head(self, *, actor_context, adoption):
        scope = self._scope(actor_context)
        revocations = tuple(v for v in self._history(scope,TenderOwnerRevocationV1) if v.adoption == adoption.pin())
        if len(revocations)>1:
            raise ValueError('OWNER_ADOPTION_AMBIGUOUS_HEAD')
        if revocations:
            proposal=self._get(scope,adoption.proposal,TenderOwnerAdoptionProposalV1)
            validate_revocation(adoption=adoption,proposal=proposal,revocation=revocations[0],
                decision=self._get(scope,revocations[0].decision,DecisionRecord))
        return revocations[0].pin() if revocations else adoption.pin()

    def read(self, *, actor_context, adoption_pin):
        scope = self._scope(actor_context)
        adoption = self._get(scope,adoption_pin,TenderOwnerAdoptionV1)
        proposal = self._get(scope,adoption.proposal,TenderOwnerAdoptionProposalV1)
        return validate_frozen(adoption=adoption,proposal=proposal,
            source_decision=self._get(scope,adoption.source_decision,DecisionRecord),
            interpretation_decision=self._get(scope,adoption.interpretation_decision,DecisionRecord))

    def validate_live(self, *, actor_context, adoption):
        scope = self._scope(actor_context)
        adoption = self.read(actor_context=actor_context,adoption_pin=adoption.pin())
        owner = RuntimeContextDTO(**scope,actor_id=adoption.actor_id)
        self.authority.require_current_owner(actor_context=owner)
        if self.head(actor_context=actor_context,adoption=adoption) != adoption.pin():
            raise ValueError('OWNER_ADOPTION_REVOKED')
        proposal = self._get(scope,adoption.proposal,TenderOwnerAdoptionProposalV1)
        self._verify_proposal(actor_context,proposal)
        return proposal

    def list(self, *, actor_context):
        scope = self._scope(actor_context)
        adoptions = self._history(scope,TenderOwnerAdoptionV1)
        entries=[]
        for proposal in self._history(scope,TenderOwnerAdoptionProposalV1):
            matches=[a for a in adoptions if a.proposal==proposal.pin()]
            if not matches:
                entries.append(dict(proposal=proposal,adoption=None,head=None,status='PROPOSED'))
            for adoption in matches:
                self.read(actor_context=actor_context,adoption_pin=adoption.pin())
                head=self.head(actor_context=actor_context,adoption=adoption)
                entries.append(dict(proposal=proposal,adoption=adoption,head=head,
                    status='ADOPTED' if head==adoption.pin() else 'REVOKED'))
        for entry in entries:
            proposal=entry['proposal']
            plan=self._get(scope,proposal.plan,CompiledRulePlanV1)
            compiled=next(r for r in plan.rules if r.rule_id==proposal.rule_ref)
            candidate=NormativeSemanticProductionService.read_persisted(store=self.store,**scope,candidate_pin=proposal.candidate)
            schema,_=NormativeSchemaProductionService.read_persisted(store=self.store,**scope,schema_pin=proposal.schema_)
            frame=next(f for f in candidate.frames if f.frame_id==proposal.rule_ref)
            definitions={f.fact_ref:f.definition for f in (schema.fact_definitions or ())}
            def describe(predicate):
                return dict(label=definitions.get(predicate.fact_ref,predicate.fact_ref),operator=predicate.operator,expected_value=predicate.expected_value)
            entry['interpretation']={**compiled.model_dump(mode='json'), 'display_name':frame.display_name,
                'condition_descriptions':[describe(p) for p in compiled.predicates],
                'exception_descriptions':[describe(p) for p in compiled.exception_predicates],
                'frames':[f.model_dump(mode='json') for f in candidate.frames]}
        return entries

    def select(self, *, actor_context, canonical_pattern_ref, ontology_pin, canonical_pattern_hash):
        scope=self._scope(actor_context)
        ontology=self._get(scope,ontology_pin,Ontology)
        compatibility=ontology_contract_hash(ontology)
        selected=[]
        for entry in self.list(actor_context=actor_context):
            proposal=entry['proposal']
            if entry['status']=='ADOPTED' and proposal.canonical_pattern_ref==canonical_pattern_ref and proposal.canonical_pattern_hash==canonical_pattern_hash and proposal.ontology_contract_hash==compatibility:
                try:
                    self.validate_live(actor_context=actor_context,adoption=entry['adoption'])
                except ValueError as exc:
                    if str(exc) in {'FORBIDDEN_ACTOR','CURRENT_ORGANIZATION_OWNER_REQUIRED','OWNER_ADOPTION_REVOKED'}:
                        continue
                    raise
                selected.append(entry)
        return tuple(selected)

    validate_frozen = staticmethod(validate_frozen)
