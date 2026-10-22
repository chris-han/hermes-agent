"""Authenticated DEVELOPMENT adapter over existing source-consumer quality owners."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4
from contracts.artifact_pin import ArtifactPinV1
from contracts.policy_revision import PolicyRevisionRecordV1 as Record
from contracts.source_consumer_verification import (
    ConsumerSourceRequirementV1, ConsumerTargetV1, ConsumerBindingV1,
    ConsumerCaseEvidenceV1, ConsumerObservationV1,
)
from contracts.ontology_quality import QualityProfileV1, QualityRequirementV1, QualityEvaluationRunV1
from contracts.ontology_workbench_context import UploadWorkspaceArtifactCommandV1
from services.source_consumer_verification import digest, raw_hash, observe, seal_verification, read_retained_consumers
from services.policy_revision_service import PolicyRevisionService
from services.teaching_candidate_scope import require_teaching_scope

SEED_HASH='sha256:4ebe4ba333d7bae95520e2a6223ec4b559026e0c536c17cf2c50f60847602876'


class SourceConsumerProductService:
    def __init__(self, service):
        self.service=service
        self.policy=PolicyRevisionService(ctx=service.ctx,store=service.business.policy_owner.store,policy_owner=service.business.policy_owner)
        self.ctx,self.store,self.scope=service.ctx,service.business.policy_owner.store,self.policy.scope
        require_teaching_scope(self.ctx,**self.scope)
        path=Path(self.ctx.hermes_home)/'demo-materials/expense-policy-source-consumer-seed.json'
        try: raw=path.read_bytes()
        except FileNotFoundError as exc: raise ValueError('CONSUMER_BASIS_UNAVAILABLE') from exc
        if raw_hash(raw)!=SEED_HASH: raise ValueError('CONSUMER_SEED_HASH_MISMATCH')
        self.seed=json.loads(raw)
        self.session=service.bound.session_id

    def record(self,kind,data):
        return Record.create(**self.scope,kind=kind,actor_ref='user:'+self.ctx.user_id,
            produced_at=datetime.now(timezone.utc),data=dict(session_ref=self.session,**data))

    def records(self,kind):
        return sorted((r for r in self.policy._list(kind) if r.data['session_ref']==self.session and r.data.get('lane') not in ('source_change','source_required_inventory','procurement_context','incident_witness','procurement_fee_form','component_formation','fee_document','fee_declaration','fee_form','fee_formation','fee_facts_run','development_measure_fixture','development_measure_input','development_measure_declaration','development_measure_formation','development_measure_run')),key=lambda r:r.produced_at)

    def candidate(self,ref,hash_):
        r=self.policy.read(ref,hash_)
        if r.kind!='consumer_candidate' or r.data['session_ref']!=self.session or r.data.get('lane') in ('source_change','source_required_inventory','procurement_context','incident_witness','procurement_fee_form','component_formation','fee_document','fee_declaration','fee_form','fee_formation','fee_facts_run','development_measure_fixture','development_measure_input','development_measure_declaration','development_measure_formation','development_measure_run'):
            raise PermissionError('CONSUMER_CANDIDATE_SCOPE_MISMATCH')
        return r

    def initialize(self):
        records=self.records('consumer_candidate')
        if records:return records[0]
        source=self.policy.initialize()
        r=self.record('consumer_candidate',dict(parent=None,label='Original bindings',source=self.policy.pin(source),
            binding_ids=[b['binding_id'] for b in self.seed['bindings']],seed_hash=SEED_HASH))
        return self.policy._operation('consumer-original:'+self.session,digest(dict(source=r.data['source'],seed_hash=SEED_HASH)),r)

    def successor(self,*,parent_ref,parent_hash,binding_ids):
        parent=self.candidate(parent_ref,parent_hash)
        if len(binding_ids)!=len(set(binding_ids)) or not set(binding_ids).issubset({b['binding_id'] for b in self.seed['bindings']}):
            raise ValueError('CONSUMER_BINDING_MEMBERSHIP_INVALID')
        r=self.record('consumer_candidate',dict(parent=self.policy.pin(parent),label='Binding successor',source=parent.data['source'],
            binding_ids=sorted(binding_ids),seed_hash=SEED_HASH))
        self.store.put_many((r,));return r

    def read_run(self,ref,hash_):
        record=self.policy.read(ref,hash_)
        if record.kind!='consumer_run' or record.data['session_ref']!=self.session or record.data.get('lane') in ('source_change','fee_facts_run','development_measure_fixture','development_measure_input','development_measure_declaration','development_measure_formation','development_measure_run'):
            raise PermissionError('CONSUMER_RUN_SCOPE_MISMATCH')
        candidate=self.candidate(**dict(ref=record.data['candidate']['artifact_ref'],hash_=record.data['candidate']['artifact_hash']))
        if candidate.data['source']!=record.data['source']:raise ValueError('CONSUMER_RUN_SOURCE_MISMATCH')
        run=read_retained_consumers(store=self.store,**self.scope,run_pin=ArtifactPinV1(**record.data['run']),
            expected_plan=ArtifactPinV1(**self.seed['plan']),source_hash=self.seed['source_hash'],session_ref=self.session)
        result=run.results[0];v=result.consumer_verification
        return dict(record.model_dump(mode='json'),assessment=dict(status=result.status,run=run.pin().model_dump(mode='json'),
            source_hash=v.source_hash,plan=v.plan.model_dump(mode='json'),
            targets=[a.model_dump(mode='json') for a in v.target_assessments],
            requirements=[a.model_dump(mode='json') for a in v.requirement_assessments],
            findings=[f.model_dump(mode='json') for f in result.findings],
            observations=[o.model_dump(mode='json') for o in v.observations],
            cases=[dict(case_id=c.case_id,request_hash=c.request_hash,response_hash=c.response_hash,
                result_ref=c.response.get('result_ref'),result_hash=c.response.get('result_hash'),http_status=c.http_status) for c in v.cases]))

    def context(self):
        original=self.initialize()
        source=self.policy.read(**dict(ref=original.data['source']['artifact_ref'],expected_hash=original.data['source']['artifact_hash']))
        return dict(session_ref=self.session,development_only=True,reference_status='NOT_QUALIFIED',
            source=self.policy.present(source),requirements=self.seed['requirements'],targets=self.seed['targets'],bindings=self.seed['bindings'],
            candidates=[c.model_dump(mode='json') for c in self.records('consumer_candidate')],
            runs=[self.read_run(r.artifact_ref,r.artifact_hash) for r in self.records('consumer_run')])

    async def execute(self,*,candidate_ref,candidate_hash,invoke):
        candidate=self.candidate(candidate_ref,candidate_hash)
        seed=deepcopy(self.seed)
        invoice=base64.b64decode(seed['assets']['invoice_pdf'])
        # Only the explicit selected DEVELOPMENT example imports these public bytes.
        collection=self.service.artifacts.list(ctx=self.ctx)
        request_id='consumer:'+uuid4().hex
        command=UploadWorkspaceArtifactCommandV1(schema_version='upload_workspace_artifact_command.v1',
            request_id=request_id,idempotency_key=request_id,ingress='DIRECT',
            expected_collection_revision=collection.collection_revision,original_name='source-consumer-example.pdf',
            declared_media_type='application/pdf',byte_length=len(invoice),byte_hash=raw_hash(invoice))
        receipt=self.service.artifacts.upload(ctx=self.ctx,command=command,byte_stream=BytesIO(invoice))
        pin=receipt.artifact_pin.model_dump(mode='json')
        upload=dict(command=command.model_dump(mode='json'),response=dict(artifact_pin=pin))
        encoded=json.dumps(upload).encode();seed['assets']['upload']=base64.b64encode(encoded).decode();seed['asset_hashes']['upload']=raw_hash(encoded)
        cases=[];results=[]
        requests=[('missing-invoice',dict(submitted_on='2026-10-15',applicant_fields={}))]
        requests += [('fields-'+str(mask),dict(artifact_pin=pin,submitted_on='2026-10-15',applicant_fields={f:'supplied' if mask&(1<<i) else '  ' for i,f in enumerate(('name','department','purpose'))})) for mask in range(8)]
        for case_id,request in requests:
            before=self.service.business.current() if case_id=='missing-invoice' else None
            status,response=await invoke(request)
            if case_id=='missing-invoice':
                if status!=400 or self.service.business.current()!=before:raise ValueError('INLET_REJECTION_CREATED_RESULT')
            elif status!=200:raise ValueError('CONSUMER_NATIVE_CASE_FAILED')
            deps={p['ref']:base64.b64encode(self.service.business._path(p['ref']).read_bytes()).decode() for p in response.get('pins',[])}
            cases.append(ConsumerCaseEvidenceV1(case_id=case_id,session_ref=self.session,request=request,request_hash=digest(request),
                http_status=status,response=response,response_hash=digest(response),dependencies=deps))
            if status==200:results.append(response)
        draft=self.service.business.save_draft(results[0]['result_ref'],results[0]['result_hash'])
        cases[1]=cases[1].model_copy(update={'draft':draft})
        request=dict(artifact_pin=pin,submitted_on='2026-10-15',applicant_fields={f:'supplied' for f in ('name','department','purpose')},
            predecessor_result={k:results[0][k] for k in ('result_ref','result_hash')})
        status,response=await invoke(request)
        if status!=200 or response['disposition']!=results[0]['disposition']:raise ValueError('CONSUMER_CORRECTION_FAILED')
        deps={p['ref']:base64.b64encode(self.service.business._path(p['ref']).read_bytes()).decode() for p in response['pins']}
        cases.append(ConsumerCaseEvidenceV1(case_id='corrected',session_ref=self.session,request=request,request_hash=digest(request),http_status=status,
            response=response,response_hash=digest(response),dependencies=deps))
        seed['requirements']=tuple(ConsumerSourceRequirementV1(**r) for r in seed['requirements'])
        seed['targets']=tuple(ConsumerTargetV1(**t) for t in seed['targets'])
        seed['bindings']=tuple(ConsumerBindingV1(**b) for b in seed['bindings'] if b['binding_id'] in candidate.data['binding_ids'])
        seed['plan']=ArtifactPinV1(**seed['plan'])
        seed['source_identity_ref']=candidate.data['source']['artifact_ref']
        obs=tuple(ConsumerObservationV1(binding_id=b.binding_id,case_id=c.case_id,evidence_hash=digest(c.model_dump(mode='json')),actual=observe(b,c))
            for b in seed['bindings'] for c in cases if c.case_id in {e.case_id for t in seed['targets'] if t.target_id==b.target_id for e in t.cases})
        v=seal_verification(**self.scope,session_ref=self.session,**seed,cases=tuple(cases),observations=obs)
        from services.ontology_quality_evaluators import ConsumerBindingQualityEvaluator
        profile=QualityProfileV1.create(**self.scope,name='source_consumer_development',requirements=(QualityRequirementV1(requirement_id='source_consumer_binding',gate='FITNESS'),))
        result=ConsumerBindingQualityEvaluator().evaluate(verification=v,profile=profile)
        run=QualityEvaluationRunV1.create(**self.scope,quality_profile=profile.pin(),profile_definition=profile,subject=v.plan,
            competency=ArtifactPinV1(artifact_kind='source_competency_inventory',artifact_ref='source:CQ02-CQ07',artifact_hash=digest([r.competency_ids for r in v.requirements])),
            validation=ArtifactPinV1(artifact_kind='source_consumer_declaration',artifact_ref='source:R01-R03',artifact_hash=digest(dict(requirements=[r.model_dump(mode='json') for r in v.requirements],targets=[t.model_dump(mode='json') for t in v.targets]))),
            results=(result,),executed_at=datetime.now(timezone.utc))
        record=self.record('consumer_run',dict(candidate=self.policy.pin(candidate),source=candidate.data['source'],run=run.pin().model_dump(mode='json')))
        self.store.put_many((run,record))
        return self.read_run(record.artifact_ref,record.artifact_hash)

    def freeze_required_inventory(self, *, source, requirements, dependencies, inputs,
                                  predecessor_pin=None, withdrawals=None):
        """Retain a separate source annotation before evaluating any candidate.

        Existing validation vocabulary is pinned as context, not a review of the
        new prose. This operation grants no independent review or policy authority.
        """
        from contracts.source_change_consumer_verification import SourceRequiredInventoryV1
        from services.source_consumer_verification import verify_required_declaration
        self.policy._access()
        if self.policy.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=self.session:
            raise PermissionError('REQUIRED_INVENTORY_SOURCE_ACCESS_MISMATCH')
        prepared=self.service.business.policy_owner.current()
        validation=ArtifactPinV1(artifact_kind='ontology_validation_contract',artifact_ref=prepared['plan'].validation_contract_ref,
            artifact_hash=prepared['plan'].validation_contract_hash)
        inventory=SourceRequiredInventoryV1(source=ArtifactPinV1(artifact_kind='policy_revision',**self.policy.pin(source)),
            source_hash=source.data['source_hash'],**self.scope,session_ref=self.session,effective_on=source.data['effective_on'],
            validation_basis=validation,requirements=requirements,dependencies=dependencies,inputs=inputs,
            predecessor_pin=predecessor_pin,withdrawals=withdrawals or {})
        verify_required_declaration(inventory,source)
        if predecessor_pin is not None:
            parent=self.policy.read(predecessor_pin.artifact_ref,predecessor_pin.artifact_hash)
            if predecessor_pin.artifact_kind!='policy_revision' or parent.kind!='consumer_candidate' or parent.data.get('lane')!='source_required_inventory' or parent.data.get('session_ref')!=self.session:
                raise ValueError('REQUIRED_INVENTORY_PARENT_MISMATCH')
            prior=SourceRequiredInventoryV1.model_validate(parent.data['inventory'])
            if source.data['parent']!={'artifact_ref':prior.source.artifact_ref,'artifact_hash':prior.source.artifact_hash}:
                raise ValueError('REQUIRED_INVENTORY_SOURCE_LINEAGE_MISMATCH')
        elif withdrawals:raise ValueError('REQUIRED_WITHDRAWAL_PARENT_MISSING')
        data=inventory.model_dump(mode='json');hashed=digest(data)
        for existing in self.policy._list('consumer_candidate'):
            if existing.data.get('lane')=='source_required_inventory' and existing.data.get('session_ref')==self.session and existing.data['inventory']['source']==data['source']:
                if existing.data['inventory_hash']!=hashed:raise ValueError('REQUIRED_INVENTORY_ALREADY_FROZEN')
                return existing
        record=self.record('consumer_candidate',dict(lane='source_required_inventory',inventory=data,inventory_hash=hashed))
        # Existing operation-result uniqueness also protects concurrent freezes;
        # a read/list check alone is not an atomic source-version boundary.
        return self.policy._operation('required-inventory:'+digest([self.session,source.artifact_ref]),hashed,record)

    def prepare_source_revision(self, *, source, requirements, dependencies, inputs, frames,
                                supports=(), dependency_inventory_complete=True, coverage_version=1, required_inventory_pin=None,
                                candidate_template=None, predecessor_candidate=None):
        """Compile explicitly authored typed semantics; never infer or admit them."""
        if getattr(self,'domain',None)=='procurement':
            if coverage_version!={None:3,'controlled-procurement.incident-evidence.v1.1':4,'controlled-procurement.component-amount.v1':5}[getattr(self,'profile_id',None)]:raise ValueError('PROCUREMENT_STRONGER_VERSION_REQUIRED')
            return self.prepare_procurement_revision(source=source,requirements=requirements,dependencies=dependencies,inputs=inputs,frames=frames,supports=supports,dependency_inventory_complete=dependency_inventory_complete,required_inventory_pin=required_inventory_pin,candidate_template=candidate_template,predecessor_candidate=predecessor_candidate)
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.source_change_consumer_verification import SourceChangeBasisV1
        from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler
        from services.source_consumer_verification import verify_source_basis
        self.policy._access()
        retained=self.policy.read(source.artifact_ref,source.artifact_hash)
        if retained!=source or source.data.get('session_ref')!=self.session:
            raise PermissionError('SOURCE_REVISION_SCOPE_MISMATCH')
        prepared=self.service.business.policy_owner.current()
        # The historical qualified adapter supplies field/effect names only. The
        # new source/candidate/plan remain non-authoritative DEVELOPMENT evidence.
        candidate=prepared['candidate'].model_dump(mode='json')
        canonical=self.policy.canonical_consumer_source(source)
        canonical_hash=digest(canonical.model_dump(mode='json',by_alias=True))
        candidate.update(**self.scope,source_identity_ref=source.artifact_ref,source_hash=source.data['source_hash'],
            canonical_source_ir_ref='canonical-source-ir:'+canonical_hash[7:],canonical_source_ir_hash=canonical_hash,
            frames=[f.model_dump(mode='json') if hasattr(f,'model_dump') else f for f in frames],
            predecessor_candidate_set_ref=prepared['candidate'].candidate_set_ref,
            predecessor_candidate_set_hash=prepared['candidate'].candidate_set_hash,
            proposal_evidence_ref=None,proposal_evidence_hash=None,source_review_ref=None,source_review_hash=None,
            compiler_ref='assistant:typed-source-revision',compiler_hash=digest([source.artifact_hash,[r.model_dump(mode='json') for r in requirements],[d.model_dump(mode='json') for d in dependencies]]),
            provider_ref='ASSISTANT_AUTHORED',model_ref='NOT_RUN',prompt_hash=digest('NO_MODEL_CALL'),
            config_hash=digest([[r.model_dump(mode='json') for r in requirements],[d.model_dump(mode='json') for d in dependencies]]),compiled_at=datetime.now(timezone.utc).isoformat())
        # Rebind evidence to the exact retained successor, not the historical PDF.
        def rebind_evidence(value):
            if isinstance(value,dict):
                return {k:([source.artifact_ref] if k=='evidence_refs' else rebind_evidence(v)) for k,v in value.items()}
            if isinstance(value,list):return [rebind_evidence(v) for v in value]
            return value
        candidate['frames']=rebind_evidence(candidate['frames'])
        candidate.pop('candidate_set_ref');candidate.pop('candidate_set_hash')
        ch=digest(candidate)
        candidate=NormativeSemanticCandidateSetV1(candidate_set_ref='normative-candidate:'+ch[7:],candidate_set_hash=ch,**candidate)
        candidate=candidate.model_copy(update={'frames':tuple(f.model_copy(update={'frame_hash':digest(f.model_dump(mode='json',exclude={'frame_hash'}))}) for f in candidate.frames)})
        ch=digest(candidate.model_dump(mode='json',exclude={'candidate_set_ref','candidate_set_hash'}))
        candidate=candidate.model_copy(update={'candidate_set_ref':'normative-candidate:'+ch[7:],'candidate_set_hash':ch})
        assets={k:base64.b64encode(Path(__file__).with_name(name).read_bytes()).decode() for k,name in dict(
            input_requirement='deterministic_rule_evaluator.py',calendar='deterministic_rule_evaluator.py',
            set_output='expense_reimbursement_service.py',compiler='ontology_rule_plan_compiler.py').items()}
        assets['input_prerequisite']=base64.b64encode((Path(__file__).parents[1]/'contracts/expense_reimbursement.py').read_bytes()).decode()
        plan=OntologyRulePlanCompiler(compiler_ref='compiler:source-consumer-revision:v1',
            compiler_hash=raw_hash(base64.b64decode(assets['compiler']))).compile(candidate=candidate,
            validation_contract_ref=prepared['plan'].validation_contract_ref,validation_contract_hash=prepared['plan'].validation_contract_hash,
            competency_refs=tuple(sorted({ref for rule in prepared['plan'].rules for ref in rule.competency_refs})),
            compiled_at=datetime.now(timezone.utc),teaching_context=self.ctx)
        if coverage_version not in (1,2) or (coverage_version==1 and required_inventory_pin is not None):raise ValueError('SOURCE_COVERAGE_VERSION_INVALID')
        extra={}
        basis_type=SourceChangeBasisV1
        if coverage_version==2:
            from contracts.source_change_consumer_verification import SourceChangeBasisV2
            basis_type=SourceChangeBasisV2
            extra=dict(required_inventory_pin=required_inventory_pin,required_inventory=self.policy.read(required_inventory_pin.artifact_ref,required_inventory_pin.artifact_hash) if required_inventory_pin else None)
        basis=basis_type(**extra,source=source,canonical_source=canonical,session_ref=self.session,requirements=requirements,dependencies=dependencies,
            inputs=inputs,candidate=candidate,plan=plan,adapter_policy=prepared['policy'],supports=supports,
            dependency_inventory_complete=dependency_inventory_complete,consumer_assets=assets)
        verify_source_basis(basis)
        return basis

    def execute_source_revision(self, *, predecessor, successor, equivalences, correction_parent=None):
        """Selected cases traverse the same compiler/native expense evaluation seam."""
        if getattr(self,'domain',None)=='procurement':
            if equivalences:raise ValueError('PROCUREMENT_EQUIVALENCE_NOT_IN_BOUNDED_SLICE')
            return self.execute_procurement_revision(predecessor=predecessor,successor=successor,correction_parent=correction_parent)
        from contracts.expense_reimbursement import ExpenseReimbursementRequest
        from contracts.source_change_consumer_verification import SourceChangeAssessmentV1,SourceNativeObservationV1
        from services.source_consumer_verification import (verify_source_basis,source_change_projection,source_change_findings)
        from services.expense_reimbursement_service import ExpenseReimbursementService
        from pydantic import ValidationError
        self.policy._access()
        if (getattr(predecessor,'schema_version',None)=='source_change_basis.v2') != (getattr(successor,'schema_version',None)=='source_change_basis.v2'):raise ValueError('SOURCE_COVERAGE_LANE_MISMATCH')
        for b in (predecessor,successor):
            verify_source_basis(b)
            if (b.source.organization_id,b.source.workspace_id,b.session_ref)!=(self.ctx.organization_id,self.ctx.workspace_id,self.session):
                raise PermissionError('SOURCE_REVISION_EXECUTION_SCOPE_MISMATCH')
            if self.policy.read(b.source.artifact_ref,b.source.artifact_hash)!=b.source:
                raise ValueError('SOURCE_REVISION_RETAINED_SOURCE_MISMATCH')
            reference=getattr(b,'required_inventory',None)
            if reference is not None and self.policy.read(reference.artifact_ref,reference.artifact_hash)!=reference:
                raise ValueError('REQUIRED_INVENTORY_RETAINED_JOIN_MISSING')
        if correction_parent is not None:
            from services.source_consumer_verification import read_retained_source_change
            prior=read_retained_source_change(store=self.store,**self.scope,session_ref=self.session,run_pin=correction_parent,
                expected_source_hash=successor.source.data['source_hash'],expected_evaluator_hash=raw_hash(base64.b64decode(successor.consumer_assets['calendar'])))
            if prior.predecessor!=predecessor or prior.successor.source!=successor.source:
                raise ValueError('SOURCE_CORRECTION_ANCESTRY_MISMATCH')
        env,impact,selected=source_change_projection(predecessor,successor,equivalences)
        observations=[]
        cases={c.case_id:c for c in successor.inputs}
        for case_id in selected:
            c=cases[case_id]
            if not c.invoice_attached:
                # Execute the existing core input contract; rejection is not MATCH.
                try:
                    ExpenseReimbursementRequest(action='evaluate',submitted_on=c.submitted_on,applicant_fields=c.applicant_fields)
                except ValidationError:
                    response=dict(input_status='REJECTED',result_created=False,owner='contracts.expense_reimbursement.ExpenseReimbursementRequest')
                else:raise ValueError('SOURCE_MISSING_INVOICE_NOT_REJECTED')
            else:
                response=ExpenseReimbursementService.evaluate_policy(policy=successor.adapter_policy,plan=successor.plan,
                    candidate=successor.candidate,invoice_date=c.invoice_date,submitted_on=c.submitted_on,
                    applicant_fields=c.applicant_fields,target_evidence_refs=[successor.source.artifact_ref])
            observations.append(SourceNativeObservationV1(case_id=case_id,input_hash=digest(c.model_dump(mode='json')),
                plan_hash=successor.plan.plan_hash,response=response,response_hash=digest(response),observed_at=datetime.now(timezone.utc)))
        stronger=getattr(successor,'schema_version',None)=='source_change_basis.v2'
        if stronger != (getattr(predecessor,'schema_version',None)=='source_change_basis.v2'):raise ValueError('SOURCE_COVERAGE_LANE_MISMATCH')
        extra={}
        assessment_type=SourceChangeAssessmentV1
        findings=source_change_findings(predecessor,successor,equivalences)
        if stronger:
            from contracts.source_change_consumer_verification import SourceChangeAssessmentV2
            from services.source_consumer_verification import required_inventory_findings,required_coverage_status
            assessment_type=SourceChangeAssessmentV2
            findings=required_inventory_findings(predecessor,successor,equivalences,observations)
            extra=dict(coverage_status=required_coverage_status(findings))
        assessment=assessment_type(**extra,predecessor=predecessor,successor=successor,equivalences=equivalences,
            correction_parent=correction_parent,delta=env,impact=impact,selected_case_ids=selected,observations=tuple(observations),
            findings=findings,produced_at=datetime.now(timezone.utc))
        record=self.record('consumer_run',dict(lane='source_change',assessment=assessment.model_dump(mode='json')))
        self.store.put(record)
        return record

    @classmethod
    def for_procurement(cls, *, ctx, store, session_ref, profile_id=None):
        """Same product owner/auth/store, with no expense adapter or seed."""
        if not isinstance(session_ref,str) or not session_ref:raise ValueError('PROCUREMENT_SESSION_REQUIRED')
        from storage.ontology_build_store import OntologyBuildStore
        if type(store) is not OntologyBuildStore:raise TypeError('PROCUREMENT_NATIVE_STORE_REQUIRED')
        obj=cls.__new__(cls)
        from contracts.source_change_consumer_verification import INCIDENT_PROFILE, AMOUNT_PROFILE, FEE_FACTS_PROFILE, DEVELOPMENT_MEASURE_PROFILE
        if profile_id not in (None,INCIDENT_PROFILE,AMOUNT_PROFILE,FEE_FACTS_PROFILE,DEVELOPMENT_MEASURE_PROFILE):raise ValueError('INCIDENT_PROFILE_SELECTION_INVALID')
        obj.profile_id=profile_id
        obj.domain='procurement';obj.ctx=ctx;obj.store=store;obj.session=session_ref
        obj.policy=PolicyRevisionService(ctx=ctx,store=store,policy_owner=None)
        obj.scope=obj.policy.scope
        return obj

    def freeze_procurement_inventory(self, *, source, requirements, dependencies, inputs, questions, boundary, witness_pins=()):
        from contracts.source_change_consumer_verification import ProcurementRequiredInventoryV1
        from contracts.ontology_building import CanonicalEvidenceBoundaryV1
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from services.competency_contract_service import CompetencyContractService
        from services.source_consumer_verification import verify_required_declaration
        self.policy._access()
        if self.policy.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=self.session or source.data.get('domain')!='procurement':raise PermissionError('PROCUREMENT_SOURCE_ACCESS_MISMATCH')
        if source.data.get('profile_id')!=getattr(self,'profile_id',None):raise ValueError('INCIDENT_EXPLICIT_PROFILE_REQUIRED')
        canonical=self.policy.canonical_consumer_source(source);ch=digest(canonical.model_dump(mode='json',by_alias=True))
        now=datetime.now(timezone.utc)
        values=dict(**self.scope,source_identity_ref=source.artifact_ref,source_hash=source.data['source_hash'],canonical_source_ir_ref='canonical-source-ir:'+ch[7:],canonical_source_ir_hash=ch,intended_domain='controlled software subscription procurement',intended_use=boundary['included'],target_time=now,jurisdiction='synthetic internal policy',explicit_exclusions=tuple(boundary['excluded']),created_at=now)
        h=digest({k:v.isoformat() if isinstance(v,datetime) else v for k,v in values.items()})
        declared_boundary=CanonicalEvidenceBoundaryV1(boundary_ref='boundary:'+h[7:],boundary_hash=h,**values)
        competency=CompetencyContractService().propose(boundary=declared_boundary,graph_intent_brief_ref=source.artifact_ref+'#declared-source-questions',graph_intent_brief_hash=digest(questions),questions=tuple(dict(q,evidence_refs=tuple(source.artifact_ref+'#'+r.removeprefix('evidence:') for r in q['evidence_refs']),acceptance_obligations=('Receipt coverage only; entire competency remains unqualified.',)) for q in questions),created_at=now)
        # Repeated freeze uses the actual stored context, avoiding timestamp drift.
        contexts=[r for r in self.policy._list('consumer_candidate') if r.data.get('lane')=='procurement_context' and r.data.get('source')==self.policy.pin(source) and r.data.get('session_ref')==self.session]
        witnesses=[]
        if self.profile_id=='controlled-procurement.incident-evidence.v1.1':
            for pin in witness_pins:
                witness=self.policy.read(pin.artifact_ref,pin.artifact_hash)
                if witness.data.get('lane')!='incident_witness' or witness.data.get('session_ref')!=self.session:raise PermissionError('INCIDENT_WITNESS_ROSTER_SCOPE_MISMATCH')
                witnesses.append(witness.model_dump(mode='json'))
        elif witness_pins:raise ValueError('INCIDENT_PROFILE_REQUIRED_FOR_WITNESSES')
        extra=dict(profile_id=self.profile_id,incident_witnesses=witnesses) if self.profile_id=='controlled-procurement.incident-evidence.v1.1' else dict(profile_id=self.profile_id) if self.profile_id is not None else {}
        context=self.record('consumer_candidate',dict(**extra,lane='procurement_context',domain='procurement',source=self.policy.pin(source),questions=list(questions),boundary=declared_boundary.model_dump(mode='json'),competency=competency.model_dump(mode='json'),native_frame_schema=NormativeSemanticCandidateSetV1.model_json_schema(),schema_status='DECLARED_NOT_ADMITTED'))
        # One atomic context operation protects a competing source/session freeze.
        request_hash=digest(dict(source=self.policy.pin(source),questions=questions,boundary=boundary,**extra))
        if contexts:
            if (self.profile_id=='controlled-procurement.incident-evidence.v1.1' and contexts[0].data.get('incident_witnesses')!=witnesses) or contexts[0].data['questions']!=list(questions) or contexts[0].data['boundary']['intended_use']!=boundary['included'] or contexts[0].data['boundary']['explicit_exclusions']!=list(boundary['excluded']):raise ValueError('PROCUREMENT_CONTEXT_ALREADY_FROZEN')
            context=contexts[0]
        else:context=self.policy._operation('procurement-context:'+digest([self.session,source.artifact_ref]),request_hash,context)
        inventory_type=ProcurementRequiredInventoryV1
        if self.profile_id=='controlled-procurement.incident-evidence.v1.1':
            from contracts.source_change_consumer_verification import ProcurementRequiredInventoryV2
            inventory_type=ProcurementRequiredInventoryV2
        if self.profile_id=='controlled-procurement.component-amount.v1':
            from contracts.source_change_consumer_verification import ProcurementRequiredInventoryV3
            inventory_type=ProcurementRequiredInventoryV3
        inventory=inventory_type(**(dict(profile_id=self.profile_id) if self.profile_id else {}),domain='procurement',source=ArtifactPinV1(artifact_kind='policy_revision',**self.policy.pin(source)),source_hash=source.data['source_hash'],**self.scope,session_ref=self.session,effective_on=source.data['effective_on'],validation_basis=ArtifactPinV1(artifact_kind='policy_revision',**self.policy.pin(context)),requirements=requirements,dependencies=dependencies,inputs=inputs,source_document_hash=source.data['source_document_hash'],canonical_hash=ch)
        verify_required_declaration(inventory,source)
        data=inventory.model_dump(mode='json');h=digest(data)
        reference=self.record('consumer_candidate',dict(lane='source_required_inventory',inventory=data,inventory_hash=h))
        return self.policy._operation('required-inventory:'+digest([self.session,source.artifact_ref]),h,reference)

    def prepare_procurement_revision(self, *, source, requirements, dependencies, inputs, frames, supports,
                                     dependency_inventory_complete, required_inventory_pin, candidate_template, predecessor_candidate):
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.source_change_consumer_verification import SourceChangeBasisV3
        from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler
        from services.source_consumer_verification import verify_procurement_basis,required_inventory_type
        self.policy._access()
        if self.policy.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=self.session or source.data.get('domain')!='procurement':raise PermissionError('PROCUREMENT_SOURCE_ACCESS_MISMATCH')
        if required_inventory_pin is None:raise ValueError('PROCUREMENT_PREPARATION_DECLARATION_REQUIRED')
        reference=self.policy.read(required_inventory_pin.artifact_ref,required_inventory_pin.artifact_hash)
        inv=required_inventory_type(reference.data['inventory']).model_validate(reference.data['inventory'])
        if getattr(inv,'domain',None)!='procurement':raise ValueError('PROCUREMENT_DECLARATION_DOMAIN_MISMATCH')
        context=self.policy.read(inv.validation_basis.artifact_ref,inv.validation_basis.artifact_hash)
        if source.data.get('profile_id')!=getattr(self,'profile_id',None):raise ValueError('INCIDENT_EXPLICIT_PROFILE_REQUIRED')
        canonical=self.policy.canonical_consumer_source(source);ch=digest(canonical.model_dump(mode='json',by_alias=True))
        if not isinstance(candidate_template,NormativeSemanticCandidateSetV1):raise TypeError('PROCUREMENT_TYPED_TEMPLATE_REQUIRED')
        candidate=candidate_template.model_dump(mode='json',exclude={'candidate_set_ref','candidate_set_hash'})
        def rebind(value):
            if isinstance(value,dict):
                return {k:([source.artifact_ref+'#'+ref.removeprefix('evidence:') for ref in v] if k=='evidence_refs' else rebind(v)) for k,v in value.items()}
            if isinstance(value,list):return [rebind(v) for v in value]
            return value
        if self.profile_id=='controlled-procurement.incident-evidence.v1.1':frames=self._incident_frames(frames)
        rebound=rebind([f.model_dump(mode='json') if hasattr(f,'model_dump') else f for f in frames])
        for f in rebound:f['frame_hash']=digest({k:v for k,v in f.items() if k!='frame_hash'})
        assets={k:base64.b64encode(Path(__file__).with_name(name).read_bytes()).decode() for k,name in dict(compiler='ontology_rule_plan_compiler.py',predicate='deterministic_rule_evaluator.py',verifier='source_consumer_verification.py').items()}
        if self.profile_id is not None:
            assets.update(input_contract=base64.b64encode((Path(__file__).parents[1]/'contracts/source_change_consumer_verification.py').read_bytes()).decode(),product=base64.b64encode(Path(__file__).read_bytes()).decode())
        candidate.update(teaching_candidate=True,**self.scope,source_identity_ref=source.artifact_ref,source_hash=source.data['source_hash'],canonical_source_ir_ref='canonical-source-ir:'+ch[7:],canonical_source_ir_hash=ch,frames=rebound,compiled_at=datetime.now(timezone.utc).isoformat(),compiler_ref='assistant:typed-procurement-input:v1',compiler_hash=digest(rebound),provider_ref='ASSISTANT_AUTHORED',model_ref='NOT_RUN',prompt_hash=raw_hash(b''),config_hash=digest(inv.model_dump(mode='json')),profile_ref=reference.artifact_ref,profile_hash=reference.data['inventory_hash'],normative_schema_ref=context.artifact_ref+'#declared-frame-schema',normative_schema_hash=digest(context.data['native_frame_schema']),competency_contract_ref=context.data['competency']['competency_contract_ref'],competency_contract_hash=context.data['competency']['competency_contract_hash'],proposal_evidence_ref=None,proposal_evidence_hash=None,source_review_ref=None,source_review_hash=None,predecessor_candidate_set_ref=predecessor_candidate.candidate_set_ref if predecessor_candidate else None,predecessor_candidate_set_hash=predecessor_candidate.candidate_set_hash if predecessor_candidate else None,change_cause='bounded_repair' if predecessor_candidate else 'initial_compilation',triggering_ref='explicit-procurement-correction' if predecessor_candidate else None)
        # Serialize once before hashing, preserving the producing JSON UTC recipe.
        parsed=NormativeSemanticCandidateSetV1.model_validate(dict(candidate_set_ref='pending',candidate_set_hash='sha256:'+'0'*64,**candidate))
        payload=parsed.model_dump(mode='json',exclude={'candidate_set_ref','candidate_set_hash'});h=digest(payload)
        candidate=NormativeSemanticCandidateSetV1(candidate_set_ref='normative-candidate:'+h[7:],candidate_set_hash=h,**payload)
        plan=OntologyRulePlanCompiler(compiler_ref='compiler:source-consumer-procurement:v1',compiler_hash=raw_hash(base64.b64decode(assets['compiler']))).compile(candidate=candidate,validation_contract_ref=context.artifact_ref,validation_contract_hash=context.artifact_hash,competency_refs=tuple(q['competency_id'] for q in context.data['competency']['questions']),compiled_at=datetime.now(timezone.utc),teaching_context=self.ctx)
        basis_type=SourceChangeBasisV3
        if self.profile_id=='controlled-procurement.incident-evidence.v1.1':
            from contracts.source_change_consumer_verification import SourceChangeBasisV4
            basis_type=SourceChangeBasisV4
        component_values={}
        if self.profile_id=='controlled-procurement.component-amount.v1':
            from contracts.source_change_consumer_verification import SourceChangeBasisV5
            basis_type=SourceChangeBasisV5
            formations=tuple(self.form_procurement_amount(source=source,case=case,require_dispatch=True) for case in inputs)
            forms=tuple(self.policy.read(case.form_pin.artifact_ref,case.form_pin.artifact_hash) for case in inputs)
            bindings={}
            for d in dependencies:
                if d.consumer in ('component_receipt','formation_handoff'):
                    case=next((c for c in inputs if c.case_id in d.case_ids),None)
                    if case is not None:
                        f=next(f for f in formations if f.data['case_id']==case.case_id)
                        bindings[d.target_id]=f.artifact_ref+'#'+d.selector
            component_values=dict(forms=forms,formations=formations,receipt_bindings=bindings)
        basis=basis_type(**component_values,**(dict(profile_id=self.profile_id) if self.profile_id else {}),domain='procurement',source=source,canonical_source=canonical,context=context,session_ref=self.session,requirements=requirements,dependencies=dependencies,dependency_inventory_complete=dependency_inventory_complete,inputs=inputs,candidate=candidate,plan=plan,supports=supports,consumer_assets=assets,required_inventory_pin=required_inventory_pin,required_inventory=reference)
        verify_procurement_basis(basis)
        return basis

    def execute_procurement_revision(self, *, predecessor, successor, correction_parent):
        from contracts.source_change_consumer_verification import SourceChangeAssessmentV3,SourceNativeObservationV1
        from contracts.ontology_paired_evaluation import ObservedFactsV1
        from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
        from services.source_consumer_verification import (verify_procurement_basis,source_change_projection,required_inventory_findings,required_coverage_status,read_retained_source_change)
        self.policy._access()
        for b in (predecessor,successor):
            if getattr(b,'profile_id',None)!=self.profile_id:raise ValueError('INCIDENT_EXPLICIT_PROFILE_REQUIRED')
            if self.profile_id=='controlled-procurement.incident-evidence.v1.1':
                for raw_record in b.context.data['incident_witnesses']:
                    if self.policy.read(raw_record['artifact_ref'],raw_record['artifact_hash']).model_dump(mode='json')!=raw_record:raise ValueError('INCIDENT_WITNESS_RETAINED_MISMATCH')
            verify_procurement_basis(b)
            if (b.source.organization_id,b.source.workspace_id,b.session_ref)!=(self.ctx.organization_id,self.ctx.workspace_id,self.session):raise PermissionError('PROCUREMENT_EXECUTION_SCOPE_MISMATCH')
            retained_records=[b.source,b.context,b.required_inventory]
            if self.profile_id=='controlled-procurement.component-amount.v1':retained_records.extend((*b.forms,*b.formations))
            for r in retained_records:
                if r is not None and self.policy.read(r.artifact_ref,r.artifact_hash)!=r:raise ValueError('PROCUREMENT_RETAINED_BASIS_MISMATCH')
        delta,impact,selected=source_change_projection(predecessor,successor,{})
        observations=[]
        for case in successor.inputs:
            if case.case_id not in selected:continue
            now=datetime.now(timezone.utc)
            formation=None
            if getattr(successor,'profile_id',None)=='controlled-procurement.component-amount.v1':
                from services.source_consumer_verification import component_native_projection
                native,formation=component_native_projection(case,successor)
                incident=None
            elif getattr(successor,'profile_id',None)=='controlled-procurement.incident-evidence.v1.1':
                from services.source_consumer_verification import incident_projection
                native,incident=incident_projection(case,successor)
            else:native,incident=case.native_facts(),None
            facts=ObservedFactsV1.create(**self.scope,facts=native,evidence_refs=(successor.source.artifact_ref,),frozen_at=now)
            epoch='procurement-case:'+digest([successor.source.artifact_ref,self.session,case.case_id,facts.facts_hash])[7:]
            bundle=DeterministicRuleEvaluator().evaluate(plan=successor.plan,execution_epoch_ref=epoch,observed_facts=facts.facts,target_evidence_refs=(facts.facts_ref,),evaluated_at=now,include_all_rules=True)
            response=dict(facts=facts.model_dump(mode='json'),bundle=bundle.model_dump(mode='json'))
            if formation is not None:response['formation']=formation.model_dump(mode='json')
            if incident is not None:
                from services.source_consumer_verification import incident_outputs
                incident['outputs']=incident_outputs(case,successor,response['bundle'],incident)
                response['incident']=incident
            observations.append(SourceNativeObservationV1(case_id=case.case_id,input_hash=digest(case.model_dump(mode='json')),plan_hash=successor.plan.plan_hash,response=response,response_hash=digest(response),observed_at=now))
        findings=required_inventory_findings(predecessor,successor,{},observations)
        assessment_type=SourceChangeAssessmentV3
        if self.profile_id=='controlled-procurement.incident-evidence.v1.1':
            from contracts.source_change_consumer_verification import SourceChangeAssessmentV4
            assessment_type=SourceChangeAssessmentV4
        if self.profile_id=='controlled-procurement.component-amount.v1':
            from contracts.source_change_consumer_verification import SourceChangeAssessmentV5
            assessment_type=SourceChangeAssessmentV5
        assessment=assessment_type(domain='procurement',predecessor=predecessor,successor=successor,equivalences={},correction_parent=correction_parent,delta=delta,impact=impact,selected_case_ids=selected,observations=tuple(observations),findings=findings,coverage_status=required_coverage_status(findings),verifier_hash=raw_hash(base64.b64decode(successor.consumer_assets['verifier'])),produced_at=datetime.now(timezone.utc))
        record=self.record('consumer_run',dict(lane='source_change',assessment=assessment.model_dump(mode='json')))
        # Validate retained correction closure inside the existing atomic owner.
        def verify():
            read_retained_source_change(store=self.store,**self.scope,session_ref=self.session,run_pin=ArtifactPinV1(artifact_kind='policy_revision',**self.policy.pin(record)),expected_source_hash=successor.source.data['source_hash'],expected_evaluator_hash=raw_hash(base64.b64decode(successor.consumer_assets['predicate'])),expected_verifier_hash=assessment.verifier_hash)
        return self.policy._operation('procurement-run:'+record.artifact_ref,digest(record.data),record,verify=verify)

    @staticmethod
    def _incident_frames(frames):
        """Source-bound v1.1 lowering; no references/case IDs or output lookup."""
        from contracts.normative_semantics import SemanticPredicateV1
        from eos.artifact_hashing import content_hash
        changes={
            'procurement.incident_ref':('procurement.recording_established','IE-02'),
            'procurement.incident_recorded_by':('procurement.recorder_qualified','IE-02'),
            'procurement.emergency_security_remediation':('procurement.emergency_established','IE-04'),
        }
        lowered=[]
        for frame in frames:
            if frame.frame_kind!='exception':lowered.append(frame);continue
            conditions=[]
            for p in frame.conditions:
                if p.fact_ref in changes:
                    fact,clause=changes[p.fact_ref]
                    p=SemanticPredicateV1(predicate_id=p.predicate_id,fact_ref=fact,operator='EQ',expected_value=True,evidence_refs=('evidence:'+clause,))
                conditions.append(p)
            payload=frame.model_dump(mode='json',exclude={'frame_hash'})
            payload.update(conditions=[p.model_dump(mode='json') for p in conditions],
                source_region_refs=['policy.exception','IE-02','IE-04'],evidence_refs=['evidence:policy.exception','evidence:IE-02','evidence:IE-04'])
            lowered.append(type(frame)(frame_hash='sha256:'+content_hash(payload),**payload))
        return tuple(lowered)

    def retain_procurement_fee_form(self, *, source, case_id, form, component_ids, origin_hash, amendment_parent=None):
        """Materialize authored facts/roster, not institutional evidence authority."""
        from contracts.source_change_consumer_verification import AMOUNT_PROFILE, ProcurementFeeFormV1
        self.policy._access()
        if self.profile_id!=AMOUNT_PROFILE or source.data.get('profile_id')!=AMOUNT_PROFILE:
            raise ValueError('AMOUNT_EXPLICIT_PROFILE_REQUIRED')
        if self.policy.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=self.session:
            raise PermissionError('AMOUNT_SOURCE_SCOPE_MISMATCH')
        parsed=ProcurementFeeFormV1.model_validate(form)
        ids=tuple(c.component_id for c in parsed.fee_components)
        if tuple(component_ids)!=ids:raise ValueError('AMOUNT_ROSTER_MISMATCH')
        if amendment_parent is not None:
            parent=self.policy.read(amendment_parent.artifact_ref,amendment_parent.artifact_hash)
            if (amendment_parent.artifact_kind!='policy_revision' or parent.data.get('lane')!='component_formation' or
                parent.data.get('session_ref')!=self.session or parent.data['formation']['source_pin']!=dict(artifact_kind='policy_revision',**self.policy.pin(source))):
                raise ValueError('AMOUNT_AMENDMENT_PARENT_MISMATCH')
        locators={}
        for index,c in enumerate(parsed.fee_components):
            expected='/components/'+c.component_id
            if c.evidence.get('artifact_ref')!='synthetic-fee-form:v1' or c.evidence.get('locator')!=expected:
                raise ValueError('AMOUNT_VIRTUAL_LOCATOR_INVALID')
            locators[c.component_id]=dict(original=dict(c.evidence),retained_locator='/form/fee_components/'+str(index))
        record=self.record('consumer_candidate',dict(lane='procurement_fee_form',profile_id=AMOUNT_PROFILE,
            source_pin=dict(artifact_kind='policy_revision',**self.policy.pin(source)),case_id=case_id,
            form=form,component_ids=list(ids),roster_hash=digest(ids),origin_hash=origin_hash,
            locator_mapping=locators,amendment_parent=amendment_parent.model_dump(mode='json') if amendment_parent else None,
            authority='SYNTHETIC_DECLARED_NOT_AUTHENTICATED'))
        return self.policy._operation('amount-form:'+digest([self.session,case_id,record.data]),digest(record.data),record)

    def form_procurement_amount(self, *, source, case, require_dispatch=False):
        """Only warm amount producer. Never reads expected/reference outcomes."""
        from contracts.source_change_consumer_verification import (AMOUNT_PROFILE, ProcurementFeeFormV1,
            ProcurementFormationV1,ProcurementComponentReceiptV1)
        from services.source_consumer_verification import verify_component_form
        self.policy._access()
        if self.profile_id!=AMOUNT_PROFILE:raise ValueError('AMOUNT_EXPLICIT_PROFILE_REQUIRED')
        if self.policy.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=self.session:raise PermissionError('AMOUNT_SOURCE_SCOPE_MISMATCH')
        material=self.policy.read(case.form_pin.artifact_ref,case.form_pin.artifact_hash)
        verify_component_form(case,source,material,self.session)
        parsed=ProcurementFeeFormV1.model_validate(case.form);receipts=[]
        for raw,c in zip(case.form['fee_components'],parsed.fee_components,strict=True):
            state,reason='UNRESOLVED','CLASSIFICATION_UNRESOLVED'
            amount=c.amount_usd if type(c.amount_usd) is int else None
            if c.fee_kind in ('subscription','usage') and type(c.separately_cancellable) is bool:
                if c.separately_cancellable:
                    state,reason='EXCLUDED','SEPARATELY_CANCELLABLE_USAGE' if c.fee_kind=='usage' else 'CANCELLABLE_SUBSCRIPTION'
                elif c.fee_kind=='subscription':
                    period=c.due_period
                    if hasattr(period,'start') and (period.start,period.end_exclusive)==('2026-06-01','2027-06-01') and type(period.first_twelve_months) is bool and c.evidence.get('period_terms'):
                        if not period.first_twelve_months:state,reason='EXCLUDED','OUTSIDE_STIPULATED_WINDOW'
                        elif amount is None:state,reason='UNRESOLVED','AMOUNT_UNRESOLVED'
                        else:state,reason='INCLUDED','NON_CANCELLABLE_SUBSCRIPTION_IN_WINDOW'
                    else:reason='PERIOD_UNRESOLVED'
                else:reason='UNSUPPORTED_FEE_TERMS'
                if not c.evidence.get('classification_terms'):state,reason='UNRESOLVED','CLASSIFICATION_EVIDENCE_UNRESOLVED'
            receipts.append(ProcurementComponentReceiptV1(component_id=c.component_id,component_hash=digest(raw),
                disposition=state,reason=reason,amount=amount,supplied_keys=tuple(sorted(raw)),
                evidence_refs=(material.artifact_ref+'#'+material.data['locator_mapping'][c.component_id]['retained_locator'],)))
        complete=all(r.disposition!='UNRESOLVED' for r in receipts)
        amount=sum(r.amount for r in receipts if r.disposition=='INCLUDED') if complete else None
        key='procurement.first_year_non_cancellable_usd';supplied=key in case.form['facts'];caller=case.form['facts'].get(key)
        formed=ProcurementFormationV1(source_pin=dict(artifact_kind='policy_revision',**self.policy.pin(source)),form_pin=case.form_pin,
            input_hash=digest(case.model_dump(mode='json')),roster_hash=material.data['roster_hash'],producer_hash=raw_hash(Path(__file__).read_bytes()),
            status='COMPLETE' if complete else 'UNKNOWN',amount=amount,caller_supplied=supplied,caller_assertion=caller,
            dispatch='AGGREGATE_MISMATCH' if amount is not None and caller is not None and caller!=amount else 'READY',receipts=tuple(receipts))
        record=self.record('consumer_candidate',dict(lane='component_formation',profile_id=AMOUNT_PROFILE,case_id=case.case_id,
            input=case.model_dump(mode='json'),material=material.model_dump(mode='json'),formation=formed.model_dump(mode='json')))
        retained=self.policy._operation('amount-formation:'+digest([self.session,formed.input_hash]),digest(record.data),record)
        if require_dispatch and formed.dispatch!='READY':raise ValueError('AGGREGATE_MISMATCH:'+retained.artifact_ref)
        return retained

    def _fee_access(self):
        from contracts.source_change_consumer_verification import FEE_FACTS_PROFILE
        self.policy._access()
        if self.profile_id != FEE_FACTS_PROFILE:
            raise ValueError('FEE_EXPLICIT_PROFILE_REQUIRED')

    def _fee_read(self, pin, lane):
        self._fee_access()
        value = self.policy.read(pin.artifact_ref, pin.artifact_hash)
        if (pin.artifact_kind != 'policy_revision' or value.kind != 'consumer_candidate'
                or value.data.get('lane') != lane or value.data.get('profile_id') != self.profile_id
                or value.data.get('session_ref') != self.session):
            raise PermissionError('FEE_RECORD_SCOPE_MISMATCH')
        return value

    def _fee_retain(self, lane, **data):
        self._fee_access()
        payload = dict(lane=lane, profile_id=self.profile_id, **data)
        record = self.record('consumer_candidate', payload)
        return self.policy._operation('fee:'+lane+':'+digest(payload), digest(payload), record)

    @staticmethod
    def _fee_pin(record):
        return ArtifactPinV1(artifact_kind='policy_revision', artifact_ref=record.artifact_ref, artifact_hash=record.artifact_hash)

    def retain_fee_document(self, *, artifacts, artifact_pin, judgments, pages, assertion_page_refs):
        """Import exact governed original bytes; declarations never grant authority."""
        from services.workspace_artifact_service import WorkspaceArtifactService
        from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
        self._fee_access()
        if type(artifacts) is not WorkspaceArtifactService or artifacts._store is not self.store:
            raise TypeError('FEE_GOVERNED_ARTIFACT_OWNER_REQUIRED')
        descriptor = artifacts.resolve(ctx=self.ctx, artifact_pin=artifact_pin)
        if descriptor.media_type != 'application/pdf':
            raise ValueError('FEE_PDF_SOURCE_REQUIRED')
        with WorkspaceArtifactBlobStore(ctx=self.ctx,max_bytes=32*1024**2).exact_get(
                byte_owner_ref=descriptor.byte_owner_ref,expected_hash=descriptor.content_hash,expected_size=descriptor.byte_size) as stream:
            raw = stream.read()
        if judgments['source_sha256'] != descriptor.content_hash.removeprefix('sha256:'):
            raise ValueError('FEE_SOURCE_JUDGMENT_PIN_MISMATCH')
        if (judgments['provenance'] != ['DEVELOPMENT','ASSISTANT_AUTHORED','NOT_QUALIFIED']
                or judgments['independent_reviews'] != 0):
            raise ValueError('FEE_UNQUALIFIED_SOURCE_REQUIRED')
        if set(pages) != set(judgments['inspection']['render_hashes']):
            raise ValueError('FEE_PAGE_EVIDENCE_MISSING')
        for name, encoded in pages.items():
            if raw_hash(base64.b64decode(encoded,validate=True)) != 'sha256:'+judgments['inspection']['render_hashes'][name]:
                raise ValueError('FEE_PAGE_PIN_MISMATCH')
        rows = judgments['rows']
        if not rows or len({r['row'] for r in rows}) != len(rows):
            raise ValueError('FEE_SOURCE_ROSTER_INVALID')
        row_ids={'row-'+str(r['row']) for r in rows}
        if set(assertion_page_refs)!=row_ids:raise ValueError('FEE_SOURCE_ASSERTION_ROSTER_INVALID')
        for roles in assertion_page_refs.values():
            if set(roles)!={'service_period','cancellation','due_condition'}:raise ValueError('FEE_SOURCE_ASSERTION_ROLES_INVALID')
            for refs in roles.values():
                if not refs or len(refs)!=len(set(refs)) or any(not ref.startswith('p') or not ref[1:].isdigit() or 'p'+ref[1:].zfill(2)+'.png' not in pages for ref in refs):
                    raise ValueError('FEE_SOURCE_ASSERTION_PAGE_INVALID')
        return self._fee_retain('fee_document',descriptor=descriptor.model_dump(mode='json'),
            source_bytes=base64.b64encode(raw).decode(),judgments=judgments,pages=pages,assertion_page_refs=assertion_page_refs,
            evidence_role='FACTUAL_ASSERTIONS_NOT_POLICY',source_authority='NOT_QUALIFIED')

    def freeze_fee_inventory(self, *, document_pin, roster, source_kind='SOURCE_TRANSCRIPTION'):
        """Freeze independently declared source membership before candidate bindings."""
        from contracts.source_change_consumer_verification import SourceRequirementInstanceV1,FeeFactsDependencyV1
        document = self._fee_read(document_pin,'fee_document')
        rows = {'row-'+str(r['row']):r for r in document.data['judgments']['rows']}
        if not roster or len({r['component_id'] for r in roster}) != len(roster):
            raise ValueError('FEE_DECLARED_ROSTER_INVALID')
        if source_kind not in ('SOURCE_TRANSCRIPTION','SYNTHETIC_METAMORPHIC'):raise ValueError('FEE_SOURCE_KIND_INVALID')
        if source_kind=='SOURCE_TRANSCRIPTION' and (len(roster)!=len(rows) or {r['source_row_id'] for r in roster}!=set(rows)):
            raise ValueError('FEE_SOURCE_ROSTER_INCOMPLETE_OR_DUPLICATED')
        requirements=[];dependencies=[]
        for row in roster:
            source=rows.get(row['source_row_id'])
            if source is None: raise ValueError('FEE_SOURCE_ROW_UNKNOWN')
            cid=row['component_id'];rid='fee:'+cid;tid='receipt:'+cid;quote=source['service_raw']
            instance='fee-source-instance:'+digest([document.artifact_hash,row['source_row_id'],source['locator']])[7:]
            requirements.append(SourceRequirementInstanceV1(logical_id=rid,instance_ref=instance,char_start=0,
                char_end=len(quote),quote=quote,target_ids=(tid,)))
            dependencies.append(FeeFactsDependencyV1(requirement_id=rid,target_id=tid,consumer='fee_component',selector=cid))
        quote=document.data['judgments']['table_total']['raw']
        requirements.append(SourceRequirementInstanceV1(logical_id='fee:table',instance_ref='fee-table:'+document.artifact_hash[7:],
            char_start=0,char_end=len(quote),quote=quote,target_ids=('handoff:total',)))
        dependencies.append(FeeFactsDependencyV1(requirement_id='fee:table',target_id='handoff:total',consumer='fee_total',selector='total'))
        return self._fee_retain('fee_declaration',document=self._fee_pin(document).model_dump(mode='json'),roster=list(roster),source_kind=source_kind,
            requirements=[r.model_dump(mode='json') for r in requirements],dependencies=[d.model_dump(mode='json') for d in dependencies])

    def retain_fee_facts(self, *, document_pin, declaration_pin, form, amendment_parent=None):
        from contracts.source_change_consumer_verification import FeeFactsFormV1
        document=self._fee_read(document_pin,'fee_document');declared=self._fee_read(declaration_pin,'fee_declaration')
        if declared.data['document']!=document_pin.model_dump(mode='json'):raise ValueError('FEE_DECLARATION_SOURCE_MISMATCH')
        parsed=FeeFactsFormV1.model_validate(form)
        if parsed.source_kind!=declared.data['source_kind']:raise ValueError('FEE_DECLARED_SOURCE_KIND_MISMATCH')
        expected={r['component_id']:r['source_row_id'] for r in declared.data['roster']}
        if {c.component_id:c.source_row_id for c in parsed.components}!=expected:raise ValueError('FEE_FORM_ROSTER_MISMATCH')
        rows={'row-'+str(r['row']):r for r in document.data['judgments']['rows']}
        for c in parsed.components:
            if (c.locator,c.source_label)!=(rows[c.source_row_id]['locator'],rows[c.source_row_id]['service_raw']):
                raise ValueError('FEE_SOURCE_LOCATOR_MISMATCH')
        for c in parsed.components:
            for role in ('service_period','cancellation','due_condition'):
                if set(getattr(c,role).evidence_refs)!=set(document.data['assertion_page_refs'][c.source_row_id][role]):
                    raise ValueError('FEE_SOURCE_ASSERTION_LOCATION_MISMATCH')
        if parsed.source_kind=='SOURCE_TRANSCRIPTION':
            for c in parsed.components:
                row=rows[c.source_row_id]
                if (c.currency,c.source_unit,c.fee_kind,c.amount_basis)!=('CNY',row['unit'],'maintenance','tax_inclusive_stated_price') or (c.raw_amount not in (None,'UNKNOWN',row['price_raw'])):
                    raise ValueError('FEE_SOURCE_AMOUNT_ASSERTION_CONFLICT')
                expected_state='CONFLICT' if row['row']==document.data['judgments']['period_conflict']['row'] else 'KNOWN'
                if (c.service_period.state,c.service_period.raw)!=(expected_state,row['service_period_raw']) or (c.cancellation.state,c.cancellation.raw)!=('UNKNOWN',None) or (c.due_condition.state,c.due_condition.raw)!=('UNKNOWN',None):
                    raise ValueError('FEE_SOURCE_ASSERTION_CONFLICT')
                for assertion in (c.service_period,c.cancellation,c.due_condition):
                    if not assertion.evidence_refs or any(not ref.startswith('p') or not ref[1:].isdigit() or ('p'+ref[1:].zfill(2)+'.png') not in document.data['judgments']['inspection']['render_hashes'] for ref in assertion.evidence_refs):
                        raise ValueError('FEE_ASSERTION_PAGE_REFERENCE_INVALID')
        expected_context={k:document.data['judgments'][k] for k in ('period_conflict','payment','signing','scope_facts')}
        if form['source_context']!=expected_context:raise ValueError('FEE_SOURCE_CONTEXT_MISMATCH')
        if amendment_parent:
            parent=self._fee_read(amendment_parent,'fee_form')
            if parent.data['document']!=document_pin.model_dump(mode='json') or {k:v for k,v in parent.data['form'].items() if k!='case_id'}=={k:v for k,v in form.items() if k!='case_id'}:
                raise ValueError('FEE_AMENDMENT_INPUT_OR_SOURCE_MISMATCH')
        return self._fee_retain('fee_form',document=document_pin.model_dump(mode='json'),declaration=declaration_pin.model_dump(mode='json'),
            form=deepcopy(form),amendment_parent=amendment_parent.model_dump(mode='json') if amendment_parent else None)

    def form_fee_facts(self, *, form_pin):
        from decimal import Decimal, localcontext
        import re
        from contracts.source_change_consumer_verification import FeeFactsFormV1,FeeFactsFormationV1,FeeMoneyV1,FeeNormalizationReceiptV1
        material=self._fee_read(form_pin,'fee_form');parsed=FeeFactsFormV1.model_validate(material.data['form'])
        document=self._fee_read(ArtifactPinV1(**material.data['document']),'fee_document')
        if len({c.currency for c in parsed.components})!=1 or any(c.amount_basis!=parsed.amount_basis for c in parsed.components):
            raise ValueError('FEE_AGGREGATION_UNSUPPORTED')
        receipts=[]
        for c,raw in zip(parsed.components,material.data['form']['components'],strict=True):
            amount=None
            if c.raw_amount not in (None,'UNKNOWN'):
                if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',c.raw_amount) or len(c.raw_amount)>80:
                    raise ValueError('FEE_AMOUNT_LEXICAL_INVALID')
                with localcontext() as context:
                    context.prec=100
                    value=Decimal(c.raw_amount)*(1000000 if c.source_unit=='万元' else 100)
                    if value!=value.to_integral_value():raise ValueError('FEE_MINOR_PRECISION_UNSUPPORTED')
                    amount=int(value)
            money=FeeMoneyV1(currency=c.currency,amount_minor=amount,status='UNKNOWN' if amount is None else 'COMPLETE')
            receipts.append(FeeNormalizationReceiptV1(component_id=c.component_id,source_row_id=c.source_row_id,
                component_hash=digest(raw),source_hash=document.data['descriptor']['content_hash'],locator=c.locator,
                raw_amount=c.raw_amount,source_unit=c.source_unit,money=money))
        complete=all(r.money.status=='COMPLETE' for r in receipts)
        total=FeeMoneyV1(currency=receipts[0].money.currency,amount_minor=sum(r.money.amount_minor for r in receipts) if complete else None,
            status='COMPLETE' if complete else 'UNKNOWN')
        assertion=parsed.declared_total_minor
        status='ABSENT' if assertion is None else 'UNKNOWN' if not complete else 'MATCH' if assertion==total.amount_minor else 'MISMATCH'
        formation=FeeFactsFormationV1(producer_hash=raw_hash(Path(__file__).read_bytes()),source_kind=parsed.source_kind,source_context=deepcopy(parsed.source_context),component_assertions={c.component_id:{k:getattr(c,k) for k in ('service_period','cancellation','due_condition')} for c in parsed.components},amount_basis=parsed.amount_basis,receipts=tuple(receipts),total=total,
            caller_assertion=assertion,assertion_status=status)
        return self._fee_retain('fee_formation',form=form_pin.model_dump(mode='json'),document=material.data['document'],
            declaration=material.data['declaration'],formation=formation.model_dump(mode='json'))

    def consume_fee_facts(self, *, formation_pin, bindings, correction_parent=None):
        from services.source_consumer_verification import fee_facts_assessment,read_retained_fee_facts
        formed=self._fee_read(formation_pin,'fee_formation')
        parent=self.policy.read(correction_parent.artifact_ref,correction_parent.artifact_hash) if correction_parent else None
        if parent and (parent.data.get('lane')!='fee_facts_run' or parent.data.get('session_ref')!=self.session
                or parent.data.get('profile_id')!=self.profile_id or parent.data['formation']!=formation_pin.model_dump(mode='json')
                or parent.data['assessment']['coverage_status']!='DEFICIENT'):
            raise ValueError('FEE_SAME_INPUT_CORRECTION_REQUIRED')
        material=self._fee_read(ArtifactPinV1(**formed.data['form']),'fee_form')
        declaration=self._fee_read(ArtifactPinV1(**formed.data['declaration']),'fee_declaration')
        assessment=fee_facts_assessment(formed.data['formation'],declaration.data,bindings)
        record=self.record('consumer_run',dict(lane='fee_facts_run',profile_id=self.profile_id,formation=formation_pin.model_dump(mode='json'),
            assessment=assessment,bindings=bindings,correction_parent=correction_parent.model_dump(mode='json') if correction_parent else None))
        self.store.put(record)
        # The same pure closure checks used by a fresh retained reader apply warm.
        read_retained_fee_facts(store=self.store,**self.scope,session_ref=self.session,run_pin=self._fee_pin(record),
            expected_document_hash=self._fee_read(ArtifactPinV1(**material.data['document']),'fee_document').data['descriptor']['content_hash'],
            expected_verifier_hash=raw_hash(Path(__file__).with_name('source_consumer_verification.py').read_bytes()),expected_producer_hash=raw_hash(Path(__file__).read_bytes()))
        return record

    def _development_access(self):
        from contracts.source_change_consumer_verification import DEVELOPMENT_MEASURE_PROFILE
        self.policy._access()
        if self.profile_id!=DEVELOPMENT_MEASURE_PROFILE:raise ValueError('DEVELOPMENT_EXPLICIT_PROFILE_REQUIRED')

    def _development_read(self,pin,lane):
        self._development_access()
        if pin.artifact_kind!='policy_revision':raise ValueError('DEVELOPMENT_RECORD_KIND_MISMATCH')
        value=self.policy.read(pin.artifact_ref,pin.artifact_hash)
        if (value.kind!='consumer_candidate' or value.data.get('lane')!=lane or value.data.get('profile_id')!=self.profile_id
                or value.data.get('session_ref')!=self.session):raise PermissionError('DEVELOPMENT_RECORD_SCOPE_MISMATCH')
        return value

    def _development_retain(self,lane,**data):
        self._development_access()
        payload=dict(lane=lane,profile_id=self.profile_id,**data)
        record=self.record('consumer_candidate',payload)
        return self.policy._operation('development:'+lane+':'+digest(payload),digest(payload),record)

    def retain_development_fixture(self,*,artifacts,artifact_pin):
        """Retain exact governed JSON text bytes and hypothetical references."""
        from services.workspace_artifact_service import WorkspaceArtifactService
        from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
        from services.source_consumer_verification import development_fixture
        self._development_access()
        if type(artifacts) is not WorkspaceArtifactService or artifacts._store is not self.store:raise TypeError('DEVELOPMENT_GOVERNED_ARTIFACT_OWNER_REQUIRED')
        descriptor=artifacts.resolve(ctx=self.ctx,artifact_pin=artifact_pin)
        if descriptor.media_type!='text/plain' or descriptor.byte_size>1024**2:raise ValueError('DEVELOPMENT_JSON_TEXT_REQUIRED')
        with WorkspaceArtifactBlobStore(ctx=self.ctx,max_bytes=1024**2).exact_get(byte_owner_ref=descriptor.byte_owner_ref,expected_hash=descriptor.content_hash,expected_size=descriptor.byte_size) as stream:
            raw=stream.read()
        development_fixture(json.loads(raw))
        return self._development_retain('development_measure_fixture',descriptor=descriptor.model_dump(mode='json'),source_bytes=base64.b64encode(raw).decode(),evidence_role='HYPOTHETICAL_DEVELOPMENT_ONLY',authority='NOT_QUALIFIED')

    def _development_package(self,fixture):
        from services.source_consumer_verification import development_fixture
        raw=base64.b64decode(fixture.data['source_bytes'],validate=True)
        if raw_hash(raw)!=fixture.data['descriptor']['content_hash']:raise ValueError('DEVELOPMENT_FIXTURE_BYTES_MISMATCH')
        return development_fixture(json.loads(raw))

    def retain_development_input(self,*,fixture_pin,fixture_key,amendment_parent=None):
        from services.source_consumer_verification import development_input
        fixture=self._development_read(fixture_pin,'development_measure_fixture')
        package=self._development_package(fixture);raw=development_input(package,fixture_key)
        parent=self._development_read(amendment_parent,'development_measure_input') if amendment_parent else None
        if parent:
            if parent.data['fixture']!=fixture_pin.model_dump(mode='json') or parent.data['profile_hash']!=digest(package.profile):raise ValueError('DEVELOPMENT_AMENDMENT_FIXTURE_CHANGED')
            if {k:v for k,v in parent.data['input'].items() if k!='case_id'}=={k:v for k,v in raw.items() if k!='case_id'}:raise ValueError('DEVELOPMENT_AMENDMENT_INPUT_UNCHANGED')
        return self._development_retain('development_measure_input',fixture=fixture_pin.model_dump(mode='json'),fixture_key=fixture_key,input=deepcopy(raw),input_hash=digest(raw),profile_hash=digest(package.profile),amendment_parent=amendment_parent.model_dump(mode='json') if amendment_parent else None)

    def declare_development_inventory(self,*,input_pin):
        from services.source_consumer_verification import development_declaration
        material=self._development_read(input_pin,'development_measure_input')
        fixture=self._development_read(ArtifactPinV1(**material.data['fixture']),'development_measure_fixture')
        declaration=development_declaration(self._development_package(fixture),material.data['fixture_key'],fixture.data['descriptor']['content_hash'])
        return self._development_retain('development_measure_declaration',input=input_pin.model_dump(mode='json'),fixture=material.data['fixture'],**declaration)

    def form_development_measure(self,*,input_pin,declaration_pin):
        from services.procurement_development_measure import evaluate
        from contracts.source_change_consumer_verification import DevelopmentMeasureFormationV1,DevelopmentMeasureResultV1
        from services.source_consumer_verification import development_declaration,development_field_receipts,verify_development_formation
        material=self._development_read(input_pin,'development_measure_input')
        declaration=self._development_read(declaration_pin,'development_measure_declaration')
        fixture=self._development_read(ArtifactPinV1(**material.data['fixture']),'development_measure_fixture')
        package=self._development_package(fixture);key=material.data['fixture_key'];fixture_hash=fixture.data['descriptor']['content_hash']
        expected=development_declaration(package,key,fixture_hash)
        if declaration.data['input']!=input_pin.model_dump(mode='json') or declaration.data['fixture']!=material.data['fixture'] or any(declaration.data[k]!=v for k,v in expected.items()):raise ValueError('DEVELOPMENT_DECLARATION_INPUT_MISMATCH')
        details=evaluate(package.profile,material.data['input'])
        result=DevelopmentMeasureResultV1(status=details['status'],hypothetical_exposure_fen=details['exposure_fen'],details=details)
        core_hash=raw_hash(Path(__file__).with_name('procurement_development_measure.py').read_bytes())
        formed=DevelopmentMeasureFormationV1(input_hash=material.data['input_hash'],profile_hash=material.data['profile_hash'],core_hash=core_hash,result=result,receipts=development_field_receipts(package,key,fixture_hash))
        verify_development_formation(formed,package,key,fixture_hash,core_hash)
        return self._development_retain('development_measure_formation',input=input_pin.model_dump(mode='json'),declaration=declaration_pin.model_dump(mode='json'),formation=formed.model_dump(mode='json'))

    def consume_development_measure(self,*,formation_pin,bindings,correction_parent=None):
        from services.source_consumer_verification import development_assessment
        formed=self._development_read(formation_pin,'development_measure_formation')
        declaration=self._development_read(ArtifactPinV1(**formed.data['declaration']),'development_measure_declaration')
        parent=self.policy.read(correction_parent.artifact_ref,correction_parent.artifact_hash) if correction_parent else None
        if parent and (correction_parent.artifact_kind!='policy_revision' or parent.kind!='consumer_run' or parent.data.get('lane')!='development_measure_run'
                or parent.data.get('session_ref')!=self.session or parent.data.get('profile_id')!=self.profile_id
                or parent.data['formation']!=formation_pin.model_dump(mode='json') or parent.data['assessment']['coverage_status']!='DEFICIENT'):
            raise ValueError('DEVELOPMENT_SAME_INPUT_CORRECTION_REQUIRED')
        assessment=development_assessment(formed.data['formation'],declaration.data,bindings)
        record=self.record('consumer_run',dict(lane='development_measure_run',profile_id=self.profile_id,formation=formation_pin.model_dump(mode='json'),assessment=assessment,bindings=dict(bindings),correction_parent=correction_parent.model_dump(mode='json') if correction_parent else None))
        self.store.put(record)
        return self.policy.read(record.artifact_ref,record.artifact_hash)
