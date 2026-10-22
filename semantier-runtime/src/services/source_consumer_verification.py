"""Bounded, read-only native consumer verification and retained reduction.

Imports only contracts/hash utilities: reopening evidence cannot execute expense
policy, parser, OCR, compiler or model code. No general assertion language.
"""
import base64
from hashlib import sha256
import json
from datetime import date
from contracts.source_consumer_verification import (
    ConsumerVerificationV1, ConsumerTargetAssessmentV1, ConsumerRequirementAssessmentV1,
    ConsumerObservationV1,
)
from eos.artifact_hashing import content_hash


def digest(value):
    return 'sha256:' + content_hash(value)


def raw_hash(raw):
    return 'sha256:' + sha256(raw).hexdigest()


def _unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError('duplicate ' + label)


def verify_basis(v):
    _unique([r.requirement_id for r in v.requirements], 'requirement')
    _unique([r.target_id for r in v.targets], 'target')
    _unique([r.binding_id for r in v.bindings], 'binding')
    _unique([r.case_id for r in v.cases], 'case')
    _unique([(o.binding_id, o.case_id) for o in v.observations], 'observation')
    if v.source_hash != raw_hash(v.source_text.encode()):
        raise ValueError('source hash mismatch')
    if set(v.assets) != set(v.asset_hashes):
        raise ValueError('missing retained asset')
    assets = {k: base64.b64decode(b, validate=True) for k, b in v.assets.items()}
    if any(raw_hash(raw) != v.asset_hashes[k] for k, raw in assets.items()):
        raise ValueError('retained asset hash mismatch')
    if not {'prepared','source_pdf','correspondence','input_prerequisite','input_requirement','set_output','upload','invoice_pdf','inventory'}.issubset(assets):
        raise ValueError('missing retained consumer dependency')
    prepared = json.loads(assets['prepared'])
    from contracts.ontology_validation import CompiledRulePlanV1
    CompiledRulePlanV1.model_validate(prepared['plan'])
    upload = json.loads(assets['upload'])
    if upload['command']['byte_hash'] != raw_hash(assets['invoice_pdf']) or upload['command']['byte_length'] != len(assets['invoice_pdf']):
        raise ValueError('upload custody mismatch')
    if (v.plan.artifact_kind, v.plan.artifact_ref, v.plan.artifact_hash) != (
        'compiled_rule_plan', prepared['plan']['plan_ref'], prepared['plan']['plan_hash']):
        raise ValueError('consumer plan pin mismatch')
    if raw_hash(assets['source_pdf']) != v.pdf_hash or prepared['policy']['source_hash'] != v.pdf_hash:
        raise ValueError('source PDF correspondence mismatch')
    # Retain the actual prior normalization receipt, distinct Markdown/PDF hashes.
    correspondence = json.loads(assets['correspondence'])
    if not correspondence['full_body_match'] or correspondence['source_hash'] != v.source_hash or correspondence['pdf_hash'] != v.pdf_hash:
        raise ValueError('source occurrence correspondence mismatch')
    inventory = {i['id']:i for i in json.loads(assets['inventory'])['items']}
    requirements = {r.requirement_id:r for r in v.requirements}
    targets = {t.target_id:t for t in v.targets}
    declared = [tid for r in v.requirements for tid in r.target_ids]
    _unique(declared, 'declared target')
    if set(declared) != set(targets):
        raise ValueError('independent target coverage mismatch')
    for r in v.requirements:
        item = inventory.get(r.requirement_id)
        if item is None or tuple(item['competency_question_ids']) != r.competency_ids or not any(
            (s['char_start'],s['char_end'],s['byte_start'],s['byte_end'],s['quote']) ==
            (r.char_start,r.char_end,r.byte_start,r.byte_end,r.quote) for s in item['source_support']):
            raise ValueError('source inventory/CQ/occurrence mismatch')
        if v.source_text[r.char_start:r.char_end] != r.quote or v.source_text.encode()[r.byte_start:r.byte_end].decode() != r.quote:
            raise ValueError('source occurrence mismatch')
        if not any(s['id'] == r.requirement_id and s['quote'] == r.quote and tuple(s['competency_ids']) == r.competency_ids
                   for s in correspondence['occurrences']):
            raise ValueError('source requirement/CQ mismatch')
        if any(targets[t].requirement_id != r.requirement_id for t in r.target_ids):
            raise ValueError('target requirement mismatch')
    for t in v.targets:
        _unique([c.case_id for c in t.cases], 'expected case')
        if t.requirement_id not in requirements:
            raise ValueError('undeclared requirement')
        for c in t.cases:
            if t.kind == 'input_prerequisite' and c.expected not in ('ACCEPTED','REJECTED'):
                raise ValueError('inlet expectation requires acceptance/rejection')
            if t.kind == 'input_requirement' and c.expected not in ('SATISFIED','FAILED','UNKNOWN'):
                raise ValueError('input expectation requires native check outcome')
            if t.kind == 'set_output' and not isinstance(c.expected, tuple):
                raise ValueError('set output requires set-valued expectation')
            if isinstance(c.expected, tuple) and len(c.expected) != len(set(c.expected)):
                raise ValueError('duplicate expected list item')
    for b in v.bindings:
        t = targets.get(b.target_id)
        if t is None:
            raise ValueError('undeclared binding target')
        r = requirements[t.requirement_id]
        if (b.requirement_id, b.competency_ids, b.occurrence_ref, b.source_hash, b.kind, b.fact_ref) != (
            r.requirement_id, r.competency_ids, r.occurrence_ref, v.source_hash, t.kind, t.fact_ref):
            raise ValueError('binding source/target basis mismatch')
        if b.consumer_ref != b.kind or v.asset_hashes.get(b.consumer_ref) != b.consumer_hash:
            raise ValueError('consumer code pin mismatch')
        if b.kind == 'input_requirement':
            predicates = [p for rule in prepared['plan']['rules'] for p in rule.get('input_requirements') or []]
            if not any(p['predicate_id'] == b.selector and p['fact_ref'] == t.fact_ref and p['operator'] == 'EXISTS'
                       and prepared['policy']['field_bindings'].get(p['fact_ref']) == t.field for p in predicates):
                raise ValueError('native predicate/fact correspondence mismatch')
        elif b.selector != ('artifact_pin' if b.kind == 'input_prerequisite' else 'missing_fields'):
            raise ValueError('unsupported consumer selector')
    for c in v.cases:
        if date.fromisoformat(c.request['submitted_on']).isoformat() != c.request['submitted_on'] or not isinstance(c.request['applicant_fields'],dict):
            raise ValueError('invalid captured request')
        if set(c.request['applicant_fields']) - {'name','department','purpose'} or any(not isinstance(x,str) for x in c.request['applicant_fields'].values()):
            raise ValueError('invalid captured applicant fields')
        if c.session_ref != v.session_ref:
            raise ValueError('cross-session case')
        if c.http_status == 200:
            result = c.response
            # Expense snapshots use their own canonical JSON hash convention.
            basis = {k:val for k,val in result.items() if k not in {'result_ref','result_hash'}}
            encoded = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            if raw_hash(encoded) != result['result_hash'] or result['session_id'] != v.session_ref:
                raise ValueError('result identity/session mismatch')
            if result['rule_trace']['plan_hash'] != v.plan.artifact_hash or result['rule_trace']['plan_ref'] != v.plan.artifact_ref:
                raise ValueError('result plan mismatch')
            if result['policy']['source_hash'] != v.pdf_hash:
                raise ValueError('result source mismatch')
            if result['source_material_pin'] != c.request['artifact_pin'] or c.request['artifact_pin'] != upload['response']['artifact_pin']:
                raise ValueError('request/result artifact mismatch')
            if result['submitted_on'] != c.request['submitted_on'] or any(
                result['applicant_fields'][k]['value'] != val for k,val in c.request['applicant_fields'].items()):
                raise ValueError('request/result input mismatch')
            if result['invoice_source']['source_hash'] != raw_hash(assets['invoice_pdf']):
                raise ValueError('invoice source custody mismatch')
            if c.draft is not None and (c.draft['result_ref'],c.draft['result_hash'],c.draft['session_id']) != (result['result_ref'],result['result_hash'],v.session_ref):
                raise ValueError('draft result mismatch')
            for pin in result['pins']:
                if pin['ref'] not in c.dependencies or raw_hash(base64.b64decode(c.dependencies[pin['ref']], validate=True)) != pin['hash']:
                    raise ValueError('missing/tampered result dependency')
        elif c.http_status != 400 or c.current_result is not None or c.response != {'detail':'INVALID_DOCUMENTARY_REQUEST'} or set(c.request) != {'submitted_on','applicant_fields'}:
            raise ValueError('unsupported inlet evidence')
    bindings = {b.binding_id:b for b in v.bindings}
    cases = {c.case_id:c for c in v.cases}
    for o in v.observations:
        if o.binding_id not in bindings or o.case_id not in cases:
            raise ValueError('unbound observation')
        b, c = bindings[o.binding_id], cases[o.case_id]
        if o.evidence_hash != digest(c.model_dump(mode='json')) or o.actual != observe(b, c):
            raise ValueError('observation evidence mismatch')


def observe(binding, case):
    if binding.kind == 'input_prerequisite':
        return 'ACCEPTED' if case.http_status == 200 else 'REJECTED'
    if case.http_status != 200:
        return 'UNKNOWN'
    if binding.kind == 'set_output':
        return tuple(case.response['missing_fields'])
    checks = [p for rule in case.response['rule_trace']['rule_evaluations']
              for p in rule.get('input_requirement_results') or []
              if p['predicate_id'] == binding.selector and p['fact_ref'] == binding.fact_ref and p['operator'] == 'EXISTS']
    return checks[0]['result'] if len(checks) == 1 else 'UNKNOWN'


def assess_consumers(v):
    """AND across declared targets/cases, OR across complete binding alternatives."""
    assessments = []
    observations = {(o.binding_id,o.case_id):o for o in v.observations}
    for t in v.targets:
        bindings = [b for b in v.bindings if b.target_id == t.target_id]
        states = []
        for b in bindings:
            states.append(['NOT_RUN' if (o := observations.get((b.binding_id,c.case_id))) is None else
                'UNKNOWN' if o.actual == 'UNKNOWN' else
                'PASS' if (set(o.actual) == set(c.expected) and len(o.actual) == len(set(o.actual))
                           if isinstance(c.expected, tuple) and isinstance(o.actual, tuple) else o.actual == c.expected) else 'FAIL'
                for c in t.cases])
        passed = [b.binding_id for b,s in zip(bindings,states) if all(x == 'PASS' for x in s)]
        status = ('UNKNOWN' if v.eligibility != 'ELIGIBLE' else 'PASS' if passed else
                  'FAIL' if any('FAIL' in s for s in states) else 'UNKNOWN' if any('UNKNOWN' in s for s in states) else 'NOT_RUN')
        assessments.append(ConsumerTargetAssessmentV1(requirement_id=t.requirement_id, target_id=t.target_id,
            mapping_status='UNKNOWN' if v.eligibility != 'ELIGIBLE' else 'MAPPED' if bindings else 'MISSING',
            witness_status=status, supporting_binding_ids=tuple(sorted(set(passed))) if status == 'PASS' else (),
            case_ids=tuple(sorted({o.case_id for o in v.observations if o.binding_id in {b.binding_id for b in bindings}}))))
    requirements = []
    for r in v.requirements:
        states = [a.witness_status for a in assessments if a.target_id in r.target_ids]
        coverage = 'UNKNOWN' if v.eligibility != 'ELIGIBLE' else 'FULL' if all(s == 'PASS' for s in states) else 'PARTIAL' if 'PASS' in states else 'NONE'
        requirements.append(ConsumerRequirementAssessmentV1(requirement_id=r.requirement_id, scope=r.scope, coverage=coverage))
    return tuple(assessments), tuple(requirements)


def seal_verification(**values):
    """Capture actual projections, then validate the retained native representation."""
    probe = ConsumerVerificationV1.model_construct(**values, target_assessments=(), requirement_assessments=())
    targets, requirements = assess_consumers(probe)
    return ConsumerVerificationV1(**values, target_assessments=targets, requirement_assessments=requirements)


def read_retained_consumers(*, store, organization_id, workspace_id, run_pin,
                            expected_plan, source_hash, session_ref):
    """Reopen pinned native quality evidence without business recomputation."""
    from contracts.ontology_quality import QualityEvaluationRunV1
    if run_pin.artifact_kind != 'quality_evaluation_run':
        raise ValueError('consumer replay run kind mismatch')
    run = store.get(organization_id=organization_id, workspace_id=workspace_id,
                    artifact_ref=run_pin.artifact_ref, contract_type=QualityEvaluationRunV1)
    if run is None or run.pin() != run_pin or run.subject != expected_plan:
        raise ValueError('consumer replay missing or mismatched run')
    results = [r for r in run.results if r.requirement_id == 'source_consumer_binding']
    if len(results) != 1 or results[0].consumer_verification is None:
        raise ValueError('consumer replay evidence unavailable')
    v = results[0].consumer_verification
    if (v.organization_id,v.workspace_id,v.source_hash,v.session_ref,v.plan) != (
        organization_id,workspace_id,source_hash,session_ref,expected_plan):
        raise ValueError('consumer replay scope/basis mismatch')
    return run


# The source-evolution reducer below inspects retained typed bytes only. It never
# imports compilation, business consumers, extraction, provider or parser owners.
def _revision_maps(basis):
    return ({r.logical_id:r for r in basis.requirements},
            {(d.requirement_id,d.target_id):d for d in basis.dependencies})


def source_instance_ref(source_hash, logical_id, start, end, quote):
    return 'requirement-instance:' + digest([source_hash,logical_id,start,end,quote])[7:]


def _meaning(basis, rid):
    reqs,deps=_revision_maps(basis)
    r=reqs[rid]
    return digest(dict(quote=r.quote,state=r.state,targets=r.target_ids,depends_on=r.depends_on,
        dependencies=[d.model_dump(mode='json') for k,d in sorted(deps.items()) if k[0]==rid]))


def source_equivalence(predecessor, successor, rid):
    if _meaning(predecessor,rid)!=_meaning(successor,rid):
        raise ValueError('SOURCE_EQUIVALENCE_MEANING_MISMATCH')
    return digest([predecessor.source.artifact_hash,successor.source.artifact_hash,
                   predecessor.session_ref,rid,_meaning(predecessor,rid)])


def verify_source_basis(b):
    if getattr(b,'schema_version',None) in ('source_change_basis.v3','source_change_basis.v4','source_change_basis.v5'):
        return verify_procurement_basis(b)
    from datetime import date
    if b.source.kind not in ('source','draft') or b.source.data.get('session_ref')!=b.session_ref:
        raise ValueError('SOURCE_REVISION_SESSION_MISMATCH')
    if date.fromisoformat(b.source.data['effective_on']).isoformat()!=b.source.data['effective_on']:
        raise ValueError('SOURCE_EFFECTIVITY_INVALID')
    scope=(b.source.organization_id,b.source.workspace_id)
    if (b.candidate.organization_id,b.candidate.workspace_id)!=scope or (b.plan.organization_id,b.plan.workspace_id)!=scope:
        raise ValueError('SOURCE_NATIVE_SCOPE_MISMATCH')
    if (b.candidate.source_identity_ref,b.candidate.source_hash)!=(b.source.artifact_ref,b.source.data['source_hash']):
        raise ValueError('SOURCE_CANDIDATE_PIN_MISMATCH')
    if (b.plan.candidate_set_ref,b.plan.candidate_set_hash)!=(b.candidate.candidate_set_ref,b.candidate.candidate_set_hash):
        raise ValueError('SOURCE_PLAN_CANDIDATE_MISMATCH')
    ir=b.canonical_source
    if ir.source_identity_ref!=b.source.artifact_ref or digest(ir.model_dump(mode='json',by_alias=True))!=b.candidate.canonical_source_ir_hash:
        raise ValueError('SOURCE_CANONICAL_PIN_MISMATCH')
    if b.candidate.canonical_source_ir_ref!='canonical-source-ir:'+b.candidate.canonical_source_ir_hash[7:]:raise ValueError('SOURCE_CANONICAL_REF_MISMATCH')
    blocks=b.source.data['blocks']
    if len(ir.blocks)!=len(blocks) or len(ir.source_map.mappings)!=len(blocks):raise ValueError('SOURCE_CANONICAL_BLOCKS_MISSING')
    from contracts.canonical_source_ir import source_block_identity,content_hash as block_hash
    for index,(block,native,mapping) in enumerate(zip(ir.blocks,blocks,ir.source_map.mappings)):
        if len(mapping.source_locators)!=1:raise ValueError('SOURCE_CANONICAL_MAPPING_AMBIGUOUS')
        loc=mapping.source_locators[0]
        if (block.ordinal,block.content,block.content_hash)!=(index,native['text'],block_hash(native['text'])) or b.source.data['markdown'][native['start']:native['end']]!=block.content:
            raise ValueError('SOURCE_CANONICAL_BLOCK_MISMATCH')
        if (loc.source_identity_ref,loc.source_hash,loc.section_id)!=(b.source.artifact_ref,b.source.data['source_hash'],native['block_id']) or block.block_id!=source_block_identity(b.source.data['source_hash'],loc,'policy_clause') or mapping.block_ids!=[block.block_id]:
            raise ValueError('SOURCE_CANONICAL_LOCATOR_MISMATCH')
    for rule in b.plan.rules:
        if any(p.operator=='DATE_ON_OR_AFTER' and p.ir_node_ref.startswith('scope.') and p.expected_value!=b.source.data['effective_on'] for p in rule.predicates):
            raise ValueError('SOURCE_EFFECTIVITY_PLAN_MISMATCH')
    for frame in b.candidate.frames:
        if frame.frame_hash!=digest(frame.model_dump(mode='json',exclude={'frame_hash'})) or frame.evidence_refs!=(b.source.artifact_ref,):raise ValueError('SOURCE_FRAME_HASH_OR_EVIDENCE_MISMATCH')
    cp=b.candidate.model_dump(mode='json',exclude={'candidate_set_ref','candidate_set_hash'})
    if b.candidate.candidate_set_hash!=digest(cp):raise ValueError('SOURCE_CANDIDATE_BYTES_MISMATCH')
    pp=b.plan.model_dump(mode='json'); retained_plan={k:pp[k] for k in ('candidate_set_ref','candidate_set_hash','validation_contract_ref','validation_contract_hash','compiler_ref','compiler_hash','semantic_profile','rules','teaching_candidate')}
    if 'precedences' in pp:retained_plan['precedences']=pp['precedences']
    # Compiler's production-time identity uses Python isoformat (+00:00).
    retained_plan['compiled_at']=b.plan.compiled_at.isoformat()
    if b.plan.plan_hash!=digest(retained_plan):raise ValueError('SOURCE_PLAN_BYTES_MISMATCH')
    required={'input_requirement','set_output','calendar','compiler','input_prerequisite'}
    if set(b.consumer_assets)!=required:raise ValueError('SOURCE_CONSUMER_ASSETS_MISSING')
    code={k:raw_hash(base64.b64decode(v,validate=True)) for k,v in b.consumer_assets.items()}
    if b.plan.compiler_ref!='compiler:source-consumer-revision:v1' or b.plan.compiler_hash!=code['compiler']:raise ValueError('SOURCE_COMPILER_IDENTITY_MISMATCH')
    reqs,deps=_revision_maps(b)
    _unique([r.logical_id for r in b.requirements],'source requirement')
    _unique([r.instance_ref for r in b.requirements],'source instance')
    _unique([tid for r in b.requirements for tid in r.target_ids],'source target identity')
    _unique([(d.requirement_id,d.target_id) for d in b.dependencies],'source dependency')
    _unique([c.case_id for c in b.inputs],'source input')
    _unique([s.binding_id for s in b.supports],'source support')
    cases={c.case_id for c in b.inputs}
    for c in b.inputs:
        date.fromisoformat(c.submitted_on)
        if set(c.applicant_fields)-{'name','department','purpose'}:raise ValueError('SOURCE_INPUT_FIELD_INVALID')
    declared=set()
    for r in b.requirements:
        if b.source.data['markdown'][r.char_start:r.char_end]!=r.quote or r.instance_ref!=source_instance_ref(b.source.data['source_hash'],r.logical_id,r.char_start,r.char_end,r.quote):
            raise ValueError('SOURCE_INSTANCE_SPAN_MISMATCH')
        _unique(r.target_ids,'source target')
        declared.update((r.logical_id,t) for t in r.target_ids)
        if set(r.depends_on)-set(reqs) or r.logical_id in r.depends_on:raise ValueError('SOURCE_DEPENDENCY_ENDPOINT_MISSING')
    if b.dependency_inventory_complete and declared!=set(deps):raise ValueError('SOURCE_DEPENDENCY_INVENTORY_MISMATCH')
    if set(deps)-declared or any(set(d.case_ids)-cases for d in b.dependencies):raise ValueError('SOURCE_CASE_DEPENDENCY_MISMATCH')
    if getattr(b,"schema_version",None)=="source_change_basis.v2":required_inventory(b)
    return code


def source_change_selection(old,new,equivalences):
    """Discover bounded direct/transitive impact, independent of support survival."""
    before,_=_revision_maps(old);after,_=_revision_maps(new)
    stronger=getattr(new,"schema_version",None) in ("source_change_basis.v2","source_change_basis.v3","source_change_basis.v4","source_change_basis.v5")
    if stronger:verify_inventory_transition(old,new)
    for rid,witness in equivalences.items():
        if rid not in before or rid not in after or witness!=source_equivalence(old,new,rid):
            raise ValueError('SOURCE_EQUIVALENCE_WITNESS_MISMATCH')
    changed={rid for rid in set(before)|set(after) if rid not in before or rid not in after or _meaning(old,rid)!=_meaning(new,rid)}
    possible={rid for rid in set(before)&set(after)-changed if old.source!=new.source and rid not in equivalences}
    direct=set(changed);affected=changed|possible
    while True:
        trans={r.logical_id for r in (*old.requirements,*new.requirements,*_required_view(old).requirements,*_required_view(new).requirements) if set(r.depends_on)&affected}
        if trans<=affected:break
        affected|=trans
    deps={(d.requirement_id,d.target_id,d.consumer,d.selector,d.case_ids) for d in (*old.dependencies,*new.dependencies) if d.requirement_id in affected}
    selected=tuple(sorted({c for _,_,_,_,ids in deps for c in ids}))
    if old.source==new.source:
        selected=tuple(sorted({cid for d in new.dependencies for cid in d.case_ids}))
    if stronger:
        inventory=required_inventory(new)
        if inventory is not None:selected=tuple(sorted({c for d in inventory.dependencies for c in d.case_ids}))
    return direct,possible,affected,selected


def _source_dependency_available(b,d):
    if d.consumer=='input_prerequisite':return d.selector=='financial_ref'
    if d.consumer=='set_output':return d.selector=='missing_fields'
    checks=[p for rule in b.plan.rules for p in ((rule.input_requirements or ()) if d.consumer=='input_requirement' else rule.predicates)]
    operator='EXISTS' if d.consumer=='input_requirement' else 'CALENDAR_YEARS_AFTER'
    return sum(p.predicate_id==d.selector and p.fact_ref==d.fact_ref and p.operator==operator for p in checks)==1


def _source_dependency_executable(b,d):
    checks=[p for rule in b.plan.rules for p in (rule.input_requirements or ())]
    if d.consumer=='input_prerequisite':return d.selector=='financial_ref'
    if d.consumer=='input_requirement':
        return sum(p.predicate_id==d.selector and p.fact_ref==d.fact_ref and p.operator=='EXISTS' for p in checks)==1
    if d.consumer=='calendar':
        return sum(p.predicate_id==d.selector and p.fact_ref==d.fact_ref and p.operator=='CALENDAR_YEARS_AFTER' and p.expected_value==d.operand for rule in b.plan.rules for p in rule.predicates)==1
    fields=tuple(sorted({b.adapter_policy.field_bindings[p.fact_ref] for p in checks if p.operator=='EXISTS' and b.adapter_policy.field_bindings.get(p.fact_ref) in ('name','department','purpose')}))
    return d.selector=='missing_fields' and isinstance(d.operand,tuple) and fields==tuple(sorted(d.operand))


def source_change_findings(old,new,equivalences):
    from contracts.source_change_consumer_verification import SourceImpactFindingV1
    reqs,deps=_revision_maps(new);before,prior_deps=_revision_maps(old)
    direct,possible,affected,_=source_change_selection(old,new,equivalences)
    code=verify_source_basis(new)
    findings=[]
    for rid,r in sorted(reqs.items()):
        for tid in r.target_ids:
            d=deps.get((rid,tid));bindings=[s for s in new.supports if s.target_id==tid]
            exact=[s for s in bindings if d and s.dependency_hash==digest(d.model_dump(mode='json')) and s.consumer_hash==code[d.consumer] and ((s.instance_ref==r.instance_ref and s.source_hash==new.source.data['source_hash']) or (rid in equivalences and rid in before and s.instance_ref==before[rid].instance_ref and s.source_hash==old.source.data['source_hash'])) and _source_dependency_executable(new,d)]
            if r.state=='UNRESOLVED' or d is None or not _source_dependency_available(new,d) or rid in possible or not new.dependency_inventory_complete:
                state,reason='UNKNOWN','unreviewed semantics, equivalence or dependency inventory unavailable'
            elif len(bindings)>1:
                state,reason='CONFLICT','multiple support alternatives are not qualified for this revision'
            elif exact:
                state,reason='SUPPORTED','exact source instance, dependency, consumer and operand'
            elif bindings:
                state,reason='STALE','support pins a different source instance, dependency, operand or consumer'
            else:
                state,reason='MISSING','declared target has no surviving support'
            findings.append(SourceImpactFindingV1(requirement_id=rid,instance_ref=r.instance_ref,target_id=tid,status=state,reason=reason,
                binding_ids=tuple(s.binding_id for s in bindings),case_ids=tuple(sorted(d.case_ids)) if d else ()))
    for (rid,tid),d in sorted(prior_deps.items()):
        if rid not in reqs or tid not in reqs[rid].target_ids:
            findings.append(SourceImpactFindingV1(requirement_id=rid,instance_ref=before[rid].instance_ref,target_id=tid,status='WITHDRAWN',
                reason='explicit successor target withdrawal; old support cannot qualify the successor',case_ids=tuple(sorted(d.case_ids))))
    return tuple(findings)


def source_change_projection(old,new,equivalences):
    from contracts.governed_delta import (ArtifactVersionPinV1,DeltaLens,DeltaLensResultV1,DeltaOperationV1,DeltaTargetType,AffectedArtifactV1)
    from services.governed_delta_service import GovernedDeltaService
    direct,possible,affected,selected=source_change_selection(old,new,equivalences)
    def pin(b):
        return ArtifactVersionPinV1(artifact_type='source',artifact_id='source-consumer-policy',version_id=b.source.artifact_ref,
            content_hash=b.source.data['source_hash'],owner_ref='policy_revision_service',organization_id=b.source.organization_id,
            workspace_id=b.source.workspace_id,effective_from=b.source.data['effective_on'])
    ops=tuple(DeltaOperationV1(operation_id='change:'+rid,lens='STRUCTURAL',operation_kind='ADD' if rid not in {r.logical_id for r in old.requirements} else 'REMOVE' if rid not in {r.logical_id for r in new.requirements} else 'MODIFY',subject_ref=rid,
        changed_fields=('source_instance',),materiality='MATERIAL',evidence_refs=(new.source.artifact_ref,)) for rid in sorted(direct))
    lenses={l:DeltaLensResultV1(lens=l,status='AVAILABLE' if l==DeltaLens.STRUCTURAL else 'UNAVAILABLE',
        operations=ops if l==DeltaLens.STRUCTURAL else ()) for l in DeltaLens}
    env=GovernedDeltaService.build_envelope(delta_id='source-delta:'+digest([old.source.artifact_hash,new.source.artifact_hash])[7:],
        target_type=DeltaTargetType.POLICY,predecessor=pin(old),successor_candidate=pin(new),lens_results=lenses)
    rows=[]
    projection_old,projection_new=_required_view(old),_required_view(new)
    reqs,_=_revision_maps(new);prior,_=_revision_maps(old)
    reqs.update(_revision_maps(projection_new)[0]);prior.update(_revision_maps(projection_old)[0])
    for rid in sorted(affected):
        req=reqs.get(rid,prior.get(rid));status='POSSIBLE' if rid in possible else 'DIRECT' if rid in direct else 'TRANSITIVE'
        ds={digest(d.model_dump(mode='json')):d for d in (*projection_old.dependencies,*projection_new.dependencies) if d.requirement_id==rid}
        missing_targets=set(req.target_ids)-{d.target_id for d in new.dependencies if d.requirement_id==rid}
        if not ds or missing_targets or not new.dependency_inventory_complete:
            rows.append(AffectedArtifactV1(artifact_ref=req.instance_ref,relation='BINDS',impact_kind='EVIDENCE',impact_status='UNAVAILABLE',basis_refs=(rid,)))
        for dh,d in sorted(ds.items()):
            bound_req=reqs[rid] if d in projection_new.dependencies else prior[rid]
            witness=(bound_req.instance_ref,dh,d.consumer,d.selector)
            rows.append(AffectedArtifactV1(artifact_ref='dependency:'+dh[7:],relation='BINDS',impact_kind='EVIDENCE',impact_status=status,basis_refs=witness))
            rows.append(AffectedArtifactV1(artifact_ref=d.consumer+':'+d.target_id,relation='EXECUTES',impact_kind='EXECUTION_BINDING',impact_status=status,basis_refs=witness))
            for cid in sorted(d.case_ids):
                rows.append(AffectedArtifactV1(artifact_ref='case:'+cid,relation='VERIFIES',impact_kind='EVIDENCE',impact_status=status,basis_refs=witness))
    return env,GovernedDeltaService.build_impact_projection(envelope=env,affected_artifacts=rows),selected


def verify_source_change(v):
    if v.schema_version in ('source_change_consumer_assessment.v3','source_change_consumer_assessment.v4','source_change_consumer_assessment.v5'):
        return verify_procurement_assessment(v)
    from datetime import timedelta
    for b in (v.predecessor,v.successor):verify_source_basis(b)
    old,new=v.predecessor,v.successor
    if (old.source.organization_id,old.source.workspace_id,old.session_ref)!=(new.source.organization_id,new.source.workspace_id,new.session_ref):
        raise ValueError('SOURCE_CHANGE_SCOPE_MISMATCH')
    if old.source!=new.source and new.source.data.get('parent')!={'artifact_ref':old.source.artifact_ref,'artifact_hash':old.source.artifact_hash}:
        raise ValueError('SOURCE_CHANGE_ANCESTRY_MISMATCH')
    env,impact,selected=source_change_projection(old,new,v.equivalences)
    if (v.delta,v.impact,v.selected_case_ids)!=(env,impact,selected):raise ValueError('SOURCE_IMPACT_RETAINED_MISMATCH')
    stronger=v.schema_version=='source_change_consumer_assessment.v2'
    findings=required_inventory_findings(old,new,v.equivalences,v.observations) if stronger else source_change_findings(old,new,v.equivalences)
    if v.findings!=findings:raise ValueError('SOURCE_FINDING_RETAINED_MISMATCH')
    if stronger and v.coverage_status!=required_coverage_status(findings):raise ValueError('REQUIRED_COVERAGE_RETAINED_MISMATCH')
    if v.produced_at.utcoffset()!=timedelta(0):raise ValueError('SOURCE_CHANGE_UTC_REQUIRED')
    _unique([o.case_id for o in v.observations],'source observation')
    if tuple(sorted(o.case_id for o in v.observations))!=selected:raise ValueError('SOURCE_SELECTED_EXECUTION_MISMATCH')
    cases={c.case_id:c for c in new.inputs}
    for o in v.observations:
        if o.observed_at.utcoffset()!=timedelta(0) or o.observed_at>v.produced_at:raise ValueError('SOURCE_OBSERVATION_TIME_MISMATCH')
        if o.input_hash!=digest(cases[o.case_id].model_dump(mode='json')) or o.plan_hash!=new.plan.plan_hash or o.response_hash!=digest(o.response):
            raise ValueError('SOURCE_OBSERVATION_BASIS_MISMATCH')
        if cases[o.case_id].invoice_attached:
            trace=o.response.get('rule_trace',{})
            if (trace.get('plan_ref'),trace.get('plan_hash'))!=(new.plan.plan_ref,new.plan.plan_hash):raise ValueError('SOURCE_NATIVE_TRACE_MISMATCH')
            values=dict(invoice_date=cases[o.case_id].invoice_date,submitted_on=cases[o.case_id].submitted_on,is_draft=False,
                **{k:cases[o.case_id].applicant_fields.get(k,'').strip() for k in ('name','department','purpose')})
            facts={ref:values[field] for ref,field in new.adapter_policy.field_bindings.items() if values[field] is not None}
            if o.response.get('observed_facts')!=facts or trace.get('observed_facts_hash')!=digest(facts):raise ValueError('SOURCE_NATIVE_INPUT_TRACE_MISMATCH')
            from contracts.ontology_validation import CandidateRuleEvaluationV1
            evaluations=tuple(CandidateRuleEvaluationV1.model_validate(row) for row in trace.get('rule_evaluations',()))
            if tuple(row.rule_id for row in evaluations)!=tuple(rule.rule_id for rule in new.plan.rules):raise ValueError('SOURCE_NATIVE_RULE_INVENTORY_MISMATCH')
            if stronger:verify_native_receipt_inventory(new,trace)
        elif o.response.get('input_status')!='REJECTED':raise ValueError('SOURCE_INLET_REJECTION_MISMATCH')
    return v


def read_retained_source_change(*,store,organization_id,workspace_id,session_ref,run_pin,expected_source_hash,expected_evaluator_hash,expected_verifier_hash=None):
    from contracts.policy_revision import PolicyRevisionRecordV1
    from contracts.source_change_consumer_verification import SourceChangeAssessmentV1
    if run_pin.artifact_kind!='policy_revision':raise ValueError('SOURCE_REPLAY_KIND_MISMATCH')
    record=store.get(organization_id=organization_id,workspace_id=workspace_id,artifact_ref=run_pin.artifact_ref,contract_type=PolicyRevisionRecordV1)
    if record is None or record.artifact_hash!=run_pin.artifact_hash or record.kind!='consumer_run' or record.data.get('lane')!='source_change':
        raise ValueError('SOURCE_REPLAY_RECORD_MISMATCH')
    v=source_change_assessment_type(record.data['assessment']).model_validate(record.data['assessment'])
    if (v.successor.source.organization_id,v.successor.source.workspace_id,v.successor.session_ref)!=(organization_id,workspace_id,session_ref):raise ValueError('SOURCE_REPLAY_SCOPE_MISMATCH')
    if v.schema_version in ('source_change_consumer_assessment.v3','source_change_consumer_assessment.v4','source_change_consumer_assessment.v5'):
        return read_procurement_closure(store=store,scope=dict(organization_id=organization_id,workspace_id=workspace_id),record=record,assessment=v,expected_source_hash=expected_source_hash,expected_evaluator_hash=expected_evaluator_hash,expected_verifier_hash=expected_verifier_hash)
    if v.successor.source.data['source_hash']!=expected_source_hash or raw_hash(base64.b64decode(v.successor.consumer_assets['calendar']))!=expected_evaluator_hash:
        raise ValueError('SOURCE_REPLAY_CURRENT_BASIS_MISMATCH')
    # Resolve each retained source and correction witness through the existing
    # store. An embedded hash alone cannot stand in for a missing join.
    seen=set()
    current=v
    while True:
        for basis in (current.predecessor,current.successor):
            source=store.get(organization_id=organization_id,workspace_id=workspace_id,
                artifact_ref=basis.source.artifact_ref,contract_type=PolicyRevisionRecordV1)
            if source!=basis.source:raise ValueError('SOURCE_REPLAY_SOURCE_JOIN_MISMATCH')
            reference=getattr(basis,'required_inventory',None)
            if reference is not None:
                _read_required_inventory_chain(store,organization_id,workspace_id,reference,current.produced_at)
        parent=current.correction_parent
        if parent is None:break
        if parent.artifact_ref in seen or parent.artifact_kind!='policy_revision':raise ValueError('SOURCE_REPLAY_CORRECTION_CYCLE')
        seen.add(parent.artifact_ref)
        previous=store.get(organization_id=organization_id,workspace_id=workspace_id,
            artifact_ref=parent.artifact_ref,contract_type=PolicyRevisionRecordV1)
        if previous is None or previous.artifact_hash!=parent.artifact_hash or previous.kind!='consumer_run':
            raise ValueError('SOURCE_REPLAY_CORRECTION_JOIN_MISSING')
        prior=source_change_assessment_type(previous.data['assessment']).model_validate(previous.data['assessment'])
        if prior.predecessor!=current.predecessor or prior.successor.source!=current.successor.source or prior.produced_at>=current.produced_at:
            raise ValueError('SOURCE_REPLAY_CORRECTION_ANCESTRY_MISMATCH')
        current=prior
    return v


def verify_required_declaration(inventory, source):
    """Integrity of an exposed finite declaration; no semantic review claim."""
    from datetime import date
    if (inventory.source.artifact_kind,inventory.source.artifact_ref,inventory.source.artifact_hash)!=(
            'policy_revision',source.artifact_ref,source.artifact_hash):
        raise ValueError('REQUIRED_INVENTORY_SOURCE_PIN_MISMATCH')
    if (inventory.organization_id,inventory.workspace_id,inventory.session_ref,inventory.source_hash,inventory.effective_on)!=(
            source.organization_id,source.workspace_id,source.data.get('session_ref'),source.data['source_hash'],source.data['effective_on']):
        raise ValueError('REQUIRED_INVENTORY_SOURCE_SCOPE_MISMATCH')
    if date.fromisoformat(inventory.effective_on).isoformat()!=inventory.effective_on:raise ValueError('REQUIRED_INVENTORY_EFFECTIVITY_INVALID')
    req={r.logical_id:r for r in inventory.requirements};cases={i.case_id for i in inventory.inputs}
    _unique([r.logical_id for r in inventory.requirements],'required requirement')
    _unique([r.instance_ref for r in inventory.requirements],'required instance')
    _unique([t for r in inventory.requirements for t in r.target_ids],'required target')
    _unique([i.case_id for i in inventory.inputs],'required input')
    keys=[(d.requirement_id,d.target_id) for d in inventory.dependencies]
    _unique(keys,'required dependency')
    if set(keys)!={(r.logical_id,t) for r in inventory.requirements for t in r.target_ids}:raise ValueError('REQUIRED_INVENTORY_DEPENDENCY_MISSING')
    for r in inventory.requirements:
        if source.data['markdown'][r.char_start:r.char_end]!=r.quote or r.instance_ref!=source_instance_ref(inventory.source_hash,r.logical_id,r.char_start,r.char_end,r.quote):raise ValueError('REQUIRED_INVENTORY_SPAN_MISMATCH')
        if set(r.depends_on)-set(req) or r.logical_id in r.depends_on:raise ValueError('REQUIRED_INVENTORY_EDGE_MISMATCH')
    if any(set(d.case_ids)-cases for d in inventory.dependencies):raise ValueError('REQUIRED_INVENTORY_CASE_MISSING')
    if set(inventory.withdrawals.values())-{r.instance_ref for r in inventory.requirements}:raise ValueError('REQUIRED_WITHDRAWAL_EVIDENCE_MISSING')
    if inventory.schema_version=='source_required_inventory.v4':
        expected={(case.case_id,c['component_id'],'component_receipt') for case in inventory.inputs for c in case.form['fee_components']}
        expected.update((case.case_id,'native_amount','formation_handoff') for case in inventory.inputs)
        actual=[(cid,d.selector,d.consumer) for d in inventory.dependencies if d.consumer!='predicate' for cid in d.case_ids]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('AMOUNT_REQUIRED_FORMATION_INVENTORY_MISMATCH')



def required_inventory(b):
    """Resolve embedded bytes against the separately supplied immutable pin."""
    from contracts.source_change_consumer_verification import SourceRequiredInventoryV1
    from contracts.policy_revision import PolicyRevisionRecordV1
    pin=getattr(b,'required_inventory_pin',None);record=getattr(b,'required_inventory',None)
    if pin is None and record is None:return None
    if pin is None or record is None:raise ValueError('REQUIRED_INVENTORY_PIN_OR_BYTES_MISSING')
    record=PolicyRevisionRecordV1.model_validate(record.model_dump(mode='json'))
    if (pin.artifact_kind,pin.artifact_ref,pin.artifact_hash)!=('policy_revision',record.artifact_ref,record.artifact_hash) or record.kind!='consumer_candidate' or record.data.get('lane')!='source_required_inventory':raise ValueError('REQUIRED_INVENTORY_PIN_MISMATCH')
    inventory=required_inventory_type(record.data['inventory']).model_validate(record.data['inventory'])
    if record.data.get('inventory_hash')!=digest(inventory.model_dump(mode='json')):raise ValueError('REQUIRED_INVENTORY_BYTES_MISMATCH')
    if (record.organization_id,record.workspace_id,record.data.get('session_ref'))!=(b.source.organization_id,b.source.workspace_id,b.session_ref):raise ValueError('REQUIRED_INVENTORY_RECORD_SCOPE_MISMATCH')
    verify_required_declaration(inventory,b.source)
    if (inventory.validation_basis.artifact_ref,inventory.validation_basis.artifact_hash)!=(b.plan.validation_contract_ref,b.plan.validation_contract_hash):raise ValueError('REQUIRED_INVENTORY_VALIDATION_BASIS_MISMATCH')
    cases={c.case_id:c for c in b.inputs}
    if any(cases.get(c.case_id)!=c for c in inventory.inputs):raise ValueError('REQUIRED_INVENTORY_INPUT_MISMATCH')
    return inventory


def _required_view(b):
    inventory=required_inventory(b)
    return b.model_copy(update={'requirements':inventory.requirements,'dependencies':inventory.dependencies,'inputs':inventory.inputs}) if inventory else b


def verify_inventory_transition(old,new):
    before,after=required_inventory(old),required_inventory(new)
    if before is None or after is None:return
    if old.source==new.source:
        if old.required_inventory_pin!=new.required_inventory_pin:raise ValueError('REQUIRED_INVENTORY_SAME_SOURCE_REPLACED')
        return
    if after.predecessor_pin!=old.required_inventory_pin:raise ValueError('REQUIRED_INVENTORY_LINEAGE_MISMATCH')
    removed={t for r in before.requirements for t in r.target_ids}-{t for r in after.requirements for t in r.target_ids}
    if removed!=set(after.withdrawals):raise ValueError('REQUIRED_WITHDRAWAL_DECLARATION_MISMATCH')
    prior={r.logical_id:r for r in before.requirements}
    witnesses={r.instance_ref:r for r in after.requirements}
    for target,ref in after.withdrawals.items():
        witness=witnesses[ref]
        if before.source_hash==after.source_hash or (witness.logical_id in prior and witness.quote==prior[witness.logical_id].quote):raise ValueError('REQUIRED_WITHDRAWAL_SOURCE_WITNESS_MISMATCH')


def _receipt_deficiency(b,d,observations):
    if getattr(b,'domain',None)=='development_measure':
        return development_receipt_deficiency(b,d)
    if getattr(b,'domain',None)=='fee_facts':
        return fee_facts_receipt_deficiency(b,d)
    if getattr(b,'domain',None)=='procurement':
        return procurement_receipt_deficiency(b,d,observations)
    """Join actual evaluated receipts; FAIL/UNKNOWN are execution, not MATCH."""
    observed={o.case_id:o for o in observations};cases={c.case_id:c for c in b.inputs}
    for cid in d.case_ids:
        if cid not in observed:return 'missing actual case receipt: '+cid
        response=observed[cid].response;case=cases[cid]
        rows=response.get('rule_trace',{}).get('rule_evaluations',())
        checks=[p for r in rows for p in r.get('input_requirement_results',())]
        if d.consumer=='input_prerequisite':
            if not case.invoice_attached:
                if response!=dict(input_status='REJECTED',result_created=False,owner='contracts.expense_reimbursement.ExpenseReimbursementRequest'):return 'missing native inlet rejection receipt: '+cid
            elif not rows:return 'missing accepted input consumer receipt: '+cid
        elif d.consumer in ('input_requirement','calendar'):
            receipts=checks if d.consumer=='input_requirement' else [p for r in rows for p in r.get('predicate_results',())]
            matches=[p for p in receipts if (p['predicate_id'],p['fact_ref'],p['operator'])==(d.selector,d.fact_ref,'EXISTS' if d.consumer=='input_requirement' else 'CALENDAR_YEARS_AFTER')]
            if len(matches)!=1:return 'missing evaluated '+d.consumer+' receipt: '+cid
            p=matches[0]
            if p['observed_value']!=response.get('observed_facts',{}).get(d.fact_ref):return 'wrong observed fact receipt: '+cid
            if d.consumer=='calendar' and (p['expected_value']!=d.operand or p.get('comparison_observed_value')!=response.get('observed_facts',{}).get(p.get('comparison_fact_ref'))):return 'wrong calendar operand receipt: '+cid
        else:
            fields=set(d.operand or ());field_receipts={b.adapter_policy.field_bindings.get(p['fact_ref']):p for p in checks if p['operator']=='EXISTS' and b.adapter_policy.field_bindings.get(p['fact_ref']) in fields}
            output=response.get(d.selector)
            if set(field_receipts)!=fields or not isinstance(output,list) or len(output)!=len(set(output)):return 'missing scoped set-output receipt: '+cid
            if set(output)&fields!={f for f,p in field_receipts.items() if p['result']=='FAILED'}:return 'mismatched scoped set-output receipt: '+cid
    return None


def required_inventory_findings(old,new,equivalences,observations):
    from contracts.source_change_consumer_verification import SourceImpactFindingV1
    inventory=required_inventory(new)
    legacy=procurement_support_findings(new) if getattr(new,'domain',None)=='procurement' else source_change_findings(old,new,equivalences)
    if inventory is None:
        return tuple(f.model_copy(update={'status':'UNKNOWN','reason':'required inventory unavailable; stronger coverage not assessed'}) for f in legacy)
    return compare_required_inventory(new,inventory,legacy,observations)


def compare_required_inventory(new,inventory,legacy,observations):
    """One fixed-inventory comparison for both explicit domain lanes."""
    from contracts.source_change_consumer_verification import SourceImpactFindingV1
    actual,dependencies=_revision_maps(new);legacy={(f.requirement_id,f.target_id):f for f in legacy};findings=[]
    expected_deps={(d.requirement_id,d.target_id):d for d in inventory.dependencies}
    for r in sorted(inventory.requirements,key=lambda r:r.logical_id):
        candidate=actual.get(r.logical_id)
        for tid in r.target_ids:
            key=(r.logical_id,tid);expected=expected_deps[key];d=dependencies.get(key);prior=legacy.get(key)
            reason=None
            if candidate is None:reason='required requirement omitted'
            elif tid not in candidate.target_ids:reason='required target omitted'
            elif d is None:reason='required dependency omitted'
            elif set(r.depends_on)-set(candidate.depends_on):reason='required transitive support edge omitted'
            elif candidate.instance_ref!=r.instance_ref:reason='required source instance differs'
            else:
                # Check the candidate's claimed case edge before the independently
                # required one. An inlet rejection cannot discharge a field check.
                reason=_receipt_deficiency(new,d,observations)
                left=d.model_dump(mode='json');right=expected.model_dump(mode='json')
                left['case_ids']=sorted(left['case_ids']);right['case_ids']=sorted(right['case_ids'])
                if reason is None and left!=right:reason='required consumer or case association differs'
                if reason is None:reason=_receipt_deficiency(new,expected,observations)
            if reason:
                findings.append(SourceImpactFindingV1(requirement_id=r.logical_id,instance_ref=r.instance_ref,target_id=tid,status='MISSING',reason=reason,case_ids=tuple(sorted(expected.case_ids))))
            elif prior:findings.append(prior)
    # Only the separately declared successor inventory can justify withdrawal.
    for key,f in legacy.items():
        if key not in expected_deps and f.status=='WITHDRAWN':findings.append(f)
    return tuple(findings)


def required_coverage_status(findings):
    if any(f.status in ('MISSING','STALE','CONFLICT') for f in findings):return 'DEFICIENT'
    if not findings or any(f.status=='UNKNOWN' for f in findings):return 'UNKNOWN'
    return 'SUPPORTED'


def source_change_assessment_type(data):
    from contracts.source_change_consumer_verification import SourceChangeAssessmentV1,SourceChangeAssessmentV2
    version=data.get('schema_version')
    if version=='source_change_consumer_assessment.v5':
        from contracts.source_change_consumer_verification import SourceChangeAssessmentV5
        return SourceChangeAssessmentV5
    if version=='source_change_consumer_assessment.v1':return SourceChangeAssessmentV1
    if version=='source_change_consumer_assessment.v2':return SourceChangeAssessmentV2
    if version=='source_change_consumer_assessment.v4':
        from contracts.source_change_consumer_verification import SourceChangeAssessmentV4
        return SourceChangeAssessmentV4
    if version=='source_change_consumer_assessment.v3':
        from contracts.source_change_consumer_verification import SourceChangeAssessmentV3
        return SourceChangeAssessmentV3
    raise ValueError('SOURCE_ASSESSMENT_VERSION_UNSUPPORTED')


def verify_native_receipt_inventory(b,trace):
    """Verify executed trace membership and operands, without running predicates."""
    rows=trace['rule_evaluations']
    for rule,row in zip(b.plan.rules,rows,strict=True):
        for field,compiled in [('predicate_results',rule.predicates),('input_requirement_results',rule.input_requirements or ()),('exception_results',rule.exception_predicates)]:
            receipts=row.get(field,())
            expected=[(p.predicate_id,p.operator,p.fact_ref,p.model_dump(mode='json')['expected_value'],p.ir_node_ref,tuple(p.evidence_refs),p.comparison_fact_ref) for p in compiled]
            actual=[(p['predicate_id'],p['operator'],p['fact_ref'],p.get('expected_value'),p['ir_node_ref'],tuple(p['evidence_refs']),p.get('comparison_fact_ref')) for p in receipts]
            if expected!=actual:raise ValueError('SOURCE_NATIVE_RECEIPT_INVENTORY_MISMATCH')


def _read_required_inventory_chain(store,organization_id,workspace_id,reference,produced_at):
    from contracts.policy_revision import PolicyRevisionRecordV1
    from contracts.source_change_consumer_verification import SourceRequiredInventoryV1
    seen=set()
    while reference is not None:
        retained=store.get(organization_id=organization_id,workspace_id=workspace_id,artifact_ref=reference.artifact_ref,contract_type=PolicyRevisionRecordV1)
        if retained!=reference or retained.produced_at>produced_at:raise ValueError('REQUIRED_INVENTORY_RETAINED_JOIN_MISSING')
        if retained.artifact_ref in seen:raise ValueError('REQUIRED_INVENTORY_LINEAGE_CYCLE')
        seen.add(retained.artifact_ref)
        inventory=required_inventory_type(retained.data['inventory']).model_validate(retained.data['inventory'])
        source=store.get(organization_id=organization_id,workspace_id=workspace_id,artifact_ref=inventory.source.artifact_ref,contract_type=PolicyRevisionRecordV1)
        if source is None:raise ValueError('REQUIRED_INVENTORY_SOURCE_JOIN_MISSING')
        verify_required_declaration(inventory,source)
        parent=inventory.predecessor_pin
        if parent is None:break
        reference=store.get(organization_id=organization_id,workspace_id=workspace_id,artifact_ref=parent.artifact_ref,contract_type=PolicyRevisionRecordV1)
        if reference is None or parent.artifact_kind!='policy_revision' or reference.artifact_hash!=parent.artifact_hash or reference.kind!='consumer_candidate' or reference.data.get('lane')!='source_required_inventory':raise ValueError('REQUIRED_INVENTORY_PARENT_JOIN_MISSING')
        previous=required_inventory_type(reference.data['inventory']).model_validate(reference.data['inventory'])
        if previous.session_ref!=inventory.session_ref or source.data['parent']!={'artifact_ref':previous.source.artifact_ref,'artifact_hash':previous.source.artifact_hash}:raise ValueError('REQUIRED_INVENTORY_SOURCE_LINEAGE_MISMATCH')


def canonical_source_for_record(source):
    from contracts.canonical_source_ir import (CanonicalSourceDocument,SourceLocator,SourceBlock,
        SourceMap,SourceMapping,Transformation,ParserTrace,source_block_identity,content_hash)
    blocks=[];mappings=[]
    procurement=source.data.get('domain')=='procurement'
    provider='source_consumer_verification.procurement_source_content' if procurement else 'policy_revision_service.markdown_blocks'
    for index,block in enumerate(source.data['blocks']):
        locator=SourceLocator(source_identity_ref=block.get('supplemental_source_ref',source.artifact_ref),source_hash=block.get('supplemental_source_hash',source.data['source_hash']),
            media_type='text/markdown' if block.get('supplemental_source_ref') else 'application/json' if procurement else 'text/markdown',section_id=block['block_id'],native_metadata=dict(start=block['start'],end=block['end'],**(dict(source_document_hash=block.get('supplemental_document_hash',source.data['source_document_hash']),**{k:block[k] for k in ('source_locator','raw_byte_start','raw_byte_end')}) if procurement else {})))
        bid=source_block_identity(source.data['source_hash'],locator,'policy_clause')
        blocks.append(SourceBlock(block_id=bid,block_type='paragraph',ordinal=index,content=block['text'],content_hash=content_hash(block['text'])))
        mappings.append(SourceMapping(source_locators=[locator],block_ids=[bid],transformation=Transformation(kind='one_to_one',transform_ref=provider)))
    return CanonicalSourceDocument(source_identity_ref=source.artifact_ref,blocks=blocks,source_map=SourceMap(mappings=mappings),
        parser_trace=ParserTrace(provider=provider,adapter_version='1',canonical_ir_version='canonical_source_ir.v1'))


def procurement_source_content(raw):
    """Exact controlled JSON-section mapping, not extraction or interpretation."""
    if len(raw)>100_000:raise ValueError('PROCUREMENT_SOURCE_SIZE_INVALID')
    sections=json.loads(raw)['source_sections']
    if len(sections)!=7:raise ValueError('PROCUREMENT_SOURCE_SECTIONS_INVALID')
    _unique([s['section_id'] for s in sections],'procurement section')
    text='';blocks=[]
    for index,section in enumerate(sections):
        quote=section['text'];prefix=section['section_id']+'\n'
        start=len(text)+len(prefix);end=start+len(quote)
        native=json.dumps(quote,ensure_ascii=False)[1:-1].encode()
        if raw.count(native)!=1:raise ValueError('PROCUREMENT_SOURCE_LOCATOR_AMBIGUOUS')
        raw_start=raw.index(native)
        blocks.append(dict(block_id=section['section_id'],text=quote,start=start,end=end,
            block_hash=raw_hash(quote.encode()),source_locator=f'/source_sections/{index}/text',
            raw_byte_start=raw_start,raw_byte_end=raw_start+len(native)))
        text+=prefix+quote+'\n\n'
    return text,blocks


def required_inventory_type(data):
    from contracts.source_change_consumer_verification import SourceRequiredInventoryV1,ProcurementRequiredInventoryV1
    if data.get('schema_version')=='source_required_inventory.v4':
        from contracts.source_change_consumer_verification import ProcurementRequiredInventoryV3
        return ProcurementRequiredInventoryV3
    if data.get('schema_version')=='source_required_inventory.v3':
        from contracts.source_change_consumer_verification import ProcurementRequiredInventoryV2
        return ProcurementRequiredInventoryV2
    if data.get('schema_version')=='source_required_inventory.v1':return SourceRequiredInventoryV1
    if data.get('schema_version')=='source_required_inventory.v2' and data.get('domain')=='procurement':return ProcurementRequiredInventoryV1
    raise ValueError('REQUIRED_INVENTORY_VERSION_UNSUPPORTED')


def _procurement_checks(b, d):
    """Resolve a bounded declared selector to its actual compiled role."""
    rules=[r for r in b.plan.rules if r.rule_id==d.rule_id]
    if len(rules)!=1:return ()
    rule=rules[0]
    if d.role=='precedence':
        edges=[p for p in (b.plan.precedences or ()) if p.ir_node_ref==d.precedence_ref and p.lower_priority_ref==d.rule_id]
        return tuple(p for edge in edges for p in edge.conditions if p.predicate_id==d.selector)
    checks={'condition':rule.predicates,'input_requirement':rule.input_requirements or (), 'exception':rule.exception_predicates}[d.role]
    return tuple(p for p in checks if p.predicate_id==d.selector)


def _procurement_metadata(p):
    return (p.predicate_id,p.operator,p.fact_ref,p.model_dump(mode='json')['expected_value'],p.ir_node_ref,tuple(p.evidence_refs),p.comparison_fact_ref)


def procurement_support_findings(b):
    from contracts.source_change_consumer_verification import SourceImpactFindingV1
    code=verify_source_basis(b);findings=[]
    for r in b.requirements:
        for tid in r.target_ids:
            ds=[d for d in b.dependencies if d.requirement_id==r.logical_id and d.target_id==tid]
            bindings=[s for s in b.supports if s.target_id==tid]
            exact=[]
            for d in ds:
                checks=_procurement_checks(b,d) if d.consumer=='predicate' else ()
                executable=len(checks)==1 and (checks[0].operator,checks[0].fact_ref,checks[0].model_dump(mode='json')['expected_value'],checks[0].ir_node_ref)==(d.operator,d.fact_ref,d.model_dump(mode='json')['operand'],d.frame_ref)
                if d.consumer!='predicate':executable=component_consumer_valid(d) if getattr(b,'profile_id',None)=='controlled-procurement.component-amount.v1' else incident_consumer_valid(d)
                consumer_code=code['predicate'] if d.consumer=='predicate' else code['verifier']
                exact.extend(s for s in bindings if executable and s.instance_ref==r.instance_ref and s.source_hash==b.source.data['source_hash'] and s.consumer_hash==consumer_code and s.dependency_hash==digest(d.model_dump(mode='json')))
            status='UNKNOWN' if r.state=='UNRESOLVED' or not b.dependency_inventory_complete else 'CONFLICT' if len(bindings)>1 else 'SUPPORTED' if exact else 'STALE' if bindings else 'MISSING'
            findings.append(SourceImpactFindingV1(requirement_id=r.logical_id,instance_ref=r.instance_ref,target_id=tid,status=status,reason='exact source, native selector and typed dependency' if status=='SUPPORTED' else 'procurement support unavailable, ambiguous or stale',binding_ids=tuple(s.binding_id for s in bindings),case_ids=tuple(sorted(ds[0].case_ids)) if ds else ()))
    return tuple(findings)


def procurement_receipt_deficiency(b,d,observations):
    if d.consumer in ('component_receipt','formation_handoff'):
        return component_receipt_deficiency(b,d,observations)
    cases={o.case_id:o for o in observations}
    if d.consumer!='predicate':
        if not incident_consumer_valid(d):return 'missing incident consumer: '+d.selector
        for cid in d.case_ids:
            if cid not in cases or 'incident' not in cases[cid].response:return 'missing incident receipt: '+cid
            receipt=cases[cid].response['incident']
            if d.consumer=='normalization' and receipt['normalization']['consumer_ref']!='procurement.incident.strip.v1':return 'normalization receipt missing: '+cid
            if d.consumer=='qualification' and receipt['qualification']['consumer_ref']!='procurement.incident.qualify.v1':return 'qualification receipt missing: '+cid
            if d.consumer=='projection' and d.selector not in receipt['outputs']:return 'projection receipt missing: '+cid
        return None
    checks=_procurement_checks(b,d)
    if len(checks)!=1:return 'missing compiled '+d.role+' consumer: '+d.selector
    for cid in d.case_ids:
        if cid not in cases:return 'missing actual case receipt: '+cid
        bundle=cases[cid].response['bundle']
        rows=[r for r in bundle['candidate_rule_evaluations'] if r['rule_id']==d.rule_id]
        field={'condition':'predicate_results','input_requirement':'input_requirement_results','exception':'exception_results','precedence':'exception_results'}[d.role]
        receipts=[p for r in rows for p in (r.get(field) or ()) if p['predicate_id']==d.selector]
        if len(receipts)!=1:return 'missing evaluated '+d.role+' receipt: '+cid
        # Precedence has its own explicit runtime trace in addition to the LE receipt.
        if d.role=='precedence':
            relation=next(e for e in b.plan.precedences if e.ir_node_ref==d.precedence_ref)
            if not any(c.get('ir_node_ref')==d.precedence_ref for r in rows for c in (r.get('precedence_results') or ())):
                return 'missing evaluated precedence receipt: '+cid
            if relation.higher_priority_kind!='exception':return 'unsupported procurement precedence role: '+cid
    return None


def _verify_procurement_plan(b):
    """Check frame→plan projection/roles, never compute a predicate outcome."""
    from contracts.ontology_validation import CompiledPredicateV1
    frames={f.frame_id:f for f in b.candidate.frames}
    rule_frames=[f for f in b.candidate.frames if f.frame_kind=='rule']
    if tuple(f.frame_id for f in rule_frames)!=tuple(r.rule_id for r in b.plan.rules):raise ValueError('PROCUREMENT_PLAN_RULE_JOIN_MISMATCH')
    def predicate(owner,p,strict=None):
        return CompiledPredicateV1(predicate_id=p.predicate_id,operator=p.operator,fact_ref=p.fact_ref,expected_value=p.expected_value,ir_node_ref=owner,evidence_refs=p.evidence_refs,comparison_fact_ref=p.comparison_fact_ref,strict_operands=strict)
    for f,r in zip(rule_frames,b.plan.rules,strict=True):
        if f.epistemic_status!='FOUND':raise ValueError('PROCUREMENT_UNRESOLVED_NATIVE_RULE')
        if f.rule_effects:raise ValueError('PROCUREMENT_EFFECT_NOT_IN_BOUNDED_SLICE')
        conditions=[predicate(f.frame_id,p) for p in f.conditions];inputs=[]
        for ref in f.scope_refs:
            frame=frames[ref]
            if frame.applies_to_rule_ref!=f.frame_id:raise ValueError('PROCUREMENT_SCOPE_BINDING_MISMATCH')
            conditions.extend(predicate(ref,p) for p in frame.conditions)
        for ref in f.temporal_condition_refs:
            frame=frames[ref]
            for op,value in [('DATE_ON_OR_AFTER',frame.effective_from),('DATE_ON_OR_BEFORE',frame.effective_until)]:
                pid='predicate:temporal:'+content_hash(dict(rule=f.frame_id,frame=ref,operator=op))
                conditions.append(CompiledPredicateV1(predicate_id=pid,operator=op,fact_ref=frame.event_time_ref,expected_value=value,ir_node_ref=ref,evidence_refs=frame.evidence_refs))
        for ref in f.qualification_refs:inputs.extend(predicate(ref,p) for p in frames[ref].conditions or ())
        for frame in b.candidate.frames:
            if frame.frame_kind=='entity_distinction':
                for effect in frame.rule_effects or ():
                    if effect.rule_ref==f.frame_id:
                        (inputs if effect.effect=='input_requirement' else conditions).extend(predicate(frame.frame_id,p,True) for p in effect.conditions)
        if any(frames[ref].applies_to_rule_ref!=f.frame_id for ref in f.exception_refs):raise ValueError('PROCUREMENT_EXCEPTION_BINDING_MISMATCH')
        exceptions=tuple(predicate(ref,p) for ref in f.exception_refs for p in frames[ref].conditions)
        if [[p.model_dump(mode='json') for p in values] for values in (conditions,inputs,exceptions)]!=[[p.model_dump(mode='json') for p in values] for values in (r.predicates,r.input_requirements or (),r.exception_predicates)]:raise ValueError('PROCUREMENT_PLAN_PREDICATE_JOIN_MISMATCH')
        if (r.ir_node_ref,r.evidence_refs,r.interpretation_required,r.source_modality)!=(f.frame_id,f.evidence_refs,f.interpretation_required,None):raise ValueError('PROCUREMENT_PLAN_RULE_METADATA_MISMATCH')
        groups=tuple((e.ir_node_ref,e.predicate_ids,e.evidence_refs) for e in (r.exception_groups or ()))
        expected=tuple((ref,tuple(p.predicate_id for p in frames[ref].conditions),frames[ref].evidence_refs) for ref in f.exception_refs)
        if groups!=expected:raise ValueError('PROCUREMENT_EXCEPTION_GROUP_JOIN_MISMATCH')
    predicates={p.predicate_id:(f,p) for f in b.candidate.frames for p in (getattr(f,'conditions',()) or ())}
    edges=[f for f in b.candidate.frames if f.frame_kind=='precedence']
    if len(edges)!=len(b.plan.precedences or ()):raise ValueError('PROCUREMENT_PRECEDENCE_JOIN_MISMATCH')
    for f,e in zip(edges,b.plan.precedences or (),strict=True):
        expected=tuple(predicate(predicates[ref][0].frame_id,predicates[ref][1]) for ref in f.condition_refs)
        if (e.ir_node_ref,e.higher_priority_ref,e.lower_priority_ref,e.higher_priority_kind,e.conditions,e.evidence_refs)!=(f.frame_id,f.higher_priority_ref,f.lower_priority_ref,frames[f.higher_priority_ref].frame_kind,expected,f.evidence_refs):raise ValueError('PROCUREMENT_PRECEDENCE_JOIN_MISMATCH')


def verify_procurement_basis(b):
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from datetime import timedelta
    source=Record.model_validate(b.source.model_dump(mode='json'))
    if source.data.get('profile_id')!=getattr(b,'profile_id',None):raise ValueError('INCIDENT_PROFILE_VERSION_MISMATCH')
    if source.data.get('domain')!='procurement' or source.data.get('session_ref')!=b.session_ref:raise ValueError('PROCUREMENT_SOURCE_SCOPE_MISMATCH')
    raw=base64.b64decode(source.data['original_json_b64'],validate=True)
    text,blocks=procurement_profile_content(source.data) if source.data.get('profile_id')=='controlled-procurement.incident-evidence.v1.1' else procurement_source_content(raw)
    if raw_hash(raw)!=source.data['source_document_hash'] or (text,blocks)!=(source.data['markdown'],source.data['blocks']):raise ValueError('PROCUREMENT_SOURCE_MAPPING_MISMATCH')
    if canonical_source_for_record(source)!=b.canonical_source:raise ValueError('PROCUREMENT_CANONICAL_MAPPING_MISMATCH')
    canonical_hash=digest(b.canonical_source.model_dump(mode='json',by_alias=True))
    c,p=b.candidate,b.plan
    if c.completeness_state!='COMPLETE' or c.teaching_candidate is not True or p.teaching_candidate is not True:raise ValueError('PROCUREMENT_NATIVE_DEVELOPMENT_BASIS_REQUIRED')
    scope=(source.organization_id,source.workspace_id)
    if (c.organization_id,c.workspace_id)!=(scope) or (p.organization_id,p.workspace_id)!=scope:raise ValueError('PROCUREMENT_NATIVE_SCOPE_MISMATCH')
    if (c.source_identity_ref,c.source_hash,c.canonical_source_ir_ref,c.canonical_source_ir_hash)!=(source.artifact_ref,source.data['source_hash'],'canonical-source-ir:'+canonical_hash[7:],canonical_hash):raise ValueError('PROCUREMENT_SOURCE_CANDIDATE_JOIN_MISMATCH')
    if c.candidate_set_hash!=digest(c.model_dump(mode='json',exclude={'candidate_set_ref','candidate_set_hash'})) or c.candidate_set_ref!='normative-candidate:'+c.candidate_set_hash[7:]:raise ValueError('PROCUREMENT_CANDIDATE_BYTES_MISMATCH')
    evidence={source.artifact_ref+'#'+block['block_id'] for block in blocks}
    for f in c.frames:
        if f.frame_kind not in ('rule','scope','qualification','entity_distinction','exception','precedence','temporal_condition'):raise ValueError('PROCUREMENT_FRAME_NOT_IN_BOUNDED_SLICE')
        if f.frame_hash!=digest(f.model_dump(mode='json',exclude={'frame_hash'})):raise ValueError('PROCUREMENT_FRAME_HASH_MISMATCH')
        if set(f.source_region_refs)-{x['block_id'] for x in blocks} or set(f.evidence_refs)!={source.artifact_ref+'#'+x for x in f.source_region_refs}:raise ValueError('PROCUREMENT_FRAME_SOURCE_MISMATCH')
        for q in getattr(f,'conditions',()) or ():
            if q.operator not in ('EQ','IN','GE','GT','LE','LT','EXISTS','DATE_ON_OR_AFTER','DATE_ON_OR_BEFORE'):raise ValueError('PROCUREMENT_OPERATOR_NOT_IN_BOUNDED_SLICE')
            if not set(q.evidence_refs)<=evidence:raise ValueError('PROCUREMENT_PREDICATE_EVIDENCE_MISMATCH')
    if (p.candidate_set_ref,p.candidate_set_hash)!=(c.candidate_set_ref,c.candidate_set_hash):raise ValueError('PROCUREMENT_PLAN_CANDIDATE_MISMATCH')
    payload=p.model_dump(mode='json');keys=('candidate_set_ref','candidate_set_hash','validation_contract_ref','validation_contract_hash','compiler_ref','compiler_hash','semantic_profile','rules')
    basis={k:payload[k] for k in keys}
    for k in ('precedences','teaching_candidate'):
        if k in payload:basis[k]=payload[k]
    basis['compiled_at']=p.compiled_at.isoformat()
    if p.plan_hash!=digest(basis) or p.plan_ref!='compiled-rule-plan:'+p.plan_hash[7:]:raise ValueError('PROCUREMENT_PLAN_BYTES_MISMATCH')
    if set(b.consumer_assets)!=({'compiler','predicate','verifier','input_contract','product'} if getattr(b,'profile_id',None) else {'compiler','predicate','verifier'}):raise ValueError('PROCUREMENT_CONSUMER_ASSETS_MISSING')
    code={k:raw_hash(base64.b64decode(v,validate=True)) for k,v in b.consumer_assets.items()}
    if p.compiler_ref!='compiler:source-consumer-procurement:v1' or p.compiler_hash!=code['compiler']:raise ValueError('PROCUREMENT_COMPILER_IDENTITY_MISMATCH')
    context=Record.model_validate(b.context.model_dump(mode='json'))
    if (context.organization_id,context.workspace_id,context.data.get('session_ref'),context.data.get('source'))!=(source.organization_id,source.workspace_id,b.session_ref,dict(artifact_ref=source.artifact_ref,artifact_hash=source.artifact_hash)) or context.data.get('lane')!='procurement_context' or context.data.get('domain')!='procurement' or context.data.get('schema_status')!='DECLARED_NOT_ADMITTED':raise ValueError('PROCUREMENT_CONTEXT_SOURCE_MISMATCH')
    competency=context.data['competency'];schema=context.data['native_frame_schema']
    from contracts.ontology_building import CanonicalEvidenceBoundaryV1,CompetencyContractV1
    boundary=CanonicalEvidenceBoundaryV1.model_validate(context.data['boundary'])
    cq=CompetencyContractV1.model_validate(competency)
    bp=boundary.model_dump(mode='json',exclude={'boundary_ref','boundary_hash','schema_version'})
    bp.update(target_time=boundary.target_time.isoformat(),created_at=boundary.created_at.isoformat())
    h=digest(bp)
    if (boundary.boundary_ref,boundary.boundary_hash)!=('boundary:'+h[7:],h) or (boundary.organization_id,boundary.workspace_id,boundary.source_identity_ref,boundary.source_hash,boundary.canonical_source_ir_hash)!=(source.organization_id,source.workspace_id,source.artifact_ref,source.data['source_hash'],canonical_hash):raise ValueError('PROCUREMENT_BOUNDARY_CONTEXT_MISMATCH')
    cp={k:competency[k] for k in ('organization_id','workspace_id','boundary_ref','boundary_hash','graph_intent_brief_ref','graph_intent_brief_hash','intended_domain','questions')}
    h=digest(cp)
    if (cq.competency_contract_ref,cq.competency_contract_hash)!=('competency-contract:'+h[7:],h) or (cq.boundary_ref,cq.boundary_hash)!=(boundary.boundary_ref,boundary.boundary_hash):raise ValueError('PROCUREMENT_COMPETENCY_CONTEXT_HASH_MISMATCH')
    if (cq.graph_intent_brief_ref,cq.graph_intent_brief_hash)!=(source.artifact_ref+'#declared-source-questions',digest(context.data['questions'])) or tuple(q.competency_id for q in cq.questions)!=tuple(q['competency_id'] for q in context.data['questions']) or any(q.review_state!='unresolved' or not set(q.evidence_refs)<=evidence for q in cq.questions):raise ValueError('PROCUREMENT_COMPETENCY_SOURCE_MISMATCH')
    if (c.competency_contract_ref,c.competency_contract_hash,c.normative_schema_ref,c.normative_schema_hash)!=(competency['competency_contract_ref'],competency['competency_contract_hash'],context.artifact_ref+'#declared-frame-schema',digest(schema)):raise ValueError('PROCUREMENT_CONTEXT_CANDIDATE_MISMATCH')
    if (p.validation_contract_ref,p.validation_contract_hash)!=(context.artifact_ref,context.artifact_hash):raise ValueError('PROCUREMENT_VALIDATION_CONTEXT_MISMATCH')
    if any(tuple(r.competency_refs)!=tuple(q['competency_id'] for q in competency['questions']) for r in p.rules):raise ValueError('PROCUREMENT_COMPETENCY_PLAN_MISMATCH')
    times=(source.produced_at,context.produced_at,c.compiled_at,p.compiled_at)
    if any(t.utcoffset()!=timedelta(0) for t in times) or list(times)!=sorted(times):raise ValueError('PROCUREMENT_PRODUCTION_TIME_MISMATCH')
    _unique([r.logical_id for r in b.requirements],'procurement requirement')
    _unique([r.instance_ref for r in b.requirements],'procurement instance')
    _unique([t for r in b.requirements for t in r.target_ids],'procurement target')
    _unique([(d.requirement_id,d.target_id) for d in b.dependencies],'procurement dependency')
    _unique([i.case_id for i in b.inputs],'procurement case')
    _unique([s.binding_id for s in b.supports],'procurement binding')
    reqs,deps=_revision_maps(b)
    for r in b.requirements:
        if text[r.char_start:r.char_end]!=r.quote or r.instance_ref!=source_instance_ref(source.data['source_hash'],r.logical_id,r.char_start,r.char_end,r.quote):raise ValueError('PROCUREMENT_REQUIREMENT_SPAN_MISMATCH')
        if set(r.depends_on)-set(reqs):raise ValueError('PROCUREMENT_REQUIREMENT_EDGE_ENDPOINT_MISSING')
    if set(deps)-{(r.logical_id,t) for r in b.requirements for t in r.target_ids}:raise ValueError('PROCUREMENT_TARGET_ENDPOINT_MISSING')
    if any(set(d.case_ids)-{i.case_id for i in b.inputs} for d in b.dependencies):raise ValueError('PROCUREMENT_CASE_ENDPOINT_MISSING')
    if any(f.effective_from!=source.data['effective_on'] for f in c.frames if f.frame_kind=='temporal_condition'):raise ValueError('PROCUREMENT_SOURCE_EFFECTIVITY_MISMATCH')
    _verify_procurement_plan(b)
    inventory=required_inventory(b)
    if inventory is not None:
        if not (b.context.produced_at<=b.required_inventory.produced_at<=c.compiled_at):raise ValueError('PROCUREMENT_DECLARATION_TIME_MISMATCH')
        if inventory.domain!='procurement' or (inventory.source_document_hash,inventory.canonical_hash)!=(source.data['source_document_hash'],canonical_hash):raise ValueError('PROCUREMENT_DECLARATION_MAPPING_MISMATCH')
        if c.profile_hash!=digest(inventory.model_dump(mode='json')):raise ValueError('PROCUREMENT_DECLARATION_CANDIDATE_MISMATCH')
    if getattr(b,'profile_id',None)=='controlled-procurement.component-amount.v1':
        if context.data.get('profile_id')!=b.profile_id:raise ValueError('AMOUNT_CONTEXT_PROFILE_MISMATCH')
        verify_component_basis(b,code)
    if getattr(b,'profile_id',None)=='controlled-procurement.incident-evidence.v1.1':
        if b.profile_id!=source.data.get('profile_id') or context.data.get('profile_id')!=b.profile_id:raise ValueError('INCIDENT_PROFILE_BASIS_MISMATCH')
        for raw_record in context.data['incident_witnesses']:
            record=Record.model_validate(raw_record)
            if (record.organization_id,record.workspace_id,record.data.get('session_ref'))!=(source.organization_id,source.workspace_id,b.session_ref) or record.data.get('lane')!='incident_witness':raise ValueError('INCIDENT_ROSTER_SCOPE_MISMATCH')
    return code


def verify_procurement_observation(b,o,produced_at):
    from datetime import timedelta
    if o.observed_at.utcoffset()!=timedelta(0):raise ValueError('PROCUREMENT_OBSERVATION_UTC_REQUIRED')
    from contracts.ontology_paired_evaluation import ObservedFactsV1
    from contracts.ontology_validation import RuntimeSemanticEvaluationBundleV1
    case=next(c for c in b.inputs if c.case_id==o.case_id)
    facts=ObservedFactsV1.model_validate(o.response['facts'])
    bundle=RuntimeSemanticEvaluationBundleV1.model_validate(o.response['bundle'])
    scope=(b.source.organization_id,b.source.workspace_id)
    if (facts.organization_id,facts.workspace_id)!=scope or (bundle.organization_id,bundle.workspace_id)!=scope:raise ValueError('PROCUREMENT_RECEIPT_SCOPE_MISMATCH')
    expected_facts=component_native_projection(case,b)[0] if getattr(b,'profile_id',None)=='controlled-procurement.component-amount.v1' else incident_projection(case,b)[0] if getattr(b,'profile_id',None)=='controlled-procurement.incident-evidence.v1.1' else case.native_facts()
    if getattr(b,'profile_id',None)=='controlled-procurement.component-amount.v1':
        if o.response.get('formation')!=component_native_projection(case,b)[1].model_dump(mode='json'):raise ValueError('AMOUNT_NATIVE_HANDOFF_MISMATCH')
    if facts.facts!=expected_facts or bundle.observed_facts!=facts.facts or o.input_hash!=digest(case.model_dump(mode='json')):raise ValueError('PROCUREMENT_FACT_RECEIPT_MISMATCH')
    if (o.plan_hash,bundle.plan_ref,bundle.plan_hash)!=(b.plan.plan_hash,b.plan.plan_ref,b.plan.plan_hash) or o.response_hash!=digest(o.response):raise ValueError('PROCUREMENT_RECEIPT_PLAN_MISMATCH')
    if bundle.execution_epoch_ref!='procurement-case:'+digest([b.source.artifact_ref,b.session_ref,case.case_id,facts.facts_hash])[7:] or bundle.target_evidence_refs!=(facts.facts_ref,):raise ValueError('PROCUREMENT_CASE_RECEIPT_MISMATCH')
    data=bundle.model_dump(mode='json');basis={k:data[k] for k in ('execution_epoch_ref','plan_ref','plan_hash','result','observed_facts','target_evidence_refs','candidate_rule_evaluations','justification','all_rule_justifications')}
    basis['evaluated_at']=bundle.evaluated_at.isoformat()
    h=digest(basis)
    if (bundle.bundle_ref,bundle.bundle_hash)!=('runtime-semantic-evaluation:'+h[7:],h):raise ValueError('PROCUREMENT_BUNDLE_HASH_MISMATCH')
    if bundle.all_rule_justifications is None or tuple(r.rule_id for r in bundle.candidate_rule_evaluations)!=tuple(r.rule_id for r in b.plan.rules):raise ValueError('PROCUREMENT_RECEIPT_RULE_INVENTORY_MISMATCH')
    if bundle.result!=bundle.candidate_rule_evaluations[0].outcome or bundle.justification!=bundle.all_rule_justifications[0]:raise ValueError('PROCUREMENT_RECEIPT_SUMMARY_MISMATCH')
    first=bundle.candidate_rule_evaluations[0]
    if (bundle.satisfied_predicates,bundle.failed_predicates,bundle.exception_checks)!=(tuple(p.predicate_id for p in first.predicate_results if p.result=='SATISFIED'),tuple(p.predicate_id for p in first.predicate_results if p.result=='FAILED'),first.exception_results):raise ValueError('PROCUREMENT_RECEIPT_SUMMARY_MISMATCH')
    proof=bundle.all_rule_justifications[0]
    if (bundle.competency_refs,bundle.ir_node_refs,bundle.source_evidence_refs)!=(proof.competency_refs,proof.ir_node_refs,proof.source_evidence_refs):raise ValueError('PROCUREMENT_RECEIPT_SUMMARY_EVIDENCE_MISMATCH')
    if not (b.plan.compiled_at<=facts.frozen_at==o.observed_at==bundle.evaluated_at<=produced_at):raise ValueError('PROCUREMENT_RECEIPT_TIME_MISMATCH')
    for rule,row in zip(b.plan.rules,bundle.candidate_rule_evaluations,strict=True):
        for compiled,receipts in [(rule.predicates,row.predicate_results),(rule.input_requirements or (),row.input_requirement_results or ()),(rule.exception_predicates,row.exception_results)]:
            if tuple(map(_procurement_metadata,compiled))!=tuple(map(_procurement_metadata,receipts)):raise ValueError('PROCUREMENT_RECEIPT_ROLE_METADATA_MISMATCH')
            if any(p.observed_value!=facts.facts.get(p.fact_ref) or p.comparison_observed_value!=facts.facts.get(p.comparison_fact_ref) for p in receipts):raise ValueError('PROCUREMENT_RECEIPT_OPERAND_MISMATCH')
        edges=tuple(e for e in (b.plan.precedences or ()) if e.lower_priority_ref==rule.rule_id)
        receipts=row.precedence_results or ()
        if len(edges)!=len(receipts):raise ValueError('PROCUREMENT_PRECEDENCE_RECEIPT_MISMATCH')
        for e,r in zip(edges,receipts,strict=True):
            if (r.ir_node_ref,r.higher_priority_ref,r.lower_priority_ref,r.evidence_refs)!=(e.ir_node_ref,e.higher_priority_ref,e.lower_priority_ref,e.evidence_refs) or tuple(map(_procurement_metadata,r.condition_results))!=tuple(map(_procurement_metadata,e.conditions)):raise ValueError('PROCUREMENT_PRECEDENCE_RECEIPT_MISMATCH')
            if any(p.observed_value!=facts.facts.get(p.fact_ref) for p in r.condition_results):raise ValueError('PROCUREMENT_PRECEDENCE_FACT_MISMATCH')
    return bundle


def verify_procurement_assessment(v):
    from datetime import timedelta
    old,new=v.predecessor,v.successor
    for b in (old,new):verify_procurement_basis(b)
    if (old.source,old.session_ref,old.context,old.required_inventory_pin,old.inputs)!=(new.source,new.session_ref,new.context,new.required_inventory_pin,new.inputs):raise ValueError('PROCUREMENT_ASSESSMENT_BASIS_MISMATCH')
    if v.equivalences:raise ValueError('PROCUREMENT_EQUIVALENCE_NOT_IN_BOUNDED_SLICE')
    if v.verifier_hash!=raw_hash(base64.b64decode(new.consumer_assets['verifier'])):raise ValueError('PROCUREMENT_VERIFIER_IDENTITY_MISMATCH')
    env,impact,selected=source_change_projection(old,new,{})
    if (v.delta,v.impact,v.selected_case_ids)!=(env,impact,selected):raise ValueError('PROCUREMENT_SELECTION_MISMATCH')
    if v.produced_at.utcoffset()!=timedelta(0):raise ValueError('PROCUREMENT_ASSESSMENT_TIME_INVALID')
    _unique([o.case_id for o in v.observations],'procurement observation')
    if tuple(sorted(o.case_id for o in v.observations))!=selected:raise ValueError('PROCUREMENT_SELECTED_RECEIPT_MISSING')
    for o in v.observations:
        bundle=verify_procurement_observation(new,o,v.produced_at)
        if getattr(new,'profile_id',None)=='controlled-procurement.incident-evidence.v1.1':
            case=next(c for c in new.inputs if c.case_id==o.case_id)
            facts,receipt=incident_projection(case,new)
            receipt['outputs']=incident_outputs(case,new,o.response['bundle'],receipt)
            if o.response.get('incident')!=receipt:raise ValueError('INCIDENT_RETAINED_RECEIPT_MISMATCH')
    findings=required_inventory_findings(old,new,{},v.observations)
    if v.findings!=findings or v.coverage_status!=required_coverage_status(findings):raise ValueError('PROCUREMENT_RETAINED_COVERAGE_MISMATCH')
    return v


def read_procurement_closure(*,store,scope,record,assessment,expected_source_hash,expected_evaluator_hash,expected_verifier_hash):
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    v=assessment
    if v.successor.source.data['source_hash']!=expected_source_hash or raw_hash(base64.b64decode(v.successor.consumer_assets['predicate']))!=expected_evaluator_hash or expected_verifier_hash is None or v.verifier_hash!=expected_verifier_hash:raise ValueError('PROCUREMENT_REPLAY_MEASUREMENT_BASIS_MISMATCH')
    seen=set()
    while True:
        verify_procurement_assessment(v)
        for b in (v.predecessor,v.successor):
            embedded_records=[b.source,b.context]
            if getattr(b,'profile_id',None)=='controlled-procurement.incident-evidence.v1.1':
                embedded_records.extend(Record.model_validate(b.source.data[k]) for k in ('base_source','supplement'))
                embedded_records.extend(Record.model_validate(r) for r in b.context.data['incident_witnesses'])
            if getattr(b,'profile_id',None)=='controlled-procurement.component-amount.v1':
                embedded_records.extend((*b.forms,*b.formations))
                for form in b.forms:read_component_amendments(store,scope,form,b.source,b.session_ref,raw_hash(base64.b64decode(b.consumer_assets['product'])),v.produced_at)
            for embedded in embedded_records:
                retained=store.get(**scope,artifact_ref=embedded.artifact_ref,contract_type=Record)
                if retained!=embedded or retained.produced_at>v.produced_at:raise ValueError('PROCUREMENT_REPLAY_SOURCE_CONTEXT_JOIN_MISSING')
            reference=b.required_inventory
            if reference is not None:_read_required_inventory_chain(store,scope['organization_id'],scope['workspace_id'],reference,v.produced_at)
        if record.produced_at<v.produced_at:raise ValueError('PROCUREMENT_REPLAY_RECORD_TIME_MISMATCH')
        parent=v.correction_parent
        if parent is None:break
        if parent.artifact_ref in seen or parent.artifact_kind!='policy_revision':raise ValueError('PROCUREMENT_CORRECTION_CYCLE')
        seen.add(parent.artifact_ref)
        record=store.get(**scope,artifact_ref=parent.artifact_ref,contract_type=Record)
        if record is None or record.artifact_hash!=parent.artifact_hash or record.kind!='consumer_run' or record.data.get('lane')!='source_change':raise ValueError('PROCUREMENT_CORRECTION_PARENT_MISSING')
        prior=source_change_assessment_type(record.data['assessment']).model_validate(record.data['assessment'])
        if prior.schema_version!=v.schema_version or prior.predecessor!=v.predecessor or prior.successor.source!=v.successor.source or prior.successor.required_inventory_pin!=v.successor.required_inventory_pin or prior.successor.context!=v.successor.context or prior.successor.inputs!=v.successor.inputs or prior.produced_at>=v.produced_at:raise ValueError('PROCUREMENT_CORRECTION_BASIS_MISMATCH')
        candidate=v.successor.candidate
        if (candidate.predecessor_candidate_set_ref,candidate.predecessor_candidate_set_hash)!=(prior.successor.candidate.candidate_set_ref,prior.successor.candidate.candidate_set_hash):raise ValueError('PROCUREMENT_CORRECTION_CANDIDATE_JOIN_MISMATCH')
        if prior.coverage_status!='DEFICIENT':raise ValueError('PROCUREMENT_CORRECTION_PARENT_NOT_DEFICIENT')
        v=prior
    return assessment


def procurement_profile_content(data):
    """Exact base+amendment carriage, not extraction or normative inference."""
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.source_change_consumer_verification import INCIDENT_PROFILE
    base=Record.model_validate(data['base_source']);supplement=Record.model_validate(data['supplement'])
    raw=base64.b64decode(base.data['original_json_b64'],validate=True)
    text,blocks=procurement_source_content(raw)
    if base.data['markdown']!=text or base.data['blocks']!=blocks or raw_hash(raw)!=base.data['source_document_hash']:
        raise ValueError('INCIDENT_BASE_SOURCE_BYTES_MISMATCH')
    amended=base64.b64decode(supplement.data['text_b64'],validate=True)
    if data['profile_id']!=INCIDENT_PROFILE or raw_hash(amended)!='sha256:8fc931d2abac59e76b6550c22612f6cfe50d34cb25539af7ed17f17890d98cda' or supplement.data['document_hash']!=raw_hash(amended):
        raise ValueError('INCIDENT_SUPPLEMENT_BYTES_MISMATCH')
    if (base.organization_id,base.workspace_id,base.data['session_ref'])!=(supplement.organization_id,supplement.workspace_id,supplement.data['session_ref']):
        raise ValueError('INCIDENT_SUPPLEMENT_SCOPE_MISMATCH')
    import re
    clauses=re.findall(r'^\*\*(IE-\d\d) ([^\n]+)$',amended.decode(),re.M)
    if [key for key,_ in clauses]!=['IE-%02d'%i for i in range(1,9)]:raise ValueError('INCIDENT_CLAUSE_MAPPING_MISMATCH')
    for key,body in clauses:
        quote=body;start=len(text)+len(key)+1;raw_start=amended.index(quote.encode())
        blocks.append(dict(block_id=key,text=quote,start=start,end=start+len(quote),block_hash=raw_hash(quote.encode()),
            source_locator=key,raw_byte_start=raw_start,raw_byte_end=raw_start+len(quote.encode()),
            supplemental_source_ref=supplement.artifact_ref,supplemental_source_hash=supplement.artifact_hash,
            supplemental_document_hash=raw_hash(amended)))
        text+=key+'\n'+quote+'\n\n'
    return text,blocks


def incident_projection(case,b):
    """Pure receipt qualification against the frozen synthetic roster only.

    Shared by native intake and retained consistency checks; not a policy engine,
    institutional authentication, a store fallback or an expected-output lookup.
    """
    from contracts.source_change_consumer_verification import ProcurementSubmittedFactsV2,SyntheticIncidentWitnessV1
    import json
    form=case.form;key='procurement.incident_ref';raw=form.get(key)
    reason=('MISSING_KEY' if key not in form else 'EXPLICIT_NULL' if raw is None else
            'STRUCTURED_ASSERTION' if isinstance(raw,dict) else 'EMPTY' if raw=='' else
            'WHITESPACE' if not raw.strip() else 'TRIMMED' if raw!=raw.strip() else 'ASSERTION')
    normalized=raw.strip() if isinstance(raw,str) and raw.strip() else None
    normalization=dict(consumer_ref='procurement.incident.strip.v1',source_hash=b.source.data['source_hash'],
        supplied_keys=sorted(form),raw_form=form,submitted=key in form,raw=raw,normalized=normalized,reason=reason)
    roster={r['artifact_ref']:r for r in b.context.data['incident_witnesses']}
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    proofs=[];polarities=set()
    def read(pin):
        r=roster.get(pin.artifact_ref)
        if pin.artifact_kind!='policy_revision' or r is None or r['artifact_hash']!=pin.artifact_hash:return None,None
        record=Record.model_validate(r)
        if (record.organization_id,record.workspace_id,record.data.get('session_ref'))!=(b.source.organization_id,b.source.workspace_id,b.session_ref):return None,None
        return record,SyntheticIncidentWitnessV1.model_validate(record.data['witness'])
    source_pin=dict(artifact_kind='policy_revision',artifact_ref=b.source.artifact_ref,artifact_hash=b.source.artifact_hash)
    for pin in case.evidence_pins:
        record,w=read(pin);event=False;actor=False;why='UNRESOLVED_OR_OUT_OF_SCOPE_PIN'
        basis_pin=None
        if w is not None:
            event=(w.kind=='recording' and w.profile_id==b.profile_id and w.source_pin.model_dump(mode='json')==source_pin and
                   (w.subject_ref,w.commitment_ref,w.scope_ref,w.incident_ref)==(case.subject_ref,case.commitment_ref,case.scope_ref,normalized) and
                   w.valid_from<=case.as_of<=w.valid_until and w.recorded_at<=w.observed_at<=case.as_of)
            basis_pin=w.basis_pin
            br,bw=read(basis_pin) if basis_pin else (None,None)
            actor=(bw is not None and bw.kind=='recorder_basis' and bw.actor_ref==w.actor_ref and bw.role=='Security Director' and
                   bw.profile_id==b.profile_id and bw.source_pin.model_dump(mode='json')==source_pin and
                   bw.valid_from<=w.recorded_at<=bw.valid_until and bw.observed_at<=case.as_of)
            why='QUALIFIED_SYNTHETIC_BASIS' if event and actor else 'EVENT_SCOPE_TIME_OR_RECORDER_BASIS_NOT_ESTABLISHED'
            if event and actor:polarities.add(w.polarity)
        proofs.append(dict(pin=pin.model_dump(mode='json'),event_match=event,recorder_match=actor,
            basis_pin=basis_pin.model_dump(mode='json') if basis_pin else None,reason=why))
    state=('CONFLICT' if len(polarities)==2 else 'ESTABLISHED' if 'POSITIVE' in polarities else
           'KNOWN_ABSENT' if 'NEGATIVE' in polarities else 'UNKNOWN')
    # An actual incompatible assertion pair is adverse unresolved information,
    # not qualified positive/negative proof. A bare CONFLICT label has no effect.
    observations=raw.get('observations',[]) if isinstance(raw,dict) else []
    assertion_conflict=any(
        isinstance(a,dict) and isinstance(n,dict) and isinstance(a.get('value'),str) and a['value'].strip() and
        n.get('state')=='KNOWN_ABSENT' and a.get('recorded_by')==n.get('recorded_by') and
        a.get('observed_at')==n.get('observed_at') for a in observations for n in observations)
    if assertion_conflict:state='CONFLICT'
    general_refs=('procurement.department','procurement.software_subscription','procurement.budget',
        'procurement.commitment_date','procurement.first_year_non_cancellable_usd','procurement.entity_kind')
    general_known=all(form.get(ref) is not None for ref in general_refs)
    facts=ProcurementSubmittedFactsV2.model_validate_json(json.dumps(form)).model_dump(mode='json',by_alias=True,exclude_none=True)
    facts.pop(key,None)
    if normalized is not None:facts[key]=normalized
    facts.update({'procurement.recording_established':state=='ESTABLISHED',
                  'procurement.recorder_qualified':any(p['event_match'] and p['recorder_match'] for p in proofs),
                  'procurement.emergency_established':general_known and form.get('procurement.emergency_security_remediation') is True})
    qualification=dict(consumer_ref='procurement.incident.qualify.v1',profile_id=b.profile_id,
        source_hash=b.source.data['source_hash'],proposition='incident_reference_recorded_by_required_actor',
        subject_ref=case.subject_ref,commitment_ref=case.commitment_ref,scope_ref=case.scope_ref,
        as_of=case.as_of.isoformat(),recording_evidence_state=state,proofs=proofs,
        general_inputs_known=general_known,assertion_conflict=bool(assertion_conflict),
        evidence_level='QUALIFIED_SYNTHETIC' if polarities else 'UNQUALIFIED_ASSERTIONS',
        institutional_authentication='NOT_ESTABLISHED')
    return facts,dict(normalization=normalization,qualification=qualification)


def incident_outputs(case,b,bundle,receipt):
    """Project actual native outcomes; evidence state never asserts approval."""
    finance=next(r for r in bundle['candidate_rule_evaluations'] if r['rule_id']=='rule.finance_approval')
    state=receipt['qualification']['recording_evidence_state'];form=case.form
    outside=any(p['result']=='FAILED' for p in finance['predicate_results'] if p['fact_ref'] in (
        'procurement.department','procurement.software_subscription','procurement.budget','procurement.commitment_date','procurement.entity_kind'))
    result=finance['outcome']
    requirement='UNKNOWN' if result=='UNKNOWN' else 'OUT_OF_SCOPE' if outside else 'REQUIRED' if result=='MATCH' else 'NOT_REQUIRED_BY_THIS_RULE'
    emergency=form.get('procurement.emergency_security_remediation')
    # Interpret actual native condition receipts, never reevaluate thresholds.
    amount_failed=any(p['result']=='FAILED' and p['fact_ref']=='procurement.first_year_non_cancellable_usd'
                      for p in (*finance['exception_results'],*finance['predicate_results']))
    emergency_failed=emergency is False and any(p['result']=='FAILED' and p['fact_ref']=='procurement.emergency_established'
                                               for p in finance['exception_results'])
    not_applicable=outside or emergency_failed or amount_failed
    native_exemption=any(p['outcome']=='MATCH' for p in (finance['precedence_results'] or ()))
    reliance=('UNRESOLVED' if requirement=='UNKNOWN' else 'NOT_APPLICABLE' if not_applicable else 'UNRESOLVED' if emergency is None else
              'ESTABLISHED' if native_exemption else 'NOT_ESTABLISHED' if state=='KNOWN_ABSENT' else state if state=='CONFLICT' else 'UNRESOLVED')
    return dict(profile_id=b.profile_id,recording_evidence_state=state,exemption_reliance=reliance,
                finance_requirement=requirement,approval_evidence='NOT_ASSESSED')


def incident_consumer_valid(d):
    return (d.consumer,d.role,d.selector) in {
        ('normalization','normalization','incident_ref'),
        ('qualification','qualification','recording'),('qualification','qualification','conflict'),
        ('projection','projection','approval_evidence'),('projection','projection','profile_id')}


def component_consumer_valid(d):
    return (d.consumer in ('component_receipt','formation_handoff') and d.role=='formation'
        and d.fact_ref=='procurement.first_year_non_cancellable_usd' and d.operator=='EXISTS'
        and d.frame_ref=='qualification.annual_committed_value'
        and ((d.consumer=='formation_handoff')==(d.selector=='native_amount')))


def verify_component_form(case,source,material,session):
    """Pinned raw membership only; never classifies components or adds amounts."""
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.source_change_consumer_verification import AMOUNT_PROFILE
    material=Record.model_validate(material.model_dump(mode='json'));data=material.data
    expected_pin=dict(artifact_kind='policy_revision',artifact_ref=source.artifact_ref,artifact_hash=source.artifact_hash)
    if source.data.get('profile_id')!=AMOUNT_PROFILE or data.get('profile_id')!=AMOUNT_PROFILE or data.get('lane')!='procurement_fee_form' or material.kind!='consumer_candidate':raise ValueError('AMOUNT_FORM_PROFILE_MISMATCH')
    if (material.organization_id,material.workspace_id,data.get('session_ref'))!=(source.organization_id,source.workspace_id,session) or data['source_pin']!=expected_pin:raise ValueError('AMOUNT_FORM_SCOPE_MISMATCH')
    if (case.form_pin.artifact_kind,case.form_pin.artifact_ref,case.form_pin.artifact_hash)!=('policy_revision',material.artifact_ref,material.artifact_hash) or data['case_id']!=case.case_id or data['form']!=case.form:raise ValueError('AMOUNT_INPUT_FORM_MISMATCH')
    ids=[c['component_id'] for c in case.form['fee_components']]
    if data['component_ids']!=ids or len(set(ids))!=len(ids) or data['roster_hash']!=digest(tuple(ids)):raise ValueError('AMOUNT_ROSTER_MISMATCH')
    expected={c['component_id']:dict(original=c['evidence'],retained_locator='/form/fee_components/'+str(i)) for i,c in enumerate(case.form['fee_components'])}
    if data['locator_mapping']!=expected:raise ValueError('AMOUNT_LOCATOR_MAPPING_MISMATCH')
    if material.produced_at<source.produced_at or material.produced_at.utcoffset().total_seconds()!=0:raise ValueError('AMOUNT_FORM_TIME_MISMATCH')
    return material


def verify_component_record(case,source,material,record,session,producer_hash):
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.source_change_consumer_verification import ProcurementFormationV1,AMOUNT_PROFILE
    record=Record.model_validate(record.model_dump(mode='json'));data=record.data
    verify_component_form(case,source,material,session)
    f=ProcurementFormationV1.model_validate(data['formation'])
    if record.kind!='consumer_candidate' or data.get('lane')!='component_formation' or data.get('profile_id')!=AMOUNT_PROFILE:raise ValueError('AMOUNT_RECEIPT_PROFILE_MISMATCH')
    if (record.organization_id,record.workspace_id,data.get('session_ref'))!=(source.organization_id,source.workspace_id,session):raise ValueError('AMOUNT_RECEIPT_SCOPE_MISMATCH')
    if data['input']!=case.model_dump(mode='json') or data['material']!=material.model_dump(mode='json') or data['case_id']!=case.case_id:raise ValueError('AMOUNT_RECEIPT_INPUT_MISMATCH')
    if f.source_pin.model_dump(mode='json')!=dict(artifact_kind='policy_revision',artifact_ref=source.artifact_ref,artifact_hash=source.artifact_hash) or f.form_pin!=case.form_pin:raise ValueError('AMOUNT_RECEIPT_SOURCE_MISMATCH')
    if f.input_hash!=digest(case.model_dump(mode='json')) or f.roster_hash!=material.data['roster_hash'] or f.producer_hash!=producer_hash:raise ValueError('AMOUNT_RECEIPT_BASIS_MISMATCH')
    if [r.component_id for r in f.receipts]!=material.data['component_ids']:raise ValueError('AMOUNT_RECEIPT_ROSTER_MISMATCH')
    for raw,r in zip(case.form['fee_components'],f.receipts,strict=True):
        amount=raw.get('amount_usd') if type(raw.get('amount_usd')) is int else None
        if r.component_hash!=digest(raw) or r.amount!=amount or r.supplied_keys!=tuple(sorted(raw)):raise ValueError('AMOUNT_COMPONENT_FACT_MISMATCH')
        expected=material.artifact_ref+'#'+material.data['locator_mapping'][r.component_id]['retained_locator']
        if r.evidence_refs!=(expected,):raise ValueError('AMOUNT_COMPONENT_EVIDENCE_MISMATCH')
    key='procurement.first_year_non_cancellable_usd'
    if f.caller_supplied!=(key in case.form['facts']) or f.caller_assertion!=case.form['facts'].get(key):raise ValueError('AMOUNT_CALLER_ASSERTION_MISMATCH')
    if not source.produced_at<=material.produced_at<=record.produced_at or record.produced_at.utcoffset().total_seconds()!=0:raise ValueError('AMOUNT_RECEIPT_TIME_MISMATCH')
    return f


def verify_component_basis(b,code):
    if len(b.forms)!=len(b.inputs) or len(b.formations)!=len(b.inputs):raise ValueError('AMOUNT_BASIS_INVENTORY_MISMATCH')
    _unique([r.artifact_ref for r in b.forms],'amount form');_unique([r.artifact_ref for r in b.formations],'amount formation')
    for case,form,formation in zip(b.inputs,b.forms,b.formations,strict=True):
        f=verify_component_record(case,b.source,form,formation,b.session_ref,code['product'])
        if f.dispatch!='READY':raise ValueError('AGGREGATE_MISMATCH:'+formation.artifact_ref)


def component_native_projection(case,b):
    """Project an already formed receipt; no component production/arithmetic."""
    record=next(r for r in b.formations if r.data['case_id']==case.case_id)
    f=record.data['formation']
    if f['dispatch']!='READY':raise ValueError('AGGREGATE_MISMATCH:'+record.artifact_ref)
    from contracts.source_change_consumer_verification import ProcurementFeeFormV1
    native=ProcurementFeeFormV1.model_validate(case.form).facts.model_dump(mode='json',by_alias=True,exclude_none=True)
    key='procurement.first_year_non_cancellable_usd';native.pop(key,None)
    if f['status']=='COMPLETE':native[key]=f['amount']
    return native,record


def component_receipt_deficiency(b,d,observations):
    if not component_consumer_valid(d):return 'unsupported formation consumer: '+d.selector
    observed={o.case_id:o for o in observations}
    for cid in d.case_ids:
        if cid not in observed:return 'missing actual formation case: '+cid
        record=next((r for r in b.formations if r.data['case_id']==cid),None)
        if record is None:return 'missing formation receipt: '+cid
        expected=record.artifact_ref+'#'+d.selector
        linked=b.receipt_bindings.get(d.target_id)
        if linked!=expected:return 'missing or wrong formation receipt binding: '+cid+':'+d.selector
        if d.consumer=='component_receipt' and not any(r['component_id']==d.selector for r in record.data['formation']['receipts']):return 'missing component receipt: '+cid+':'+d.selector
        if d.consumer=='formation_handoff' and observed[cid].response.get('formation')!=record.model_dump(mode='json'):return 'missing formation-native amount handoff: '+cid
    return None


def read_component_amendments(store,scope,form,source,session,producer_hash,at):
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.source_change_consumer_verification import ProcurementInputV3
    seen=set()
    while True:
        actual=store.get(**scope,artifact_ref=form.artifact_ref,contract_type=Record)
        if actual!=form or form.produced_at>at:raise ValueError('AMOUNT_RETAINED_FORM_JOIN_MISSING')
        pin=form.data.get('amendment_parent')
        if pin is None:return
        if pin['artifact_kind']!='policy_revision' or pin['artifact_ref'] in seen:raise ValueError('AMOUNT_AMENDMENT_CYCLE_OR_KIND')
        seen.add(pin['artifact_ref'])
        parent=store.get(**scope,artifact_ref=pin['artifact_ref'],contract_type=Record)
        if parent is None or parent.artifact_hash!=pin['artifact_hash'] or parent.produced_at>form.produced_at:raise ValueError('AMOUNT_AMENDMENT_PARENT_MISSING')
        previous=ProcurementInputV3.model_validate(parent.data['input']);material=Record.model_validate(parent.data['material'])
        verify_component_record(previous,source,material,parent,session,producer_hash)
        if previous.form==form.data['form']:raise ValueError('AMOUNT_AMENDMENT_INPUT_UNCHANGED')
        form=material


def fee_facts_receipt_deficiency(basis, dependency):
    """One factual receipt join plugged into the existing fixed comparer."""
    bound=basis.bindings.get(dependency.target_id)
    if bound is None:return 'required factual binding missing'
    if bound!=dependency.selector:return 'wrong existing factual receipt binding'
    if dependency.consumer=='fee_component':
        if len([r for r in basis.formation.receipts if r.component_id==bound])!=1:
            return 'required normalization receipt missing'
    elif bound!='total':return 'formation consumer handoff missing'
    return None


def fee_facts_assessment(formation, declaration, bindings):
    from contracts.source_change_consumer_verification import (FeeFactsCoverageBasisV1,FeeFactsFormationV1,
        FeeFactsDependencyV1,SourceRequirementInstanceV1,SourceImpactFindingV1)
    formed=FeeFactsFormationV1.model_validate(formation)
    requirements=tuple(SourceRequirementInstanceV1.model_validate(r) for r in declaration['requirements'])
    dependencies=tuple(FeeFactsDependencyV1.model_validate(d) for d in declaration['dependencies'])
    inventory=FeeFactsCoverageBasisV1(requirements=requirements,dependencies=dependencies,bindings={},formation=formed)
    basis=inventory.model_copy(update={'bindings':dict(bindings)})
    if set(bindings)-{d.target_id for d in dependencies}:raise ValueError('FEE_BINDING_TARGET_UNKNOWN')
    legacy=tuple(SourceImpactFindingV1(requirement_id=d.requirement_id,instance_ref=next(r.instance_ref for r in requirements if r.logical_id==d.requirement_id),
        target_id=d.target_id,status='SUPPORTED',reason='exact factual receipt bound',case_ids=d.case_ids) for d in dependencies)
    findings=compare_required_inventory(basis,inventory,legacy,())
    consumer=dict(producer_hash=formed.producer_hash,source_kind=formed.source_kind,total=formed.total.model_dump(mode='json'),total_basis=formed.total_basis,amount_basis=formed.amount_basis,
        source_context=formed.source_context,component_assertions={cid:{k:v.model_dump(mode='json') for k,v in a.items()} for cid,a in formed.component_assertions.items()},qualified_commitment=formed.qualified_commitment,finance=formed.finance,finance_reason=formed.finance_reason,
        assertion_status=formed.assertion_status,source_uncertainty='RETAINED_UNRESOLVED')
    return dict(coverage_status=required_coverage_status(findings),findings=[f.model_dump(mode='json') for f in findings],consumer=consumer)


def read_retained_fee_facts(*,store,organization_id,workspace_id,session_ref,run_pin,expected_document_hash,expected_verifier_hash,expected_producer_hash):
    """Read pinned factual closure only. No normalization, PDF parsing or policy."""
    from pathlib import Path
    from contracts.artifact_pin import ArtifactPinV1
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.ontology_workbench_context import WorkspaceArtifactV1
    from contracts.source_change_consumer_verification import FEE_FACTS_PROFILE,FeeFactsFormV1,FeeFactsFormationV1
    if expected_verifier_hash!=raw_hash(Path(__file__).read_bytes()):raise ValueError('FEE_VERIFIER_PIN_MISMATCH')
    memo={};active=set();verified=set()
    def get(pin,lane):
        if isinstance(pin,dict):pin=ArtifactPinV1.model_validate(pin)
        if pin.artifact_kind!='policy_revision':raise ValueError('FEE_RECORD_KIND_MISMATCH')
        key=(pin.artifact_ref,pin.artifact_hash)
        if key not in memo:
            value=store.get(organization_id=organization_id,workspace_id=workspace_id,artifact_ref=pin.artifact_ref,contract_type=Record)
            if value is None:raise ValueError('FEE_RETAINED_RECORD_MISSING')
            value=Record.model_validate(value.model_dump(mode='json'))
            if value.artifact_hash!=pin.artifact_hash:raise ValueError('FEE_RETAINED_PIN_MISMATCH')
            if (value.organization_id,value.workspace_id)!=(organization_id,workspace_id):raise ValueError('FEE_RETAINED_SCOPE_MISMATCH')
            if value.data.get('session_ref')!=session_ref or value.data.get('profile_id')!=FEE_FACTS_PROFILE:
                raise ValueError('FEE_SESSION_PROFILE_MISMATCH')
            memo[key]=value
        value=memo[key]
        if value.data.get('lane')!=lane:raise ValueError('FEE_RETAINED_LANE_MISMATCH')
        if (lane=='fee_facts_run')!=(value.kind=='consumer_run'):raise ValueError('FEE_RETAINED_OWNER_MISMATCH')
        return value
    def document(pin):
        doc=get(pin,'fee_document')
        if doc.artifact_ref in verified:return doc
        d=doc.data;descriptor=WorkspaceArtifactV1.model_validate(d['descriptor'])
        if (descriptor.organization_id,descriptor.workspace_id)!=(organization_id,workspace_id):raise ValueError('FEE_DOCUMENT_SCOPE_MISMATCH')
        raw=base64.b64decode(d['source_bytes'],validate=True)
        if descriptor.content_hash!=expected_document_hash or raw_hash(raw)!=descriptor.content_hash or len(raw)!=descriptor.byte_size:
            raise ValueError('FEE_DOCUMENT_BYTES_MISMATCH')
        j=d['judgments']
        if j['source_sha256']!=descriptor.content_hash[7:] or j['provenance']!=['DEVELOPMENT','ASSISTANT_AUTHORED','NOT_QUALIFIED'] or j['independent_reviews']!=0:
            raise ValueError('FEE_DOCUMENT_JUDGMENT_MISMATCH')
        if d['evidence_role']!='FACTUAL_ASSERTIONS_NOT_POLICY' or d['source_authority']!='NOT_QUALIFIED':raise ValueError('FEE_FALSE_AUTHORITY')
        if set(d['pages'])!=set(j['inspection']['render_hashes']):raise ValueError('FEE_RETAINED_PAGES_MISSING')
        for name,encoded in d['pages'].items():
            if raw_hash(base64.b64decode(encoded,validate=True))!='sha256:'+j['inspection']['render_hashes'][name]:raise ValueError('FEE_RETAINED_PAGE_HASH_MISMATCH')
        if set(d['assertion_page_refs'])!={'row-'+str(r['row']) for r in j['rows']}:raise ValueError('FEE_SOURCE_ASSERTION_ROSTER_INVALID')
        for roles in d['assertion_page_refs'].values():
            if set(roles)!={'service_period','cancellation','due_condition'}:raise ValueError('FEE_SOURCE_ASSERTION_ROLES_INVALID')
            for refs in roles.values():
                if not refs or len(refs)!=len(set(refs)) or any(not ref.startswith('p') or not ref[1:].isdigit() or 'p'+ref[1:].zfill(2)+'.png' not in d['pages'] for ref in refs):
                    raise ValueError('FEE_SOURCE_ASSERTION_PAGE_INVALID')
        verified.add(doc.artifact_ref);return doc
    def form(pin):
        value=get(pin,'fee_form')
        if value.artifact_ref in active:raise ValueError('FEE_ANCESTRY_CYCLE')
        active.add(value.artifact_ref)
        d=value.data;doc=document(d['document']);declaration=get(d['declaration'],'fee_declaration')
        if declaration.data['document']!=d['document']:raise ValueError('FEE_DECLARATION_PIN_MISMATCH')
        raw=d['form'];parsed=FeeFactsFormV1.model_validate(raw)
        roster=declaration.data['roster']
        if len({r['component_id'] for r in roster})!=len(roster) or {c.component_id:c.source_row_id for c in parsed.components}!={r['component_id']:r['source_row_id'] for r in roster}:
            raise ValueError('FEE_RETAINED_ROSTER_MISMATCH')
        rows={'row-'+str(r['row']):r for r in doc.data['judgments']['rows']}
        if parsed.source_kind!=declaration.data['source_kind']:raise ValueError('FEE_RETAINED_SOURCE_KIND_MISMATCH')
        if parsed.source_kind=='SOURCE_TRANSCRIPTION' and (len(roster)!=len(rows) or {r['source_row_id'] for r in roster}!=set(rows)):
            raise ValueError('FEE_RETAINED_SOURCE_ROSTER_INCOMPLETE_OR_DUPLICATED')
        for c in parsed.components:
            if c.source_row_id not in rows or (c.locator,c.source_label)!=(rows[c.source_row_id]['locator'],rows[c.source_row_id]['service_raw']):
                raise ValueError('FEE_RETAINED_LOCATOR_MISMATCH')
        for c in parsed.components:
            for role in ('service_period','cancellation','due_condition'):
                if set(getattr(c,role).evidence_refs)!=set(doc.data['assertion_page_refs'][c.source_row_id][role]):
                    raise ValueError('FEE_RETAINED_ASSERTION_LOCATION_MISMATCH')
        if parsed.source_kind=='SOURCE_TRANSCRIPTION':
            for c in parsed.components:
                row=rows[c.source_row_id]
                if (c.currency,c.source_unit,c.fee_kind,c.amount_basis)!=('CNY',row['unit'],'maintenance','tax_inclusive_stated_price') or c.raw_amount not in (None,'UNKNOWN',row['price_raw']):
                    raise ValueError('FEE_RETAINED_SOURCE_AMOUNT_CONFLICT')
                expected_state='CONFLICT' if row['row']==doc.data['judgments']['period_conflict']['row'] else 'KNOWN'
                if (c.service_period.state,c.service_period.raw)!=(expected_state,row['service_period_raw']) or (c.cancellation.state,c.cancellation.raw)!=('UNKNOWN',None) or (c.due_condition.state,c.due_condition.raw)!=('UNKNOWN',None):
                    raise ValueError('FEE_SOURCE_ASSERTION_CONFLICT')
                for assertion in (c.service_period,c.cancellation,c.due_condition):
                    if not assertion.evidence_refs or any(not ref.startswith('p') or not ref[1:].isdigit() or ('p'+ref[1:].zfill(2)+'.png') not in doc.data['judgments']['inspection']['render_hashes'] for ref in assertion.evidence_refs):
                        raise ValueError('FEE_ASSERTION_PAGE_REFERENCE_INVALID')
        if raw['source_context']!={k:doc.data['judgments'][k] for k in ('period_conflict','payment','signing','scope_facts')}:
            raise ValueError('FEE_RETAINED_SOURCE_CONTEXT_MISMATCH')
        reqs=declaration.data['requirements'];deps=declaration.data['dependencies']
        expected={('fee:'+c.component_id,'receipt:'+c.component_id,c.component_id) for c in parsed.components}|{('fee:table','handoff:total','total')}
        if {(d['requirement_id'],d['target_id'],d['selector']) for d in deps}!=expected or len(deps)!=len(expected):raise ValueError('FEE_DECLARED_TARGET_MEMBERSHIP_MISMATCH')
        if {(r['logical_id'],t) for r in reqs for t in r['target_ids']}!={(a,b) for a,b,_ in expected} or len(reqs)!=len(expected):raise ValueError('FEE_REQUIRED_SOURCE_MEMBERSHIP_MISMATCH')
        table=next(r for r in reqs if r['logical_id']=='fee:table')
        if table['instance_ref']!='fee-table:'+doc.artifact_hash[7:] or table['quote']!=doc.data['judgments']['table_total']['raw']:
            raise ValueError('FEE_TABLE_SOURCE_INSTANCE_MISMATCH')
        for c in parsed.components:
            r=next(r for r in reqs if r['logical_id']=='fee:'+c.component_id)
            if r['instance_ref']!='fee-source-instance:'+digest([doc.artifact_hash,c.source_row_id,c.locator])[7:] or r['quote']!=c.source_label:raise ValueError('FEE_SOURCE_INSTANCE_MISMATCH')
        if d['amendment_parent']:
            parent=form(d['amendment_parent'])
            if parent.data['document']!=d['document'] or {k:v for k,v in parent.data['form'].items() if k!='case_id'}=={k:v for k,v in raw.items() if k!='case_id'}:raise ValueError('FEE_AMENDMENT_ANCESTRY_MISMATCH')
        active.remove(value.artifact_ref);return value
    def run(pin):
        value=get(pin,'fee_facts_run')
        if value.artifact_ref in active:raise ValueError('FEE_ANCESTRY_CYCLE')
        active.add(value.artifact_ref)
        d=value.data;formed=get(d['formation'],'fee_formation');material=form(formed.data['form'])
        if (formed.data['document'],formed.data['declaration'])!=(material.data['document'],material.data['declaration']):raise ValueError('FEE_FORMATION_SOURCE_MISMATCH')
        parsed=FeeFactsFormV1.model_validate(material.data['form']);formation=FeeFactsFormationV1.model_validate(formed.data['formation'])
        if formation.producer_hash!=expected_producer_hash:raise ValueError('FEE_PRODUCER_PIN_MISMATCH')
        if len(formation.receipts)!=len(parsed.components):raise ValueError('FEE_NORMALIZATION_MEMBERSHIP_MISMATCH')
        for c,raw,receipt in zip(parsed.components,material.data['form']['components'],formation.receipts,strict=True):
            if (receipt.component_id,receipt.source_row_id,receipt.component_hash,receipt.source_hash,receipt.locator,receipt.raw_amount,receipt.source_unit)!=(c.component_id,c.source_row_id,digest(raw),expected_document_hash,c.locator,c.raw_amount,c.source_unit):
                raise ValueError('FEE_NORMALIZATION_INPUT_JOIN_MISMATCH')
            if receipt.money.currency!=c.currency or receipt.money.scale!=2:raise ValueError('FEE_NORMALIZATION_CURRENCY_MISMATCH')
        if len({c.currency for c in parsed.components})!=1 or any(c.amount_basis!=parsed.amount_basis for c in parsed.components):raise ValueError('FEE_RETAINED_AGGREGATION_UNSUPPORTED')
        if formation.total.currency!=parsed.components[0].currency or formation.amount_basis!=parsed.amount_basis or formation.caller_assertion!=parsed.declared_total_minor:
            raise ValueError('FEE_FORMATION_BASIS_MISMATCH')
        if formation.source_kind!=parsed.source_kind or formation.source_context!=parsed.source_context or {cid:{k:v.model_dump(mode='json') for k,v in a.items()} for cid,a in formation.component_assertions.items()}!={c.component_id:{k:getattr(c,k).model_dump(mode='json') for k in ('service_period','cancellation','due_condition')} for c in parsed.components}:
            raise ValueError('FEE_QUALIFICATION_EVIDENCE_MISMATCH')
        # Verify carriage/equality of retained outputs, never re-form arithmetic.
        declaration=get(material.data['declaration'],'fee_declaration')
        expected=fee_facts_assessment(formation.model_dump(mode='json'),declaration.data,d['bindings'])
        if expected!=d['assessment']:raise ValueError('FEE_CONSUMER_FINDINGS_MISMATCH')
        if d['correction_parent']:
            parent=run(d['correction_parent'])
            if parent.data['formation']!=d['formation'] or parent.data['assessment']['coverage_status']!='DEFICIENT':raise ValueError('FEE_CORRECTION_ANCESTRY_MISMATCH')
        active.remove(value.artifact_ref);return value
    return run(run_pin).data['assessment']


def development_fixture(package):
    """Validate finite declared profile/reference definitions, never outcomes."""
    from contracts.source_change_consumer_verification import DevelopmentMeasureFixtureV1,DEVELOPMENT_MEASURE_PROFILE_HASH
    parsed=DevelopmentMeasureFixtureV1.model_validate(package)
    if digest(parsed.profile)!=DEVELOPMENT_MEASURE_PROFILE_HASH:raise ValueError('DEVELOPMENT_PROFILE_PIN_MISMATCH')
    refs=parsed.reference_definitions
    expected={ref:dict(role='HYPOTHETICAL_DEVELOPMENT_FIELD_REFERENCE',assertion='ASSUMED_FOR_TEST',family=ref.split('/')[2],authority='NOT_QUALIFIED') for ref in parsed.profile['fixture_refs']}
    if refs!=expected:raise ValueError('DEVELOPMENT_REFERENCE_DEFINITION_MISMATCH')
    return parsed


def development_input(package,key):
    """Intake rejects ambiguous references; core owns business/amount refusals."""
    from contracts.source_change_consumer_verification import DEVELOPMENT_MEASURE_PROFILE
    if key not in package.inputs:raise ValueError('DEVELOPMENT_FIXTURE_KEY_MISSING')
    raw=package.inputs[key]
    if raw.get('profile_id')!=DEVELOPMENT_MEASURE_PROFILE:raise ValueError('DEVELOPMENT_PROFILE_PIN_MISMATCH')
    if ({'owner_adoption','authority','independent_reviews','domain_review'} & set(raw)
            or raw.get('judgment_assertion')!='ASSUMED_FOR_TEST'):raise ValueError('DEVELOPMENT_FALSE_AUTHORITY')
    for field,ref in [('proposed_version_ref','fixture://proposed-version/dev-v1'),('knowledge_cutoff_ref','fixture://cutoff/dev-v1')]:
        if raw.get(field)!=ref or ref not in package.reference_definitions:raise ValueError('DEVELOPMENT_FIXTURE_REFERENCE_MISSING:'+field)
    if not isinstance(raw.get('premises',{}),dict) or not isinstance(raw.get('obligations'),list):raise ValueError('DEVELOPMENT_INPUT_SHAPE')
    for k,v in raw.get('premises',{}).items():
        if v is None:continue
        if not isinstance(v,dict) or v.get('fixture_ref')!='fixture://judgment/dev-v1':raise ValueError('DEVELOPMENT_FIXTURE_REFERENCE_MISSING:premises.'+k)
        if v.get('assertion')!='ASSUMED_FOR_TEST':raise ValueError('DEVELOPMENT_FALSE_AUTHORITY:premises.'+k)
    roster=package.profile['rosters'].get(raw.get('roster_id'))
    if roster is None:raise ValueError('DEVELOPMENT_ROSTER_REFERENCE')
    obligations=raw.get('obligations',[])
    if not all(isinstance(o,dict) and isinstance(o.get('economic_id'),str) and isinstance(o.get('parts'),list) for o in obligations):raise ValueError('DEVELOPMENT_INPUT_SHAPE')
    ids=[o.get('economic_id') for o in obligations]
    if len(set(ids))!=len(ids):raise ValueError('DEVELOPMENT_AMBIGUOUS_ECONOMIC_REFERENCE')
    declared={r['economic_id']:r for r in roster}
    for o in obligations:
        if o.get('economic_id') not in declared:raise ValueError('DEVELOPMENT_ROSTER_REFERENCE')
        rule=declared[o['economic_id']]
        if o.get('partition') not in rule['partitions']:raise ValueError('DEVELOPMENT_PARTITION_REFERENCE')
        if not all(isinstance(p,dict) and isinstance(p.get('part_id'),str) for p in o['parts']):raise ValueError('DEVELOPMENT_INPUT_SHAPE')
        part_ids=[p.get('part_id') for p in o.get('parts',[])]
        if len(set(part_ids))!=len(part_ids):raise ValueError('DEVELOPMENT_AMBIGUOUS_PART_REFERENCE')
        if set(part_ids)-set(rule['partitions'][o['partition']]):raise ValueError('DEVELOPMENT_PARTITION_REFERENCE')
        for part in o.get('parts',[]):
            if part.get('fixture_ref')!='fixture://money/dev-v1':raise ValueError('DEVELOPMENT_FIXTURE_REFERENCE_MISSING:obligations')
    return raw


def _development_pointer(value):
    return value.replace('~','~0').replace('/','~1')


def development_field_receipts(package,key,fixture_hash):
    """Field carriage only: no exposure arithmetic or interpretation."""
    from contracts.source_change_consumer_verification import DevelopmentFieldReceiptV1
    raw=development_input(package,key);profile=package.profile
    prefix='/inputs/'+_development_pointer(key)
    rows=[]
    def receipt(slot,pointer,supplied,value,ref):
        rows.append(DevelopmentFieldReceiptV1(receipt_id=slot,fixture_hash=fixture_hash,fixture_key=key,
            pointer=pointer,supplied=supplied,raw=value,reference=ref,
            reference_pointer='/reference_definitions/'+_development_pointer(ref)))
    premises=raw.get('premises',{})
    keys=list(profile['required_premises'])
    condition=premises.get('condition_materiality')
    if isinstance(condition,dict) and condition.get('state')=='SUPPLIED' and condition.get('value')=='CONSTITUTIVE':keys.append('constitutive_condition')
    for name in keys:
        receipt('premise:'+name,prefix+'/premises/'+name,name in premises,premises.get(name),'fixture://judgment/dev-v1')
    obligations=raw.get('obligations',[])
    for rule in profile['rosters'][raw['roster_id']]:
        found=next(((i,o) for i,o in enumerate(obligations) if o['economic_id']==rule['economic_id']),None)
        oi,o=found if found else (len(obligations),{})
        parts=o.get('parts',[])
        for part_id in rule['partitions'][o.get('partition','whole')]:
            part=next(((i,p) for i,p in enumerate(parts) if p['part_id']==part_id),None)
            pi,value=part if part else (len(parts),None)
            receipt('obligation:'+rule['economic_id']+':'+part_id,prefix+'/obligations/'+str(oi)+'/parts/'+str(pi),part is not None,value,'fixture://money/dev-v1')
    return tuple(sorted(rows,key=lambda r:r.receipt_id))


def development_declaration(package,key,fixture_hash):
    """Required profile slots precede and do not inspect candidate bindings."""
    receipts=development_field_receipts(package,key,fixture_hash)
    requirements=[];dependencies=[]
    for slot in [r.receipt_id for r in receipts]+['handoff']:
        quote=json.dumps(dict(fixture_hash=fixture_hash,fixture_key=key,required_slot=slot),sort_keys=True)
        requirements.append(dict(logical_id=slot,instance_ref='synthetic-field:'+digest([fixture_hash,key,slot])[7:],char_start=0,char_end=len(quote),quote=quote,target_ids=[slot],depends_on=[]))
        dependencies.append(dict(requirement_id=slot,target_id=slot,consumer='development_handoff' if slot=='handoff' else 'development_field',selector=slot,case_ids=['hypothetical-measure']))
    return dict(requirements=requirements,dependencies=dependencies,locator_kind='SYNTHETIC_JSON_FIELD_NOT_NATURAL_LANGUAGE_SPAN')


def development_receipt_deficiency(basis,dependency):
    bound=basis.bindings.get(dependency.target_id)
    if bound is None:return 'required hypothetical receipt binding missing: '+dependency.target_id
    if bound!=dependency.selector:return 'wrong existing hypothetical receipt binding: '+dependency.target_id
    if dependency.consumer=='development_field' and len([r for r in basis.formation.receipts if r.receipt_id==bound])!=1:
        return 'required hypothetical field receipt missing: '+dependency.target_id
    if dependency.consumer=='development_handoff' and bound!='handoff':return 'hypothetical formation consumer handoff missing'
    return None


def development_assessment(formation,declaration,bindings):
    from contracts.source_change_consumer_verification import (DevelopmentMeasureCoverageBasisV1,DevelopmentMeasureFormationV1,
        DevelopmentMeasureDependencyV1,SourceRequirementInstanceV1,SourceImpactFindingV1)
    formed=DevelopmentMeasureFormationV1.model_validate(formation)
    req=tuple(SourceRequirementInstanceV1.model_validate(r) for r in declaration['requirements'])
    deps=tuple(DevelopmentMeasureDependencyV1.model_validate(d) for d in declaration['dependencies'])
    if set(bindings)-{d.target_id for d in deps}:raise ValueError('DEVELOPMENT_BINDING_TARGET_UNKNOWN')
    inventory=DevelopmentMeasureCoverageBasisV1(requirements=req,dependencies=deps,bindings={},formation=formed)
    basis=inventory.model_copy(update={'bindings':dict(bindings)})
    seeds=tuple(SourceImpactFindingV1(requirement_id=d.requirement_id,instance_ref=next(r.instance_ref for r in req if r.logical_id==d.requirement_id),target_id=d.target_id,status='SUPPORTED',reason='exact retained hypothetical field/handoff receipt',case_ids=d.case_ids) for d in deps)
    findings=compare_required_inventory(basis,inventory,seeds,())
    return dict(coverage_status=required_coverage_status(findings),findings=[f.model_dump(mode='json') for f in findings],consumer=formed.result.model_dump(mode='json'))


def verify_development_formation(formation,package,key,fixture_hash,core_hash):
    """Retained projection/receipt integrity, expressly no amount reexecution."""
    from contracts.source_change_consumer_verification import DevelopmentMeasureFormationV1
    f=DevelopmentMeasureFormationV1.model_validate(formation);raw=development_input(package,key)
    if (f.input_hash,f.profile_hash,f.core_hash)!=(digest(raw),digest(package.profile),core_hash):raise ValueError('DEVELOPMENT_FORMATION_BASIS_MISMATCH')
    if f.receipts!=development_field_receipts(package,key,fixture_hash):raise ValueError('DEVELOPMENT_FIELD_RECEIPT_MISMATCH')
    d=f.result.details
    if d.get('input_hash')!=digest(raw).removeprefix('sha256:') or d.get('profile_hash')!=digest(package.profile).removeprefix('sha256:'):
        raise ValueError('DEVELOPMENT_RESULT_BASIS_MISMATCH')
    if d.get('profile_id')!=package.profile['profile_id']:raise ValueError('DEVELOPMENT_RESULT_PROFILE_MISMATCH')
    if 'premise_observations' in d and d['premise_observations']!=raw.get('premises',{}):raise ValueError('DEVELOPMENT_PREMISE_PROJECTION_MISMATCH')
    if 'observations' in d and d['observations']!=raw.get('observations',{}):raise ValueError('DEVELOPMENT_OBSERVATION_PROJECTION_MISMATCH')
    # Compare every emitted ledger fact with its exact raw part. Never sum it.
    seen=set()
    for row in d.get('ledger',[]):
        identity=(row['economic_id'],row['part_id'])
        if identity in seen:raise ValueError('DEVELOPMENT_LEDGER_DUPLICATION')
        seen.add(identity)
        receipt=next((r for r in f.receipts if r.receipt_id=='obligation:'+':'.join(identity)),None)
        if receipt is None or not receipt.supplied:raise ValueError('DEVELOPMENT_LEDGER_FIELD_MISSING')
        part=receipt.raw;value=part.get('amount_fen');state='MISSING' if 'amount_fen' not in part else 'NULL' if value is None else 'UNKNOWN' if value=='UNKNOWN' else 'SUPPLIED'
        if state=='SUPPLIED' and (type(value) is not int or not 0<=value<=2**63-1):raise ValueError('DEVELOPMENT_LEDGER_MONEY_FORMAT')
        if row!={'economic_id':identity[0],'part_id':identity[1],'amount_fen':value if state=='SUPPLIED' else None,'state':state,'fixture_ref':part['fixture_ref']}:
            raise ValueError('DEVELOPMENT_LEDGER_FIELD_MISMATCH')
    if f.result.status=='HYPOTHETICAL_EXPOSURE' and (d['findings'] or seen!={(r.receipt_id.split(':')[1],r.receipt_id.split(':')[2]) for r in f.receipts if r.receipt_id.startswith('obligation:')}):
        raise ValueError('DEVELOPMENT_COMPLETE_LEDGER_MISMATCH')
    return f


def read_retained_development_measure(*,store,organization_id,workspace_id,session_ref,run_pin,
        expected_fixture_hash,expected_profile_hash,expected_core_hash,expected_verifier_hash,
        trusted_inputs,trusted_declarations):
    """Closed-store hypothetical closure. No import or invocation of calculation."""
    from pathlib import Path
    from contracts.artifact_pin import ArtifactPinV1
    from contracts.policy_revision import PolicyRevisionRecordV1 as Record
    from contracts.ontology_workbench_context import WorkspaceArtifactV1
    from contracts.source_change_consumer_verification import DEVELOPMENT_MEASURE_PROFILE,DEVELOPMENT_MEASURE_PROFILE_HASH
    if expected_verifier_hash!=raw_hash(Path(__file__).read_bytes()):raise ValueError('DEVELOPMENT_VERIFIER_PIN_MISMATCH')
    if expected_profile_hash!=DEVELOPMENT_MEASURE_PROFILE_HASH:raise ValueError('DEVELOPMENT_PROFILE_PIN_MISMATCH')
    scope=dict(organization_id=organization_id,workspace_id=workspace_id);memo={};active=set();done={}
    def get(pin,lane):
        pin=ArtifactPinV1.model_validate(pin)
        if pin.artifact_kind!='policy_revision':raise ValueError('DEVELOPMENT_RECORD_KIND_MISMATCH')
        key=(pin.artifact_ref,pin.artifact_hash)
        if key not in memo:
            r=store.get(**scope,artifact_ref=pin.artifact_ref,contract_type=Record)
            if r is None:raise ValueError('DEVELOPMENT_RETAINED_RECORD_MISSING')
            r=Record.model_validate(r.model_dump(mode='json'))
            if r.artifact_hash!=pin.artifact_hash:raise ValueError('DEVELOPMENT_RETAINED_PIN_MISMATCH')
            if (r.organization_id,r.workspace_id)!=(organization_id,workspace_id):raise ValueError('DEVELOPMENT_SCOPE_MISMATCH')
            if r.data.get('session_ref')!=session_ref or r.data.get('profile_id')!=DEVELOPMENT_MEASURE_PROFILE:raise ValueError('DEVELOPMENT_SESSION_PROFILE_MISMATCH')
            memo[key]=r
        r=memo[key]
        if r.data.get('lane')!=lane or r.kind!=('consumer_run' if lane=='development_measure_run' else 'consumer_candidate'):raise ValueError('DEVELOPMENT_LANE_OWNER_MISMATCH')
        if r.produced_at.utcoffset().total_seconds()!=0:raise ValueError('DEVELOPMENT_TIME_MISMATCH')
        return r
    def before(earlier,later):
        if earlier.produced_at>later.produced_at:raise ValueError('DEVELOPMENT_ANCESTRY_TIME_MISMATCH')
    def fixture(pin):
        record=get(pin,'development_measure_fixture');d=record.data;descriptor=WorkspaceArtifactV1.model_validate(d['descriptor'])
        if (descriptor.organization_id,descriptor.workspace_id)!=(organization_id,workspace_id):raise ValueError('DEVELOPMENT_FIXTURE_SCOPE_MISMATCH')
        retained_descriptor=store.get(**scope,artifact_ref=descriptor.artifact_ref,contract_type=WorkspaceArtifactV1)
        if retained_descriptor!=descriptor:raise ValueError('DEVELOPMENT_GOVERNED_FIXTURE_ARTIFACT_MISSING')
        if descriptor.created_at>record.produced_at:raise ValueError('DEVELOPMENT_FIXTURE_TIME_MISMATCH')
        raw=base64.b64decode(d['source_bytes'],validate=True)
        if descriptor.media_type!='text/plain' or descriptor.content_hash!=expected_fixture_hash or raw_hash(raw)!=expected_fixture_hash or len(raw)!=descriptor.byte_size:
            raise ValueError('DEVELOPMENT_FIXTURE_BYTES_MISMATCH')
        package=development_fixture(json.loads(raw))
        if d['evidence_role']!='HYPOTHETICAL_DEVELOPMENT_ONLY' or d['authority']!='NOT_QUALIFIED':raise ValueError('DEVELOPMENT_FALSE_AUTHORITY')
        return record,package
    def input_record(pin):
        r=get(pin,'development_measure_input')
        if r.artifact_ref in active:raise ValueError('DEVELOPMENT_ANCESTRY_CYCLE')
        if r.artifact_ref in done:return done[r.artifact_ref]
        if trusted_inputs.get(r.artifact_ref)!=r.artifact_hash:raise ValueError('DEVELOPMENT_EXTERNAL_INPUT_PIN_MISMATCH')
        active.add(r.artifact_ref);d=r.data;source,package=fixture(d['fixture']);before(source,r)
        raw=development_input(package,d['fixture_key'])
        if d['input']!=raw or d['input_hash']!=digest(raw) or d['profile_hash']!=expected_profile_hash:raise ValueError('DEVELOPMENT_INPUT_FIXTURE_MISMATCH')
        parent=d.get('amendment_parent')
        if parent:
            prior,_,_=input_record(parent);before(prior,r)
            if prior.data['fixture']!=d['fixture'] or prior.data['profile_hash']!=d['profile_hash']:raise ValueError('DEVELOPMENT_AMENDMENT_FIXTURE_CHANGED')
            if {k:v for k,v in prior.data['input'].items() if k!='case_id'}=={k:v for k,v in raw.items() if k!='case_id'}:raise ValueError('DEVELOPMENT_AMENDMENT_INPUT_UNCHANGED')
        active.remove(r.artifact_ref);done[r.artifact_ref]=(r,source,package)
        return done[r.artifact_ref]
    def formation(pin):
        r=get(pin,'development_measure_formation');d=r.data;material,source,package=input_record(d['input']);before(material,r)
        declaration=get(d['declaration'],'development_measure_declaration');before(material,declaration);before(declaration,r)
        if trusted_declarations.get(declaration.artifact_ref)!=declaration.artifact_hash:raise ValueError('DEVELOPMENT_EXTERNAL_DECLARATION_PIN_MISMATCH')
        if declaration.data['input']!=d['input'] or declaration.data['fixture']!=material.data['fixture']:raise ValueError('DEVELOPMENT_DECLARATION_INPUT_MISMATCH')
        expected=development_declaration(package,material.data['fixture_key'],expected_fixture_hash)
        if any(declaration.data.get(k)!=v for k,v in expected.items()):raise ValueError('DEVELOPMENT_DECLARATION_CLOSURE_MISMATCH')
        verify_development_formation(d['formation'],package,material.data['fixture_key'],expected_fixture_hash,expected_core_hash)
        return r,declaration
    def run(pin):
        r=get(pin,'development_measure_run')
        if r.artifact_ref in active:raise ValueError('DEVELOPMENT_ANCESTRY_CYCLE')
        active.add(r.artifact_ref);d=r.data;formed,declaration=formation(d['formation']);before(formed,r)
        assessment=development_assessment(formed.data['formation'],declaration.data,d['bindings'])
        if d['assessment']!=assessment:raise ValueError('DEVELOPMENT_RETAINED_ASSESSMENT_MISMATCH')
        parent=d.get('correction_parent')
        if parent:
            prior=run(parent);before(prior,r)
            if prior.data['formation']!=d['formation'] or prior.data['assessment']['coverage_status']!='DEFICIENT':raise ValueError('DEVELOPMENT_CORRECTION_ANCESTRY_MISMATCH')
        active.remove(r.artifact_ref)
        return r
    return run(run_pin)
