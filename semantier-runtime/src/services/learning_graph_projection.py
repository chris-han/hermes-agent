"""Bounded read projection of existing learning receipts, never an authority owner."""
from datetime import datetime, timezone
import json
from urllib.parse import urlencode
from contracts.ontology_building import ArtifactPinV1
from contracts.learning_trace_evidence import ActionableLearningLessonV1, LearningTraceObservationV1
from contracts.learned_skill_publication import LearnedSkillProposalV1
from contracts.retained_context_items import RetainedContextItemChangeV1
from eos.artifact_hashing import content_hash
from services.memory_workbench_projection_service import MemoryWorkbenchProjectionService, _OWNERS, _bindings


class LearningGraphProjection:
    def __init__(self, *, store, learning):
        self.store = store
        self.learning = learning
        self.reader = MemoryWorkbenchProjectionService(store=store, learning=learning)

    def read(self, *, context, as_of=None, snapshot_hash=None, offset=0, limit=200, after_ref=''):
        self.reader._scope(context)
        if context.membership_status != 'active':
            raise PermissionError('active workspace membership required')
        if not 1 <= limit <= 200 or offset < 0 or (offset and not (as_of and snapshot_hash)):
            raise ValueError('continuation requires exact snapshot and bounds')
        instant = datetime.fromisoformat(as_of.replace('Z','+00:00')) if as_of else datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError('as_of requires a timezone')
        as_of = instant.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
        records = self.store.list_learning_graph_records(organization_id=context.organization_id,
            workspace_id=context.workspace_id,owner_ref=context.user_id,as_of=as_of,after_ref=after_ref)
        nodes,edges,unresolved = {},[],[]
        values = {}
        pending = [ArtifactPinV1(artifact_kind=_OWNERS[r['schema_version']][1],artifact_ref=r['artifact_ref'],artifact_hash=r['artifact_hash']) for r in records[:200]]
        cohort = {p.artifact_ref for p in pending}
        attempted = set()
        while pending and len(attempted) < 400:
            pin = pending.pop(0)
            key = pin.artifact_ref+'@'+pin.artifact_hash
            if key in attempted: continue
            attempted.add(key)
            # Avoid the older reader's unbounded embedded WorkingContext fallback.
            if pin.artifact_kind == 'working_context':
                unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'embedded_context_resolution_not_qualified'})
                continue
            try:
                value,spec = self.reader._read(context,pin)
            except (PermissionError, ValueError):
                unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'unavailable_in_current_scope_or_hash_mismatch'})
                continue
            if value is None:
                unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'unresolved_owner_reference'})
                continue
            body = value.model_dump(mode='json')
            values[key] = value
            bindings = list(_bindings(value))
            sources = [p.model_dump(mode='json') for field,p,forward in bindings if not forward and field != 'version']
            if isinstance(value, LearnedSkillProposalV1) and (value.origin or {}).get('lesson_json'):
                try:
                    lesson = ActionableLearningLessonV1.model_validate_json(value.origin['lesson_json'])
                    for field in ('evidence_refs','counter_evidence_refs'):
                        for raw in getattr(lesson,field):
                            evidence = ArtifactPinV1.model_validate(raw)
                            bindings.append(('origin.lesson_json.'+field,evidence,False))
                            sources.append(evidence.model_dump(mode='json'))
                except ValueError:
                    unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'lesson_binding_not_supported'})
            target = getattr(value,'version',None)
            if not isinstance(target,ArtifactPinV1): target=None
            node = {'id':key,'pin':pin.model_dump(mode='json'),'label':spec[1],
                'operation':getattr(value,'action',spec[1]),'recorded_at':body.get('recorded_at',body.get('captured_at')),
                'source_set':sources,'source_bindings':[{'relation':field,'pin':p.model_dump(mode='json')} for field,p,forward in bindings if not forward and field != 'version'],
                'predecessor':getattr(value,'predecessor',None).model_dump(mode='json') if isinstance(getattr(value,'predecessor',None),ArtifactPinV1) else None,
                'target':target.model_dump(mode='json') if target else None,
                'body':body,'source_href':None,'before':None,'after':None}
            if isinstance(value, LearnedSkillProposalV1):
                node['source_href']='/learning?'+urlencode({'kind':'candidate','ref':value.candidate.artifact_ref})
            if spec[1] in ('change_candidate','skill_artifact_version'):
                node['source_href']='/learning?'+urlencode({'kind':'candidate' if spec[1]=='change_candidate' else 'procedure','ref':pin.artifact_ref})
            if isinstance(value,RetainedContextItemChangeV1):
                node.update(before=value.prior_content,after=value.content)
            if isinstance(value,LearningTraceObservationV1) and not value.occurrence_id:
                unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'native_incarnation_occurrence_unqualified'})
            nodes[key]=node
            for field,other,forward in bindings:
                otherkey=other.artifact_ref+'@'+other.artifact_hash
                edge={'id':'sha256:'+content_hash([key,field,otherkey,forward]),
                      'source':key if forward else otherkey,'target':otherkey if forward else key,'relation':field}
                edges.append(edge)
                pending.append(other)
        for pin in pending:
            if pin.artifact_ref+'@'+pin.artifact_hash not in attempted:
                unresolved.append({'pin':pin.model_dump(mode='json'),'reason':'reference_read_bound'})
        event_count=0
        for key,value in sorted(values.items()):
            if getattr(value,'ARTIFACT_KIND',None) != 'change_candidate': continue
            if event_count >= 200:
                unresolved.append({'pin':value.pin().model_dump(mode='json'),'reason':'governance_event_bound'})
                continue
            events=self.learning._governance.learning_graph_events(candidate=value.pin(),owner_ref=context.user_id,as_of=as_of,limit=201-event_count)
            if len(events)>200-event_count:
                unresolved.append({'pin':value.pin().model_dump(mode='json'),'reason':'governance_event_bound'})
            for event in events[:200-event_count]:
                eid=event['candidate_event_id']; event_count+=1
                nodes[eid]={'id':eid,'pin':None,'label':event['event_type'],'operation':event['event_type'],
                    'recorded_at':event['created_at'],'body':event,'source_set':[value.pin().model_dump(mode='json')],
                    'target':value.pin().model_dump(mode='json'),'source_href':nodes[key]['source_href'],'before':None,'after':None}
                edges.append({'id':eid+':candidate','source':key,'target':eid,'relation':'candidate_event_receipt'})
        # Exact local package delta; retained bytes only, never current disk state.
        for node in nodes.values():
            for side,pin in [('after',node['target']),('before',node.get('predecessor'))]:
                if pin:
                    value=values.get(pin['artifact_ref']+'@'+pin['artifact_hash'])
                    if hasattr(value,'files'):
                        node[side]=next((f.content_bytes().decode('utf-8',errors='replace') for f in value.files if f.relative_path=='SKILL.md'),None)
        ordered=sorted(nodes.values(),key=lambda n:n['id'])
        edges=sorted({e['id']:e for e in edges}.values(),key=lambda e:e['id'])
        unresolved=sorted({json.dumps(r,sort_keys=True):r for r in unresolved}.values(),key=lambda r:json.dumps(r,sort_keys=True))
        digest='sha256:'+content_hash({'revision':'learning_graph.v1','scope':[context.organization_id,context.workspace_id,context.user_id],
            'as_of':as_of,'after_ref':after_ref,'nodes':ordered,'edges':edges,'unresolved':unresolved})
        if snapshot_hash and snapshot_hash!=digest: raise ValueError('snapshot changed or unavailable; open a new snapshot')
        return {'schema_version':'learning_graph.v1','as_of':as_of,'snapshot_hash':digest,'after_ref':after_ref,
            'nodes':ordered[offset:offset+limit],'edges':edges,'unresolved_references':unresolved,
            'total_nodes':len(ordered),'next_offset':offset+limit if offset+limit<len(ordered) else None,
            'next_cohort_after':records[199]['artifact_ref'] if len(records)>200 else None,
            'coverage':{'cohort_records':len(cohort),'resolved_records':len(values),'reference_read_limit':400,'event_limit':200},
            'authority':'read_projection_non_authority','quality':'unmeasured','supplier_cost':None,'causal_attribution':'unknown'}
