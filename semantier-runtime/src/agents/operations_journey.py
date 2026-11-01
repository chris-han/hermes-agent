"""Versioned deterministic operations projection. Observations never grant authority."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

CATALOGS = {version:json.loads((Path(__file__).parent / f'operations/{version}.json').read_text()) for version in ('metric-definition-v1','metric-definition-v2')}
CATALOG = CATALOGS['metric-definition-v2']
DIRECT_EVENTS = {'start_now_clicked','start_section_viewed','workspace_registration_clicked','download_catalog_resolved','download_catalog_failed','download_resolution_started','download_resolution_completed','download_resolution_failed','download_initiated','download_retry_clicked'}
ATTEMPT_EVENTS = {'download_clicked','download_resolution_started','download_resolution_completed','download_resolution_failed','download_initiated'}
CLIENT_EVENTS = {
    'landing': {'page_entered','section_viewed','scenario_selected','case_opened','trial_cta_clicked','download_clicked'} | DIRECT_EVENTS,
    'workspace': {'registration_started','client_ready','client_failed','client_resumed','download_clicked'},
    'windows': {'app_first_open','registration_started','client_ready','client_failed','client_resumed'},
}
PROPERTIES = {'page_version','case_id','section_id','scenario','role','has_current_task','source','campaign','reason_code','classification','feedback','action','visibility_definition','flow_version','cta_location','attempt_id','retry_of_attempt','release_version','artifact_ref','phase','duration_ms','client_version','channel'}
SAFE_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')


def stamp(value: str | datetime) -> datetime:
    if not isinstance(value,(str,datetime)):raise ValueError('Timestamp required')
    at = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z','+00:00'))
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return at.astimezone(timezone.utc)


def utc(value: datetime) -> str:
    return stamp(value).isoformat().replace('+00:00','Z')


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def canonical_event(body: dict, *, surface: str, subject_ref: str, now: datetime) -> dict:
    if not isinstance(body,dict) or set(body)-{'event_id','event_name','occurred_at','properties','installation_id','producer_version'}:
        raise ValueError('Unsupported observation fields')
    name=body.get('event_name')
    if name not in CLIENT_EVENTS.get(surface,set()):
        raise ValueError('Client cannot assert this event')
    eid=body.get('event_id','')
    if not isinstance(eid,str) or not SAFE_ID.fullmatch(eid):raise ValueError('Bounded event id required')
    at=stamp(body.get('occurred_at',''))
    props=body.get('properties',{})
    if not isinstance(props,dict) or set(props)-PROPERTIES:raise ValueError('Unsupported observation properties')
    for key,value in props.items():
        if key=='duration_ms':
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=120000:raise ValueError('Bounded duration required')
            continue
        if not isinstance(value,(str,bool)) or isinstance(value,str) and (len(value)>128 or not re.fullmatch(r'[\w .:/-]*',value,re.ASCII)):
            raise ValueError('Bounded non-sensitive categorical properties required')
    direct=props.get('flow_version')=='direct_download_v1'
    if 'channel' in props and (surface!='landing' or not direct or not name.startswith('download_') or props['channel'] not in {'channel1','channel2'}):raise ValueError('Unsupported download channel')
    if 'flow_version' in props and not direct:raise ValueError('Unsupported flow version')
    if name in DIRECT_EVENTS and not direct:raise ValueError('Direct-download flow required')
    if direct:
        if not all(isinstance(props.get(k),str) and props[k] for k in ('page_version','cta_location')):raise ValueError('Page and CTA binding required')
        if name in ATTEMPT_EVENTS and not props.get('attempt_id'):raise ValueError('Technical attempt required')
    for key in ('attempt_id','retry_of_attempt'):
        if key in props and (not isinstance(props[key],str) or not re.fullmatch(r'[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}',props[key])):raise ValueError('UUID attempt required')
    if 'phase' in props and props['phase'] not in {'initial','click','retry'}:raise ValueError('Unsupported resolution phase')
    if 'release_version' in props and not re.fullmatch(r'(0|[1-9]\d*)(\.(0|[1-9]\d*)){3}',str(props['release_version'])):raise ValueError('Four-part release version required')
    if 'release_version' in props and any(int(part)>65535 for part in props['release_version'].split('.')):raise ValueError('Windows release version out of range')
    if 'artifact_ref' in props and not re.fullmatch(r'sha256:[a-f0-9]{64}',str(props['artifact_ref'])):raise ValueError('Opaque artifact digest required')
    if name in {'download_catalog_resolved','download_resolution_completed','download_initiated'} and not all(props.get(k) for k in ('release_version','artifact_ref')):raise ValueError('Resolved artifact binding required')
    if name.startswith('download_catalog_') and props.get('phase') not in {'initial','retry'}:raise ValueError('Availability phase required')
    if name=='download_retry_clicked' and props.get('phase')!='retry':raise ValueError('Retry phase required')
    if name.startswith('download_resolution_') and props.get('phase')!='click':raise ValueError('Click phase required')
    if name in {'section_viewed','start_section_viewed'} and props.get('visibility_definition')!='half-visible-2s-v1':
        raise ValueError('Visibility definition required')
    install=body.get('installation_id')
    if install is not None and (surface!='windows' or not isinstance(install,str) or not SAFE_ID.fullmatch(install)):
        raise ValueError('Bounded Windows installation id required')
    version=body.get('producer_version','v1')
    if not isinstance(version,str) or not SAFE_ID.fullmatch(version):raise ValueError('Producer version required')
    return dict(event_id=eid,schema_version='journey-event-v2' if direct else 'journey-event-v1',event_name=name,
                producer_id=surface,producer_version=version,surface=surface,
                subject_ref=subject_ref,occurred_at=utc(at),received_at=utc(now),
                evidence_type='self_reported' if name in {'feedback_submitted','action_recorded'} else 'client_observation',
                installation_id=install,properties=props,
                clock_status='anomalous' if at>now+timedelta(minutes=5) or at<now-timedelta(days=30) else 'accepted')


def summarize_edge(events: list[dict], *, start: str, end: str, days: int, as_of: datetime,
                   covered: bool, start_status=None, end_status=None, cohort_from=None, cohort_to=None, start_alternatives=()) -> dict:
    by_subject=defaultdict(list)
    for row in events:
        if stamp(row['received_at'])<=as_of and stamp(row['occurred_at'])<=as_of and row.get('clock_status')!='anomalous':
            by_subject[row['subject_ref']].append(row)
    result=dict(eligible=0,mature=0,completed=0,immature_completed=0,waiting=0,failed=0,cancelled=0,immature=0,unknown=0,not_observed=0,rate=None,subjects={})
    for subject, rows in sorted(by_subject.items()):
        rows.sort(key=lambda e:(stamp(e['occurred_at']),e.get('event_id','')))
        origins=[e for e in rows if e['event_name'] in {start,*start_alternatives} and (start_status is None or e.get('status')==start_status)]
        if not origins:continue
        origin=stamp(origins[0]['occurred_at']); deadline=origin+timedelta(days=days)
        if cohort_from and origin<stamp(cohort_from) or cohort_to and origin>=stamp(cohort_to):continue
        mature=deadline<=as_of
        result['eligible']+=1;result['mature']+=int(mature)
        after=[e for e in rows if stamp(e['occurred_at'])>=origin]
        success=any(e['event_name']==end and stamp(e['occurred_at'])<=deadline and (end_status is None or e.get('status')==end_status) for e in after)
        if success:category='completed' if mature else 'immature_completed'
        else:
            # Only current owner-confirmed waits/cancellations or explicit relevant
            # errors count. A later ready/resumed event supersedes earlier failure.
            relevant={
                'download_initiated':{'download_resolution_started','download_resolution_completed','download_resolution_failed','download_initiated'},
                'download_resolution_completed':{'download_resolution_started','download_resolution_completed','download_resolution_failed'},
                'client_ready':{'client_failed','client_ready','client_resumed','registration_failed'},
                'payment_status_changed':{'payment_status_changed'},
                'entitlement_status_changed':{'entitlement_status_changed'},
                'task_completed':{'task_started','task_status_changed','task_completed'},
                'registration_completed':{'invitation_status_changed','registration_completed','registration_failed'},
                'invitation_status_changed':{'invitation_application_submitted','invitation_status_changed'},
                'independent_task_completed':{'task_status_changed','task_completed','independent_task_completed'},
                'invitation_application_submitted':{'invitation_status_changed','invitation_application_submitted'},
            }.get(end,{end})
            states=[e for e in after if e['event_name'] in relevant]
            latest=states[-1] if states else {}
            status=latest.get('status')
            if status in {'pending','awaiting_payment','awaiting_invitation','requested','queued','running'} or status=='approved' and end=='registration_completed' and not mature:category='waiting'
            elif status in {'cancelled','rejected','revoked','expired'}:category='cancelled'
            elif latest.get('event_name') in {'client_failed','download_resolution_failed'} or status=='failed':category='failed'
            elif not covered:category='unknown'
            elif not mature:category='immature'
            else:category='not_observed'
        result[category]+=1;result['subjects'][subject]=category
    if covered and result['mature']:
        result['rate']=result['completed']/result['mature']
    return result


def projection(events: list[dict], *, as_of: datetime, coverage: list[dict], excluded: set[str], scope: dict | None = None, current_invitations=(), definition_version='metric-definition-v2') -> dict:
    catalog=CATALOGS[definition_version]
    scope=scope or {}
    rows=[dict(e) for e in events if e['subject_ref'] not in excluded]
    # Accounts are merged only by explicit server-owned identity_linked edges.
    candidates=defaultdict(set)
    for e in rows:
        if e['event_name']=='identity_linked' and e.get('evidence_type')=='server_fact':
            candidates[e['subject_ref']].add(e['linked_subject_ref'])
    ambiguous={subject for subject,targets in candidates.items() if len(targets)>1}
    links={subject:next(iter(targets)) for subject,targets in candidates.items() if len(targets)==1}
    acquisition_rows=[dict(e) for e in rows]
    for e in rows:
        if e['subject_ref'] in links:e['source_subject_ref']=e['subject_ref']
        e['subject_ref']=links.get(e['subject_ref'],e['subject_ref'])
    completed=defaultdict(dict)
    for e in sorted(rows,key=lambda e:(e['occurred_at'],e['event_id'])):
        task=e.get('properties',{}).get('task_id')
        if e['event_name']=='task_completed' and e.get('properties',{}).get('classification')=='real' and task:
            completed[e['subject_ref']].setdefault(task,e)
    for tasks in completed.values():
        for e in list(tasks.values())[1:]:
            rows.append(e|{'event_name':'independent_task_completed','event_id':'derived-reuse:'+e['event_id']})
    coverage_start=stamp(scope['cohort_from']) if scope.get('cohort_from') else as_of-timedelta(days=catalog['cohort_days'])
    complete={c['producer_id'] for c in coverage if c['status']=='CONNECTED' and c.get('coverage_from') and c.get('coverage_until') and stamp(c['coverage_until'])>=as_of and stamp(c['coverage_from'])<=coverage_start}
    dimensions={}
    for subject in {e['subject_ref'] for e in rows}:
        subject_events=sorted([e for e in rows if e['subject_ref']==subject and e.get('clock_status')!='anomalous' and stamp(e['occurred_at'])<=as_of],key=lambda e:(e['occurred_at'],e['event_id']))
        surfaces={e['surface'] for e in subject_events if e['surface'] in {'workspace','windows'}}
        values={'branch':'both' if len(surfaces)>1 else next(iter(surfaces)) if surfaces else 'unknown'}
        known_surfaces=[e['surface'] for e in subject_events if e['surface'] in {'landing','workspace','windows'}]
        values.update(first_touch_surface=known_surfaces[0] if known_surfaces else 'unknown',current_surface=known_surfaces[-1] if known_surfaces else 'unknown')
        for origin_key in ('registration_origin','purchase_origin'):
            origins=[e.get('properties',{}).get(origin_key) for e in subject_events if e.get('evidence_type')=='server_fact' and e.get('properties',{}).get(origin_key)]
            values[origin_key]=origins[0] if origins else 'unknown'
        for key in ('source','campaign','role','scenario','has_current_task','page_version','flow_version','cta_location','release_version'):
            declared=[e.get('properties',{}).get(key) for e in subject_events if key in e.get('properties',{})]
            values[key]=str(declared[0] if key in {'source','campaign','page_version'} else declared[-1]).lower() if declared else 'unknown'
        dimensions[subject]=values
    for e in acquisition_rows:
        dimensions.setdefault(e['subject_ref'],dimensions.get(links.get(e['subject_ref']),{'branch':'unknown'}))
    filters={k:v for k,v in scope.items() if k in {'branch','source','role','scenario','has_current_task','page_version'} and v!='all'}
    rows=[e for e in rows if all(dimensions[e['subject_ref']].get(k,'unknown')==v for k,v in filters.items())]
    acquisition_rows=[e for e in acquisition_rows if all(dimensions[e['subject_ref']].get(k,'unknown')==v for k,v in filters.items())]
    # Registration belongs to the verified account; map only its confirmed
    # continuation into each explicitly linked acquisition unit for this edge.
    for anonymous,target in links.items():
        original=[e for e in acquisition_rows if e['subject_ref']==anonymous]
        if original:
            origin=min(stamp(e['occurred_at']) for e in original)
            acquisition_rows.extend(e|{'subject_ref':anonymous,'source_subject_ref':target} for e in rows if e['subject_ref']==target and e['event_name'] in {'registration_completed','app_first_open'} and stamp(e['occurred_at'])>=origin)
    attempt_rows=[]
    for e in acquisition_rows:
        attempt=e.get('properties',{}).get('attempt_id')
        if attempt and e.get('properties',{}).get('flow_version')=='direct_download_v1':
            key='attempt:'+fingerprint([e['subject_ref'],attempt])
            dimensions.setdefault(key,dict(dimensions.get(e['subject_ref'],{})))
            for dimension in catalog['dimensions']:
                if dimension in e.get('properties',{}):dimensions[key][dimension]=str(e['properties'][dimension]).lower()
            attempt_rows.append(e|{'subject_ref':key,'source_subject_ref':e['subject_ref']})
    analyses=[]
    for analysis in catalog['analyses']:
        selected=acquisition_rows if analysis["unit"]=="anonymous_journey" else rows
        if analysis.get('eligible')=='approved_invitation':
            eligible={e['subject_ref'] for e in rows if e['event_name']=='invitation_status_changed' and e.get('status')=='approved'}
            selected=[e for e in rows if e['subject_ref'] in eligible]
        if analysis['unit']=='verified_account':selected=[e for e in selected if e['subject_ref'].startswith('user:')]
        if analysis.get('flow_version'):selected=[e for e in selected if e.get('properties',{}).get('flow_version')==analysis['flow_version'] or e['event_name']=='app_first_open' and e.get('source_subject_ref')]
        edges=[]
        for edge in analysis['edges']:
            required=set(edge['requires'])
            if scope.get('branch') in {'workspace','windows'}:required.discard('windows' if scope['branch']=='workspace' else 'workspace')
            edge_rows=attempt_rows if edge.get('unit')=='download_attempt' else selected
            valid=[e for e in edge_rows if e['event_name']!='task_completed' or e.get('properties',{}).get('classification')=='real']
            r=summarize_edge(valid,start=edge['start'],end=edge['end'],days=edge['days'],as_of=as_of,covered=required<=complete,start_status=edge.get('start_status'),end_status=edge.get('end_status'),start_alternatives=edge.get('start_alternatives',()),cohort_from=scope.get('cohort_from'),cohort_to=scope.get('cohort_to'))
            groups={}
            for dimension in catalog['dimensions']:
                groups[dimension]=[]
                for value in sorted({dimensions[e['subject_ref']].get(dimension,'unknown') for e in valid}):
                    subset=[e for e in valid if dimensions[e['subject_ref']].get(dimension,'unknown')==value]
                    group=summarize_edge(subset,start=edge['start'],end=edge['end'],days=edge['days'],as_of=as_of,covered=required<=complete,start_status=edge.get('start_status'),end_status=edge.get('end_status'),start_alternatives=edge.get('start_alternatives',()),cohort_from=scope.get('cohort_from'),cohort_to=scope.get('cohort_to'))
                    groups[dimension].append(group|{'id':value})
                groups[dimension].sort(key=lambda g:(-g['eligible'],g['id']))
            edges.append(edge|r|{'groups':groups,'coverage':'complete' if required<=complete else 'insufficient'})
        analyses.append(analysis|{'edges':edges})
    journeys=[]
    for unit,source in [('verified_account',[e for e in rows if e['subject_ref'].startswith('user:')]),('anonymous_journey',[e for e in acquisition_rows if e['subject_ref'].startswith('anon:')]),('download_attempt',attempt_rows)]:
        for subject in sorted({e['subject_ref'] for e in source}):
            timeline={e['event_id']:e for e in source if e['subject_ref']==subject}
            linked=links.get(subject)
            if linked:
                # Preserve the anonymous analytical unit while exposing the exact
                # server-linked account continuation, without merging visits.
                origin=min(stamp(r['occurred_at']) for r in timeline.values())
                for e in rows:
                    if e['subject_ref']==linked and not e.get('source_subject_ref') and stamp(e['occurred_at'])>=origin:
                        timeline.setdefault(e['event_id'],e)
            subject_rows=sorted(timeline.values(),key=lambda e:(e['occurred_at'],e['event_id']))
            states={}
            for e in subject_rows:states[e['event_name']]=e.get('status') or 'observed'
            current_states={}
            for e in subject_rows:
                if e.get('clock_status')=='anomalous' or stamp(e['occurred_at'])>as_of:continue
                name=e['event_name'];status=e.get('status')
                if name=='registration_completed':current_states.update(registration='registered',registration_attempt='completed')
                elif name in {'registration_started','registration_failed'}:current_states['registration_attempt']=status or ('started' if name.endswith('started') else 'failed')
                elif name in {'invitation_status_changed','invitation_application_submitted'}:current_states['invitation']=status or 'unknown'
                elif name=='app_first_open':current_states['installation']='first_open_observed'
                elif name in {'client_ready','client_failed','client_resumed'}:current_states['client']=name.removeprefix('client_')
                elif name=='trial_access_status_changed':current_states['trial_access']=status or 'unknown'
                elif name=='entitlement_status_changed':current_states['paid_access']=status or 'unknown'
                elif name=='payment_status_changed':current_states['payment']=status or 'unknown'
                elif name in {'task_started','task_completed','task_status_changed'}:current_states['task']=status or name.removeprefix('task_')
                elif name in {'feedback_submitted','action_recorded'}:current_states['feedback_self_report' if name=='feedback_submitted' else 'action_self_report']=e.get('properties',{}).get('value','reported')
                elif name=='result_viewed':current_states['delivery']='view_acknowledged'
            invitation=next((r for r in current_invitations if r['journey_ref']==subject or subject.startswith('user:') and r.get('consumed_by')==subject[5:]),None)
            if invitation:
                current_states['invitation']='expired' if invitation['status']=='approved' and invitation.get('expires_at') and stamp(invitation['expires_at'])<=as_of else invitation['status']
            journeys.append({'unit':unit,'subject_ref':subject,'identity':'ambiguous' if subject in ambiguous else 'verified' if subject.startswith('user:') or linked else 'unlinked','events':subject_rows,'dimensions':dimensions.get(subject,{}),'states':states,'current_states':current_states,'invitation_basis':invitation,'last_event_at':subject_rows[-1]['occurred_at']})
    first_open={e.get('installation_id') or e['subject_ref']:e for e in rows
                if e['event_name']=='app_first_open' and stamp(e['received_at'])<=as_of
                and stamp(e['occurred_at'])<=as_of and e.get('clock_status')!='anomalous'}
    linked_first_open=set()
    for journey in journeys:
        if journey['unit']!='anonymous_journey' or journey['identity']!='verified':continue
        starts=[stamp(e['occurred_at']) for e in journey['events'] if e['event_name']=='download_initiated'
                and e.get('properties',{}).get('flow_version')=='direct_download_v1']
        if starts:
            linked_first_open.update(e['event_id'] for e in journey['events']
                                     if e['event_name']=='app_first_open' and stamp(e['occurred_at'])>=min(starts))
    download_evidence={'file_receipt':'unavailable','installation':'unavailable',
                       'first_open_installations_observed':len(first_open),
                       'first_open_installations_without_download_journey':sum(e['event_id'] not in linked_first_open for e in first_open.values())}
    return {'download_evidence':download_evidence,'analyses':analyses,'journeys':journeys,'coverage':coverage,'definition':catalog,'projection_version':'journey-projection-v2','comparison_dimensions':catalog['dimensions']}


def snapshot_page(payload, *, view='funnels',analysis='acquisition',edge=None,dimension='branch',group=None,journey=None,cursor=0,sort='newest',limit=25):
    """Presentation selection/pagination over exactly one retained server snapshot."""
    if view not in {'funnels','cohorts','journeys','collection'} or sort not in {'newest','oldest'} or dimension not in payload.get('comparison_dimensions',CATALOG['dimensions']):
        raise ValueError('Unsupported snapshot selection')
    if not 0<=cursor<=100000 or not 1<=limit<=100:raise ValueError('Bounded page required')
    result=dict(payload)
    rows=payload['journeys']
    definition=next((a for a in payload['analyses'] if a['id']==analysis),None)
    if definition is None:raise ValueError('Analysis unavailable')
    edge_definition=next((e for e in definition['edges'] if e['id']==edge),{}) if edge else {}
    unit=edge_definition.get('unit',definition.get('unit','verified_account'))
    rows=[r for r in rows if r.get('unit','verified_account' if r['subject_ref'].startswith('user:') else 'anonymous_journey')==unit]
    if unit=='anonymous_journey' and payload.get('projection_version')!='journey-projection-v2' and view=='journeys':
        raise LookupError('Historical snapshot lacks anonymous journey detail; refresh explicitly')
    frozen_dimensions=payload.get('comparison_dimensions', ['branch','source','role','scenario','has_current_task','page_version'])
    if dimension not in frozen_dimensions:raise LookupError('Historical snapshot does not support this comparison; refresh explicitly')
    selected_edge=next((e for e in definition['edges'] if e['id']==edge),None) if edge else None
    if edge and selected_edge is None:raise ValueError('Transition unavailable')
    subjects=selected_edge['subjects'] if selected_edge else None
    if group:
        candidate=next((g for g in (selected_edge or {}).get('groups',{}).get(dimension,[]) if g['id']==group),None)
        if candidate is None:raise ValueError('Group unavailable in this transition')
        subjects=candidate['subjects']
    if subjects is not None:rows=[r for r in rows if r['subject_ref'] in subjects]
    rows=sorted(rows,key=lambda r:(r['last_event_at'],r['subject_ref']),reverse=sort=='newest')
    result['dimension_values']={key:sorted({r['dimensions'].get(key,'unknown') for r in payload['journeys']}) for key in ('source','role','scenario','has_current_task','page_version')}
    result['journeys_total']=len(rows);result['journeys_page']=True;result['cursor']=cursor
    if view!='journeys':result['journeys']=[]
    elif journey:
        result['journeys']=[r for r in rows if r['subject_ref']==journey]
        result['selection_status']='selected' if result['journeys'] else 'not_in_result'
    else:result['journeys']=rows[cursor:cursor+limit]
    result['journeys']=[row|{'transition_state':subjects.get(row['subject_ref']) if subjects is not None else None} for row in result['journeys']]
    # Membership remains server-owned; a paginated reply must not leak every
    # subject ID through otherwise aggregate metric/group dictionaries.
    result['analyses']=[analysis_row|{'edges':[
        {key:value for key,value in edge_row.items() if key not in {'subjects','groups'}}|{'groups':{key:[{k:v for k,v in group_row.items() if k!='subjects'} for group_row in groups] for key,groups in edge_row.get('groups',{}).items()}}
        for edge_row in analysis_row['edges']]} for analysis_row in payload['analyses']]
    return result
