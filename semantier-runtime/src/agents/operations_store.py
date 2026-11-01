"""PostgreSQL operations evidence owner. Explicit migration; append-only events/snapshots."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from uuid import uuid4
from psycopg.types.json import Jsonb
from agents.auth_db_postgres import connection
from agents.operations_journey import fingerprint, projection, stamp, utc

COLLECTION = json.loads((__import__('pathlib').Path(__file__).parent / 'operations/collection-definition-v2.json').read_text())

DDL = '''
CREATE SCHEMA IF NOT EXISTS semantier_operations;
CREATE TABLE IF NOT EXISTS semantier_operations.events (
 event_id text PRIMARY KEY, payload_hash text NOT NULL, subject_ref text NOT NULL,
 event_name text NOT NULL, occurred_at timestamptz NOT NULL, received_at timestamptz NOT NULL,
 payload jsonb NOT NULL);
CREATE INDEX IF NOT EXISTS operations_event_subject ON semantier_operations.events(subject_ref,received_at,event_id);
CREATE TABLE IF NOT EXISTS semantier_operations.handoffs (ref_hash text PRIMARY KEY,journey_ref text NOT NULL,expires_at timestamptz NOT NULL,claimed_at timestamptz);
CREATE TABLE IF NOT EXISTS semantier_operations.snapshots (
 snapshot_id text PRIMARY KEY, created_at timestamptz NOT NULL, definition_version text NOT NULL,
 scope_hash text NOT NULL, payload jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_operations.producers (
 producer_id text PRIMARY KEY, status text NOT NULL CHECK(status IN ('CONNECTED','UNAVAILABLE','NOT_IMPLEMENTED')),
 coverage_from timestamptz, version text, verification_ref text, updated_at timestamptz NOT NULL DEFAULT now());
ALTER TABLE semantier_operations.producers ADD COLUMN IF NOT EXISTS stage_coverage jsonb NOT NULL DEFAULT '{}'::jsonb;
CREATE TABLE IF NOT EXISTS semantier_operations.test_traffic (
 subject_ref text PRIMARY KEY, actor_user_id text NOT NULL REFERENCES semantier_identity.users(user_id),
 verification_ref text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS semantier_operations.receipt_counts (
 producer_id text NOT NULL, kind text NOT NULL, count bigint NOT NULL DEFAULT 0,
 last_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(producer_id,kind));
'''
PRODUCERS = ('landing','invitation','identity','workspace','windows','commerce','entitlement','task')


def migrate(dsn: str, *, app_role='semantier_app'):
    import psycopg
    from psycopg import sql
    with psycopg.connect(dsn) as c:
        c.execute("SELECT pg_advisory_xact_lock(hashtext('semantier-operations-migration'))")
        c.execute(DDL)
        from agents.trial_invitations import DDL as invitation_ddl
        c.execute(invitation_ddl)
        c.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE ON semantier_identity.trial_invitations TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT,INSERT ON semantier_identity.trial_invitation_history TO {}').format(sql.Identifier(app_role)))
        for name in PRODUCERS:
            c.execute("INSERT INTO semantier_operations.producers(producer_id,status) VALUES(%s,'NOT_IMPLEMENTED') ON CONFLICT DO NOTHING",(name,))
        c.execute(sql.SQL('GRANT USAGE ON SCHEMA semantier_operations TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT,INSERT ON semantier_operations.events,semantier_operations.snapshots TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE ON semantier_operations.handoffs TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT ON semantier_operations.test_traffic TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT ON semantier_operations.producers TO {}').format(sql.Identifier(app_role)))
        c.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE ON semantier_operations.receipt_counts TO {}').format(sql.Identifier(app_role)))


def count_receipt(c, producer, kind):
    c.execute('''INSERT INTO semantier_operations.receipt_counts(producer_id,kind,count) VALUES(%s,%s,1)
        ON CONFLICT(producer_id,kind) DO UPDATE SET count=semantier_operations.receipt_counts.count+1,last_at=now()''',(producer,kind))


def append_event(event: dict) -> str:
    # Receipt time is the receiving owner clock and does not affect retry identity.
    valid={k:v for k,v in event.items() if k!='received_at'}
    digest=fingerprint(valid)
    collision=False
    with connection() as c:
        with c.transaction():
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('ops-event:'+event['event_id'],))
            old=c.execute('SELECT payload_hash FROM semantier_operations.events WHERE event_id=%s',(event['event_id'],)).fetchone()
            if old:
                collision=old['payload_hash']!=digest
                count_receipt(c,event['producer_id'],'collision' if collision else 'duplicate')
                outcome='duplicate'
            else:
                c.execute('INSERT INTO semantier_operations.events VALUES(%s,%s,%s,%s,%s,%s,%s)',
                          (event['event_id'],digest,event['subject_ref'],event['event_name'],stamp(event['occurred_at']),stamp(event['received_at']),Jsonb(event)))
                count_receipt(c,event['producer_id'],'accepted')
                outcome='accepted'
    if collision:raise ValueError('Event identity collision')
    return outcome


def capture_fact(*, name, user_id, producer, receipt_ref, status=None, surface='unknown', properties=None, occurred_at=None, linked_subject_ref=None, evidence_type='server_fact'):
    """Called only by core owners, with an owner receipt after successful mutation.

    Join an existing identity transaction when present. A savepoint isolates
    telemetry failure from its business transaction; no sensitive exception prose.
    """
    now=datetime.now(timezone.utc)
    event=dict(event_id='fact:'+fingerprint([producer,name,receipt_ref,status]),schema_version='journey-event-v1',
               event_name=name,producer_id=producer,producer_version='v1',surface=surface,
               subject_ref='user:'+user_id,occurred_at=utc(stamp(occurred_at) if occurred_at else now),received_at=utc(now),
               evidence_type=evidence_type,receipt_ref=receipt_ref,properties=properties or {},status=status,clock_status='accepted')
    if linked_subject_ref:
        event['subject_ref']=linked_subject_ref;event['linked_subject_ref']='user:'+user_id
    # Stable receipt IDs are idempotent even when the caller retries later.
    if not occurred_at:
        with connection() as c:
            try:
                with c.transaction():
                    previous=c.execute('SELECT payload FROM semantier_operations.events WHERE event_id=%s',(event['event_id'],)).fetchone()
                    if previous:event['occurred_at']=previous['payload']['occurred_at']
            except Exception:
                return False
    try:
        append_event(event)
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Operations fact capture unavailable: producer=%s',producer)
        return False


def collection(c):
    producers=c.execute('SELECT * FROM semantier_operations.producers ORDER BY producer_id').fetchall()
    counts=c.execute('SELECT * FROM semantier_operations.receipt_counts ORDER BY producer_id,kind').fetchall()
    for p in producers:
        last=c.execute('SELECT max(received_at) AS last_at FROM semantier_operations.events WHERE payload->>\'producer_id\'=%s',(p['producer_id'],)).fetchone()
        p['last_received_at']=utc(last['last_at']) if last['last_at'] else None
        p['coverage_from']=utc(p['coverage_from']) if p['coverage_from'] else None
        p['updated_at']=utc(p['updated_at'])
        p['counts']={r['kind']:r['count'] for r in counts if r['producer_id']==p['producer_id']}
        p['traffic_state']='observed' if p['last_received_at'] else 'unknown_no_baseline'
        configured=p.pop('stage_coverage',{})
        p['collection_definition']=COLLECTION['version']
        # Preserve legacy qualification entries. Only unchanged producer semantics
        # can reuse them; direct-download Landing stages require v2 receipts.
        p['stages']=[]
        for name in COLLECTION['stages'][p['producer_id']]:
            key=COLLECTION['version']+':'+name
            basis=configured.get(key)
            if basis is None and p['producer_id']!='landing' and name in configured:
                basis=configured[name]|{'evidence_reuse':'EXACT_REUSE','collection_definition_origin':'collection-definition-v1'}
            p['stages'].append({'event_name':name,**(basis or {'status':'NOT_IMPLEMENTED','verification_ref':None,'coverage_from':None})})
        p['status']='CONNECTED' if all(s['status']=='CONNECTED' for s in p['stages']) else 'UNAVAILABLE'
        p['coverage_until']=min((s.get('coverage_until') or '' for s in p['stages']),default='') or None
        p['coverage_from']=max((s.get('coverage_from') or '' for s in p['stages']),default='') or None if p['status']=='CONNECTED' else None
        if p['coverage_from'] and p['coverage_until'] and stamp(p['coverage_from'])>stamp(p['coverage_until']):
            p['coverage_from']=p['coverage_until']=None
        anomalies=c.execute("SELECT count(*) AS retained_events,count(*) FILTER(WHERE payload->>'clock_status'='anomalous') AS clock_anomalies,count(*) FILTER(WHERE received_at-occurred_at>interval '5 minutes') AS received_after_five_minutes FROM semantier_operations.events WHERE payload->>'producer_id'=%s",(p['producer_id'],)).fetchone()
        p['counts'].update(anomalies)
    return producers


def snapshot(*, scope: dict, snapshot_id: str | None = None, definition_version: str | None = None) -> dict:
    scope_hash=fingerprint(scope)
    with connection() as c:
        if snapshot_id:
            row=c.execute('SELECT payload,scope_hash FROM semantier_operations.snapshots WHERE snapshot_id=%s',(snapshot_id,)).fetchone()
            if row is None:raise LookupError('Snapshot unavailable; refresh explicitly')
            if row['scope_hash']!=scope_hash:raise ValueError('Snapshot scope mismatch')
            if definition_version and row['payload']['definition_version']!=definition_version:raise ValueError('Snapshot definition mismatch')
            return row['payload']
        # Single repeatable-read transaction freezes sources, coverage and exclusion
        # identities. Subsequent detail reads use this retained payload only.
        c.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        as_of=c.execute('SELECT transaction_timestamp() AS at').fetchone()['at']
        data=c.execute('SELECT payload FROM semantier_operations.events WHERE received_at<=%s ORDER BY occurred_at,event_id',(as_of,)).fetchall()
        events=[r['payload'] for r in data]
        users=c.execute('SELECT user_id,payload_json FROM semantier_identity.users').fetchall()
        excluded={row['journey_ref'] for row in c.execute("SELECT journey_ref FROM semantier_identity.trial_invitations WHERE traffic_class='TEST'").fetchall()}
        excluded.update(r['subject_ref'] for r in c.execute('SELECT subject_ref FROM semantier_operations.test_traffic').fetchall())
        never_include=set()
        for row in users:
            u=json.loads(row['payload_json'])
            if u.get('platform_role') or u.get('ops_bootstrap') or u.get('demo_account') or u.get('test_account'):
                excluded.add('user:'+row['user_id'])
            if u.get('platform_role') or u.get('ops_bootstrap') or u.get('demo_account'):never_include.add('user:'+row['user_id'])
        links={e['subject_ref'] for e in events if e['event_name']=='identity_linked' and e.get('linked_subject_ref') in excluded}
        if scope.get('traffic','REAL')=='REAL':excluded |= links
        else:
            # TEST remains an explicitly separate set; it cannot inflate REAL.
            included=excluded|links
            excluded={e['subject_ref'] for e in events if e['subject_ref'] not in included}
            excluded |= never_include | {e['subject_ref'] for e in events if e['event_name']=='identity_linked' and e.get('linked_subject_ref') in never_include}
        invitations=c.execute('SELECT application_ref,journey_ref,status,expires_at,consumed_by,updated_at FROM semantier_identity.trial_invitations ORDER BY updated_at DESC,application_ref').fetchall()
        for invitation in invitations:
            invitation['updated_at']=utc(invitation['updated_at'])
            invitation['expires_at']=utc(invitation['expires_at']) if invitation['expires_at'] else None
        # Current owner state is separate from historical conversion events.
        result=projection(events,as_of=as_of,coverage=collection(c),excluded=excluded,scope=scope,current_invitations=invitations,definition_version=definition_version or 'metric-definition-v2')
        result['conditional_collection']=COLLECTION.get('conditional_stages',[])
        sid=uuid4().hex
        result.update(snapshot_id=sid,as_of=utc(as_of),scope=scope,scope_hash=scope_hash,
                      definition_version=result['definition']['version'],definition_hash=fingerprint(result['definition']),
                      projection_hash=fingerprint(__import__('pathlib').Path(__file__).with_name('operations_journey.py').read_text()),
                      watermark=max((e['received_at'] for e in events),default=None))
        c.execute('INSERT INTO semantier_operations.snapshots VALUES(%s,%s,%s,%s,%s)',(sid,as_of,result['definition_version'],scope_hash,Jsonb(result)))
        return result


def verify_stage(dsn, *, producer, event_name, status, version, verification_ref, coverage_from, coverage_until=None):
    """Explicit operator capability records checked collection readiness, not user facts."""
    if producer not in COLLECTION['stages'] or event_name not in COLLECTION['stages'][producer]:
        raise ValueError('Declared collection stage required')
    if status not in {'CONNECTED','UNAVAILABLE','NOT_IMPLEMENTED'} or not version or not verification_ref:
        raise ValueError('Explicit status/version/verification required')
    at=utc(stamp(coverage_from)) if coverage_from else None
    until=utc(stamp(coverage_until)) if coverage_until else None
    if status=='CONNECTED' and not at:raise ValueError('Connected stage requires coverage origin')
    if until and (not at or stamp(until)<stamp(at) or stamp(until)>datetime.now(timezone.utc)):
        raise ValueError('Coverage interval requires an ordered, observed end')
    import psycopg
    with psycopg.connect(dsn) as c:
        row=c.execute('SELECT stage_coverage FROM semantier_operations.producers WHERE producer_id=%s FOR UPDATE',(producer,)).fetchone()
        if not row:raise ValueError('Migration required')
        stages=row[0]
        stages[COLLECTION['version']+':'+event_name]={'status':status,'version':version,'verification_ref':verification_ref,'coverage_from':at,'coverage_until':until,'verified_at':utc(datetime.now(timezone.utc))}
        active=[stages.get(COLLECTION['version']+':'+name,stages.get(name,{}) if producer!='landing' else {}) for name in COLLECTION['stages'][producer]]
        ready=all(s.get('status')=='CONNECTED' for s in active)
        overall='CONNECTED' if ready else 'UNAVAILABLE'
        origin=max(s['coverage_from'] for s in active) if ready else None
        c.execute('UPDATE semantier_operations.producers SET stage_coverage=%s,status=%s,coverage_from=%s,version=%s,verification_ref=%s,updated_at=now() WHERE producer_id=%s',(Jsonb(stages),overall,origin,version,verification_ref,producer))
    return {'producer_id':producer,'event_name':event_name,'status':status,'producer_status':overall}


def record_rejected_observation(producer):
    """Aggregate only: never retain rejected payloads, credentials or identities."""
    try:
        with connection() as c:
            with c.transaction():count_receipt(c,producer,'rejected')
    except Exception:
        pass


def append_observation(event, *, verified_user_id=None):
    """Retry across cookie rotation only with an unambiguous server-owned link."""
    if verified_user_id:
        with connection() as c:
            old=c.execute('SELECT payload,payload_hash FROM semantier_operations.events WHERE event_id=%s',(event['event_id'],)).fetchone()
            if old and old['payload']['subject_ref']!=event['subject_ref']:
                targets=c.execute("SELECT DISTINCT payload->>'linked_subject_ref' AS target FROM semantier_operations.events WHERE subject_ref=%s AND event_name='identity_linked' AND payload->>'evidence_type'='server_fact'",(old['payload']['subject_ref'],)).fetchall()
                if {r['target'] for r in targets}=={'user:'+verified_user_id}:
                    replay=event|{'subject_ref':old['payload']['subject_ref']}
                    if fingerprint({k:v for k,v in replay.items() if k!='received_at'})==old['payload_hash']:
                        return append_event(replay)
    return append_event(event)


def classify_test_traffic(dsn, *, subject_ref, actor_user_id, verification_ref):
    """Migration-owner capability marks retained QA traffic; never creates an invitation.

    Immutable declaration; application credentials receive SELECT only. Caller must
    also name a current DB-backed platform administrator and a concrete run receipt.
    """
    if not verification_ref or len(verification_ref)>256:raise ValueError('Bounded qualification reference required')
    import psycopg
    with psycopg.connect(dsn) as c:
        actor=c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s FOR SHARE',(actor_user_id,)).fetchone()
        from agents.operations_admin import is_operations_admin
        if not actor or not is_operations_admin(json.loads(actor[0])):raise PermissionError('Current platform administrator required')
        if not c.execute('SELECT 1 FROM semantier_operations.events WHERE subject_ref=%s LIMIT 1',(subject_ref,)).fetchone():raise ValueError('Retained event subject required')
        c.execute('INSERT INTO semantier_operations.test_traffic(subject_ref,actor_user_id,verification_ref) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING',(subject_ref,actor_user_id,verification_ref))
    return {'subject_ref':subject_ref,'traffic_class':'TEST'}
