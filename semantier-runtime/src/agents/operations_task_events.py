"""Task-owner facts, written in the same execution transaction as the job state."""
from datetime import datetime,timezone
from psycopg.types.json import Jsonb
from agents.operations_journey import fingerprint,utc


def capture_task(conn, *, user_id, job):
    if job.get('kind')!='independent_task':return
    state=job.get('state')
    name='task_completed' if state=='completed' else 'task_started' if state=='running' else 'task_status_changed'
    receipt=job['id']+':'+str(job.get('run_id') or 'admission')+':'+str(state)
    eid='task:'+fingerprint(receipt)
    now=datetime.now(timezone.utc)
    classification=job.get('value_purpose','unknown')
    event=dict(event_id=eid,schema_version='journey-event-v1',event_name=name,producer_id='task',producer_version='v1',surface='server',subject_ref='user:'+user_id,occurred_at=job.get('updated_at') or utc(now),received_at=utc(now),evidence_type='server_fact',receipt_ref=receipt,status=state,clock_status='accepted',properties={'task_id':job['id'],'attempt_id':str(job.get('run_id') or 'admission'),'classification':classification,'classification_basis':'declared_task_intent' if classification!='unknown' else 'unavailable'})
    try:
        with conn.transaction():
            conn.execute('INSERT INTO semantier_operations.events VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(eid,fingerprint({k:v for k,v in event.items() if k!='received_at'}),event['subject_ref'],name,event['occurred_at'],now,Jsonb(event)))
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Task observation unavailable')
