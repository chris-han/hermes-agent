"""Minimal invitation eligibility owner. No messaging, CRM or inferred approval."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from agents.auth_db_postgres import connection, write_transaction
from agents.operations_store import capture_fact
from agents.operations_journey import utc

DDL='''
CREATE TABLE IF NOT EXISTS semantier_identity.trial_invitations (
 application_ref text PRIMARY KEY, journey_ref text UNIQUE NOT NULL,
 traffic_class text NOT NULL DEFAULT 'REAL' CHECK(traffic_class IN ('REAL','TEST')),
 status text NOT NULL CHECK(status IN ('requested','approved','rejected','revoked','expired','consumed')),
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 expires_at timestamptz, consumed_by text REFERENCES semantier_identity.users(user_id));
ALTER TABLE semantier_identity.trial_invitations ADD COLUMN IF NOT EXISTS traffic_class text NOT NULL DEFAULT 'REAL';
CREATE TABLE IF NOT EXISTS semantier_identity.trial_invitation_history (
 event_ref text PRIMARY KEY, application_ref text NOT NULL REFERENCES semantier_identity.trial_invitations,
 status text NOT NULL, actor_user_id text, occurred_at timestamptz NOT NULL DEFAULT now());
'''


def apply(journey_ref):
    with write_transaction(),connection() as c:
        c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",('trial:'+journey_ref,))
        row=c.execute('SELECT application_ref,status FROM semantier_identity.trial_invitations WHERE journey_ref=%s',(journey_ref,)).fetchone()
        if row:return row
        ref=uuid4().hex
        c.execute("INSERT INTO semantier_identity.trial_invitations(application_ref,journey_ref,status) VALUES(%s,%s,'requested')",(ref,journey_ref))
        c.execute("INSERT INTO semantier_identity.trial_invitation_history(event_ref,application_ref,status) VALUES(%s,%s,'requested')",('application:'+ref,ref))
        # The subject is a server-issued anonymous journey, not an inferred user.
        from agents.operations_store import append_event
        from agents.operations_journey import fingerprint
        now=utc(datetime.now(timezone.utc))
        event=dict(event_id='invitation:'+ref,schema_version='journey-event-v1',event_name='invitation_application_submitted',producer_id='invitation',producer_version='v1',surface='unknown',subject_ref=journey_ref,occurred_at=now,received_at=now,evidence_type='server_fact',receipt_ref='application:'+ref,properties={},status='requested',clock_status='accepted')
        try:
            with c.transaction():append_event(event)
        except Exception:pass
        return {'application_ref':ref,'status':'requested'}


def current(journey_ref):
    with connection() as c:
        row=c.execute('SELECT application_ref,status,expires_at FROM semantier_identity.trial_invitations WHERE journey_ref=%s',(journey_ref,)).fetchone()
        if not row:return {'status':'not_applied'}
        if row['expires_at'] and row['expires_at']<datetime.now(timezone.utc) and row['status']=='approved':row['status']='expired'
        row['expires_at']=utc(row['expires_at']) if row['expires_at'] else None
        return row


def require_approved(c,journey_ref):
    row=c.execute('SELECT * FROM semantier_identity.trial_invitations WHERE journey_ref=%s FOR UPDATE',(journey_ref,)).fetchone()
    if not row or row['status']!='approved' or row['expires_at'] is None or row['expires_at']<=datetime.now(timezone.utc):
        raise PermissionError('Approved trial invitation required')
    return row


def decide(*,application_ref,actor_user_id,status,traffic_class=None):
    if traffic_class not in {None,"REAL","TEST"}:raise ValueError("Unsupported traffic class")
    if status not in {'approved','rejected','revoked'}:raise ValueError('Unsupported invitation decision')
    with write_transaction(),connection() as c:
        from agents.operations_admin import is_operations_admin
        import json
        actor=c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s',(actor_user_id,)).fetchone()
        if not actor or not is_operations_admin(json.loads(actor['payload_json'])):raise PermissionError('Current platform administrator required')
        row=c.execute('SELECT * FROM semantier_identity.trial_invitations WHERE application_ref=%s FOR UPDATE',(application_ref,)).fetchone()
        if row is None:raise LookupError('Application unavailable')
        if traffic_class is None:traffic_class=row['traffic_class']
        if row['status']=='consumed':raise ValueError('Consumed invitation cannot be rewritten')
        if row['status']==status:return {'application_ref':application_ref,'status':status}
        now=datetime.now(timezone.utc);expires=now+timedelta(days=7) if status=='approved' else None
        c.execute('UPDATE semantier_identity.trial_invitations SET status=%s,expires_at=%s,updated_at=%s,traffic_class=%s WHERE application_ref=%s',(status,expires,now,traffic_class,application_ref))
        receipt=uuid4().hex
        c.execute('INSERT INTO semantier_identity.trial_invitation_history(event_ref,application_ref,status,actor_user_id) VALUES(%s,%s,%s,%s)',(receipt,application_ref,status,actor_user_id))
        from agents.operations_store import append_event
        event=dict(event_id='invitation-decision:'+receipt,schema_version='journey-event-v1',event_name='invitation_status_changed',producer_id='invitation',producer_version='v1',surface='unknown',subject_ref=row['journey_ref'],occurred_at=utc(now),received_at=utc(now),evidence_type='server_fact',receipt_ref=receipt,properties={},status=status,clock_status='accepted')
        try:
            with c.transaction():append_event(event)
        except Exception:pass
        return {'application_ref':application_ref,'status':status}


def register_invited(*,journey_ref,login_name,password,device_intent=None,model_preference='platform_recommended'):
    """Invitation eligibility and canonical signup commit atomically."""
    from agents.gateway_identity import register_password_user
    with write_transaction(),connection() as c:
        invitation=require_approved(c,journey_ref)
        surface='workspace'
        if device_intent:
            device=c.execute("SELECT user_id FROM semantier_gateway.device_connections WHERE intent_ref=%s AND status='pending' AND expires_at>now()",(device_intent,)).fetchone()
            if not device or device['user_id'] is not None:raise ValueError('Device registration context expired; restart sign-in from Windows')
            surface='windows'
        user=register_password_user(login_name=login_name,password=password,model_preference=model_preference)
        from agents import auth_db
        users=auth_db.load_users()
        users[user['user_id']]['registration_origin']=surface
        if invitation.get('traffic_class')=='TEST':users[user['user_id']]['test_account']=True
        auth_db.save_users(users)
        user.update(users[user['user_id']])
        now=datetime.now(timezone.utc)
        c.execute("UPDATE semantier_identity.trial_invitations SET status='consumed',consumed_by=%s,updated_at=%s WHERE application_ref=%s",(user['user_id'],now,invitation['application_ref']))
        c.execute("INSERT INTO semantier_identity.trial_invitation_history(event_ref,application_ref,status,actor_user_id) VALUES(%s,%s,'consumed',%s)",('consumed:'+invitation['application_ref'],invitation['application_ref'],user['user_id']))
        capture_fact(name='invitation_status_changed',user_id=user['user_id'],producer='invitation',receipt_ref=invitation['application_ref'],status='consumed',occurred_at=utc(now))
        capture_fact(name='identity_linked',user_id=user['user_id'],producer='identity',receipt_ref=journey_ref+':'+user['user_id'],linked_subject_ref=journey_ref,occurred_at=utc(now))
        capture_fact(name='registration_completed',user_id=user['user_id'],producer='identity',receipt_ref=user['user_id'],surface=surface,occurred_at=user['created_at'],properties={'registration_origin':surface,'origin_basis':'server_device_intent' if device_intent else 'browser_registration'})
        # Observe the access this signup owner actually committed. This is not
        # wallet funding, a subscription, or an invented trial credit grant.
        try:
            with c.transaction():
                access=c.execute("SELECT g.permissions FROM semantier_gateway.account_access a JOIN semantier_gateway.grants g USING(organization_id,user_id) WHERE a.user_id=%s AND a.organization_id=%s AND a.status='ACTIVE'",(user['user_id'],user.get('organization_id'))).fetchone()
                if access and 'model:invoke' in access['permissions']:
                    capture_fact(name='trial_access_status_changed',user_id=user['user_id'],producer='entitlement',receipt_ref=invitation['application_ref']+':application-access',status='active',occurred_at=utc(now),properties={'access_kind':'invited_application_access','monetary_credit_granted':False,'application_ref':invitation['application_ref']})
        except Exception:
            import logging
            logging.getLogger(__name__).warning('Invited access observation unavailable')


        return user
