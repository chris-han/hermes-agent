"""Target-bound observations and manually submitted, sanitized diagnostics."""
from datetime import datetime,timedelta,timezone
import json
from uuid import uuid4
from contracts.client_operations import canonical,digest,sanitize,summarize_attempts,MAX_DIAGNOSTIC_BYTES
from .publication import OperationError


class Feedback:
    def __init__(self,store,channel):self.store,self.channel=store,channel

    def enroll(self,user_id,installation_ref,consent_version):
        if consent_version!='semantier-operational-reporting/1':raise OperationError(422,'explicit_consent_version_required')
        with self.store.connection() as c:
            old=c.execute('SELECT * FROM installations WHERE installation_ref=%s FOR UPDATE',(installation_ref,)).fetchone()
            if old and old['user_id']!=user_id:raise OperationError(409,'installation_scope_conflict')
            c.execute('INSERT INTO installations(installation_ref,user_id,channel,consent_version) VALUES(%s,%s,%s,%s) ON CONFLICT(installation_ref) DO UPDATE SET revoked_at=NULL,consent_version=excluded.consent_version,consented_at=now()', (installation_ref,user_id,self.channel,consent_version))
            c.execute('UPDATE principals SET revoked_at=now() WHERE installation_ref=%s AND kind=\'reporting\'',(installation_ref,))
            self.store.audit(c,user_id,'reporting_consent',installation_ref,{'consent_version':consent_version})
            expires=datetime.now(timezone.utc)+timedelta(days=30)
            token=self.store.issue_in(c,kind='reporting',channel=self.channel,user_id=user_id,installation_ref=installation_ref,expires_at=expires)
        return {'installation_ref':installation_ref,'capability':token,'expires_at':expires.isoformat(),'purpose':'update_results_only','destination':'Semantier Cloud'}

    def withdraw(self,user_id,installation_ref):
        with self.store.connection() as c:
            row=c.execute('UPDATE installations SET revoked_at=now() WHERE installation_ref=%s AND user_id=%s RETURNING installation_ref',(installation_ref,user_id)).fetchone()
            if not row:raise OperationError(404,'installation_unavailable')
            c.execute('UPDATE principals SET revoked_at=now() WHERE installation_ref=%s',(installation_ref,))
            self.store.audit(c,user_id,'reporting_withdrawn',installation_ref,{})
        return {'revoked':True,'installation_ref':installation_ref}

    def events(self,principal,batch):
        receipts=[]
        with self.store.connection() as c:
            owner=c.execute('SELECT * FROM installations WHERE installation_ref=%s AND revoked_at IS NULL FOR UPDATE',(principal['installation_ref'],)).fetchone()
            if not owner:raise OperationError(401,'reporting_revoked')
            for event in batch.events:
                data=event.model_dump();key=(event.installation_ref,event.event_id)
                if event.installation_ref!=owner['installation_ref']:
                    receipts.append({'event_id':event.event_id,'result':'rejected','code':'installation_scope_mismatch'});continue
                release=c.execute('SELECT body FROM releases WHERE release_id=%s',(event.release_id,)).fetchone()
                if not release:
                    receipts.append({'event_id':event.event_id,'result':'rejected','code':'release_unavailable'});continue
                target=release['body']['manifest']['packages'][0]
                if event.target_version!=target['version'] or event.target_sha256!=target['artifact']['sha256'] or event.package_family!=release['body']['native_receipt']['package_family']:
                    receipts.append({'event_id':event.event_id,'result':'rejected','code':'target_binding_mismatch'});continue
                old=c.execute('SELECT body_hash FROM events WHERE installation_ref=%s AND event_id=%s',key).fetchone()
                result='accepted'
                if old:result='duplicate' if old['body_hash']==digest(data) else 'rejected'
                else:
                    attempt=c.execute('SELECT body FROM events WHERE installation_ref=%s AND update_id=%s LIMIT 1',(event.installation_ref,event.update_id)).fetchone()
                    fields=('release_id','target_sha256','from_version','target_version','package_family')
                    if attempt and any(attempt['body'][k]!=data[k] for k in fields):result='rejected'
                    else:c.execute('INSERT INTO events(installation_ref,event_id,update_id,release_id,body_hash,body) VALUES(%s,%s,%s,%s,%s,%s::jsonb)',(*key,event.update_id,event.release_id,digest(data),json.dumps(data)))
                receipts.append({'event_id':event.event_id,'result':result})
        return {'items':receipts,'source':'client_reported'}

    def attempts(self,release_id=None,days=7):
        with self.store.connection() as c:
            rows=c.execute('SELECT * FROM events WHERE (%s::text IS NULL OR release_id=%s) ORDER BY received_at,event_id',(release_id,release_id)).fetchall()
            grouped={}
            rank={'pending':0,'unknown':1,'failed':2,'target_confirmed':3}
            for row in rows:
                key=(row['installation_ref'],row['update_id']);old=grouped.get(key)
                if old is None or rank[row['body']['outcome']]>=rank[old['body']['outcome']]:grouped[key]=row
            cutoff=datetime.now(timezone.utc)-timedelta(days=days)
            items=[row for row in grouped.values() if row['received_at']>=cutoff]
            return {'items':items,'summary':summarize_attempts(r['body']['outcome'] for r in items),'reporting_installations':len({r['installation_ref'] for r in items}),'coverage':'observed_only','source':'client_reported'}

    def submit(self,user_id,diagnostic):
        body=diagnostic.model_dump()
        if len(canonical(body))>MAX_DIAGNOSTIC_BYTES:raise OperationError(413,'diagnostic_too_large')
        # Independently redact on the server; the client preview is not authority.
        transformations=0
        for key,value in list(body['sections'].items()):
            body['sections'][key],n=sanitize(value);transformations+=n
        body['note'],n=sanitize(body['note']);transformations+=n
        payload=canonical({'sections':body.pop('sections'),'note':body.pop('note')}).decode()
        request_hash=digest(diagnostic.model_dump())
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(user_id+':'+diagnostic.submission_key,))
            old=c.execute('SELECT * FROM diagnostics WHERE user_id=%s AND submission_key=%s',(user_id,diagnostic.submission_key)).fetchone()
            if old:
                if old['body_hash']!=request_hash:raise OperationError(409,'submission_idempotency_conflict')
                return self.receipt(old)
            if body['release_id'] and not c.execute('SELECT 1 FROM releases WHERE release_id=%s',(body['release_id'],)).fetchone():raise OperationError(422,'release_unavailable')
            if body['update_id']:
                related=c.execute('SELECT 1 FROM events e JOIN installations i USING(installation_ref) WHERE e.update_id=%s AND e.release_id=%s AND i.user_id=%s',(body['update_id'],body['release_id'],user_id)).fetchone()
                if not related:raise OperationError(422,'diagnostic_update_relation_mismatch')
            identifier=uuid4().hex;body['server_redactions']=transformations
            row=c.execute('INSERT INTO diagnostics(diagnostic_id,user_id,submission_key,body_hash,release_id,update_id,metadata,payload,classification) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb) RETURNING *',(identifier,user_id,diagnostic.submission_key,request_hash,body['release_id'],body['update_id'],json.dumps(body),payload,json.dumps({'category':'unclassified','severity':'unassessed','status':'new','fix_release_id':None}))).fetchone()
            self.store.audit(c,user_id,'diagnostic_submitted',identifier,{'payload_bytes':len(payload.encode()),'server_redactions':transformations})
            return self.receipt(row)

    @staticmethod
    def receipt(row):
        return {k:row[k] for k in ('diagnostic_id','received_at','expires_at','deleted_at','release_id','revision')}

    def erase(self,user_id,identifier):
        with self.store.connection() as c:
            row=c.execute('UPDATE diagnostics SET payload=NULL,deleted_at=coalesce(deleted_at,now()),revision=revision+CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END WHERE diagnostic_id=%s AND user_id=%s RETURNING *',(identifier,user_id)).fetchone()
            if not row:raise OperationError(404,'diagnostic_unavailable')
            self.store.audit(c,user_id,'diagnostic_payload_removed',identifier,{})
            return self.receipt(row)

    def classify(self,user_id,identifier,classification):
        with self.store.connection() as c:
            if classification.fix_release_id and not c.execute('SELECT 1 FROM releases WHERE release_id=%s',(classification.fix_release_id,)).fetchone():raise OperationError(422,'fix_release_unavailable')
            body=classification.model_dump(exclude={'expected_revision'});body['operator_note']=sanitize(body['operator_note'])[0]
            row=c.execute('UPDATE diagnostics SET classification=%s::jsonb,revision=revision+1 WHERE diagnostic_id=%s AND revision=%s RETURNING *',(json.dumps(body),identifier,classification.expected_revision)).fetchone()
            if not row:raise OperationError(409,'classification_revision_conflict')
            self.store.audit(c,user_id,'diagnostic_classified',identifier,body)
            return row
