"""Possession-bound non-inference authorization; no provider/config mutations."""
from datetime import datetime,timedelta,timezone
import hashlib
import hmac
import json
import secrets
from uuid import uuid4
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .artifact_io import OperationError


class FeedbackAuthorizations:
    def __init__(self,store,channel,key,portal):
        if len(key)!=32:raise ValueError('Separate client authorization AEAD key required')
        from urllib.parse import urlsplit
        url=urlsplit(portal)
        if (url.scheme!='https' and not(url.scheme=='http' and url.hostname in {'127.0.0.1','localhost','::1'}) or url.username or url.password or url.path not in {'','/'} or url.query or url.fragment):raise ValueError('Validated feedback portal origin required')
        self.store,self.channel,self.crypto,self.portal=store,channel,AESGCM(key),portal.rstrip('/')

    def start(self,request):
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.store.schema+':feedback-authorization',))
            old=c.execute('SELECT * FROM authorizations WHERE intent_ref=%s',(request.intent_ref,)).fetchone()
            if old:
                if any(old[k]!=v for k,v in request.model_dump().items()):raise OperationError(409,'authorization_identity_conflict')
                return self.public(old)
            active=c.execute("SELECT count(*) AS n FROM authorizations WHERE expires_at>now() AND status='pending'").fetchone()['n']
            if active>=1000:raise OperationError(429,'authorization_capacity')
            row=c.execute('INSERT INTO authorizations(intent_ref,proof_challenge,purpose,profile_ref) VALUES(%s,%s,%s,%s) RETURNING *',(request.intent_ref,request.proof_challenge,request.purpose,request.profile_ref)).fetchone()
        return self.public(row)

    def public(self,row):
        return {'intent_ref':row['intent_ref'],'purpose':row['purpose'],'profile_ref':row['profile_ref'],'status':row['status'] if row['expires_at']>datetime.now(timezone.utc) else 'expired','expires_at':row['expires_at'].isoformat(),'authorization_url':self.portal+'/client-feedback/authorize?intent_ref='+row['intent_ref']}

    def view(self,intent):
        with self.store.connection() as c:row=c.execute('SELECT * FROM authorizations WHERE intent_ref=%s',(intent,)).fetchone()
        if not row:raise OperationError(404,'authorization_unavailable')
        return self.public(row)

    def decide(self,intent,user_id,decision):
        with self.store.connection() as c:
            row=c.execute('SELECT * FROM authorizations WHERE intent_ref=%s FOR UPDATE',(intent,)).fetchone()
            if not row or row['expires_at']<=datetime.now(timezone.utc):raise OperationError(409,'authorization_expired')
            if row['status']!='pending':
                if row['user_id']==user_id and row['status']==('approved' if decision=='approve' else 'denied'):return self.public(row)
                raise OperationError(409,'authorization_already_decided')
            if decision=='deny':
                c.execute("UPDATE authorizations SET status='denied',user_id=%s WHERE intent_ref=%s",(user_id,intent));return {**self.public(row),'status':'denied'}
            expiry=datetime.now(timezone.utc)+timedelta(days=30);installation_ref=None
            if row['purpose']=='reporting':
                installation_ref=uuid4().hex
                c.execute('INSERT INTO installations(installation_ref,user_id,channel,consent_version) VALUES(%s,%s,%s,%s)',(installation_ref,user_id,self.channel,'semantier-operational-reporting/1'))
            token=self.store.issue_in(c,kind=row['purpose'],channel=self.channel,user_id=user_id,installation_ref=installation_ref,expires_at=expiry)
            payload=json.dumps({'capability':token,'installation_ref':installation_ref,'capability_expires_at':expiry.isoformat()}).encode()
            nonce=secrets.token_bytes(12);cipher=self.crypto.encrypt(nonce,payload,intent.encode())
            c.execute("UPDATE authorizations SET status='approved',user_id=%s,ciphertext=%s,nonce=%s,token_hash=%s WHERE intent_ref=%s",(user_id,cipher,nonce,hashlib.sha256(token.encode()).hexdigest(),intent))
            self.store.audit(c,user_id,'feedback_authorized',intent,{'purpose':row['purpose'],'profile_ref':row['profile_ref']})
        return {**self.public(row),'status':'approved'}

    @staticmethod
    def possession(row,proof):
        if not row or not hmac.compare_digest(row['proof_challenge'],hashlib.sha256(proof.encode()).hexdigest()):raise OperationError(401,'authorization_possession_required')

    def poll(self,intent,proof):
        with self.store.connection() as c:row=c.execute('SELECT * FROM authorizations WHERE intent_ref=%s',(intent,)).fetchone()
        self.possession(row,proof);result=self.public(row)
        if result['status']=='approved':result.update(json.loads(self.crypto.decrypt(bytes(row['nonce']),bytes(row['ciphertext']),intent.encode())))
        return result

    def cancel(self,intent,proof):
        with self.store.connection() as c:
            row=c.execute('SELECT * FROM authorizations WHERE intent_ref=%s FOR UPDATE',(intent,)).fetchone();self.possession(row,proof)
            if row['token_hash']:c.execute('UPDATE principals SET revoked_at=now() WHERE token_hash=%s',(row['token_hash'],))
            c.execute("UPDATE authorizations SET status='cancelled',ciphertext=NULL,nonce=NULL WHERE intent_ref=%s",(intent,))
        return {**self.public(row),'status':'cancelled'}
