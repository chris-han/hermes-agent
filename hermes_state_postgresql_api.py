"""Explicit PostgreSQL stores for embedded API replay and response chaining."""
import hmac
import json
from pathlib import Path
import time
from psycopg import sql


class PostgreSQLAPIStore:
    def __init__(self,db_path=None):
        from hermes_constants import get_hermes_home
        from hermes_state_backend import open_session_db
        if db_path==':memory:': raise ValueError('Hosted API requires durable PostgreSQL storage')
        self.store=open_session_db(home=Path(db_path).parent if db_path else get_hermes_home())

    def close(self):
        self.store.close()


class PostgreSQLRunIdempotencyStore(PostgreSQLAPIStore):
    RETENTION_SECONDS=86400
    ACKNOWLEDGED_RETENTION_SECONDS=86400
    durable=True

    def _prune(self,conn):
        from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES
        now=time.time()
        conn.execute(sql.SQL('DELETE FROM {} WHERE (acknowledged_at<=%s OR (retention_until>0 AND retention_until<=%s) OR (retention_until<=0 AND updated_at<%s)) AND status_json::jsonb->>\'status\'=ANY(%s)').format(self.store.table('run_idempotency')),(now-self.ACKNOWLEDGED_RETENTION_SECONDS,now,now-self.RETENTION_SECONDS,list(TERMINAL_STATUSES)))

    def _record(self,row):
        return dict(run_id=row['run_id'],status=json.loads(row['status_json']),owner_pid=row['owner_pid'],owner_started=row['owner_started'],updated_at=row['updated_at'])

    def _lookup(self,conn,scope,key,fingerprint,retention_until=0):
        self._prune(conn)
        row=conn.execute(sql.SQL('SELECT * FROM {} WHERE scope=%s AND idempotency_key=%s FOR UPDATE').format(self.store.table('run_idempotency')),(scope,key)).fetchone()
        if row is None: return ('missing',None)
        same=hmac.compare_digest(row['fingerprint'],fingerprint)
        if same and retention_until:
            conn.execute(sql.SQL('UPDATE {} SET retention_until=GREATEST(retention_until,%s) WHERE scope=%s AND idempotency_key=%s').format(self.store.table('run_idempotency')),(max(0,float(retention_until)),scope,key))
        return ('reused' if same else 'conflict',self._record(row))

    def reserve(self,scope,key,fingerprint,run_id,status,*,owner_pid=0,owner_started=0,retention_until=0):
        with self.store._transaction() as conn:
            self.store._lock_conversation(conn,'run:'+scope+'\x1f'+key)
            outcome,old=self._lookup(conn,scope,key,fingerprint,retention_until)
            if old is not None: return outcome,old
            now=time.time()
            values=dict(scope=scope,idempotency_key=key,fingerprint=fingerprint,run_id=run_id,status_json=json.dumps(status,sort_keys=True,separators=(',',':')),
                owner_pid=int(owner_pid or 0),owner_started=int(owner_started or 0),retention_until=max(0,float(retention_until or 0)),created_at=now,updated_at=now)
            self.store._insert(conn,'run_idempotency',values)
            return 'created',self._record(values)

    def lookup(self,scope,key,fingerprint,*,retention_until=0):
        with self.store._transaction() as conn: return self._lookup(conn,scope,key,fingerprint,retention_until)

    def status_for_run(self,scope,run_id,*,retention_until=0):
        with self.store._transaction() as conn:
            if retention_until: conn.execute(sql.SQL('UPDATE {} SET retention_until=GREATEST(retention_until,%s) WHERE scope=%s AND run_id=%s').format(self.store.table('run_idempotency')),(max(0,float(retention_until)),scope,run_id))
            row=conn.execute(sql.SQL('SELECT * FROM {} WHERE scope=%s AND run_id=%s').format(self.store.table('run_idempotency')),(scope,run_id)).fetchone()
            if row is None: return None
            record=self._record(row);record.pop('run_id');return record

    def extend_retention(self,scope,run_id,until):
        if not until:return False
        with self.store._transaction() as conn:
            return conn.execute(sql.SQL('UPDATE {} SET retention_until=GREATEST(retention_until,%s) WHERE scope=%s AND run_id=%s').format(self.store.table('run_idempotency')),(max(0,float(until)),scope,run_id)).rowcount==1

    def owns_run(self,scope,run_id):
        return self.store._one(sql.SQL('SELECT 1 FROM {} WHERE scope=%s AND run_id=%s').format(self.store.table('run_idempotency')),(scope,run_id)) is not None

    def update_status(self,run_id,status):
        with self.store._transaction() as conn:
            self.store._update(conn,'run_idempotency',dict(status_json=json.dumps(status,sort_keys=True,separators=(',',':')),updated_at=time.time()),'run_id',run_id)

    def forget(self,scope,key):
        with self.store._transaction() as conn:
            conn.execute(sql.SQL('DELETE FROM {} WHERE scope=%s AND idempotency_key=%s').format(self.store.table('run_idempotency')),(scope,key))


class PostgreSQLResponseStore(PostgreSQLAPIStore):
    def __init__(self,max_size=100,db_path=None,max_identities=None):
        super().__init__(db_path)
        self._max_size,self._max_identities=max_size,max_identities
        with self.store._transaction() as conn:self._evict(conn)

    def get(self,response_id):
        with self.store._transaction() as conn:
            row=conn.execute(sql.SQL('SELECT data FROM {} WHERE response_id=%s').format(self.store.table('responses')),(response_id,)).fetchone()
            if row is None:return None
            try:data=json.loads(row['data'])
            except (ValueError,TypeError):
                conn.execute(sql.SQL('DELETE FROM {} WHERE response_id=%s').format(self.store.table('responses')),(response_id,));return None
            self.store._update(conn,'responses',{'accessed_at':time.time()},'response_id',response_id)
            return data

    def put(self,response_id,data):
        with self.store._transaction() as conn:
            query=sql.SQL('INSERT INTO {}(response_id,data,accessed_at) VALUES(%s,%s,%s) ON CONFLICT(response_id) ').format(self.store.table('responses'))
            query+=sql.SQL('DO NOTHING' if response_id.startswith('idem:') else 'DO UPDATE SET data=excluded.data,accessed_at=excluded.accessed_at')
            conn.execute(query,(response_id,json.dumps(data,default=str),time.time()))
            if response_id.startswith('idem:'):
                response=data.get('response')
                if response is not None:
                    conn.execute(sql.SQL('UPDATE {} SET settled_at=%s,response_id=%s WHERE request_key=%s AND settled_at IS NULL').format(self.store.table('response_keys')),(time.time(),response.get('id') if isinstance(response,dict) else None,response_id))
            self._evict(conn)

    def _forget(self,conn,keys):
        for table,column in [('response_output_order','request_key'),('response_admissions','request_key'),('responses','response_id'),('response_keys','request_key')]:
            conn.execute(sql.SQL('DELETE FROM {} WHERE {}=ANY(%s)').format(self.store.table(table),sql.Identifier(column)),(keys,))

    def _settle(self,conn):
        from hermes_state_terminal import ADMISSION_PREFIX
        from gateway.platforms.api_server_response_identity import durable_response_id
        rows=conn.execute(sql.SQL('SELECT k.request_key,a.admission_id,k.created_at FROM {} k JOIN {} a USING(request_key) WHERE k.settled_at IS NULL AND (EXISTS(SELECT 1 FROM {} s WHERE s.admission_id=a.admission_id AND s.status=\'terminal\') OR EXISTS(SELECT 1 FROM {} m WHERE m.key=%s||a.admission_id))').format(self.store.table('response_keys'),self.store.table('response_admissions'),self.store.table('session_admissions'),self.store.table('state_meta')),(ADMISSION_PREFIX,)).fetchall()
        for row in rows:self.store._update(conn,'response_keys',{'settled_at':time.time(),'response_id':durable_response_id(row['admission_id'])},'request_key',row['request_key'])

    def _evict(self,conn):
        # One namespace-level lock gives LRU/identity eviction the same atomic owner as puts.
        self.store._lock_conversation(conn,'__response_lru__')
        conn.execute(sql.SQL('DELETE FROM {} WHERE response_id IN(SELECT response_id FROM {} ORDER BY accessed_at DESC,response_id DESC OFFSET %s)').format(self.store.table('responses'),self.store.table('responses')),(self._max_size,))
        conn.execute(sql.SQL('DELETE FROM {} c WHERE NOT EXISTS(SELECT 1 FROM {} r WHERE r.response_id=c.response_id)').format(self.store.table('conversations'),self.store.table('responses')))
        self._settle(conn)
        bound=self._max_identities or 10*self._max_size
        rows=conn.execute(sql.SQL('SELECT request_key FROM {} WHERE settled_at IS NOT NULL ORDER BY settled_at DESC,request_key DESC OFFSET %s').format(self.store.table('response_keys')),(bound,)).fetchall()
        if rows:self._forget(conn,[row['request_key'] for row in rows])

    def bind_request_key(self,key,fingerprint,*,model=None):
        with self.store._transaction() as conn:
            self.store._insert(conn,'response_keys',dict(request_key=key,fingerprint=fingerprint,created_at=int(time.time()),model_json=json.dumps(model)),conflict=True)
            return conn.execute(sql.SQL('SELECT fingerprint FROM {} WHERE request_key=%s').format(self.store.table('response_keys')),(key,)).fetchone()['fingerprint']==fingerprint

    def request_model(self,key,fallback):
        with self.store._transaction() as conn:
            conn.execute(sql.SQL('UPDATE {} SET model_json=%s WHERE request_key=%s AND model_json IS NULL').format(self.store.table('response_keys')),(json.dumps(fallback),key))
            row=conn.execute(sql.SQL('SELECT model_json FROM {} WHERE request_key=%s').format(self.store.table('response_keys')),(key,)).fetchone()
            return json.loads(row['model_json']) if row else fallback

    def output_index(self,key,item_id):
        with self.store._transaction() as conn:
            self.store._lock_conversation(conn,'response-output:'+key)
            conn.execute(sql.SQL('INSERT INTO {}(request_key,item_id,output_index) SELECT %s,%s,COALESCE(MAX(output_index)+1,0) FROM {} WHERE request_key=%s ON CONFLICT(request_key,item_id) DO NOTHING').format(self.store.table('response_output_order'),self.store.table('response_output_order')),(key,item_id,key))
            return conn.execute(sql.SQL('SELECT output_index FROM {} WHERE request_key=%s AND item_id=%s').format(self.store.table('response_output_order')),(key,item_id)).fetchone()['output_index']

    def order_output(self,key,items):
        return sorted(items,key=lambda item:self.output_index(key,item['id']))

    def _key_value(self,key,column):
        row=self.store._one(sql.SQL('SELECT {} FROM {} WHERE request_key=%s').format(sql.Identifier(column),self.store.table('response_keys')),(key,))
        return row[column] if row else None

    def request_created_at(self,key):return self._key_value(key,'created_at')
    def request_key_matches(self,key,fingerprint):
        found=self._key_value(key,'fingerprint');return found is None or found==fingerprint

    def request_admission(self,key):
        row=self.store._one(sql.SQL('SELECT admission_id FROM {} WHERE request_key=%s').format(self.store.table('response_admissions')),(key,))
        return row['admission_id'] if row else None

    def bind_request_admission(self,key,admission_id):
        from hermes_state_runtime import RuntimeStoreError
        with self.store._transaction() as conn:
            self.store._insert(conn,'response_admissions',dict(request_key=key,admission_id=admission_id),conflict=True)
            row=conn.execute(sql.SQL('SELECT admission_id FROM {} WHERE request_key=%s').format(self.store.table('response_admissions')),(key,)).fetchone()
            if row['admission_id']!=admission_id:raise RuntimeStoreError('admission_conflict')

    def forget_unadmitted_request(self,key):
        with self.store._transaction() as conn:
            conn.execute(sql.SQL('DELETE FROM {} k WHERE request_key=%s AND NOT EXISTS(SELECT 1 FROM {} a WHERE a.request_key=k.request_key)').format(self.store.table('response_keys'),self.store.table('response_admissions')),(key,))

    def delete(self,response_id):
        with self.store._transaction() as conn:
            found=conn.execute(sql.SQL('DELETE FROM {} WHERE response_id=%s').format(self.store.table('responses')),(response_id,)).rowcount>0
            conn.execute(sql.SQL('DELETE FROM {} WHERE response_id=%s').format(self.store.table('conversations')),(response_id,))
            self._settle(conn)
            rows=conn.execute(sql.SQL('SELECT request_key FROM {} WHERE settled_at IS NOT NULL AND response_id=%s').format(self.store.table('response_keys')),(response_id,)).fetchall()
            if rows:self._forget(conn,[row['request_key'] for row in rows])
            return found or bool(rows)

    def get_conversation(self,name):
        row=self.store._one(sql.SQL('SELECT response_id FROM {} WHERE name=%s').format(self.store.table('conversations')),(name,))
        return row['response_id'] if row else None

    def set_conversation(self,name,response_id):
        with self.store._transaction() as conn:
            conn.execute(sql.SQL('INSERT INTO {}(name,response_id) VALUES(%s,%s) ON CONFLICT(name) DO UPDATE SET response_id=excluded.response_id').format(self.store.table('conversations')),(name,response_id))

    def __len__(self):
        return self.store._one(sql.SQL('SELECT COUNT(*) AS n FROM {}').format(self.store.table('responses')))['n']
