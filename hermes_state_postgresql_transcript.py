"""PostgreSQL rows for the shared native transcript CAS/repair algorithm."""
from psycopg import sql
from agent.transcript_repair import _REPAIR_COLUMNS,_OWNED_COLUMNS,transcript_row_snapshot


def native_row(row):
    if row is None: return None
    row=dict(row)
    content=row.pop('content_bytes',None)
    if content is not None: row['content']=bytes(content).decode('utf-8')
    for key,value in row.items():
        if isinstance(value,memoryview): row[key]=bytes(value)
    return row


def pg_values(values):
    values=dict(values)
    content=values.get('content')
    values['content_bytes']=content.encode('utf-8') if isinstance(content,str) and '\x00' in content else None
    if values['content_bytes'] is not None: values['content']=None
    return values


class PostgreSQLTranscriptRows:
    def __init__(self,store,conn):
        self.store,self.conn=store,conn

    def fetch(self,session_id,row_id):
        return native_row(self.conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND id=%s').format(self.store.table('messages')),(session_id,row_id)).fetchone())

    def logical(self,session_id,role,uid):
        if not uid: return None
        return native_row(self.conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND role=%s AND message_uid=%s AND active=1 ORDER BY id DESC LIMIT 1').format(self.store.table('messages')),(session_id,role,uid)).fetchone())

    def active(self,session_id,row_id,role):
        row=self.fetch(session_id,row_id)
        if not row or row['role']!=role: return None
        if row['active']==1 or row['display_identity'] is None: return row
        # Native compaction clone identity includes every durable payload field.
        values=pg_values({column:row[column] for column in _REPAIR_COLUMNS})
        clauses=sql.SQL(' AND ').join(sql.SQL('{} IS NOT DISTINCT FROM %s').format(sql.Identifier(column)) for column in values)
        query=sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 AND role=%s AND display_identity IS NOT DISTINCT FROM %s AND timestamp=%s AND id<>%s AND {} ORDER BY id DESC LIMIT 2').format(self.store.table('messages'),clauses)
        clones=self.conn.execute(query,[session_id,role,row['display_identity'],row['timestamp'],row_id,*values.values()]).fetchall()
        return native_row(clones[0]) if len(clones)==1 else row

    def rewrite(self,session_id,target,serialized):
        values=pg_values({column:serialized[column] for column in _OWNED_COLUMNS})
        self.store._update(self.conn,'messages',values,'id',target['id'])
        # Display projection derives the same native identity in Python; no trigger shadow state.
        self.store._update(self.conn,'messages',{'display_identity':serialized['display_identity'],'display_order':target['id']},'id',target['id'])

    def fill_blank(self,session_id,row_id,target,content):
        # Conversation advisory lock and FOR UPDATE fence the CAS throughout the transaction.
        return bool(self.store._update(self.conn,'messages',pg_values({'content':content}),'id',row_id))


def append_on_conn(store,conn,session_id,messages):
    from hermes_state import SessionDB
    from hermes_state_messages import _parse_tool_calls,_coerce_timestamp
    from agent.transcript_repair import resolve_and_repair_transcript_batch
    from agent.message_metadata import stamp_message_uid,DB_ROW_SNAPSHOT
    # Reuse native codecs without opening SQLite; the invoked helpers are pure serialization.
    codec=object.__new__(SessionDB)
    rows=PostgreSQLTranscriptRows(store,conn)
    fresh=resolve_and_repair_transcript_batch(conn,session_id,messages,
        encode_content_fn=codec._encode_content,decode_content_fn=codec._decode_content,
        serialize_message_fn=lambda message,timestamp:codec._serialized_message_row(session_id,message,timestamp),
        decode_row_fn=lambda row:codec._row_to_message_dict(row,warn_context='PG transcript repair',summary_flag=True),
        row_store=rows)
    ids=[]; uid_index={}
    for message in fresh:
        stamp_message_uid(message)
        codec._stamp_tool_call_uids(message,_parse_tool_calls(message.get('tool_calls')),uid_index)
        timestamp=_coerce_timestamp(message.get('timestamp'),__import__('time').time())
        values=pg_values(codec._serialized_message_row(session_id,message,timestamp))
        row_id=store._insert(conn,'messages',values,returning='id').fetchone()['id']
        message['_row_id']=row_id; message['timestamp']=timestamp
        message[DB_ROW_SNAPSHOT]=transcript_row_snapshot(rows.fetch(session_id,row_id)); ids.append(row_id)
    store._recount_on_conn(conn,session_id)
    return ids
