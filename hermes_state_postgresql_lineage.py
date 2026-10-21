"""PostgreSQL continuation selectors for native transaction callbacks.

Opaque legacy model_config values use the same tolerant parser as SessionDB;
malformed JSON is never cast through a PostgreSQL JSON operator.
"""
from psycopg import sql
from hermes_state_sessions import _parse_model_config


def _continuation(row):
    config=_parse_model_config(row['model_config'])
    return row['source']!='tool' and all(config.get(k) is None for k in ('_branched_from','_delegate_from','_reset_from'))


def compression_step(conn,parent_id):
    sessions=sql.Identifier(conn.hermes_scope.schema,'sessions')
    messages=sql.Identifier(conn.hermes_scope.schema,'messages')
    rows=conn.execute(sql.SQL("SELECT child.*, GREATEST(COALESCE(child.last_activity_at,0),child.started_at,COALESCE((SELECT MAX(m.timestamp) FROM {} m WHERE m.session_id=child.id),0)) AS _last_active FROM {} parent JOIN {} child ON child.parent_session_id=parent.id WHERE parent.id=%s AND parent.end_reason='compression'").format(messages,sessions,sessions),(parent_id,)).fetchall()
    candidates=[row for row in rows if _continuation(row)]
    if not candidates:return None
    def order(row):
        priority=0 if row['end_reason']=='compression' else (1 if row['ended_at'] is None else 2)
        return -priority,row['_last_active'],row['started_at'],row['id']
    return max(candidates,key=order)


def expand_compression_lineage_ids(conn,ids):
    sessions=sql.Identifier(conn.hermes_scope.schema,'sessions')
    rows=conn.execute(sql.SQL("SELECT child.id,child.source,child.model_config,parent.id AS _parent FROM {} child JOIN {} parent ON parent.id=child.parent_session_id WHERE parent.end_reason='compression' AND (parent.id=ANY(%s) OR child.id=ANY(%s))").format(sessions,sessions),(ids,ids)).fetchall()
    targets=set(ids)
    return [target for row in rows if _continuation(row) for target in
            ((row['id'],) if row['_parent'] in targets else ())+
            ((row['_parent'],) if row['id'] in targets else ())]
