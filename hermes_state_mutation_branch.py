"""An independent local route, transcript and frozen policy in one receipt commit."""
import json
from hermes_state_sql import statement as _storage_statement, is_postgres
import uuid

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_mutation_guards import require_idle
from hermes_state_runtime import RuntimeStoreError, _json
from hermes_state_titles import lineage_title_on_conn


def _branch_boundary(conn, target, through):
    """Highest row id the branch copies. ``through`` must name an active row of the physical
    target (after compression that is the continuation, whose ids the viewer was shown); rows
    appended later, even in this same commit window, have larger ids and stay out. The tool
    results that directly follow an assistant boundary answer its calls, so they belong to it."""
    if through is None:
        return conn.execute(_storage_statement(sqlite='SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=? AND active=1', postgresql='SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=%s AND active=1', connection=conn),
                            (target,)).fetchone()[0]
    row = conn.execute(_storage_statement(sqlite='SELECT role FROM messages WHERE id=? AND session_id=? AND active=1', postgresql='SELECT role FROM messages WHERE id=%s AND session_id=%s AND active=1', connection=conn),
                       (through, target)).fetchone()
    if row is None:
        raise RuntimeStoreError('invalid_params')
    if row['role'] == 'assistant':
        following = conn.execute(_storage_statement(sqlite='SELECT id, role FROM messages WHERE session_id=? AND active=1 AND id>? '
                                 'ORDER BY id', postgresql='SELECT id, role FROM messages WHERE session_id=%s AND active=1 AND id>%s ORDER BY id', connection=conn), (target, through))
        for later in following:
            if later['role'] != 'tool':
                break
            through = later['id']
    return through


def branch_in_transaction(db, conn, session_id, payload):
    row = conn.execute(_storage_statement(sqlite='SELECT value FROM state_meta WHERE key=?', postgresql='SELECT value FROM state_meta WHERE key=%s', connection=conn),
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_idle(db, conn, list({session_id, target}))
    # Late: the routing entry/key are gateway types (the gateway imports this store at module level).
    from gateway.session import SessionEntry, SessionSource
    from gateway.config import Platform
    from hermes_state_keys import local_identity, profile_from_session_key
    request_id = 'branch:' + uuid.uuid4().hex
    child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
    # The child keeps the parent's owning profile: a served secondary's origin names it
    # (``local_source``), and restore compares the stored origin against that exactly.
    source = SessionSource(Platform.LOCAL, child, user_id=saved['principal_id'], chat_type='dm',
                           profile=saved['entry']['origin'].get('profile'))
    from gateway.session import build_session_key
    route = build_session_key(source, profile=profile_from_session_key(saved['route']))
    from gateway.session_lifecycle import _now
    now = _now()
    entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
    parent = conn.execute(_storage_statement(sqlite='SELECT * FROM sessions WHERE id=?', postgresql='SELECT * FROM sessions WHERE id=%s', connection=conn), (target,)).fetchone()
    policy = saved['policy']
    db._publish_child_session_row(conn, parent, parent_session_id=target,
        child_session_id=child, source=policy['source'], model=policy['model'],
        model_config={'_branched_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute(_storage_statement(sqlite='UPDATE sessions SET session_key=?,chat_id=?,origin_json=?,system_prompt_hash=? WHERE id=?', postgresql='UPDATE sessions SET session_key=%s,chat_id=%s,origin_json=%s,system_prompt_hash=%s WHERE id=%s', connection=conn),
                 (route, child, _json(source.to_dict()), parent['system_prompt_hash'], child))
    ids, tools = db._tail_rows_after_watermark(conn,
        _storage_statement(sqlite='SELECT id,tool_calls FROM messages WHERE session_id=? AND active=1 AND id<=? ORDER BY id',
            postgresql='SELECT id,tool_calls FROM messages WHERE session_id=%s AND active=1 AND id<=%s ORDER BY id',connection=conn),
        [target, _branch_boundary(conn, target, payload.get('through_message_id'))])
    db._clone_message_rows(conn, ids, session_id=child)
    conn.execute(_storage_statement(sqlite='UPDATE sessions SET message_count=?,tool_call_count=? WHERE id=?', postgresql='UPDATE sessions SET message_count=%s,tool_call_count=%s WHERE id=%s', connection=conn), (len(ids), tools, child))
    if payload.get('title'):
        db._set_session_title_in_transaction(conn, child, payload['title'], source=db.TITLE_SOURCE_USER)
    else:
        # Untitled branch: the next name in the parent's lineage ("my session" → "my session #2"),
        # same rule as the legacy TUI /branch, so the child is its own titled sidebar row.
        db._set_session_title_in_transaction(conn, child, lineage_title_on_conn(conn, parent['title'] or 'branch'),
                                             source=db.TITLE_SOURCE_DERIVED)
    receipt = dict(profile_id=saved['profile_id'], principal_id=saved['principal_id'],
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    conn.execute(_storage_statement(sqlite="INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)", postgresql="INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',%s,%s,%s)", connection=conn),
                 (route, _json(entry.to_dict()), now.timestamp()))
    conn.execute(_storage_statement(sqlite='INSERT INTO state_meta(key,value) VALUES(?,?)', postgresql='INSERT INTO state_meta(key,value) VALUES(%s,%s)', connection=conn), (POLICY_PREFIX + child, _json(receipt)))
    return {session_id}, {'branched_session_id': child, 'copied_messages': len(ids)}
