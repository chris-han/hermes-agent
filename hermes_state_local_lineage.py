"""Local physical-target handoff inside the existing publication transaction.

The creation ID remains the admission/execution owner. Transcript rotation must
not re-key idempotency digests, queued inputs, subscribers or uncertain claims.
"""
import json
from hermes_state_sql import statement as _storage_statement, is_postgres
from hermes_state_sql import compression_step as _postgres_compression_step
import time

from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import RuntimeStoreError, _epoch, _json
import itertools


def validate_local_lineage(conn, receipt):
    from hermes_state_compression import _CHAIN_STEP_SQL
    from hermes_state_keys import local_identity
    if ('legacy_session_id' not in receipt and
            receipt['session_id'] != local_identity(receipt['profile_id'], receipt['principal_id'], receipt['request_id'])):
        raise RuntimeStoreError('storage_unavailable')
    from hermes_state_local_migration import legacy_lineage_root
    root = legacy_lineage_root(conn, receipt)
    lineage = receipt.get('lineage', [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt['entry']['session_id']):
        raise RuntimeStoreError('storage_unavailable')
    for parent_id, child_id in itertools.pairwise(lineage):
        parent = conn.execute(_storage_statement(sqlite='SELECT end_reason FROM sessions WHERE id=?', postgresql='SELECT end_reason FROM sessions WHERE id=%s', connection=conn), (parent_id,)).fetchone()
        child = conn.execute(_storage_statement(sqlite='SELECT parent_session_id,model_config FROM sessions WHERE id=?', postgresql='SELECT parent_session_id,model_config FROM sessions WHERE id=%s', connection=conn), (child_id,)).fetchone()
        if parent is None or child is None or child['parent_session_id'] != parent_id:
            raise RuntimeStoreError('storage_unavailable')
        if parent['end_reason'] == 'session_reset':
            valid = json.loads(child['model_config'] or '{}').get('_reset_from') == parent_id
        else:
            tip = (_postgres_compression_step(conn, parent_id) if is_postgres(conn) else conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone())
            valid = parent['end_reason'] == 'compression' and tip is not None and tip['id'] == child_id
        if not valid:
            raise RuntimeStoreError('storage_unavailable')
    if (_postgres_compression_step(conn, lineage[-1]) if is_postgres(conn) else conn.execute(_CHAIN_STEP_SQL, (lineage[-1],)).fetchone()) is not None:
        raise RuntimeStoreError('storage_unavailable')
    return lineage[-1]


def local_physical_target(conn, session_id):
    """Resolve a local owner's current transcript on the caller's transaction."""
    saved = conn.execute(_storage_statement(sqlite='SELECT value FROM state_meta WHERE key=?', postgresql='SELECT value FROM state_meta WHERE key=%s', connection=conn),
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return session_id
    receipt = json.loads(saved[0])
    if receipt['session_id'] != session_id:
        raise RuntimeStoreError('storage_unavailable')
    return validate_local_lineage(conn, receipt)


def advance_local_target(conn, parent_session_id, child_session_id, *, entry=None):
    parent = conn.execute(_storage_statement(sqlite='SELECT * FROM sessions WHERE id=?', postgresql='SELECT * FROM sessions WHERE id=%s', connection=conn), (parent_session_id,)).fetchone()
    logical_id = parent['chat_id'] if parent else None
    if not logical_id:
        return
    key = POLICY_PREFIX + logical_id
    raw = conn.execute(_storage_statement(sqlite='SELECT value FROM state_meta WHERE key=?', postgresql='SELECT value FROM state_meta WHERE key=%s', connection=conn), (key,)).fetchone()
    if raw is None:
        from hermes_state_local_migration import LEGACY_PREFIX
        legacy = conn.execute(_storage_statement(sqlite='SELECT 1 FROM state_meta WHERE key=?', postgresql='SELECT 1 FROM state_meta WHERE key=%s', connection=conn), (LEGACY_PREFIX + logical_id,)).fetchone()
        if logical_id.startswith('local-') or legacy:
            raise RuntimeStoreError('storage_unavailable')
        return
    try:
        receipt = json.loads(raw[0])
        if (receipt['session_id'] != logical_id or receipt['entry']['session_id'] != parent_session_id
                or receipt['principal_id'] != parent['user_id'] or receipt['route'] != parent['session_key']):
            raise RuntimeStoreError('storage_unavailable')
        route_row = conn.execute(_storage_statement(sqlite="SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?", postgresql="SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=%s", connection=conn),
                                 (receipt['route'],)).fetchone()
        routed = json.loads(route_row[0]) if route_row else None
        if routed is None or routed['session_id'] != parent_session_id:
            raise RuntimeStoreError('admission_conflict')
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt['entry'] = candidate
        receipt['lineage'] = [*receipt.get('lineage', [logical_id]), child_session_id]
        validate_local_lineage(conn, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeStoreError):
            raise
        raise RuntimeStoreError('storage_unavailable') from exc
    conn.execute(_storage_statement(sqlite="UPDATE gateway_routing SET entry_json=? WHERE scope='' AND session_key=?", postgresql="UPDATE gateway_routing SET entry_json=%s WHERE scope='' AND session_key=%s", connection=conn),
                 (_json(candidate), receipt['route']))
    conn.execute(_storage_statement(sqlite='UPDATE state_meta SET value=? WHERE key=?', postgresql='UPDATE state_meta SET value=%s WHERE key=%s', connection=conn), (_json(receipt), key))
    conn.execute(_storage_statement(sqlite='UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', postgresql='UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=%s', connection=conn), (logical_id,))


def reset_local_target(db, *, epoch, parent_session_id, entry):
    def write(conn):
        _epoch(conn, epoch)
        parent = conn.execute(_storage_statement(sqlite='SELECT * FROM sessions WHERE id=?', postgresql='SELECT * FROM sessions WHERE id=%s', connection=conn), (parent_session_id,)).fetchone()
        if parent is None or parent['end_reason'] == 'compression':
            raise RuntimeStoreError('admission_conflict')
        saved = conn.execute(_storage_statement(sqlite='SELECT value FROM state_meta WHERE key=?', postgresql='SELECT value FROM state_meta WHERE key=%s', connection=conn),
                             (POLICY_PREFIX + parent['chat_id'],)).fetchone()
        if saved is None:
            raise RuntimeStoreError('storage_unavailable')
        receipt = json.loads(saved[0])
        logical_id = receipt['session_id']
        if validate_local_lineage(conn, receipt) != parent_session_id:
            raise RuntimeStoreError('admission_conflict')
        from hermes_state_mutation_guards import require_not_executing
        require_not_executing(conn, list({logical_id, parent_session_id}))
        policy = receipt['policy']
        db._publish_child_session_row(conn, parent, parent_session_id=parent_session_id,
            child_session_id=entry['session_id'], source=policy['source'], model=policy['model'],
            model_config={'_reset_from': parent_session_id}, system_prompt=None,
            cwd=policy['cwd'], profile_name=parent['profile_name'])
        conn.execute(_storage_statement(sqlite="UPDATE sessions SET ended_at=?,end_reason='session_reset' WHERE id=?", postgresql="UPDATE sessions SET ended_at=%s,end_reason='session_reset' WHERE id=%s", connection=conn),
                     (time.time(), parent_session_id))
        db._bump_conversation_generation(conn, parent_session_id, 'session_reset')
        advance_local_target(conn, parent_session_id, entry['session_id'], entry=entry)
        conn.execute(_storage_statement(sqlite='UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', postgresql='UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=%s', connection=conn), (logical_id,))
    db._execute_write(write)
