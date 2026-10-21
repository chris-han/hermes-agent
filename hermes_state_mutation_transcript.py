"""Transaction-scoped rewind shared by storage and canonical mutations."""
from hermes_state_messages import _ACTIVE_IDS_SQL, _SET_COUNTERS_SQL, _placeholders
from hermes_state_sql import statement as _storage_statement, is_postgres


def rewind_in_transaction(self, conn, session_id, target_message_id, *,
                          preserve_compaction_handoff=False, expected_active_ids=None,
                          expected_target_content=None):
    self._check_transcript_write_guards(
        conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
    if expected_active_ids is not None:
        active_rows = conn.execute(_storage_statement(sqlite=_ACTIVE_IDS_SQL,postgresql='SELECT id FROM messages WHERE session_id=%s AND active=1 ORDER BY id',connection=conn), (session_id,)).fetchall()
        if [int(r[0]) for r in active_rows] != expected_active_ids:
            raise RuntimeError("active transcript changed before the rewind could be persisted")
    row = conn.execute(
        _storage_statement(sqlite="SELECT * FROM messages WHERE id = ? AND session_id = ?", postgresql='SELECT * FROM messages WHERE id = %s AND session_id = %s', connection=conn), (target_message_id, session_id)).fetchone()
    if row is None:
        raise ValueError(f"message {target_message_id} not found in session {session_id}")
    if is_postgres(conn):
        from hermes_state_postgresql_transcript import native_row
        target_row=native_row(row)
    else:target_row = dict(row)
    if target_row.get("role") != "user":
        raise ValueError(
            f"rewind target must be a 'user' message (got role={target_row.get('role')!r}, id={target_message_id})")
    replacement_message_id = replacement = None
    if preserve_compaction_handoff or expected_target_content is not None:
        replacement = self._split_rewind_target(target_row, expected_target_content, preserve_compaction_handoff)
    ids = [r[0] for r in conn.execute(_storage_statement(sqlite="SELECT id FROM messages WHERE session_id = ? AND id >= ? AND active = 1", postgresql='SELECT id FROM messages WHERE session_id = %s AND id >= %s AND active = 1', connection=conn),
                                     (session_id, target_message_id)).fetchall()]
    if ids:
        conn.execute(_storage_statement(sqlite=f"UPDATE messages SET active = 0 WHERE id IN ({_placeholders(ids)})",
            postgresql=f"UPDATE messages SET active=0 WHERE id IN ({','.join('%s' for _ in ids)})",connection=conn), ids)
    if replacement is not None:
        self._insert_message_rows(conn, session_id, [replacement])
        replacement_message_id = int(replacement['_row_id']) if is_postgres(conn) else int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        _storage_statement(sqlite="UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = ?", postgresql='UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = %s', connection=conn), (session_id,))
    message_count, tool_call_count = self._active_transcript_counts(conn, session_id)
    conn.execute(_storage_statement(sqlite=f"{_SET_COUNTERS_SQL} WHERE id = ?",postgresql='UPDATE sessions SET message_count=%s,tool_call_count=%s WHERE id=%s',connection=conn), (message_count, tool_call_count, session_id))
    head_id = conn.execute(
        _storage_statement(sqlite="SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1", postgresql='SELECT MAX(id) FROM messages WHERE session_id = %s AND active = 1', connection=conn), (session_id,)).fetchone()[0]
    return target_row, ids, head_id, replacement_message_id
