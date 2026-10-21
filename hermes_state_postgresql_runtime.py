"""PostgreSQL owner statements preserving the native route/config/cooldown operations."""
from __future__ import annotations
from contextlib import contextmanager
import json
import logging
import time
from typing import Any, Callable, Optional
import psycopg
from psycopg import sql
from hermes_state_sessions import _parse_model_config, _MODEL_CONFIG_ROW_MISSING
from hermes_state_compression import _cooldown_row
from hermes_state import SessionDB as _NativeSessionDB
from hermes_state_sql import _row_factory, native_write_transaction
logger = logging.getLogger(__name__)
_COOLDOWN_ROW_SQL = 'SELECT compression_failure_cooldown_until,compression_failure_error FROM sessions WHERE id=%s'

_PEER_BY_KEY_PG="\n                SELECT s.*,\n                       COALESCE(sp.prompt, s.system_prompt)\n                           AS _system_prompt_resolved,\n                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (\n                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1\n                       )) AS _has_messages\n                FROM sessions s\n                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash\n                WHERE s.session_key = %s\n                  AND s.source = %s\n                  AND (s.ended_at IS NULL OR s.end_reason IN ('agent_close', 'ws_orphan_reap', 'superseded_by_resume', 'startup_orphan_reap')\n                       OR (s.handoff_state = 'completed'))\n                  AND NOT EXISTS (\n                      SELECT 1 FROM sessions b\n                      WHERE b.session_key = s.session_key\n                        AND b.source = s.source\n                        AND b.ended_at IS NOT NULL\n                        AND b.end_reason IN ('session_reset', 'session_switch', 'idle', 'daily', 'suspended', 'resume_pending_expired')\n                        AND b.ended_at\n                            > COALESCE(s.last_activity_at, s.started_at)\n                  )\n                ORDER BY _has_messages DESC,\n                         COALESCE(s.last_activity_at, s.started_at) DESC\n                LIMIT 1\n                "
_PEER_BY_TUPLE_PG="\n                SELECT s.*,\n                       COALESCE(sp.prompt, s.system_prompt)\n                           AS _system_prompt_resolved,\n                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (\n                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1\n                       )) AS _has_messages\n                FROM sessions s\n                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash\n                WHERE s.source = %s\n                  AND COALESCE(s.user_id, '') = COALESCE(%s, '')\n                  AND COALESCE(s.chat_id, '') = COALESCE(%s, '')\n                  AND COALESCE(s.chat_type, '') = COALESCE(%s, '')\n                  AND COALESCE(s.thread_id, '') = COALESCE(%s, '')\n                  AND (%s::text IS NULL OR COALESCE(s.profile_name, %s) = %s)\n                  AND (s.ended_at IS NULL OR s.end_reason IN ('agent_close', 'ws_orphan_reap', 'superseded_by_resume', 'startup_orphan_reap')\n                       OR (s.handoff_state = 'completed'))\n                  AND (COALESCE(s.message_count, 0) > 0 OR EXISTS (\n                      SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1\n                  ))\n                  AND NOT EXISTS (\n                      SELECT 1 FROM sessions b\n                      WHERE b.source = s.source\n                        AND COALESCE(b.user_id, '') = COALESCE(s.user_id, '')\n                        AND COALESCE(b.chat_id, '') = COALESCE(s.chat_id, '')\n                        AND COALESCE(b.chat_type, '') = COALESCE(s.chat_type, '')\n                        AND COALESCE(b.thread_id, '') = COALESCE(s.thread_id, '')\n                        AND (%s::text IS NULL OR COALESCE(b.profile_name, %s) = %s)\n                        AND b.ended_at IS NOT NULL\n                        AND b.end_reason IN ('session_reset', 'session_switch', 'idle', 'daily', 'suspended', 'resume_pending_expired')\n                        AND b.ended_at\n                            > COALESCE(s.last_activity_at, s.started_at)\n                  )\n                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC\n                LIMIT 1\n                "

class PostgreSQLRuntimeMixin:
    _TRANSCRIPT_WRITE_PATIENCE_S = _NativeSessionDB._TRANSCRIPT_WRITE_PATIENCE_S
    _WRITE_PATIENCE_S = _NativeSessionDB._WRITE_PATIENCE_S
    _TITLE_SOURCE_RANK = _NativeSessionDB._TITLE_SOURCE_RANK
    _WRITE_RETRY_MIN_S = _NativeSessionDB._WRITE_RETRY_MIN_S
    _WRITE_RETRY_MAX_S = _NativeSessionDB._WRITE_RETRY_MAX_S
    _WRITE_RETRY_SLOW_AFTER_S = _NativeSessionDB._WRITE_RETRY_SLOW_AFTER_S
    _WRITE_RETRY_SLOW_MIN_S = _NativeSessionDB._WRITE_RETRY_SLOW_MIN_S
    _WRITE_RETRY_SLOW_MAX_S = _NativeSessionDB._WRITE_RETRY_SLOW_MAX_S
    _sleep_before_write_retry = _NativeSessionDB._sleep_before_write_retry

    @contextmanager
    def _read_ctx(self):
        with self._canonical_connection() as conn:
            conn.execute('SET default_transaction_read_only=on')
            yield conn

    @contextmanager
    def _canonical_connection(self):
        if self._closed:
            raise RuntimeError('Session store closed')
        with psycopg.connect(self._dsn, autocommit=True, row_factory=_row_factory) as conn:
            conn.execute(sql.SQL('SET search_path TO {},pg_catalog').format(sql.Identifier(self.scope.schema)))
            conn.hermes_scope = self.scope
            conn.hermes_read_only = self.read_only
            conn.hermes_home = self.home
            yield conn

    def _execute_write(self, callback, **kwargs):
        with self._canonical_connection() as conn:
            with native_write_transaction(conn,patience_s=kwargs.get('patience_s',self._WRITE_PATIENCE_S)):
                return callback(conn)

    def _read_one(self, query, params=()):
        with self._read_ctx() as conn:
            return conn.execute(query, params).fetchone()

    def _read_all(self, query, params=()):
        with self._read_ctx() as conn:
            return conn.execute(query, params).fetchall()

    def _write_rowcount(self, query, params=()):
        return self._execute_write(lambda conn: conn.execute(query, params).rowcount)

    def _write_sql(self, query, params=(), *, many=False):
        if many:
            return self._execute_write(lambda conn: conn.cursor().executemany(query, params))
        return self._write_rowcount(query, params)

    def _write_sql_logged(self, operation, session_id, query, params=()):
        return self._write_sql(query, params)

    def _set_lineage_column(self, column, session_id, value):
        with self._transaction() as conn:
            self._lock_conversation(conn, session_id)
            ids = self.get_compression_lineage(session_id) or [session_id]
            return bool(conn.execute(sql.SQL('UPDATE {} SET {}=%s WHERE id=ANY(%s)').format(self.table('sessions'), sql.Identifier(column)), (value, ids)).rowcount)

    def update_session_cwd(self, session_id: str, cwd: str, git_branch: Optional[str]=None, git_repo_root: Optional[str]=None, replace_git_meta: bool=False) -> Optional[int]:
        """Persist the authoritative cwd and claim a Git metadata generation. git fields are written
        only when non-empty (a probe failure never clobbers a value) except under ``replace_git_meta``
        (a workspace MOVE overwrites the old repo identity). Async probes publish with the returned
        generation so an older worker cannot overwrite a newer claim (A -> B -> A)."""
        if not session_id or not cwd:
            return None
        branch = (git_branch or '').strip()
        repo_root = (git_repo_root or '').strip()

        def _do(conn):
            current = conn.execute('SELECT cwd FROM sessions WHERE id = %s', (session_id,)).fetchone()
            if current is None:
                return None
            sets = ['cwd = %s', 'git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1']
            params: list[Any] = [cwd]
            if current[0] != cwd or replace_git_meta:
                sets.extend(('git_branch = %s', 'git_repo_root = %s'))
                params.extend((branch or None, repo_root or None))
            else:
                for (col, val) in (('git_branch', branch), ('git_repo_root', repo_root)):
                    if val:
                        sets.append(f'{col} = %s')
                        params.append(val)
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = %s", [*params, session_id])
            row = conn.execute('SELECT git_metadata_generation FROM sessions WHERE id = %s', (session_id,)).fetchone()
            return None if row is None else int(row[0])
        return self._execute_write(_do)

    def publish_session_git_metadata(self, session_id: str, cwd: str, generation: int, git_branch: Optional[str]=None, git_repo_root: Optional[str]=None) -> bool:
        """Publish async Git enrichment only while its cwd claim is current."""
        valid_generation = isinstance(generation, int) and (not isinstance(generation, bool)) and (generation >= 1)
        if not session_id or not cwd or (not valid_generation):
            return False
        fields = [(col, val) for (col, val) in (('git_branch', (git_branch or '').strip()), ('git_repo_root', (git_repo_root or '').strip())) if val]
        if not fields:
            return False
        return self._write_rowcount(f"UPDATE sessions SET {', '.join((f'{col} = %s' for (col, _) in fields))} WHERE id = %s AND cwd = %s AND git_metadata_generation = %s", [val for (_, val) in fields] + [session_id, cwd, generation]) == 1

    def backfill_repo_roots(self, cwd_to_root: dict[str, str]) -> None:
        """Backfill git repo roots for cwds without one; never clobbers a recorded root."""
        pairs = [(root, cwd) for (cwd, root) in cwd_to_root.items() if root and cwd]
        if pairs:
            self._write_sql("UPDATE sessions SET git_repo_root = %s WHERE cwd = %s AND COALESCE(git_repo_root, '') = ''", pairs, many=True)

    def update_session_model(self, session_id: str, model: str, provider: Optional[str]=None, *, base_url: Optional[str]=None, api_mode: Optional[str]=None) -> None:
        """Set the model after a mid-session /model switch (unconditionally) and drop any Browser
        runtime lock (lineage markers survive).

        Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness.

        When *provider* is given the whole route is written, in both shapes resume reads (top-level
        keys for the TUI/Desktop, ``gateway_runtime`` for the CLI), so a later resume recombines the
        model with the provider that serves it (#79536). ``base_url``/``api_mode`` are always
        replaced then (``None`` deletes): the previous provider's endpoint must not survive a switch,
        or resume sends the new provider's model to the old host. Callers without provider knowledge
        leave the stored route untouched.
        """
        self.flush_token_counts()
        patch: dict[str, Any] = {'browser_model_lock': None}
        if model:
            patch['model'] = model
        if provider:
            route = {'provider': provider, 'base_url': base_url or None, 'api_mode': api_mode or None}
            patch.update(route, gateway_runtime=route)
        self._write_model_config_patch(session_id, patch, 'UPDATE sessions SET model = %s, model_config = %s WHERE id = %s', lambda merged: (model, merged, session_id))

    def _write_model_config_patch(self, session_id: str, patch: dict[str, Any], sql: str='UPDATE sessions SET model_config = %s WHERE id = %s', params: Optional[Callable[[Optional[str]], tuple]]=None) -> None:
        """Merge ``patch`` into model_config then run ``sql`` with ``params(merged)`` in one write
        transaction; no-op when the row doesn't exist."""

        def _do(conn):
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(sql, params(merged) if params else (merged, session_id))
        self._execute_write(_do)

    def _merge_model_config_json(self, conn, session_id: str, patch: dict[str, Any], *, on_missing: str='skip'):
        """SELECT + tolerant-parse + merge ``patch`` into model_config (the one place that keeps
        ``_branched_from``/``_delegate_from`` alive); ``None`` deletes a key. Returns serialized JSON
        (``None`` when empty) or ``_MODEL_CONFIG_ROW_MISSING`` (``on_missing="raise"`` → ValueError)."""
        row = conn.execute('SELECT model_config FROM sessions WHERE id = %s', (session_id,)).fetchone()
        if row is None:
            if on_missing == 'raise':
                raise ValueError(f'Session not found: {session_id}')
            return _MODEL_CONFIG_ROW_MISSING
        config = _parse_model_config(row[0])
        for (key, value) in patch.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value
        return json.dumps(config) if config else None

    def patch_session_model_config(self, session_id: str, patch: dict[str, Any]) -> None:
        """Merge ``patch`` into model_config atomically (``None`` removes a key);
        no-op when the row or patch is empty."""
        if not session_id or not patch:
            return
        self._write_model_config_patch(session_id, patch)

    def get_session_model_config_value(self, session_id: str, key: str, default: Any=None) -> Any:
        """Read one key out of a session's model_config JSON (tolerant parse)."""
        session = self.get_session(session_id) or {}
        return _parse_model_config(session.get('model_config')).get(key, default)

    def set_session_yolo(self, session_id: str, enabled: bool) -> None:
        """Persist the per-session YOLO flag so ``/yolo`` survives ``--resume``; no-op without a row."""
        if not session_id:
            return
        self._write_model_config_patch(session_id, {'yolo_mode': bool(enabled)})

    @staticmethod
    def session_yolo_enabled(session_meta: Optional[dict[str, Any]]) -> bool:
        """Persisted YOLO flag; False on any parse failure (resume must never enable the bypass)."""
        return bool(_parse_model_config((session_meta or {}).get('model_config')).get('yolo_mode'))

    def set_session_read(self, session_id: str, read: bool=True) -> bool:
        """Mark read/unread across the compression lineage. ``last_read_at`` is a watermark: unread when
        activity postdates it (no write on the message path). NULL = never tracked = read; 0 = unread."""
        return self._set_lineage_column('last_read_at', session_id, time.time() if read else 0.0)

    @staticmethod
    def session_unread(session_row: dict[str, Any]) -> bool:
        """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
        last_read = session_row.get('last_read_at')
        if last_read is None:
            return False
        last_active = session_row.get('last_active') or session_row.get('started_at')
        return float(last_active or 0) > float(last_read)

    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: Optional[str]=None) -> None:
        """Persist the active compression-failure cooldown. Merge-max with any longer live deadline so a
        later shorter write can't reopen the thrash window; error always takes the latest diagnostic."""
        if not session_id:
            return
        self._write_sql_logged('record_compression_failure_cooldown', session_id, 'UPDATE sessions SET compression_failure_cooldown_until = CASE WHEN compression_failure_cooldown_until IS NOT NULL  AND compression_failure_cooldown_until > %s THEN compression_failure_cooldown_until ELSE %s END, compression_failure_error = %s WHERE id = %s', (cooldown_until, cooldown_until, error, session_id))

    def get_compression_failure_cooldown(self, session_id: str) -> Optional[dict[str, Any]]:
        """Return the active (unexpired) compression-failure cooldown, or None."""
        now = time.time()
        row = self._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
        if row is None or row[0] is None or float(row[0]) <= now:
            return None
        return {'cooldown_until': float(row[0]), 'remaining_seconds': float(row[0]) - now, 'error': row[1]}

    def get_compression_failure_cooldown_row(self, session_id: str) -> dict[str, Any]:
        """Exact stored cooldown columns, no expiry filtering, so compression cancellation
        can roll back an expired, partially-null, or absent row exactly."""
        row = self._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
        return _cooldown_row(False, None, None) if row is None else _cooldown_row(True, row[0], row[1])

    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: dict[str, Any]) -> None:
        """Restore and verify an exact cooldown-row snapshot. Unlike record/clear this
        rollback API propagates write and verification failures: cancellation must not be
        reported mutation-free when compensation failed. The tolerated exception is a
        session row that vanished mid-attempt — before or after the compensating write —
        since its cooldown died with it and there is nothing left to restore (#106271)."""
        if not snapshot.get('session_exists', False):
            if self.get_compression_failure_cooldown_row(session_id).get('session_exists', False):
                raise RuntimeError('cannot restore absent compression cooldown row: session now exists')
            return
        deadline = snapshot.get('cooldown_until')
        error = snapshot.get('error')

        def _do(conn):
            cursor = conn.execute('UPDATE sessions SET compression_failure_cooldown_until = %s, compression_failure_error = %s WHERE id = %s', (deadline, error, session_id))
            return cursor.rowcount == 1
        if not self._execute_write(_do):
            logger.warning('compression cooldown rollback session missing: %s', session_id)
            return
        actual = self.get_compression_failure_cooldown_row(session_id)
        expected = _cooldown_row(True, deadline, error)
        if actual != expected:
            if not actual.get('session_exists', False):
                logger.warning('compression cooldown rollback session missing after restore: %s', session_id)
                return
            raise RuntimeError(f'compression cooldown rollback verification failed: expected={expected!r}, actual={actual!r}')

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        """Clear any persisted compression-failure cooldown for a session."""
        if not session_id:
            return
        self._write_sql_logged('clear_compression_failure_cooldown', session_id, 'UPDATE sessions SET compression_failure_cooldown_until = NULL, compression_failure_error = NULL WHERE id = %s', (session_id,))

    def _read_session_number(self, column: str, session_id: str, cast: type, zero: Any) -> Any:
        """Read one numeric ``sessions`` column clamped at ``zero``; a missing session,
        NULL, or unparsable value also reads as ``zero``."""
        row = self._read_one(f'SELECT {column} FROM sessions WHERE id = %s', (session_id,)) if session_id else None
        try:
            return zero if row is None else max(zero, cast(row[0] or zero))
        except (TypeError, ValueError):
            return zero

    def _write_session_column(self, column: str, session_id: str, value: Any) -> None:
        self._write_sql(f'UPDATE sessions SET {column} = %s WHERE id = %s', (value, session_id))

    def get_compression_fallback_streak(self, session_id: str) -> int:
        """Return the persisted deterministic-fallback streak."""
        return self._read_session_number('compression_fallback_streak', session_id, int, 0)

    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        """Persist the deterministic-fallback streak for one session."""
        if session_id:
            self._write_session_column('compression_fallback_streak', session_id, max(0, int(streak)))

    def get_compression_ineffective_count(self, session_id: str) -> int:
        """Persisted ineffective-compaction strike count — the durable half of the built-in
        compressor's anti-thrash guard, so a fresh compressor bound to a resumed session
        inherits an armed/tripped guard across restarts."""
        return self._read_session_number('compression_ineffective_count', session_id, int, 0)

    def set_compression_ineffective_count(self, session_id: str, count: int) -> None:
        """Persist the ineffective-compaction strike count for one session."""
        if session_id:
            self._write_session_column('compression_ineffective_count', session_id, max(0, int(count)))

    def get_compression_recovery_deadline(self, session_id: str) -> float:
        """Persisted anti-thrash recovery deadline (epoch; ``0.0`` = not armed). Durable
        because the gateway rebuilds the compressor every turn / cache eviction.

        The deadline is the durable half of the 14694 recovery clock: the gateway rebuilds the compressor on
        every turn / cache eviction, so a process-local deadline restarted the wait on each rebuild and a
        tripped session never earned its probe (#100185).
        """
        return self._read_session_number('compression_recovery_deadline', session_id, float, 0.0)

    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None:
        """Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it."""
        if not session_id:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        self._write_session_column('compression_recovery_deadline', session_id, normalized or None)

    def get_compression_overload_streak(self, session_id: str) -> int:
        """Return the persisted sustained-overload abort streak (#123167)."""
        return self._read_session_number('compression_overload_streak', session_id, int, 0)

    def set_compression_overload_streak(self, session_id: str, streak: int) -> None:
        """Persist the sustained-overload abort streak for one session."""
        if session_id:
            self._write_session_column('compression_overload_streak', session_id, max(0, int(streak)))

    def increment_compression_overload_streak(self, session_id: str) -> Optional[int]:
        """Atomically bump the overload streak and return the new value (None when no row).
        One UPDATE ... RETURNING, so concurrent agents on one session cannot lose a strike."""
        if not session_id:
            return None

        def _do(conn):
            row = conn.execute('UPDATE sessions SET compression_overload_streak = compression_overload_streak + 1 WHERE id = %s RETURNING compression_overload_streak', (session_id,)).fetchone()
            return None if row is None else int(row[0])
        return self._execute_write(_do)

    def admit_native_api_route(self, session_key: str, entry_json: str, *, scope: str) -> str:
        """Bind a real native transcript without replacing another origin or route.

        The authenticated host supplies provenance; this method grants no authority.
        Check and insertion share the existing SQLite writer transaction.
        """
        entry = json.loads(entry_json)

        def admit(conn):
            session = conn.execute('SELECT * FROM sessions WHERE id=%s', (entry['session_id'],)).fetchone()
            if session is None or session['ended_at'] is not None:
                raise ValueError('API_ORIGIN_SESSION_UNAVAILABLE')
            if session['session_key'] != session_key or session['source'] != 'api_server':
                raise ValueError('API_ORIGIN_SESSION_MISMATCH')
            old = conn.execute('SELECT entry_json FROM gateway_routing WHERE scope=%s AND session_key=%s', (scope, session_key)).fetchone()
            if old:
                previous = json.loads(old[0])
                if previous['session_id'] != entry['session_id'] or previous.get('origin') != entry.get('origin'):
                    raise ValueError('API_ORIGIN_REBINDING_FORBIDDEN')
                return old[0]
            conn.execute('INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES(%s,%s,%s,%s)', (scope, session_key, entry_json, time.time()))
            return entry_json
        return self._execute_write(admit)

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        """Overwrite a title's provenance without touching the text (a title copied across a
        compression rotation keeps the original's authority)."""
        if source not in self._TITLE_SOURCE_RANK:
            raise ValueError(f'invalid title source: {source!r}')
        return self._write_rowcount('UPDATE sessions SET title_source = %s WHERE id = %s AND title IS NOT NULL', (source, session_id)) > 0

    # The canonical wait loop calls only the owned lease verb and wall-clock callbacks.
    def acquire_session_turn_lease(
        self, session_id: str, holder: str, *, ttl_seconds: float = 300.0,
        wait_seconds: float = 1800.0, poll_interval_seconds: float = 1.0, on_wait=None,
        wait_notice_interval_seconds: float = 15.0, should_abort=None, acquire_patience_s: float = 0.5,
        on_contended=None,
    ) -> bool:
        """Wait for a cross-process turn lease without holding a PostgreSQL lock. ``on_wait(elapsed)`` is
        best-effort: called when another holder has the lease and about every
        ``wait_notice_interval_seconds`` after. A busy database is not a holder: the attempt is
        retried at once with a longer write patience, and ``on_contended()`` is called instead,
        since the busy writer may be the holder's last flush. ``should_abort()`` True (e.g.
        ``/stop``) returns False at once."""
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        patience = acquire_patience_s
        wait_started = None
        last_notice_at = None
        notice_every = max(0.0, float(wait_notice_interval_seconds))
        while True:
            if should_abort is not None:
                try:
                    if should_abort():
                        return False
                except Exception:
                    logger.debug("session turn lease should_abort callback failed", exc_info=True)
            try:
                if self.try_acquire_session_turn_lease(
                    session_id, holder, ttl_seconds=ttl_seconds, patience_s=patience):
                    return True
            except (psycopg.errors.LockNotAvailable,psycopg.errors.DeadlockDetected) as exc:
                # Another writer's transaction outlasted the write patience. That is not a lease
                # holder, so no poll sleep or "another process" notice; the retry waits on the write
                # lock itself (the poll interval moves into its patience) and wins it once free.
                if on_contended is not None:
                    try:
                        on_contended()
                    except Exception:
                        logger.debug("session turn lease on_contended callback failed", exc_info=True)
                patience = min(acquire_patience_s + max(0.0, float(poll_interval_seconds)),
                               max(acquire_patience_s, deadline - time.monotonic()))
                if not self._sleep_before_write_retry(deadline, 0.0):
                    return False
                continue
            patience = acquire_patience_s
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False
            if wait_started is None:
                wait_started = now
            if on_wait is not None and (
                last_notice_at is None or notice_every == 0.0 or (now - last_notice_at) >= notice_every
            ):
                try:
                    on_wait(max(0.0, now - wait_started))
                except Exception:
                    logger.debug("session turn lease on_wait callback failed", exc_info=True)
                last_notice_at = now
            time.sleep(min(max(0.01, float(poll_interval_seconds)), remaining))
    session_unread = staticmethod(_NativeSessionDB.session_unread)

    def session_count_ge(self,minimum):
        return bool(self._one(sql.SQL('SELECT COUNT(*)>=%s AS enough FROM {}').format(self.table('sessions')),(minimum,))['enough'])

    def get_conversation_root(self,session_id):
        with self._transaction(read_only=True) as conn:
            return self._lease_key_on_conn(conn,session_id)

    def get_session_model_config_value(self,session_id,key,default=None):
        return _parse_model_config((self.get_session(session_id) or {}).get('model_config')).get(key,default)

    def update_conversation_model_config_value(self,session_id,key,update):
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            tip=self.get_compression_tip(session_id) or session_id
            row=self._session_on_conn(conn,tip,lock=True)
            if not row:return None
            config=_parse_model_config(row['model_config'])
            value=update(config.get(key))
            if value is None:config.pop(key,None)
            else:config[key]=value
            self._update(conn,'sessions',{'model_config':json.dumps(config)},'id',tip)
            return tip

    def get_recent_session_model_route(self,session_id):
        return self._one(sql.SQL("SELECT model,billing_provider,billing_base_url,billing_mode,api_call_count FROM {} WHERE session_id=%s AND task='' AND model<>'unknown' AND billing_provider<>'' ORDER BY last_seen DESC,native_row_order DESC LIMIT 1").format(self.table('session_model_usage')),(session_id,))

    def set_latest_user_api_content(self,session_id,content,api_content):
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            row=conn.execute(sql.SQL("SELECT * FROM {} WHERE session_id=%s AND role='user' AND active=1 ORDER BY id DESC LIMIT 1 FOR UPDATE").format(self.table('messages')),(session_id,)).fetchone()
            return self._set_pg_api_content(conn,row,content,api_content)

    def set_message_api_content(self,session_id,row_id,content,api_content):
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            row=conn.execute(sql.SQL("SELECT * FROM {} WHERE session_id=%s AND id=%s AND role='user' AND active=1 FOR UPDATE").format(self.table('messages')),(session_id,row_id)).fetchone()
            return self._set_pg_api_content(conn,row,content,api_content)

    def _set_pg_api_content(self,conn,row,content,api_content):
        from hermes_state_postgresql_transcript import native_row
        from hermes_state_messages import _scrub_surrogates
        if not row or native_row(row)['content']!=self._encode_content(content):return 0
        return self._update(conn,'messages',{'api_content':_scrub_surrogates(api_content)},'id',row['id'])

    def backfill_null_session_profiles(self,profile_name):
        value=(profile_name or '').strip()
        if not value:return 0
        return self._write_rowcount("UPDATE sessions SET profile_name=%s WHERE profile_name IS NULL OR TRIM(profile_name)=''",(value,))

    _is_explicit_fork_child_row = _NativeSessionDB._is_explicit_fork_child_row
    CANONICAL_BOT_CHAT_TITLE = _NativeSessionDB.CANONICAL_BOT_CHAT_TITLE
    declared_scope_identity = _NativeSessionDB.declared_scope_identity
    is_explicit_fork_child = _NativeSessionDB.is_explicit_fork_child

    def has_gateway_input_owner(self,session_id,owner):
        rows=self._all(sql.SQL("SELECT display_metadata FROM {} WHERE session_id=%s AND role='user' AND observed=0 AND (active=1 OR compacted=1)").format(self.table('messages')),(session_id,))
        for row in rows:
            try:value=json.loads(row['display_metadata'] or 'null')
            except (ValueError,TypeError):continue
            if isinstance(value,dict) and value.get('gateway_input_owner')==owner:return True
        return False

    def session_ids_with_model_config_key(self,key):
        return [row['id'] for row in self._all(sql.SQL('SELECT id,model_config FROM {} ORDER BY started_at DESC').format(self.table('sessions'))) if key in _parse_model_config(row['model_config']) and _parse_model_config(row['model_config'])[key] is not None]

    def set_user_message_content(self,session_id,row_id,content):
        if not session_id or isinstance(row_id,bool) or not isinstance(row_id,int) or row_id<=0:return 0
        from hermes_state_postgresql_transcript import pg_values
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            row=conn.execute(sql.SQL("SELECT id FROM {} WHERE id=%s AND session_id=%s AND role='user' AND active=1 FOR UPDATE").format(self.table('messages')),(row_id,session_id)).fetchone()
            if not row:return 0
            count=self._update(conn,'messages',pg_values({'content':self._encode_content(content)}),'id',row_id)
            self._reconcile_pg_display_orders(conn,session_id)
            return count

    def find_latest_gateway_session_for_peer(self,*,source,user_id=None,session_key=None,chat_id=None,chat_type=None,thread_id=None):
        if not session_key:return None
        row=self._read_one(_PEER_BY_KEY_PG,(session_key,source))
        if row is None and chat_id is not None and chat_type is not None:
            owner=self._own_profile_name()
            row=self._read_one(_PEER_BY_TUPLE_PG,(source,user_id,chat_id,chat_type,thread_id,owner,owner,owner,owner,owner,owner))
        return _NativeSessionDB._session_row_dict(row) if row else None

    def _own_profile_name(self):
        import re
        from hermes_constants import get_default_hermes_root
        root=get_default_hermes_root().resolve()
        if self.home==root:return 'default'
        if self.home.parent==root/'profiles' and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',self.home.name):return self.home.name
        return None

    def resolve_session_by_title(self,title):
        from hermes_state_common import escape_like
        exact=self.get_session_by_title(title)
        if exact and title==self.CANONICAL_BOT_CHAT_TITLE:return exact['id']
        rows=self._all(sql.SQL("SELECT id FROM {} WHERE title LIKE %s ESCAPE E'\\\\' ORDER BY started_at DESC").format(self.table('sessions')),(escape_like(title)+' #%',))
        return rows[0]['id'] if rows else exact['id'] if exact else None

    def latest_conversation_boundary(self,session_key,source):
        if not session_key or not source:return None
        row=self._one(sql.SQL('SELECT generation FROM {} WHERE source=%s AND session_key=%s').format(self.table('conversation_generations')),(source,session_key))
        generation=int(row['generation'] or 0) if row else 0
        return generation if generation>0 else None
