"""Native PostgreSQL usage statements with canonical accounting and route attribution."""
from __future__ import annotations
import time
from typing import Any,Optional
from psycopg import sql
_TOKEN_UPDATE_ABSOLUTE_SQL='UPDATE sessions SET\n                   input_tokens = %s,\n                   output_tokens = %s,\n                   cache_read_tokens = %s,\n                   cache_write_tokens = %s,\n                   reasoning_tokens = %s,\n                   estimated_cost_usd = COALESCE(%s, 0),\n                   actual_cost_usd = CASE\n                       WHEN %s::double precision IS NULL THEN actual_cost_usd\n                       ELSE %s\n                   END,\n                   cost_status = COALESCE(%s, cost_status),\n                   cost_source = COALESCE(%s, cost_source),\n                   pricing_version = COALESCE(%s, pricing_version),\n                   billing_provider = COALESCE(billing_provider, %s),\n                   billing_base_url = COALESCE(billing_base_url, %s),\n                   billing_mode = COALESCE(billing_mode, %s),\n                   model = COALESCE(model, %s),\n                   api_call_count = %s\n                   WHERE id = %s'
_TOKEN_UPDATE_DELTA_SQL='UPDATE sessions SET\n                   input_tokens = input_tokens + %s,\n                   output_tokens = output_tokens + %s,\n                   cache_read_tokens = cache_read_tokens + %s,\n                   cache_write_tokens = cache_write_tokens + %s,\n                   reasoning_tokens = reasoning_tokens + %s,\n                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(%s, 0),\n                   actual_cost_usd = CASE\n                       WHEN %s::double precision IS NULL THEN actual_cost_usd\n                       ELSE COALESCE(actual_cost_usd, 0) + %s\n                   END,\n                   cost_status = COALESCE(%s, cost_status),\n                   cost_source = COALESCE(%s, cost_source),\n                   pricing_version = COALESCE(%s, pricing_version),\n                   billing_provider = COALESCE(billing_provider, %s),\n                   billing_base_url = COALESCE(billing_base_url, %s),\n                   billing_mode = COALESCE(billing_mode, %s),\n                   model = COALESCE(model, %s),\n                   api_call_count = COALESCE(api_call_count, 0) + %s\n                   WHERE id = %s'
_MODEL_USAGE_UPSERT_SQL='INSERT INTO session_model_usage (\n                   session_id, model, billing_provider, billing_base_url, billing_mode,\n                   task, api_call_count, input_tokens, output_tokens,\n                   cache_read_tokens, cache_write_tokens, reasoning_tokens,\n                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,\n                   first_seen, last_seen\n               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)\n               DO UPDATE SET\n                   api_call_count = session_model_usage.api_call_count + excluded.api_call_count,\n                   input_tokens = session_model_usage.input_tokens + excluded.input_tokens,\n                   output_tokens = session_model_usage.output_tokens + excluded.output_tokens,\n                   cache_read_tokens = session_model_usage.cache_read_tokens + excluded.cache_read_tokens,\n                   cache_write_tokens = session_model_usage.cache_write_tokens + excluded.cache_write_tokens,\n                   reasoning_tokens = session_model_usage.reasoning_tokens + excluded.reasoning_tokens,\n                   estimated_cost_usd = session_model_usage.estimated_cost_usd + excluded.estimated_cost_usd,\n                   actual_cost_usd = session_model_usage.actual_cost_usd + excluded.actual_cost_usd,\n                   cost_status = COALESCE(excluded.cost_status, session_model_usage.cost_status),\n                   cost_source = COALESCE(excluded.cost_source, session_model_usage.cost_source),\n                   last_seen = excluded.last_seen'
_MODEL_USAGE_FIELDS=frozenset({'actual_cost_usd', 'input_tokens', 'billing_mode', 'cache_read_tokens', 'api_call_count', 'output_tokens', 'cost_source', 'reasoning_tokens', 'cost_status', 'cache_write_tokens', 'estimated_cost_usd', 'billing_base_url', 'model', 'billing_provider'})

class PostgreSQLUsageMixin:
    def queue_token_counts(self,session_id,**kwargs):
        self.update_token_counts(session_id,**kwargs)

    def flush_token_counts(self,timeout=5):return True

    def _insert_session_row(self,session_id,source,**kwargs):
        return self.create_session(session_id,source,**kwargs)

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str] = None,
    ) -> None:
        """Unconditionally set the billing route (``update_token_counts`` only COALESCE-fills
        NULLs) so the dashboard reflects the latest /model switch.

        Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness.

        See #48173, #48248.
        """
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        def _do(conn):
            conn.execute("""UPDATE sessions SET
                   billing_provider = %s,
                   billing_base_url = %s,
                   billing_mode = COALESCE(%s, billing_mode)
                   WHERE id = %s""", (provider, base_url, billing_mode, session_id))
        self._execute_write(_do)


    def update_token_counts(
        self, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str | None=None, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None,
        pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None,
        billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False,
        source: Optional[str]=None, task: str = "",
    ) -> None:
        """Update totals and route attribution, ensuring the legacy missing-row backfill.
        *absolute*=False increments (per-API-call deltas, CLI path); *absolute*=True sets directly
        (gateway path, where the cached agent holds cumulative totals). ``source`` is the session's
        real surface for the row-existence guard; callers that don't know it leave the placeholder.
        ``task`` names a per-turn route (``voice_chat``): its calls count in the session totals and
        land in ``session_model_usage`` under that task, but never become the session's recorded route."""
        values = dict(locals())
        values.pop('self')
        values.pop('session_id')
        values.pop('source')
        # Ensure the row exists: under concurrent load create_session() may have failed on
        # locking, and the UPDATE would silently affect 0 rows. When this guard is the first
        # writer it must carry the agent's real source: the turn lease treats an existing row as
        # proof the create already happened, so the creator never returns to repair an anonymous
        # ``unknown`` placeholder and the session stays a phantom for life (#111999).
        self._insert_session_row(session_id, source or "unknown", model=model)
        self._execute_write(lambda conn: self._update_token_counts_in_transaction(conn, session_id, **values))


    def _update_token_counts_in_transaction(
        self, conn, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str | None=None, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None,
        pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None,
        billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False, task: str="",
    ) -> None:
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        sql = _TOKEN_UPDATE_ABSOLUTE_SQL if absolute else _TOKEN_UPDATE_DELTA_SQL
        has_usage = bool(input_tokens or output_tokens or cache_read_tokens or cache_write_tokens or reasoning_tokens
                         or api_call_count or estimated_cost_usd)
        has_accounted_usage = bool(has_usage or actual_cost_usd)
        params = (
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
            estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None, model if has_accounted_usage else None,
            api_call_count, session_id)
        # Per-model attribution: the sessions row keeps one (model, provider) pair, so a
        # mid-session /model switch would attribute every token to the initial model. Only
        # the incremental path records here — absolute cumulative updates cannot be split
        # back into routes; Insights reconciles the residual instead.
        # ``update_token_counts`` is the single chokepoint every per-API-call delta flows through (CLI,
        # gateway, cron, delegated runs — see conversation_loop / codex_runtime), and each call carries the
        # model/provider *active at the time of that call*. Recording the per-call delta into
        # session_model_usage keyed by the live model preserves an accurate per-model breakdown regardless
        # of how many times the user switches. See #51607.
        record_model_usage = (not absolute) and has_usage

        row = conn.execute(
            "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = %s", (session_id,),
        ).fetchone()
        existing = dict(row) if row is not None else {}
        # create_session records the requested route before any API call. If that fails
        # and fallback succeeds, the first accounted usage is the authoritative route;
        # after that keep the row as is (one row cannot represent mixed usage).
        first_accounted_route = (
            not task and int(existing.get("api_call_count") or 0) == 0 and has_accounted_usage and bool(model)
            and bool(billing_provider)
            and (existing.get("model") != model or existing.get("billing_provider") != billing_provider)
        )
        if first_accounted_route:
            conn.execute("""UPDATE sessions
                   SET model = %s, billing_provider = %s,
                   billing_base_url = %s, billing_mode = %s
                   WHERE id = %s""", (model, billing_provider, billing_base_url, billing_mode, session_id))
        conn.execute(sql, params)
        if record_model_usage:
            self._record_model_usage(conn, session_id, task=task, **usage)


    def _record_model_usage(
        self, conn, session_id: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, billing_mode: Optional[str]=None, input_tokens: int=0,
        output_tokens: int=0, cache_read_tokens: int=0, cache_write_tokens: int=0, reasoning_tokens: int=0,
        estimated_cost_usd: Optional[float]=None, actual_cost_usd: Optional[float]=None,
        cost_status: Optional[str]=None, cost_source: Optional[str]=None, api_call_count: int=0, task: str="",
    ) -> None:
        """Accumulate a per-API-call usage delta into session_model_usage, inside the caller's
        write txn after the ``sessions`` UPDATE. A missing model/provider falls back to
        the session row — except for aux rows (``task`` set), which must NOT inherit the
        main-loop route (vision on gemini while the main loop runs anthropic): missing
        info stays 'unknown'/empty.

        ``task`` distinguishes what kind of work consumed the tokens: ``''`` (empty) is the main agent loop;
        auxiliary calls record their task name (``vision``, ``compression``, ``title_generation``, ...) via
        :meth:`record_auxiliary_usage` (issue #23270).
        """
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode FROM sessions WHERE id = %s", (session_id,),
        ).fetchone()
        sess = dict(row) if (row is not None and not task) else {}
        counts = [v or 0 for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)]
        now = time.time()
        conn.execute(_MODEL_USAGE_UPSERT_SQL, (
            session_id, model or sess.get("model") or "unknown",
            billing_provider or sess.get("billing_provider") or "",
            billing_base_url or sess.get("billing_base_url") or "",
            billing_mode or sess.get("billing_mode") or "", task or "", api_call_count or 0, *counts,
            float(estimated_cost_usd or 0.0), float(actual_cost_usd or 0.0), cost_status, cost_source, now, now))


    def record_auxiliary_usage(
        self, session_id: str, task: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        api_call_count: int=1,
    ) -> None:
        """Record an auxiliary LLM call's usage (vision, compression, title generation, ...)
        as a per-(model, provider, task) delta in ``session_model_usage`` WITHOUT touching
        the ``sessions`` summary row (the gateway overwrites those counters with absolute
        main-loop totals). ``api_call_count`` may aggregate N calls. Best-effort.

        See #23270.
        Background-review forks record an aggregate of N fork API calls in one write with
        ``task='background_review'`` (issue #87250).
        """
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        if not session_id or not task:
            return
        usage["api_call_count"] = 1 if api_call_count is None else int(api_call_count)
        # FK to sessions.id: same guard as update_token_counts; the aux path carries no surface, so
        # the placeholder stays repairable by the creator's upsert (_insert_session_row).
        self._insert_session_row(session_id, "unknown")
        self._execute_write(lambda conn: self._record_model_usage(conn, session_id, task=task, **usage))


    def usage_totals(self, *, min_message_count: int = 1, include_archived: bool = False) -> dict[str, float]:
        """Tokens and spend across the whole store (one scan), so the sidebar total does not
        shrink with paging. Spend prefers the billed figure over the estimate."""
        where = ["parent_session_id IS NULL", "message_count >= %s"]
        params: list[Any] = [min_message_count]
        if not include_archived:
            where.append("COALESCE(archived, 0) = 0")
        row = self._read_one(f"""
            SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0),
                   COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0)
              FROM sessions
             WHERE {' AND '.join(where)}
            """, params)
        return {"tokens": int(row[0] or 0), "cost_usd": float(row[1] or 0.0)}
