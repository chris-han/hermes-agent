"""Cleanup of legacy transcript markers, inherited by SessionMessagesMixin."""
from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_state_common import _placeholders

logger = logging.getLogger("hermes_state")


class SessionMessageCleanupMixin:
    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
        """Permanently clear bare tool-call marker content ("[memory]") left by pre-fix sessions
        (``_rows_to_conversation`` repairs it in memory; this stops the re-scan). Only ``content`` is touched.
        ``backup``: ``VACUUM INTO`` snapshot first (none when nothing changes)."""
        from hermes_state import _STALE_TOOL_CALL_MARKER_RE
        def _find_affected(conn) -> list[int]:
            cursor = conn.execute("SELECT id, content FROM messages "
                "WHERE role = 'assistant' AND tool_calls IS NOT NULL AND tool_calls != ''")
            return [row["id"] for row in cursor.fetchall()
                if isinstance(row["content"], str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(row["content"].strip())]
        def _result(affected, backup_path=None):
            return {"dry_run": dry_run, "rows_affected": len(affected), "row_ids": affected, "backup_path": backup_path}
        with self._read_ctx() as conn:
            affected_ids = _find_affected(conn)
        if dry_run or not affected_ids:
            return _result(affected_ids)
        backup_path: Optional[str] = None
        if backup:
            import datetime
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.db_path.with_name(f"{self.db_path.name}.pre-clean-markers-backup-{stamp}"))
            with self._lock:
                self._conn.execute("VACUUM INTO ?", (backup_path,))
            logger.info("Backed up state.db to %s before clean-markers write", backup_path)
        def _do(conn):
            ids = _find_affected(conn)
            if ids:
                conn.execute(f"UPDATE messages SET content = '' WHERE id IN ({_placeholders(ids)})", ids)
            return ids
        affected_ids = self._execute_write(_do)
        if affected_ids:
            logger.info("Permanently cleared %d stale tool-call marker row(s) in state.db (#78148)", len(affected_ids))
        return _result(affected_ids, backup_path)
