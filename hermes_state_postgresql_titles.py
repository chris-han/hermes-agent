"""PostgreSQL statements preserving canonical title identity and conflict rules."""
from __future__ import annotations
from typing import Optional
from psycopg import sql
from hermes_state import SessionDB
from hermes_state_titles import SessionTitlesMixin, _NUMBERED_TITLE_RE
from hermes_state_common import escape_like


class PostgreSQLTitlesMixin:
    TITLE_SOURCE_USER = SessionDB.TITLE_SOURCE_USER
    TITLE_SOURCE_DERIVED = SessionDB.TITLE_SOURCE_DERIVED
    TITLE_SOURCE_LLM = SessionDB.TITLE_SOURCE_LLM
    CANONICAL_BOT_CHAT_TITLE = SessionDB.CANONICAL_BOT_CHAT_TITLE
    _TITLE_SOURCE_RANK = SessionDB._TITLE_SOURCE_RANK
    _EMPTY_SESSION_WHERE = SessionDB._EMPTY_SESSION_WHERE
    _title_rank = classmethod(SessionTitlesMixin._title_rank.__func__)

    def _is_compression_ancestor(self, conn, *, ancestor_id, descendant_id):
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        return conn.execute(sql.SQL("""WITH RECURSIVE ancestors(id) AS (
            SELECT %s::text UNION
            SELECT parent.id FROM ancestors a
            JOIN {} child ON child.id=a.id
            JOIN {} parent ON parent.id=child.parent_session_id
            WHERE parent.end_reason='compression')
            SELECT 1 FROM ancestors WHERE id=%s AND id<>%s LIMIT 1""").format(
                self.table('sessions'),self.table('sessions')),
                (descendant_id,ancestor_id,descendant_id)).fetchone() is not None

    def get_next_title_in_lineage(self, base_title):
        match = _NUMBERED_TITLE_RE.match(base_title)
        base = match.group(1) if match else base_title
        rows = self._read_all("SELECT title FROM sessions WHERE title=%s OR title LIKE %s ESCAPE '\\'",(base,f"{escape_like(base)} #%"))
        if not rows:
            return base
        numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row[0]) for row in rows) if m]
        return f"{base} #{max([1,*numbers])+1}"

    def _set_session_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Write a title, enforcing provenance precedence. A ``user`` write always lands;
        ``derived``/``llm`` land only when the row is untitled or holds strictly lower
        authority (derived upgrades to llm exactly once, nothing overwrites a user name,
        re-running the titler on an llm row is a no-op). No writer may move a hidden
        canonical Bot Chat off its title. Read and write are one compare-and-swap
        transaction, so a manual ``/title`` racing an in-flight generation is not clobbered."""
        return bool(self._execute_write(
            lambda conn: self._set_session_title_in_transaction(conn, session_id, title, source=source)
        ))

    def _set_session_title_in_transaction(self, conn, session_id: str, title: str, *, source: str):
        """Reuse title rules within an authority-owned transaction; never commit here."""
        title = self.sanitize_title(title)
        is_user = source == self.TITLE_SOURCE_USER
        new_rank = self._title_rank(source) if not is_user else None
        affected = []

        current = conn.execute(
            "SELECT title, title_source, hidden FROM sessions WHERE id = %s", (session_id,),
        ).fetchone()
        if current is None:
            return []
        # The canonical Bot Chat's NAME is its identity (Bot Mode resolves it by
        # exact-title lookup on every open), so a rename orphans the conversation. Hidden
        # is the discriminator: canonical chats are born hidden; a visible session merely
        # named "Bot Chat" stays renameable. Provenance-blind.
        if ((current["title"] or "") == self.CANONICAL_BOT_CHAT_TITLE and bool(current["hidden"])
                and title != self.CANONICAL_BOT_CHAT_TITLE):
            if is_user:
                raise ValueError("This is the bot's canonical Bot Chat — its name is its "
                                 "identity, and renaming it would orphan the conversation. "
                                 "To start fresh, create a new bot instead.")
            return []
        if not is_user and current["title"] is not None and self._title_rank(current["title_source"]) >= new_rank:
            return []
        if title:
            cleared = self._resolve_title_conflict(conn, session_id, title)
            if cleared:
                affected.append(cleared)
        # CAS on the values just read (``IS`` is NULL-safe): a concurrent write between
        # the SELECT and here loses instead of being overwritten.
        changed = conn.execute(
            "UPDATE sessions SET title = %s, title_source = %s WHERE id = %s AND title IS NOT DISTINCT FROM %s AND title_source IS NOT DISTINCT FROM %s",
            (title, source if title else None, session_id, current["title"], current["title_source"]),
        ).rowcount
        return [*affected, session_id] if changed else affected

    def _resolve_title_conflict(self, conn, session_id: str, title: str) -> Optional[str]:
        """Free ``title`` for ``session_id`` inside the caller's write transaction, or raise
        ValueError when another session legitimately holds it. The one place the uniqueness
        rule lives, shared by renames and the API server's create-with-title. Returns the id
        whose title was cleared (an affected row for mutation receipts), else None."""
        conflict = conn.execute(
            f"SELECT id, archived, hidden, {self._EMPTY_SESSION_WHERE} AS ghost "
            "FROM sessions WHERE title = %s AND id != %s", (title, session_id),
        ).fetchone()
        if not conflict:
            return None
        conflict_id = conflict["id"]
        # A hidden compressed ancestor holding the title cannot be freed by the
        # user, so transfer it onto the tip (uniqueness + lineage kept).
        if self._is_compression_ancestor(conn, ancestor_id=conflict_id, descendant_id=session_id):
            conn.execute("UPDATE sessions SET title = NULL WHERE id = %s", (conflict_id,))
            return conflict_id
        # A deliberately archived hidden Bot Chat is a retired registry
        # entry, not a live identity. Retire its name in the same title
        # transaction so a replacement can become the sole canonical row;
        # the old session remains archived and otherwise untouched.
        # An ended, empty, visible ghost (abandoned new chat that listings filter
        # out, so the user cannot find it to free the name) yields too (#81888).
        # The full index stays: a ``message_count > 0`` partial index would make
        # the ghost's first append_message fail with IntegrityError.
        if (title == self.CANONICAL_BOT_CHAT_TITLE and bool(conflict["archived"])
              and bool(conflict["hidden"])) or (conflict["ghost"] and not conflict["hidden"]):
            conn.execute(
                "UPDATE sessions SET title = NULL, title_source = NULL WHERE id = %s",
                (conflict_id,),
            )
            return conflict_id
        raise ValueError(f"Title '{title}' is already in use by session {conflict_id}")

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a title on the user's behalf (``user`` provenance). Empty clears it. Raises
        ValueError on conflict or validation failure."""
        return self._set_session_title(session_id, title, source=self.TITLE_SOURCE_USER)

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Set an automatic title; False (untouched) when a higher-authority title holds the row."""
        if source not in (self.TITLE_SOURCE_DERIVED, self.TITLE_SOURCE_LLM):
            raise ValueError(f"invalid automatic title source: {source!r}")
        return self._set_session_title(session_id, title, source=source)
