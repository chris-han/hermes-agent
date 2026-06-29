"""
Session mirroring for cross-platform message delivery.

When a message is sent to a platform (via send_message or cron delivery),
this module appends a "delivery-mirror" record to the target session's
transcript so the receiving-side agent has context about what was sent.

Standalone -- works from CLI, cron, and gateway contexts without needing
the full SessionStore machinery.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workspace-session lookup helpers — patchable at module level for tests.
# Both functions perform lazy imports of agents.workspace_session_logs so
# that mirror.py remains importable in Hermes-only contexts where the
# Semantier package is not on sys.path.
# ---------------------------------------------------------------------------

def _workspace_find_matches(
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    origin_user_id: Optional[str] = None,
    canonical_session_id: Optional[str] = None,
) -> List[dict]:
    """Return workspace session index rows matching the given filters.

    Falls back to an empty list when the Semantier package is unavailable.
    Patchable by tests as ``gateway.mirror._workspace_find_matches``.
    """
    try:
        from agents.workspace_session_logs import find_workspace_session_index_matches
        return find_workspace_session_index_matches(
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            origin_user_id=origin_user_id,
            canonical_session_id=canonical_session_id,
        ) or []
    except Exception:
        return []


def _workspace_jsonl_path(session_id: str) -> Optional[Path]:
    """Return the workspace-scoped JSONL transcript path for *session_id*.

    Returns ``None`` when no workspace home is registered for the session or
    when the Semantier package is unavailable.
    Patchable by tests as ``gateway.mirror._workspace_jsonl_path``.
    """
    try:
        from agents.workspace_session_logs import find_workspace_session_index_matches, _session_jsonl_path
        matches = find_workspace_session_index_matches(canonical_session_id=session_id)
        if matches:
            workspace_home = matches[0].get("workspace_hermes_home")
            if isinstance(workspace_home, Path):
                return _session_jsonl_path(workspace_home, session_id)
    except Exception:
        pass
    return None


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then writes a mirror entry to both the JSONL transcript and SQLite DB.

    Returns True if mirrored successfully, False if no matching session or error.
    All errors are caught -- this is never fatal.
    """
    try:
        session_id = _find_session_id(
            platform,
            str(chat_id),
            thread_id=thread_id,
            user_id=user_id,
            session_id=session_id,
        )
        if not session_id:
            logger.debug(
                "Mirror: no session found for %s:%s:%s:%s",
                platform,
                chat_id,
                thread_id,
                user_id,
            )
            return False

        mirror_msg = {
            "role": "assistant",
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_jsonl(session_id, mirror_msg)
        _append_to_sqlite(session_id, mirror_msg)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug(
            "Mirror failed for %s:%s:%s:%s: %s",
            platform,
            chat_id,
            thread_id,
            user_id,
            e,
        )
        return False


def _find_session_id(
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[str]:
    """
    Find the active session_id for a platform + chat_id pair.

    Queries workspace session indexes for a match where origin.chat_id ==
    chat_id on the right platform.  DM session keys don't embed the chat_id
    (e.g. "agent:main:telegram:dm"), so we check the origin dict.

    When *user_id* is provided, prefer exact sender matches. If multiple
    same-chat candidates exist and none matches the user, return None instead
    of guessing and contaminating another participant's session.
    """
    candidates = _workspace_find_matches(
        platform=platform.lower(),
        chat_id=str(chat_id),
        thread_id=thread_id,
        origin_user_id=user_id,
        canonical_session_id=session_id,
    )
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda item: str(item.get("updated_at") or ""),
        reverse=True,
    )

    if session_id is None and user_id is None and thread_id is None and len(candidates) > 1:
        distinct_user_ids = {
            str((item.get("origin") or {}).get("user_id") or "").strip()
            for item in candidates
            if str((item.get("origin") or {}).get("user_id") or "").strip()
        }
        if len(distinct_user_ids) > 1:
            return None

    return str(candidates[0].get("session_id") or "") or None


def _append_to_jsonl(session_id: str, message: dict) -> None:
    """Append a message to the workspace-scoped JSONL transcript file.

    Writes only to the workspace-local path resolved by
    ``_workspace_jsonl_path``.  If no workspace home is known for the
    session the write is skipped rather than falling back to a shared
    filesystem store — preserving the contract that no shared session
    store surface is written to by authenticated session traffic.
    """
    transcript_path = _workspace_jsonl_path(session_id)
    if transcript_path is None:
        logger.debug(
            "Mirror: no workspace transcript path for session %s; skipping JSONL write",
            session_id,
        )
        return

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Mirror JSONL write failed: %s", e)


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database."""
    db = None
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        db.append_message(
            session_id=session_id,
            role=message.get("role", "assistant"),
            content=message.get("content"),
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
    finally:
        if db is not None:
            db.close()
