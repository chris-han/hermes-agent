"""Shared conversation-continuity selection policies for gateways."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


History = List[Dict[str, Any]]


def select_continuity_history(
    stored_history: Optional[History],
    fallback_history: Optional[History],
) -> History:
    """Select the transcript to replay for a continued session.

    Stored history is authoritative when it exists. If a caller also provided
    history and the stored transcript is empty or unavailable, preserve that
    caller history instead of erasing context for short follow-ups like "yes".
    """
    if stored_history:
        return list(stored_history)
    return list(fallback_history or [])


def select_persisted_transcript(
    primary_history: Optional[History],
    fallback_history: Optional[History],
) -> History:
    """Select between two persisted transcript sources.

    Use the longer source because partial migrations or partial writes can
    leave one persisted store with only the newest turn while another store
    still has the complete transcript.
    """
    primary = list(primary_history or [])
    fallback = list(fallback_history or [])
    if len(fallback) > len(primary):
        return fallback
    return primary
