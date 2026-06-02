"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None

logger = logging.getLogger(__name__)

_TRAJECTORY_METADATA_ALLOWLIST = {
    "timestamp",
    "session_id",
    "workspace_id",
    "source_gateway",
    "platform_session_key",
    "chat_id",
    "thread_id",
    "origin_user_id",
}


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def save_trajectory(
    trajectory: List[Dict[str, Any]],
    model: str,
    completed: bool,
    filename: str = None,
    metadata: dict[str, Any] | None = None,
):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    sanitized_metadata = _sanitize_metadata(metadata)
    entry = {
        "conversations": trajectory,
        "model": model,
        "completed": completed,
    }
    if "timestamp" in sanitized_metadata:
        entry["timestamp"] = sanitized_metadata.pop("timestamp")
    if sanitized_metadata:
        entry.update(sanitized_metadata)

    try:
        path = Path(filename)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)


def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in _TRAJECTORY_METADATA_ALLOWLIST:
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "timestamp":
            normalized = _normalize_timestamp(value)
            if normalized is not None:
                sanitized[key] = normalized
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
    return sanitized


def _normalize_timestamp(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None
