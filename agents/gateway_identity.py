"""Optional gateway identity shim for non-Semantier checkouts."""

from __future__ import annotations

from pathlib import Path


def list_user_records() -> list[dict]:
    """Return no records when the Semantier identity store is unavailable."""
    return []


def get_user_record(_user_id: str) -> dict | None:
    """Return no identity record when the Semantier store is unavailable."""
    return None


def ensure_workspace_paths(workspace_id: str) -> tuple[Path, Path]:
    """Return the conventional workspace-root pair for tests that patch this shim."""
    workspace_root = Path.cwd() / "workspaces" / str(workspace_id)
    return workspace_root, workspace_root / ".hermes"
