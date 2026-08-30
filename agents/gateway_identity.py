"""Optional governed identity seam supplied by Semantier core."""

from __future__ import annotations

from pathlib import Path


def list_user_records() -> list[dict]:
    return []


def get_user_record(_user_id: str) -> dict | None:
    return None


def ensure_workspace_paths(workspace_id: str) -> tuple[Path, Path]:
    workspace_root = Path.cwd() / "workspaces" / str(workspace_id)
    return workspace_root, workspace_root / ".hermes"
