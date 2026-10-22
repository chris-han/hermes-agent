from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def _model_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_connected_edge(row: dict[str, Any], parent_id: str) -> bool:
    if str(row.get("parent_session_id") or "") != parent_id:
        return False
    source = str(row.get("source") or "")
    if source in {"tool", "kanban", "oneshot"}:
        return False
    config = _model_config(row.get("model_config"))
    if config.get("_delegate_from") or config.get("_reset_from"):
        return False
    return True


def build_session_lineage(
    db: Any,
    session_id: str,
    *,
    belongs_to_workspace: Callable[[str], bool],
) -> dict[str, Any] | None:
    """Project the user-visible session lineage from canonical SessionDB state."""

    resolved = db.resolve_session_id(session_id)
    if (
        not resolved
        or not belongs_to_workspace(resolved)
        or db.get_session(resolved) is None
    ):
        return None

    rows = db.list_session_records()
    branch_rows = db.get_branch_records()
    by_id = {
        str(row.get("id") or ""): row
        for row in rows
        if str(row.get("id") or "") and belongs_to_workspace(str(row.get("id") or ""))
    }
    if resolved not in by_id:
        return None

    root = resolved
    seen = {root}
    while True:
        parent = str(by_id[root].get("parent_session_id") or "")
        if (
            not parent
            or parent not in by_id
            or parent in seen
            or not _is_connected_edge(by_id[root], parent)
        ):
            break
        root = parent
        seen.add(root)

    children: dict[str, list[str]] = {}
    for child_id, row in by_id.items():
        parent_id = str(row.get("parent_session_id") or "")
        if parent_id in by_id and _is_connected_edge(row, parent_id):
            children.setdefault(parent_id, []).append(child_id)

    for child_ids in children.values():
        child_ids.sort(
            key=lambda child_id: (
                float(by_id[child_id].get("started_at") or 0),
                child_id,
            )
        )

    ordered: list[str] = []
    frontier = [root]
    visited: set[str] = set()
    while frontier:
        current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        ordered.append(current)
        frontier.extend(children.get(current, []))

    branch_meta: dict[str, dict[str, Any]] = {}
    for row in branch_rows:
        item = dict(row)
        branch_meta[str(item.get("child_session_id") or "")] = item

    sessions: list[dict[str, Any]] = []
    for item_id in ordered:
        row = by_id[item_id]
        branch = branch_meta.get(item_id, {})
        title = str(row.get("title") or row.get("display_name") or "").strip()
        sessions.append(
            {
                "id": item_id,
                "title": title or item_id,
                "parent_session_id": str(row.get("parent_session_id") or "") or None,
                "started_at": float(row.get("started_at") or 0),
                "branch_point_message_id": branch.get("branch_point_message_id"),
                "branch_point_sequence": branch.get("branch_point_sequence"),
                "is_current": item_id == resolved,
            }
        )

    return {
        "current_session_id": resolved,
        "root_session_id": root,
        "sessions": sessions,
    }
