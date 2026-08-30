"""Trusted workspace runtime bindings for gateway operations."""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from typing import Any, Iterator


def _session_segment(target_home: Path, session_id: str) -> str:
    segment = str(session_id or "").strip()
    prefix = f"{target_home.name}:"
    if segment.startswith(prefix):
        segment = segment[len(prefix) :]
    if not segment or segment in {".", ".."}:
        raise ValueError("session_id required")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", segment):
        raise ValueError("session_id must be an ASCII-stable path segment")
    if segment != Path(segment).name or any(sep in segment for sep in ("/", "\\")):
        raise ValueError("session_id must be a safe single path segment")
    return segment


@contextlib.contextmanager
def bound_workspace_hermes_home(
    raw_home: str | os.PathLike[str] | None,
    session_id: str | None = None,
) -> Iterator[Path | None]:
    value = str(raw_home or "").strip()
    if not value:
        yield None
        return
    target_home = Path(value).expanduser().resolve()
    wiki_root = target_home / "wiki"
    wiki_governance_root = wiki_root / ".governance"
    wiki_contracts_root = wiki_governance_root / "contracts"
    wiki_reports_root = wiki_governance_root / "reports"
    for root in (wiki_contracts_root, wiki_reports_root):
        root.mkdir(parents=True, exist_ok=True)
    updates = {
        "HERMES_HOME": str(target_home),
        "TERMINAL_CWD": str(target_home),
        "WIKI_PATH": str(wiki_root.resolve()),
        "WIKI_GOVERNANCE_PATH": str(wiki_governance_root.resolve()),
        "WIKI_CONTRACTS_PATH": str(wiki_contracts_root.resolve()),
        "WIKI_REPORTS_PATH": str(wiki_reports_root.resolve()),
        "WIKI_DEPENDENCY_GRAPH_PATH": str(
            (wiki_governance_root / "dependency-graph.json").resolve()
        ),
    }
    if session_id:
        segment = _session_segment(target_home, session_id)
        session_root = target_home / "sessions" / segment
        runs_root = session_root / "runs"
        uploads_root = session_root / "uploads"
        artifacts_root = session_root / "artifacts"
        for root in (runs_root, uploads_root, artifacts_root):
            root.mkdir(parents=True, exist_ok=True)
        updates.update(
            {
                "HERMES_WRITE_ALLOWED_ROOTS": ",".join(
                    str(root.resolve())
                    for root in (runs_root, uploads_root, artifacts_root)
                ),
                "SEMANTIER_WORKSPACE_RUNS_DIR": str(runs_root.resolve()),
                "SEMANTIER_WORKSPACE_ARTIFACTS_DIR": str(artifacts_root.resolve()),
                "HERMES_PROVIDER_FALLBACK_ENABLED": "0",
            }
        )
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield target_home
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def merge_workspace_tool_config(
    base_config: dict[str, Any], workspace_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge only workspace-owned tool/plugin configuration surfaces."""
    merged = dict(base_config)
    for key in ("platform_toolsets", "mcp_servers", "plugins"):
        workspace_values = workspace_config.get(key)
        if not isinstance(workspace_values, dict):
            continue
        combined = dict(base_config.get(key) or {})
        for item_key, item_value in workspace_values.items():
            if isinstance(item_value, list):
                existing = combined.get(item_key)
                values = list(existing) if isinstance(existing, list) else []
                values.extend(value for value in item_value if value not in values)
                combined[item_key] = values
            else:
                combined[item_key] = item_value
        merged[key] = combined
    return merged


def discover_workspace_plugins_and_config(
    base_config: dict[str, Any],
    workspace_home: str | os.PathLike[str] | None,
    *,
    session_id: str | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    """Discover workspace plugins under a trusted bound home."""
    if not workspace_home:
        return base_config
    with bound_workspace_hermes_home(workspace_home, session_id=session_id):
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins(force=True)
        except Exception:
            pass
        try:
            from hermes_cli.config import read_raw_config

            workspace_config = read_raw_config()
        except Exception:
            workspace_config = None
    if not isinstance(workspace_config, dict) or not workspace_config:
        return base_config
    return merge_workspace_tool_config(base_config, workspace_config) if merge else workspace_config
