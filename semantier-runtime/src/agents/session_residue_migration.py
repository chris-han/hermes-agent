"""Legacy session-residue migration (P0.4).

Implements the deterministic policy for `.semantier-home/sessions/` leftovers:

  * Relocate: if a session file's id has the form `{workspace_id}:{tail}` and
    that workspace exists under `workspaces/`, move the file (and its sibling
    `.jsonl` / `.trajectory.jsonl`) into the owning workspace's session dir.
  * Quarantine (default): otherwise, move the file to
    `.semantier-home/sessions/_quarantine/` so it is no longer read by any
    authoritative session path.
  * Delete: only via the explicit `delete_quarantine` operator flag; never the
    default.

Authenticated session authority lives exclusively under
`workspaces/<workspace_id>/sessions/`. After migration, the shared
`.semantier-home/sessions/` directory must contain no top-level session files;
only the `_quarantine/` subdirectory may remain.

See `docs/operational/saas-multitenancy-roadmap.md` P0.4.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from runtime_paths import platform_runtime_root, workspace_root_path

_LOGGER = logging.getLogger(__name__)

# Session file naming follows workspace_session_logs.py:
#   session_<quoted_id>.json
#   <quoted_id>.jsonl
#   session_<quoted_id>.trajectory.jsonl
# We match either the canonical form (URL-quoted id) or the legacy raw form.
_SESSION_LOG_RE = re.compile(r"^session_(?P<id>.+?)(?<!\.trajectory)\.json$")
_SESSION_TRAJ_RE = re.compile(r"^session_(?P<id>.+)\.trajectory\.jsonl$")
_SESSION_JSONL_RE = re.compile(r"^(?P<id>.+)\.jsonl$")


@dataclass
class MigrationReport:
    relocated: list[tuple[Path, Path]] = field(default_factory=list)
    quarantined: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def _shared_sessions_dir() -> Path:
    return platform_runtime_root() / "sessions"


def _decode_session_id(quoted: str) -> str:
    from urllib.parse import unquote

    return unquote(quoted)


def _session_id_from_filename(name: str) -> str | None:
    for pattern in (_SESSION_LOG_RE, _SESSION_TRAJ_RE):
        m = pattern.match(name)
        if m:
            return _decode_session_id(m.group("id"))
    m = _SESSION_JSONL_RE.match(name)
    if m and not name.endswith(".trajectory.jsonl"):
        return _decode_session_id(m.group("id"))
    return None


def _workspace_id_of(session_id: str) -> str | None:
    if ":" not in session_id:
        return None
    candidate, _ = session_id.split(":", 1)
    candidate = candidate.strip()
    return candidate or None


def _local_session_id(workspace_id: str, session_id: str) -> str:
    prefix = f"{workspace_id}:"
    if session_id.startswith(prefix):
        return session_id[len(prefix) :]
    return session_id


def _target_workspace_session_log_path(target_workspace: Path, session_id: str, filename: str) -> Path:
    workspace_id = target_workspace.name
    local_session_id = _local_session_id(workspace_id, session_id)
    return target_workspace / "sessions" / local_session_id / "logs" / filename


def _workspace_dir_exists(workspace_id: str) -> Path | None:
    try:
        ws_root = workspace_root_path(workspace_id)
    except ValueError:
        return None
    return ws_root if ws_root.exists() else None


def _move(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def migrate_legacy_session_residue(
    *,
    dry_run: bool = False,
    delete_quarantine: bool = False,
    sessions_dir: Path | None = None,
) -> MigrationReport:
    """Move residue out of the shared sessions dir using the documented policy.

    Args:
        dry_run: when True, no filesystem changes are performed.
        delete_quarantine: when True, files that would otherwise be quarantined
            are deleted instead. Default False (quarantine is the safe default).
        sessions_dir: override of the shared residue location (for tests).
    """
    shared = sessions_dir if sessions_dir is not None else _shared_sessions_dir()
    report = MigrationReport()
    if not shared.exists() or not shared.is_dir():
        return report

    quarantine_dir = shared / "_quarantine"
    for entry in sorted(shared.iterdir()):
        if not entry.is_file():
            # subdirectories (including _quarantine) are left alone
            continue
        session_id = _session_id_from_filename(entry.name)
        if session_id is None:
            report.skipped.append(entry)
            continue
        workspace_id = _workspace_id_of(session_id)
        target_workspace = _workspace_dir_exists(workspace_id) if workspace_id else None
        if target_workspace is not None:
            target = _target_workspace_session_log_path(target_workspace, session_id, entry.name)
            if target.exists():
                # Workspace already owns a canonical copy; the shared one is
                # stale residue and goes to quarantine to avoid silent merges.
                dst = quarantine_dir / entry.name
            else:
                dst = target
                report.relocated.append((entry, dst))
                _move(entry, dst, dry_run=dry_run)
                continue
        else:
            dst = quarantine_dir / entry.name

        if delete_quarantine:
            report.deleted.append(entry)
            if not dry_run:
                entry.unlink()
        else:
            report.quarantined.append(entry)
            _move(entry, dst, dry_run=dry_run)

    return report
