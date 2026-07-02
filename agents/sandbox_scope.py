from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal


_SANDBOX_KEY_ENV = "SEMANTIER_SANDBOX_KEY"
_scope_ctx: ContextVar["SandboxScope | None"] = ContextVar(
    "_semantier_sandbox_scope",
    default=None,
)

SandboxLane = Literal["interactive_session", "cron_job_run"]


@dataclass(frozen=True)
class SandboxScope:
    workspace_id: str
    lane: SandboxLane
    scope_id: str
    adapter_key: str | None
    network_enabled: bool


def sandbox_key_for_request(scope: SandboxScope) -> str:
    if scope.lane == "interactive_session":
        return f"ws:{scope.workspace_id}:session:{scope.scope_id}"
    if scope.lane == "cron_job_run":
        return f"ws:{scope.workspace_id}:cron:{scope.scope_id}"
    raise ValueError(f"unsupported sandbox scope lane: {scope.lane}")


def format_sandbox_run_timestamp_utc(value: datetime | str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("run timestamp required")
        return normalized
    if value.tzinfo is None:
        raise ValueError("run timestamp must be timezone-aware UTC")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cron_job_scope(
    *,
    workspace_id: str,
    job_id: str,
    run_timestamp_utc: datetime | str,
    adapter_key: str | None = None,
    network_enabled: bool = False,
) -> SandboxScope:
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise ValueError("job_id required")
    return SandboxScope(
        workspace_id=str(workspace_id or "").strip(),
        lane="cron_job_run",
        scope_id=f"{normalized_job_id}:{format_sandbox_run_timestamp_utc(run_timestamp_utc)}",
        adapter_key=adapter_key,
        network_enabled=network_enabled,
    )


def workspace_id_from_workdir(workdir: str | Path | None) -> str | None:
    value = str(workdir or "").strip()
    if not value:
        return None
    resolved = Path(value).expanduser().resolve()
    try:
        from runtime_paths import workspace_root_path
    except ImportError:
        return None
    preferred_candidates = [
        candidate
        for candidate in (resolved, *resolved.parents)
        if (candidate / ".semantier-home").exists()
    ]
    candidates = preferred_candidates or [resolved, *resolved.parents]
    for candidate in candidates:
        name = candidate.name.strip()
        if not name:
            continue
        try:
            if candidate == workspace_root_path(name):
                return name
        except ValueError:
            continue
    for candidate in (resolved, *resolved.parents):
        parent = candidate.parent
        if parent.name == "workspaces" and candidate.name.strip():
            return candidate.name.strip()
    return None


def cron_job_scope_if_resolvable(
    *,
    workdir: str | Path | None,
    job_id: str,
    run_timestamp_utc: datetime | str,
    adapter_key: str | None = None,
    network_enabled: bool = False,
) -> SandboxScope | None:
    workspace_id = workspace_id_from_workdir(workdir)
    if workspace_id is None:
        return None
    return cron_job_scope(
        workspace_id=workspace_id,
        job_id=job_id,
        run_timestamp_utc=run_timestamp_utc,
        adapter_key=adapter_key,
        network_enabled=network_enabled,
    )


def current_sandbox_scope() -> SandboxScope | None:
    return _scope_ctx.get()


def current_sandbox_key() -> str | None:
    scope = current_sandbox_scope()
    if scope is not None:
        return sandbox_key_for_request(scope)
    raw = os.environ.get(_SANDBOX_KEY_ENV)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def child_background_scope(parent_scope: SandboxScope | None) -> SandboxScope | None:
    return parent_scope


@contextlib.contextmanager
def bind_sandbox_scope(scope: SandboxScope | None) -> Iterator[None]:
    if scope is None:
        yield
        return

    key = sandbox_key_for_request(scope)
    prev = os.environ.get(_SANDBOX_KEY_ENV)
    token: Token = _scope_ctx.set(scope)
    os.environ[_SANDBOX_KEY_ENV] = key
    try:
        yield
    finally:
        _scope_ctx.reset(token)
        if prev is None:
            os.environ.pop(_SANDBOX_KEY_ENV, None)
        else:
            os.environ[_SANDBOX_KEY_ENV] = prev
