from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Protocol


@dataclass(frozen=True)
class ExecutionBoundaryRequest:
    source: str
    session_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryPaths:
    hermes_home: Path | None = None
    terminal_cwd: Path | None = None
    runs_root: Path | None = None
    uploads_root: Path | None = None
    artifacts_root: Path | None = None


@dataclass(frozen=True)
class BoundaryPolicy:
    provider_fallback_enabled: bool = True
    require_boundary: bool = False


@dataclass(frozen=True)
class ExecutionBoundary:
    source: str
    session_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    paths: BoundaryPaths = field(default_factory=BoundaryPaths)
    policy: BoundaryPolicy = field(default_factory=BoundaryPolicy)
    audit_metadata: Mapping[str, object] = field(default_factory=dict)
    subprocess_env: Mapping[str, str] = field(default_factory=dict)


class GovernedExecutionBoundaryRequired(RuntimeError):
    pass


class ExecutionBoundaryProvider(Protocol):
    def resolve(self, request: ExecutionBoundaryRequest) -> ExecutionBoundary | None:
        ...


_provider: ExecutionBoundaryProvider | None = None
_current_boundary: ContextVar[ExecutionBoundary | None] = ContextVar(
    "current_execution_boundary", default=None
)


def register_execution_boundary_provider(provider: ExecutionBoundaryProvider) -> None:
    global _provider
    if _provider is not None and _provider is not provider:
        raise RuntimeError("execution boundary provider already registered")
    _provider = provider


def replace_execution_boundary_provider(provider: ExecutionBoundaryProvider) -> None:
    global _provider
    _provider = provider


def clear_execution_boundary_provider() -> None:
    global _provider
    _provider = None


def get_execution_boundary_provider() -> ExecutionBoundaryProvider | None:
    return _provider


def current_execution_boundary() -> ExecutionBoundary | None:
    return _current_boundary.get()


def resolve_execution_boundary(
    request: ExecutionBoundaryRequest,
) -> ExecutionBoundary | None:
    provider = get_execution_boundary_provider()
    if provider is None:
        return None
    return provider.resolve(request)


def _boundary_env(boundary: ExecutionBoundary) -> dict[str, str]:
    env: dict[str, str] = {}
    if boundary.paths.hermes_home is not None:
        env["HERMES_HOME"] = str(boundary.paths.hermes_home)
    if boundary.paths.terminal_cwd is not None:
        env["TERMINAL_CWD"] = str(boundary.paths.terminal_cwd)
    roots = [
        boundary.paths.runs_root,
        boundary.paths.uploads_root,
        boundary.paths.artifacts_root,
    ]
    safe_roots = [str(path) for path in roots if path is not None]
    if safe_roots:
        env["HERMES_WRITE_ALLOWED_ROOTS"] = ",".join(safe_roots)
    env["HERMES_PROVIDER_FALLBACK_ENABLED"] = (
        "1" if boundary.policy.provider_fallback_enabled else "0"
    )
    env.update(dict(boundary.subprocess_env))
    return env


@contextlib.contextmanager
def bind_execution_boundary(boundary: ExecutionBoundary | None) -> Iterator[None]:
    if boundary is None:
        yield
        return

    token = _current_boundary.set(boundary)
    updates = _boundary_env(boundary)
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        _current_boundary.reset(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
