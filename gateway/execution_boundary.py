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
    logs_root: Path | None = None


@dataclass(frozen=True)
class BoundaryPolicy:
    provider_fallback_enabled: bool = True
    require_boundary: bool = False
    allowed_read_roots: tuple[Path, ...] = ()
    allowed_write_roots: tuple[Path, ...] = ()
    scratch_roots: tuple[Path, ...] = (Path("/tmp"),)


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


class ExecutionBoundaryProvider(Protocol):
    def resolve(self, request: ExecutionBoundaryRequest) -> ExecutionBoundary | None: ...


class GovernedExecutionBoundaryRequired(RuntimeError):
    pass


class BoundaryPathRejected(RuntimeError):
    pass


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
    roots = list(boundary.policy.allowed_write_roots) or [
        boundary.paths.runs_root,
        boundary.paths.uploads_root,
        boundary.paths.artifacts_root,
        boundary.paths.logs_root,
    ]
    safe_roots = [str(path) for path in roots if path is not None]
    if safe_roots:
        env["HERMES_WRITE_ALLOWED_ROOTS"] = ",".join(safe_roots)
    if boundary.paths.artifacts_root is not None:
        env["SEMANTIER_WORKSPACE_ARTIFACTS_DIR"] = str(boundary.paths.artifacts_root)
    if boundary.paths.runs_root is not None:
        env["SEMANTIER_WORKSPACE_RUNS_DIR"] = str(boundary.paths.runs_root)
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
    sandbox_context = contextlib.nullcontext()
    if boundary.workspace_id and boundary.session_id:
        from agents.sandbox_scope import SandboxScope, bind_sandbox_scope

        sandbox_context = bind_sandbox_scope(
            SandboxScope(
                workspace_id=boundary.workspace_id,
                lane="interactive_session",
                scope_id=boundary.session_id,
                adapter_key=str(boundary.audit_metadata.get("adapter_key") or "")
                or None,
                network_enabled=bool(
                    boundary.audit_metadata.get("network_enabled", False)
                ),
            )
        )
    try:
        with sandbox_context:
            yield
    finally:
        _current_boundary.reset(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class DiskIOBoundaryGuard:
    """Resolve authenticated file IO through an active execution boundary."""

    def __init__(self, boundary: ExecutionBoundary | None) -> None:
        if boundary is None:
            raise GovernedExecutionBoundaryRequired(
                "SEMANTIER_EXECUTION_BOUNDARY_REQUIRED: active execution boundary required"
            )
        self.boundary = boundary

    @classmethod
    def current(cls) -> "DiskIOBoundaryGuard":
        return cls(current_execution_boundary())

    def _read_roots(self) -> tuple[Path, ...]:
        roots = list(self.boundary.policy.allowed_read_roots)
        if not roots:
            roots.extend(
                root
                for root in (
                    self.boundary.paths.hermes_home,
                    self.boundary.paths.terminal_cwd,
                    self.boundary.paths.uploads_root,
                    self.boundary.paths.artifacts_root,
                )
                if root is not None
            )
        return tuple(_resolve_path(root) for root in roots)

    def _write_roots(self) -> tuple[Path, ...]:
        roots = list(self.boundary.policy.allowed_write_roots)
        if not roots:
            roots.extend(
                root
                for root in (
                    self.boundary.paths.runs_root,
                    self.boundary.paths.uploads_root,
                    self.boundary.paths.artifacts_root,
                    self.boundary.paths.logs_root,
                )
                if root is not None
            )
        return tuple(_resolve_path(root) for root in roots)

    def resolve_read_path(
        self,
        requested_path: str | Path,
        *,
        purpose: str = "read",
        manifest_allowed_files: set[Path] | None = None,
    ) -> Path:
        resolved = _resolve_path(requested_path)
        if purpose == "skill_asset":
            allowed = {_resolve_path(path) for path in (manifest_allowed_files or set())}
            if resolved in allowed:
                return resolved
            raise BoundaryPathRejected(
                f"BOUNDARY_READ_REJECTED: {resolved} is not pinned in the bundled skill manifest"
            )
        if any(_is_under(resolved, root) for root in self._read_roots()):
            return resolved
        raise BoundaryPathRejected(
            f"BOUNDARY_READ_REJECTED: {resolved} is outside the active execution boundary"
        )

    def resolve_write_path(
        self, requested_path: str | Path, *, purpose: str = "write"
    ) -> Path:
        resolved = _resolve_path(requested_path)
        if any(_is_under(resolved, root) for root in self._write_roots()):
            return resolved
        artifacts_root = self.boundary.paths.artifacts_root
        if artifacts_root is not None:
            for scratch_root in self.boundary.policy.scratch_roots:
                scratch = _resolve_path(scratch_root)
                if _is_under(resolved, scratch):
                    return (_resolve_path(artifacts_root) / scratch.name / resolved.relative_to(scratch)).resolve()
        raise BoundaryPathRejected(
            f"BOUNDARY_WRITE_REJECTED: {resolved} is outside the active execution boundary"
        )
