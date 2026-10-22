import contextlib
import os
import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Optional


_DEFAULT_RUNTIME_ROOT = ".semantier-home"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACES_ROOT = _REPO_ROOT / "workspaces"

_HERMES_HOME_ENV = "HERMES_HOME"
_WORKSPACE_RUNS_DIR_ENV = "SEMANTIER_WORKSPACE_RUNS_DIR"
_WORKSPACE_ARTIFACTS_DIR_ENV = "SEMANTIER_WORKSPACE_ARTIFACTS_DIR"
_WRITE_SAFE_ROOT_ENV = "HERMES_WRITE_SAFE_ROOT"
_WRITE_ALLOWED_ROOTS_ENV = "HERMES_WRITE_ALLOWED_ROOTS"
_TERMINAL_CWD_ENV = "TERMINAL_CWD"
_WIKI_PATH_ENV = "WIKI_PATH"
_SEMANTIER_WORKSPACE_ID_ENV = "SEMANTIER_WORKSPACE_ID"
_SEMANTIER_DISABLE_PROVIDER_FALLBACK_ENV = "SEMANTIER_DISABLE_PROVIDER_FALLBACK"
_SESSION_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._:-]+$")

# Per-context workspace bindings — set by bind_workspace_env and readable via
# current_workspace_hermes_home() / current_workspace_runs_dir().
# ContextVar values are context-local (thread/task-local when entered via
# asyncio.run_in_executor or asyncio.create_task), so concurrent requests for
# different workspaces see their own bindings without process-global races.
# os.environ is still mutated in parallel for subprocess inheritance, but all
# in-process readers must use the ContextVar accessors below.
_workspace_hermes_home_ctx: ContextVar[Optional[str]] = ContextVar(
    "_semantier_hermes_home", default=None
)
_workspace_runs_dir_ctx: ContextVar[Optional[str]] = ContextVar(
    "_semantier_workspace_runs_dir", default=None
)


def current_workspace_hermes_home() -> Optional[str]:
    """Return the workspace HERMES_HOME for the current execution context.

    Reads the ContextVar set by ``bind_workspace_env``. Falls back to
    ``os.environ`` for legacy callers that have not yet been updated, or when
    called from a subprocess context where ContextVars are not propagated.
    """
    ctx_val = _workspace_hermes_home_ctx.get()
    if ctx_val is not None:
        return ctx_val
    return os.environ.get(_HERMES_HOME_ENV)


def current_workspace_runs_dir() -> Optional[str]:
    """Return the active-session runs directory for the current execution context.

    Reads the ContextVar set by ``bind_workspace_env``. Falls back to
    ``os.environ`` for legacy callers that have not yet been updated, or when
    called from a subprocess context where ContextVars are not propagated.
    """
    ctx_val = _workspace_runs_dir_ctx.get()
    if ctx_val is not None:
        return ctx_val
    return os.environ.get(_WORKSPACE_RUNS_DIR_ENV)


def _validate_workspace_id(workspace_id: str) -> str:
    normalized = workspace_id.strip()
    if not normalized:
        raise ValueError("workspace_id required")
    if normalized != Path(normalized).name:
        raise ValueError("workspace_id must be a single path segment")
    if any(sep in normalized for sep in ("/", "\\")):
        raise ValueError("workspace_id must not contain path separators")
    return normalized


def _validate_session_segment(session_id: str) -> str:
    normalized = str(session_id or "").strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("session_id required")
    if not _SESSION_SEGMENT_RE.fullmatch(normalized):
        raise ValueError("session_id must be an ASCII-stable path segment")
    if normalized != Path(normalized).name or any(sep in normalized for sep in ("/", "\\")):
        raise ValueError("session_id must be a safe single path segment")
    return normalized


def workspace_session_segment(workspace_id: str, session_id: str) -> str:
    """Return the safe filesystem segment for a workspace-scoped session.

    ``session_id`` may be the canonical identity used by SQLite/audit
    (``<workspace_id>:session_xxx``) or an already-local session segment
    (``session_xxx``). Filesystem paths under ``workspaces/<id>/sessions`` must
    use the returned segment, never the canonical identity directly.
    """
    workspace = _validate_workspace_id(workspace_id)
    normalized = _validate_session_segment(session_id)
    prefix = f"{workspace}:"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
        normalized = _validate_session_segment(normalized)
    return normalized


def _workspace_session_segment(workspace_id: str, session_id: str) -> str:
    return workspace_session_segment(workspace_id, session_id)


def _session_segment_from_workspace_home(workspace_home: Path, session_id: str) -> str:
    return workspace_session_segment(workspace_home.name, session_id)


def platform_runtime_root() -> Path:
    """Return the shared launcher-managed runtime root."""
    raw = os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or _DEFAULT_RUNTIME_ROOT
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspace_root_path(workspace_id: str) -> Path:
    normalized = _validate_workspace_id(workspace_id)
    workspaces_root = _workspaces_root()
    base = workspaces_root.resolve()
    root = (workspaces_root / normalized).resolve()
    if root != base and base not in root.parents:
        raise ValueError("workspace path escapes workspaces root")
    return root


def _workspaces_root() -> Path:
    raw = os.environ.get("SEMANTIER_WORKSPACES_ROOT", "").strip()
    return Path(raw).expanduser() if raw else _WORKSPACES_ROOT


def workspace_runtime_home_path(workspace_id: str) -> Path:
    return workspace_root_path(workspace_id)


def workspace_hermes_home_path(workspace_id: str) -> Path:
    return workspace_runtime_home_path(workspace_id)


def workspace_wiki_root(workspace_id: str) -> Path:
    return workspace_root_path(workspace_id) / "wiki"


def workspace_wiki_governance_root(workspace_id: str) -> Path:
    return workspace_wiki_root(workspace_id) / ".governance"


def workspace_wiki_contracts_root(workspace_id: str) -> Path:
    return workspace_wiki_governance_root(workspace_id) / "contracts"


def workspace_wiki_reports_root(workspace_id: str) -> Path:
    return workspace_wiki_governance_root(workspace_id) / "reports"


def workspace_wiki_dependency_graph_path(workspace_id: str) -> Path:
    return workspace_wiki_governance_root(workspace_id) / "dependency-graph.json"


def workspace_sessions_root(workspace_id: str) -> Path:
    return workspace_runtime_home_path(workspace_id) / "sessions"


def _validate_runs_kind(kind: str) -> str:
    normalized = str(kind or "").strip()
    if not normalized:
        raise ValueError("runs kind required")
    if normalized != Path(normalized).name:
        raise ValueError("runs kind must be a single path segment")
    if any(sep in normalized for sep in ("/", "\\")) or normalized in (".", ".."):
        raise ValueError("runs kind must be a safe single segment")
    return normalized


def workspace_runs_root(workspace_id: str) -> Path:
    """Reject legacy flat per-workspace run roots for new writes."""
    _validate_workspace_id(workspace_id)
    raise ValueError("workspace runs require session_id; use workspace_session_runs_root")


def workspace_session_root(workspace_id: str, session_id: str) -> Path:
    return workspace_root_path(workspace_id) / "sessions" / workspace_session_segment(
        workspace_id, session_id
    )


def workspace_session_uploads_root(workspace_id: str, session_id: str) -> Path:
    return workspace_session_root(workspace_id, session_id) / "uploads"


def workspace_session_runs_root(workspace_id: str, session_id: str) -> Path:
    return workspace_session_root(workspace_id, session_id) / "runs"


def workspace_uploads_root(workspace_id: str) -> Path:
    """Reject legacy flat per-workspace upload roots for new writes."""
    raise ValueError("workspace uploads require session_id; use workspace_session_uploads_root")


def workspace_runs_dir(workspace_id: str, kind: str) -> Path:
    """Reject legacy flat per-workspace run roots for new writes."""
    _validate_runs_kind(kind)
    return workspace_runs_root(workspace_id)


def workspace_session_runs_dir(workspace_id: str, session_id: str, kind: str) -> Path:
    """Return the canonical per-session runs sub-directory for a run kind."""
    safe_kind = _validate_runs_kind(kind)
    return workspace_session_runs_root(workspace_id, session_id) / safe_kind


def _workspace_root_from_hermes_home(hermes_home: Path) -> Path:
    resolved = Path(hermes_home).expanduser().resolve()
    if resolved == platform_runtime_root().resolve():
        raise ValueError("not a workspace runtime home")
    workspaces_root = _workspaces_root().resolve()
    if resolved.parent == workspaces_root:
        return resolved
    raise ValueError("not a workspace runtime home")


def workspace_runs_root_from_hermes_home(hermes_home: Path) -> Path:
    """Reject legacy flat per-workspace run roots for new writes."""
    _workspace_root_from_hermes_home(hermes_home)
    raise ValueError("workspace runs require session_id; use workspace_session_runs_root_from_hermes_home")


def workspace_session_runs_root_from_hermes_home(
    hermes_home: Path, session_id: str
) -> Path:
    """Derive the canonical active-session runs root from a workspace home."""
    workspace_root = _workspace_root_from_hermes_home(hermes_home)
    session_segment = _session_segment_from_workspace_home(workspace_root, session_id)
    return workspace_root / "sessions" / session_segment / "runs"


def runtime_root() -> Path:
    """Return the shared local runtime root for SQLite-backed state."""
    return platform_runtime_root()


def sqlite_db_path(filename: str, *legacy_paths: str) -> Path:
    """Resolve a database path under the shared runtime root.

    Existing legacy files are moved into the unified runtime root on first use
    when the target file does not already exist.
    """
    target = runtime_root() / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return target

    for legacy_raw in legacy_paths:
        legacy = Path(legacy_raw).expanduser()
        if not legacy.exists():
            continue
        try:
            if legacy.resolve() == target.resolve():
                return target
        except OSError:
            pass
        legacy.replace(target)
        break

    return target


def bind_workspace_env(target_home: Path | str | None) -> contextlib.AbstractContextManager[None]:
    """Cross-gateway workspace env binding.

    Sets `HERMES_HOME` to `target_home` and, when `target_home` is a
    per-workspace runtime root (``workspaces/<id>``), exports workspace identity
    and safe-root context only. `SEMANTIER_WORKSPACE_RUNS_DIR` and
    `HERMES_WRITE_ALLOWED_ROOTS` require an active session id and are added by
    `bind_workspace_session_env()`. All env vars are restored on exit.

    Every gateway adapter (api_server, webapi, weixin, ...) must wrap its
    per-request handoff to LLM/tool execution in this context manager so
    downstream file-IO tools and tool subprocesses see a consistent
    workspace context regardless of which gateway started the request.
    """
    value = str(target_home or "").strip()
    if not value:
        raise ValueError("workspace HERMES_HOME binding requires a workspace runtime home")
    resolved_home = Path(value).expanduser().resolve()
    _workspace_root_from_hermes_home(resolved_home)
    return _bind_workspace_env_impl(resolved_home)


@contextlib.contextmanager
def _bind_workspace_env_impl(resolved_home: Path) -> Iterator[None]:
    resolved_home.mkdir(parents=True, exist_ok=True)

    # Semantier product defaults are seeded into every workspace HERMES_HOME.
    # Explicit user disable remains authoritative inside the seeder.
    from agents.default_workspace_plugins import ensure_default_workspace_plugins

    ensure_default_workspace_plugins(resolved_home)

    # --- os.environ (subprocess inheritance) ---
    prev_home = os.environ.get(_HERMES_HOME_ENV)
    os.environ[_HERMES_HOME_ENV] = str(resolved_home)

    prev_runs = os.environ.get(_WORKSPACE_RUNS_DIR_ENV)
    prev_write_safe_root = os.environ.get(_WRITE_SAFE_ROOT_ENV)
    prev_write_allowed_roots = os.environ.get(_WRITE_ALLOWED_ROOTS_ENV)
    prev_terminal_cwd = os.environ.get(_TERMINAL_CWD_ENV)
    prev_wiki_path = os.environ.get(_WIKI_PATH_ENV)
    prev_workspace_id = os.environ.get(_SEMANTIER_WORKSPACE_ID_ENV)
    runs_root = None
    workspace_root_env_set = False
    workspace_id_env_set = False
    workspace_root = _workspace_root_from_hermes_home(resolved_home)
    workspace_id = workspace_root.name
    wiki_root = workspace_root / "wiki"
    workspace_root.mkdir(parents=True, exist_ok=True)
    os.environ[_SEMANTIER_WORKSPACE_ID_ENV] = workspace_id
    os.environ[_WRITE_SAFE_ROOT_ENV] = str(workspace_root)
    os.environ.pop(_WORKSPACE_RUNS_DIR_ENV, None)
    os.environ.pop(_WRITE_ALLOWED_ROOTS_ENV, None)
    os.environ[_TERMINAL_CWD_ENV] = str(workspace_root)
    os.environ[_WIKI_PATH_ENV] = str(wiki_root)
    workspace_root_env_set = True
    workspace_id_env_set = True

    # --- ContextVar (in-process thread/task isolation) ---
    # ContextVar.set() returns a Token that can be passed to .reset() to
    # restore the previous value — safe for both concurrent threads and
    # nested bindings.  asyncio.run_in_executor copies the context
    # automatically, so executor threads see the correct per-request binding.
    home_token: Token = _workspace_hermes_home_ctx.set(str(resolved_home))
    runs_token: Optional[Token] = (
        _workspace_runs_dir_ctx.set(str(runs_root)) if runs_root is not None else None
    )
    # Bridge to hermes_constants so get_hermes_home() resolves the workspace
    # home inside this binding.  Set both the normal active-home ContextVar and
    # the higher-priority profile override used by cron profile jobs; otherwise
    # workspace-scoped tool code running inside a profile job would keep
    # resolving cron/state paths against the profile home.
    _reset_hho = None
    _hho_token = None
    try:
        from hermes_constants import (
            reset_hermes_home_override as _reset_hho,
            set_hermes_home_override as _set_hho,
        )

        _hho_token = _set_hho(resolved_home)
    except Exception:
        if _reset_hho is not None and _hho_token is not None:
            try:
                _reset_hho(_hho_token)
            except Exception:
                pass
        _hho_token = None

    try:
        yield
    finally:
        # Restore ContextVar first (cheapest, no side-effects)
        _workspace_hermes_home_ctx.reset(home_token)
        if runs_token is not None:
            _workspace_runs_dir_ctx.reset(runs_token)
        if _reset_hho is not None and _hho_token is not None:
            try:
                _reset_hho(_hho_token)
            except Exception:
                pass
        # Restore os.environ for subprocess inheritance
        if prev_home is None:
            os.environ.pop(_HERMES_HOME_ENV, None)
        else:
            os.environ[_HERMES_HOME_ENV] = prev_home
        if prev_runs is None:
            os.environ.pop(_WORKSPACE_RUNS_DIR_ENV, None)
        else:
            os.environ[_WORKSPACE_RUNS_DIR_ENV] = prev_runs
        if workspace_root_env_set:
            if prev_write_safe_root is None:
                os.environ.pop(_WRITE_SAFE_ROOT_ENV, None)
            else:
                os.environ[_WRITE_SAFE_ROOT_ENV] = prev_write_safe_root
            if prev_write_allowed_roots is None:
                os.environ.pop(_WRITE_ALLOWED_ROOTS_ENV, None)
            else:
                os.environ[_WRITE_ALLOWED_ROOTS_ENV] = prev_write_allowed_roots
            if prev_terminal_cwd is None:
                os.environ.pop(_TERMINAL_CWD_ENV, None)
            else:
                os.environ[_TERMINAL_CWD_ENV] = prev_terminal_cwd
            if prev_wiki_path is None:
                os.environ.pop(_WIKI_PATH_ENV, None)
            else:
                os.environ[_WIKI_PATH_ENV] = prev_wiki_path
        if workspace_id_env_set:
            if prev_workspace_id is None:
                os.environ.pop(_SEMANTIER_WORKSPACE_ID_ENV, None)
            else:
                os.environ[_SEMANTIER_WORKSPACE_ID_ENV] = prev_workspace_id


@contextlib.contextmanager
def bind_workspace_session_env(
    target_home: Path | str | None, session_id: str | None
) -> Iterator[None]:
    """Bind workspace env and allow writes to active session run/upload/artifact roots."""
    with bind_workspace_env(target_home):
        if target_home and session_id:
            workspace_root = Path(target_home).expanduser().resolve()
            session_root = workspace_root / "sessions" / _session_segment_from_workspace_home(
                workspace_root, session_id
            )
            uploads_root = session_root / "uploads"
            uploads_root.mkdir(parents=True, exist_ok=True)
            runs_root = session_root / "runs"
            runs_root.mkdir(parents=True, exist_ok=True)
            artifacts_root = session_root / "artifacts"
            artifacts_root.mkdir(parents=True, exist_ok=True)
            previous = os.environ.get(_WRITE_ALLOWED_ROOTS_ENV)
            previous_runs = os.environ.get(_WORKSPACE_RUNS_DIR_ENV)
            previous_artifacts = os.environ.get(_WORKSPACE_ARTIFACTS_DIR_ENV)
            previous_provider_fallback = os.environ.get(_SEMANTIER_DISABLE_PROVIDER_FALLBACK_ENV)
            os.environ[_WRITE_ALLOWED_ROOTS_ENV] = ",".join(
                [
                    str(runs_root.resolve()),
                    str(uploads_root.resolve()),
                    str(artifacts_root.resolve()),
                ]
            )
            os.environ[_WORKSPACE_RUNS_DIR_ENV] = str(runs_root.resolve())
            os.environ[_WORKSPACE_ARTIFACTS_DIR_ENV] = str(artifacts_root.resolve())
            os.environ[_SEMANTIER_DISABLE_PROVIDER_FALLBACK_ENV] = "1"
            runs_token = _workspace_runs_dir_ctx.set(str(runs_root.resolve()))
            try:
                yield
            finally:
                _workspace_runs_dir_ctx.reset(runs_token)
                if previous is None:
                    os.environ.pop(_WRITE_ALLOWED_ROOTS_ENV, None)
                else:
                    os.environ[_WRITE_ALLOWED_ROOTS_ENV] = previous
                if previous_runs is None:
                    os.environ.pop(_WORKSPACE_RUNS_DIR_ENV, None)
                else:
                    os.environ[_WORKSPACE_RUNS_DIR_ENV] = previous_runs
                if previous_artifacts is None:
                    os.environ.pop(_WORKSPACE_ARTIFACTS_DIR_ENV, None)
                else:
                    os.environ[_WORKSPACE_ARTIFACTS_DIR_ENV] = previous_artifacts
                if previous_provider_fallback is None:
                    os.environ.pop(_SEMANTIER_DISABLE_PROVIDER_FALLBACK_ENV, None)
                else:
                    os.environ[_SEMANTIER_DISABLE_PROVIDER_FALLBACK_ENV] = previous_provider_fallback
        else:
            yield
