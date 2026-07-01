"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written to the active
   governed execution boundary's artifact root when one is present. Standalone
   Hermes runs without a governed boundary keep the historical sandbox temp-dir
   behavior. The in-context content is replaced with a preview + file path
   reference. The model can read_file to access the full output on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import logging
import os
from pathlib import Path
import shlex
import uuid

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_GOVERNED_ARTIFACTS_ENV = "SEMANTIER_WORKSPACE_ARTIFACTS_DIR"


class GovernedToolResultStorageError(RuntimeError):
    """Raised when governed tool-result persistence cannot stay in-boundary."""


def _current_boundary():
    try:
        from gateway.execution_boundary import current_execution_boundary
    except Exception:
        return None
    return current_execution_boundary()


def _is_governed_storage_required() -> bool:
    boundary = _current_boundary()
    if boundary is not None:
        return True
    return bool(os.environ.get(_GOVERNED_ARTIFACTS_ENV))


def _resolve_governed_artifacts_root() -> str | None:
    boundary = _current_boundary()
    if boundary is not None:
        root = boundary.paths.artifacts_root
        if root is None:
            raise GovernedToolResultStorageError(
                "GOVERNED_TOOL_RESULT_ARTIFACTS_ROOT_REQUIRED: "
                "active execution boundary has no artifacts_root"
            )
        return str(Path(root).resolve())

    raw = os.environ.get(_GOVERNED_ARTIFACTS_ENV)
    if raw:
        return str(Path(raw).expanduser().resolve())
    return None


def _is_under_root(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _mirror_path_under_governed_root(path: str, governed_root: str) -> str:
    """Map an absolute sandbox path beneath the governed artifacts root.

    Example: /tmp/hermes-results -> <artifacts_root>/tmp/hermes-results.
    This preserves the original backend subpath while keeping disk IO inside
    the session artifact boundary.
    """
    source = Path(path)
    if not source.is_absolute():
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_ABSOLUTE_PATH_REQUIRED: "
            f"{path} is not absolute"
        )
    relative_parts = source.parts[1:]
    if not relative_parts:
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_PATH_REQUIRED: cannot mirror filesystem root"
        )
    return str(Path(governed_root, *relative_parts).resolve())


def _resolve_storage_dir(env) -> str:
    """Return the storage dir for persisted tool results.

    Governed Semantier execution must store spill files under the active
    session artifact root. The temp-dir fallback is only for standalone Hermes
    execution without a governed boundary.
    """
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if isinstance(temp_dir, str) and temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    storage_dir = f"{temp_dir}/hermes-results"
                    governed_root = _resolve_governed_artifacts_root()
                    if governed_root is not None:
                        return _mirror_path_under_governed_root(storage_dir, governed_root)
                    return storage_dir

    governed_root = _resolve_governed_artifacts_root()
    if governed_root is not None:
        return _mirror_path_under_governed_root(STORAGE_DIR, governed_root)
    return STORAGE_DIR


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    governed_root = _resolve_governed_artifacts_root()
    if governed_root is not None and not _is_under_root(remote_path, governed_root):
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_PATH_OUT_OF_BOUNDARY: "
            f"{remote_path} is outside {governed_root}"
        )

    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _write_to_governed_root(content: str, remote_path: str) -> bool:
    """Write directly when the target is already inside the governed root."""
    governed_root = _resolve_governed_artifacts_root()
    if governed_root is None:
        return False
    if not _is_under_root(remote_path, governed_root):
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_PATH_OUT_OF_BOUNDARY: "
            f"{remote_path} is outside {governed_root}"
        )
    target = Path(remote_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return True


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Governed Semantier execution must
    write under the active session artifact root or fail closed; standalone
    Hermes execution falls back to inline truncation if persistence is not
    available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{tool_use_id}.txt"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)
    governed_required = _is_governed_storage_required()

    if governed_required and env is None:
        try:
            if _write_to_governed_root(content, remote_path):
                logger.info(
                    "Persisted large governed tool result directly: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except GovernedToolResultStorageError:
            raise
        except Exception as exc:
            raise GovernedToolResultStorageError(
                "GOVERNED_TOOL_RESULT_WRITE_FAILED: "
                f"failed to persist {tool_use_id} under {storage_dir}: {exc}"
            ) from exc
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_ARTIFACTS_ROOT_REQUIRED: "
            "cannot persist governed tool result without artifacts_root"
        )

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            if governed_required:
                raise GovernedToolResultStorageError(
                    "GOVERNED_TOOL_RESULT_WRITE_FAILED: "
                    f"failed to persist {tool_use_id} under {storage_dir}: {exc}"
                ) from exc
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    if governed_required:
        raise GovernedToolResultStorageError(
            "GOVERNED_TOOL_RESULT_WRITE_FAILED: "
            f"failed to persist {tool_use_id} under {storage_dir}"
        )

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages
