"""Tests for task-scoped CWD sandboxing in terminal and file I/O tools.

These tests guard against the regression where a generated script (e.g.
pltr_data.py) called `cd /repo/agent && python /abs/script.py`, causing
`open('output.json', 'w')` inside the script to land at the process cwd
rather than the session's run_dir.

The fix: `terminal_tool` now always passes the registered task cwd to
`env.execute()` on every foreground call, not just when `workdir` is
explicitly provided.  This prevents `env.self.cwd` drift from persisting
across tool calls when a command uses an internal `cd` to escape the anchor.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure project root is on path (mirrors conftest.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_config(cwd: str = "/tmp", env_type: str = "local", **kwargs):
    """Minimal _get_env_config()-shaped dict, overridable."""
    base = {
        "env_type": env_type,
        "timeout": 30,
        "cwd": cwd,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "local_persistent": False,
    }
    base.update(kwargs)
    return base


def _mock_env(output: str = "ok", returncode: int = 0) -> MagicMock:
    """Return a mock environment whose execute() returns a successful result."""
    env = MagicMock()
    env.execute.return_value = {"output": output, "returncode": returncode}
    env.env = {}
    return env


def _run_terminal(task_id: str, command: str, mock_env: MagicMock, config: dict,
                  workdir: str | None = None, **extra):
    """Call terminal_tool with a pre-wired mock environment and config."""
    from tools.terminal_tool import terminal_tool

    active = {task_id: mock_env}
    last_activity = {task_id: 0.0}

    with patch("tools.terminal_tool._get_env_config", return_value=config), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._active_environments", active), \
         patch("tools.terminal_tool._last_activity", last_activity), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        return json.loads(terminal_tool(
            command=command,
            task_id=task_id,
            workdir=workdir,
            **extra,
        ))


# ===========================================================================
# 1. Registered task CWD is always passed to env.execute()
# ===========================================================================

class TestRegisteredCwdAnchor:
    """terminal_tool uses the task override cwd on every call."""

    def test_registered_cwd_used_when_no_workdir(self, tmp_path):
        """The registered task cwd is passed even when workdir is not supplied."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        run_dir = str(tmp_path / "artifacts")
        task_id = "test-task-anchor"
        register_task_env_overrides(task_id, {"cwd": run_dir})

        env = _mock_env()
        try:
            _run_terminal(
                task_id=task_id,
                command="echo hello",
                mock_env=env,
                config=_make_env_config(cwd=run_dir),
            )
        finally:
            clear_task_env_overrides(task_id)

        # env.execute must have been called with cwd=run_dir
        env.execute.assert_called_once()
        call_kwargs = env.execute.call_args[1]
        assert call_kwargs.get("cwd") == run_dir

    def test_global_config_cwd_used_when_no_override(self, tmp_path):
        """When no task override is set, global config cwd is used."""
        from tools.terminal_tool import _task_env_overrides

        global_cwd = str(tmp_path / "global")
        task_id = "test-task-global"
        # Ensure no override is registered
        _task_env_overrides.pop(task_id, None)

        env = _mock_env()
        _run_terminal(
            task_id=task_id,
            command="echo hello",
            mock_env=env,
            config=_make_env_config(cwd=global_cwd),
        )

        env.execute.assert_called_once()
        call_kwargs = env.execute.call_args[1]
        assert call_kwargs.get("cwd") == global_cwd

    def test_explicit_workdir_overrides_registered_cwd(self, tmp_path):
        """When LLM passes explicit workdir, it wins over the registered anchor."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        run_dir = str(tmp_path / "artifacts")
        explicit_workdir = str(tmp_path / "explicit")
        task_id = "test-task-explicit-workdir"
        register_task_env_overrides(task_id, {"cwd": run_dir})

        env = _mock_env()
        try:
            _run_terminal(
                task_id=task_id,
                command="echo hello",
                mock_env=env,
                config=_make_env_config(cwd=run_dir),
                workdir=explicit_workdir,
            )
        finally:
            clear_task_env_overrides(task_id)

        call_kwargs = env.execute.call_args[1]
        assert call_kwargs.get("cwd") == explicit_workdir

    def test_explicit_cd_outside_safe_root_is_blocked(self, tmp_path):
        """A command cannot `cd` outside the registered task safe root."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        artifact_dir = tmp_path / "agent" / "sessions" / "abc123" / "runs" / "run1" / "artifacts"
        artifact_dir.mkdir(parents=True)
        escaped_dir = tmp_path / "agent"
        task_id = "test-task-cd-escape"
        register_task_env_overrides(task_id, {
            "cwd": str(artifact_dir),
            "safe_write_root": str(artifact_dir),
        })

        env = _mock_env()
        try:
            result = _run_terminal(
                task_id=task_id,
                command=f"cd {escaped_dir} && ./.venv/bin/python script.py",
                mock_env=env,
                config=_make_env_config(cwd=str(artifact_dir)),
            )
        finally:
            clear_task_env_overrides(task_id)

        assert result["status"] == "blocked"
        assert "task sandbox" in result["error"]
        env.execute.assert_not_called()

    def test_explicit_absolute_read_outside_safe_read_root_is_blocked(self, tmp_path):
        """A command cannot reference an absolute path outside the registered read root."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        workspace_dir = tmp_path / "agent" / "workspace"
        artifacts_dir = tmp_path / "agent" / "sessions" / "abc123" / "runs" / "run1" / "artifacts"
        workspace_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        task_id = "test-task-read-escape"
        register_task_env_overrides(task_id, {
            "cwd": str(artifacts_dir),
            "safe_read_root": str(workspace_dir),
            "safe_write_root": str(artifacts_dir),
        })

        env = _mock_env()
        try:
            result = _run_terminal(
                task_id=task_id,
                command="ls -la /mnt/c/Users/test/Desktop/report.pdf 2>/dev/null",
                mock_env=env,
                config=_make_env_config(cwd=str(artifacts_dir)),
            )
        finally:
            clear_task_env_overrides(task_id)

        assert result["status"] == "blocked"
        assert "outside the task sandbox" in result["error"]
        env.execute.assert_not_called()

    def test_cwd_passed_on_every_call(self, tmp_path):
        """Registered cwd is re-passed on each consecutive terminal call."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        run_dir = str(tmp_path / "artifacts")
        task_id = "test-task-multi-call"
        register_task_env_overrides(task_id, {"cwd": run_dir})

        env = _mock_env()
        active = {task_id: env}
        last_activity = {task_id: 0.0}
        config = _make_env_config(cwd=run_dir)

        from tools.terminal_tool import terminal_tool

        try:
            with patch("tools.terminal_tool._get_env_config", return_value=config), \
                 patch("tools.terminal_tool._start_cleanup_thread"), \
                 patch("tools.terminal_tool._active_environments", active), \
                 patch("tools.terminal_tool._last_activity", last_activity), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                terminal_tool(command="echo first", task_id=task_id)
                terminal_tool(command="echo second", task_id=task_id)
                terminal_tool(command="echo third", task_id=task_id)
        finally:
            clear_task_env_overrides(task_id)

        assert env.execute.call_count == 3
        for c in env.execute.call_args_list:
            assert c[1].get("cwd") == run_dir, (
                f"Expected cwd={run_dir!r} in every call, got {c[1].get('cwd')!r}"
            )


# ===========================================================================
# 2. CWD drift prevention (the pltr_ohlcv regression)
# ===========================================================================

class TestCwdDriftPrevention:
    """Even if a command `cd`s away internally, subsequent calls stay anchored."""

    def test_env_cwd_drift_does_not_affect_next_call(self, tmp_path):
        """Simulates: first call does `cd /other && python script.py`,
        env.cwd mutates; second call must still use the registered anchor."""
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        run_dir = str(tmp_path / "artifacts")
        other_dir = "/var/tmp"  # where a cd-escape would land
        task_id = "test-drift-prevention"
        register_task_env_overrides(task_id, {"cwd": run_dir})

        env = _mock_env()
        active = {task_id: env}
        last_activity = {task_id: 0.0}
        config = _make_env_config(cwd=run_dir)

        from tools.terminal_tool import terminal_tool

        def _execute_with_cwd_leak(command, **kwargs):
            """Simulates env.cwd being updated after a cd-escape command."""
            env.cwd = other_dir  # mimic LocalEnvironment._update_cwd after cd
            return {"output": "ok", "returncode": 0}

        env.execute.side_effect = _execute_with_cwd_leak

        try:
            with patch("tools.terminal_tool._get_env_config", return_value=config), \
                 patch("tools.terminal_tool._start_cleanup_thread"), \
                 patch("tools.terminal_tool._active_environments", active), \
                 patch("tools.terminal_tool._last_activity", last_activity), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                # First call — simulates cd-escape inside the command
                terminal_tool(command=f"cd {other_dir} && python /abs/script.py", task_id=task_id)
                # After this, env.cwd == other_dir (drift)
                assert env.cwd == other_dir

                # Second call — must use the registered anchor, not the drifted cwd
                terminal_tool(command="echo after_drift", task_id=task_id)
        finally:
            clear_task_env_overrides(task_id)

        assert env.execute.call_count == 2
        # Both calls should have passed the registered run_dir as cwd
        for c in env.execute.call_args_list:
            assert c[1].get("cwd") == run_dir, (
                f"CWD drift leaked into call: expected {run_dir!r}, got {c[1].get('cwd')!r}"
            )

    def test_no_registered_override_uses_config_cwd_not_env_drift(self, tmp_path):
        """Without an override, config cwd (not env.cwd drift) is the anchor."""
        from tools.terminal_tool import _task_env_overrides

        global_cwd = str(tmp_path / "global")
        task_id = "test-no-override-drift"
        _task_env_overrides.pop(task_id, None)

        env = _mock_env()
        env.cwd = "/drifted"  # simulate prior drift

        active = {task_id: env}
        last_activity = {task_id: 0.0}
        config = _make_env_config(cwd=global_cwd)

        _run_terminal(
            task_id=task_id,
            command="echo test",
            mock_env=env,
            config=config,
        )

        env.execute.assert_called_once()
        assert env.execute.call_args[1].get("cwd") == global_cwd


# ===========================================================================
# 3. write_file respects registered task cwd anchor
# ===========================================================================

class TestWriteFileCwdAnchor:
    """write_file_tool writes into the task-scoped environment, not the process cwd."""

    @patch("tools.file_tools._get_file_ops")
    def test_write_file_uses_task_file_ops(self, mock_get_file_ops, tmp_path):
        """write_file_tool delegates to _get_file_ops(task_id), not a global one."""
        from tools.file_tools import write_file_tool

        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"bytes_written": 42, "dirs_created": False}
        mock_ops.write_file.return_value = result_obj
        mock_get_file_ops.return_value = mock_ops

        target = str(tmp_path / "output.json")
        result = json.loads(write_file_tool(target, '{"key": "value"}', task_id="my-task"))

        assert result["bytes_written"] == 42
        mock_get_file_ops.assert_called_once_with("my-task")
        mock_ops.write_file.assert_called_once_with(target, '{"key": "value"}')

    @patch("tools.file_tools._get_file_ops")
    def test_write_file_sensitive_path_blocked(self, mock_get_file_ops):
        """write_file_tool rejects writes to sensitive system paths."""
        from tools.file_tools import write_file_tool

        result = json.loads(write_file_tool("/etc/passwd", "evil"))

        mock_get_file_ops.assert_not_called()
        assert "error" in result
        assert "sensitive" in result["error"].lower() or "refusing" in result["error"].lower()

    def test_write_file_rejects_etc_passwd(self):
        """write_file_tool blocks writes to /etc/* via _check_sensitive_path."""
        from tools.file_tools import write_file_tool

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(write_file_tool("/etc/passwd", "evil"))

        mock_get.assert_not_called()
        assert "error" in result

    def test_is_write_denied_blocks_ssh_key(self):
        """_is_write_denied (ShellFileOperations layer) blocks ~/.ssh/id_rsa."""
        from tools.file_operations import _is_write_denied

        path = os.path.expanduser("~/.ssh/id_rsa")
        assert _is_write_denied(path) is True

    @patch("tools.file_tools._get_file_ops")
    def test_write_file_absolute_run_dir_path_allowed(self, mock_get_file_ops, tmp_path):
        """Absolute paths inside the run_dir are allowed through."""
        from tools.file_tools import write_file_tool

        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"bytes_written": 10, "dirs_created": False}
        mock_ops.write_file.return_value = result_obj
        mock_get_file_ops.return_value = mock_ops

        target = str(tmp_path / "runs" / "20260413_214458" / "pltr_data.py")
        result = json.loads(write_file_tool(target, "import yfinance"))

        assert result["bytes_written"] == 10
        mock_ops.write_file.assert_called_once_with(target, "import yfinance")


# ===========================================================================
# 4. register / clear task env overrides mechanics
# ===========================================================================

class TestRegisterTaskEnvOverrides:
    """register_task_env_overrides and clear_task_env_overrides contract."""

    def test_register_sets_cwd_override(self, tmp_path):
        from tools.terminal_tool import register_task_env_overrides, _task_env_overrides

        run_dir = str(tmp_path / "run")
        register_task_env_overrides("task-A", {"cwd": run_dir})

        assert _task_env_overrides["task-A"]["cwd"] == run_dir

    def test_clear_removes_override(self, tmp_path):
        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
            _task_env_overrides,
        )

        register_task_env_overrides("task-B", {"cwd": str(tmp_path)})
        clear_task_env_overrides("task-B")

        assert "task-B" not in _task_env_overrides

    def test_clear_nonexistent_task_is_safe(self):
        from tools.terminal_tool import clear_task_env_overrides

        # Should not raise
        clear_task_env_overrides("no-such-task-xyz")

    def test_different_tasks_have_independent_cwds(self, tmp_path):
        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
            _task_env_overrides,
        )

        dir_a = str(tmp_path / "task_a")
        dir_b = str(tmp_path / "task_b")

        register_task_env_overrides("task-X", {"cwd": dir_a})
        register_task_env_overrides("task-Y", {"cwd": dir_b})

        try:
            assert _task_env_overrides["task-X"]["cwd"] == dir_a
            assert _task_env_overrides["task-Y"]["cwd"] == dir_b
        finally:
            clear_task_env_overrides("task-X")
            clear_task_env_overrides("task-Y")

    def test_register_override_replaces_previous(self, tmp_path):
        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
            _task_env_overrides,
        )

        first = str(tmp_path / "first")
        second = str(tmp_path / "second")

        register_task_env_overrides("task-Z", {"cwd": first})
        register_task_env_overrides("task-Z", {"cwd": second})

        try:
            assert _task_env_overrides["task-Z"]["cwd"] == second
        finally:
            clear_task_env_overrides("task-Z")


# ===========================================================================
# 5. BaseEnvironment._wrap_command cwd injection
# ===========================================================================

class TestWrapCommandCwdInjection:
    """_wrap_command always uses the cwd argument, not self.cwd."""

    def test_wrap_command_uses_provided_cwd(self, tmp_path):
        """The wrapped command starts with `cd <provided_cwd>`."""
        from tools.environments.base import BaseEnvironment

        class _ConcreteEnv(BaseEnvironment):
            def _run_bash(self, cmd, *, login=False, timeout=120, stdin_data=None):
                pass

            def execute(self, command, **kwargs):
                pass

            def _kill_process(self, proc):
                pass

            def _update_cwd(self, result):
                pass

            def cleanup(self):
                pass

        env = _ConcreteEnv.__new__(_ConcreteEnv)
        env.cwd = "/drifted/path"
        env.timeout = 30
        env._snapshot_ready = False
        import uuid, tempfile
        env._session_id = uuid.uuid4().hex
        td = tempfile.gettempdir()
        env._snapshot_path = f"{td}/hermes-snap-{env._session_id}.sh"
        env._cwd_file = f"{td}/hermes-cwd-{env._session_id}.txt"
        env._cwd_marker = f"__HERMES_CWD_{env._session_id}__"

        provided_cwd = "/expected/run/dir"
        wrapped = env._wrap_command("echo test", provided_cwd)

        # The wrapped script must cd to the provided cwd, not self.cwd
        assert f"cd {provided_cwd}" in wrapped or f"cd '{provided_cwd}'" in wrapped
        assert "/drifted/path" not in wrapped

    def test_wrap_command_cwd_not_same_as_self_cwd_when_drifted(self, tmp_path):
        """When self.cwd has drifted, passing a different cwd still wins."""
        from tools.environments.base import BaseEnvironment

        class _ConcreteEnv(BaseEnvironment):
            def _run_bash(self, cmd, *, login=False, timeout=120, stdin_data=None):
                pass

            def execute(self, command, **kwargs):
                pass

            def _kill_process(self, proc):
                pass

            def _update_cwd(self, result):
                pass

            def cleanup(self):
                pass

        env = _ConcreteEnv.__new__(_ConcreteEnv)
        env.cwd = "/var/tmp/leaked"
        env.timeout = 30
        env._snapshot_ready = False
        import uuid, tempfile
        env._session_id = uuid.uuid4().hex
        td = tempfile.gettempdir()
        env._snapshot_path = f"{td}/hermes-snap-{env._session_id}.sh"
        env._cwd_file = f"{td}/hermes-cwd-{env._session_id}.txt"
        env._cwd_marker = f"__HERMES_CWD_{env._session_id}__"

        anchor = "/home/user/sessions/run_dir/artifacts"
        wrapped = env._wrap_command("python script.py", anchor)

        assert "/var/tmp/leaked" not in wrapped
        assert anchor in wrapped


# ===========================================================================
# 6. LocalEnvironment execute() cwd fallback chain
# ===========================================================================

class TestLocalEnvironmentExecuteCwdFallback:
    """BaseEnvironment.execute() uses cwd arg or self.cwd — never None."""

    def test_execute_passes_cwd_arg_to_wrap_command(self, tmp_path):
        """execute(command, cwd='/explicit') uses the explicit arg."""
        from tools.environments.base import BaseEnvironment

        captured = {}

        class _TracingEnv(BaseEnvironment):
            def _run_bash(self, cmd, *, login=False, timeout=120, stdin_data=None):
                captured["wrapped"] = cmd
                proc = MagicMock()
                proc.stdout = iter([])
                proc.poll.return_value = 0
                return proc

            def _kill_process(self, proc):
                pass

            def _update_cwd(self, result):
                pass

            def cleanup(self):
                pass

        import uuid, tempfile
        session_id = uuid.uuid4().hex
        td = tempfile.gettempdir()

        env = _TracingEnv.__new__(_TracingEnv)
        env.timeout = 30
        env.cwd = "/self/cwd/drifted"
        env._snapshot_ready = False
        env._session_id = session_id
        env._snapshot_path = f"{td}/hermes-snap-{session_id}.sh"
        env._cwd_file = f"{td}/hermes-cwd-{session_id}.txt"
        env._cwd_marker = f"__HERMES_CWD_{session_id}__"
        env.env = {}
        env._stdin_mode = "pipe"

        explicit_cwd = str(tmp_path / "explicit")
        env.execute("echo test", cwd=explicit_cwd)

        assert explicit_cwd in captured["wrapped"]
        assert "/self/cwd/drifted" not in captured["wrapped"]

    def test_execute_falls_back_to_self_cwd_when_no_cwd_arg(self, tmp_path):
        """execute(command) with no cwd falls back to self.cwd."""
        from tools.environments.base import BaseEnvironment

        captured = {}

        class _TracingEnv(BaseEnvironment):
            def _run_bash(self, cmd, *, login=False, timeout=120, stdin_data=None):
                captured["wrapped"] = cmd
                proc = MagicMock()
                proc.stdout = iter([])
                proc.poll.return_value = 0
                return proc

            def _kill_process(self, proc):
                pass

            def _update_cwd(self, result):
                pass

            def cleanup(self):
                pass

        import uuid, tempfile
        session_id = uuid.uuid4().hex
        td = tempfile.gettempdir()

        self_cwd = str(tmp_path / "self_cwd")
        env = _TracingEnv.__new__(_TracingEnv)
        env.timeout = 30
        env.cwd = self_cwd
        env._snapshot_ready = False
        env._session_id = session_id
        env._snapshot_path = f"{td}/hermes-snap-{session_id}.sh"
        env._cwd_file = f"{td}/hermes-cwd-{session_id}.txt"
        env._cwd_marker = f"__HERMES_CWD_{session_id}__"
        env.env = {}
        env._stdin_mode = "pipe"

        env.execute("echo test")

        assert self_cwd in captured["wrapped"]


# ===========================================================================
# 7. HERMES_WRITE_SAFE_ROOT sandbox integration
# ===========================================================================

class TestWriteSafeRootIntegration:
    """HERMES_WRITE_SAFE_ROOT + task-scoped cwd work together."""

    def test_run_dir_inside_safe_root_allowed(self, tmp_path, monkeypatch):
        """When run_dir is under HERMES_WRITE_SAFE_ROOT, writes are allowed."""
        from tools.file_operations import _is_write_denied

        safe_root = tmp_path / "agent"
        run_dir = safe_root / "sessions" / "abc123" / "runs" / "20260413"
        run_dir.mkdir(parents=True)
        target = str(run_dir / "output.json")

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(target) is False

    def test_run_dir_outside_safe_root_blocked(self, tmp_path, monkeypatch):
        """Writes outside HERMES_WRITE_SAFE_ROOT are denied."""
        from tools.file_operations import _is_write_denied

        safe_root = tmp_path / "agent"
        safe_root.mkdir()
        outside = tmp_path / "other_project" / "output.json"

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(outside)) is True

    def test_process_cwd_escape_blocked_when_safe_root_set(self, tmp_path, monkeypatch):
        """Writing to the process cwd (when it's outside safe_root) is denied."""
        from tools.file_operations import _is_write_denied

        safe_root = tmp_path / "agent" / "sessions"
        safe_root.mkdir(parents=True)
        # Simulate writing a bare filename that would land in agent/ (process cwd)
        process_cwd_file = str(tmp_path / "agent" / "pltr_ohlcv.json")

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        # agent/pltr_ohlcv.json is under agent/ but NOT under agent/sessions/
        assert _is_write_denied(process_cwd_file) is True

    def test_symlink_traversal_blocked(self, tmp_path, monkeypatch):
        """Symlinks that escape the safe root are blocked."""
        from tools.file_operations import _is_write_denied

        safe_root = tmp_path / "workspace"
        safe_root.mkdir()

        # Symlink inside workspace pointing outside
        link = safe_root / "escape_link"
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        link.symlink_to(outside_dir)

        target = str(link / "evil.txt")  # resolves to outside/evil.txt
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(target) is True

    def test_file_tool_uses_task_safe_root_override(self, tmp_path):
        """Per-task safe_write_root is propagated into ShellFileOperations."""
        from tools.file_tools import _get_file_ops, clear_file_ops_cache
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        task_id = "test-task-file-safe-root"
        artifact_dir = tmp_path / "agent" / "sessions" / "abc123" / "runs" / "run1" / "artifacts"
        artifact_dir.mkdir(parents=True)
        register_task_env_overrides(task_id, {
            "cwd": str(artifact_dir),
            "safe_write_root": str(artifact_dir),
        })

        env = _mock_env()
        active = {task_id: env}
        last_activity = {task_id: 0.0}

        try:
            with patch("tools.terminal_tool._active_environments", active), \
                 patch("tools.terminal_tool._last_activity", last_activity), \
                 patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(artifact_dir))):
                file_ops = _get_file_ops(task_id)
        finally:
            clear_file_ops_cache(task_id)
            clear_task_env_overrides(task_id)

        assert file_ops.safe_write_root == str(artifact_dir)

    def test_file_tool_uses_task_safe_read_root_override(self, tmp_path):
        """Per-task safe_read_root is propagated into ShellFileOperations."""
        from tools.file_tools import _get_file_ops, clear_file_ops_cache
        from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

        task_id = "test-task-file-read-root"
        workspace_dir = tmp_path / "agent" / "workspace"
        artifact_dir = tmp_path / "agent" / "sessions" / "abc123" / "runs" / "run1" / "artifacts"
        workspace_dir.mkdir(parents=True)
        artifact_dir.mkdir(parents=True)
        register_task_env_overrides(task_id, {
            "cwd": str(artifact_dir),
            "safe_read_root": str(workspace_dir),
            "safe_write_root": str(artifact_dir),
        })

        env = _mock_env()
        active = {task_id: env}
        last_activity = {task_id: 0.0}

        try:
            with patch("tools.terminal_tool._active_environments", active), \
                 patch("tools.terminal_tool._last_activity", last_activity), \
                 patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(artifact_dir))):
                file_ops = _get_file_ops(task_id)
        finally:
            clear_file_ops_cache(task_id)
            clear_task_env_overrides(task_id)

        assert file_ops.safe_read_root == str(workspace_dir)

    def test_file_read_outside_safe_read_root_is_blocked(self, tmp_path):
        """read_file refuses paths outside the registered read root."""
        env = _mock_env()
        safe_read_root = tmp_path / "workspace"
        safe_read_root.mkdir()
        ops = __import__("tools.file_operations", fromlist=["ShellFileOperations"]).ShellFileOperations(
            env,
            cwd=str(safe_read_root),
            safe_read_root=str(safe_read_root),
        )

        result = ops.read_file("/mnt/c/Users/test/Desktop/report.pdf")

        assert result.error is not None
        assert "outside the allowed read root" in result.error
        env.execute.assert_not_called()
