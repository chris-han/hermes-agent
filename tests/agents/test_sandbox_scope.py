from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import ModuleType

import pytest

from agents.sandbox_scope import (
    SandboxScope,
    bind_sandbox_scope,
    cron_job_scope,
    cron_job_scope_if_resolvable,
    current_sandbox_key,
    current_sandbox_scope,
    format_sandbox_run_timestamp_utc,
    sandbox_key_for_request,
)


def test_interactive_scope_binds_and_restores_stable_sandbox_key(monkeypatch):
    monkeypatch.delenv("SEMANTIER_SANDBOX_KEY", raising=False)
    scope = SandboxScope(
        workspace_id="ws-123",
        lane="interactive_session",
        scope_id="ws-123:session-abc",
        adapter_key="weixin:ws-123:acct-1",
        network_enabled=False,
    )

    with bind_sandbox_scope(scope):
        assert current_sandbox_scope() == scope
        assert current_sandbox_key() == "ws:ws-123:session:ws-123:session-abc"

    assert current_sandbox_scope() is None
    assert current_sandbox_key() is None


def test_cron_scope_requires_aware_utc_timestamp():
    aware = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    scope = cron_job_scope(
        workspace_id="ws-123",
        job_id="job-42",
        run_timestamp_utc=aware,
    )

    assert sandbox_key_for_request(scope) == "ws:ws-123:cron:job-42:20260830T120000Z"
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        format_sandbox_run_timestamp_utc(datetime(2026, 8, 30, 12, 0))


def test_cron_scope_is_derived_only_from_a_governed_workspace(monkeypatch, tmp_path):
    workspaces_root = tmp_path / "workspaces"
    workspace = workspaces_root / "ws-123"
    governed_home = workspace / ".semantier-home"
    workdir = workspace / "project" / "src"
    governed_home.mkdir(parents=True)
    workdir.mkdir(parents=True)

    runtime_paths = ModuleType("runtime_paths")
    runtime_paths.workspace_root_path = lambda workspace_id: workspaces_root / workspace_id
    monkeypatch.setitem(sys.modules, "runtime_paths", runtime_paths)

    scope = cron_job_scope_if_resolvable(
        workdir=workdir,
        job_id="job-42",
        run_timestamp_utc="20260830T120000Z",
    )

    assert scope is not None
    assert sandbox_key_for_request(scope) == "ws:ws-123:cron:job-42:20260830T120000Z"

    outside = tmp_path / "outside"
    outside.mkdir()
    assert (
        cron_job_scope_if_resolvable(
            workdir=outside,
            job_id="job-42",
            run_timestamp_utc="20260830T120000Z",
        )
        is None
    )


def test_cron_conversation_observes_real_bound_scope(monkeypatch):
    from cron.scheduler import _run_cron_conversation_in_scope

    monkeypatch.delenv("SEMANTIER_SANDBOX_KEY", raising=False)
    observed: dict[str, object] = {}

    class ExternalAgentBoundary:
        def run_conversation(self, prompt: str):
            observed["prompt"] = prompt
            observed["scope"] = current_sandbox_scope()
            observed["key"] = current_sandbox_key()
            return {"final_response": "ok"}

    scope = SandboxScope(
        workspace_id="ws-123",
        lane="cron_job_run",
        scope_id="job-42:20260830T120000Z",
        adapter_key=None,
        network_enabled=False,
    )

    assert _run_cron_conversation_in_scope(ExternalAgentBoundary(), "run", scope) == {
        "final_response": "ok"
    }
    assert observed == {
        "prompt": "run",
        "scope": scope,
        "key": "ws:ws-123:cron:job-42:20260830T120000Z",
    }
    assert current_sandbox_scope() is None
    assert current_sandbox_key() is None
