"""Desktop window bridges for a local viewer's turn on the session authority.

A GUI session is offered the ``desktop_ui`` tools; on the authority their emitter, GUI callbacks
and process-registry sinks must reach the session's viewers, and be released with the turn.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui, local_viewer_events, session_sudo_prompt
from gateway.session_context import clear_session_vars, set_session_vars
from tools import desktop_ui
from tools.process_registry import process_registry

_ANSWERS = {
    "preview.read": {"value": "page text"},
    "terminal.read": {"value": "tick 1\ntick 2"},
    "window.read": {"value": json.dumps({"app": "Finder"})},
    "preview.act": {"value": json.dumps({"success": True, "acted": "clicked a"})},
    "tour": {"value": json.dumps({"success": True})},
    "theme.apply": {"value": json.dumps({"applied": {"accent": "violet"}})},
    "theme.pick": {"picked": {"accent": "violet"}},
    "setup_beat": {"choice": "next"},
}


class _Fanout:
    """A viewer that answers every request it knows, unless told to stay silent."""

    def __init__(self, requests, silent):
        self.requests, self.silent = requests, silent

    def transports(self):
        return [object()]

    def write(self, frame, **_kw):
        self.requests.append(frame)
        if frame.get("method") in _ANSWERS and frame["method"] not in self.silent:
            session_sudo_prompt.resolve({"id": frame["id"], "result": _ANSWERS[frame["method"]]})
        return True


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_turns", {})
    monkeypatch.setattr(local_desktop_ui, "_preview_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_tour_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_process_routes", {})
    monkeypatch.setattr(local_desktop_ui, "_emitter_installed", False)
    monkeypatch.setattr(local_desktop_ui, "_process_sinks_installed", False)
    monkeypatch.setattr(desktop_ui, "_emit", None)
    monkeypatch.setattr(process_registry, "on_output", None)
    monkeypatch.setattr(process_registry, "on_close", None)

    published, streamed, requests, silent = [], [], [], set()
    stream = SimpleNamespace(
        fanout=_Fanout(requests, silent),
        publish=lambda session_id, payload, *, event_type: streamed.append((session_id, event_type, payload)))
    authority = SimpleNamespace(sessions={"canon": SimpleNamespace(event_stream=stream)})
    turn = SimpleNamespace(
        _ctx=SimpleNamespace(session_id="phys", session_key="agent:main:local:dm:1"),
        _approval_owner=(authority, "canon", 1),
        _publish_execution=lambda kind, payload: published.append((kind, payload)) or True)
    agent = SimpleNamespace(_interrupt_requested=False)
    tokens = set_session_vars(session_id="phys")
    yield SimpleNamespace(turn=turn, agent=agent, published=published, streamed=streamed,
                          requests=requests, silent=silent)
    clear_session_vars(tokens)


def _emit(event, payload):
    return desktop_ui.emit_or_error(event, payload, "failed: ", "desktop only", {"success": True})


def test_window_events_reach_the_turns_stream(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    assert json.loads(_emit("preview.open", {"url": "https://example.com"})) == {"success": True}
    assert window.published[-1] == ("preview.open", {"url": "https://example.com"})
    local_viewer_events.restore(window.agent, sinks)
    assert "failed: " in _emit("preview.open", {"url": "https://example.com"})


def test_gui_reads_are_server_requests_to_the_session_viewers(window):
    local_viewer_events.wire(window.turn, window.agent)
    assert window.agent.read_preview_callback() == "page text"
    assert window.agent.read_terminal_callback(start=0, count=2) == "tick 1\ntick 2"
    assert json.loads(window.agent.read_window_below_callback()) == {"app": "Finder"}
    assert json.loads(window.agent.drive_preview_callback({"action": "click"}))["success"] is True
    terminal = next(f for f in window.requests if f["method"] == "terminal.read")
    assert terminal["params"] == {"session_id": "canon", "start": 0, "count": 2}


def test_set_theme_applies_and_picks(window):
    local_viewer_events.wire(window.turn, window.agent)
    assert window.agent.theme_callback({"accent": "violet"}) == {"applied": {"accent": "violet"}}
    assert window.agent.theme_callback({"picker": "accent"}) == {"success": True, "picked": {"accent": "violet"}}


def test_tour_fails_fast_once_no_window_answered(window, monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_TOUR_PROBE_TIMEOUT_S", 0.3)
    window.silent.add("tour")
    local_viewer_events.wire(window.turn, window.agent)
    assert "No Hermes Desktop window answered the tour" in window.agent.tour_callback({"action": "start"})
    sent = len(window.requests)
    assert "No Hermes Desktop window answered the tour" in window.agent.tour_callback({"action": "start"})
    assert len(window.requests) == sent


def test_background_process_output_follows_its_session_after_the_turn(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    local_viewer_events.restore(window.agent, sinks)
    proc = SimpleNamespace(id="proc_1", session_key="agent:main:local:dm:1")
    process_registry.on_output(proc, "tick 1\n")
    process_registry.on_close(proc, "proc_1")
    assert window.streamed == [
        ("canon", "agent.terminal.output", {"process_id": "proc_1", "chunk": "tick 1\n"}),
        ("canon", "terminal.close", {"process_id": "proc_1"})]


def test_restore_puts_the_agents_callbacks_back(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    local_viewer_events.restore(window.agent, sinks)
    for attr in local_desktop_ui.CALLBACK_ATTRS:
        assert getattr(window.agent, attr) is None
    assert local_desktop_ui._turns == {}



def test_tool_payload_cannot_replace_owning_session(window):
    local_viewer_events.wire(window.turn, window.agent)
    window.agent.drive_preview_callback({"action": "elements", "session_id": "foreign"})
    request = next(f for f in window.requests if f["method"] == "preview.act")
    assert request["params"]["session_id"] == "canon"


def test_released_callback_cannot_reach_a_window(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    stale = window.agent.drive_preview_callback
    local_viewer_events.restore(window.agent, sinks)
    assert json.loads(stale({"action": "click"}))["success"] is False
    assert window.requests == []


def test_stale_generation_cannot_read_or_act(window):
    local_viewer_events.wire(window.turn, window.agent)
    window.turn._approval_owner[0].executing = {"canon": 2}
    assert window.agent.read_preview_callback() == ""
    assert json.loads(window.agent.drive_preview_callback({"action": "click"}))["success"] is False
    assert window.requests == []


def test_old_release_preserves_reconnected_turn(window):
    first = local_viewer_events.wire(window.turn, window.agent)
    newer = SimpleNamespace(**vars(window.turn))
    newer._publish_execution = lambda kind, payload: window.published.append((kind, payload))
    second = local_viewer_events.wire(newer, SimpleNamespace())
    local_desktop_ui.release(first[local_viewer_events._DESKTOP_UI_TOKEN])
    assert json.loads(_emit("preview.open", {"url": "https://example.com"}))["success"]
    local_desktop_ui.release(second[local_viewer_events._DESKTOP_UI_TOKEN])


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [
    {"user_id": "owner", "profile_id": "foreign"},
    {"user_id": "owner", "capabilities": ["session:read"]},
])
async def test_foreign_or_read_only_connection_cannot_answer_preview(tmp_path, monkeypatch, identity):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return "ok"

    authority = await _authority(tmp_path, monkeypatch, answer)
    denied = AuthorityConnection(authority, object(), identity)
    owner = AuthorityConnection(authority, object(), {"user_id": "owner"})
    pending = session_sudo_prompt._Pending("s")
    session_sudo_prompt._open["srq-affinity"] = pending
    frame = {"id": "srq-affinity", "result": {"value": "authenticated pane"}}
    try:
        await denied.dispatch(frame)
        assert not pending.event.is_set()
        await owner.dispatch(frame)
        assert pending.event.is_set() and pending.value == "authenticated pane"
    finally:
        session_sudo_prompt._open.pop("srq-affinity", None)
        await denied.close()
        await owner.close()


def test_missing_preview_bridge_reports_failure_without_another_browser(window, monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_PREVIEW_ACTION_PROBE_TIMEOUT_S", 0.01)
    window.silent.add("preview.act")
    local_viewer_events.wire(window.turn, window.agent)
    answer = json.loads(window.agent.drive_preview_callback({"action": "click"}))
    assert answer["success"] is False and "No Hermes Desktop window answered" in answer["error"]
    sent = len(window.requests)
    assert json.loads(window.agent.drive_preview_callback({"action": "click"}))["success"] is False
    assert len(window.requests) == sent
    # Cooldown expires; the current renderer can answer after reconnect.
    local_desktop_ui._preview_bridge["canon"]["retry_at"] = 0
    window.silent.clear()
    assert json.loads(window.agent.drive_preview_callback({"action": "elements"}))["success"]


def test_session_removed_before_callback_cannot_read_page(window):
    local_viewer_events.wire(window.turn, window.agent)
    window.turn._approval_owner[0].sessions.clear()
    assert window.agent.read_preview_callback() == ""
    assert window.requests == []


def test_generation_change_withdraws_pending_action_and_rejects_late_reply(window, monkeypatch):
    monkeypatch.setattr(session_sudo_prompt, "_POLL_S", 0.01)
    window.silent.add("preview.act")
    authority = window.turn._approval_owner[0]
    authority.executing = {"canon": 1}
    local_viewer_events.wire(window.turn, window.agent)
    result = []
    worker = threading.Thread(target=lambda: result.append(window.agent.drive_preview_callback({"action": "click"})))
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while not window.requests:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        request = window.requests[0]
        authority.executing["canon"] = 2
        worker.join(2)
        assert not worker.is_alive()
        assert json.loads(result[0])["success"] is False
        assert any(kind == "request.cancel" for _, kind, _ in window.streamed)
        assert not session_sudo_prompt.resolve({"id": request["id"], "result": {"value": "late"}})
    finally:
        window.agent._interrupt_requested = True
        worker.join(2)


def test_appserver_owned_bridge_retains_actions_without_native_prelude(window):
    window.agent.platform = "desktop"
    window.agent.api_mode = "codex_app_server"
    sinks = local_viewer_events.wire(window.turn, window.agent)
    try:
        result = json.loads(window.agent.drive_preview_callback({"action": "click"}))
        assert result["success"] is True
        assert getattr(window.agent, "_desktop_browser_turn", None) is None
        window.turn._approval_owner[0].sessions.clear()
        assert json.loads(window.agent.drive_preview_callback({"action": "click"}))["success"] is False
    finally:
        local_viewer_events.restore(window.agent, sinks)


def test_native_desktop_turn_still_installs_browser_binding(window):
    from agent.desktop_browser_target import BrowserTurnState

    window.agent.platform = "desktop"
    window.agent.api_mode = "chat_completions"
    sinks = local_viewer_events.wire(window.turn, window.agent)
    try:
        assert isinstance(window.agent._desktop_browser_turn, BrowserTurnState)
        result = json.loads(window.agent.drive_preview_callback({"action": "click"}))
        assert result["code"] == "TARGET_UNRESOLVED"
    finally:
        local_viewer_events.restore(window.agent, sinks)


def test_nonbrowser_turn_preserves_unbound_preview_callback(window):
    window.agent.platform = "desktop"
    window.agent.api_mode = "chat_completions"
    sinks = local_viewer_events.wire(window.turn, window.agent)
    try:
        window.agent._desktop_browser_turn.select("NONE")
        # Nonbrowser turns may preview generated files through the legacy API.
        assert window.agent.read_preview_callback() == "page text"
        window.turn._approval_owner[0].sessions.clear()
        assert window.agent.read_preview_callback() == ""
    finally:
        local_viewer_events.restore(window.agent, sinks)
