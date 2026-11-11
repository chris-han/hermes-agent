"""Browser callbacks receive the same canonical validated arguments via direct/deferred routes."""
import json
from types import SimpleNamespace

import pytest

from tools.drive_preview_tool import ACT_PREVIEW_SCHEMA
from tools.registry import registry
from tools import tool_search


@pytest.fixture
def scope(monkeypatch):
    definitions = [{'type': 'function', 'function': ACT_PREVIEW_SCHEMA}]
    monkeypatch.setattr(tool_search, 'load_config_readonly', lambda: SimpleNamespace(effective_defer_tools=frozenset({'drive_preview'})))
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **_: definitions)
    return SimpleNamespace(enabled_toolsets=['desktop_ui'], disabled_toolsets=[])


@pytest.mark.parametrize(('args', 'code'), [
    ({'action': 'fill', 'selector': '#field', 'text': 'Ready'}, 'UNKNOWN_ACTION'),
    ({'action': 'click'}, 'MISSING_REQUIRED_FIELD'),
    ({'action': 'type', 'selector': '#field'}, 'MISSING_REQUIRED_FIELD'),
    ({'action': 'click', 'ref': {'selector': '#field'}}, 'INVALID_ARGUMENT'),
    ({'action': 'elements', 'max': '1.5'}, 'INVALID_ARGUMENT'),
    ({'action': 'elements', 'max': '9007199254740993'}, 'INVALID_ARGUMENT'),
])
def test_direct_and_deferred_reject_identically_before_callback(scope, args, code):
    from agent.tool_executor import _unwrap_tool_search_call
    called = []
    direct = json.loads(registry.dispatch('drive_preview', args, callback=lambda _: called.append(True) or '{"success":true}'))
    _, _, blocked = _unwrap_tool_search_call(scope, 'tool_call', {'calls': [{'name': 'drive_preview', 'arguments': args}]})
    deferred = json.loads(blocked)
    assert direct['code'] == deferred['code'] == code
    assert direct['path'] and direct['parameters'] == ACT_PREVIEW_SCHEMA['parameters']
    assert not called


def test_valid_direct_deferred_and_lossless_integer_string(scope):
    from agent.tool_executor import _unwrap_tool_search_call
    for args in [{'action': 'click', 'selector': '#button'}, {'action': 'elements', 'max': '3'}]:
        name, forwarded, blocked = _unwrap_tool_search_call(scope, 'tool_call', {'calls': json.dumps([{'name': 'drive_preview', 'arguments': args}])})
        assert name == 'drive_preview' and blocked is None
        direct_calls, deferred_calls = [], []
        registry.dispatch(name, args, callback=lambda a: direct_calls.append(a) or '{"success":true}')
        registry.dispatch(name, forwarded, callback=lambda a: deferred_calls.append(a) or '{"success":true}')
        assert direct_calls == deferred_calls and len(direct_calls) == 1


def test_ambiguous_envelope_is_structured_and_does_not_invent_name(scope):
    from model_tools import _dispatch_bridge_tool
    for raw in ['\\n[{\\"arguments\\":{\\"action\\":\\"elements\\",\\"name\\":\\"drive_preview\\"}}]\\n', json.dumps([{'arguments': {'name': 'drive_preview', 'action': 'elements'}}])]:
        result, forwarded = _dispatch_bridge_tool('tool_call', {'calls': raw}, ['desktop_ui'], [])
        error = json.loads(result)
        assert forwarded is None and error['code'] in ('INVALID_ARGUMENT', 'MISSING_REQUIRED_FIELD')
        assert error['path'] and error['recovery']['same_context']


def test_bound_target_cannot_be_changed_by_new_browser_session_argument():
    from agent.desktop_browser_target import BrowserTurnState, browser_turn_scope
    from tools.browser_use_cli import browser_exec
    state = BrowserTurnState(session_id='owner', generation=1, turn_id='t', target_kind='VISIBLE_PANE')
    before = state.binding
    with browser_turn_scope(state):
        for session in ['owned', 'foreign', 'new-session']:
            result = json.loads(browser_exec('from browser_helpers import js', session=session))
            assert result['code'] == 'TARGET_MISMATCH'
            assert result['recovery']['same_context']
    assert state.binding == before and state.native_failure is None


def test_missing_validator_fails_closed_for_browser_only(monkeypatch):
    import sys
    called = []
    monkeypatch.setitem(sys.modules, 'jsonschema.validators', None)
    result = json.loads(registry.dispatch('drive_preview', {'action': 'click', 'selector': '#button'}, callback=lambda _: called.append(True)))
    assert result['code'] == 'ARGUMENT_CONTRACT_UNAVAILABLE' and not called


def test_bound_target_rejects_registered_legacy_browser_backends(monkeypatch):
    from agent.desktop_browser_target import BrowserTurnState
    original = registry.get_entry
    monkeypatch.setattr(registry, 'get_entry', lambda name, **kw: SimpleNamespace(toolset='browser') if name == 'fixture_independent_backend' else original(name, **kw))
    state = BrowserTurnState(session_id='owner', generation=1, turn_id='t', target_kind='VISIBLE_PANE')
    assert state.guard('fixture_independent_backend', {})['code'] == 'TARGET_MISMATCH'


def test_bound_alias_canonicalization_matches_agent_and_native_bridge(scope, monkeypatch):
    from agent.desktop_browser_target import BrowserTurnState, browser_turn_scope
    from agent.tool_executor import _unwrap_tool_search_call
    from model_tools import _dispatch_bridge_tool
    from tools.preview_tool import PREVIEW_SCHEMA
    definitions = [{'type': 'function', 'function': schema} for schema in [ACT_PREVIEW_SCHEMA, PREVIEW_SCHEMA]]
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **_: definitions)
    monkeypatch.setattr(tool_search, 'load_config_readonly', lambda: SimpleNamespace(effective_defer_tools=frozenset({'drive_preview', 'desktop_preview'})))
    state = BrowserTurnState(session_id='owner', generation=1, turn_id='t', target_kind='VISIBLE_PANE')
    args = {'action': 'read'}
    wrapped = {'calls': [{'name': 'drive_preview', 'arguments': args}]}
    with browser_turn_scope(state):
        name, forwarded, error = _unwrap_tool_search_call(scope, 'tool_call', wrapped)
        result, native = _dispatch_bridge_tool('tool_call', wrapped, ['desktop_ui'], [])
    assert (name, forwarded, error) == ('desktop_preview', args, None)
    assert result is None and native == ('desktop_preview', args)


@pytest.mark.parametrize('args', [{'action': 'open'}, {'action': 'open', 'url': ' '}])
def test_open_requires_an_explicit_nonempty_target_before_callback(args):
    called = []
    result = json.loads(registry.dispatch('desktop_preview', args, callback=lambda _: called.append(True)))
    assert result['code'] in ('MISSING_REQUIRED_FIELD', 'INVALID_ARGUMENT')
    assert not called



def test_registry_boundary_rejects_independent_backend_before_handler(monkeypatch):
    from agent.desktop_browser_target import BrowserTurnState, browser_turn_scope
    state = BrowserTurnState(session_id='owner', generation=1, turn_id='t', target_kind='VISIBLE_PANE')
    calls = []
    original = registry.get_entry
    entry = SimpleNamespace(toolset='browser', schema={'name':'fixture_independent_backend','parameters':{'type':'object'}},
                            is_async=False, handler=lambda args: calls.append(args) or '{"success":true}')
    monkeypatch.setattr(registry, 'get_entry', lambda name, **kw: entry if name == 'fixture_independent_backend' else original(name, **kw))
    with browser_turn_scope(state):
        for _ in range(2):
            result = json.loads(registry.dispatch('fixture_independent_backend', {}))
            assert result['code'] == 'TARGET_MISMATCH'
    assert not calls


@pytest.mark.parametrize('args', [None, [], 'not a JSON object'])
def test_non_object_arguments_get_actionable_validation_feedback(args):
    called = []
    result = json.loads(registry.dispatch('drive_preview', args, callback=lambda _: called.append(True)))
    assert result['code'] == 'INVALID_ARGUMENT'
    assert result['path'] == 'arguments' and result['recovery']['same_context']
    assert not called
