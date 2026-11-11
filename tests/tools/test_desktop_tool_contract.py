"""Native Desktop schema disclosure, repairable coercion and scoped dispatch parity."""
import json
from types import SimpleNamespace

import pytest
from tools.drive_preview_tool import ACT_PREVIEW_SCHEMA, drive_preview_tool
from tools import tool_search


@pytest.fixture
def scoped_contract(monkeypatch):
    definitions = [{'type': 'function', 'function': ACT_PREVIEW_SCHEMA}]
    monkeypatch.setattr(tool_search, 'load_config_readonly', lambda: SimpleNamespace(effective_defer_tools=frozenset({'drive_preview'})))
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **kwargs: definitions)
    return definitions


def test_describe_preserves_the_complete_native_parameter_schema(scoped_contract):
    described = json.loads(tool_search.dispatch_tool_describe({'names': ['drive_preview']}, current_tool_defs=scoped_contract))
    assert described['tools']['drive_preview']['parameters'] == ACT_PREVIEW_SCHEMA['parameters']
    assert described['tools']['drive_preview']['description'] == ACT_PREVIEW_SCHEMA['description']


def test_valid_first_action_reaches_the_expected_callback():
    calls = []
    result = drive_preview_tool(action='type', selector='#field', text='Ready', callback=lambda args: calls.append(args) or '{"success":true}')
    assert json.loads(result)['success']
    assert calls == [{'action': 'type', 'selector': '#field', 'text': 'Ready'}]


def test_existing_schema_coercion_repairs_only_the_integer_representation():
    args = {'action': 'elements', 'max': '3'}
    assert tool_search.validate_deferred_call_args('drive_preview', args) is None
    assert args == {'action': 'elements', 'max': '3'}
    calls = []
    drive_preview_tool(action=args['action'], limit=args['max'], callback=lambda value: calls.append(value) or '{"success":true}')
    assert calls == [{'action': 'elements', 'max': 3}]


def test_unsupported_action_returns_schema_feedback_without_inventing_arguments():
    args = {'action': 'fill', 'selector': '#field', 'text': 'Ready'}
    result = json.loads(tool_search.validate_deferred_call_args('drive_preview', args))
    assert result['constraint'] == 'enum'
    assert result['parameters'] == ACT_PREVIEW_SCHEMA['parameters']
    assert args == {'action': 'fill', 'selector': '#field', 'text': 'Ready'}


def test_direct_and_deferred_calls_have_the_same_effect(scoped_contract):
    from agent.tool_executor import _unwrap_tool_search_call
    args = {'action': 'type', 'selector': '#field', 'text': 'Ready'}
    agent = SimpleNamespace(enabled_toolsets=['desktop_ui'], disabled_toolsets=[])
    name, resolved, blocked = _unwrap_tool_search_call(agent, 'tool_call', {'calls': [{'name': 'drive_preview', 'arguments': args}]})
    assert (name, blocked) == ('drive_preview', None)
    direct, deferred = [], []
    drive_preview_tool(**args, callback=lambda value: direct.append(value) or '{"success":true}')
    drive_preview_tool(**resolved, callback=lambda value: deferred.append(value) or '{"success":true}')
    assert direct == deferred == [args]


def test_foreign_scope_cannot_describe_or_unwrap_a_desktop_action(scoped_contract, monkeypatch):
    from agent.tool_executor import _unwrap_tool_search_call
    described = json.loads(tool_search.dispatch_tool_describe({'names': ['drive_preview']}, current_tool_defs=[]))
    assert 'drive_preview' not in described['tools']
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **kwargs: [])
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[])
    name, _, blocked = _unwrap_tool_search_call(agent, 'tool_call', {'calls': [{'name': 'drive_preview', 'arguments': {'action': 'click', 'selector': '#field'}}]})
    assert name == 'tool_call' and blocked
