"""Desktop task language reaches the actual platform prompt, without routing code."""
from types import SimpleNamespace

from agent import system_prompt


def test_desktop_effective_hint_teaches_user_intent_and_ordinary_clarification():
    hint = system_prompt.platform_hint(SimpleNamespace(platform='desktop'))
    assert 'desktop_preview' in hint
    assert 'drive_preview' in hint
    assert 'browser_exec' in hint
    assert 'Would you like to see and interact with the page' in hint
    assert 'get the price' in hint
    assert 'open Amazon' in hint
    assert 'Never silently substitute' in hint
    assert hint == system_prompt.platform_hint(SimpleNamespace(platform='desktop'))


def test_browser_intent_is_desktop_only_and_preserves_deliberate_override():
    assert 'Would you like to see and interact with the page' not in system_prompt.platform_hint(
        SimpleNamespace(platform='cli'))
    assert system_prompt.platform_hint(SimpleNamespace(platform='desktop', _platform_hint_overrides={
        'desktop': {'replace': 'User-owned platform instructions'}})) == 'User-owned platform instructions'
