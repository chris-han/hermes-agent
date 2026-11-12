"""Preservation counterexample on the real per-turn close dispatcher."""
import json
from agent.desktop_browser_target import BrowserTurnState
from gateway.desktop_browser_turn import close_live


def test_discovery_failure_does_not_authorize_close():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    effects = []
    def close(payload):
        effects.append(payload)
        return json.dumps({'success': True})
    result = json.loads(close_live(state, lambda _: json.dumps(page), close, {}))
    assert effects == []
    assert result['success'] is False
    assert state.closed is False


def test_same_turn_explicit_reopen_can_be_verified_before_binding_renewal():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    state.mark_inventory()
    state.closed = True
    replacement = {**page, 'binding': {**page['binding'], 'guest_id': 2}}
    # Ordinary page acceptance must remain strict. Renewal needs a distinct,
    # owner-verified path rather than weakening accept_page for every read.
    assert state.accept_page(replacement)['code'] == 'STALE_TARGET'
    assert state.closed is True
    assert state.binding.guest_id == 1
