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


def test_canceled_turn_cannot_close_bound_guest():
    live = True
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: live)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    live = False
    effects = []
    result = json.loads(close_live(state, lambda _: effects.append('read'),
                                  lambda _: effects.append('close'), {}))
    assert result['code'] == 'STALE_TARGET'
    assert effects == []


def test_repeated_bad_refs_are_not_a_destructive_recovery_grant():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    state.mark_inventory()
    effects = []
    for ref in ('guessed-control', 'another-guess', 'third-guess'):
        act_live(state, lambda _: json.dumps(page),
                 lambda _: json.dumps({'success': False, 'code': 'INVALID_ELEMENT_REF'}),
                 {'action': 'click', 'ref': ref})
    close_live(state, lambda _: json.dumps(page),
               lambda payload: effects.append(payload) or json.dumps({'success': True}), {})
    assert effects == []
    assert state.closed is False


def test_explicit_host_authorized_close_still_verifies_live_binding():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    effects = []
    result = json.loads(close_live(state, lambda _: json.dumps(page),
        lambda payload: effects.append(payload) or json.dumps({'success': True}), {},
        authorize=lambda expected: expected == page['binding']))
    assert result['success'] is True
    assert len(effects) == 1
    assert state.closed is True


def test_invalid_element_ref_invalidates_inventory_before_next_action():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}}
    assert state.accept_page(page) is None
    state.mark_inventory()
    act_live(state, lambda _: json.dumps(page),
             lambda _: json.dumps({'success': False, 'code': 'INVALID_ELEMENT_REF'}),
             {'action': 'click', 'ref': 'missing'})
    assert state.guard('drive_preview', {'action': 'click', 'ref': 'old'})['code'] == 'STALE_TARGET'


def test_tool_argument_cannot_forge_user_close_intent():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    effects = []
    result = json.loads(close_live(state, lambda _: effects.append('read'),
        lambda _: effects.append('close'), {'authorized': True, 'operation': 'close'}))
    assert result['code'] == 'USER_INTENT_REQUIRED'
    assert effects == []


def test_discovery_task_does_not_authorize_reload_or_history_navigation():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    state.operation = 'existing'
    for action in ('reload', 'back', 'forward'):
        assert state.guard('drive_preview', {'action': action})['code'] == 'USER_INTENT_REQUIRED'
        state.operation = action
        assert state.guard('drive_preview', {'action': action}) is None
        state.operation = 'existing'


def test_verified_reconnect_adopts_guest_only_after_fresh_read_and_inventory():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    page = {'success': True, 'binding': {'session_id': 'session', 'window_id': 'window',
            'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}}
    assert state.accept_page(page) is None
    state.mark_inventory()
    state.closed = True
    replacement = {**page, 'binding': {**page['binding'], 'guest_id': 2},
                   'explicit_user_selection': True, 'live_verified': True}
    assert state.reconnect(lambda expected: replacement) is None
    assert state.binding.guest_id == 2
    assert state.closed is False
    assert state.inventory is None
    assert state.guard('drive_preview', {'action': 'click', 'ref': 'old'})['code'] == 'STALE_TARGET'


import pytest


@pytest.mark.parametrize('field,value', [('session_id', 'foreign'), ('window_id', 'foreign'), ('profile_id', 'foreign')])
def test_reconnect_rejects_foreign_owner_without_changing_binding(field, value):
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    original = state.binding
    state.closed = True
    error = state.reconnect(lambda _: {'success': True, 'binding': {**binding, field: value, 'guest_id': 2},
        'explicit_user_selection': True, 'live_verified': True})
    assert error['code'] == 'WRONG_OWNER'
    assert state.binding == original
    assert state.closed is True


def test_turn_canceled_during_reconnect_cannot_adopt_guest():
    live = True
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: live)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    state.closed = True
    def verify(_):
        nonlocal live
        live = False
        return {'success': True, 'binding': {**binding, 'guest_id': 2},
                'explicit_user_selection': True, 'live_verified': True}
    assert state.reconnect(verify)['code'] == 'STALE_TARGET'
    assert state.binding.guest_id == 1
    assert state.closed is True


def test_old_inventory_callback_cannot_mark_replacement_guest_ready():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    page = {'success': True, 'binding': binding}
    state.accept_page(page)
    def delayed_inventory(_):
        assert state.reconnect(lambda _: {'success': True, 'binding': {**binding, 'guest_id': 2},
            'explicit_user_selection': True, 'live_verified': True}) is None
        return json.dumps({'success': True, 'elements': [{'ref': 'guest-a'}]})
    result = json.loads(act_live(state, lambda _: json.dumps(page), delayed_inventory, {'action': 'elements'}))
    assert result['code'] == 'STALE_TARGET'
    assert state.inventory is None


def test_ordinary_read_cannot_replace_profile_identity():
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    assert state.accept_page({'success': True, 'binding': binding}) is None
    assert state.accept_page({'success': True, 'binding': {**binding, 'profile_id': 'foreign'}})['code'] == 'WRONG_OWNER'
    assert state.binding.profile_id == 'profile'


@pytest.mark.parametrize('missing', ['explicit_user_selection', 'live_verified'])
def test_reconnect_requires_both_user_selection_and_live_verification(missing):
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    state.closed = True
    result = {'success': True, 'binding': {**binding, 'guest_id': 2},
              'explicit_user_selection': True, 'live_verified': True}
    del result[missing]
    assert state.reconnect(lambda _: result)['code'] == 'USER_INTENT_REQUIRED'
    assert state.closed is True
    assert state.binding.guest_id == 1


def test_public_read_callback_reconnects_closed_same_turn_without_replaying_action():
    from gateway.desktop_browser_turn import read_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    state.closed = True
    replacement = {**binding, 'guest_id': 2, 'user_selection': 1}
    state.reconnect_callback = lambda _: {'success': True, 'binding': replacement,
        'explicit_user_selection': True, 'live_verified': True}
    assert state.guard('desktop_preview', {'action': 'read'}) is None
    requests = []
    def read(payload):
        requests.append(payload)
        return json.dumps({'success': True, 'binding': replacement, 'text': 'fresh'})
    result = json.loads(read_live(state, read))
    assert result['text'] == 'fresh'
    assert state.binding.turn_id == 'turn'
    assert state.closed is False
    assert len(requests) == 1
    assert requests[0]['expected_binding']['guest_id'] == 2


def test_discovery_stops_guessing_after_one_fresh_inventory():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab', 'guest_id': 1, 'page_epoch': 1}
    page = {'success': True, 'binding': binding}
    state.accept_page(page)
    state.mark_inventory()
    actions = []
    def act(payload):
        actions.append(payload['action'])
        return json.dumps({'success': payload['action'] == 'elements', 'code': 'INVALID_ELEMENT_REF'})
    read = lambda _: json.dumps(page)
    act_live(state, read, act, {'action': 'click', 'ref': 'guess-1'})
    act_live(state, read, lambda payload: actions.append(payload['action']) or json.dumps({'success': True}), {'action': 'elements'})
    act_live(state, read, act, {'action': 'click', 'ref': 'guess-2'})
    result = json.loads(act_live(state, read, act, {'action': 'click', 'selector': '#guess-3'}))
    assert result['success'] is False
    assert actions == ['click', 'elements', 'click']
    assert state.closed is False


def test_manually_replaced_pane_read_uses_verified_reconnect_once():
    from gateway.desktop_browser_turn import read_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    replacement = {**binding, 'guest_id': 2, 'user_selection': 1}
    verifies = []
    state.reconnect_callback = lambda expected: verifies.append(expected) or {'success': True,
        'binding': replacement, 'explicit_user_selection': True, 'live_verified': True}
    state.discovery_failures = 2
    state.recovery_inventories = 1
    requests = []
    def read(payload):
        requests.append(payload)
        if payload['expected_binding']['guest_id'] == 1:
            return json.dumps({'success': False, 'code': 'STALE_TARGET'})
        return json.dumps({'success': True, 'binding': replacement, 'text': 'fresh'})
    result = json.loads(read_live(state, read))
    assert result['text'] == 'fresh'
    assert len(verifies) == 1
    assert len(requests) == 2
    assert state.discovery_failures == 0
    assert state.recovery_inventories == 0


def test_old_ref_cannot_reenter_after_new_guest_inventory_with_same_labels():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    replacement = {**binding, 'guest_id': 2}
    state.reconnect(lambda _: {'success': True, 'binding': replacement,
        'explicit_user_selection': True, 'live_verified': True})
    inventory = json.loads(act_live(state, lambda _: json.dumps({'success': True, 'binding': replacement}),
        lambda _: json.dumps({'success': True, 'elements': [{'ref': 'btn-save'}]}), {'action': 'elements'}))
    assert inventory['elements'][0]['ref'] != 'btn-save'
    assert state.guard('drive_preview', {'action': 'click', 'ref': 'btn-save'})['code'] == 'STALE_TARGET'
    assert state.guard('drive_preview', {'action': 'click', 'ref': inventory['elements'][0]['ref']}) is None


def test_fresh_reconnect_ref_is_unwrapped_only_for_current_guest_dispatch():
    from gateway.desktop_browser_turn import act_live
    state = BrowserTurnState(session_id='session', generation=1, turn_id='turn',
                             target_kind='VISIBLE_PANE', live_check=lambda: True)
    binding = {'session_id': 'session', 'window_id': 'window', 'tab_id': 'tab',
               'guest_id': 1, 'page_epoch': 1, 'profile_id': 'profile'}
    state.accept_page({'success': True, 'binding': binding})
    replacement = {**binding, 'guest_id': 2}
    state.reconnect(lambda _: {'success': True, 'binding': replacement,
        'explicit_user_selection': True, 'live_verified': True})
    read = lambda _: json.dumps({'success': True, 'binding': replacement})
    inventory = json.loads(act_live(state, read,
        lambda _: json.dumps({'success': True, 'elements': [{'ref': 'btn-save'}]}), {'action': 'elements'}))
    effects = []
    result = json.loads(act_live(state, read,
        lambda payload: effects.append(payload) or json.dumps({'success': True}),
        {'action': 'hover', 'ref': inventory['elements'][0]['ref']}))
    assert result['success'] is True
    assert effects[0]['ref'] == 'btn-save'
    assert effects[0]['expected_binding']['guest_id'] == 2
