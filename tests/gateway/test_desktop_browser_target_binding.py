"""Native browser binding invariants; synthetic metadata, never a browser/auth substitute."""
import json
import pytest
from agent.desktop_browser_target import BrowserTurnState, BrowserTargetBinding, browser_turn_scope, current_browser_turn


def state(kind='VISIBLE_PANE'):
    return BrowserTurnState(session_id='session-A', generation=7, turn_id='turn-1', target_kind=kind)


def page(**changes):
    return {'success': True, 'binding': {'session_id': 'session-A', 'window_id': 'window-A', 'tab_id': 'tab-A', 'guest_id': 17, 'page_epoch': 2, **changes}, 'text': 'Enabled'}


def test_case_b_normalizes_logical_open_before_direct_or_deferred_validation():
    s = state()
    assert s.canonical_call('drive_preview', {'action': 'open', 'url': 'http://localhost'}) == ('desktop_preview', {'action': 'open', 'url': 'http://localhost'})
    assert s.canonical_call('desktop_preview', {'action': 'elements'}) == ('drive_preview', {'action': 'elements'})


def test_binding_records_native_owner_generation_and_epoch():
    s=state();assert s.accept_page(page()) is None
    assert isinstance(s.binding, BrowserTargetBinding)
    assert (s.binding.session_id,s.binding.generation,s.binding.tab_id,s.binding.page_epoch)==('session-A',7,'tab-A',2)


@pytest.mark.parametrize('tool', ['browser_exec','execute_code'])
def test_visible_target_blocks_independent_execution(tool):
    s=state();assert s.guard(tool,{})['code']=='TARGET_MISMATCH'


@pytest.mark.parametrize('tool', ['desktop_preview','drive_preview'])
def test_independent_target_blocks_pane_execution(tool):
    s=state('INDEPENDENT');assert s.guard(tool,{'action':'read'})['code']=='TARGET_MISMATCH'


def test_unrelated_tools_and_unbound_turns_keep_existing_behavior():
    assert state().guard('read_file',{}) is None
    from agent.desktop_browser_target import guard_browser_call
    assert guard_browser_call('browser_exec',{}) is None
    assert state('NONE').guard('browser_exec',{})['code']=='TARGET_UNRESOLVED'


def test_case_d_missing_live_read_cannot_become_verified():
    s=state();assert s.accept_page(None)['code']=='PANE_UNAVAILABLE'
    assert s.binding.tab_id is None
    assert s.accept_page({'text':'old transcript text'})['code']=='PANE_UNAVAILABLE'


@pytest.mark.parametrize('field,value', [('session_id','session-B'),('window_id','window-B'),('tab_id','tab-B'),('guest_id',18)])
def test_wrong_owner_or_substituted_guest_rejected(field,value):
    s=state();s.accept_page(page());assert s.accept_page(page(**{field:value}))['code'] in ('WRONG_OWNER','STALE_TARGET')


def test_navigation_invalidates_ref_inventory():
    s=state();s.accept_page(page());s.mark_inventory()
    assert s.guard('drive_preview',{'action':'click','ref':'btn-save'}) is None
    assert s.accept_page(page(page_epoch=3),refresh=False)['code']=='STALE_TARGET'
    assert s.accept_page(page(page_epoch=3),refresh=True) is None
    assert s.guard('drive_preview',{'action':'click','ref':'btn-save'})['code']=='STALE_TARGET'
    s.mark_inventory();assert s.guard('drive_preview',{'action':'click','ref':'btn-save'}) is None


def test_context_is_per_turn_and_restored():
    a,b=state(),state('INDEPENDENT')
    assert current_browser_turn() is None
    with browser_turn_scope(a):
        assert current_browser_turn() is a
        with browser_turn_scope(b):assert current_browser_turn() is b
        assert current_browser_turn() is a
    assert current_browser_turn() is None


def test_ambiguous_target_does_not_admit_browser_effect():
    assert state('UNRESOLVED').guard('browser_exec',{})['code']=='TARGET_UNRESOLVED'

@pytest.mark.parametrize('wrapped',[False,True])
def test_real_native_parser_canonicalizes_case_b_before_schema_probe(wrapped,monkeypatch):
    from types import SimpleNamespace
    from agent.tool_executor import _parse_tool_call
    import agent.tool_executor as executor
    from tools import tool_search
    s=state();agent=SimpleNamespace(_desktop_browser_turn=s)
    monkeypatch.setattr(executor,'_tool_search_scoped_names',lambda _:frozenset({'desktop_preview','drive_preview'}))
    checked=[]
    monkeypatch.setattr(tool_search,'validate_deferred_call_args',lambda name,args:checked.append(name))
    name='tool_call' if wrapped else 'drive_preview'
    args={'name':'drive_preview','arguments':{'action':'open','url':'http://localhost'}} if wrapped else {'action':'open','url':'http://localhost'}
    call=SimpleNamespace(function=SimpleNamespace(name=name,arguments=json.dumps(args)))
    parsed=_parse_tool_call(agent,call)
    assert parsed.name=='desktop_preview'
    if wrapped:assert checked==['desktop_preview']


def test_browser_exec_native_entry_cannot_escape_bound_pane():
    from tools.browser_use_cli import browser_exec
    with browser_turn_scope(state()):
        assert json.loads(browser_exec('print("must not execute")'))['code']=='TARGET_MISMATCH'


def test_worker_context_carries_binding():
    from concurrent.futures import ThreadPoolExecutor
    from tools.thread_context import propagate_context_to_thread
    s=state()
    with browser_turn_scope(s),ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(propagate_context_to_thread(current_browser_turn)).result() is s


def test_selected_binding_cannot_be_overridden():
    s=state();s.accept_page(page())
    assert s.select('INDEPENDENT')['code']=='TARGET_MISMATCH'
    assert s.kind=='VISIBLE_PANE'
