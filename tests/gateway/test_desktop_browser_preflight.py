"""Preparation is native; external model and renderer replies are synthetic here."""
import json
from types import SimpleNamespace
import pytest
from agent.desktop_browser_target import BrowserTurnState, prelude, failure, recovery_text
from gateway.desktop_browser_turn import read_live, act_live


def state():return BrowserTurnState(session_id='s',generation=2,turn_id='t',target_kind='UNRESOLVED')

def live(epoch=1,session='s'):
    return {'success':True,'url':'http://localhost/owned','text':'current','binding':{'session_id':session,'window_id':'w','tab_id':'tab','guest_id':2,'page_epoch':epoch}}


def test_missing_bridge_stops_prelude_before_any_main_answer(monkeypatch):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',lambda *args:{'target_kind':'VISIBLE_PANE','operation':'existing'})
    s=state();p=prelude(s,SimpleNamespace(_current_turn_id='turn-native'),'Continue on that exact page',[])
    assert next(p)[1:] == ('desktop_preview',{'action':'read'})
    with pytest.raises(StopIteration):p.send(json.dumps(failure('PANE_UNAVAILABLE')))
    assert s.native_failure['code']=='PANE_UNAVAILABLE'
    assert 'cannot access' in recovery_text(s,'en')
    assert '无法实时访问' in recovery_text(s,'zh')


def test_visible_open_is_native_first_correct_action_and_requires_inventory(monkeypatch):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',lambda *args:{'target_kind':'VISIBLE_PANE','operation':'open','url':'http://localhost/new'})
    s=state();p=prelude(s,SimpleNamespace(_current_turn_id='turn-native'),'Open this beside our chat',[])
    assert next(p)[1:] == ('desktop_preview',{'action':'open','url':'http://localhost/new'})
    assert p.send(json.dumps(live()))[1:] == ('drive_preview',{'action':'elements'})
    with pytest.raises(StopIteration):p.send(json.dumps({'success':True,'elements':[]}))
    assert s.native_failure is None
    assert s.binding.turn_id=='turn-native'
    assert s.inventory==(2,1)


@pytest.mark.parametrize('kind',['INDEPENDENT','NONE','UNRESOLVED'])
def test_other_targets_do_not_probe_or_act_on_pane(monkeypatch,kind):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',lambda *args:{'target_kind':kind,'operation':'none'})
    s=state();assert list(prelude(s,SimpleNamespace(),'request',[]))==[]
    assert bool(s.native_failure)==(kind=='UNRESOLVED')


def test_owner_mismatch_or_navigation_prevents_mutation():
    s=state();s.select('VISIBLE_PANE');s.accept_page(live());s.mark_inventory();called=[]
    result=act_live(s,lambda _:json.dumps(live(session='foreign')),lambda p:called.append(p),{'action':'click','ref':'save'})
    assert json.loads(result)['code']=='WRONG_OWNER' and not called
    result=act_live(s,lambda _:json.dumps(live(epoch=2)),lambda p:called.append(p),{'action':'click','ref':'save'})
    assert json.loads(result)['code']=='STALE_TARGET' and not called


def test_inventory_can_refresh_same_owned_guest_but_not_foreign_window():
    s=state();s.select('VISIBLE_PANE');s.accept_page(live());s.mark_inventory();s.preflight_complete=True;asked=[]
    result=act_live(s,lambda _:json.dumps(live(epoch=2)),lambda p:asked.append(p) or json.dumps({'success':True,'elements':[]}),{'action':'elements'})
    assert json.loads(result)['success'] and s.inventory==(2,2)
    assert asked[0]['expected_binding']['page_epoch']==2


def test_cancelled_generation_never_reads_or_sends_input():
    s=state();s.select('VISIBLE_PANE');s.live_check=lambda:False
    assert json.loads(read_live(s,lambda _:pytest.fail('read after cancellation')))['code']=='STALE_TARGET'


def test_initial_inventory_cannot_adopt_navigation_after_live_read():
    s=state();s.select('VISIBLE_PANE');s.accept_page(live());called=[]
    result=act_live(s,lambda _:json.dumps(live(epoch=2)),lambda p:called.append(p),{'action':'elements'})
    assert json.loads(result)['code']=='STALE_TARGET'
    assert not called
    assert s.binding.page_epoch==1


def test_native_prelude_uses_scoped_wrapper_when_pane_tools_are_deferred(monkeypatch):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',lambda *args:{'target_kind':'VISIBLE_PANE','operation':'existing'})
    s=state();agent=SimpleNamespace(valid_tool_names={'tool_call','tool_search','tool_describe'})
    p=prelude(s,agent,'Inspect the page beside me',[])
    assert next(p)[1:] == ('tool_call',{'calls':[{'name':'desktop_preview','arguments':{'action':'read'}}]})
    assert p.send(json.dumps(live()))[1:] == ('tool_call',{'calls':[{'name':'drive_preview','arguments':{'action':'elements'}}]})
    assert p.send(json.dumps({'success':True,'elements':[]}))[1] == 'tool_describe'
    from tools.drive_preview_tool import ACT_PREVIEW_SCHEMA
    from tools.preview_tool import PREVIEW_SCHEMA
    schemas = {'tools': {schema['name']: {'parameters': schema['parameters']} for schema in (ACT_PREVIEW_SCHEMA, PREVIEW_SCHEMA)}}
    with pytest.raises(StopIteration):p.send(json.dumps(schemas))
    assert s.preflight_complete


def test_prelude_cannot_replace_an_explicit_native_target(monkeypatch):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent',lambda *args:{'target_kind':'INDEPENDENT','operation':'none'})
    s=state();s.select('VISIBLE_PANE',provenance='explicit_ui')
    assert list(prelude(s,SimpleNamespace(),'request',[]))==[]
    assert s.kind=='VISIBLE_PANE'
    assert s.native_failure['code']=='TARGET_MISMATCH'
    assert s.binding.selection_provenance=='explicit_ui'


def test_independent_turn_cannot_enter_pane_through_shared_annotation_callback():
    s=state();s.select('INDEPENDENT')
    result=act_live(s,lambda _:pytest.fail('Independent callback must not read visible page'),lambda _:pytest.fail('Independent callback must not annotate visible page'),{'action':'pin','ref':'target'})
    assert json.loads(result)['code']=='TARGET_MISMATCH'


def test_deferred_prelude_discloses_complete_scoped_contract_before_main(monkeypatch):
    from tools.drive_preview_tool import ACT_PREVIEW_SCHEMA
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent', lambda *args: {'target_kind': 'VISIBLE_PANE', 'operation': 'existing'})
    s = state()
    agent = SimpleNamespace(valid_tool_names={'tool_call', 'tool_search', 'tool_describe'})
    p = prelude(s, agent, 'Update the field on the page beside me', [])
    assert next(p)[1] == 'tool_call'
    assert p.send(json.dumps(live()))[1] == 'tool_call'
    disclosure = p.send(json.dumps({'success': True, 'elements': []}))
    assert disclosure[1:] == ('tool_describe', {'names': ['desktop_preview', 'drive_preview']})
    assert not s.preflight_complete
    schemas = {'tools': {'drive_preview': {'parameters': ACT_PREVIEW_SCHEMA['parameters']}, 'desktop_preview': {'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}}, 'required': ['action']}}}}
    with pytest.raises(StopIteration):
        p.send(json.dumps(schemas))
    assert s.preflight_complete and s.native_failure is None


def test_missing_scoped_contract_stops_without_browser_substitution(monkeypatch):
    monkeypatch.setattr('agent.desktop_browser_target.propose_intent', lambda *args: {'target_kind': 'VISIBLE_PANE', 'operation': 'existing'})
    s = state()
    p = prelude(s, SimpleNamespace(valid_tool_names={'tool_call', 'tool_describe'}), 'Update the owned page', [])
    next(p)
    p.send(json.dumps(live()))
    assert p.send(json.dumps({'success': True, 'elements': []}))[1] == 'tool_describe'
    with pytest.raises(StopIteration):
        p.send(json.dumps({'tools': {}, 'not_found': ['drive_preview']}))
    assert s.native_failure['code'] == 'TOOL_CONTRACT_UNAVAILABLE'
    assert s.kind == 'VISIBLE_PANE'


def test_native_wrapper_example_conforms_to_the_advertised_bridge_schema():
    from jsonschema import validate
    from agent.desktop_browser_target import native_pane_call
    from tools.tool_search import bridge_tool_schemas
    agent = SimpleNamespace(valid_tool_names={'tool_call', 'tool_describe', 'tool_search'})
    _, name, arguments = native_pane_call(agent, 'drive_preview', {'action': 'elements'})
    schema = next(td['function']['parameters'] for td in bridge_tool_schemas(2) if td['function']['name'] == name)
    validate(arguments, schema)
