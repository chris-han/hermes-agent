"""Ephemeral Desktop target lease. Existing session/renderer checks remain authority.

Intent is a proposal, never an owner grant. No profile preferences or browser state
are stored here. The same turn object follows native/direct/deferred execution.
"""
from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class BrowserTargetBinding:
    target_kind: str
    selection_provenance: str
    session_id: str
    generation: Any
    turn_id: str
    window_id: str | None = None
    tab_id: str | None = None
    guest_id: int | None = None
    page_epoch: int | None = None


def failure(code: str) -> dict:
    return {'success': False, 'code': code, 'error': code}


class BrowserTurnState:
    def __init__(self, *, session_id, generation, turn_id, target_kind='UNRESOLVED', live_check=None):
        self.kind = target_kind
        self.binding = BrowserTargetBinding(target_kind if target_kind != 'NONE' else 'UNRESOLVED',
                                            'proposed_intent', session_id, generation, turn_id)
        self.live_check = live_check
        self.inventory = None
        self.native_failure = None
        self.operation = 'none'
        self.requested_url = ''
        self.current_url = ''
        self.closed = False
        self.preflight_complete = False
        self.receipts = []
        self._lock = RLock()
        self._selected = target_kind != 'UNRESOLVED'

    def select(self, kind, provenance='proposed_intent'):
        if kind not in ('VISIBLE_PANE','INDEPENDENT','UNRESOLVED','NONE'):
            kind='UNRESOLVED'
        with self._lock:
            if self._selected:
                return failure('TARGET_MISMATCH') if kind != self.kind else None
            self._selected = True
            self.kind=kind
            self.binding=replace(self.binding,target_kind=kind if kind!='NONE' else 'UNRESOLVED',selection_provenance=provenance)

    def canonical_call(self, name, args):
        # The two existing pane tools share logical operations. Preserve legacy
        # schemas outside a bound Desktop turn; canonicalize BEFORE their probes.
        if self.kind=='VISIBLE_PANE':
            action=args.get('action')
            if name=='drive_preview' and action in ('open','read','close'):
                return 'desktop_preview',args
            if name=='desktop_preview' and action in ('elements','click','hover','type','scroll','press','strobe','back','forward','reload'):
                return 'drive_preview',args
        return name,args

    def guard(self,name,args):
        if self.kind == 'NONE':return None
        browser=name in ('desktop_preview','drive_preview','annotate_preview','browser_exec','browser_cdp')
        if not browser and not (self.kind=='VISIBLE_PANE' and name=='execute_code'):return None
        if self.native_failure is not None:return dict(self.native_failure)
        if self.closed:return failure('PANE_UNAVAILABLE')
        if self.live_check is not None and not self.live_check():return failure('STALE_TARGET')
        if self.kind in ('UNRESOLVED','NONE'):return failure('TARGET_UNRESOLVED')
        if self.kind=='VISIBLE_PANE' and name in ('browser_exec','browser_cdp','execute_code'):return failure('TARGET_MISMATCH')
        if self.kind=='INDEPENDENT' and name in ('desktop_preview','drive_preview','annotate_preview'):return failure('TARGET_MISMATCH')
        if name=='drive_preview' and args.get('ref') and args.get('action')!='elements':
            if self.binding.tab_id is None:return failure('PANE_UNAVAILABLE')
            if self.inventory!=(self.binding.guest_id,self.binding.page_epoch):return failure('STALE_TARGET')
        return None

    def accept_page(self,result,*,refresh=True):
        with self._lock:
            b=result.get('binding') if isinstance(result,dict) else None
            if not isinstance(b,dict) or result.get('success') is False:
                return failure(result.get('code','PANE_UNAVAILABLE') if isinstance(result,dict) else 'PANE_UNAVAILABLE')
            if b.get('session_id')!=self.binding.session_id:return failure('WRONG_OWNER')
            if not all(b.get(k) is not None for k in ('window_id','tab_id','guest_id','page_epoch')):return failure('PANE_UNAVAILABLE')
            if self.binding.tab_id is not None:
                if b['window_id']!=self.binding.window_id or b['tab_id']!=self.binding.tab_id:return failure('WRONG_OWNER')
                if b['guest_id']!=self.binding.guest_id:return failure('STALE_TARGET')
                if not refresh and b['page_epoch']!=self.binding.page_epoch:return failure('STALE_TARGET')
            self.binding=replace(self.binding,**{k:b[k] for k in ('window_id','tab_id','guest_id','page_epoch')})
            self.current_url=result.get('url','')
            return None

    def mark_inventory(self):
        with self._lock:self.inventory=(self.binding.guest_id,self.binding.page_epoch)

    def expected_page(self):
        b=self.binding
        return {k:getattr(b,k) for k in ('session_id','window_id','tab_id','guest_id','page_epoch')} if b.tab_id else None


_current: ContextVar[BrowserTurnState | None]=ContextVar('desktop_browser_turn',default=None)

def current_browser_turn():return _current.get()

@contextmanager
def browser_turn_scope(state):
    token=_current.set(state)
    try:yield state
    finally:_current.reset(token)


def guard_browser_call(name,args):
    state=current_browser_turn()
    return state.guard(name,args) if state is not None else None


def state_for_agent(agent):
    state=getattr(agent,'_desktop_browser_turn',None)
    state = state if isinstance(state,BrowserTurnState) else current_browser_turn()
    return None if state is not None and state.kind == 'NONE' else state




def assets():
    return json.loads((Path(__file__).resolve().parents[1]/'semantier-runtime/src/prompts/agents/DESKTOP_BROWSER_TARGET_INTENT.json').read_text(encoding='utf-8'))


def recovery_text(state, language):
    copy=assets()['recovery']['zh' if str(language).startswith('zh') else 'en']
    return copy.get((state.native_failure or {}).get('code'),copy['PANE_UNAVAILABLE'])


def propose_intent(agent, message, history, pane_context=None):
    """A bounded semantic proposal through the SAME configured interruptible model route.

    It cannot supply an owner identity. Account as auxiliary usage so the small
    classification context never overwrites the main conversation's usage anchor.
    """
    from agent.aux_accounting import record_aux_usage
    from agent.turn_truncation import normalize_response_for_agent
    data=assets()
    context=[{'role':m.get('role'),'content':str(m.get('content',''))[-1600:]} for m in (history or [])[-2:] if m.get('role') in ('user','assistant')]
    messages=[{'role':'system','content':data['prompt']},{'role':'user','content':json.dumps({'request':message,'reference_context':context,'owned_page_present':bool(isinstance(pane_context,dict) and pane_context.get('success'))},ensure_ascii=False)}]
    request=agent._build_api_kwargs(messages,tools_for_api=[data['schema']])
    request['stream']=False
    request['timeout']=30
    for key in ('max_tokens','max_completion_tokens','max_output_tokens'):
        if key in request:request[key]=min(int(request[key] or 512),512)
    # Respect each native transport's tool-choice shape, never a new provider.
    request.pop('tool_choice',None)
    wire_tools=request.get('tools') or []
    if wire_tools and 'function' in wire_tools[0]:
        request['tool_choice']={'type':'function','function':{'name':'browser_target_intent'}}
    elif getattr(agent,'api_mode','')=='anthropic_messages':
        request['tool_choice']={'type':'tool','name':'browser_target_intent'}
    elif 'input' in request:
        request['tool_choice']={'type':'function','name':'browser_target_intent'}
    from agent.message_sanitization import sanitize_outbound_kwargs
    from agent.semantier_gateway import attribute_request,configure_cloud_request,agent_cloud_settings
    from hermes_cli.middleware import run_llm_execution_middleware
    request_id=str(getattr(agent,'_current_turn_id','') or agent.session_id)+':browser-target'
    settings=agent_cloud_settings(agent)
    request=attribute_request(request,base_url=str(agent.base_url or ''),purpose='chat',session_id=str(agent.session_id or ''),
        **(getattr(agent,'_semantier_task_context',None) or {'task_id':str(agent.session_id or ''),'root_run_id':str(agent.session_id or ''),'run_id':str(getattr(agent,'_current_turn_id','') or '')}),
        request_id=request_id,settings=settings)
    request=configure_cloud_request(request,base_url=str(agent.base_url or ''),settings=settings,expected_revision=getattr(agent,'_semantier_cloud_revision',None))
    sanitize_outbound_kwargs(agent,request)
    response=run_llm_execution_middleware(request,agent._interruptible_api_call,original_request=request,
        task_id=str(agent.session_id or ''),turn_id=str(getattr(agent,'_current_turn_id','') or ''),api_request_id=request_id,
        session_id=str(agent.session_id or ''),platform=agent.platform or '',model=agent.model,provider=agent.provider,
        base_url=agent.base_url,api_mode=agent.api_mode,api_call_count=0,middleware_trace=[])
    record_aux_usage(response,'desktop_browser_target',provider=agent.provider,base_url=agent.base_url)
    normalized=normalize_response_for_agent(agent,response)
    calls=normalized.tool_calls or []
    if len(calls)!=1 or calls[0].function.name!='browser_target_intent':return {'target_kind':'UNRESOLVED','operation':'none'}
    raw_args=calls[0].function.arguments
    proposal=json.loads(raw_args) if isinstance(raw_args,str) else raw_args
    if (not isinstance(proposal,dict) or set(proposal)-{'target_kind','operation','url'}
        or proposal.get('target_kind') not in ('VISIBLE_PANE','INDEPENDENT','UNRESOLVED','NONE')
        or proposal.get('operation') not in ('existing','open','none')):
        return {'target_kind':'UNRESOLVED','operation':'none'}
    if proposal['target_kind']=='VISIBLE_PANE' and proposal['operation']=='open':
        url=str(proposal.get('url',''))
        if urlsplit(url).scheme not in ('http','https') or not urlsplit(url).netloc:return {'target_kind':'UNRESOLVED','operation':'none'}
    return proposal


def parse_result(raw):
    try:return json.loads(raw) if isinstance(raw,str) else raw
    except (TypeError,ValueError):return None


def native_pane_call(agent, name, arguments):
    """Use native deferred dispatch without widening the session's tool catalog."""
    valid = getattr(agent, 'valid_tool_names', None)
    if isinstance(valid, (set, frozenset, list, tuple)) and name not in valid and 'tool_call' in valid:
        return '', 'tool_call', {'calls': [{'name': name, 'arguments': arguments}]}
    return '', name, arguments


def prelude(state: BrowserTurnState,agent,message,history):
    """Native preflight is observable through the existing scripted tool-round path."""
    capture=getattr(state,'capture_context',None)
    try:pane_context=parse_result(capture()) if callable(capture) else None
    except Exception:pane_context=None
    try:
        proposal=propose_intent(agent,message,history,pane_context)
    except Exception:
        state.native_failure=failure('INTENT_UNAVAILABLE')
        return
    state.binding=replace(state.binding,turn_id=str(getattr(agent,'_current_turn_id',None) or state.binding.turn_id))
    selection_error = state.select(proposal['target_kind'])
    if selection_error:
        state.native_failure = selection_error
        state.receipts.append({'phase': 'intent', 'proposal': proposal, **selection_error})
        return
    state.operation=proposal['operation']
    state.requested_url=proposal.get('url','')
    state.receipts.append({'phase':'intent','proposal':proposal,'selection_provenance':'proposed_intent'})
    if state.kind=='UNRESOLVED':
        state.native_failure=failure('TARGET_UNRESOLVED')
        return
    if state.kind!='VISIBLE_PANE':return
    if callable(capture) and (state.operation!='open' or (isinstance(pane_context,dict) and pane_context.get('binding'))):
        error=state.accept_page(pane_context)
        if error:
            state.native_failure=error
            state.receipts.append({'phase':'owner_selection',**error})
            return
        state.receipts.append({'phase':'owner_selection','success':True,'binding':asdict(state.binding),'selection_provenance':'renderer_selection'})
    if state.operation=='open':
        raw=yield native_pane_call(agent, 'desktop_preview', {'action':'open','url':state.requested_url})
    else:
        raw=yield native_pane_call(agent, 'desktop_preview', {'action':'read'})
    result=parse_result(raw)
    error=state.accept_page(result)
    if error:
        state.native_failure=error
        state.receipts.append({'phase':'live_preflight',**error})
        return
    state.receipts.append({'phase':'live_preflight','success':True,'binding':asdict(state.binding)})
    raw=yield native_pane_call(agent, 'drive_preview', {'action':'elements'})
    result=parse_result(raw)
    if not isinstance(result,dict) or result.get('success') is not True:
        state.native_failure=failure((result or {}).get('code','PANE_UNAVAILABLE') if isinstance(result,dict) else 'PANE_UNAVAILABLE')
        return
    state.mark_inventory()
    state.receipts.append({'phase':'inventory','success':True,'binding':asdict(state.binding)})
    valid = getattr(agent, 'valid_tool_names', None)
    if isinstance(valid, (set, frozenset, list, tuple)):
        needed = [name for name in ('desktop_preview', 'drive_preview') if name not in valid]
        if needed:
            if 'tool_describe' not in valid:
                state.native_failure = failure('TOOL_CONTRACT_UNAVAILABLE')
                return
            raw = yield '', 'tool_describe', {'names': needed}
            result = parse_result(raw)
            contracts = result.get('tools') if isinstance(result, dict) else None
            if (not isinstance(contracts, dict) or any(
                    not isinstance(contracts.get(name), dict)
                    or not isinstance(contracts[name].get('parameters'), dict) for name in needed)):
                state.native_failure = failure('TOOL_CONTRACT_UNAVAILABLE')
                return
            state.receipts.append({'phase': 'tool_contract', 'success': True, 'names': needed})
    state.preflight_complete=True
