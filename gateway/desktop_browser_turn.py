"""Generation-fenced Desktop bridge operations for a native per-turn binding."""
from __future__ import annotations
import json
from agent.desktop_browser_target import BrowserTurnState, failure, parse_result

def read_live(state,raw_request,*,start=None,count=None,refresh=True,_renewed=False):
    if state.closed:
        _renewed=True
        if not callable(state.reconnect_callback):return json.dumps(failure('PANE_UNAVAILABLE'))
        error=state.reconnect(state.reconnect_callback)
        if error:return json.dumps(error)
    if state.live_check is not None and not state.live_check():return json.dumps(failure('STALE_TARGET'))
    revision=state.binding_revision
    raw=raw_request({'start':start,'count':count,'require_live':True,'expected_binding':state.expected_page(),'refresh_binding':refresh})
    if revision != state.binding_revision:return json.dumps(failure('STALE_TARGET'))
    result=parse_result(raw)
    error=state.accept_page(result,refresh=refresh)
    if error and error.get('code') in ('STALE_TARGET', 'PANE_UNAVAILABLE') and not _renewed and callable(state.reconnect_callback):
        renewed=state.reconnect(state.reconnect_callback)
        if renewed:return json.dumps(renewed)
        return read_live(state,raw_request,start=start,count=count,refresh=refresh,_renewed=True)
    return json.dumps(error) if error else raw


def open_live(state,raw_request,args):
    if state.operation!='open' or args.get('url')!=state.requested_url:return json.dumps(failure('TARGET_MISMATCH'))
    if state.live_check is not None and not state.live_check():return json.dumps(failure('STALE_TARGET'))
    raw=raw_request({'url':state.requested_url,'label':args.get('label',''),'expected_binding':state.expected_page()})
    error=state.accept_page(parse_result(raw))
    return json.dumps(error) if error else raw


def act_live(state,raw_read,raw_act,payload):
    error=state.guard('drive_preview',payload)
    if error:return json.dumps(error)
    is_inventory=payload.get('action')=='elements'
    raw=read_live(state,raw_read,refresh=is_inventory and state.preflight_complete)
    result=parse_result(raw)
    if not isinstance(result,dict) or result.get('success') is False:return raw
    error=state.guard('drive_preview',payload)
    if error:return json.dumps(error)
    revision=state.binding_revision
    action_payload=dict(payload)
    if state.binding_revision and action_payload.get('ref'):
        action_payload['ref']=action_payload['ref'].removeprefix(f'binding:{state.binding_revision}:')
    result=raw_act({**action_payload,'expected_binding':state.expected_page()})
    if revision != state.binding_revision:return json.dumps(failure('STALE_TARGET'))
    value=parse_result(result)
    if isinstance(value,dict) and value.get('success') is False and (value.get('code') in ('INVALID_ELEMENT_REF', 'INVALID_SELECTOR') or (value.get('code') == 'INVALID_ARGUMENT' and value.get('path') == 'arguments.selector')):
        with state._lock:
            state.inventory = None
            state.discovery_failures += 1
    if is_inventory and isinstance(value,dict) and value.get('success') is True:
        if state.discovery_failures:state.recovery_inventories += 1
        state.mark_inventory()
    if isinstance(value, dict) and state.binding_revision:
        def scope_refs(item):
            if isinstance(item, dict):
                return {key: (f'binding:{state.binding_revision}:'+entry if key == 'ref' and isinstance(entry, str) else scope_refs(entry)) for key, entry in item.items()}
            if isinstance(item, list):return [scope_refs(entry) for entry in item]
            return item
        result=json.dumps(scope_refs(value))
    return result


def close_live(state,raw_read,raw_close,args,*,authorize=None):
    # Authorization is supplied by the owning host, never by tool arguments or
    # element-discovery failures. Verify liveness before requesting user intent.
    if state.live_check is not None and not state.live_check():return json.dumps(failure('STALE_TARGET'))
    if authorize is None or authorize(state.expected_page()) is not True:
        return json.dumps(failure('USER_INTENT_REQUIRED'))
    if args.get('url') and args['url'] not in (state.current_url,state.requested_url):return json.dumps(failure('TARGET_MISMATCH'))
    checked=read_live(state,raw_read,refresh=False)
    value=parse_result(checked)
    if not isinstance(value,dict) or value.get('success') is False:return checked
    result=raw_close({'expected_binding':state.expected_page()})
    value=parse_result(result)
    if isinstance(value,dict) and value.get('success') is True:state.closed=True
    return result
