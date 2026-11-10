"""Generation-fenced Desktop bridge operations for a native per-turn binding."""
from __future__ import annotations
import json
from agent.desktop_browser_target import BrowserTurnState, failure, parse_result

def read_live(state,raw_request,*,start=None,count=None,refresh=True):
    if state.live_check is not None and not state.live_check():return json.dumps(failure('STALE_TARGET'))
    raw=raw_request({'start':start,'count':count,'require_live':True,'expected_binding':state.expected_page(),'refresh_binding':refresh})
    result=parse_result(raw)
    error=state.accept_page(result,refresh=refresh)
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
    result=raw_act({**payload,'expected_binding':state.expected_page()})
    value=parse_result(result)
    if is_inventory and isinstance(value,dict) and value.get('success') is True:state.mark_inventory()
    return result


def close_live(state,raw_read,raw_close,args):
    if args.get('url') and args['url'] not in (state.current_url,state.requested_url):return json.dumps(failure('TARGET_MISMATCH'))
    checked=read_live(state,raw_read,refresh=False)
    value=parse_result(checked)
    if not isinstance(value,dict) or value.get('success') is False:return checked
    result=raw_close({'expected_binding':state.expected_page()})
    value=parse_result(result)
    if isinstance(value,dict) and value.get('success') is True:state.closed=True
    return result
