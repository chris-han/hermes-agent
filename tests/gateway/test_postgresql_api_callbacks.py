"""Canonical gateway API callbacks execute on genuine PostgreSQL transactions."""
import json
from uuid import uuid4

import pytest

from tests.hermes_state.test_postgresql_sessions import pg_settings


@pytest.fixture
def pg_gateway(tmp_path,monkeypatch,pg_settings):
    from hermes_state_postgresql import SessionScope,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(
        ('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *args,**kwargs:pytest.fail('hosted callback opened SQLite'))
    from types import SimpleNamespace
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_state_backend import open_session_db
    from hermes_state_runtime import begin_runtime_epoch
    from tests.gateway.conftest import api
    from hermes_constants import set_hermes_home_override,reset_hermes_home_override
    home_token=set_hermes_home_override(tmp_path)
    import hermes_state
    monkeypatch.setattr(hermes_state,'DEFAULT_DB_PATH',tmp_path/'state.db')
    db=open_session_db(tmp_path/'state.db')
    store=SessionStore(config=GatewayConfig(),sessions_dir=tmp_path/'sessions')
    store._db=db
    runner=SimpleNamespace(_draining=False,session_store=store)
    authority=SessionAuthority(runner,profile_id='default',instance_id='first',db=db,
        epoch=begin_runtime_epoch(db,instance_id='first'))
    runner.session_authority=authority
    api_fixture=api.__wrapped__(authority,tmp_path,monkeypatch)
    adapter=next(api_fixture)
    try:yield adapter,authority
    finally:
        try:next(api_fixture)
        except StopIteration:pass
        db.close()
        reset_hermes_home_override(home_token)



def test_pg_declared_api_binding_retry_result_and_terminal_media_compaction(pg_gateway):
    from tests.gateway.test_api_authority_continuity import test_declared_api_root_restores_compression_tip_and_retries_same_admission
    from gateway.session_api_turn import api_settings,admit_api_turn
    from gateway.session_contract import SessionRef
    from gateway.platforms.api_server_authority_runs import run_admission
    from gateway.platforms.api_server_response_admissions import _request_admission_rows
    from gateway.session_results import admission_result,finish_result
    from gateway.session_api_media import compact_settled_api_payloads
    from hermes_state_runtime import claim_session_input,settle_session_input
    adapter,authority=pg_gateway
    test_declared_api_root_restores_compression_tip_and_retries_same_admission(adapter,authority)
    assert isinstance(api_settings(authority,SessionRef('default','s')),dict)
    owned,row=run_admission(adapter,'one')
    assert owned is authority and row['request_id']=='one'
    assert _request_admission_rows(authority.db,'one')[0]['admission_id']==row['admission_id']
    claimed=claim_session_input(authority.db,epoch=authority.epoch,session_id='s')
    settle_session_input(authority.db,epoch=authority.epoch,admission_id=claimed['admission_id'],
        generation=claimed['generation'],outcome='completed',result={'result':{'final_response':'retained output'},'usage':{}})
    assert admission_result(authority.db,claimed['admission_id'])['result']['final_response']=='retained output'
    assert finish_result(authority.db,epoch=authority.epoch,row=claimed,response='late replacement',outcome='failed')[1]=='retained output'
    assert compact_settled_api_payloads(authority.db)==1
    assert compact_settled_api_payloads(authority.db)==0
    assert admit_api_turn(adapter,session_id='s',request_id='one',user_message='hello',conversation_history=[],
        gateway_session_key='conversation',bind_declared_conversation=True)[2]['admission_id']==claimed['admission_id']


def test_pg_room_callback_identity_conflict_and_canonical_hosted_fifo(pg_gateway,tmp_path,monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from gateway.platforms.api_server_room_dispatch import _ensure_hosted_member_session
    from gateway.session_acp_lifecycle import end_idle_acp_sessions
    from gateway.session_a2a import storage_source
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.session_contract import SessionRef
    from gateway.session_hosted_transport import check_remote_hosted_admission
    from tests.gateway.conftest import hosted_owner
    from tests.gateway.test_session_hosted_rpc import test_room_binding_exact_retry_terminal_history_and_unknown
    adapter,authority=pg_gateway
    # The standalone API branch owns the real callback transaction too.
    adapter.gateway_runner=None
    dispatch=SimpleNamespace(room_id='legacy-room',home_install_id='home',member_id='one',target_profile='default')
    sid=asyncio.run(_ensure_hosted_member_session(adapter,dispatch))
    assert asyncio.run(_ensure_hosted_member_session(adapter,dispatch))==sid
    assert authority.db.get_session(sid)['hidden']
    with pytest.raises(RuntimeError,match='Another group'):
        asyncio.run(_ensure_hosted_member_session(adapter,SimpleNamespace(**{**vars(dispatch),'member_id':'foreign'})))
    assert not check_remote_hosted_admission(authority,SessionRef('default',sid),{})
    assert end_idle_acp_sessions(authority)==0
    assert storage_source(authority.db,SessionSource(platform=Platform.LOCAL,chat_id='unadopted'),sid,'local')=='local'
    fixture=hosted_owner.__wrapped__(tmp_path,monkeypatch)
    instance=next(fixture)
    try:
        test_room_binding_exact_retry_terminal_history_and_unknown(instance)
    finally:
        try:next(fixture)
        except StopIteration:pass
