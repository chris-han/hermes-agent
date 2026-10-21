"""Real PostgreSQL native storage contracts; private fixture contains disposable DSNs only."""
import json
import os
from pathlib import Path
import uuid

import pytest


class PrivateDSN(str):
    def __repr__(self):
        return '<private test DSN>'


@pytest.fixture
def pg_settings():
    fixture = Path(__file__).resolve().parents[3] / '.artifacts/shared-pg-v1/candidate-pg-private.json'
    if not fixture.is_file():
        pytest.skip('disposable PostgreSQL fixture unavailable')
    return {key: PrivateDSN(value) for key,value in json.loads(fixture.read_text())['dsns'].items()}


def test_colliding_native_ids_are_scoped_and_restartable(tmp_path, pg_settings):
    from hermes_state import SessionDB
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope, provision_scope
    scopes = [SessionScope('tenant-a', 'workspace', str(uuid.uuid4())),
              SessionScope('tenant-b', 'workspace', str(uuid.uuid4()))]
    for n, scope in enumerate(scopes):
        source = tmp_path / str(n) / 'state.db'
        source.parent.mkdir()
        with SessionDB(source) as db:
            db.create_session('same-session', 'api_server')
            assert db.append_message('same-session', 'user', f'Hello 世界 {n}') == 1
            assert db.try_acquire_session_turn_lease('same-session', 'same-owner')
        provision_scope(pg_settings['semantier_migrate'], scope, app_role='semantier_app')
        with PostgreSQLSessionDB(pg_settings['semantier_migrate'], scope, home=source.parent) as db:
            receipt = db.import_sqlite_snapshot(source)
            assert receipt['tables']['messages'] == 1
            assert db.import_sqlite_snapshot(source)['idempotent']
            assert db.get_messages('same-session')[0]['id'] == 1
            assert db.get_messages('same-session')[0]['content'] == f'Hello 世界 {n}'
            assert not hasattr(db, 'db_path')
        with PostgreSQLSessionDB(pg_settings['semantier_app'], scope, home=source.parent) as db:
            assert db.get_messages('same-session')[0]['content'] == f'Hello 世界 {n}'
            db.release_session_turn_lease('same-session', 'same-owner')
            assert db.append_message('same-session', 'assistant', f'reply {n}') == 2


def test_native_pg_commit_guards_branch_compaction_and_search(tmp_path, pg_settings):
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope, provision_scope
    scope = SessionScope('tenant', 'workspace', str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'], scope, app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_app'], scope, home=tmp_path) as db:
        db.create_session('parent', 'api_server', model_config={'route': 'stable'})
        db.append_messages_batch('parent', [{'role': 'user', 'content': 'Hello world 世界中文'},
                                           {'role': 'assistant', 'content': 'answer'}])
        assert db.get_active_message_ids('parent') == [1, 2]
        assert db.try_acquire_session_turn_lease('parent', 'owner')
        with pytest.raises(Exception):
            db.append_message('parent', 'user', 'forbidden', turn_lease_holder='other')
        assert db.message_count('parent') == 2
        db.release_session_turn_lease('parent', 'owner')
        for query in ['Hello', 'Hel*', '世界', '中文', '世界*']:
            assert [r['id'] for r in db.search_messages(query)] == [1]
        branch = db.branch_session('parent', 'child', title='Branch', branch_point_message_id='2',
                                   branch_point_sequence=2, idempotency_key='key', request_hash='hash')
        assert branch['message_count'] == 2
        assert db.get_messages('child')[0]['id'] != 1
        assert db.get_branch_records()[0]['branch_point_message_id'] == '2'
        db.update_token_counts('parent', input_tokens=10, output_tokens=3, model='model', api_call_count=1)
        assert db.get_session('parent')['input_tokens'] == 10
        assert db.try_acquire_compression_lock('parent', 'compressor')
        db.publish_compression_child(parent_session_id='parent', child_session_id='compressed', source='api_server',
            messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='compressor')
        assert db.get_compression_tip('parent') == 'compressed'
        assert db.get_session('parent')['end_reason'] == 'compression'
        assert db.get_messages_as_conversation('compressed')[0]['content'] == 'summary'
        assert db.export_session('parent')['messages'][0]['id'] == 1


def test_pg_reflush_cas_and_multimodal_identity(tmp_path,pg_settings):
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope, provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        messages=[{'role':'user','content':[{'type':'text','text':'first'}]},
                  {'role':'assistant','content':'initial'}]
        assert db.append_messages_batch('session',messages)==2
        assert db.append_messages_batch('session',messages)==0
        assert db.get_messages('session')[0]['content']==messages[0]['content']
        assert messages[1].get('_db_row_snapshot')
        stale=[dict(message) for message in messages]
        messages[1]['content']='winner'
        assert db.append_messages_batch('session',messages)==0
        stale[1]['content']='stale loser'
        assert db.append_messages_batch('session',stale)==0
        assert db.get_messages('session')[1]['content']=='winner'


def test_api_response_and_run_idempotency_are_durable_in_pg(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope, provision_scope
    from gateway.platforms.api_server_response_store import ResponseStore
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted API opened SQLite'))
    store=ResponseStore(db_path=str(tmp_path/'response_store.db'),max_size=1)
    assert store.bind_request_key('idem:key','fingerprint',model='model')
    assert not store.bind_request_key('idem:key','conflict')
    store.put('response-1',{'value':'中文'})
    store.set_conversation('chat','response-1')
    assert store.get('response-1')=={'value':'中文'}
    assert store.output_index('idem:key','a')==0
    assert store.output_index('idem:key','b')==1
    store.close()
    store=ResponseStore(db_path=str(tmp_path/'response_store.db'),max_size=1)
    assert store.request_model('idem:key','different')=='model'
    assert store.get_conversation('chat')=='response-1'
    store.close()
    runs=RunIdempotencyStore(str(tmp_path/'runs_idempotency.db'))
    assert runs.durable
    assert runs.reserve('scope','key','fp','run',{'status':'queued'})[0]=='created'
    runs.close()
    runs=RunIdempotencyStore(str(tmp_path/'runs_idempotency.db'))
    assert runs.reserve('scope','key','fp','other',{'status':'queued'})[0]=='reused'
    assert runs.lookup('scope','key','different')[0]=='conflict'
    assert runs.lookup('other-scope','key','fp')[0]=='missing'
    runs.update_status('run',{'status':'completed'})
    assert runs.status_for_run('scope','run')['status']=={'status':'completed'}
    runs.close()


def test_cron_pg_terminal_immutability_and_incident_ack(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope, provision_scope
    from cron import executions,incidents
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(executions,'EXECUTIONS_FILE',tmp_path/'cron'/'executions.db')
    monkeypatch.setattr(incidents,'EXECUTIONS_FILE',tmp_path/'cron'/'executions.db')
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted cron opened SQLite'))
    record=executions.create_execution('same-job',source='cron')
    assert record['status']=='claimed'
    assert executions.mark_execution_running(record['id'])['status']=='running'
    assert executions.finish_execution(record['id'],success=True)['status']=='completed'
    assert executions.finish_execution(record['id'],success=False) is None
    assert executions.latest_execution('same-job')['status']=='completed'
    identity,new=incidents.upsert_incident('same-job','timeout')
    assert new
    assert incidents.ack_incident(identity)
    assert not incidents.upsert_incident('same-job','timeout')[1]
    assert incidents.get_incident(identity)['state']=='closed'


def test_kanban_pg_canonical_claim_completion_and_parent_fence(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope, provision_scope
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted Kanban opened SQLite'))
    conn=connect(tmp_path/'kanban.db')
    try:
        parent=kb.create_task(conn,title='Parent',workspace_kind='scratch',completion_contract='local-only')
        child=kb.create_task(conn,title='Child',workspace_kind='scratch',parents=[parent],completion_contract='local-only')
        assert kb.get_task(conn,child).status=='todo'
        assert not kb.complete_task(conn,child,result='cannot bypass parent',fire_lifecycle_hook=False)
        assert kb.complete_task(conn,parent,result='parent done',fire_lifecycle_hook=False)
        assert kb.get_task(conn,parent).status=='done'
        assert kb.complete_task(conn,child,result='child done',fire_lifecycle_hook=False)
        assert kb.list_events(conn,child)
        assert kb.parent_ids(conn,child)==[parent]
    finally:conn.close()


def test_delivery_pg_retains_unknown_remote_ownership(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,provision_scope
    from gateway import delivery_ledger as ledger
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(ledger,'get_process_hermes_home',lambda:tmp_path)
    import sqlite3,time
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted delivery ledger opened SQLite'))
    ledger.record_obligation(obligation_id='known',session_key='chat',platform='telegram',chat_id='chat',thread_id=None,content='known')
    ledger.mark_delivered('known')
    with ledger._transaction() as conn:
        conn.execute("INSERT INTO delivery_obligations(obligation_id,session_key,platform,chat_id,content,state,created_at,updated_at,owner_pid,owner_started_at) VALUES('unknown','chat','telegram','chat','do not resend','attempting',%s,%s,999999999,1)",(time.time(),time.time()))
    assert ledger.sweep_recoverable(deliverable_platforms={'telegram'})==[]
    with ledger._transaction() as conn:
        row=conn.execute("SELECT state,attempts FROM delivery_obligations WHERE obligation_id='unknown'").fetchone()
        assert tuple(row)==('attempting',0)


def test_hosted_rooms_pg_log_driver_and_history_guards(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,provision_scope
    import psycopg
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted room opened SQLite'))
    from gateway import hosted_rooms as rooms,hosted_room_driver as driver
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    from gateway.hosted_rooms_common import open_sqlite,transaction
    path=tmp_path/'shared-state.db'
    room=rooms.create_room(path,room_id='room',name='中文 Room',members=[{'profile':'ops','handle':'ops'}],authority_gateway_id='gateway-a',now=10)
    assert room['authority_epoch']==1
    event=rooms.append_event(path,room_id='room',event_id='e1',kind='message.user',actor={'kind':'user','id':'user'},payload={'text':'中文 history','thread_id':'thread'},authority_gateway_id='gateway-a',authority_epoch=1,now=11)
    assert event['seq']==1
    assert rooms.probe_hosted_room(path,room_id='room')
    assert rooms.read_events(path,room_id='room')['events'][0]['payload']['text']=='中文 history'
    checkpoint=HostedRoomPolicyCheckpoint(path)
    assert checkpoint.sync(room_id='room',latest_seq=1)==1
    lease=driver.acquire_lease(path,room_id='room',gateway_id='gateway-a',authority_epoch=1,process_generation='process',ttl_seconds=30,clock=lambda:20)
    assert lease.lease_generation==1
    with pytest.raises(driver.LeaseHeldError):
        driver.acquire_lease(path,room_id='room',gateway_id='gateway-a',authority_epoch=1,process_generation='other',ttl_seconds=30,clock=lambda:21)
    identity=driver.TaskIdentity(room_id='room',task_id='task',thread_id='thread',turn_id='turn')
    admitted=driver.admit_task(path,identity,payload={'target_profile':'ops','prompt':'bounded no-dispatch','source_event_seq':1},clock=lambda:21)
    assert admitted['status']=='queued'
    from gateway import hosted_room_work_records as work
    record=work.capture(path,room_id='room',local_gateway_id='gateway-a')
    assert work.validate(record)['tasks'][0]['task_id']=='task'
    with transaction(open_sqlite,path,immediate=True) as conn:
        conn.execute("UPDATE hosted_rooms SET authority_gateway_id='gateway-b',authority_epoch=2 WHERE room_id='room'")
    with open_sqlite(path) as conn:
        assert conn.execute("SELECT disposition FROM hosted_room_work_records_source WHERE room_id='room'").fetchone()[0]=='historical'
    with pytest.raises(psycopg.errors.CheckViolation,match='historical work record is immutable'):
        with transaction(open_sqlite,path,immediate=True) as conn:
            conn.execute("DELETE FROM hosted_room_work_records_source WHERE room_id='room'")
    with pytest.raises(psycopg.errors.CheckViolation,match='invalid work evidence is immutable'):
        with transaction(open_sqlite,path,immediate=True) as conn:
            conn.execute("INSERT INTO hosted_room_work_records_invalid(source_table,room_id,revision,digest,record_json,disposition) VALUES ('legacy','room',1,'digest','{}','invalid')")


def test_pg_native_api_route_config_rewind_and_cooldown(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server',session_key='route',model_config={'_branched_from':'retained'})
        entry=json.dumps({'session_id':'session','origin':{'platform':'api_server'}})
        assert db.admit_native_api_route('route',entry,scope='scope')==entry
        assert db.admit_native_api_route('route',entry,scope='scope')==entry
        with pytest.raises(ValueError,match='REBINDING'):
            db.admit_native_api_route('route',json.dumps({'session_id':'session','origin':{}}),scope='scope')
        db.update_session_model('session','model',provider='provider',base_url='https://route.invalid')
        config=json.loads(db.get_session('session')['model_config'])
        assert config['_branched_from']=='retained'
        assert config['gateway_runtime']['provider']=='provider'
        assert db.update_session_cwd('session','/workspace',git_branch='branch')==1
        assert db.publish_session_git_metadata('session','/workspace',1,git_branch='new')
        assert not db.publish_session_git_metadata('session','/old',1,git_branch='stale')
        db.record_compression_failure_cooldown('session',9999999999,'failed')
        db.record_compression_failure_cooldown('session',9999999998,'later')
        assert db.get_compression_failure_cooldown('session')['cooldown_until']==9999999999
        assert db.get_compression_failure_cooldown('session')['error']=='later'
        db.set_compression_fallback_streak('session',2)
        assert db.get_compression_fallback_streak('session')==2
        db.append_messages_batch('session',[{'role':'user','content':'first'},{'role':'assistant','content':'answer'},{'role':'user','content':'retry'}])
        result=db.rewind_user_turn('session',-1,require_retryable=True)
        assert result.live_text=='retry'
        assert result.turns_undone==1
        assert len(db.get_active_message_ids('session'))==2


def test_pg_inplace_compaction_preserves_unseen_rows_and_payloads(tmp_path,pg_settings,monkeypatch):
    import copy
    from hermes_state import SessionDB
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source=tmp_path/'state.db'
    with SessionDB(source) as original:
        original.create_session('session','api_server',model_config={'retained':'value'})
        original.append_messages_batch('session',[
            {'role':'user','content':'held first','timestamp':1},
            {'role':'assistant','content':'unseen gap','api_content':'exact wire','timestamp':2},
            {'role':'user','content':'held last','timestamp':3},
            {'role':'assistant','content':[{'type':'text','text':'late 中文'}],'timestamp':4},
        ])
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=tmp_path) as imported:imported.import_sqlite_snapshot(source)
    original=SessionDB(source)
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    compacted=[{'role':'user','content':'summary','timestamp':10,'message_uid':uuid.uuid4().hex}]
    expected=original.archive_and_compact('session',copy.deepcopy(compacted),watermark=3,covered_ids=[1,3],model_config_patch={'patch':True})
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        assert db.archive_and_compact('session',copy.deepcopy(compacted),watermark=3,covered_ids=[1,3],model_config_patch={'patch':True})==expected==3
        actual=db.get_messages('session')
        retained=original.get_messages('session')
        for row,want in zip(actual,retained):
            for field in ('id','role','content','timestamp','message_uid','api_content','active','compacted'):
                assert row[field]==want[field],field
        assert json.loads(db.get_session('session')['model_config'])==json.loads(original.get_session('session')['model_config'])
        assert [(r['id'],r['active'],r['compacted']) for r in db.get_messages('session',include_inactive=True)]==[(r['id'],r['active'],r['compacted']) for r in original.get_messages('session',include_inactive=True)]
    original.close()


def test_pg_native_session_store_never_uses_json_authority(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    from gateway.config import GatewayConfig,Platform
    from gateway.session import SessionStore,SessionSource
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server',session_key='route')
        routes=SessionStore(tmp_path/'sessions',GatewayConfig(),session_db=db)
        monkeypatch.setattr(routes,'_import_legacy_sessions_json',lambda *a:pytest.fail('Hosted routing read sessions.json'))
        monkeypatch.setattr(routes,'_save_sessions_json',lambda *a:pytest.fail('Hosted routing wrote sessions.json'))
        entry=routes.admit_native_api_session(session_key='route',session_id='session',source=SessionSource(platform=Platform.API_SERVER,chat_id='chat',user_id='user',workspace_owner_id='workspace'))
        assert entry.session_id=='session'
        routes._save()
        assert not (tmp_path/'sessions'/'sessions.json').exists()
        def unavailable(*args,**kwargs):raise RuntimeError('database unavailable')
        monkeypatch.setattr(db,'save_gateway_routing_entry',unavailable)
        with pytest.raises(RuntimeError,match='database unavailable'):routes._save_entry('route')


def test_pg_usage_auxiliary_and_portable_history_flags(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server',model='requested')
        db.update_token_counts('session',input_tokens=10,output_tokens=2,model='fallback',billing_provider='provider',api_call_count=1)
        db.record_auxiliary_usage('session','compression',model='aux-model',billing_provider='aux',input_tokens=5)
        assert db.get_session('session')['input_tokens']==10
        assert db.get_session('session')['model']=='fallback'
        assert db.get_recent_session_model_route('session')['billing_provider']=='provider'
        payload=[{'id':'restored','source':'api_server','session_key':'untrusted-route','origin_json':'untrusted-origin','model_config':{'retained':True},'messages':[
            {'role':'user','content':'archived 中文','active':0,'compacted':1,'timestamp':1},
            {'role':'assistant','content':'live','active':1,'timestamp':2}]}]
        receipt=db.import_sessions(payload)
        assert receipt['ok'] and receipt['imported']==1
        assert db.get_session('restored')['session_key'] is None
        assert db.get_session('restored')['origin_json'] is None
        assert db.get_session('restored')['message_count']==1
        assert [r['content'] for r in db.get_messages('restored')]==['live']
        assert [r['content'] for r in db.get_messages('restored',include_inactive=True)]==['archived 中文','live']
        assert db.import_sessions(payload)['skipped']==1


def test_pg_delete_retirement_cascades_and_protects_live_work(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    from hermes_state_errors import SessionActiveWriteGuardError
    from hermes_state_runtime import RuntimeStoreError
    from hermes_state_terminal import ADMISSION_PREFIX,RESULT_PREFIX
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('parent','api_server')
        db.create_session('delegate','tool',parent_session_id='parent',model_config={'_delegate_from':'parent'})
        db.create_session('branch','api_server',parent_session_id='parent',model_config={'_branched_from':'parent'})
        db.append_message('parent','user','private history')
        db.append_message('delegate','assistant','private delegate')
        with db._transaction() as conn:
            db._insert(conn,'semantier_session_branches',dict(child_session_id='branch',source_session_id='parent',branch_point_message_id='1',branch_point_sequence=1,title='Branch',idempotency_key='branch-key',request_hash='branch-hash'))
        db.save_gateway_routing_entry('route',json.dumps({'session_id':'parent'}))
        assert db.try_acquire_session_turn_lease('delegate','busy-owner')
        with pytest.raises(SessionActiveWriteGuardError):db.delete_session('parent',exclude_active_write_guards=True)
        db.release_session_turn_lease('delegate','busy-owner')
        assert not db.delete_session('parent',expected_delete_ids=['parent'])
        with db._transaction() as conn:
            db._insert(conn,'session_admissions',dict(admission_id='admission',request_id='request',principal_id='principal',target_session_id='parent',lineage_json='[]',payload_json='{}',payload_digest='digest',intent='queue',status='terminal'))
        db.set_meta(RESULT_PREFIX+'admission',json.dumps({'result':{'final_response':'exact retry','messages':[{'content':'private history'}],'last_reasoning':'private'}}))
        assert set(db.get_session_delete_targets('parent'))=={'parent','delegate'}
        assert db.delete_session('parent',expected_delete_ids=['parent','delegate'],exclude_active_write_guards=True)
        assert db.get_session('parent') is None and db.get_session('delegate') is None
        assert db.get_session('branch')['parent_session_id'] is None
        assert db.get_branch_records()[0]['source_session_id']=='parent'
        assert db.load_gateway_routing_entries()=={}
        result=json.loads(db.get_meta(RESULT_PREFIX+'admission'))['result']
        assert result=={'final_response':'exact retry','messages':[]}
        assert json.loads(db.get_meta(ADMISSION_PREFIX+'admission'))['payload_json']=='{}'
        with pytest.raises(RuntimeStoreError):db.create_session('parent','api_server')
        assert db.get_session('parent') is None


def test_pg_native_reprovision_preserves_foreign_owner_acls(tmp_path,pg_settings):
    import psycopg
    from psycopg import sql
    from hermes_state_postgresql import SessionScope,provision_scope,postgres_owned_tables
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    assert 'hosted_room_legacy_imports' in postgres_owned_tables()
    with psycopg.connect(pg_settings['semantier_migrate']) as conn:
        conn.execute(sql.SQL('CREATE TABLE {}(source TEXT PRIMARY KEY)').format(sql.Identifier(scope.schema,'foreign_import_marker')))
        conn.execute(sql.SQL('GRANT SELECT ON {} TO semantier_app').format(sql.Identifier(scope.schema,'foreign_import_marker')))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    for table in ('foreign_import_marker','scope_binding','migration_receipts','schema_version','hosted_room_legacy_imports'):
        with psycopg.connect(pg_settings['semantier_app']) as conn:
            row=conn.execute("SELECT has_table_privilege(current_user,%s,'INSERT,UPDATE,DELETE')",(scope.schema+'.'+table,)).fetchone()
            assert row[0] is False,table
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with psycopg.connect(pg_settings['semantier_app']) as conn:
                conn.execute(sql.SQL('DELETE FROM {}').format(sql.Identifier(scope.schema,table)))


def test_pg_native_readiness_rejects_disabled_history_guard(tmp_path,pg_settings):
    import psycopg
    from psycopg import sql
    from hermes_state_postgresql import SessionScope,provision_scope,require_scope_ready
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with psycopg.connect(pg_settings['semantier_app']) as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        result=require_scope_ready(conn,scope)
        assert result['enabled_triggers']==15
        assert result['native_schema_version']==1
    with psycopg.connect(pg_settings['semantier_migrate']) as conn:
        conn.execute(sql.SQL('ALTER TABLE {} DISABLE TRIGGER trg_work_invalid_insert').format(sql.Identifier(scope.schema,'hosted_room_work_records_invalid')))
        with pytest.raises(ValueError,match='missing or disabled'):require_scope_ready(conn,scope)
        conn.rollback()
        conn.execute(sql.SQL('DROP INDEX {}').format(sql.Identifier(scope.schema,'idx_sessions_title_unique')))
        with pytest.raises(ValueError,match='indexes missing'):require_scope_ready(conn,scope)
        conn.rollback()


def test_pg_native_titles_preserve_provenance_identity_and_conflicts(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('legacy','desktop',title='Legacy user title')
        assert db.get_session('legacy')['title_source'] is None
        assert not db.set_auto_title('legacy','Autogenerated',source='llm')
        assert db.get_session_title('legacy')=='Legacy user title'
        db.create_session('bot','desktop')
        assert db.set_session_title('bot','Bot Chat')
        assert db.set_session_hidden('bot',True)
        with pytest.raises(ValueError,match='canonical Bot Chat'):db.set_session_title('bot','Rename')
        assert not db.set_auto_title('bot','Automatic rename',source='llm')
        db.create_session('other','desktop')
        with pytest.raises(ValueError,match='already in use'):db.set_session_title('other','Legacy user title')
        db.create_session('ghost','desktop')
        assert db.set_session_title('ghost','Reusable')
        db.end_session('ghost','agent_close')
        assert db.set_session_title('other','Reusable')
        assert db.get_session_title('ghost') is None
        assert db.get_next_title_in_lineage('Reusable')=='Reusable #2'
        assert db.set_session_archived('bot',True)
        db.create_session('replacement','desktop')
        assert db.set_session_title('replacement','Bot Chat')
        assert db.get_session_title('bot') is None
        db.create_session('numbered','desktop')
        assert db.set_session_title('numbered','Bot Chat #2')
        assert db.resolve_session_by_title('Bot Chat')=='replacement'
        with pytest.raises(ValueError,match='invalid automatic title source'):db.set_auto_title('other','bad',source='unknown')


def test_pg_search_matches_native_pinned_en_zh_cjk_and_substrings(tmp_path,pg_settings,monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source=tmp_path/'search-source.db'
    cases=[('needle',{}),('needle',{'sort':'oldest'}),('needle',{'sort':'newest'}),
        ('Hello world',{}),('"Hello world"',{}),('Hel*',{}),('missing Hello',{}),
        ('广西',{}),('广西 OR 阳朔',{}),('桂林',{}),('桂',{}),('youer',{}),
        ('needle',{'role_filter':['tool']}),('needle',{'source_filter':['api_server']}),
        ('needle',{'exclude_sources':['api_server']}),('needle',{'before_ts':1500}),
        ('needle',{'include_inactive':True}),('needle',{'limit':1,'offset':1}),
        ('needle NOT padding',{}),('needle OR Hello',{}),('needle AND Hello',{}),
        ('广西 AND 阳朔',{}),('yo',{}),('cafe',{}),('my-app.config.ts',{}),('P2.2',{})]
    with SessionDB(source) as native:
        native.create_session('first','api_server')
        native.create_session('second','cli')
        native._write_sql('UPDATE sessions SET started_at=? WHERE id=?',(1000,'first'))
        native._write_sql('UPDATE sessions SET started_at=? WHERE id=?',(2000,'second'))
        native.append_messages_batch('first',[
            {'role':'user','content':'needle','timestamp':1},
            {'role':'assistant','content':'needle '+'padding '*50,'timestamp':2},
            {'role':'user','content':'Hello world','timestamp':3},
            {'role':'assistant','content':'广西桂林漓江','timestamp':4},
            {'role':'user','content':'修改youer服务端','timestamp':5},
            {'role':'tool','content':'needle tool result','timestamp':6},
            {'role':'assistant','content':'needle hidden','timestamp':7,'display_kind':'hidden'},
            {'role':'user','content':'needle inactive','timestamp':8},
        ])
        native.append_messages_batch('second',[{'role':'user','content':'桂林阳朔 needle','timestamp':9},
            {'role':'assistant','content':'Café my-app.config.ts P2.2','timestamp':10}])
        native._write_sql('UPDATE messages SET active=0,compacted=0 WHERE id=8')
        expected=[native.search_messages(query,fields=['id'],**options) for query,options in cases]
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=tmp_path) as db:db.import_sqlite_snapshot(source)
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted search opened SQLite'))
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        for (query,options),want in zip(cases,expected):
            assert db.search_messages(query,fields=['id'],**options)==want,(query,options,want)


def test_pg_internal_result_projection_retries_without_duplicates(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        row_id=db.append_internal_result('session','Committed 中文',result_id='continuation:result',result_hash='hash')
        assert db.append_internal_result('session','Committed 中文',result_id='continuation:result',result_hash='hash')==row_id
        with pytest.raises(ValueError,match='INTERNAL_RESULT_IDENTITY_CONFLICT'):
            db.append_internal_result('session','Different',result_id='continuation:result',result_hash='hash')
        assert db.message_count('session')==1
        assert db.try_acquire_session_turn_lease('session','owner')
        with pytest.raises(Exception):db.append_internal_result('session','Concurrent',result_id='next',result_hash='other')
        db.release_session_turn_lease('session','owner')
        assert db.get_messages('session')[0]['content']=='Committed 中文'
        assert db.get_messages('session')[0]['display_metadata']['internal_result_hash']=='hash'


def test_pg_compression_handoff_matches_native_automatic_recovery_and_tail(tmp_path,pg_settings,monkeypatch):
    import copy
    from hermes_state import SessionDB
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source=tmp_path/'compression-source.db'
    native=SessionDB(source)
    native.create_session('parent','api_server',model_config={'onboarding_run':{'run':'stable'},'discard':'old'},
        user_id='user',session_key='route',chat_id='chat',profile_name='profile',cwd='/workspace',
        system_prompt='Parent prompt',origin_json=json.dumps({'platform':'api_server'}))
    native.update_session_tool_names('parent',['one','two'])
    native.append_messages_batch('parent',[
        {'role':'user','content':'carried original','timestamp':1,'message_uid':'original'},
        {'role':'assistant','content':'answer','timestamp':2},
        {'role':'tool','content':[{'type':'text','text':'concurrent 中文'}],'timestamp':3,'api_content':'exact wire'},
        {'role':'assistant','content':'own flush already covered','timestamp':4},
    ])
    native._write_sql('UPDATE sessions SET pinned=1,archived=1,auto_archived=1 WHERE id=?',('parent',))
    assert native.try_acquire_compression_lock('parent','compressor')
    native.end_session('parent','ws_disconnect')
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=tmp_path) as imported:imported.import_sqlite_snapshot(source)
    messages=[{'role':'user','content':'summary','timestamp':100,'message_uid':'summary'},
              {'role':'user','content':'carried original','message_uid':'original'}]
    arguments=dict(parent_session_id='parent',child_session_id='child',source='api_server',
        model='new-model',model_config={'new':'value'},system_prompt='New prompt',
        compression_lock_holder='compressor',watermark=2,watermark_ceiling=3)
    native.publish_compression_child(messages=copy.deepcopy(messages),**arguments)
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.publish_compression_child(messages=copy.deepcopy(messages),**arguments)
        for session_id in ('parent','child'):
            actual=db.get_session(session_id);expected=native.get_session(session_id)
            for field in ('source','created_source','model','model_config','system_prompt','system_prompt_hash','tool_names',
                    'profile_name','user_id','session_key','chat_id','origin_json','cwd','parent_session_id','archived','auto_archived','pinned','end_reason','message_count','tool_call_count'):
                assert actual[field]==expected[field],(session_id,field)
        for actual,expected in zip(db.get_messages('child'),native.get_messages('child')):
            for field in ('id','role','content','timestamp','message_uid','api_content','active','compacted'):
                assert actual[field]==expected[field],field
        assert db.get_compression_lock_holder('parent')==native.get_compression_lock_holder('parent')=='compressor'
        assert db.get_compression_tip('parent')=='child'
        assert db.find_live_compression_child('parent')['id']=='child'
    native.close()


def test_pg_replace_preserves_kept_prefix_and_destructive_modes(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        db.append_messages_batch('session',[{'role':'user','content':'keep'},{'role':'assistant','content':'drop'}])
        first=db.get_messages('session')[0]
        replacement=[{'role':'user','content':'keep'},{'role':'assistant','content':'new'}]
        db.replace_messages('session',replacement,archive_dropped=True)
        assert [r['content'] for r in db.get_messages('session')]==['keep','new']
        assert db.get_messages('session')[0]['id']==first['id']
        assert len(db.get_messages('session',include_inactive=True))==3
        assert replacement[0]['_row_id']==first['id']
        db.replace_messages('session',[{'role':'user','content':'fresh'}],active_only=True)
        assert [r['content'] for r in db.get_messages('session',include_inactive=True)]==['drop','fresh']
        db.replace_messages('session',[{'role':'assistant','content':'only'}])
        assert [r['content'] for r in db.get_messages('session',include_inactive=True)]==['only']
        assert db.get_session('session')['message_count']==1


def test_pg_shadowed_checkpoints_and_inherited_branch_markers(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        db.append_messages_batch('session',[
            {'role':'assistant','content':'older','codex_reasoning_items':[{'type':'compaction','encrypted_content':'old'},{'type':'reasoning','id':'keep'}]},
            {'role':'assistant','content':'newer','codex_reasoning_items':[{'type':'compaction','encrypted_content':'new'}]},
        ])
        older,newer=db.get_messages('session')
        assert json.loads(older['codex_reasoning_items'])==[{'type':'reasoning','id':'keep'}]
        assert json.loads(newer['codex_reasoning_items'])==[{'type':'compaction','encrypted_content':'new'}]
        db.create_session('old-root','api_server')
        db.create_session('branch','api_server',parent_session_id='old-root',model_config={'_branched_from':'old-root'})
        db.end_session('branch','compression')
        db.create_session('continuation','api_server',parent_session_id='branch',model_config={'_branched_from':'old-root'})
        assert db.get_compression_tip('branch')=='branch'
        assert db.get_compression_lineage('continuation')==['branch','continuation']


def test_pg_turn_lease_wait_honors_writer_contention_budget(tmp_path,pg_settings,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import psycopg,time
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        blocker=psycopg.connect(pg_settings['semantier_app'])
        blocker.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(scope.schema+'\x1fcanonical-native-writer',))
        notices=[]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(db.acquire_session_turn_lease,'session','holder',wait_seconds=.15,acquire_patience_s=.02,
                poll_interval_seconds=.01,on_contended=lambda:notices.append('contended'))
            try:
                assert future.result(timeout=1) is False
                assert notices
            finally:
                blocker.rollback();blocker.close()


def test_pg_native_runtime_admission_worker_and_restart_receipts(tmp_path,pg_settings,monkeypatch):
    import hermes_state_runtime as rt
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        epoch=rt.begin_runtime_epoch(db,instance_id='first-boot')
        arguments=dict(epoch=epoch,principal_id='principal',session_id='session',request_id='request',payload={'text':'bounded persisted work'})
        admitted=rt.admit_session_input(db,**arguments)
        assert rt.admit_session_input(db,**arguments)==admitted
        with pytest.raises(rt.RuntimeStoreError,match='admission_conflict'):
            rt.admit_session_input(db,**(arguments|{'payload':{'text':'changed'}}))
        claimed=rt.claim_session_input(db,epoch=epoch,session_id='session')
        assignment=dict(epoch=epoch,execution_id='worker',session_id='session',generation=claimed['generation'])
        rt.register_worker_execution(db,**assignment,kind='compute',adoption_secret='private-fixture')
        assert rt.mutate_worker_execution(db,**assignment,sequence=1,operation='turn.acquire',payload={'holder':'worker','ttl_seconds':300})=={'value':True}
        write=dict(sequence=2,operation='transcript.append',payload={'turn_lease_holder':'worker','messages':[{'role':'assistant','content':'durable 中文'}]})
        receipt=rt.mutate_worker_execution(db,**assignment,**write)
        assert rt.mutate_worker_execution(db,**assignment,**write)==receipt
        rt.mutate_worker_execution(db,**assignment,sequence=3,operation='usage.main',payload={'input_tokens':5,'model':'model'})
        rt.mutate_worker_execution(db,**assignment,sequence=4,operation='turn.release',payload={'holder':'worker'})
        rt.mutate_worker_execution(db,**assignment,sequence=5,operation='execution.finish',payload={})
        assert rt.settle_session_input(db,epoch=epoch,admission_id=admitted['admission_id'],generation=claimed['generation'],outcome='completed',result={'final_response':'durable 中文'})['status']=='terminal'
        pending=rt.admit_session_input(db,**(arguments|{'request_id':'pending'}))
        lost=rt.claim_session_input(db,epoch=epoch,session_id='session')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        replacement=rt.begin_runtime_epoch(db,instance_id='replacement-boot')
        assert replacement==epoch+1
        assert rt.recover_session_inputs(db,epoch=replacement)>=1
        unknown=rt.get_session_admission(db,admission_id=pending['admission_id'])
        assert unknown['status']=='unknown'
        with pytest.raises(rt.RuntimeStoreError,match='unknown_execution'):rt.claim_session_input(db,epoch=replacement,session_id='session')
        assert rt.resolve_unknown_session_input(db,epoch=replacement,admission_id=pending['admission_id'],generation=lost['generation'])['outcome']=='interrupted'
        assert db.get_messages('session')[0]['content']=='durable 中文'
        assert db.get_session('session')['input_tokens']==5
        with pytest.raises(rt.RuntimeStoreError,match='stale_epoch'):rt.admit_session_input(db,**(arguments|{'request_id':'stale'}))


def test_pg_lease_reacquisition_keeps_existing_expiry(tmp_path,pg_settings,monkeypatch):
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('session','api_server')
        assert db.try_acquire_session_turn_lease('session','owner',ttl_seconds=60)
        first=db.session_turn_lease_expires_at('session','owner')
        assert db.try_acquire_session_turn_lease('session','owner',ttl_seconds=600)
        assert db.session_turn_lease_expires_at('session','owner')==first
        assert not db.try_acquire_session_turn_lease('session','other',ttl_seconds=600)


def test_pg_snapshot_preserves_surviving_branch_reference_to_deleted_source(tmp_path,pg_settings):
    from hermes_state import SessionDB
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source=tmp_path/'branch-source.db'
    with SessionDB(source) as db:
        db.create_session('parent','api_server')
        db.append_message('parent','user','original')
        db.create_session('branch','api_server',parent_session_id='parent',model_config={'_branched_from':'parent'})
        db._write_sql('CREATE TABLE semantier_session_branches(child_session_id TEXT PRIMARY KEY,source_session_id TEXT NOT NULL,branch_point_message_id TEXT NOT NULL,branch_point_sequence INTEGER NOT NULL,title TEXT NOT NULL,idempotency_key TEXT NOT NULL,request_hash TEXT NOT NULL)')
        db._write_sql('INSERT INTO semantier_session_branches VALUES(?,?,?,?,?,?,?)',('branch','parent','1',1,'Retained branch','key','hash'))
        assert db.delete_session('parent')
        assert db.get_session('branch')['parent_session_id'] is None
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=tmp_path) as db:
        receipt=db.import_sqlite_snapshot(source)
        assert receipt['tables']['semantier_session_branches']==1
        assert db.import_sqlite_snapshot(source)['idempotent']
        assert db.get_branch_records()==[{'child_session_id':'branch','source_session_id':'parent','branch_point_message_id':'1','branch_point_sequence':1,'title':'Retained branch','idempotency_key':'key','request_hash':'hash'}]


def test_pg_real_gateway_authority_bootstrap_uses_scope_identity(tmp_path,pg_settings,monkeypatch):
    import asyncio,sqlite3
    from types import SimpleNamespace
    from hermes_state import AsyncSessionDB
    from hermes_state_backend import session_store_key
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    from gateway.session_authorities import SessionAuthorities
    from gateway.run_runtime import _build_profile_authority
    from gateway.session_cron import unbind_owner
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted gateway authority bootstrap opened SQLite'))
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        config=GatewayConfig()
        store=SessionStore(tmp_path/'sessions',config,session_db=db)
        runner=SimpleNamespace(config=config,session_store=store,_session_db=AsyncSessionDB(db),
            session_authorities=SessionAuthorities(tmp_path),session_runtime_descriptor={'instance_id':'boot'},adapters={})
        async def run():
            authority=await _build_profile_authority(runner,'default',tmp_path,register=True)
            try:
                assert authority.db is db
                assert authority.epoch==1
                assert runner.session_authorities.for_home(tmp_path) is authority
                assert store._local_authority_epochs[session_store_key(db)]==1
            finally:unbind_owner(authority)
        asyncio.run(run())
        assert not hasattr(db,'db_path')
        assert not (tmp_path/'state.db').exists()


@pytest.fixture
def native_pg_worker(tmp_path,pg_settings,monkeypatch):
    import sqlite3
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import begin_runtime_epoch,register_worker_execution,mutate_worker_execution
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(tmp_path):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted worker opened SQLite'))
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        db.create_session('owned','cli',cwd='/reserved',profile_name='fixture')
        epoch=begin_runtime_epoch(db,instance_id='worker-fixture')
        assignment=dict(epoch=epoch,execution_id='worker',session_id='owned',generation=0)
        register_worker_execution(db,**assignment,kind='compute',adoption_secret='private')
        store=RuntimeSessionStore(lambda method,**params:mutate_worker_execution(db,**params),assignment,tmp_path/'outbox')
        try:yield db,store
        finally:store.close()


def test_pg_worker_constructor_context_lifecycle_receipts(native_pg_worker):
    db,store=native_pg_worker
    assert store.create_session('owned','cli',model='requested',system_prompt='prefix',
        model_config={'max_iterations':5},cwd='/reserved',profile_name='fixture')=='owned'
    store.create_session('owned','cli',model='ignored',system_prompt='ignored',model_config={'max_iterations':10})
    assert db.get_session('owned')['model']=='requested'
    assert db.get_session_model_config_value('owned','max_iterations')==5
    before=db.get_session('owned')['runtime_revision']
    assert store.get_session('owned')['system_prompt']=='prefix'
    assert db.get_session('owned')['runtime_revision']==before
    store.update_system_prompt('owned','retained-prefix')
    store.patch_session_model_config('owned',{'_usage_anchor':{'tokens':23}})
    store.update_session_tool_names('owned',{'version':'v1','tools':['terminal']})
    assert store.get_session_model_config_value('owned','_usage_anchor')=={'tokens':23}
    assert store.get_session('owned')['tool_names']==json.dumps({'version':'v1','tools':['terminal']},sort_keys=True)
    assert store.session_lifecycle_statuses(['owned'])=={'owned':'empty'}
    store.append_messages_batch('owned',[{'role':'assistant','content':'done','finish_reason':'stop'}])
    assert store.session_lifecycle_statuses(['owned'])==db.session_lifecycle_statuses(['owned'])
    store.end_session('owned','agent_close')
    ended=db.get_session('owned')['ended_at']
    store.end_session('owned','late-close')
    assert db.get_session('owned')['ended_at']==ended
    assert store.finish()['status']=='terminal'


def test_pg_worker_compaction_and_rotation_keep_foreign_tail(native_pg_worker):
    db,store=native_pg_worker
    store.append_messages_batch('owned',[{'role':'user','content':'old'},{'role':'assistant','content':'carry'}])
    assert store.try_acquire_compression_lock('owned','holder')
    watermark=store.get_active_message_watermark('owned')
    db.append_messages_batch('owned',[{'role':'user','content':'foreign','api_content':' exact ','display_metadata':{'_accepted_input_id':'delivery'}}])
    assert store.archive_and_compact('owned',[{'role':'assistant','content':'summary'},{'role':'assistant','content':'carry'}],
        watermark=watermark,lock_holder='holder',tail_count=1,model_config_patch={'_proactive_prune_rearm_tokens':42})==3
    rows=store.get_messages_as_conversation('owned',include_row_ids=True)
    assert [row['content'] for row in rows]==['summary','carry','foreign']
    assert rows[-1]['api_content']==' exact '
    assert rows[-1]['display_metadata']=={'_accepted_input_id':'delivery'}
    assert store.get_session_model_config_value('owned','_proactive_prune_rearm_tokens')==42
    watermark=store.get_active_message_watermark('owned')
    store.publish_compression_child(parent_session_id='owned',child_session_id='child',source='cli',
        messages=[{'role':'assistant','content':'handoff'}],system_prompt='prefix',compression_lock_holder='holder',watermark=watermark)
    assert store.scope['session_id']=='child'
    store.append_messages_batch('child',[{'role':'assistant','content':'after'}])
    assert [row['content'] for row in store.get_messages_as_conversation('child')]==['handoff','after']
    assert store.get_compression_lineage('child')==['owned','child']


def test_pg_canonical_room_service_uses_real_native_home(native_pg_worker,tmp_path):
    import asyncio
    from types import SimpleNamespace
    from gateway import hosted_rooms as rooms
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    db,_=native_pg_worker
    authority=SimpleNamespace(db=db,profile_id=str(tmp_path))
    loop=asyncio.new_event_loop()
    try:
        service=CanonicalHostedRoomService(authority,loop)
        assert service.db_path is db
        assert service.root==tmp_path
        assert not service._link_load_error
        room=rooms.create_room(db,room_id='room',name='真实存储',members=[{'profile':'ops','handle':'ops'}],authority_gateway_id='gateway')
        assert room['authority_epoch']==1 and rooms.probe_hosted_room(db,room_id='room')
        rooms.append_event(db,room_id='room',event_id='event',kind='message.user',actor={'kind':'user','id':'user'},
            payload={'text':'retained 中文','thread_id':'thread'},authority_gateway_id='gateway',authority_epoch=1)
        assert HostedRoomPolicyCheckpoint(db).sync(room_id='room',latest_seq=1)==1
        attachments=HostedRoomAttachmentStore(db)
        assert attachments.root==tmp_path/'hosted-room-attachments'
        saved=attachments.put(room_id='room',upload_id='upload',kind='file',name='fixture.txt',mime='text/plain',data=b'retained attachment')
        assert saved['size']==19
        assert not (tmp_path/'state.db').exists() and not (tmp_path/'shared-state.db').exists()
    finally:loop.close()


def test_pg_legacy_message_decoding_and_later_append_match_native(tmp_path,pg_settings,monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source=tmp_path/'legacy.db'
    with SessionDB(source) as original:
        original.create_session('session','cli')
        original.append_message('session','assistant','legacy content')
        original._write_sql("UPDATE messages SET tool_calls='{broken',display_metadata='{broken' WHERE session_id='session'")
        expected=original.get_messages('session')
    scope=SessionScope('tenant','workspace',str(uuid.uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=tmp_path) as migration:
        migration.import_sqlite_snapshot(source)
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('Hosted legacy reader opened SQLite'))
    with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
        assert db.get_messages('session')==expected
        db.append_message('session','user','after migration')
        assert [message['content'] for message in db.get_messages('session')]==['legacy content','after migration']
        assert db.get_session('session')['message_count']==2


def test_pg_native_readers_media_and_kanban_owner_fences(native_pg_worker,tmp_path,monkeypatch):
    import os,psutil
    import hermes_state_runtime as rt
    from gateway.readiness import _probe_state_db
    from gateway.lifecycle_ledger import check_state_db_integrity
    from gateway.session_ingress_media import _held_media_paths,_history_mentions
    from gateway.session_kanban import _read_kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.kanban_owner_recovery import owner_reclaim_paused
    from agent.kanban_turn_recovery import worker_claim_is_live
    from gateway.run_idle_exit import _kanban_busy
    import psycopg
    db,_=native_pg_worker
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    assert _probe_state_db(tmp_path)=={'status':'ok'}
    assert check_state_db_integrity(home=tmp_path)=='ok'
    db.create_session('queue-owner','cli')
    reference={'path':str(tmp_path/'media'/'retained.png')}
    admitted=rt.admit_session_input(db,epoch=1,principal_id='principal',session_id='queue-owner',request_id='request',
        payload={'attachments_v1':{'media':[reference]},'api_turn_v1':{'media':[reference]}})
    with db._read_ctx() as conn:assert _held_media_paths(conn)=={reference['path']}
    claimed=rt.claim_session_input(db,epoch=1,session_id='queue-owner')
    rt.settle_session_input(db,epoch=1,admission_id=admitted['admission_id'],generation=claimed['generation'],outcome='completed')
    with db._read_ctx() as conn:assert _held_media_paths(conn)=={reference['path']}
    digest='a'*64
    history_path=str(tmp_path/digest/'api.png')
    db.append_message('queue-owner','user',[{'type':'text','text':history_path}])
    with db._read_ctx() as conn:
        assert _history_mentions(conn,[{'path':history_path,'sha256':digest}])=={history_path}
    path=tmp_path/'kanban.db'
    conn=connect(path)
    try:
        task_id=kb.create_task(conn,title='Retained owner',workspace_kind='scratch',completion_contract='local-only')
        task=kb.claim_task(conn,task_id)
        assert task is not None and task.current_run_id is not None
        kb._append_event(conn,task_id,'owner_admitted',{'db':'/must-never-open/source-state.db'},run_id=task.current_run_id)
        assert owner_reclaim_paused(conn,task_id)
        marker=dict(owner_home=str(tmp_path),storage_identity=list(db.storage_identity),session_id='queue-owner',request_id='request',
            pid=os.getpid(),birth=psutil.Process().create_time())
        kb._append_event(conn,task_id,'owner_admitted',marker,run_id=task.current_run_id)
        assert not owner_reclaim_paused(conn,task_id)
        kb._append_event(conn,task_id,'owner_admitted',{**marker,'storage_identity':['postgresql','foreign','workspace','source']},run_id=task.current_run_id)
        assert owner_reclaim_paused(conn,task_id)
        assert worker_claim_is_live(task_id=task_id,db_path=str(path),run_id=str(task.current_run_id),claim_lock=task.claim_lock,verify_pid=False)
        assert _kanban_busy()
    finally:conn.close()
    reader=_read_kanban(path)
    try:
        assert kb.get_task(reader,task_id).id==task_id
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):reader.execute('UPDATE tasks SET title=%s WHERE id=%s',('forbidden',task_id))
    finally:reader.close()
    assert not path.exists() and not (tmp_path/'state.db').exists()


def test_pg_canonical_mutation_receipts_metadata_rewind_and_delete(native_pg_worker):
    import hermes_state_runtime as rt
    db,_=native_pg_worker
    db.create_session('edit','cli')
    db.append_messages_batch('edit',[{'role':'user','content':'first'},{'role':'assistant','content':'reply'},{'role':'user','content':'again'}])
    args=dict(epoch=1,principal_id='principal',session_id='edit',request_id='sidebar',expected_revision=0,
        expected_generation=0,operation='sidebar',payload={'title':'中文 sidebar','pinned':True,'hidden':True,'unread':True})
    receipt=rt.mutate_runtime_session(db,**args)
    assert receipt['revision']==1 and receipt['pinned'] and receipt['hidden'] and receipt['unread']
    assert rt.mutate_runtime_session(db,**args)==receipt
    with pytest.raises(rt.RuntimeStoreError,match='revision_conflict'):
        rt.mutate_runtime_session(db,**(args|{'request_id':'stale'}))
    target=db.get_messages('edit')[-1]['id']
    rewind=dict(args,request_id='rewind',expected_revision=1,operation='rewind',payload={'target_message_id':target})
    rewound=rt.mutate_runtime_session(db,**rewind)
    assert rewound['rewound_count']==1 and rewound['target_message']['content']=='again'
    assert [row['content'] for row in db.get_messages('edit')]==['first','reply']
    assert rt.mutate_runtime_session(db,**rewind)==rewound
    delete=dict(args,request_id='delete',expected_revision=rewound['revision'],expected_generation=rewound['execution_generation'],operation='delete',payload={})
    deleted=rt.mutate_runtime_session(db,**delete)
    assert deleted['deleted_ids']==['edit'] and db.get_session('edit') is None
    assert rt.mutate_runtime_session(db,**delete)==deleted


def test_pg_canonical_local_branch_reset_and_history_import(native_pg_worker,tmp_path):
    from dataclasses import asdict
    from gateway.config import Platform
    from gateway.session import SessionEntry,SessionSource,build_session_key
    from gateway.session_policy import build_policy
    from gateway.session_lifecycle import _now
    from gateway.session_contract import Principal
    from hermes_state_keys import local_identity
    from hermes_state_local import commit_local_session,local_receipt
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_mutation_binding import import_history_control
    import hermes_state_runtime as rt
    db,_=native_pg_worker
    profile=str(tmp_path);principal='principal';request='create-local'
    sid=local_identity(profile,principal,request)
    source=SessionSource(Platform.LOCAL,sid,user_id=principal,chat_type='dm')
    route=build_session_key(source)
    now=_now()
    entry=SessionEntry(route,sid,now,now,origin=source,platform=Platform.LOCAL)
    policy=build_policy({'source':'gui','cwd':str(tmp_path)}, {},profile_terminal=False)
    created=dict(profile_id=profile,principal_id=principal,request_id=request,session_id=sid,route=route,
        entry=entry.to_dict(),policy=asdict(policy))
    assert commit_local_session(db,epoch=1,receipt=created)['session_id']==sid
    db.append_messages_batch(sid,[{'role':'user','content':'branch root'},{'role':'assistant','content':'retained answer'}])
    args=dict(epoch=1,principal_id=principal,session_id=sid,request_id='branch',expected_revision=0,
        expected_generation=0,operation='branch',payload={'title':'独立分支'})
    branched=rt.mutate_runtime_session(db,**args)
    child=branched['branched_session_id']
    assert [row['content'] for row in db.get_messages(child)]==['branch root','retained answer']
    assert local_receipt(db,child)['principal_id']==principal
    assert rt.mutate_runtime_session(db,**args)==branched
    reset_args=dict(args,request_id='reset',expected_revision=branched['revision'],operation='reset',payload={})
    reset=rt.mutate_runtime_session(db,**reset_args)
    target=reset['target_session_id']
    assert target!=sid and db.get_session(sid)['end_reason']=='session_reset'
    with db._read_ctx() as conn:assert local_physical_target(conn,sid)==target
    assert rt.mutate_runtime_session(db,**reset_args)==reset
    actor=Principal(principal,profile,frozenset({'session:create','session:read','session:control'}),'fixture')
    imported=dict(epoch=1,principal_id=principal,session_id='portable',request_id='import',expected_revision=0,
        operation='import',payload={'sessions':[{'id':'portable','source':'import','title':'Imported',
        'messages':[{'role':'user','content':'retained import'},{'role':'assistant','content':'archived','active':0,'compacted':1}]}]},
        _authorize_write=lambda conn:import_history_control(conn,actor,['portable']))
    receipt=rt.mutate_runtime_session(db,**imported)
    assert receipt['imported_ids']==['portable']
    assert [row['content'] for row in db.get_messages('portable')]==['retained import']
    assert [row['content'] for row in db.get_messages('portable',include_compacted=True)]==['retained import','archived']
    assert rt.mutate_runtime_session(db,**imported)==receipt


def test_pg_foreign_history_import_preserves_origin_and_duplicate_clicks(native_pg_worker,tmp_path):
    db,_=native_pg_worker
    origin={'tool':'claude','foreign_session_id':'foreign-id','path':'/read-only/source-export.json'}
    messages=[{'role':'user','content':'foreign 中文'},{'role':'assistant','content':'history'}]
    first=db.import_foreign_history(origin,messages,title='迁移历史',cwd=str(tmp_path),profile='fixture')
    assert not first['already_imported']
    again=db.import_foreign_history(origin,messages,title='changed label',cwd=str(tmp_path),profile='fixture')
    assert again=={'session_id':first['session_id'],'already_imported':True}
    assert db.find_foreign_import(origin)==first['session_id']
    row=db.get_session(first['session_id'])
    assert json.loads(row['origin_json'])=={'imported_from':origin} and row['profile_name']=='fixture'
    assert [message['content'] for message in db.get_messages(first['session_id'])]==['foreign 中文','history']
    other=db.import_foreign_history({**origin,'foreign_session_id':'different'},messages,title='迁移历史',cwd=str(tmp_path),profile='fixture')
    assert other['session_id']!=first['session_id']
    assert db.get_session(other['session_id'])['title'].startswith('迁移历史 (')
