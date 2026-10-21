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
