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
