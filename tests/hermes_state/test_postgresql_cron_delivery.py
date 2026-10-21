"""Actual cron notepad and queued delivery owner paths on scoped PostgreSQL."""

import json
from uuid import uuid4

import pytest

from tests.hermes_state.test_postgresql_sessions import pg_settings


def test_cron_payloads_and_gateway_drain_preserve_two_home_id_collisions(
    tmp_path, pg_settings, monkeypatch
):
    import sqlite3
    from hermes_state_postgresql import SessionScope, provision_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from cron import notepad, delivery_queue, scheduler

    homes = [tmp_path / "one", tmp_path / "two"]
    scopes = [
        SessionScope("tenant-" + str(i), "workspace", str(uuid4())) for i in range(2)
    ]
    for scope in scopes:
        provision_scope(
            pg_settings["semantier_migrate"], scope, app_role="semantier_app"
        )
    monkeypatch.setenv("HERMES_STATE_BACKEND", "postgresql")
    monkeypatch.setenv("HERMES_DATABASE_URL", pg_settings["semantier_app"])
    monkeypatch.setenv(
        "HERMES_STATE_SCOPE_MAP",
        json.dumps(
            {
                str(home): dict(
                    zip(
                        ("tenant_id", "workspace_id", "source_store_id"), scope.identity
                    )
                )
                for home, scope in zip(homes, scopes)
            }
        ),
    )
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: pytest.fail("hosted cron opened SQLite")
    )
    monkeypatch.setattr(notepad, "NOTEPAD_FILE", None)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", None)
    sent = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, content, **kw: sent.append((job["id"], content)) or None,
    )
    for i, home in enumerate(homes):
        token = set_hermes_home_override(home)
        try:
            note = "中文\x00 cursor " + str(i)
            notepad.set_note("same-job", "游标\x00key", note)
            assert notepad.get_note("same-job", "游标\x00key") == note
            assert note in notepad.render_notepad_section("same-job")
            queued = delivery_queue.enqueue(
                "same-execution",
                {"id": "same-job", "name": "测试\x00"},
                "结果\x00 " + str(i),
            )
            assert queued["status"] == "pending" and queued[
                "content"
            ] == "结果\x00 " + str(i)
            assert scheduler.drain_delivery_queue({}, None) == 1
            assert delivery_queue.get_status("same-execution")["status"] == "delivered"
            assert not list(home.rglob("*.db"))
        finally:
            reset_hermes_home_override(token)
    assert sent == [("same-job", "结果\x00 0"), ("same-job", "结果\x00 1")]
    for i, home in enumerate(homes):
        token = set_hermes_home_override(home)
        try:
            assert notepad.list_notes("same-job")[0][
                "value"
            ] == "中文\x00 cursor " + str(i)
            assert (
                delivery_queue.enqueue("same-execution", {}, "different")["status"]
                == "delivered"
            )
        finally:
            reset_hermes_home_override(token)


def test_cron_snapshot_import_preserves_nul_payloads_and_readonly_scope(tmp_path,pg_settings,monkeypatch):
    from pathlib import Path
    import hashlib
    import sqlite3
    import psycopg
    from cron import notepad,delivery_queue
    from hermes_constants import set_hermes_home_override,reset_hermes_home_override
    from hermes_state_postgresql import SessionScope,PostgreSQLSessionDB,provision_scope
    source_home=tmp_path/'source';target=tmp_path/'target'
    notes=source_home/'cron/notepad.db';deliveries=source_home/'cron/deliveries.db'
    with monkeypatch.context() as local:
        local.setenv('HERMES_STATE_BACKEND','sqlite')
        local.setattr(notepad,'NOTEPAD_FILE',notes)
        local.setattr(delivery_queue,'DELIVERY_DB',deliveries)
        notepad.set_note('same-job','游标\x00key','保留\x00值')
        delivery_queue.enqueue('same-execution',{'id':'same-job','name':'保留\x00'},'保留\x00结果')
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in (notes,deliveries)}
    scope=SessionScope('tenant','workspace',str(uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    with PostgreSQLSessionDB(pg_settings['semantier_migrate'],scope,home=target) as db:
        for family,source in [('cron_notes',notes),('cron_delivery',deliveries)]:
            receipt=db.import_sqlite_snapshot(source,family=family)
            assert not receipt['idempotent'] and receipt['table_fingerprints']
            assert db.import_sqlite_snapshot(source,family=family)['idempotent']
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(target):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(notepad,'NOTEPAD_FILE',None);monkeypatch.setattr(delivery_queue,'DELIVERY_DB',None)
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('runtime opened SQLite after import'))
    token=set_hermes_home_override(target)
    try:
        assert notepad.get_note('same-job','游标\x00key')=='保留\x00值'
        assert delivery_queue.get_status('same-execution')['content']=='保留\x00结果'
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            with notepad._transaction(read_only=True) as conn:
                conn.execute('DELETE FROM cron_notepad')
        assert notepad.get_note('same-job','游标\x00key')=='保留\x00值'
        claimed=delivery_queue.claim_next()
        assert claimed['content']=='保留\x00结果'
        with delivery_queue._transaction() as conn:
            conn.execute("UPDATE deliveries SET owner_host='foreign-host'")
        assert delivery_queue.recover_abandoned()==0
        assert delivery_queue.get_status('same-execution')['status']=='delivering'
        assert not delivery_queue._finish('same-execution',error=None)
        with delivery_queue._transaction() as conn:
            conn.execute('UPDATE deliveries SET owner_host=NULL')
        assert delivery_queue.recover_abandoned()==0
        with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=target) as db:
            with pytest.raises(PermissionError):db.import_sqlite_snapshot(notes,family='cron_notes')
        assert not list(target.rglob('*.db'))
    finally:reset_hermes_home_override(token)


def test_completed_cron_occurrence_and_tool_message_anchor_use_actual_pg_owner(tmp_path,pg_settings,monkeypatch):
    import sqlite3
    from datetime import datetime,UTC
    from hermes_state_postgresql import SessionScope,provision_scope
    from hermes_state_backend import open_session_db
    from hermes_constants import set_hermes_home_override,reset_hermes_home_override
    from cron import executions,occurrences
    from tools.session_search_tool import _get_message_storage_state
    home=tmp_path/'home';scope=SessionScope('tenant','workspace',str(uuid4()))
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(home):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity))}))
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('callback opened SQLite'))
    monkeypatch.setattr(executions,'EXECUTIONS_FILE',None)
    token=set_hermes_home_override(home)
    try:
        instant=datetime.now(UTC).isoformat()
        record=executions.create_execution('job',source='cron',scheduled_instant=instant)
        assert not occurrences.completed_occurrence({'id':'job'},instant)
        executions.finish_execution(record['id'],success=True)
        assert occurrences.completed_occurrence({'id':'job'},instant)
        with open_session_db(home=home) as db:
            db.create_session('session','cli')
            db.append_message('session','user','中文 anchor')
            row_id=db.latest_message_row_id('session')
            assert _get_message_storage_state(db,row_id)==dict(session_id='session',active=1,compacted=0)
            from types import SimpleNamespace
            from gateway import session_cron
            from cron import scheduler
            from hermes_state_registry import release_or_close
            from hermes_state_backend import session_store_key
            with monkeypatch.context() as local:
                local.setattr(session_cron,'current_execution',lambda:(SimpleNamespace(db=db),'session','job',record['id']))
                local.setenv('HERMES_CRON_SESSION_DB_TIMEOUT','0')
                borrowed=scheduler._open_cron_session_db({'id':'job'})
                assert session_store_key(borrowed)==session_store_key(db)
                assert borrowed is not db
                release_or_close(borrowed)
                assert db.get_session('session')['id']=='session'
        assert not list(home.rglob('*.db'))
    finally:
        reset_hermes_home_override(token)


def test_selected_gateway_watcher_hashes_native_rows_without_sqlite_files(tmp_path,pg_settings,monkeypatch):
    import sqlite3
    from hermes_state_postgresql import SessionScope,provision_scope
    from hermes_state_backend import open_session_db
    from tui_gateway import server
    homes=[tmp_path/'one',tmp_path/'two'];scopes=[SessionScope('tenant'+str(i),'workspace',str(uuid4())) for i in range(2)]
    for scope in scopes:provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql');monkeypatch.setenv('HERMES_DATABASE_URL',pg_settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP',json.dumps({str(home):dict(zip(('tenant_id','workspace_id','source_store_id'),scope.identity)) for home,scope in zip(homes,scopes)}))
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**k:pytest.fail('watcher opened SQLite'))
    monkeypatch.setattr(server,'_watcher_home',lambda:homes[0]);monkeypatch.setattr(server,'_served_profile_homes',[homes[1]])
    for i,home in enumerate(homes):
        with open_session_db(home=home) as db:db.create_session('same','cli',title='one'+str(i))
    before=server._sessions_sig()
    assert before[0] is not None and before[1] is not None and before[0]!=before[1]
    broadcasts=[]
    monkeypatch.setattr(server,'_CHANGE_WATCHES',{'sessions.changed':(0.5,server._sessions_sig,dict)})
    monkeypatch.setattr(server,'_change_sigs',{})
    monkeypatch.setattr(server,'_change_checked_at',{})
    monkeypatch.setattr(server,'_change_broadcast_at',{})
    monkeypatch.setattr(server,'_broadcast_global_event',lambda name,payload:broadcasts.append((name,payload)))
    server._broadcast_watched_changes(now=0)
    assert broadcasts==[]
    with open_session_db(home=homes[0]) as db:
        db.set_session_title('same','changed')
        changed=server._sessions_sig()
        assert changed[0]!=before[0] and changed[1]==before[1]
        server._broadcast_watched_changes(now=3)
        assert broadcasts==[('sessions.changed',{})]
        db.touch_session_activity('same',last_activity_description='heartbeat')
        assert server._sessions_sig()==changed
        server._broadcast_watched_changes(now=6)
        assert broadcasts==[('sessions.changed',{})]
    assert not any(home.exists() and list(home.rglob('*.db')) for home in homes)
    with pytest.raises(ValueError,match='mapping required'):
        server._session_db_content_sig(tmp_path/'unmapped'/'state.db')
