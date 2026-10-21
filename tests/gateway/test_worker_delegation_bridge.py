"""Authenticated worker ledger bridge publishes committed owner events."""
import asyncio
from contextlib import closing
import json
import os
import queue
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, register_worker_execution
from gateway.session_contract import SessionRef
from gateway.session_worker import _claim
from tools.async_delegation import _initialize_schema


def test_bridge_publishes_committed_completion_and_rejects_another_principal(tmp_path, monkeypatch):
    from gateway.session_worker_delegation import worker_delegation_request
    from tools.process_registry import process_registry
    target = queue.Queue()
    monkeypatch.setattr(process_registry, 'completion_queue', target)
    db = SessionDB(tmp_path / 'state.db')
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('parent', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        if getattr(db, 'backend', 'sqlite') == 'sqlite':
            with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
                _initialize_schema(conn)
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id='profile', _require_admission_open=lambda: None)
        actor = SimpleNamespace(profile_id='profile', subject='alice', capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        ref = SessionRef('profile', 'parent')
        scope = dict(profile_id='profile', session_id='parent', execution_id='worker', generation=0,
                     pid=worker.pid, birth=psutil.Process(worker.pid).create_time(), secret='private-worker-secret')
        claim = _claim(connection, ref, scope)
        register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                                  kind='compute', adoption_secret=claim)
        payload = dict(delegation_id='deleg_owned', task={'goal': 'inert'}, dispatched_at=time.time(),
                       parent_session_id='parent', session_key='', origin_ui_session_id='', origin_session_id='')
        params = scope | dict(epoch=epoch, sequence=1, operation='delegation.dispatch', payload=payload)
        asyncio.run(worker_delegation_request(connection, ref, params))
        event = dict(type='async_delegation', delegation_id='deleg_owned', parent_session_id='parent',
            session_key='', origin_ui_session_id='', origin_session_id='', status='completed',
            completed_at=time.time(), summary='durable')
        params = scope | dict(epoch=epoch, sequence=2, operation='delegation.complete',
                              payload={'event': event, 'result': {'summary': 'durable'}})
        response = asyncio.run(worker_delegation_request(connection, ref, params))
        assert response == {'value': None}
        assert target.get_nowait() == event | {'restored': True}
        assert asyncio.run(worker_delegation_request(connection, ref, params)) == response
        assert target.get_nowait() == event | {'restored': True}
        with db._read_ctx() as conn:
            row = conn.execute('SELECT event_json FROM async_delegations').fetchone()
            assert json.loads(row[0]) == event
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 2
        actor.subject = 'mallory'
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            asyncio.run(worker_delegation_request(connection, ref, params))
        assert target.empty()
        actor.subject = 'alice'
        for sequence, operation in enumerate(('claim', 'defer', 'claim', 'release', 'claim', 'ack'), 3):
            result = asyncio.run(worker_delegation_request(connection, ref, scope | dict(
                epoch=epoch, sequence=sequence, operation='delegation.'+operation,
                payload={'delegation_id':'deleg_owned', 'claim_id':'delivery-token'})))
            assert result == {'value':True}
        drop_payload = payload | {'delegation_id':'deleg_drop'}
        drop_event = event | {'delegation_id':'deleg_drop'}
        for sequence, operation, value in (
            (9,'dispatch',drop_payload),
            (10,'complete',{'event':drop_event, 'result':{'summary':'durable'}}),
            (11,'claim',{'delegation_id':'deleg_drop','claim_id':'drop-token'}),
            (12,'drop',{'delegation_id':'deleg_drop','claim_id':'drop-token'}),
            (13,'dispatch',payload | {'delegation_id':'unscheduled'}),
            (14,'unscheduled',{'delegation_id':'unscheduled'}),
        ):
            asyncio.run(worker_delegation_request(connection, ref, scope | dict(
                epoch=epoch, sequence=sequence, operation='delegation.'+operation, payload=value)))
        assert target.get_nowait() == drop_event | {'restored':True}
        with db._read_ctx() as conn:
            states = {row[0]:row[1] for row in conn.execute('SELECT delegation_id,delivery_state FROM async_delegations')}
            assert states == {'deleg_owned':'delivered', 'deleg_drop':'dropped'}
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 14
    finally:
        worker.terminate()
        worker.wait(timeout=5)
        db.close()
