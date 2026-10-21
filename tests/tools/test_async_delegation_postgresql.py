"""Main-process delegation recovery uses the scoped native PostgreSQL owner."""
import json
from pathlib import Path
import queue
import sqlite3
import time
import uuid

import psycopg
import pytest


class PrivateDSN(str):
    def __repr__(self):
        return '<private test DSN>'


@pytest.fixture
def delegation_scopes(tmp_path, monkeypatch):
    from hermes_state_postgresql import SessionScope, provision_scope
    path = Path(__file__).resolve().parents[3] / '.artifacts/shared-pg-v1/candidate-pg-private.json'
    if not path.is_file():
        pytest.skip('owned PostgreSQL fixture required')
    settings = {key: PrivateDSN(value) for key, value in json.loads(path.read_text())['dsns'].items()}
    homes = [tmp_path/'one', tmp_path/'two']
    for home in homes:
        home.mkdir()
    scopes = [SessionScope('tenant-'+str(index), 'workspace', str(uuid.uuid4())) for index in range(2)]
    for scope in scopes:
        provision_scope(settings['semantier_migrate'], scope, app_role='semantier_app')
    monkeypatch.setenv('HERMES_STATE_BACKEND', 'postgresql')
    monkeypatch.setenv('HERMES_DATABASE_URL', settings['semantier_app'])
    monkeypatch.setenv('HERMES_STATE_SCOPE_MAP', json.dumps({str(home): dict(zip(('tenant_id','workspace_id','source_store_id'), scope.identity)) for home, scope in zip(homes, scopes)}))
    from tools import async_delegation as ad
    selected = [homes[0]]
    monkeypatch.setattr(ad, 'get_hermes_home', lambda: selected[0])
    ad._reset_for_tests()
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **k: pytest.fail('main delegation opened SQLite'))
    try:
        yield ad, selected, homes
    finally:
        ad._reset_for_tests()
        with psycopg.connect(settings['semantier_migrate'], autocommit=True) as conn:
            for scope in scopes:
                conn.execute(psycopg.sql.SQL('DROP SCHEMA {} CASCADE').format(psycopg.sql.Identifier(scope.schema)))


def _dispatch(ad, goal='retained'):
    ad._persist_dispatch(dict(delegation_id='same-id', session_key='same-session', parent_session_id='parent', origin_ui_session_id='ui', dispatched_at=time.time(), goal=goal))


def _complete(ad, summary='中文\x00retained'):
    ad._persist_completion(dict(type='async_delegation', delegation_id='same-id', session_key='same-session', status='completed', completed_at=time.time()), {'summary':summary})


def test_main_replay_claims_and_replace_are_scoped_without_sqlite(delegation_scopes, monkeypatch):
    ad, selected, homes = delegation_scopes
    _dispatch(ad)
    _complete(ad)
    assert ad.get_durable_delegation('same-id')['result']['summary'] == '中文\x00retained'
    selected[0] = homes[1]
    assert ad.get_durable_delegation('same-id') is None
    _dispatch(ad, 'foreign')
    _complete(ad, 'other-home')
    selected[0] = homes[0]
    pending = queue.Queue()
    assert ad.restore_undelivered_completions(pending) == 1
    event = pending.get_nowait()
    assert event['restored'] is True
    ad.return_completion_offer(event)
    monkeypatch.setattr(ad, '_owner_liveness', lambda: lambda pid, started: False)
    assert ad.sweep_orphaned_completions(pending, now=time.time()+61) == 1
    assert pending.get_nowait()['delegation_id'] == 'same-id'
    assert ad.claim_completion_delivery('same-id', 'claim-one')
    assert not ad.claim_completion_delivery('same-id', 'claim-two')
    assert not ad.complete_completion_delivery('same-id', 'wrong-claim')
    assert ad.defer_completion_delivery('same-id', 'claim-one')
    assert ad.get_durable_delegation('same-id')['delivery_attempts'] == 0
    assert ad.claim_completion_delivery('same-id', 'claim-two')
    assert ad.complete_completion_delivery('same-id', 'claim-two')
    _dispatch(ad, 'replacement')
    row = ad.get_durable_delegation('same-id')
    assert row['state'] == 'running' and row['result'] is None
    assert row['completed_at'] is None and row['delivery_attempts'] == 0
    with ad._transaction() as conn:
        assert tuple(conn.execute("SELECT event_json,delivery_claim,delivered_at FROM async_delegations WHERE delegation_id='same-id'").fetchone()) == (None,None,None)
    selected[0] = homes[1]
    assert ad.get_durable_delegation('same-id')['result']['summary'] == 'other-home'
    assert all(not (home/'state.db').exists() for home in homes)


def test_owner_recovery_never_infers_foreign_or_legacy_pid_liveness(delegation_scopes, monkeypatch):
    ad, _selected, _homes = delegation_scopes
    _dispatch(ad)
    monkeypatch.setattr(ad, '_owner_liveness', lambda: lambda pid, started: False)
    with ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET owner_host='foreign-host' WHERE delegation_id='same-id'")
    assert ad.recover_abandoned_delegations() == 0
    with ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET owner_host=NULL WHERE delegation_id='same-id'")
    assert ad.recover_abandoned_delegations() == 0
    _dispatch(ad)
    assert ad.recover_abandoned_delegations() == 1
    assert ad.get_durable_delegation('same-id')['state'] == 'unknown'
    assert ad.failed_delegations_for_session('ui', 'parent')[0]['delegation_id'] == 'same-id'
    assert ad.failed_delegations_for_session('foreign') == []
    assert ad.recover_abandoned_delegations() == 0
    result = queue.Queue()
    assert ad.restore_undelivered_completions(result) == 1
    assert result.get_nowait()['status'] == 'unknown'


def test_two_real_connections_claim_once_and_rollback_keeps_state(delegation_scopes, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    ad, _selected, _homes = delegation_scopes
    _dispatch(ad)
    _complete(ad)
    # Remove only the process mutex; the real database fence must arbitrate.
    monkeypatch.setattr(ad, '_DB_LOCK', nullcontext())
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda key: ad.claim_completion_delivery('same-id', key), ('one','two')))
    assert sorted(claims) == [False,True]
    winner = ('one','two')[claims.index(True)]
    with pytest.raises(RuntimeError, match='rollback'):
        with ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET delivery_state='dropped'")
            raise RuntimeError('rollback')
    assert ad.get_durable_delegation('same-id')['delivery_state'] == 'pending'
    assert ad.release_completion_delivery('same-id', winner)


def test_actual_authenticated_worker_bridge_on_native_postgresql(delegation_scopes, monkeypatch):
    import os
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope
    from tests.gateway import test_worker_delegation_bridge as bridge
    _ad, _selected, homes = delegation_scopes
    scope = SessionScope(**json.loads(os.environ['HERMES_STATE_SCOPE_MAP'])[str(homes[0])])
    dsn = PrivateDSN(os.environ['HERMES_DATABASE_URL'])
    monkeypatch.setattr(bridge, 'SessionDB', lambda path: PostgreSQLSessionDB(dsn, scope, home=Path(path).parent))
    bridge.test_bridge_publishes_committed_completion_and_rejects_another_principal(homes[0], monkeypatch)


@pytest.mark.parametrize('case', ['reservation', 'physical_unknown', 'loss'])
def test_canonical_private_worker_reservation_and_recovery_on_pg(delegation_scopes, monkeypatch, case):
    import os
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope
    from tests.gateway import test_worker_admission_reservation as reservation
    from tests.gateway import test_unknown_resolution_physical_worker as physical
    from tests.gateway import test_managed_worker_fencing as fencing
    _ad, _selected, homes = delegation_scopes
    scope = SessionScope(**json.loads(os.environ['HERMES_STATE_SCOPE_MAP'])[str(homes[0])])
    dsn = PrivateDSN(os.environ['HERMES_DATABASE_URL'])
    module, check = {
        'reservation': (reservation, reservation.test_started_admission_binds_only_exact_owned_process_and_principal),
        'physical_unknown': (physical, physical.test_resolve_unknown_retires_the_physical_worker_row),
        'loss': (fencing, fencing.test_worker_loss_fences_same_generation_parent_result),
    }[case]
    monkeypatch.setattr(module, 'SessionDB', lambda path: PostgreSQLSessionDB(dsn, scope, home=Path(path).parent))
    check(homes[0])


def test_profile_discovery_reads_actual_pg_owner(delegation_scopes, monkeypatch):
    import os
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope
    from tests.gateway import test_session_group_controls as groups
    _ad, _selected, homes = delegation_scopes
    scope = SessionScope(**json.loads(os.environ['HERMES_STATE_SCOPE_MAP'])[str(homes[0])])
    dsn = PrivateDSN(os.environ['HERMES_DATABASE_URL'])
    monkeypatch.setattr(groups, 'SessionDB', lambda path: PostgreSQLSessionDB(dsn, scope, home=Path(path).parent))
    groups.test_profile_discovery_and_group_controls_enforce_actor_scope(homes[0], monkeypatch)


def test_child_and_bot_home_stay_pinned_to_parent_pg_scope(delegation_scopes, monkeypatch):
    from types import SimpleNamespace
    from hermes_state_backend import open_session_db, session_store_key
    from tools.delegate_tool import _parent_live_home, _open_child_session_db
    from tools.bot_mode_dm import _agent_home
    _ad, _selected, homes = delegation_scopes
    monkeypatch.setenv('HERMES_HOME', str(homes[1]))
    with open_session_db(home=homes[0]) as parent:
        parent.create_session('parent', 'cli')
        agent = SimpleNamespace(_session_db=parent)
        assert _parent_live_home(agent) == homes[0]
        assert _agent_home(agent) == str(homes[0])
        child = _open_child_session_db(agent)
        assert child is not parent
        try:
            assert session_store_key(child) == session_store_key(parent)
            child.create_session('child', 'cli', parent_session_id='parent')
            assert parent.get_session('child')['parent_session_id'] == 'parent'
            with open_session_db(home=homes[1]) as foreign:
                assert foreign.get_session('child') is None
        finally:
            child.close()
        assert parent.get_session('parent') is not None


@pytest.mark.asyncio
async def test_peer_bot_chat_enters_real_pg_authority_fifo(delegation_scopes, monkeypatch):
    from tests.gateway import test_peer_dm_bot_chat_authority as peer
    import hermes_constants
    import hermes_state
    _ad, _selected, homes = delegation_scopes
    monkeypatch.setattr(hermes_constants, '_PINNED_PROCESS_HERMES_HOME', None)
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', homes[0] / 'state.db')
    fixture = peer.peer.__wrapped__(homes[0], monkeypatch)
    instance = await anext(fixture)
    try:
        await peer.test_peer_dm_into_the_bot_chat_is_the_chats_own_admission(instance)
    finally:
        await fixture.aclose()
