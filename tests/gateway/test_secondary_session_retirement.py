"""Native secondary deletion must retire launch routing without bypassing ownership."""
import json
from datetime import datetime
from types import SimpleNamespace
import pytest
from gateway.config import GatewayConfig
from gateway.session import SessionStore, SessionEntry
from gateway.session_authority import SessionAuthority, LiveSession
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from hermes_state_mutation_retirement import retired_session

@pytest.fixture
def stores(tmp_path, monkeypatch):
    home = tmp_path / "launch"
    for name in ("cold", "warm"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SEMANTIER_RUNTIME", "0")
    store = SessionStore(home / "sessions", GatewayConfig(multiplex_profiles=True))
    # Profile-path resolution itself is covered by the real daemon test.
    store._profile_home_cache.update({name: home/"profiles"/name for name in ("cold","warm")})
    db = store._open_session_db_for_active_scope(db_path=home/"profiles"/"cold"/"state.db")
    other = store._open_session_db_for_active_scope(db_path=home/"profiles"/"warm"/"state.db")
    try:
        yield home, store, db, other
    finally:
        store.close_all_db_handles()

def route(store, name, sid):
    key = f"agent:{name}:local:dm:{sid}"
    entry = SessionEntry(session_key=key, session_id=sid,
        created_at=datetime.now(), updated_at=datetime.now())
    with store._lock:
        store._ensure_loaded_locked()
        store._entries[key] = entry
    store._save_entries()
    return key

async def delete(store, db, monkeypatch=None):
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import SessionAuthorities
    home = store._routing_home
    process_ownership.reserve([home, db.db_path.parent])
    runner = SimpleNamespace(_draining=False, session_store=store, _evict_cached_agent=lambda key: None)
    authority = SessionAuthority(runner, profile_id=str(db.db_path.parent),
        instance_id="fixture", db=db, epoch=begin_runtime_epoch(db, instance_id="fixture"))
    runner.session_authorities = SessionAuthorities(home, multiplexed=True)
    runner.session_authorities.add(home, SimpleNamespace(runner=runner, db=store._routing_db), name='default')
    runner.session_authorities.add(db.db_path.parent, authority, name='cold')
    authority.sessions["owned"] = LiveSession(None, "agent:cold:local:dm:owned")
    peer = AuthorityConnection(authority, object(), {"user_id":"human"})
    await peer.dispatch({"id":1,"method":"session.resume","params":{"session_id":"owned"}})
    params = {"session_id":"owned","request_id":"delete","expected_revision":0,
        "expected_generation":0,"operation":"delete","payload":{}}
    try:
        first = await peer.dispatch({"id":2,"method":"session.mutate","params":params})
        second = await peer.dispatch({"id":3,"method":"session.mutate","params":params})
        return first, second
    finally:
        await peer.close()
        process_ownership.release(db.db_path.parent)
        process_ownership.release(home)

@pytest.mark.asyncio
async def test_secondary_delete_persists_routes_and_exact_retry(stores):
    home, store, db, other = stores
    db.create_session("owned", source="test")
    db.create_session("kept", source="test")
    other.create_session("owned", source="test")
    cold = route(store, "cold", "owned")
    kept = route(store, "cold", "kept")
    warm = route(store, "warm", "owned")
    old_data, old_generation = store._snapshot_routing_locked()
    first, second = await delete(store, db)
    assert first.get("result", {}).get("deleted_ids") == ["owned"], first
    assert second.get("result") == first["result"], second
    assert retired_session(db, "owned")
    assert db.get_session("owned") is None and db.get_session("kept") is not None
    assert other.get_session("owned") is not None
    store._persist_routing_data(old_data, old_generation)
    saved = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    assert cold not in saved and kept in saved and warm in saved

def test_reopen_rejects_route_when_owner_has_committed_retirement(stores):
    home, store, db, other = stores
    db.create_session("owned", source="test")
    key = route(store, "cold", "owned")
    from hermes_state_runtime import mutate_runtime_session
    mutate_runtime_session(db, epoch=begin_runtime_epoch(db, instance_id="cut"),
        principal_id="human", session_id="owned", request_id="cut",
        expected_revision=0, expected_generation=0, operation="delete", payload={},
        _live_guard=lambda targets: True)
    assert key in store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    restored = SessionStore(home/"sessions", GatewayConfig(multiplex_profiles=True))
    restored._profile_home_cache.update(store._profile_home_cache)
    try:
        assert restored.lookup_by_session_key(key) is None
        assert key not in restored._routing_db.load_gateway_routing_entries(scope=restored._routing_scope())
    finally:
        restored.close_all_db_handles()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("cut", ["before_commit", "after_source", "after_route", "after_ack"])
def test_process_cut_recovery_preserves_retirement(stores, cut):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from hermes_state_mutation_retirement import RETIRED_PREFIX, prune_retired_tombstones
    home, store, db, other = stores
    db.create_session("owned", source="test")
    db.create_session("kept", source="test")
    key = route(store, "cold", "owned")
    kept = route(store, "cold", "kept")
    source = Path(__file__).resolve().parents[2]
    code = """
import asyncio, os, sys
from pathlib import Path
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB
from tests.gateway.test_secondary_session_retirement import delete
import gateway.session_mutations as mutations
home, cut = Path(sys.argv[1]), sys.argv[2]
store = SessionStore(home/'sessions', GatewayConfig(multiplex_profiles=True))
store._profile_home_cache.update({name:home/'profiles'/name for name in ('cold','warm')})
db = store._open_session_db_for_active_scope(db_path=home/'profiles'/'cold'/'state.db')
def die(*args, **kwargs):
    os._exit(85)
if cut == 'before_commit':
    import hermes_state_mutations as actions
    original = actions.apply_action
    def apply(*args, **kwargs):
        original(*args, **kwargs)
        die()
    actions.apply_action = apply
elif cut == 'after_source':
    mutations._project_committed = die
elif cut == 'after_route':
    store._ack_route_retirement = die
else:
    original = mutations._project_committed
    def publish(*args, **kwargs):
        original(*args, **kwargs)
        die()
    mutations._project_committed = publish
asyncio.run(delete(store, db))
raise AssertionError('cut was not exercised')
"""
    environment = {**os.environ, "PYTHONPATH":str(source), "HERMES_HOME":str(home), "SEMANTIER_RUNTIME":"0"}
    child = subprocess.run([sys.executable, "-c", code, str(home), cut],
        cwd=source, env=environment, capture_output=True, text=True, timeout=45)
    assert child.returncode == 85, (child.returncode, child.stdout, child.stderr)
    if cut == "before_commit":
        assert db.get_session("owned") is not None
    else:
        assert db.get_session("owned") is None
        with db._read_ctx() as conn:
            marker = json.loads(conn.execute("SELECT value FROM state_meta WHERE key=?",
                (RETIRED_PREFIX+"owned",)).fetchone()[0])
        assert marker["routing_pending"] is (cut != "after_ack")
        if marker["routing_pending"]:
            def age(conn):
                conn.execute("UPDATE state_meta SET value=json_set(value, '$.retired_at', 0) WHERE key=?",
                    (RETIRED_PREFIX+"owned",))
                prune_retired_tombstones(conn)
            db._execute_write(age)
            assert retired_session(db, "owned")
    verify = """
import json, sys
from pathlib import Path
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB
from hermes_state_mutation_retirement import RETIRED_PREFIX
home = Path(sys.argv[1])
store = SessionStore(home/'sessions', GatewayConfig(multiplex_profiles=True))
store._profile_home_cache.update({name:home/'profiles'/name for name in ('cold','warm')})
key, kept = sys.argv[2:4]
try:
    owned = store.lookup_by_session_key(key) is not None
    db = store._open_session_db_for_active_scope(db_path=home/'profiles'/'cold'/'state.db')
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (RETIRED_PREFIX+'owned',)).fetchone()
    pending = json.loads(row[0]).get('routing_pending') if row else None
    print(json.dumps({'owned':owned, 'pending':pending,
        'kept':store.lookup_by_session_key(kept) is not None,
        'durable':key in store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())}))
finally:
    store.close_all_db_handles()
"""
    reopened = subprocess.run([sys.executable, "-c", verify, str(home), key, kept],
        cwd=source, env=environment, capture_output=True, text=True, timeout=45)
    assert reopened.returncode == 0, reopened.stderr
    assert json.loads(reopened.stdout.splitlines()[-1]) == {
        "owned":cut == "before_commit", "kept":True, "durable":cut == "before_commit",
        "pending":True if cut == "after_route" else False if cut in ("after_source", "after_ack") else None}

@pytest.mark.platforms('linux')
def test_secondary_native_http_delete_via_multiplexer(tmp_path):
    """Use the desktop's profile ticket and mutation snapshot over a real gateway."""
    import asyncio
    import sqlite3
    import signal
    import subprocess
    import sys
    from pathlib import Path
    import httpx
    from tests.gateway.test_multiplex_cold_profile_sidebar_mutation import (
        multiplex_daemon, desktop_ensure, native_headers, create_via_ws,
    )

    with multiplex_daemon(tmp_path, profiles=('cold', 'warm')) as (home, descriptor, env):
        source = Path(__file__).resolve().parents[2]
        endpoint = desktop_ensure(env, source, 'cold')
        other = desktop_ensure(env, source, 'warm')
        sid = asyncio.run(create_via_ws(endpoint, 'delete-real-secondary'))
        kept_sid = asyncio.run(create_via_ws(endpoint, 'keep-real-secondary'))
        with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
            path = f'/api/sessions/{sid}'
            kept_path = f'/api/sessions/{kept_sid}'
            assert client.get(kept_path, params={'profile':'cold'}, headers=native_headers(endpoint)).status_code == 200
            snap = client.get(path + '/mutation-snapshot', params={'profile':'cold'}, headers=native_headers(endpoint))
            assert snap.status_code == 200, snap.text
            identity = {'request_id':'delete-real-secondary', 'expected_revision':snap.json()['runtime_revision'],
                        'expected_generation':snap.json()['runtime_generation']}
            foreign = client.delete(path, params={**identity, 'profile':'cold'}, headers=native_headers(other))
            assert foreign.status_code == 403, foreign.text
            reply = client.delete(path, params={**identity, 'profile':'cold'}, headers=native_headers(endpoint))
            assert reply.status_code == 200, reply.text
            assert client.get(path, params={'profile':'cold'}, headers=native_headers(endpoint)).status_code == 404
            assert client.get(kept_path, params={'profile':'cold'}, headers=native_headers(endpoint)).status_code == 200
        with sqlite3.connect(home / 'profiles' / 'cold' / 'state.db') as db:
            assert db.execute('SELECT count(*) FROM sessions WHERE id=?', (sid,)).fetchone() == (0,)
            assert db.execute('SELECT count(*) FROM sessions WHERE id=?', (kept_sid,)).fetchone() == (1,)

    # Reopen the same stores with a fresh owner and fresh native HTTP tickets.
    # This catches a launch-store route or mirror that resurrects a deleted row.
    from tests.gateway.test_normal_runtime_boot import control
    from tests.waiting import wait_until
    with (tmp_path / 'gateway-restart.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=source, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            def ready():
                if process.poll() is not None:
                    raise AssertionError(f'gateway restart exited: {process.returncode}')
                try:
                    result = control(home, 'identify')
                    return result if result.get('state') == 'ready' else None
                except (OSError, ValueError, AssertionError):
                    return None

            restarted = wait_until(ready, timeout=60, what='restarted multiprofile owner')
            assert restarted['instance_id'] != descriptor['instance_id']
            new_endpoint = desktop_ensure(env, source, 'cold')
            with httpx.Client(base_url=restarted['api_origin'], trust_env=False, timeout=30) as client:
                assert client.get(path, params={'profile':'cold'}, headers=native_headers(new_endpoint)).status_code == 404
                assert client.get(kept_path, params={'profile':'cold'}, headers=native_headers(new_endpoint)).status_code == 200
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)




def test_coordination_refuses_unregistered_and_unreserved_authorities(stores):
    """An accessible same-path database and caller-supplied profile do not prove ownership."""
    from gateway.session_authorities import SessionAuthorities
    home, store, db, _ = stores
    runner = SimpleNamespace(session_store=store, session_authorities=SessionAuthorities(home, multiplexed=True))
    forged = SimpleNamespace(runner=runner, db=db, profile_id=str(db.db_path.parent), retired=False, retiring=False)
    assert not store.can_coordinate_retirement(forged)
    runner.session_authorities.add(forged.profile_id, forged, name='cold')
    assert not store.can_coordinate_retirement(forged)


@pytest.mark.parametrize('postgres_flag', [False, True])
def test_coordination_requires_live_registry_and_both_reservations(stores, monkeypatch, postgres_flag):
    # This is the ownership predicate test, not PostgreSQL SQL integration.
    monkeypatch.setattr('gateway.session_persistence.postgres_enabled', lambda: postgres_flag)
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import SessionAuthorities
    home, store, db, other = stores
    if postgres_flag:
        # Predicate-only fixture: emulate the PostgreSQL concrete type while the
        # real four-container test proves PostgreSQL persistence and publication.
        import hermes_state_postgresql
        monkeypatch.setattr(hermes_state_postgresql, 'PostgreSQLSessionDB', SessionDB)
    runner = SimpleNamespace(session_store=store, session_authorities=SessionAuthorities(home, multiplexed=True))
    launch = SimpleNamespace(runner=runner, db=store._routing_db)
    owner = SimpleNamespace(runner=runner, db=db, profile_id=str(db.db_path.parent), retired=False, retiring=False)
    runner.session_authorities.add(home, launch, name='default')
    runner.session_authorities.add(owner.profile_id, owner, name='cold')
    process_ownership.reserve([home])
    try:
        assert not store.can_coordinate_retirement(owner)
        process_ownership.reserve([db.db_path.parent])
        assert store.can_coordinate_retirement(owner)
        assert not store.can_coordinate_retirement(SimpleNamespace(**vars(owner)))
        # A different connection to the same path is not the runner's lent handle.
        duplicate = SessionDB(db_path=db.db_path)
        try:
            owner.db = duplicate
            assert not store.can_coordinate_retirement(owner)
        finally:
            owner.db = db
            duplicate.close()
        launch.db = other
        assert not store.can_coordinate_retirement(owner)
        launch.db = store._routing_db
        owner.retiring = True
        assert not store.can_coordinate_retirement(owner)
    finally:
        process_ownership.release(db.db_path.parent)
        process_ownership.release(home)


def test_retirement_ack_refuses_unresolvable_route_owner(stores):
    """A matching route with unknown scope cannot be declared durably retired."""
    home, store, db, _ = stores
    db.create_session('owned', source='test')
    key = route(store, 'cold', 'owned')
    from hermes_state_runtime import mutate_runtime_session
    mutate_runtime_session(db, epoch=begin_runtime_epoch(db, instance_id='cut'),
        principal_id='human', session_id='owned', request_id='cut', expected_revision=0,
        expected_generation=0, operation='delete', payload={}, _live_guard=lambda targets: True)
    store._profile_home_cache.pop('cold')
    store._profile_home_for_key = lambda route_key: None
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        store._ack_route_retirement(db, ['owned'])
    assert key in store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())


@pytest.mark.asyncio
async def test_secondary_delete_preserves_foreign_unpublished_route_hint(stores):
    home, store, db, other = stores
    db.create_session('owned', source='test')
    other.create_session('owned', source='test')
    route(store, 'cold', 'owned')
    foreign_hint = 'agent:warm:local:dm:owned'
    store._session_owner_hints['owned'] = foreign_hint
    first, _ = await delete(store, db)
    assert first.get('result', {}).get('deleted_ids') == ['owned'], first
    assert store._session_owner_hints.get('owned') == foreign_hint
    assert other.get_session('owned') is not None


def test_route_write_failure_keeps_pending_until_durable_retry(stores, monkeypatch):
    """A mirror fallback cannot acknowledge an old authoritative routing row."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    from hermes_state_runtime import RuntimeStoreError, mutate_runtime_session
    home, store, db, _ = stores
    db.create_session('owned', source='test')
    key = route(store, 'cold', 'owned')
    mutate_runtime_session(db, epoch=begin_runtime_epoch(db, instance_id='cut'),
        principal_id='human', session_id='owned', request_id='cut', expected_revision=0,
        expected_generation=0, operation='delete', payload={}, _live_guard=lambda targets: True)
    root = store._routing_db
    original = root.replace_gateway_routing_entries
    def fail(*args, **kwargs):
        raise OSError('fixture routing write failure')
    monkeypatch.setattr(root, 'replace_gateway_routing_entries', fail)
    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        store.retire_runtime_sessions(['owned'], session_db=db)
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (RETIRED_PREFIX+'owned',)).fetchone()
    assert json.loads(row[0])['routing_pending'] is True
    assert key in root.load_gateway_routing_entries(scope=store._routing_scope())
    monkeypatch.setattr(root, 'replace_gateway_routing_entries', original)
    store.retire_runtime_sessions(['owned'], session_db=db)
    assert key not in root.load_gateway_routing_entries(scope=store._routing_scope())
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (RETIRED_PREFIX+'owned',)).fetchone()
    assert json.loads(row[0])['routing_pending'] is False


def test_mirror_write_failure_keeps_pending_until_durable_retry(stores, monkeypatch):
    """An old mirror must not outlive the profile's acknowledged retirement fence."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    from hermes_state_runtime import RuntimeStoreError, mutate_runtime_session
    home, store, db, _ = stores
    db.create_session('owned', source='test')
    key = route(store, 'cold', 'owned')
    mirror = store.sessions_dir / 'sessions.json'
    assert key in json.loads(mirror.read_text())
    mutate_runtime_session(db, epoch=begin_runtime_epoch(db, instance_id='cut'),
        principal_id='human', session_id='owned', request_id='cut', expected_revision=0,
        expected_generation=0, operation='delete', payload={}, _live_guard=lambda targets: True)
    original = store._save_sessions_json
    def fail(*args, **kwargs):
        raise OSError('fixture mirror write failure')
    monkeypatch.setattr(store, '_save_sessions_json', fail)
    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        store.retire_runtime_sessions(['owned'], session_db=db)
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (RETIRED_PREFIX+'owned',)).fetchone()
    assert json.loads(row[0])['routing_pending'] is True
    assert key in json.loads(mirror.read_text())
    monkeypatch.setattr(store, '_save_sessions_json', original)
    store.retire_runtime_sessions(['owned'], session_db=db)
    assert key not in json.loads(mirror.read_text())
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (RETIRED_PREFIX+'owned',)).fetchone()
    assert json.loads(row[0])['routing_pending'] is False
