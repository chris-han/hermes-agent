"""Profile and cron ownership use actual storage identity without invented DB paths."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_storage_identity_and_legacy_authority_home(tmp_path):
    from hermes_state_backend import session_store_home, session_store_key
    from gateway.session_authorities import authority_for_home
    identity = ('postgresql', 'tenant', 'workspace', 'source')
    hosted = SimpleNamespace(home=tmp_path, storage_identity=identity)
    authority = SimpleNamespace(db=hosted)
    runner = SimpleNamespace(session_authority=authority)
    assert session_store_home(hosted) == tmp_path.resolve()
    assert session_store_key(hosted) == identity
    assert authority_for_home(runner, tmp_path) is authority
    assert authority_for_home(runner, tmp_path/'other') is None
    sqlite = SimpleNamespace(db_path=tmp_path/'state.db')
    assert session_store_home(sqlite) == tmp_path.resolve()
    assert session_store_key(sqlite) == (tmp_path/'state.db').resolve()
    with pytest.raises(ValueError): session_store_key(SimpleNamespace(home=tmp_path, storage_identity=('postgresql',)))
    with pytest.raises(ValueError): session_store_home(SimpleNamespace())


def test_hosted_cron_owner_binds_only_actual_home(tmp_path):
    from gateway.session_cron import bind_owner, unbind_owner, owner_for_home
    authority = SimpleNamespace(db=SimpleNamespace(home=tmp_path, storage_identity=('postgresql', 'tenant', 'workspace', 'source')))
    async def check():
        bind_owner(authority)
        try:
            assert owner_for_home(tmp_path)[0] is authority
            assert owner_for_home(tmp_path/'foreign') is None
        finally: unbind_owner(authority)
        assert owner_for_home(tmp_path) is None
    asyncio.run(check())


def test_hosted_profile_config_and_bot_scope_use_real_home(tmp_path):
    from dataclasses import replace
    import json
    from gateway.session_bot import _home
    from gateway.session_contract import Principal
    from gateway.session_policy import LocalSessionPolicy
    from gateway.session_policy_credentials import config_reference, recover_config_secrets
    from gateway.session_mutation_model import _alias_credential
    from hermes_state_runtime import RuntimeStoreError
    (tmp_path/'config.yaml').write_text('model: local-owned-model\n')
    authority = SimpleNamespace(db=SimpleNamespace(home=tmp_path, storage_identity=('postgresql','tenant','workspace','source')), profile_id=str(tmp_path))
    actor = Principal('owner', str(tmp_path), frozenset({'session:submit'}), 'owned-test')
    assert _home(authority, actor, 'default') == tmp_path.resolve()
    with pytest.raises(RuntimeStoreError): _home(authority, replace(actor, profile_id='foreign'), 'default')
    policy = LocalSessionPolicy(source='cli',platform='local',cwd=str(tmp_path),model='local-owned-model',toolsets=(),config_json='{}',request_json=json.dumps({'follow_profile_config':True}),terminal_json='{}')
    assert policy.read_profile(authority)['model']['default']=='local-owned-model'
    reference = config_reference(authority,'session',policy,{})
    policy = replace(policy,config_secret_ref=reference)
    assert recover_config_secrets(authority,policy)=={}
    foreign = SimpleNamespace(db=SimpleNamespace(home=tmp_path/'foreign', storage_identity=('postgresql','foreign','workspace','source')),profile_id=str(tmp_path))
    with pytest.raises(RuntimeStoreError): recover_config_secrets(foreign,policy)
    assert _alias_credential(authority,SimpleNamespace(success=True,resolved_via_alias='missing')) == (None,None,None)


@pytest.fixture
def pg_settings():
    from tests.hermes_state.test_postgresql_sessions import PrivateDSN
    import json
    path = Path(__file__).resolve().parents[3]/'.artifacts/shared-pg-v1/candidate-pg-private.json'
    if not path.is_file(): pytest.skip('owned PostgreSQL fixture required')
    return {key:PrivateDSN(value) for key,value in json.loads(path.read_text())['dsns'].items()}


def test_native_history_claim_and_bot_retry_queries_use_actual_connection(tmp_path, pg_settings):
    import psycopg
    from uuid import uuid4
    from hermes_state_postgresql import PostgreSQLSessionDB, SessionScope, provision_scope
    from gateway.session_local_migration import require_history_claim, resolve_local_target
    from gateway.session_bot import _retry_admission
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    scope = SessionScope('tenant','workspace','profile-query-'+uuid4().hex)
    provision_scope(pg_settings['semantier_migrate'],scope,app_role='semantier_app')
    try:
        with PostgreSQLSessionDB(pg_settings['semantier_app'],scope,home=tmp_path) as db:
            db.create_session('history','cli')
            authority = SimpleNamespace(db=db,profile_id=str(tmp_path))
            actor = Principal('owner',str(tmp_path),frozenset({'session:read','session:create'}),'owned-ticket')
            with db._read_ctx() as conn:
                with pytest.raises(RuntimeStoreError,match='permission_denied'):
                    require_history_claim(authority,conn,actor,'history')
                authority._native_legacy_transports = {actor.transport_id:actor}
                require_history_claim(authority,conn,actor,'history')
            authority._native_legacy_transports.clear()
            with pytest.raises(RuntimeStoreError,match='not_found'):
                resolve_local_target(authority,actor,'history')
            assert _retry_admission(authority,{'session_id':'history','delivery_id':'owned-delivery'}) is None
            assert not (tmp_path/'state.db').exists()
    finally:
        with psycopg.connect(pg_settings['semantier_migrate'],autocommit=True) as conn:
            conn.execute(psycopg.sql.SQL('DROP SCHEMA {} CASCADE').format(psycopg.sql.Identifier(scope.schema)))
