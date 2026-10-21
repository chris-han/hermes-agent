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
