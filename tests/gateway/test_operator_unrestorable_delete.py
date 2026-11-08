"""A verified profile operator can retire an empty failed creation without executing it."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.asyncio
async def test_operator_retires_unrestorable_creation_with_existing_transaction_fences(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import stage_local_session
    from gateway.session_mutations import mutate_session
    from hermes_state_local import commit_local_session
    from hermes_state_mutation_retirement import retired_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store=SessionStore(tmp_path/'sessions',GatewayConfig())
    runner=SimpleNamespace(session_store=store,_session_db=store._db,adapters={},_draining=False)
    authority=await initialize_session_authority(runner,profile_id=str(tmp_path),instance_id='fixture')
    owner=Principal('original-owner',str(tmp_path),frozenset({'session:create','session:read','session:control'}),'original')
    receipt=stage_local_session(authority,owner,{'request_id':'failed-creation','source':'cli','cwd':str(tmp_path)})
    assert receipt['policy']['model']==''
    commit_local_session(authority.db,epoch=authority.epoch,receipt=receipt)
    ref=SessionRef(authority.profile_id,receipt['session_id'])
    operator=replace(owner,subject='verified-operator',capabilities=owner.capabilities|{'session:operator'})
    params=dict(session_id=ref.session_id,request_id='retire-empty',expected_revision=0,expected_generation=0,operation='delete',payload={})
    try:
        for denied in [replace(operator,profile_id='foreign'),replace(operator,capabilities=operator.capabilities-{'session:control'}),replace(operator,capabilities=operator.capabilities-{'session:operator'})]:
            with pytest.raises(RuntimeStoreError):await mutate_session(authority,denied,ref,params)
            assert authority.db.get_session(ref.session_id) is not None
        authority.retired=True
        with pytest.raises(RuntimeStoreError,match='runtime_retired'):
            await mutate_session(authority,operator,ref,params)
        authority.retired=False
        with pytest.raises(RuntimeStoreError,match='revision_conflict'):
            await mutate_session(authority,operator,ref,{**params,'request_id':'stale','expected_revision':1})
        with pytest.raises(RuntimeStoreError,match='stale_generation'):
            await mutate_session(authority,operator,ref,{**params,'request_id':'stale-generation','expected_generation':1})
        result=await mutate_session(authority,operator,ref,params)
        assert ref.session_id in result['deleted_ids']
        assert authority.db.get_session(ref.session_id) is None
        assert retired_session(authority.db,ref.session_id)
        assert await mutate_session(authority,operator,ref,params)==result
    finally:store._db.close()
