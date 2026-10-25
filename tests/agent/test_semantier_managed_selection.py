"""Managed role and conversation bindings refuse drift before inference."""
import pytest

def test_scoped_auxiliary_assignment_refuses_explicit_provider_override(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump({'auxiliary':{'compression':{'provider':'custom','model':'authorized','base_url':'http://127.0.0.1:8899/v1','key_env':'HERMES_CUSTOM_SEMANTIER_CLOUD_TEST_API_KEY','account_id':'a'}}}))
    from agent.auxiliary_client import _resolve_task_provider_model
    with pytest.raises(ValueError,match='cannot be overridden'):
        _resolve_task_provider_model('compression',provider='alibaba',model='poison')

def test_conversation_binding_does_not_accept_an_unmanaged_secret_pointer(tmp_path,monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    from agent.semantier_gateway import session_cloud_route
    with pytest.raises(ValueError,match='unavailable'):
        session_cloud_route({'semantier_cloud':{'key_env':'SUPPLIER_KEY','base_url':'https://private.invalid/v1'}},'alias')


def test_managed_auxiliary_uses_frozen_assignment_after_profile_change(monkeypatch):
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux,'_read_auxiliary_task_config',lambda task:{'model':'account-b'})
    frozen={'cloud_route':'http://127.0.0.1:8899/v1','auxiliary_snapshot':{'compression':{'model':'account-a'}}}
    with aux.scoped_runtime_main(frozen):
        assert aux._get_auxiliary_task_config('compression')['model']=='account-a'
        assert aux._get_auxiliary_task_config('title_generation')=={}
    assert aux._get_auxiliary_task_config('compression')['model']=='account-b'


def test_cloud_config_redacts_nested_credentials_and_retains_server_pointers():
    from hermes_cli.web_server_config import _redact_managed_config
    assert _redact_managed_config({'fallback_chain':[{'api_key':'private','key_env':'POINTER'}]})=={'fallback_chain':[{'api_key':'[configured]','key_env':'POINTER'}]}


def test_cloud_request_refuses_revision_drift_before_dispatch(monkeypatch):
    from agent import semantier_gateway as cloud
    monkeypatch.setattr(cloud,'cloud_model_profile',lambda *args,**kwargs:{'revision':'new'})
    with pytest.raises(ValueError,match='selection changed'):
        cloud.configure_cloud_request({'model':'alias'},base_url='http://127.0.0.1:8899/v1',settings=cloud.GatewaySettings('http://127.0.0.1:8899/v1','private'),expected_revision='old')


def test_cloud_session_cannot_override_reserved_snapshot_from_request(monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter
    from agent import semantier_gateway as cloud
    monkeypatch.setattr(cloud,'session_cloud_route',lambda *args:{'_semantier_cloud_bound':True})
    adapter=object.__new__(APIServerAdapter)
    options={'semantier_cloud':{'account_id':'a'},'semantier_auxiliary':{'compression':{'account_id':'a'}}}
    session={'model_config':{'browser_model_lock':{'confirmed':True,'provider':'semantier-cloud','model':'alias','model_options':options}}}
    resolved=adapter._runtime_request_from_persisted_session_lock(session,{'model_options':{'semantier_auxiliary':{'compression':{'account_id':'b'}}}})
    assert resolved['model_options']==options


def test_unsupported_managed_purpose_cannot_inherit_cloud_but_custom_is_retained(monkeypatch):
    from agent import semantier_gateway as cloud
    monkeypatch.setattr(cloud,'is_gateway_route',lambda base:base=='https://cloud.invalid/v1')
    with pytest.raises(ValueError,match='unavailable through managed Cloud'):
        cloud.require_managed_auxiliary_purpose('review','https://cloud.invalid/v1')
    cloud.require_managed_auxiliary_purpose('review','http://localhost:8000/v1')
    cloud.require_managed_auxiliary_purpose('compression','https://cloud.invalid/v1')


def test_missing_managed_auxiliary_secret_never_borrows_main(monkeypatch):
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux,'_get_auxiliary_task_config',lambda task:{'account_id':'a','provider':'custom','model':'alias','base_url':'https://cloud.invalid/v1','key_env':'HERMES_CUSTOM_SEMANTIER_CLOUD_MISSING_API_KEY'})
    monkeypatch.setattr(aux,'_scoped_key_env',lambda name:'')
    with pytest.raises(ValueError,match='credential unavailable'):
        aux._resolve_task_provider_model('compression')


def test_vision_resolver_may_forward_the_exact_scoped_role_credential(monkeypatch):
    from agent import auxiliary_client as aux
    role={'account_id':'a','provider':'custom','model':'alias','base_url':'https://cloud.invalid/v1','key_env':'HERMES_CUSTOM_SEMANTIER_CLOUD_TEST_API_KEY'}
    monkeypatch.setattr(aux,'_get_auxiliary_task_config',lambda task:role)
    monkeypatch.setattr(aux,'_scoped_key_env',lambda name:'scoped-role-credential')
    resolved=aux._resolve_task_provider_model('vision',provider='custom',model='alias',base_url=role['base_url'],api_key='scoped-role-credential')
    assert resolved[3]=='scoped-role-credential'
    with pytest.raises(ValueError,match='cannot be overridden'):
        aux._resolve_task_provider_model('vision',api_key='another-account')

def test_retained_cloud_connection_is_recognized_after_custom_default(monkeypatch):
    from types import SimpleNamespace
    from agent import semantier_gateway as cloud
    from hermes_cli import config
    monkeypatch.setattr(cloud,'cloud_connection_config',lambda:None)
    monkeypatch.setattr(config,'load_config_readonly',lambda:{'model':{'provider':'explicit-custom'},'providers':{'semantier-cloud':{'base_url':'http://127.0.0.1:8899/v1'}}})
    agent=SimpleNamespace(base_url='http://127.0.0.1:8899/v1',api_key='retained-scoped-secret')
    cloud.capture_agent_route(agent)
    assert cloud.agent_cloud_settings(agent)==cloud.GatewaySettings(agent.base_url,agent.api_key)

@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['semantier-cloud','custom-fixture'])
async def test_agent_worker_resolves_existing_lock_before_later_default(monkeypatch,tmp_path,provider):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from agent import semantier_gateway as cloud
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    adapter=APIServerAdapter(PlatformConfig(enabled=True))
    options={'semantier_auxiliary':{'compression':{'account_id':'a'}}}
    session={'model_config':{'browser_model_lock':{'confirmed':True,'provider':provider,'model':'session-alias','model_options':options}}}
    monkeypatch.setattr(adapter,'_ensure_session_db',lambda:SimpleNamespace(get_session=lambda sid:session))
    monkeypatch.setattr(cloud,'session_cloud_route',lambda *args: {'provider':'custom','model':'session-alias','_semantier_cloud_bound':True} if provider=='semantier-cloud' else None)
    fake=MagicMock();fake.provider='custom';fake.model='session-alias';fake._hermes_api_runtime={'provider':'custom','model':'session-alias'}
    fake.run_conversation.return_value={'final_response':'ok'}
    fake.session_prompt_tokens=1;fake.session_completion_tokens=1;fake.session_total_tokens=2
    create=MagicMock(return_value=fake);monkeypatch.setattr(adapter,'_create_agent',create)
    await adapter._run_agent(user_message='hello',conversation_history=[],session_id='s',requested_model='later-default')
    saved=create.call_args.kwargs
    assert saved['route']['model']=='session-alias'
    assert saved['confirmed_runtime_lock'] is True
    assert saved['model_options']==options
