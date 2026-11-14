"""Cloud profile request configuration through the profile-owned production bridge."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
import yaml


def test_profile_scoped_catalog_and_actual_request_fields(tmp_path, monkeypatch):
    from hermes_cli.config import load_config_readonly
    from agent.semantier_gateway import cloud_catalog, configure_cloud_request
    seen=[]
    profile={'id':'private-alias','revision':'r1','context_window':8000,'max_output_tokens':100,
        'capabilities':{'tools':None,'vision':False,'reasoning':False,'structured_output':False,'stream':True},
        'request_parameters':{'output':'max_tokens','reasoning_controls':[]}}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            seen.append(self.headers['Authorization'])
            data=json.dumps({'object':'list','data':[profile]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/v1'
    monkeypatch.setenv('CLOUD_TEST_KEY','scoped-test-key')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    cfg={'semantier_gateway':{'enabled':True,'base_url':base,'api_key_env':'CLOUD_TEST_KEY'},
         'model':{'default':'private-alias','provider':'custom','base_url':base}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    try:
        assert cloud_catalog()['data'][0]['id']=='private-alias'
        assert cloud_catalog()['data'][0]['revision']=='r1' and len(seen)==1
        body=configure_cloud_request({'model':'private-alias','messages':[{'role':'user','content':'hello'}],
            'tools':[{'function':{'name':'test'}}],'reasoning_effort':'high','temperature':.7},base_url=base)
        assert body['max_tokens']==100 and 'tools' not in body and 'reasoning_effort' not in body
        assert body['temperature']==.7
        assert body['extra_headers']['X-Semantier-Capability-Revision']=='r1'
        from hermes_cli.model_switch import switch_model,apply_model_selection
        from hermes_cli.runtime_provider import resolve_runtime_provider
        # Native draft/session picker passes its display slug through these real owners.
        profile['protocol']='chat'
        from agent.semantier_gateway import clear_cloud_catalog
        clear_cloud_catalog()
        result=switch_model('private-alias','custom','old',current_base_url=base,explicit_provider='semantier-cloud')
        assert result.success and result.target_provider=='custom' and result.base_url==base
        saved=apply_model_selection(cfg['model'],result)
        assert saved['key_env']=='CLOUD_TEST_KEY' and 'api_key' not in saved and saved['capability_revision']=='r1'
        resolved=resolve_runtime_provider(requested='semantier-cloud',target_model='private-alias')
        assert resolved['provider']=='custom' and resolved['base_url']==base and resolved['api_key']=='scoped-test-key'
        embedded=resolve_runtime_provider(requested='custom',explicit_base_url=base,explicit_api_key='embedded-control-only',target_model='private-alias')
        assert embedded['api_key']=='scoped-test-key' and embedded['source']=='semantier-cloud'

        other=tmp_path/'other';other.mkdir();monkeypatch.setenv('HERMES_HOME',str(other))
        (other/'config.yaml').write_text(yaml.safe_dump(cfg))
        before=len(seen)
        cloud_catalog();assert len(seen)==before+1
        assert configure_cloud_request({'model':'byok'},base_url='http://127.0.0.1:2/v1')=={'model':'byok'}
    finally:
        server.shutdown();server.server_close();thread.join()


def test_cleared_scope_does_not_accept_a_late_catalog_response(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from agent.semantier_gateway import cloud_catalog,clear_cloud_catalog
    ready,release=Event(),Event();calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            calls.append(1);version=len(calls)
            if version==1:
                ready.set();assert release.wait(3)
            data=json.dumps({'object':'list','data':[{'id':'private','revision':str(version)}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('HERMES_HOME',str(tmp_path));monkeypatch.setenv('CLOUD_TEST_KEY','owned')
    (tmp_path/'config.yaml').write_text(yaml.safe_dump({'semantier_gateway':{'enabled':True,'base_url':f'http://127.0.0.1:{server.server_port}/v1','api_key_env':'CLOUD_TEST_KEY'}}))
    clear_cloud_catalog()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            old=pool.submit(cloud_catalog);assert ready.wait(3)
            clear_cloud_catalog()
            fresh=pool.submit(cloud_catalog)
            assert fresh.result(timeout=3)['data'][0]['revision']=='2'
            release.set();assert old.result(timeout=3)['data'][0]['revision']=='1'
        assert cloud_catalog()['data'][0]['revision']=='2' and len(calls)==2
    finally:
        release.set();server.shutdown();thread.join();server.server_close();clear_cloud_catalog()


def test_discovery_snapshot_is_shared_with_warm_execution_and_restart(tmp_path, monkeypatch):
    """Only the external catalog is a responder; metadata/request owners are real."""
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.desktop_model_discovery import read_cloud_snapshot
    from agent.semantier_gateway import GatewaySettings, configure_cloud_request, clear_cloud_catalog
    from agent.model_metadata import get_model_context_length
    from hermes_cli import desktop_model_discovery as discovery
    calls=[]
    profile={'id':'never-bundled-model','revision':'profile-1','deployment_revision':'deployment-1',
        'protocol':'openai_chat_completions','context_window':131072,'max_output_tokens':1200,
        'account_id':'account-one','provider_id':'OPENAI','source_model_id':'never-bundled-model',
        'capabilities':{'tools':False,'vision':False,'reasoning':False,'structured_output':False,'stream':True},
        'request_parameters':{'output':'max_tokens','reasoning_controls':[]},
        'sources':{'context_window':'deployment'},'unknown_reason':None}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            calls.append(self.path)
            data=json.dumps({'object':'list','data':[profile]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/v1';settings=GatewaySettings(base,'fixture-key')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path));monkeypatch.setenv('CLOUD_TEST_KEY','fixture-key')
    (tmp_path/'config.yaml').write_text(yaml.safe_dump({'semantier_gateway':{'enabled':True,'base_url':base,'api_key_env':'CLOUD_TEST_KEY'}}))
    clear_cloud_catalog()
    try:
        first=read_cloud_snapshot(settings,refresh=True)
        assert first.model_profiles[profile['id']]['deployment_revision']=='deployment-1'
        assert first.model_profiles[profile['id']]['sources']==profile['sources']
        def construct(_):
            assert get_model_context_length(profile['id'],base_url=base)==131072
            result=configure_cloud_request({'model':profile['id'],'messages':[{'role':'user','content':'fixture'}],'tools':[{'function':{'name':'unused'}}]},base_url=base)
            assert 'tools' not in result and result['max_tokens']==1200
            assert result['extra_headers']['X-Semantier-Capability-Revision']=='profile-1'
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(construct,range(20)))
        assert len(calls)==1
        # Process-local memory loss preserves the existing persisted owner snapshot.
        discovery._cloud_rows.clear()
        assert read_cloud_snapshot(settings).model_profiles[profile['id']]['context_window']==131072
        assert len(calls)==1
        profile.update(revision='profile-2',deployment_revision='deployment-2',context_window=262144)
        read_cloud_snapshot(settings,refresh=True)
        assert get_model_context_length(profile['id'],base_url=base)==262144
        assert len(calls)==2
    finally:
        server.shutdown();server.server_close();thread.join();clear_cloud_catalog()


@pytest.mark.parametrize('context,output',[(262144,40000),(64000,100)])
def test_explicit_cloud_limits_override_exact_bundled_defaults(monkeypatch,context,output):
    from agent import semantier_gateway as owner
    monkeypatch.setattr(owner,'_bundled_cloud_limits',lambda p:{'context_window':128000,'max_output_tokens':1024})
    profile={'id':'known','context_window':context,'max_output_tokens':output,
        'capabilities':{'tools':False,'vision':False,'stream':False}}
    result=owner.resolve_cloud_profile(profile)
    assert result['context_window']==context and result['max_output_tokens']==output
    assert result['capabilities']['tools'] is False and result['capabilities']['stream'] is False


def test_absent_cloud_field_can_use_only_exact_verified_default(monkeypatch):
    from agent import semantier_gateway as owner
    monkeypatch.setattr(owner,'_bundled_cloud_limits',lambda p:{'context_window':128000})
    result=owner.resolve_cloud_profile({'id':'known','max_output_tokens':1024})
    assert result['context_window']==128000 and result['sources']['context_window']=='exact_local_catalog'


@pytest.mark.parametrize('value',[None,0,-1,True,'8192'])
def test_explicit_unknown_or_invalid_cloud_limit_never_uses_default(monkeypatch,value):
    from agent import semantier_gateway as owner
    monkeypatch.setattr(owner,'_bundled_cloud_limits',lambda p:{'context_window':128000})
    with pytest.raises(ValueError,match='context_window'):
        owner.resolve_cloud_profile({'id':'known','context_window':value,'max_output_tokens':1024})


def test_unknown_cloud_model_does_not_get_a_family_limit(monkeypatch):
    from agent import semantier_gateway as owner
    from agent import models_dev
    from types import SimpleNamespace
    monkeypatch.setattr(models_dev,'get_model_info',lambda *a,**kw:SimpleNamespace(id='known-family',provider_id='openai',context_window=128000,max_output=1024))
    with pytest.raises(ValueError,match='context_window'):
        owner.resolve_cloud_profile({'id':'unknown-alias','provider_id':'OPENAI','source_model_id':'known-family-unseen','max_output_tokens':1024})


def test_cloud_request_uses_server_output_and_separate_input_ceiling(monkeypatch):
    from agent import semantier_gateway as owner
    p={'id':'new-id','revision':'r1','context_window':262144,'max_output_tokens':40000,
       'max_input_tokens':8192,'capabilities':{'stream':False,'tools':False},
       'request_parameters':{'output':'max_tokens','reasoning_controls':[]}}
    monkeypatch.setattr(owner,'cloud_model_profile',lambda *a,**kw:p)
    settings=owner.GatewaySettings('https://fixture.test/v1','fixture-key')
    body={'model':'new-id','messages':[{'role':'user','content':'small'}]}
    result=owner.configure_cloud_request(body,base_url=settings.base_url,settings=settings)
    assert result['max_tokens']==40000 and result['stream'] is False
    with pytest.raises(ValueError,match='input budget'):
        owner.configure_cloud_request({**body,'messages':[{'role':'user','content':'x'*9000}]},base_url=settings.base_url,settings=settings)


def test_context_budget_uses_bound_session_credential_snapshot(tmp_path,monkeypatch):
    from hermes_cli import desktop_model_discovery as discovery
    from agent.semantier_gateway import GatewaySettings,clear_cloud_catalog
    from agent.model_metadata import get_model_context_length
    monkeypatch.setenv('HERMES_HOME',str(tmp_path));monkeypatch.setenv('CLOUD_TEST_KEY','current')
    base='https://fixture.test/v1'
    (tmp_path/'config.yaml').write_text(yaml.safe_dump({'semantier_gateway':{'enabled':True,'base_url':base,'api_key_env':'CLOUD_TEST_KEY'}}))
    reads=[]
    def query(endpoint,key):
        reads.append(key)
        return ['same-alias'],{'same-alias':{'id':'same-alias','account_id':key,'revision':key,
            'context_window':256000 if key=='bound-session' else 128000,'max_output_tokens':1024,
            'capabilities':{},'protocol':'chat','request_parameters':{'output':'max_tokens','reasoning_controls':[]}}}
    monkeypatch.setattr(discovery,'_cloud_snapshot_values',query)
    clear_cloud_catalog()
    discovery.read_cloud_snapshot(GatewaySettings(base,'current'))
    discovery.read_cloud_snapshot(GatewaySettings(base,'bound-session'))
    for _ in range(20):assert get_model_context_length('same-alias',base_url=base,api_key='bound-session')==256000
    assert len(reads)==2


@pytest.mark.parametrize('requested,expected',[(None,16000),(1000,1000),(50000,16000)])
def test_cloud_compaction_reserves_the_same_server_output_budget(monkeypatch,requested,expected):
    from types import SimpleNamespace
    from agent.agent_init import _compressor_max_tokens
    from agent import semantier_gateway as owner
    settings=owner.GatewaySettings('https://fixture.test/v1','bound-session')
    monkeypatch.setattr(owner,'agent_cloud_settings',lambda agent:settings)
    monkeypatch.setattr(owner,'cloud_model_profile',lambda model,**kw:{'max_output_tokens':16000})
    agent=SimpleNamespace(model='new-model',provider='custom',base_url=settings.base_url,api_key=settings.api_key,max_tokens=requested)
    assert _compressor_max_tokens(agent)==expected
