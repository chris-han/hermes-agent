"""Native source-only DPAPI/restart probe. Never installer or joined acceptance."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--read',action='store_true')
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--chat',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--task-id',default='windows-source-explicit-chat')
    parser.add_argument('--title',action='store_true')
    parser.add_argument('--expect-denied',action='store_true')
    parser.add_argument('--account-map',type=Path)
    args = parser.parse_args()
    assert os.name == 'nt', 'real Windows is required'
    source=args.source.resolve();home=args.root.resolve()
    sys.path.insert(0,str(source))
    os.environ['HERMES_HOME']=str(home)
    from hermes_cli import cloud_device_client as device
    if args.status:
        print(json.dumps(device.status()))
        return
    if args.account_map:
        import httpx
        from openai import OpenAI, APIStatusError
        from agent.semantier_gateway import gateway_settings, attribute_request, configure_cloud_request
        fixture_path=source.parent/'tests/fixtures/cloud-task-finops-v1/contract.json'
        fixture=json.loads(fixture_path.read_text())
        mapped=json.loads(args.account_map.read_text())
        replacements={fixture['account_override_authority'][key]:value for key,value in mapped.items()}
        def translate(value):
            if isinstance(value,dict):return {k:translate(v) for k,v in value.items()}
            if isinstance(value,str):return replacements.get(value,value)
            return value
        settings=gateway_settings()
        results=[]
        with OpenAI(base_url=settings.base_url,api_key=settings.api_key,max_retries=0,
                http_client=httpx.Client(trust_env=False,timeout=15)) as client:
            for case in fixture['account_override_cases']:
                try:
                    attribute_request({},base_url=settings.base_url,**translate(case['correlation']))
                except TypeError as error:
                    assert "account_id" in str(error)
                else:raise AssertionError('untrusted account entered native correlation')
                wire=attribute_request({'model':'device-controlled','messages':[{'role':'user','content':'Controlled account metadata rejection.'}],
                    'max_tokens':16,**translate(case['metadata'])},base_url=settings.base_url,
                    **fixture['cases'][0]['correlation'])
                wire=configure_cloud_request(wire,base_url=settings.base_url)
                assert wire['extra_headers']['X-Semantier-Account-Id']==mapped['bound_account_id']
                try:client.chat.completions.create(**wire)
                except APIStatusError as error:
                    assert error.status_code==case['expected']['metadata_http_status']
                    assert error.response.json()['detail']==case['expected']['metadata_detail']
                    results.append({'case':case['id'],'status':error.status_code,'detail':error.response.json()['detail'],
                        'forged_account_id':translate(case['correlation'])['account_id'],
                        'bound_account_id':mapped['bound_account_id'],'untrusted_correlation_refused':True})
                else:raise AssertionError('account metadata override accepted')
        print(json.dumps({'produced_at':datetime.now(UTC).isoformat(),'scope':'REAL_NATIVE_ACCOUNT_OVERRIDE_NEGATIVES',
            'fixture_revision':fixture['fixture_revision'],'fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            'cases':results,'package_acceptance':False}))
        return
    if args.chat:
        profile=home.parent/'profile'
        for key,folder in {'HOME':profile,'USERPROFILE':profile,'APPDATA':profile/'AppData/Roaming','LOCALAPPDATA':profile/'AppData/Local','TEMP':home.parent/'tmp','TMP':home.parent/'tmp'}.items():
            folder.mkdir(parents=True,exist_ok=True)
            os.environ[key]=str(folder)
        os.chdir(home)
        for key in list(os.environ):
            if key.endswith(('_API_KEY','_TOKEN','_SECRET','_PASSWORD','_CREDENTIALS')):
                os.environ.pop(key,None)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from run_agent import AIAgent
        runtime=resolve_runtime_provider(requested='semantier-cloud',target_model='device-controlled')
        runtime.pop('source',None)
        agent=AIAgent(model='device-controlled',**runtime,max_iterations=1,enabled_toolsets=[],
            quiet_mode=True,skip_context_files=True,skip_memory=True,skip_background_review=True)
        observed=[]
        native_call=agent._interruptible_api_call
        def observe_native_call(*args,**kwargs):
            try:
                response=native_call(*args,**kwargs)
            except Exception as error:
                observed.append(getattr(error,'status_code',type(error).__name__))
                raise
            observed.append('completed')
            return response
        agent._interruptible_api_call=observe_native_call
        result=agent.run_conversation('Controlled native source acceptance. Reply briefly.',task_id=args.task_id)
        receipt={'scope':'WINDOWS_NATIVE_SOURCE_ONLY','produced_at':datetime.now(UTC).isoformat(),
            'task':agent._semantier_task_context,'native_call_results':observed,'package_acceptance':False}
        if args.expect_denied:
            assert 402 in observed and 'Device controlled response.' not in str(result.get('final_response','')), 'native cap denial absent'
            receipt['cap_denied']=True
        else:
            assert 'Device controlled response.' in str(result.get('final_response','')), 'controlled native response absent'
            receipt['explicit_chat']=True
            if args.title:
                import uuid
                from agent.auxiliary_client import call_llm
                frozen={key:getattr(agent,key) for key in ('provider','model','base_url','api_key','session_id')}
                frozen.update(agent._semantier_task_context,parent_run_id=agent._semantier_task_context['run_id'],run_id='title-'+uuid.uuid4().hex,cloud_route=agent.base_url)
                options=dict(provider='custom',model='device-controlled',base_url=agent.base_url,api_key=agent.api_key,
                    main_runtime=frozen,messages=[{'role':'user','content':'Controlled title acceptance.'}],max_tokens=16)
                title=call_llm('title_generation',**options)
                assert 'Device controlled response.' in title.choices[0].message.content
                receipt['title']=True
                try:
                    call_llm('compression',**options)
                except ValueError as error:
                    assert 'primary chat and title' in str(error)
                    receipt['unqualified_compression_refused']=True
                else:
                    raise AssertionError('unqualified native auxiliary caller dispatched')
        print(json.dumps(receipt))
        return
    if args.inspect:
        state=device._read()
        result={k:state.get(k) for k in ('status','origin','portal_origin','expires_at','credential_ref','account_id')}
        result['credential_count']=len(state.get('credentials',{}))
        try:
            result['poll']=device.poll(state['session_id'])
        except ValueError as error:
            result['error_type']=type(error).__name__
            result['error']=str(error)
        print(json.dumps(result))
        return
    if args.read:
        state=device._read()
        assert state['proof']=='disposable-native-custody-probe'
        print(json.dumps({'restart_read':True}))
        return
    home.mkdir(parents=True,exist_ok=False)
    (home/'config.yaml').write_text('{}\n')
    original={'proof':'disposable-native-custody-probe','session_id':'cloud-source-probe','status':'pending'}
    device._write(original)
    ciphertext=(home/'.cloud-device.bin').read_bytes()
    assert original['proof'].encode() not in ciphertext
    assert device._read()['proof']==original['proof']
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--root',str(home),'--read'],capture_output=True,text=True,check=True)
    assert json.loads(child.stdout)['restart_read'] is True
    (home/'.cloud-device.bin').write_bytes(ciphertext[:-8]+b'tampered')
    try:
        device._read()
    except ValueError:
        pass
    else:
        raise AssertionError('tampered ciphertext accepted')
    print(json.dumps({'scope':'WINDOWS_NATIVE_SOURCE_ONLY','dpapi_encrypted':True,
        'process_restart_read':True,'tamper_rejected':True,
        'module_sha256':hashlib.sha256((source/'hermes_cli/cloud_device_client.py').read_bytes()).hexdigest(),
        'python':sys.version.split()[0],'package_acceptance':False}))


if __name__=='__main__':
    main()
