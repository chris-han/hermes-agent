"""Real Windows source lifecycle/fault probe; never package acceptance."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--fixture-root',type=Path,required=True)
parser.add_argument('--operation',choices=['lifecycle','fault','expiry','runtime'],required=True)
args=parser.parse_args()
assert os.name=='nt'
source=args.source.resolve();home=args.root.resolve()
assert home.parent.parent.name=='windows-cloud-client-r1'
os.environ['HERMES_HOME']=str(home)
profile=home.parent/'profile'
for key,folder in {'HOME':profile,'USERPROFILE':profile,'APPDATA':profile/'AppData/Roaming','LOCALAPPDATA':profile/'AppData/Local','TEMP':home.parent/'tmp','TMP':home.parent/'tmp'}.items():
    folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
sys.path.insert(0,str(source))
from hermes_cli import cloud_device_client as device


def child_command(values):
    payload=os.environ.get('HERMES_QUALIFICATION_PAYLOAD')
    if not payload:
        return [sys.executable,*values]
    packages=str(Path(payload)/'venv/Lib/site-packages')
    if values[0]=='-c':
        prefix='import sys;sys.path[:0]='+repr([str(source),packages])+';'
        return [sys.executable,'-I','-c',prefix+values[1],*values[2:]]
    return [sys.executable,'-I',os.environ['HERMES_QUALIFICATION_RUNNER'],'--payload',payload,
        '--script',values[0],'--',*values[1:]]

if args.operation=='fault':
    from hermes_cli import config
    original=config.atomic_config_replace
    def crash_after_real_publication(*values,**options):
        original(*values,**options)
        os._exit(77)
    config.atomic_config_replace=crash_after_real_publication
    device.poll(device._read()['session_id'])
    raise AssertionError('fault publication was not reached')

home.mkdir(parents=True,exist_ok=False)
(home.parent/'.owner.json').write_text(json.dumps({'owner':'windows-cloud-client-r1'}))
local={'base_url':'http://127.0.0.1:55479/v1','api_key':'fixture-only-local-preserved'}
initial={'model':{'provider':'local-preserved','base_url':local['base_url'],'default':'local-model'},
    'providers':{'local-preserved':local,'semantier-cloud':{'base_url':'http://127.0.0.1:55474/v1','portal_origin':'http://127.0.0.1:3358'}},'memory':{'enabled':True}}
(home/'config.yaml').write_text(json.dumps(initial))
sentinels={}
for name in ['memories/fixture.md','skills/fixture/SKILL.md','history-preservation-sentinel.txt']:
    path=home/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('Owned retention fixture only.\n')
    sentinels[name]=hashlib.sha256(path.read_bytes()).hexdigest()
original_config=(home/'config.yaml').read_bytes()
import httpx
identity=json.loads((args.fixture_root/'identity.json').read_text())
account=json.loads((args.fixture_root/'account.json').read_text())['account_id']
checks=[]
with httpx.Client(base_url='http://127.0.0.1:55474',trust_env=False,timeout=20) as browser:
    assert browser.post('/auth/password/login',json={k:identity[k] for k in ('login','password')}).status_code==200
    def decide(started,decision,selected=account,expected=200):
        intent=parse_qs(urlsplit(started['verification_url']).query)['intent_ref'][0]
        reply=browser.post('/api/cloud-provider/device/'+intent+'/approve',json={'contract_version':device.CONTRACT,'account_id':selected,'decision':decision},headers={'origin':'http://127.0.0.1:55474','x-semantier-csrf':'1'})
        assert reply.status_code==expected,reply.text
        return intent
    started=device.start()
    private=device._read()
    assert private['proof'].encode() not in (home/'.cloud-device.bin').read_bytes()
    assert private['proof'] not in json.dumps(started)
    if args.operation=='runtime':
        decide(started,'approve')
        other=home.parent.parent/'runtime-other-user'/'hermes-home'
        other.mkdir(parents=True,exist_ok=False)
        (other.parent/'.owner.json').write_text(json.dumps({'owner':'windows-cloud-client-r1'}))
        (other/'config.yaml').write_text(json.dumps(initial))
        other_before=(other/'config.yaml').read_bytes()
        probe = """import json,sys
sys.path.insert(0,sys.argv[1])
from hermes_cli import cloud_device_client as device


def child_command(values):
    payload=os.environ.get('HERMES_QUALIFICATION_PAYLOAD')
    if not payload:
        return [sys.executable,*values]
    packages=str(Path(payload)/'venv/Lib/site-packages')
    if values[0]=='-c':
        prefix='import sys;sys.path[:0]='+repr([str(source),packages])+';'
        return [sys.executable,'-I','-c',prefix+values[1],*values[2:]]
    return [sys.executable,'-I',os.environ['HERMES_QUALIFICATION_RUNNER'],'--payload',payload,
        '--script',values[0],'--',*values[1:]]
try:device.poll(sys.argv[2])
except ValueError:print(json.dumps({'foreign_runtime_refused':True}))
else:raise AssertionError('foreign runtime consumed authorization')
"""
        child_env={**os.environ,'HERMES_HOME':str(other)}
        foreign=subprocess.run(child_command(['-c',probe,str(source),started['session_id']]),env=child_env,capture_output=True,text=True,timeout=45,check=False)
        assert foreign.returncode==0,foreign.stderr[-500:]
        assert json.loads(foreign.stdout)['foreign_runtime_refused']
        assert (home/'config.yaml').read_bytes()==original_config
        assert (other/'config.yaml').read_bytes()==other_before
        assert not (other/'.cloud-device.bin').exists()
        resume="""import json,sys
sys.path.insert(0,sys.argv[1])
from hermes_cli import cloud_device_client as device


def child_command(values):
    payload=os.environ.get('HERMES_QUALIFICATION_PAYLOAD')
    if not payload:
        return [sys.executable,*values]
    packages=str(Path(payload)/'venv/Lib/site-packages')
    if values[0]=='-c':
        prefix='import sys;sys.path[:0]='+repr([str(source),packages])+';'
        return [sys.executable,'-I','-c',prefix+values[1],*values[2:]]
    return [sys.executable,'-I',os.environ['HERMES_QUALIFICATION_RUNNER'],'--payload',payload,
        '--script',values[0],'--',*values[1:]]
started=device.start()
assert started['session_id']==sys.argv[2]
print(json.dumps({'status':device.poll(started['session_id'])['status']}))
"""
        resumed=subprocess.run(child_command(['-c',resume,str(source),started['session_id']]),capture_output=True,text=True,timeout=90,check=False)
        assert resumed.returncode==0,resumed.stderr[-500:]
        assert json.loads(resumed.stdout)['status']=='approved'
        assert (other/'config.yaml').read_bytes()==other_before
        assert device.disconnect()['remote_revoked']
        checks=['separate_windows_runtime_cannot_consume_foreign_session','foreign_runtime_no_credential_or_config_publication','original_runtime_new_process_resumes_same_intent','original_runtime_only_installation','remote_revocation_confirmed']
    elif args.operation=='expiry':
        print(json.dumps({'stage':'waiting_real_server_ttl','expires_at':private['expires_at']}),flush=True)
        while device._now() <= device._time(private['expires_at'])+.3:
            time.sleep(min(15,max(.1,device._time(private['expires_at'])+.4-device._now())))
        assert device.poll(started['session_id'])['status']=='expired'
        intent=private['intent_ref']
        view=browser.get('/api/cloud-provider/device/'+intent)
        assert view.status_code==200 and view.json()['status']=='expired'
        assert (home/'config.yaml').read_bytes()==original_config
        checks=['real_600_second_native_and_server_expiry','no_credential_or_config_publication']
    else:
        assert device.cancel(started['session_id'])['remote_cancelled']
        assert device.poll(started['session_id'])['status']=='cancelled'
        assert (home/'config.yaml').read_bytes()==original_config
        checks.append('native_cancel_and_terminal_fence')
        denied=device.start();decide(denied,'deny')
        assert device.poll(denied['session_id'])['status']=='denied'
        assert (home/'config.yaml').read_bytes()==original_config
        checks.append('real_browser_owner_denial_no_publication')
        install=device.start()
        decide(install,'approve','windows-inaccessible-r1',403)
        assert (home/'config.yaml').read_bytes()==original_config
        decide(install,'approve')
        child=subprocess.run(child_command([str(Path(__file__).resolve()),'--source',str(source),'--root',str(home),'--fixture-root',str(args.fixture_root),'--operation','fault']),capture_output=True,text=True,timeout=45,check=False)
        assert child.returncode==77,(child.returncode,child.stderr[-500:])
        assert device._read()['status']=='installing'
        staged_ref=device._read()['credential_ref']
        recovered=device.poll(install['session_id'])
        assert recovered['status']=='approved'
        assert device._read()['credential_ref']==staged_ref
        assert device._config()['providers']['local-preserved']==local
        checks.extend(['foreign_approval403_no_publication','real_process_death_after_atomic_config_publication','same_generation_claim_recovery_and_ack'])
        from agent.semantier_gateway import gateway_settings
        settings=gateway_settings()
        assert device._wire('http://127.0.0.1:55474','renew',bearer=settings.api_key)['status']=='reauthorization_required'
        assert device._wire('http://127.0.0.1:55474','revoke',bearer=settings.api_key)['status']=='revoked'
        private=device._read()
        time.sleep(max(0,min(31,device._time(private['next_readiness'])-device._now()+.3)))
        assert not device.status()['logged_in']
        try:device.credential(staged_ref,base_url=settings.base_url)
        except ValueError:pass
        else:raise AssertionError('revoked native credential remained usable')
        checks.extend(['renew_requires_explicit_reauthorization','actual_remote_revocation_readiness_and_local_key_retirement'])
        reauth=device.start();decide(reauth,'approve')
        assert device.poll(reauth['session_id'])['status']=='approved'
        assert device.status()['logged_in']
        assert device.disconnect()['remote_revoked']
        assert not device.status()['logged_in']
        assert device._config()['providers']['local-preserved']==local
        checks.extend(['same_paid_account_reauthorization_without_purchase','native_disconnect_confirmed_remote_revocation'])
for name,digest in sentinels.items():assert hashlib.sha256((home/name).read_bytes()).hexdigest()==digest
print(json.dumps({'produced_at':datetime.now(UTC).isoformat(),'scope':'REAL_WINDOWS_NATIVE_SOURCE_LIFECYCLE','checks':checks,'sentinels_preserved':list(sentinels),'purchase_calls':0,'inference_calls':0,'package_acceptance':False}))
