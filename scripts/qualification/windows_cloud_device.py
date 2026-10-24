"""Native source-only DPAPI/restart probe. Never installer or joined acceptance."""
import argparse
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
    args = parser.parse_args()
    assert os.name == 'nt', 'real Windows is required'
    source=args.source.resolve();home=args.root.resolve()
    sys.path.insert(0,str(source))
    os.environ['HERMES_HOME']=str(home)
    from hermes_cli import cloud_device_client as device
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
