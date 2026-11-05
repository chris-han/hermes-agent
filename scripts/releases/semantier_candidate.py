"""Stage signed native CI output; never promote a channel or weaken full-matrix admission."""
import argparse
import hashlib
import hmac
import json
import os
import re
import stat
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
CI_DELIVERABLE_ROOT=Path('/mnt/d/repo/deliverables')
sys.path.insert(0,str(ROOT))
from scripts.releases.semantier_windows import is_acceptance_request, validate_native_graph
from scripts.bundles.release_artifacts import write_appinstaller


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def native_path(value):return Path(subprocess.check_output(['wslpath','-u',value],text=True).strip())

def copy_artifact(source, destination):
    # WSL sendfile can fail with ENOMEM after a partial multi-GB DrvFS copy.
    # Bound userspace memory and preserve exclusive immutable staging creation.
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def admit_target_binding(path):
    """Submit the staged binding to the existing authenticated owner after CI."""
    from scripts.releases import semantier_native_ci as ci
    path=Path(path)
    run=re.fullmatch(r'SemantierAgent-client-feedback-ci-([0-9]+)-([0-9]+)',path.parent.name)
    if (path.name!='channel2-target-binding.json' or not run or
        path.parent.parent!=CI_DELIVERABLE_ROOT or path.parent.is_symlink()):
        raise ValueError('Unexpected target binding path')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>2*1024*1024:raise ValueError('Invalid target binding file')
        raw=stream.read(2*1024*1024+1);after=os.fstat(stream.fileno())
    named=path.lstat()
    if not stat.S_ISREG(named.st_mode) or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or (after.st_dev,after.st_ino)!=(named.st_dev,named.st_ino):
        raise ValueError('Target binding changed during read')
    binding=json.loads(raw)
    receipt=binding.get('descriptor_receipt',{})
    if binding.get('delivery_target_id')!='channel2' or binding.get('candidate',{}).get('manifest',{}).get('request',{}).get('publicBase')!='https://update2.semantier.com' or str(receipt.get('workflow_run_id',''))!=run.group(1) or receipt.get('workflow_run_attempt')!=int(run.group(2)):
        raise ValueError('Cloudflare target binding required')
    candidate=binding['candidate']
    expected_variant=hashlib.sha256(canonical({'target_id':'channel2','manifest':{**candidate['manifest'],'notes':candidate['notes']},
        'descriptor_receipt':receipt,'native_receipt':candidate['native_receipt']})).hexdigest()
    token=os.environ.get('SEMANTIER_CLIENT_CI_CAPABILITY','')
    if not token:raise ValueError('CI principal capability unavailable')
    parts=subprocess.check_output(['docker','inspect',ci.CI_RUNTIME_CONTAINER,'--format','{{.Id}} {{.Image}} {{.State.Running}}'],text=True).strip().split()
    if len(parts)!=3 or not re.fullmatch('[a-f0-9]{64}',parts[0]) or parts[1]!=ci.CI_RUNTIME_IMAGE or parts[2]!='true':raise ValueError('Runtime identity unavailable')
    mounts=json.loads(subprocess.check_output(['docker','inspect',parts[0],'--format','{{json .Mounts}}'],text=True))
    if not any(m.get('Source')==str(ci.CI_INCOMING_ROOT.parents[1]) and m.get('Destination')=='/fixture' for m in mounts) or any(m.get('Destination')=='/app' or str(m.get('Destination','')).startswith('/app/') for m in mounts):raise ValueError('Runtime custody mount mismatch')
    ci.verify_runtime_identity(parts[0])
    ci.verify_cloudflare_custody(parts[0],ci.CI_INCOMING_ROOT)
    program='''import sys,json,http.client,os,pathlib
argv=pathlib.Path('/proc/1/cmdline').read_bytes().split(b'\\0')
if argv[1:7]!=[b'-m',b'uvicorn',b'apps.combined:app',b'--host',b'0.0.0.0',b'--port'] or argv[7:8]!=[b'8899'] or not os.path.samefile('/proc/1/exe',sys.executable):
 raise SystemExit('Runtime entrypoint mismatch')
owned={p.readlink().name[8:-1] for p in pathlib.Path('/proc/1/fd').iterdir() if p.is_symlink() and p.readlink().name.startswith('socket:[')}
listening=[]
for table in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in pathlib.Path(table).read_text().splitlines()[1:]:
  row=line.split()
  if row[3]=='0A' and int(row[1].split(':')[1],16)==8899 and row[9] in owned:listening.append(row[9])
if len(listening)!=1:raise SystemExit('Runtime listener mismatch')
x=json.load(sys.stdin)
c=http.client.HTTPConnection('127.0.0.1',8899,timeout=2700)
c.request('POST','/api/client-operations/delivery-targets/bindings',body=json.dumps(x['binding']).encode(),headers={'Authorization':'Bearer '+x['token'],'Content-Type':'application/json'})
r=c.getresponse();data=r.read(2097153)
print(json.dumps({'status':r.status,'body':json.loads(data) if len(data)<=2097152 else {}}))'''
    try:
        result=subprocess.run(['docker','exec','-i',parts[0],'python','-c',program],input=json.dumps({'token':token,'binding':binding}),
                              text=True,capture_output=True,timeout=2760,check=True)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
        raise RuntimeError('Target admission outcome UNKNOWN; reconcile before retry via the existing delivery-targets catalog for release '+candidate['release_id']+' and variant '+expected_variant) from error
    try:
        response=json.loads(result.stdout)
        admitted=response['body']['binding']
        scope=response['body']['authorized_scope']
        if not all(isinstance(value,dict) for value in (response,admitted,scope)):
            raise ValueError('Invalid target admission response')
    except (ValueError,TypeError,KeyError):
        raise RuntimeError('Target admission outcome UNKNOWN; reconcile before retry via the existing delivery-targets catalog for release '+candidate['release_id']+' and variant '+expected_variant) from None
    if response.get('status')!=200 or admitted.get('release_id')!=candidate['release_id'] or admitted.get('target_id')!='channel2' or admitted.get('variant_hash')!=expected_variant or scope.get('channel')!=candidate['manifest']['request']['channel'] or scope.get('platform')!='win32' or scope.get('arch')!='x64':
        raise ValueError('Target admission not confirmed; reconcile before retry')
    return {'release_id':admitted['release_id'],'target_id':'channel2','variant_hash':admitted['variant_hash'],'state':'target_binding_admitted_not_published'}


def stage(output,native,incoming,notes,target_incoming=None):
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='chris-han/hermes-agent':raise ValueError('Real governed CI context required')
    result=read(output/'result.json');request=read(output/'channel-request.json');signed=read(output/'signed/receipt.json')
    if not is_acceptance_request(request) or result['result']!='PASS' or request['commit']!=os.environ['GITHUB_SHA'] or result['sourceCommit']!=request['commit']:raise ValueError('Exact completed source required')
    controller=os.environ.get('ACCEPTANCE_CONTROLLER_COMMIT','')
    if not re.fullmatch('[0-9a-f]{40}',controller) or request.get('controllerCommit')!=controller or result.get('controllerCommit')!=controller:
        raise ValueError('Exact completed controller required')
    if 'nativeGraph' in signed:
        validate_native_graph(signed['nativeGraph'],build_id=request['buildId'],
            identity=request['identity']['msixAppIdWithOrg'],publisher=signed['publisher'],
            version=request['windowsVersion'],arch='x64')
        from scripts.releases.semantier_native_candidate import stage_native_graph
        if target_incoming is None:raise ValueError('Cloudflare target incoming required')
        return stage_native_graph(output,native,incoming,notes,request,signed,target_incoming)
    installed=read(native/'installed-identity.json');exit_result=read(native/'native-install-exit.json');artifact_meta=read(native/'artifact.json')
    if exit_result.get('exitCode')!=0 or signed.get('signing')!='PASS' or not signed.get('verification','').startswith('PASS:'):raise ValueError('Native installation and signature gates required')
    if signed['packageName']!=request['identity']['msixAppIdWithOrg'] or signed['publisher']!='CN=Semantier Technology, O=Semantier Technology' or signed['packageVersion']!=request['windowsVersion']:raise ValueError('Package identity mismatch')
    if installed['version']!=signed['packageVersion'] or installed['publisher']!=signed['publisher'] or installed['stamp']['commit']!=request['commit']:raise ValueError('Installed source mismatch')
    distribution=output/'signed/distribution';files=list(distribution.glob('*.msix'))
    if len(files)!=1 or files[0].is_symlink():raise ValueError('One signed MSIX required')
    artifact=files[0];digest=sha(artifact)
    if digest!=signed['signedSha256'] or digest!=artifact_meta['sha256']:raise ValueError('Artifact bytes mismatch')
    # Reuse the Hermes-owned publisher's chunk contract. The composition
    # verification independently binds the frozen Runtime payload.
    from scripts.bundles.parent_composition import verify
    composition = ROOT/'semantier-runtime'
    verify(composition)
    from client_operations.chunks import prepare_chunks
    prefix='releases/channel-builds/'+request['buildId']+'/'
    key=prefix+artifact.name;feed_key=prefix+'win32/stable.appinstaller'
    target=incoming/key;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('Fresh immutable staging namespace required')
    copy_artifact(artifact,target)
    if sha(target)!=digest:raise ValueError('Staged artifact mismatch')
    feed=incoming/feed_key
    write_appinstaller(feed,identity=signed['packageName'],publisher=signed['publisher'],version=signed['packageVersion'],self_uri=request['publicBase']+'/'+feed_key,artifact_uri=request['publicBase']+'/'+key,update_policy='pinned',package_arch='x64')
    package={'platform':'win32','arch':'x64','variant':'bundled','version':signed['packageVersion'],'identity':signed['packageName'],'publisher':signed['publisher'],'artifact':{'key':key,'sha256':digest,'size':artifact.stat().st_size},'feed':{'key':feed_key,'channel':'stable'}}
    package['artifact']['chunks'] = prepare_chunks(target, incoming, request['buildId'], package['artifact'])
    archive=output/'signed'/signed['archiveName']
    if archive.name!=signed['archiveName'] or archive.is_symlink() or not archive.is_file():raise ValueError('Signed distribution archive required')
    archive_digest=sha(archive)
    if archive.with_name(archive.name+'.sha256').read_text(encoding='utf-8-sig').split()[0]!=archive_digest:raise ValueError('Distribution archive checksum mismatch')
    # Keep the required signed ZIP locally; update delivery already needs the
    # whole MSIX for older receivers plus chunks. A third copy prevents a second
    # protected multi-GB release from fitting within the R2 storage budget.
    archive_fact={'name':archive.name,'sha256':archive_digest,'size':archive.stat().st_size}
    full=installed['packageFullName'];family=signed['packageName']+'_'+full.rsplit('_',1)[1]
    receipt={'source_commit':request['commit'],'controller_commit':request['controllerCommit'],'build_id':request['buildId'],'repository':'chris-han/hermes-agent','platform':'win32','arch':'x64','identity':signed['packageName'],'publisher':signed['publisher'],'application_id':request['identity']['appNamePascal'],'windows_version':signed['packageVersion'],'sha256':digest,'size':artifact.stat().st_size,'signature_result':'PASS','qualification':'windows-x64-native','workflow_run_id':os.environ['GITHUB_RUN_ID'],'workflow_run_attempt':int(os.environ['GITHUB_RUN_ATTEMPT']),'workflow_result':'native_gates_passed','package_family':family,'notes_sha256':hashlib.sha256(canonical(notes)).hexdigest(),'native_receipt_sha256':sha(native/'installed-identity.json'),'trusted_timestamp':signed['trustedTimestamp']}
    receipt['local_distribution_archive']=archive_fact
    receipt['chunk_descriptor_sha256']=hashlib.sha256(canonical(package['artifact']['chunks'])).hexdigest()
    capabilities=installed['stamp'].get('receiverCapabilities')
    if capabilities is not None:
        if capabilities!={'windowsPackageUri':1} or type(capabilities.get('windowsPackageUri')) is not int:
            raise ValueError('Unsupported installed native receiver capability')
        receipt['receiver_capabilities']=capabilities
    secret=os.environ['SEMANTIER_CLIENT_RECEIPT_KEY'].encode()
    if len(secret)<32:raise ValueError('CI-only receipt key required')
    candidate={'release_id':request['buildId'],'manifest':{'schema':1,'request':request,'packages':[package]},'notes':notes,'native_receipt':receipt,'receipt_hmac':hmac.new(secret,canonical(receipt),hashlib.sha256).hexdigest()}
    if capabilities is not None:candidate['manifest']['receiverCapabilities']=capabilities
    candidates=incoming/'ci-candidates';candidates.mkdir(exist_ok=True)
    destination=candidates/(request['buildId']+'.json')
    with destination.open('x') as stream:json.dump(candidate,stream,indent=2);stream.write('\n')
    return {'build_id':request['buildId'],'source_commit':request['commit'],'candidate_path':str(destination),'sha256':digest,'state':'staged_awaiting_completed_github_run_and_explicit_promotion'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--native',type=Path);p.add_argument('--incoming',type=Path);p.add_argument('--notes',type=Path);p.add_argument('--target-incoming',type=Path);p.add_argument('--admit-target-binding',type=Path);a=p.parse_args()
    if a.admit_target_binding:
        if any((a.output,a.native,a.incoming,a.notes,a.target_incoming)):p.error('target admission is separate from staging')
        print(json.dumps(admit_target_binding(a.admit_target_binding)))
    else:
        if not all((a.output,a.native,a.incoming,a.notes)):p.error('staging requires output/native/incoming/notes')
        print(json.dumps(stage(a.output,a.native,a.incoming,read(a.notes),a.target_incoming)))
