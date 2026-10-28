"""Stage signed native CI output; never promote a channel or weaken full-matrix admission."""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.releases.semantier_windows import is_acceptance_request
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


def stage(output,native,incoming,notes):
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='chris-han/hermes-agent':raise ValueError('Real governed CI context required')
    result=read(output/'result.json');request=read(output/'channel-request.json');signed=read(output/'signed/receipt.json')
    if not is_acceptance_request(request) or result['result']!='PASS' or request['commit']!=os.environ['GITHUB_SHA'] or result['sourceCommit']!=request['commit']:raise ValueError('Exact completed source required')
    installed=read(native/'installed-identity.json');exit_result=read(native/'native-install-exit.json');artifact_meta=read(native/'artifact.json')
    if exit_result.get('exitCode')!=0 or signed.get('signing')!='PASS' or not signed.get('verification','').startswith('PASS:'):raise ValueError('Native installation and signature gates required')
    if signed['packageName']!=request['identity']['msixAppIdWithOrg'] or signed['publisher']!='CN=Semantier Technology, O=Semantier Technology' or signed['packageVersion']!=request['windowsVersion']:raise ValueError('Package identity mismatch')
    if installed['version']!=signed['packageVersion'] or installed['publisher']!=signed['publisher'] or installed['stamp']['commit']!=request['commit']:raise ValueError('Installed source mismatch')
    distribution=output/'signed/distribution';files=list(distribution.glob('*.msix'))
    if len(files)!=1 or files[0].is_symlink():raise ValueError('One signed MSIX required')
    artifact=files[0];digest=sha(artifact)
    if digest!=signed['signedSha256'] or digest!=artifact_meta['sha256']:raise ValueError('Artifact bytes mismatch')
    # Reuse the artifact-pinned parent publisher's chunk contract, never a
    # second CI-only splitter. Verify its composed source before consuming it.
    from scripts.bundles.parent_composition import verify
    composition = ROOT/'semantier-runtime'
    verify(composition)
    sys.path.insert(0, str(composition/'src'))
    from agents.client_operations.chunks import prepare_chunks
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
    secret=os.environ['SEMANTIER_CLIENT_RECEIPT_KEY'].encode()
    if len(secret)<32:raise ValueError('CI-only receipt key required')
    candidate={'release_id':request['buildId'],'manifest':{'schema':1,'request':request,'packages':[package]},'notes':notes,'native_receipt':receipt,'receipt_hmac':hmac.new(secret,canonical(receipt),hashlib.sha256).hexdigest()}
    candidates=incoming/'ci-candidates';candidates.mkdir(exist_ok=True)
    destination=candidates/(request['buildId']+'.json')
    with destination.open('x') as stream:json.dump(candidate,stream,indent=2);stream.write('\n')
    return {'build_id':request['buildId'],'source_commit':request['commit'],'candidate_path':str(destination),'sha256':digest,'state':'staged_awaiting_completed_github_run_and_explicit_promotion'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--incoming',type=Path,required=True);p.add_argument('--notes',type=Path,required=True);a=p.parse_args()
    print(json.dumps(stage(a.output,a.native,a.incoming,read(a.notes))))
