"""Stage a complete native graph only after the existing genuine CI boundary."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
from datetime import UTC,datetime
from uuid import UUID

ROOT=Path(__file__).resolve().parents[2]


def read(path,maximum=4*1024*1024):
    if path.is_symlink() or path.stat().st_size>maximum:raise ValueError('Bounded native evidence file required')
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value,dict):raise ValueError('Native evidence object required')
    return value


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def verify_uri_proof(proof,graph,graph_sha,expected_sid):
    if (not re.fullmatch(r'S-1-5-21-[0-9-]+',expected_sid)
            or expected_sid in ('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011') or proof.get('graphSha256')!=graph_sha
            or proof.get('nativeRemoteDigestEnforcement')!='PASS'
            or proof.get('sourceBuildId')!=graph['root'].split('/')[2]):
        raise ValueError('Native URI proof graph/CI actor binding required')
    before,after=proof['contextBefore'],proof['contextAfter']
    for context in (before,after):
        if (context['sid']!=expected_sid or context['session_sid']!=expected_sid or context['wts_state']!=0
                or context['interactive'] is not True or type(context['session']) is not int or context['session']<=0):
            raise ValueError('Actual active CI session required')
    if before['session']!=after['session']:raise ValueError('CI session changed')
    rows=graph['objects'];refusals=proof['refusals']
    if (len(refusals)!=len(rows) or {row['key'] for row in refusals}!={row['key'] for row in rows}
            or any(row['result'].get('status')!='FAILURE' or row['result'].get('hresult')!='80080219' or row.get('registrationAbsentAfterRefusal') is not True for row in refusals)):
        raise ValueError('Every native object requires exact SDK digest refusal')
    if (len(proof['uris'])!=len(rows) or proof['uris']!=[proof['uris'][0].split('/releases/')[0]+'/'+row['key'] for row in rows]
            or proof['digests']!=[row['nativeDigest'] for row in rows]):raise ValueError('Native URI/digest map mismatch')
    positive=proof['positive']
    if positive.get('status')!='SUCCESS' or positive.get('hresult')!='00000000' or UUID(positive['activityId']).int==0:
        raise ValueError('Native URI success unproved')


def verify_installed(native,request,graph,graph_sha):
    installed=read(native/'installed-identity.json');registration=read(native/'native-installed-graph.json')
    runtime=read(native/'native-payload-runtime.json');folding=read(native/'native-payload-folding.json')
    if (read(native/'native-install-exit.json').get('exitCode')!=0 or registration.get('nativeRemoteDigestEnforcement')!='PASS'
            or registration['graphSha256']!=graph_sha or installed['stamp']['commit']!=request['commit']
            or installed['stamp']['channelBuild']!=request or runtime['sourceCommit']!=request['commit']
            or runtime['buildId']!=request['buildId'] or runtime['packageFullName']!=installed['packageFullName']
            or installed['version']!=request['windowsVersion'] or installed['architecture'].lower()!='x64'):
        raise ValueError('Exact installed graph/source proof required')
    if (runtime.get('polarsSum')!=6 or not runtime.get('nativeModules')
            or runtime['sensevoice'].get('modelLoaded') is not True or runtime['sensevoice'].get('silenceDecodeCompleted') is not True
            or runtime['skills'].get('filesRead',0)<1 or folding['payloadInventorySha256']!=graph['payloadInventorySha256']):
        raise ValueError('Complete installed native runtime proof required')
    expected={row['resourceId'] for row in graph['objects'] if row['kind']=='asset-package'}
    actual=registration['related']
    if len(actual)!=len(expected) or {row['resource_id'] for row in actual}!=expected:
        raise ValueError('Complete registered native resource set required')
    identity=request['identity']['msixAppIdWithOrg'];family=identity+'_'+installed['packageFullName'].rsplit('_',1)[1]
    if any(row['family']!=family or row['name']!=identity or row['publisher']!=installed['publisher'] or row['version']!=request['windowsVersion'] for row in actual):
        raise ValueError('Native related identity continuity mismatch')
    return installed,folding,family


def stage_native_graph(output,native,incoming,notes,request,signed):
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='chris-han/hermes-agent' or os.environ.get('GITHUB_SHA')!=request['commit']:
        raise ValueError('Real governed CI context required')
    if signed.get('signing')!='PASS' or not signed.get('verification','').startswith('PASS:') or signed.get('sourceCommit')!=request['commit']:
        raise ValueError('Final signed source proof required')
    from scripts.bundles.parent_composition import verify
    composition=ROOT/'semantier-runtime';composition_info=verify(composition);sys.path.insert(0,str(composition/'src'))
    from agents.client_operations.package_graph import decode_package_graph
    from agents.client_operations.artifact_io import immutable_copy
    graph=signed['nativeGraph'];file=output/'signed/distribution/native-graph.json';graph_sha=sha(file)
    root=next(row for row in graph['objects'] if row['key']==graph['root'])
    descriptor={'key':graph['root'].rsplit('/',1)[0]+'/native-graph.json','size':file.stat().st_size,'sha256':graph_sha}
    package={'platform':'win32','arch':'x64','variant':'native-msix-assets','identity':request['identity']['msixAppIdWithOrg'],
             'publisher':signed['publisher'],'version':request['windowsVersion'],
             'artifact':{key:root[key] for key in ('key','size','sha256')},'feed':{'key':descriptor['key'],'channel':'stable'},
             'native':{'schema':1,'api':'PackageManager.AddPackageByUriAsync','digestMethod':'IAppxDigestProvider.GetDigest','digestEncoding':'base64','graph':descriptor}}
    if signed.get('signedSha256')!=root['sha256']:raise ValueError('Signed native root binding mismatch')
    if read(native/'artifact.json').get('sha256')!=root['sha256']:raise ValueError('Qualified native root binding mismatch')
    if decode_package_graph(file.read_bytes(),package,request['buildId'])!=graph:raise ValueError('Signed graph bytes differ')
    verify_uri_proof(read(native/'native-uri-proof.json'),graph,graph_sha,os.environ.get('ACCEPTANCE_CI_SID',''))
    installed,folding,family=verify_installed(native,request,graph,graph_sha)
    if (installed['publisher']!=package['publisher'] or signed.get('packageName')!=package['identity']
            or signed.get('packageVersion')!=package['version'] or signed['publisher']!='CN=Semantier Technology, O=Semantier Technology'):
        raise ValueError('Final signed and installed identity continuity required')
    plan=read(output/'signed/native-layout/layout-plan.json',maximum=256*1024*1024)
    if read(native/'native-installed-graph.json').get('layoutPlanSha256')!=sha(output/'signed/native-layout/layout-plan.json') or folding['filesChecked']!=len(plan['assignment']) or folding['bytesChecked']!=sum(row['size'] for row in plan['files'] if row['path'] in plan['assignment']):
        raise ValueError('Complete installed file conservation proof required')
    for row in [*graph['objects'],descriptor]:
        source=file.parent/row['key'].rsplit('/',1)[1]
        immutable_copy(source,incoming/row['key'],row)
    receipt={'produced_at':datetime.now(UTC).isoformat(),'qualification_contract':'windows-native-uri-graph/v1',
             'native_qualified_at':read(native/'native-uri-proof.json')['producedAt'],
             'native_payload_verified_at':read(native/'native-payload-runtime.json')['producedAt'],
             'source_commit':request['commit'],'controller_commit':request['controllerCommit'],'build_id':request['buildId'],
             'repository':'chris-han/hermes-agent','platform':'win32','arch':'x64','identity':package['identity'],'publisher':package['publisher'],
             'application_id':request['identity']['appNamePascal'],'windows_version':package['version'],'sha256':root['sha256'],'size':root['size'],
             'signature_result':'PASS','qualification':'windows-x64-native-uri-graph','workflow_run_id':os.environ['GITHUB_RUN_ID'],
             'workflow_run_attempt':int(os.environ['GITHUB_RUN_ATTEMPT']),'workflow_result':'native_gates_passed','package_family':family,
             'notes_sha256':hashlib.sha256(canonical(notes)).hexdigest(),'trusted_timestamp':signed['trustedTimestamp'],
             'runtime_parent_commit':composition_info['parent_commit'],'runtime_composition_sha256':sha(composition/'composition.json'),
             'native_graph_sha256':graph_sha,'native_objects_sha256':hashlib.sha256(canonical(graph['objects'])).hexdigest(),
             'native_remote_digest_enforcement':'PASS','installed_graph_result':'PASS','payload_result':'PASS'}
    for name in ('installed-identity.json','native-uri-proof.json','native-installed-graph.json','native-payload-runtime.json','native-payload-folding.json'):
        source=native/name;fact={'sha256':sha(source),'size':source.stat().st_size}
        immutable_copy(source,incoming/'ci-evidence'/request['buildId']/name,fact)
        receipt[name.replace('.json','').replace('-','_')+'_sha256']=fact['sha256']
    secret=os.environ['SEMANTIER_CLIENT_RECEIPT_KEY'].encode()
    if len(secret)<32:raise ValueError('CI-only receipt key required')
    candidate={'release_id':request['buildId'],'manifest':{'schema':2,'request':request,'packages':[package]},'notes':notes,
               'native_receipt':receipt,'receipt_hmac':hmac.new(secret,canonical(receipt),hashlib.sha256).hexdigest()}
    directory=incoming/'ci-candidates';directory.mkdir(exist_ok=True)
    destination=directory/(request['buildId']+'.json')
    with destination.open('x',encoding='utf-8') as stream:json.dump(candidate,stream,indent=2);stream.write('\n')
    return {'build_id':request['buildId'],'source_commit':request['commit'],'candidate_path':str(destination),'sha256':root['sha256'],
            'state':'staged_awaiting_completed_github_run_and_explicit_promotion'}
