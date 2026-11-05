"""Native CI proof validation fixtures; no native execution or admission claims."""
from copy import deepcopy
import pytest
from scripts.releases.semantier_native_candidate import verify_uri_proof


def proof_case():
    sid='S-1-5-21-9-8-7-1003'  # Another host's RID is not the excluded ChrisH SID.
    rows=[dict(key='releases/channel-builds/'+'a'*32+'/'+name,nativeDigest='fixture') for name in ('main.msix','asset.msix')]
    graph={'root':rows[0]['key'],'objects':rows}
    context=dict(sid=sid,session_sid=sid,session=2,wts_state=0,interactive=True)
    proof=dict(graphSha256='b'*64,nativeRemoteDigestEnforcement='PASS',sourceBuildId='a'*32,
               contextBefore=context,contextAfter=deepcopy(context),
               refusals=[dict(key=row['key'],registrationAbsentAfterRefusal=True,result=dict(status='FAILURE',hresult='80080219')) for row in rows],
               uris=['http://127.0.0.1:12345/'+row['key'] for row in rows],digests=['fixture']*2,
               positive=dict(status='SUCCESS',hresult='00000000',activityId='11111111-1111-1111-1111-111111111111'))
    return proof,graph,sid


def test_every_object_refusal_and_separate_actual_ci_actor_are_required():
    proof,graph,sid=proof_case()
    verify_uri_proof(proof,graph,'b'*64,sid)


@pytest.mark.parametrize('damage',['missing-object','wrong-error','registered','wrong-digest-map','wrong-uri-map','inactive','cross-session','unknown-positive','wrong-graph'])
def test_native_ci_proof_refuses_incomplete_or_unbound_evidence(damage):
    proof,graph,sid=proof_case()
    mutations={
        'missing-object':(proof,'refusals',proof['refusals'][:1]),
        'wrong-error':(proof['refusals'][0]['result'],'hresult','80070002'),
        'registered':(proof['refusals'][0],'registrationAbsentAfterRefusal',False),
        'wrong-digest-map':(proof,'digests',['other']*2),
        'wrong-uri-map':(proof,'uris',[]),
        'inactive':(proof['contextAfter'],'wts_state',4),
        'cross-session':(proof['contextAfter'],'session',3),
        'unknown-positive':(proof['positive'],'status','UNKNOWN'),
        'wrong-graph':(proof,'graphSha256','c'*64),
    }
    value,key,changed=mutations[damage];value[key]=changed
    with pytest.raises(ValueError):verify_uri_proof(proof,graph,'b'*64,sid)


@pytest.mark.parametrize('corrupt_object',[False,True,'symlink'])
def test_real_native_stager_and_composed_publisher_bind_complete_final_graph(tmp_path,monkeypatch,corrupt_object):
    """Only signing/Windows/GitHub results are fixtures; use real local wiring."""
    import hashlib,hmac,json
    from pathlib import Path
    from test_semantier_windows_channel import native_graph
    from scripts.releases.semantier_windows import acceptance_request
    from scripts.releases.semantier_candidate import stage
    from scripts.releases.semantier_native_candidate import canonical
    graph=native_graph();asset=graph['objects'][-1]
    asset.update(kind='asset-package',identity='Semantier.Agent.Acceptance',arch='neutral',resourceId='runtime0001',allowExecution=True)
    request=acceptance_request(commit='a'*40,controller='b'*40,version='0.0.59',source_version='0.22.2',base='https://updates.semantier.com')
    prefix='releases/channel-builds/'+request['buildId']+'/'
    old='releases/channel-builds/'+'a'*32+'/'
    output=tmp_path/'output';native=tmp_path/'native';incoming=tmp_path/'incoming';target_incoming=tmp_path/'incoming_r2';target_incoming.mkdir()
    distribution=output/'signed/distribution';distribution.mkdir(parents=True);native.mkdir()
    for row in graph['objects']:
        row['key']=row['key'].replace(old,prefix);row['references']=[key.replace(old,prefix) for key in row['references']]
        body=('final signed boundary fixture '+row['kind']).encode()
        row.update(size=len(body),sha256=hashlib.sha256(body).hexdigest())
        (distribution/Path(row['key']).name).write_bytes(body)
    graph['root']=graph['root'].replace(old,prefix)
    graph_file=distribution/'native-graph.json';graph_file.write_bytes(canonical(graph));graph_sha=hashlib.sha256(graph_file.read_bytes()).hexdigest()
    plan=output/'signed/native-layout/layout-plan.json';plan.parent.mkdir();plan.write_text(json.dumps({'assignment':{'payload':'main'},'files':[{'path':'payload','size':17}]}))
    proof,_,sid=proof_case();proof.update(graphSha256=graph_sha,sourceBuildId=request['buildId'],producedAt='2026-10-29T15:00:00+00:00',
        uris=['http://127.0.0.1:12345/'+row['key'] for row in graph['objects']],digests=[row['nativeDigest'] for row in graph['objects']],
        refusals=[dict(key=row['key'],registrationAbsentAfterRefusal=True,result=dict(status='FAILURE',hresult='80080219')) for row in graph['objects']])
    publisher=graph['objects'][0]['publisher'];full='Semantier.Agent.Acceptance_0.0.59.0_x64__fixture'
    signed=dict(signing='PASS',verification='PASS: signing boundary fixture',sourceCommit=request['commit'],nativeGraph=graph,
        signedSha256=graph['objects'][0]['sha256'],packageName='Semantier.Agent.Acceptance',packageVersion=request['windowsVersion'],publisher=publisher,trustedTimestamp=False)
    records={output/'result.json':dict(result='PASS',sourceCommit=request['commit'],controllerCommit=request['controllerCommit']),output/'channel-request.json':request,
        output/'signed/receipt.json':signed,native/'artifact.json':dict(sha256=signed['signedSha256']),
        native/'installed-identity.json':dict(version=request['windowsVersion'],architecture='X64',publisher=publisher,packageFullName=full,stamp=dict(commit=request['commit'],channelBuild=request)),
        native/'native-install-exit.json':dict(exitCode=0),native/'native-uri-proof.json':proof,
        native/'native-installed-graph.json':dict(graphSha256=graph_sha,nativeRemoteDigestEnforcement='PASS',layoutPlanSha256=hashlib.sha256(plan.read_bytes()).hexdigest(),related=[dict(resource_id='runtime0001',family='Semantier.Agent.Acceptance_fixture',name='Semantier.Agent.Acceptance',publisher=publisher,version=request['windowsVersion'])]),
        native/'native-payload-runtime.json':dict(sourceCommit=request['commit'],buildId=request['buildId'],packageFullName=full,polarsSum=6,nativeModules=['fixture'],sensevoice=dict(modelLoaded=True,silenceDecodeCompleted=True),skills=dict(filesRead=58),producedAt='2026-10-29T15:00:30+00:00'),
        native/'native-payload-folding.json':dict(payloadInventorySha256=graph['payloadInventorySha256'],filesChecked=1,bytesChecked=17)}
    for path,value in records.items():path.write_text(json.dumps(value))
    for key,value in dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY='chris-han/hermes-agent',GITHUB_SHA=request['commit'],GITHUB_RUN_ID='external-boundary-fixture',GITHUB_RUN_ATTEMPT='1',ACCEPTANCE_CI_SID=sid,SEMANTIER_CLIENT_RECEIPT_KEY='x'*32,ACCEPTANCE_CONTROLLER_COMMIT=request['controllerCommit']).items():monkeypatch.setenv(key,value)
    import scripts.releases.semantier_native_ci as ci
    monkeypatch.setattr(ci,'CI_INCOMING_ROOT',incoming)
    monkeypatch.setattr(ci,'verify_runtime_identity',lambda _cid:None)
    monkeypatch.setattr(ci,'verify_cloudflare_custody',lambda _cid,_root:None)
    original_check=ci.subprocess.check_output
    def fake_runtime(argv,**kwargs):
        if argv[:2]==['docker','inspect']:return 'f'*64+' '+ci.CI_RUNTIME_IMAGE+' true\n'
        return original_check(argv,**kwargs)
    monkeypatch.setattr(ci.subprocess,'check_output',fake_runtime)
    monkeypatch.setenv('ACCEPTANCE_CONTROLLER_COMMIT','c'*40)
    with pytest.raises(ValueError,match='Exact completed controller required'):
        stage(output,native,incoming,{'en':'Fixture, not native acceptance'},target_incoming)
    assert not (incoming/'ci-candidates').exists()
    monkeypatch.setenv('ACCEPTANCE_CONTROLLER_COMMIT',request['controllerCommit'])
    if corrupt_object is True:
        (distribution/Path(asset['key']).name).write_bytes(b'corrupt final object')
        with pytest.raises(Exception,match='artifact_integrity_mismatch'):stage(output,native,incoming,{'en':'Fixture, not native acceptance'},target_incoming)
        assert not (incoming/'ci-candidates').exists()
        return
    if corrupt_object=='symlink':
        outside=tmp_path/'outside';outside.mkdir();(target_incoming/'releases').symlink_to(outside,target_is_directory=True)
        with pytest.raises(OSError):stage(output,native,incoming,{'en':'Fixture, not native acceptance'},target_incoming)
        assert not list(outside.iterdir()) and not (incoming/'ci-candidates').exists()
        return
    result=stage(output,native,incoming,{'en':'Fixture, not native acceptance'},target_incoming)
    candidate=json.loads(Path(result['candidate_path']).read_text())
    binding=json.loads(Path(result['target_binding_path']).read_text())
    assert binding['delivery_target_id']=='channel2' and binding['candidate']['native_receipt']==candidate['native_receipt']
    assert binding['candidate']['manifest']['request']['publicBase']=='https://update2.semantier.com'
    assert binding['descriptor_receipt']['descriptor_sha256']==graph_sha
    assert hmac.compare_digest(binding['descriptor_receipt_hmac'],hmac.new(b'x'*32,canonical(binding['descriptor_receipt']),hashlib.sha256).hexdigest())
    for row in graph['objects']:assert (target_incoming/row['key']).read_bytes()==(distribution/Path(row['key']).name).read_bytes()
    assert (target_incoming/binding['candidate']['manifest']['packages'][0]['feed']['key']).read_bytes()==graph_file.read_bytes()
    assert candidate['manifest']['schema']==2 and 'distributionArchive' not in candidate['manifest']['packages'][0]
    receipt=candidate['native_receipt']
    assert receipt['controller_commit']==request['controllerCommit']!=receipt['source_commit']
    assert receipt['runtime_parent_commit']==json.loads(Path('semantier-runtime/composition.json').read_text())['parent_commit']
    assert hmac.compare_digest(candidate['receipt_hmac'],hmac.new(b'x'*32,canonical(receipt),hashlib.sha256).hexdigest())
    from client_operations.publication import Publisher
    from client_operations.contracts import ReleaseCandidate
    owner=Publisher(None,tmp_path/'public',incoming,request['publicBase'],b'x'*32,request['identity'],request['channel'],workflow_verifier=lambda _: {'scope':'external-GitHub-boundary-fixture'})
    from client_operations.artifact_io import OperationError
    wrong_owner=Publisher(None,tmp_path/'wrong-public',incoming,'http://127.0.0.1:8926',b'x'*32,request['identity'],request['channel'],workflow_verifier=lambda _: {'scope':'external-GitHub-boundary-fixture'})
    with pytest.raises(OperationError,match='manifest_contract_mismatch'):
        wrong_owner.admit(ReleaseCandidate(**candidate),{'principal_id':'ci-fixture'},persist=False)
    admitted=owner.admit(ReleaseCandidate(**candidate),{'principal_id':'ci-fixture'},persist=False)
    assert admitted['qualification']=='verified'
    for row in graph['objects']:assert (tmp_path/'public'/row['key']).read_bytes()==(distribution/Path(row['key']).name).read_bytes()
    assert not (tmp_path/'public/releases/channels/acceptance.json').exists()
