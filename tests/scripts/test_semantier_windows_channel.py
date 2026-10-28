import pytest
from scripts.releases.semantier_windows import acceptance_request,IDENTITY
from scripts.bundles.desktop_prepare import validate_channel_request


def test_consecutive_versions_keep_family_and_provenance():
    first=acceptance_request(commit='a'*40,controller='b'*40,version='0.0.36',source_version='0.22.2',base='http://127.0.0.1:8926')
    second=acceptance_request(commit='c'*40,controller='b'*40,version='0.0.37',source_version='0.22.2',base='http://127.0.0.1:8926')
    assert first['bundleEnv']==second['bundleEnv']=={'HERMES_HOME':None,'HERMES_DATA_DIR_SUFFIX':'-semantier-acceptance'}
    assert first['identity']==second['identity']==IDENTITY
    assert first['commit']!=second['commit'] and first['windowsVersion']!=second['windowsVersion']
    assert validate_channel_request(first)==first
    with pytest.raises(ValueError):validate_channel_request({**first,'semantierScope':'other'})
    with pytest.raises(ValueError):acceptance_request(commit='a'*40,controller='b'*40,version='0.0.36',source_version='0.22.2',base='http://untrusted.example')


def native_graph():
    import base64
    prefix='releases/channel-builds/'+'a'*32+'/'
    common=dict(publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64',
                size=1000,sha256='b'*64,nativeDigest=base64.b64encode(b'd'*32).decode())
    return dict(schema=1,transport='windows-package-uri',root=prefix+'main.msixbundle',payloadInventorySha256='c'*64,objects=[
        dict(common,key=prefix+'main.msixbundle',kind='main-bundle',identity='Semantier.Agent.Acceptance',references=[prefix+'main.msix',prefix+'runtime.msix']),
        dict(common,key=prefix+'main.msix',kind='main-package',identity='Semantier.Agent.Acceptance',references=[]),
        dict(common,key=prefix+'runtime.msix',kind='optional-package',identity='Semantier.Agent.Acceptance.Runtime',references=[])])


def test_native_graph_binds_entire_remote_set():
    from scripts.releases.semantier_windows import validate_native_graph
    graph=native_graph()
    assert validate_native_graph(graph,build_id='a'*32,identity='Semantier.Agent.Acceptance',
        publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64')==graph


@pytest.mark.parametrize('damage',['oversize','outer_oversize','missing','unreachable','cycle','publisher','version','arch','digest','traversal','duplicate','different_build'])
def test_native_graph_refuses_unbound_or_incomplete_sets(damage):
    from scripts.releases.semantier_windows import validate_native_graph
    graph=native_graph();row=graph['objects'][-1]
    if damage=='oversize':row['size']=480000001
    elif damage=='outer_oversize':graph['objects'][0]['size']=480000001
    elif damage=='missing':graph['objects'].pop()
    elif damage=='unreachable':graph['objects'][0]['references'].pop()
    elif damage=='cycle':row['references']=[graph['root']]
    elif damage in ('publisher','version','arch'):row[damage]='wrong'
    elif damage=='digest':row['nativeDigest']='b'*64
    elif damage=='traversal':row['key']=row['key'].replace('runtime.msix','../runtime.msix')
    elif damage=='duplicate':graph['objects'].append(dict(row))
    elif damage=='different_build':row['key']=row['key'].replace('a'*32,'d'*32)
    with pytest.raises(ValueError):validate_native_graph(graph,build_id='a'*32,identity='Semantier.Agent.Acceptance',
        publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64')


def test_native_graph_cannot_mint_legacy_ci_qualification(tmp_path,monkeypatch):
    import json
    from scripts.releases.semantier_candidate import stage
    request=acceptance_request(commit='a'*40,controller='b'*40,version='0.0.59',source_version='0.22.2',base='http://127.0.0.1:8926')
    request['buildId']='a'*32
    (tmp_path/'signed').mkdir()
    records={'result.json':{'result':'PASS','sourceCommit':request['commit']},'channel-request.json':request,
        'signed/receipt.json':{'publisher':'CN=Semantier Technology, O=Semantier Technology','nativeGraph':native_graph()}}
    for name,value in records.items():(tmp_path/name).write_text(json.dumps(value))
    for key,value in {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'chris-han/hermes-agent','GITHUB_SHA':request['commit']}.items():monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='graph-bound installed qualification'):stage(tmp_path,tmp_path,tmp_path,{'en':'test'})


def test_native_graph_rejects_same_identity_at_two_remote_keys():
    from scripts.releases.semantier_windows import validate_native_graph
    graph=native_graph()
    other=dict(graph['objects'][-1]);other['key']=other['key'].replace('runtime.msix','other.msix')
    graph['objects'].append(other);graph['objects'][0]['references'].append(other['key'])
    with pytest.raises(ValueError,match='Ambiguous native package identity'):
        validate_native_graph(graph,build_id='a'*32,identity='Semantier.Agent.Acceptance',
            publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64')
