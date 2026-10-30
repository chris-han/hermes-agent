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
        dict({**common,'arch':'neutral'},key=prefix+'main.msixbundle',kind='main-bundle',identity='Semantier.Agent.Acceptance',references=[prefix+'main.msix',prefix+'runtime.msix']),
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
    mutations={
        'oversize':lambda:row.update(size=480000001),
        'outer_oversize':lambda:graph['objects'][0].update(size=480000001),
        'missing':lambda:graph['objects'].pop(),
        'unreachable':lambda:graph['objects'][0]['references'].pop(),
        'cycle':lambda:row.update(references=[graph['root']]),
        'publisher':lambda:row.update(publisher='wrong'),
        'version':lambda:row.update(version='wrong'),
        'arch':lambda:row.update(arch='wrong'),
        'digest':lambda:row.update(nativeDigest='b'*64),
        'traversal':lambda:row.update(key=row['key'].replace('runtime.msix','../runtime.msix')),
        'duplicate':lambda:graph['objects'].append(dict(row)),
        'different_build':lambda:row.update(key=row['key'].replace('a'*32,'d'*32)),
    }
    mutations[damage]()
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
    for name,value in records.items():(tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    for key,value in {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'chris-han/hermes-agent','GITHUB_SHA':request['commit']}.items():monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='Final signed source proof required'):stage(tmp_path,tmp_path,tmp_path,{'en':'test'})
    assert not (tmp_path/'ci-candidates').exists()


def test_native_graph_rejects_same_identity_at_two_remote_keys():
    from scripts.releases.semantier_windows import validate_native_graph
    graph=native_graph()
    other=dict(graph['objects'][-1]);other['key']=other['key'].replace('runtime.msix','other.msix')
    graph['objects'].append(other);graph['objects'][0]['references'].append(other['key'])
    with pytest.raises(ValueError,match='Ambiguous native package identity'):
        validate_native_graph(graph,build_id='a'*32,identity='Semantier.Agent.Acceptance',
            publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64')


def test_asset_partition_conserves_paths_and_reuses_existing_assignment():
    from scripts.releases.semantier_windows import partition_native_assets
    rows=[{'path':'app/main.exe','bytes':1000000,'compressed_bytes':500000},
          {'path':'app/resources/agent-payload/venv/module.pyd','bytes':25000000,'compressed_bytes':23000000},
          {'path':'app/resources/agent-payload/model.onnx','bytes':25000000,'compressed_bytes':23000000}]
    first=partition_native_assets(rows,max_bytes=30000000)
    assert set(first['assignment'])=={r['path'] for r in rows}
    assert first['assignment']['app/main.exe']=='main'
    assert len(first['groups'])==2
    changed=[*rows,{'path':'app/resources/agent-payload/new.py','bytes':100,'compressed_bytes':100}]
    second=partition_native_assets(changed,max_bytes=30000000,previous=first['assignment'])
    assert all(second['assignment'][p]==g for p,g in first['assignment'].items())
    assert sum(len(g['files']) for g in second['groups'])==len(changed)


def test_high_multiplicity_content_avoids_folded_two_version_link_overflow():
    import hashlib
    from scripts.releases.semantier_windows import partition_native_assets
    metadata=[{'path':f'app/resources/agent-payload/venv/pkg{i}.dist-info/INSTALLER',
               'bytes':2,'compressed_bytes':2,'sha256':hashlib.sha256(b'uv').hexdigest()}
              for i in range(265)]
    other='app/resources/agent-payload/ordinary.bin'
    rows=[{'path':'app/main.exe','bytes':1,'compressed_bytes':1},
          {'path':other,'bytes':100,'compressed_bytes':100},*metadata]
    previous={r['path']:'runtime0001' for r in rows};previous['app/main.exe']='main'
    plan=partition_native_assets(rows,previous=previous)
    assert all(plan['assignment'][r['path']]=='main' for r in metadata)
    assert plan['assignment'][other]=='runtime0001'
    assert set(plan['assignment'])==set(previous)
    # Installed Windows observed 531 links for this 265-copy folded content.
    # A second folded version would need 1061; two main-only versions need 531.
    assert 1+sum(1 if plan['assignment'][r['path']]=='main' else 2 for r in metadata)*2<=1023


def test_native_content_that_cannot_fit_two_main_graphs_refuses_before_sdk():
    import hashlib
    from scripts.releases.semantier_windows import partition_native_assets
    rows=[{'path':f'app/resources/agent-payload/duplicate/{i}',
           'bytes':2,'compressed_bytes':2,'sha256':hashlib.sha256(b'uv').hexdigest()}
          for i in range(512)]
    with pytest.raises(ValueError,match='two-version hard-link budget'):
        partition_native_assets(rows)


def test_native_link_policy_does_not_relayout_distinct_same_size_content():
    import hashlib
    from scripts.releases.semantier_windows import partition_native_assets
    rows=[{'path':f'app/resources/agent-payload/metadata/{i}',
           'bytes':4,'compressed_bytes':4,'sha256':hashlib.sha256(i.to_bytes(4,'little')).hexdigest()}
          for i in range(265)]
    previous={r['path']:'runtime0001' for r in rows}
    assert partition_native_assets(rows,previous=previous)['assignment']==previous


def test_real_layout_producer_uses_verified_content_for_link_safe_placement(tmp_path):
    import base64,hashlib,json,zipfile
    from scripts.releases.semantier_windows import prepare_native_asset_layout
    manifest=b'<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="Semantier.Agent.Acceptance" Publisher="CN=Semantier Technology, O=Semantier Technology" ProcessorArchitecture="x64" Version="0.0.62.0"/><Applications><Application Executable="app/main.exe"/></Applications></Package>'
    metadata={f'app/resources/agent-payload/venv/pkg{i}.dist-info/INSTALLER':b'uv' for i in range(265)}
    other='app/resources/agent-payload/ordinary.bin'
    files={'AppxManifest.xml':manifest,'app/main.exe':b'app',other:b'ordinary',**metadata}
    blockmap='<BlockMap xmlns="http://schemas.microsoft.com/appx/2010/blockmap" HashMethod="http://www.w3.org/2001/04/xmlenc#sha256">'+''.join(
        '<File Name="'+name+'" Size="'+str(len(data))+'"><Block Hash="'+base64.b64encode(hashlib.sha256(data).digest()).decode()+'"/></File>'
        for name,data in files.items())+'</BlockMap>'
    package=tmp_path/'input.msix';directory=tmp_path/'unpacked';directory.mkdir()
    with zipfile.ZipFile(package,'w') as archive:
        for name,data in files.items():
            archive.writestr(name,data);file=directory/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
        archive.writestr('AppxBlockMap.xml',blockmap)
    previous={name:'runtime0001' for name in files if name!='AppxManifest.xml'};previous['app/main.exe']='main'
    old=tmp_path/'previous.json';old.write_text(json.dumps({'assignment':previous}))
    plan=prepare_native_asset_layout(package,directory,tmp_path/'layout',
        expected_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),previous=old)
    assert all(plan['assignment'][name]=='main' for name in metadata)
    assert plan['assignment'][other]=='runtime0001'
    assert {row['path']:row['sha256'] for row in plan['files']}=={
        name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    assert all((directory/name).read_bytes()==data for name,data in files.items())


@pytest.mark.parametrize('path',['../escape','/absolute','app/../escape','app/CON','app/a:stream','app/a\\b'])
def test_asset_partition_rejects_unsafe_members(path):
    from scripts.releases.semantier_windows import partition_native_assets
    with pytest.raises(ValueError):partition_native_assets([{'path':path,'bytes':1,'compressed_bytes':1}])


def test_asset_partition_refuses_indivisible_zip_and_case_collision():
    from scripts.releases.semantier_windows import partition_native_assets
    rows=[{'path':'app/resources/agent-payload/uv-cache/seed.zip','bytes':516677051,'compressed_bytes':516677051}]
    with pytest.raises(ValueError,match='seed.zip'):partition_native_assets(rows)
    rows=[{'path':p,'bytes':1,'compressed_bytes':1} for p in ('app/A','app/a')]
    with pytest.raises(ValueError,match='collision'):partition_native_assets(rows)


def test_asset_layout_uses_native_folding_without_path_relocation(tmp_path):
    import xml.etree.ElementTree as ET
    from scripts.releases.semantier_windows import partition_native_assets,write_native_asset_layout
    rows=[{'path':'app/main.exe','bytes':1,'compressed_bytes':1},
          {'path':'app/resources/agent-payload/module1.pyd','bytes':15000000,'compressed_bytes':14000000},
          {'path':'app/resources/agent-payload/module2.pyd','bytes':15000000,'compressed_bytes':14000000}]
    plan=partition_native_assets(rows,max_bytes=30000000)
    file=tmp_path/'layout.xml'
    write_native_asset_layout(file,plan,manifest=r'D:\owned\AppxManifest.xml',source=r'D:\owned')
    root=ET.fromstring(file.read_bytes());ns={'m':'http://schemas.microsoft.com/appx/makeappx/2017'}
    family=root.find('m:PackageFamily',ns)
    assert family.get('FlatBundle')=='true' and family.get('ResourceManager')=='false'
    assets=family.findall('m:AssetPackage',ns)
    assert len(assets)==1 and assets[0].get('AllowExecution')=='true'
    files=root.findall('.//m:File',ns)
    assert {f.get('DestinationPath').replace('\\','/') for f in files}=={r['path'] for r in rows}
    assert all(f.get('SourcePath').startswith('D:\\owned\\') for f in files)


def test_native_graph_asset_packages_bind_resource_id_and_execution():
    from scripts.releases.semantier_windows import validate_native_graph
    graph=native_graph();row=graph['objects'][-1]
    row.update(kind='asset-package',identity='Semantier.Agent.Acceptance',arch='neutral',resourceId='runtime0001',allowExecution=True)
    args=dict(build_id='a'*32,identity='Semantier.Agent.Acceptance',publisher='CN=Semantier Technology, O=Semantier Technology',version='0.0.59.0',arch='x64')
    validate_native_graph(graph,**args)
    row['allowExecution']=False
    with pytest.raises(ValueError,match='asset'):validate_native_graph(graph,**args)


def test_native_layout_rejects_modified_unpacked_payload(tmp_path):
    import base64,hashlib,zipfile
    from scripts.releases.semantier_windows import prepare_native_asset_layout
    manifest=b'<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="Semantier.Agent.Acceptance" Publisher="CN=Semantier Technology, O=Semantier Technology" ProcessorArchitecture="x64" Version="0.0.59.0"/><Applications><Application Executable="app/main.exe"/></Applications></Package>'
    files={'AppxManifest.xml':manifest,'app/main.exe':b'actual payload'}
    blockmap='<BlockMap xmlns="http://schemas.microsoft.com/appx/2010/blockmap" xmlns:b4="http://schemas.microsoft.com/appx/2021/blockmap" HashMethod="http://www.w3.org/2001/04/xmlenc#sha256">'+''.join('<File Name="'+name+'" Size="'+str(len(data))+'"><Block Hash="'+base64.b64encode(hashlib.sha256(data).digest()).decode()+'"/><b4:FileHash Hash="'+base64.b64encode(hashlib.sha256(data).digest()).decode()+'"/></File>' for name,data in files.items())+'</BlockMap>'
    package=tmp_path/'input.msix';directory=tmp_path/'unpacked';directory.mkdir()
    with zipfile.ZipFile(package,'w') as z:
        for name,data in files.items():
            z.writestr(name,data);f=directory/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
        z.writestr('AppxBlockMap.xml',blockmap)
    digest=hashlib.sha256(package.read_bytes()).hexdigest()
    result=prepare_native_asset_layout(package,directory,tmp_path/'valid',expected_sha256=digest)
    assert result['files'][1]['sha256']==hashlib.sha256(files['app/main.exe']).hexdigest()
    (directory/'app/main.exe').write_bytes(b'wrong! payload')
    with pytest.raises(ValueError,match='block mismatch'):
        prepare_native_asset_layout(package,directory,tmp_path/'invalid',expected_sha256=digest)
    assert not (tmp_path/'invalid').exists()


def test_final_graph_rejects_foreign_payload_and_embedded_bundle(tmp_path):
    import base64,hashlib,json,zipfile
    from scripts.releases.semantier_windows import inspect_native_asset_graph
    payload=b'real runtime'
    identity='Name="Semantier.Agent.Acceptance" Publisher="CN=Semantier Technology, O=Semantier Technology" Version="0.0.59.0"'
    stamp=json.dumps({'commit':'a'*40,'channelBuild':{'buildId':'b'*32}}).encode()
    files={'app/main.exe':payload,'app/resources/install-stamp.json':stamp}
    plan={'identity':{'Name':'Semantier.Agent.Acceptance','Publisher':'CN=Semantier Technology, O=Semantier Technology','Version':'0.0.59.0','ProcessorArchitecture':'x64'},'payloadInventorySha256':'c'*64,'groups':[{'id':'main'}],'assignment':{p:'main' for p in files},'files':[{'path':p,'size':len(v),'sha256':hashlib.sha256(v).hexdigest()} for p,v in files.items()]}
    def package(damage=False,dependency=False):
        with zipfile.ZipFile(tmp_path/'SemantierAgent-main.msix','w') as z:
            z.writestr('AppxManifest.xml','<Package><Identity '+identity+' ProcessorArchitecture="x64"/>'+('<Dependencies><PackageDependency Name="Unexpected.Framework"/></Dependencies>' if dependency else '')+'</Package>')
            z.writestr('AppxSignature.p7x',b'fixture signature; signature policy checked by signer')
            for p,v in files.items():z.writestr(p,b'foreign' if damage and p.endswith('.exe') else v)
    package()
    bundle=tmp_path/'SemantierAgent.msixbundle'
    with zipfile.ZipFile(bundle,'w') as z:
        z.writestr('AppxMetadata/AppxBundleManifest.xml','<Bundle><Identity '+identity+'/><Packages><Package Type="application" Architecture="x64" Version="0.0.59.0" FileName="SemantierAgent-main.msix"/></Packages></Bundle>')
        z.writestr('AppxSignature.p7x',b'fixture signature')
    digests={p.name:base64.b64encode(b'd'*32).decode() for p in tmp_path.iterdir() if p.is_file()}
    graph=inspect_native_asset_graph(tmp_path,plan,digests,source_commit='a'*40)
    assert len(graph['objects'])==2
    package(True)
    with pytest.raises(ValueError,match='payload'):inspect_native_asset_graph(tmp_path,plan,digests,source_commit='a'*40)
    package()
    package(dependency=True)
    with pytest.raises(ValueError):inspect_native_asset_graph(tmp_path,plan,digests,source_commit='a'*40)
    package()
    with zipfile.ZipFile(bundle,'a') as z:z.writestr('SemantierAgent-main.msix',b'embedded')
    with pytest.raises(ValueError,match='external'):inspect_native_asset_graph(tmp_path,plan,digests,source_commit='a'*40)


def test_ci_bridge_selects_only_the_complete_source_bound_graph(tmp_path):
    import hashlib,json
    from scripts.releases.semantier_native_ci import select_native_candidate
    graph=native_graph();distribution=tmp_path/'signed/distribution';distribution.mkdir(parents=True)
    request={'commit':'a'*40,'buildId':'a'*32,'windowsVersion':'0.0.59.0','identity':IDENTITY}
    body=b'selection fixture; not native qualification'
    for row in graph['objects']:row.update(size=len(body),sha256=hashlib.sha256(body).hexdigest())
    signed={'sourceCommit':'a'*40,'publisher':'CN=Semantier Technology, O=Semantier Technology','packageName':IDENTITY['msixAppIdWithOrg'],'packageVersion':'0.0.59.0','signedSha256':graph['objects'][0]['sha256'],'nativeGraph':graph}
    for name,value in [('channel-request.json',request),('result.json',{'result':'PASS','sourceCommit':'a'*40}),('signed/receipt.json',signed)]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    for row in graph['objects']:(distribution/row['key'].rsplit('/',1)[1]).write_bytes(body)
    (distribution/'native-graph.json').write_text(json.dumps(graph),encoding='utf-8')
    layout=tmp_path/'signed/native-layout/layout-plan.json';layout.parent.mkdir();layout.write_text('{}',encoding='utf-8')
    assert select_native_candidate(tmp_path,'a'*40)['artifact'].name=='main.msixbundle'
    with pytest.raises(ValueError,match='source'):select_native_candidate(tmp_path,'c'*40)
    (distribution/'runtime.msix').write_bytes(b'altered bytes')
    with pytest.raises(ValueError,match='object bytes'):select_native_candidate(tmp_path,'a'*40)
    (distribution/'runtime.msix').unlink()
    with pytest.raises(ValueError,match='complete'):select_native_candidate(tmp_path,'a'*40)


def test_new_native_assignment_keeps_room_for_small_successor_without_relayout():
    from scripts.releases.semantier_windows import partition_native_assets
    rows=[{'path':'app/main.exe','bytes':1000,'compressed_bytes':1000}]+[
        {'path':f'app/resources/agent-payload/source/{index:04d}.bin','bytes':1000000,'compressed_bytes':1000000}
        for index in range(600)]
    first=partition_native_assets(rows)
    assert len(first['groups'])==2
    assert first['groups'][0]['estimatedBytes']<=472000000
    successor=[{**row,'bytes':1001000,'compressed_bytes':1001000} if row['path']=='app/main.exe' else row for row in rows]
    second=partition_native_assets(successor,previous=first['assignment'])
    assert second['assignment']==first['assignment']


def test_ci_preflight_rejects_checkout_and_secret_mismatch_before_custody(tmp_path, monkeypatch):
    from scripts.releases.semantier_native_ci import validate_ci_preflight
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'chris-han/hermes-agent')
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    with pytest.raises(ValueError, match='source checkout'):
        validate_ci_preflight('b' * 40, 'c' * 40, 'c' * 40, 'S-1-5-21-1-2-3-4', 'x' * 32)
    with pytest.raises(ValueError, match='controller checkout'):
        validate_ci_preflight('a' * 40, 'b' * 40, 'c' * 40, 'S-1-5-21-1-2-3-4', 'x' * 32)
    with pytest.raises(ValueError, match='receipt key'):
        validate_ci_preflight('a' * 40, 'c' * 40, 'c' * 40, 'S-1-5-21-1-2-3-4', 'short')
    with pytest.raises(ValueError, match='Separate'):
        validate_ci_preflight('a' * 40, 'c' * 40, 'c' * 40, 'S-1-5-21-1928370875-4287869425-2092521300-1003', 'x' * 32)


def test_ci_custody_probe_requires_runtime_readback_and_removes_only_owned_probe(tmp_path):
    from scripts.releases.semantier_native_ci import custody_probe
    observed = []
    before = set(tmp_path.iterdir())
    def runtime_read(name):
        observed.append(name)
        import hashlib
        return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    custody_probe(tmp_path, runtime_read)
    assert len(observed) == 1
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError, match='Runtime readback'):
        custody_probe(tmp_path, lambda name: b'incorrect')
    assert set(tmp_path.iterdir()) == before


def test_ci_custody_probe_never_removes_a_replaced_file(tmp_path):
    from scripts.releases.semantier_native_ci import custody_probe
    replaced = []
    def replace(name):
        path = tmp_path / name
        path.unlink()
        path.write_bytes(b'other owner')
        replaced.append(path)
        return 'wrong'
    with pytest.raises(ValueError, match='Runtime readback'):
        custody_probe(tmp_path, replace)
    assert replaced[0].read_bytes() == b'other owner'


def test_ci_receipt_key_preflight_refuses_mismatched_runtime_key():
    import hashlib
    import hmac
    from scripts.releases.semantier_native_ci import verify_receipt_key
    key = 'x' * 32
    challenge = b'fixture-challenge'
    def same(value):
        return hmac.new(key.encode(), value, hashlib.sha256).hexdigest()
    verify_receipt_key(key, challenge, same)
    with pytest.raises(ValueError, match='receipt key mismatch'):
        verify_receipt_key(key, challenge, lambda value: '0' * 64)


def test_ci_primary_publication_base_rejects_loopback_and_runtime_mismatch():
    from scripts.releases.semantier_native_ci import validate_primary_base
    validate_primary_base('https://updates.semantier.com', 'https://updates.semantier.com')
    with pytest.raises(ValueError, match='publication base'):
        validate_primary_base('http://127.0.0.1:8926', 'https://updates.semantier.com')
    with pytest.raises(ValueError, match='publication base'):
        validate_primary_base('https://updates.semantier.com', 'https://other.example')
