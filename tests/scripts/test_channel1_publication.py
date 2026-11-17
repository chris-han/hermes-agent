"""Current-only Channel 1 publication uses real files and the native operator seam."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import zipfile

import pytest

from hermes_cli import release_channels as protocol
from scripts.releases import channel1


def package(stage, sequence):
    stage.mkdir(parents=True, exist_ok=True)
    request = {
        'schema': 1, 'buildId': f'{sequence:032x}', 'channel': 'semantier-acceptance',
        'sequence': sequence, 'repository': 'chris-han/hermes-agent', 'commit': 'a'*40,
        'controllerCommit': 'a'*40, 'sourceVersion': '0.0.0', 'version': f'0.0.{sequence}',
        'windowsVersion': f'0.0.{sequence}.0', 'releaseTag': f'v0.0.{sequence}',
        'publicBase': 'https://download.semantier.cn',
        'bundleEnv': {'HERMES_HOME': None, 'HERMES_DATA_DIR_SUFFIX': '-semantier-acceptance'},
        'identity': {'token': 'b0fdc2a7e4b08d16', 'displayName': 'Semantier Agent Acceptance',
                     'appId': 'com.semantier.agent.acceptance', 'appNamePascal': 'SemantierAgentAcceptance',
                     'artifactNamePascal': 'SemantierAgent', 'cliName': 'semantier-agent-acceptance',
                     'msixAppIdWithOrg': 'Semantier.Agent.Acceptance', 'windowsExecutableName': 'Semantier Agent'}}
    path = stage/'SemantierAgent.msix'
    # Disposable ZIP metadata fixture, never an installable/signed application.
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(zipfile.ZipInfo('AppxBlockMap.xml'), '<BlockMap/>')
        archive.writestr(zipfile.ZipInfo('AppxManifest.xml'), f'<Package><Identity Name="Semantier.Agent.Acceptance" Publisher="CN=Semantier Technology, O=Semantier Technology" Version="{request["windowsVersion"]}" ProcessorArchitecture="x64"/></Package>')
        archive.writestr(zipfile.ZipInfo('app/resources/install-stamp.json'), json.dumps({'commit': request['commit'], 'dirty': False, 'channelBuild': request}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = stage/'channel1-metadata.json'
    channel1.prepare(path, metadata, expected_sha256=digest, source_commit=request['commit'],
                     windows_version=request['windowsVersion'], build_id=request['buildId'], native_digest='AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=')
    return request, digest, hashlib.sha256(metadata.read_bytes()).hexdigest()


def publish(root, stage, sequence):
    request, digest, metadata_digest = package(stage, sequence)
    result = channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    return request, digest, result


def test_real_metadata_and_repeated_publish_are_current_only(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    request, digest, first = publish(root, tmp_path/'stage1', 109)
    record_path = root/'releases/channels/semantier-acceptance.json'
    original = record_path.read_bytes()
    record = protocol.validate_record(json.loads(original))
    manifest = json.loads((root/record['nativeHead']['manifestKey']).read_bytes())
    protocol.validate_manifest(manifest, record, request['publicBase'], allow_native=True)
    artifact = root/manifest['packages'][0]['artifact']['key']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    assert artifact.samefile(root/'releases/SemantierAgent.msix')
    _, _, second = publish(root, tmp_path/'stage2', 109)
    assert record_path.read_bytes() == original
    assert first['buildId'] == second['buildId'] == request['buildId']
    assert second['alreadyPublished']
    assert not (root/'releases/.channel1-package.tmp').exists()
    marker = root/'unrelated.txt'; marker.write_text('preserve', encoding='utf-8')
    unrelated = root/'releases/channels/another.json'; unrelated.write_text('{}', encoding='utf-8')
    publish(root, tmp_path/'stage3', 110)
    current = json.loads(record_path.read_bytes())
    assert current['nativeHead']['sequence'] == 110 and current['head'] is None and 'nativeArchive' not in current
    assert not (root/record['nativeHead']['manifestKey']).exists()
    assert not (artifact.parent/'win32/stable.appinstaller').exists()
    assert artifact.exists()  # package retirement remains the existing operator's owner
    assert marker.read_text(encoding='utf-8-sig') == 'preserve' and unrelated.read_text(encoding='utf-8-sig') == '{}'


def test_pointer_is_last_and_failure_restores_old_release(tmp_path, monkeypatch):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 109)
    pointer = root/'releases/channels/semantier-acceptance.json'
    old_pointer = pointer.read_bytes()
    old_package = (root/'releases/SemantierAgent.msix').read_bytes()
    stage = tmp_path/'new'
    request, digest, metadata_digest = package(stage, 110)
    replace = channel1.os.replace
    def fail_activation(source, target):
        if Path(target) == pointer:
            assert pointer.read_bytes() == old_pointer
            manifest = root/f'releases/channel-builds/{request["buildId"]}/build.json'
            assert manifest.exists()
            raise OSError('disposable activation failure')
        return replace(source, target)
    with monkeypatch.context() as context:
        context.setattr(channel1.os, 'replace', fail_activation)
        with pytest.raises(OSError, match='activation failure'):
            channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    assert pointer.read_bytes() == old_pointer
    assert (root/'releases/SemantierAgent.msix').read_bytes() == old_package
    assert not (root/f'releases/channel-builds/{request["buildId"]}').exists()
    channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    assert json.loads(pointer.read_bytes())['nativeHead']['sequence'] == 110


def test_tampered_upload_or_downgrade_never_activates(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'current', 110)
    pointer = root/'releases/channels/semantier-acceptance.json'
    before = pointer.read_bytes()
    stage = tmp_path/'old'
    _, digest, metadata_digest = package(stage, 109)
    with pytest.raises(ValueError, match='older|sequence|downgrade'):
        channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    stage = tmp_path/'tampered'
    _, digest, metadata_digest = package(stage, 111)
    (stage/'SemantierAgent.msix').write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError, match='hash|SHA|integrity'):
        channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    assert pointer.read_bytes() == before


def test_cleanup_failure_restores_metadata_before_restoring_pointer(tmp_path, monkeypatch):
    root = tmp_path/'public';root.mkdir()
    request, _, _ = publish(root, tmp_path/'old', 109)
    pointer = root/'releases/channels/semantier-acceptance.json'
    old_pointer = pointer.read_bytes()
    old_manifest = root/f'releases/channel-builds/{request["buildId"]}/build.json'
    old_feed = old_manifest.parent/'native-graph.json'
    stage = tmp_path/'new'; _, digest, metadata_digest = package(stage, 110)
    unlink, replace = Path.unlink, channel1.os.replace
    def fail_cleanup(path, *args, **kwargs):
        if path == old_feed: raise OSError('disposable cleanup failure')
        return unlink(path, *args, **kwargs)
    def observe_restore(source, target):
        if Path(target) == pointer and Path(source).read_bytes() == old_pointer:
            assert old_manifest.exists() and old_feed.exists()
        return replace(source, target)
    with monkeypatch.context() as context:
        context.setattr(Path, 'unlink', fail_cleanup)
        context.setattr(channel1.os, 'replace', observe_restore)
        with pytest.raises(OSError, match='cleanup failure'):
            channel1.publish(root, stage, metadata_sha256=metadata_digest, package_sha256=digest, protocol=protocol)
    assert pointer.read_bytes() == old_pointer
    protocol.validate_manifest(json.loads(old_manifest.read_bytes()), json.loads(old_pointer), request['publicBase'], allow_native=True)


def test_preparation_rejects_unbound_package_facts(tmp_path):
    request, digest, _ = package(tmp_path/'input', 109)
    for overrides in ({'expected_sha256': '0'*64}, {'source_commit': 'c'*40},
                      {'windows_version': '0.0.110.0'}, {'build_id': 'e'*32}):
        arguments = dict(expected_sha256=digest, source_commit=request['commit'],
                         windows_version=request['windowsVersion'], build_id=request['buildId'], native_digest='AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=')
        arguments.update(overrides)
        target = tmp_path/'invalid.json'
        with pytest.raises(ValueError): channel1.prepare(tmp_path/'input/SemantierAgent.msix', target, **arguments)
        assert not target.exists()


@pytest.mark.platforms('windows')
def test_native_powershell_owner_publishes_twice_and_preserves_current_cleanup(tmp_path):
    source = Path(__file__).resolve().parents[2]
    stage = tmp_path/'upload'; stage.mkdir()
    prepared = tmp_path/'prepared'
    request, digest, _ = package(prepared, 111)
    bridge = tmp_path/'ssh.py'
    bridge.write_text('import subprocess,sys,shlex\na=shlex.split(sys.argv[1]);sys.exit(subprocess.run([sys.executable,*a[1:]],check=False).returncode)\n', encoding='utf-8')
    script = tmp_path/'simulate.ps1'
    script.write_text(r'''
param($Source,$Python,$Public,$Stage,$Signed,$Metadata,$Digest,$Build,$Bridge)
$ErrorActionPreference='Stop'
foreach($file in @('deploy-ecs.ps1','deploy-ecs-channel1.ps1')){
 $ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $Source ('scripts/'+$file)),[ref]$null,[ref]$null)
 foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Hash','Publish-Channel1Artifacts','Get-Channel1ProofCommand','Get-Channel1CleanupCommand')},$true)){
  # SDK digest is the external OS boundary; this disposable ZIP is not signed/installable.
  $text=$f.Extent.Text.Replace('$nativeDigest=Get-NativePackageDigest $Signed', '$nativeDigest="AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="')
  . ([scriptblock]::Create($text))
 }
}
function Run($Exe,$Arguments){ & $Exe @Arguments | Out-Host; if($LASTEXITCODE -ne 0){throw 'Local metadata preparation failed'} }
function Upload($Path,$RemoteFile){Copy-Item -LiteralPath $Path -Destination $RemoteFile}
function Ssh($Command){$r=& $Python $Bridge $Command; if($LASTEXITCODE -ne 0){throw 'Simulated remote CLI failed'};return $r}
$BuildPython=$Python;$SourceCommit='a'*40;$Version='0.0.111.0';$BuildId=$Build
foreach($iteration in 1..2){
 New-Item -ItemType Directory -Path $Stage -Force | Out-Null
 $r=Publish-Channel1Artifacts -Signed $Signed -Root $Public -Stage $Stage -PackageSha256 $Digest -Metadata $Metadata -Publisher (Join-Path $Source 'scripts/releases/channel1.py')
 $value=$r | ConvertFrom-Json
 if($value.buildId -cne $Build -or $value.stageCleanup -cne 'removed'){throw 'Publication binding or stage cleanup mismatch'}
 if($iteration -eq 2 -and !$value.alreadyPublished){throw 'Repeated deployment was not idempotent'}
}
Get-Channel1ProofCommand -Root $Public -PackageSha256 $Digest -Build $Build | Set-Content (Join-Path (Split-Path -Parent $Metadata) 'proof.sh')
Get-Channel1CleanupCommand -Root $Public -Build $Build | Set-Content (Join-Path (Split-Path -Parent $Metadata) 'cleanup.sh')
'NATIVE_CHANNEL1_SIMULATION_PASS'
''', encoding='utf-8')
    root = tmp_path/'public'; root.mkdir()
    (root/'releases/installers').mkdir(parents=True)
    (root/'releases/installers/semantier-setup-ch1.exe').write_bytes(b'existing fixture')
    result = subprocess.run(['pwsh.exe', '-NoProfile', '-File', str(script), str(source), sys.executable,
                             root.as_posix(), stage.as_posix(), str(prepared/'SemantierAgent.msix'), str(tmp_path/'metadata.json'),
                             digest, request['buildId'], str(bridge)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'NATIVE_CHANNEL1_SIMULATION_PASS' in result.stdout
    git = shutil.which('git.exe')
    bash = next((p/'bin/bash.exe' for p in Path(git).parents if (p/'bin/bash.exe').is_file()), None) if git else None
    if not bash or not bash.is_file(): pytest.skip('Git Bash is required for the native shell cleanup fixture')
    tools = tmp_path/'bin';tools.mkdir()
    python3 = tools/'python3'
    python3.write_text('#!/bin/sh\nexec '+shlex.quote(Path(sys.executable).as_posix())+' "$@"\n', encoding='utf-8')
    python3.chmod(0o755)
    environment = dict(__import__('os').environ)
    environment['PATH'] = str(tools)+';'+environment['PATH']
    old = root/'releases/channel-builds'/('e'*32);old.mkdir()
    (old/'obsolete.msix').write_bytes(b'old fixture')
    (old/'semantier-setup-ch1.exe').write_bytes(b'preserve setup')
    for name in ('proof.sh','cleanup.sh','cleanup.sh','proof.sh'):
        path = tmp_path/name;path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
        result = subprocess.run([str(bash), str(path)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20, env=environment, check=False)
        assert result.returncode == 0, result.stdout+result.stderr
    assert not (old/'obsolete.msix').exists()
    assert (old/'semantier-setup-ch1.exe').exists()
    assert (root/'releases/channels/semantier-acceptance.json').exists()

def test_descriptor_free_single_msix_metadata(tmp_path):
    _, digest, _ = package(tmp_path/'stage', 112)
    data = json.loads((tmp_path/'stage/channel1-metadata.json').read_bytes())
    assert 'feed' not in data
    assert data['manifest']['schema'] == 2
    row = data['manifest']['packages'][0]
    assert row['variant'] == 'bundled'
    assert row['feed']['key'].endswith('/native-graph.json')
    graph = data['graph']
    assert len(graph['objects']) == 1
    assert graph['root'] == row['artifact']['key']
    assert graph['objects'][0]['kind'] == 'main-package'
    assert graph['objects'][0]['sha256'] == digest
    assert graph['objects'][0]['nativeDigest'] == 'AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE='


def test_uploaded_native_metadata_rejects_rebound_graph_and_identity(tmp_path):
    root=tmp_path/'public';root.mkdir()
    _, _, _ = publish(root, tmp_path/'current', 110)
    pointer=root/channel1.POINTER
    before=pointer.read_bytes()
    for field, value in (('publisher','CN=Wrong'),('identity','Wrong.Identity'),('version','0.0.1.0'),('nativeDigest','invalid')):
        stage=tmp_path/field
        _, digest, _=package(stage,111)
        file=stage/'channel1-metadata.json';data=json.loads(file.read_bytes())
        data['graph']['objects'][0][field]=value
        file.write_bytes(protocol.canonical_json(data))
        with pytest.raises(ValueError):
            channel1.publish(root,stage,metadata_sha256=channel1.sha(file),package_sha256=digest,protocol=protocol)
        assert pointer.read_bytes()==before


@pytest.mark.platforms('windows')
def test_production_native_graph_reader_accepts_single_msix_and_refuses_bad_bindings(tmp_path):
    request, _, _=package(tmp_path/'stage',112)
    data=json.loads((tmp_path/'stage/channel1-metadata.json').read_bytes())
    native={'graph':data['graph'],'base':channel1.BASE,'identity':'Semantier.Agent.Acceptance',
            'publisher':'CN=Semantier Technology, O=Semantier Technology','version':request['windowsVersion']}
    native['graph']['objects'][0]['size']=3_000_000_000
    file=tmp_path/'request.json';file.write_text(json.dumps(native),encoding='utf-8')
    source=Path(__file__).resolve().parents[2]/'apps/desktop/scripts/native-package-update.ps1'
    script=tmp_path/'reader.ps1'
    script.write_text("param($Source,$InputFile)\n$ErrorActionPreference='Stop'\n. $Source\n$r=Get-Content -Raw $InputFile|ConvertFrom-Json\n$rows=@(Read-NativeUpdateGraph $r)\nif($rows.Count -ne 1){throw 'Unexpected graph'}\nforeach($field in @('publisher','identity','version','nativeDigest')){\n $copy=$r|ConvertTo-Json -Depth 16|ConvertFrom-Json; $copy.graph.objects[0].$field='invalid'\n $refused=$false;try{$null=Read-NativeUpdateGraph $copy}catch{$refused=$true}\n if(!$refused){throw ('Unsafe graph accepted: '+$field)}\n}\n'NATIVE_SINGLE_GRAPH_PASS'\n",encoding='utf-8')
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',str(script),str(source),str(file)],capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'NATIVE_SINGLE_GRAPH_PASS' in result.stdout


def test_new_sequence_cannot_publish_a_lower_version(tmp_path):
    root=tmp_path/'public';root.mkdir()
    publish(root,tmp_path/'current',110)
    stage=tmp_path/'new';_,digest,_=package(stage,111)
    file=stage/'channel1-metadata.json';data=json.loads(file.read_bytes())
    data['manifest']['request']['windowsVersion']='0.0.109.0'
    data['manifest']['packages'][0]['version']='0.0.109.0'
    file.write_bytes(protocol.canonical_json(data))
    before=(root/channel1.POINTER).read_bytes()
    with pytest.raises(ValueError):
        channel1.publish(root,stage,metadata_sha256=channel1.sha(file),package_sha256=digest,protocol=protocol)
    assert (root/channel1.POINTER).read_bytes()==before


def test_legacy_descriptor_and_metadata_remain_after_native_promotion(tmp_path):
    root=tmp_path/'public';root.mkdir()
    _,_,_=publish(root,tmp_path/'current',110)
    pointer=root/channel1.POINTER;record=json.loads(pointer.read_bytes())
    native_head=record.pop('nativeHead');record['head']=native_head
    old_path=root/native_head['manifestKey'];old=json.loads(old_path.read_bytes())
    old['schema']=1;row=old['packages'][0];row.pop('native')
    row['feed']['key']=row['artifact']['key'].rsplit('/',1)[0]+'/win32/stable.appinstaller'
    old_body=protocol.canonical_json(old);old_path.write_bytes(old_body)
    record['head']['sha256']=hashlib.sha256(old_body).hexdigest();pointer.write_bytes(protocol.canonical_json(record))
    descriptor=root/row['feed']['key'];descriptor.parent.mkdir();descriptor.write_bytes(b'legacy descriptor preserved')
    publish(root,tmp_path/'new',111)
    assert old_path.read_bytes()==old_body
    assert descriptor.read_bytes()==b'legacy descriptor preserved'
    assert json.loads(pointer.read_bytes())['head'] is None

@pytest.mark.platforms('windows')
def test_production_native_options_close_only_target_after_review(tmp_path):
    """Real SDK options + production caller; intercept only OS reads/deployment, never install."""
    source=Path(__file__).resolve().parents[2]/'apps/desktop/scripts/native-package-update.ps1'
    script=tmp_path/'native-options.ps1'
    script.write_text(r'''
param($Source,$Directory)
$ErrorActionPreference='Stop'
. $Source
$binding=Get-Content (Join-Path (Split-Path -Parent $Source) 'native-package-digest.ps1') -Raw
$call='$operation=$manager.AddPackageByUriAsync([Uri]$Root,$options)'
if(!$binding.Contains($call)){throw 'Native SDK boundary changed; inspect before testing'}
$replacement='$script:CapturedOptions=$options; return @{status="SUCCESS";startedAt=[DateTime]::UtcNow.ToString("o");producedAt=[DateTime]::UtcNow.ToString("o");activityId="11111111-1111-1111-1111-111111111111";isRegistered=$false}'
. ([scriptblock]::Create($binding.Replace($call,$replacement)))
function Get-AppxPackage { [pscustomobject]@{Name='Fixture';PackageFullName='fixture-baseline';Publisher='CN=Fixture'} }
$r=@{identity='Fixture';fullName='fixture-baseline';publisher='CN=Fixture';base='https://localhost:1';graph=@{root='releases/channel-builds/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/fixture.msix'};manifestSha256=('a'*64);authorizedSid='fixture';session=1}
$rows=@(@{key=$r.graph.root;nativeDigest=[Convert]::ToBase64String((,[byte]1*32))})
Invoke-NativePackageUpdate @{request=$r;rows=$rows;directory=$Directory}
if(!$script:CapturedOptions.ForceTargetAppShutdown){throw 'Production native update must opt into target-package shutdown'}
if($script:CapturedOptions.ForceAppShutdown -or $script:CapturedOptions.ForceUpdateFromAnyVersion){throw 'Dependency shutdown or downgrade was enabled'}
$null=Invoke-BoundNativePackageUri @('https://localhost:1/fixture.msix') @([Convert]::ToBase64String((,[byte]1*32))) 'https://localhost:1/fixture.msix' 1
if($script:CapturedOptions.ForceTargetAppShutdown -or $script:CapturedOptions.ForceAppShutdown -or $script:CapturedOptions.ForceUpdateFromAnyVersion){throw 'Default SDK caller gained shutdown or downgrade authority'}
'SDK_OPTIONS_ONLY_PASS_NO_DEPLOYMENT'
''',encoding='utf-8')
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',str(script),str(source),str(tmp_path)],capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'SDK_OPTIONS_ONLY_PASS_NO_DEPLOYMENT' in result.stdout
