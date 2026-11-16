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
        archive.writestr(zipfile.ZipInfo('AppxManifest.xml'), f'<Package><Identity Name="Semantier.Agent.Acceptance" Publisher="CN=Semantier Technology, O=Semantier Technology" Version="{request["windowsVersion"]}" ProcessorArchitecture="x64"/></Package>')
        archive.writestr(zipfile.ZipInfo('app/resources/install-stamp.json'), json.dumps({'commit': request['commit'], 'dirty': False, 'channelBuild': request}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = stage/'channel1-metadata.json'
    channel1.prepare(path, metadata, expected_sha256=digest, source_commit=request['commit'],
                     windows_version=request['windowsVersion'], build_id=request['buildId'])
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
    manifest = json.loads((root/record['head']['manifestKey']).read_bytes())
    protocol.validate_manifest(manifest, record, request['publicBase'])
    artifact = root/manifest['packages'][0]['artifact']['key']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    assert artifact.samefile(root/'releases/SemantierAgent.msix')
    _, _, second = publish(root, tmp_path/'stage2', 109)
    assert record_path.read_bytes() == original
    assert first['buildId'] == second['buildId'] == request['buildId']
    assert second['alreadyPublished']
    marker = root/'unrelated.txt'; marker.write_text('preserve', encoding='utf-8')
    unrelated = root/'releases/channels/another.json'; unrelated.write_text('{}', encoding='utf-8')
    publish(root, tmp_path/'stage3', 110)
    current = json.loads(record_path.read_bytes())
    assert current['head']['sequence'] == 110 and not {'nativeHead', 'nativeArchive'} & current.keys()
    assert not (root/record['head']['manifestKey']).exists()
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
    assert json.loads(pointer.read_bytes())['head']['sequence'] == 110


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
    old_feed = old_manifest.parent/'win32/stable.appinstaller'
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
    protocol.validate_manifest(json.loads(old_manifest.read_bytes()), json.loads(old_pointer), request['publicBase'])


def test_preparation_rejects_unbound_package_facts(tmp_path):
    request, digest, _ = package(tmp_path/'input', 109)
    for overrides in ({'expected_sha256': '0'*64}, {'source_commit': 'c'*40},
                      {'windows_version': '0.0.110.0'}, {'build_id': 'e'*32}):
        arguments = dict(expected_sha256=digest, source_commit=request['commit'],
                         windows_version=request['windowsVersion'], build_id=request['buildId'])
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
  . ([scriptblock]::Create($f.Extent.Text))
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
