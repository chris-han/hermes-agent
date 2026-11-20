"""Current-only Channel 1 publication uses real files and the native operator seam."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import zipfile
import types

import pytest

from hermes_cli import release_channels as protocol
from scripts.releases import channel1


def package(stage, sequence, *, version=None):
    stage.mkdir(parents=True, exist_ok=True)
    request = {
        'schema': 1, 'buildId': f'{sequence:032x}', 'channel': 'semantier-acceptance',
        'sequence': sequence, 'repository': 'chris-han/hermes-agent', 'commit': 'a'*40,
        'controllerCommit': 'a'*40, 'sourceVersion': '0.0.0', 'version': f'0.0.{sequence}',
        'windowsVersion': f'0.0.{sequence}.0', 'releaseTag': f'v0.0.{sequence}',
        'publicBase': 'https://cloud.semantier.cn',
        'bundleEnv': {'HERMES_HOME': None, 'HERMES_DATA_DIR_SUFFIX': '-semantier-acceptance'},
        'identity': {'token': 'b0fdc2a7e4b08d16', 'displayName': 'Semantier Agent Acceptance',
                     'appId': 'com.semantier.agent.acceptance', 'appNamePascal': 'SemantierAgentAcceptance',
                     'artifactNamePascal': 'SemantierAgent', 'cliName': 'semantier-agent-acceptance',
                     'msixAppIdWithOrg': 'Semantier.Agent.Acceptance', 'windowsExecutableName': 'Semantier Agent'}}
    if version is not None:
        request.update(buildId=version,windowsVersion=version,version='.'.join(version.split('.')[:3]),semantierScope='semantier-windows-x64/v1')
        request['releaseTag']='v'+request['version']
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


def test_expected_previous_is_checked_under_lock_before_mutation(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    pointer = root/channel1.POINTER
    before = pointer.read_bytes()
    _, digest, metadata = package(tmp_path/'new', 122)
    with pytest.raises(ValueError, match='Concurrent'):
        channel1.publish(root, tmp_path/'new', metadata_sha256=metadata,
                         package_sha256=digest, protocol=protocol,
                         expected_previous_sha256='0'*64)
    assert pointer.read_bytes() == before
    assert not (root/'releases/channel-builds'/f'{122:032x}').exists()


def test_durable_rollback_preserves_previous_closure_and_never_downgrades_os(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    pointer = root/channel1.POINTER
    before = pointer.read_bytes()
    old_package = (root/'releases/SemantierAgent.msix').read_bytes()
    _, digest, metadata = package(tmp_path/'new', 122)
    backup = root.parent/'.channel1-recovery-test'
    channel1.publish(root, tmp_path/'new', metadata_sha256=metadata,
                     package_sha256=digest, protocol=protocol,
                     expected_previous_sha256=hashlib.sha256(before).hexdigest(), recovery=backup)
    # The old native metadata and package must still be available to in-flight readers.
    assert (root/'releases/channel-builds'/f'{121:032x}'/'build.json').exists()
    result = channel1.restore(root, backup)
    assert result['osDowngrade'] == 'NOT_ATTEMPTED'
    assert pointer.read_bytes() == before
    assert (root/'releases/SemantierAgent.msix').read_bytes() == old_package
    assert channel1.restore(root, backup)['result'] == 'POINTER_RESTORED'
    assert backup.exists()


def test_rollback_refuses_concurrent_pointer(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    _, digest, metadata = package(tmp_path/'new', 122)
    backup = root.parent/'.channel1-recovery-test'
    channel1.publish(root, tmp_path/'new', metadata_sha256=metadata,
                     package_sha256=digest, protocol=protocol,
                     expected_previous_sha256=channel1.sha(root/channel1.POINTER), recovery=backup)
    publish(root, tmp_path/'other', 123)
    with pytest.raises(ValueError, match='Concurrent'):
        channel1.restore(root, backup)


def test_recovery_tampering_refuses_republish_before_pointer_change(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    _, digest, metadata = package(tmp_path/'new', 122)
    backup = root.parent/'.channel1-recovery-test'
    args = dict(metadata_sha256=metadata, package_sha256=digest, protocol=protocol,
                expected_previous_sha256=channel1.sha(root/channel1.POINTER), recovery=backup)
    channel1.publish(root, tmp_path/'new', **args)
    before = (root/channel1.POINTER).read_bytes()
    (backup/'2').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Recovery SHA256'):
        channel1.publish(root, tmp_path/'new', **args)
    assert (root/channel1.POINTER).read_bytes() == before


def test_native_staging_probes_before_pointer_and_restores_route(tmp_path, monkeypatch):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    before = (root/channel1.POINTER).read_bytes()
    caddy = tmp_path/'Caddyfile'
    old = '/releases/channel-builds/'+f'{121:032x}'+'/'
    body = '\n'.join(old+name for name in ('SemantierAgent.msix', 'build.json', 'native-graph.json')).encode()
    caddy.write_bytes(body)
    real_path = Path
    monkeypatch.setattr(channel1, 'Path', lambda p: caddy if str(p) == '/opt/semantier-cloud-live-v1/Caddyfile' else real_path(p))
    monkeypatch.setattr(channel1, '_apply_route', lambda path, data: path.write_bytes(data))
    _, digest, metadata = package(tmp_path/'new', 122)
    backup = root.parent/'.channel1-recovery-test'
    args = dict(metadata_sha256=metadata, package_sha256=digest, protocol=protocol,
                expected_previous_sha256=hashlib.sha256(before).hexdigest(), recovery=backup)
    channel1.stage_native(root, tmp_path/'new', caddy_sha256=hashlib.sha256(body).hexdigest(), **args)
    assert (root/channel1.POINTER).read_bytes() == before
    assert (root/'releases/channel-builds'/f'{122:032x}'/'SemantierAgent.msix').exists()
    staged_caddy = caddy.read_bytes()
    assert old.encode() in staged_caddy and f'{122:032x}'.encode() in staged_caddy
    channel1.stage_native(root, tmp_path/'new', caddy_sha256=hashlib.sha256(body).hexdigest(), **args)
    assert caddy.read_bytes() == staged_caddy
    channel1.publish(root, tmp_path/'new', **args)
    channel1.restore(root, backup)
    assert (root/channel1.POINTER).read_bytes() == before
    assert caddy.read_bytes() == body
    # A prior safe rollback is an owned resumable state, not a foreign writer.
    channel1.stage_native(root, tmp_path/'new', caddy_sha256=hashlib.sha256(body).hexdigest(), **args)
    assert (root/channel1.POINTER).read_bytes() == before
    assert caddy.read_bytes() == staged_caddy


@pytest.mark.parametrize('admin_off', [True, False])
def test_live_route_uses_restart_only_when_admin_is_disabled(tmp_path, monkeypatch, admin_off):
    path = tmp_path/'Caddyfile'; path.write_bytes(b'old')
    body = b'{\n admin off\n}\n' if admin_off else b':80 { respond "ok" }\n'
    calls = []
    inspected = [{'Id': 'edge-id', 'State': {'Running': True},
                  'Mounts': [{'Source': str(path), 'Destination': '/etc/caddy/Caddyfile'}]}]
    monkeypatch.setattr(channel1.subprocess, 'check_output', lambda *a, **k: json.dumps(inspected).encode())
    def run(args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(channel1.subprocess, 'run', run)
    channel1._apply_route(path, body)
    assert path.read_bytes() == body
    assert calls[0][:3] == ['docker', 'exec', '-i']
    assert calls[-1] == (['docker', 'restart', 'edge-id'] if admin_off else
                        ['docker', 'exec', 'edge-id', 'caddy', 'reload', '--config', '/etc/caddy/Caddyfile', '--adapter', 'caddyfile'])


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('failure', ['421', 'http1', 'wrong-bytes', 'wrong-graph', None])
def test_native_public_probe_checks_real_response_predicates(tmp_path, monkeypatch, failure,legacy,capsys):
    _, _, _ = package(tmp_path/'new', 122)
    data = json.loads((tmp_path/'new/channel1-metadata.json').read_bytes())
    manifest = data['manifest']; graph = protocol.canonical_json(data['graph'])
    if legacy:
        manifest['request']['publicBase']='https://download.semantier.cn'
        manifest['packages'][0]['native']['schema']=1
        del manifest['packages'][0]['native']['packageBase']
    probe = tmp_path/'probe.msix'; probe.write_bytes(bytes(range(256))*1024)
    manifest['packages'][0]['artifact']['size'] = probe.stat().st_size
    # Exercise the actual embedded Python probe, mocking only external HTTP.
    ps = (Path(__file__).resolve().parents[2]/'scripts/deploy-ecs-channel1.ps1').read_text()
    code = ps.split('function Test-NativePublic {', 1)[1].split("$code=@'\n", 1)[1].split("\n'@", 1)[0]

    class Session:
        def __init__(self, **kwargs):
            self.version = {'v2':3,'v3only':30,'v1':2}[kwargs['http_version']]
            assert kwargs['curl_options'][2] in (65536,4194304)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            package_base='https://download.semantier.cn/' if legacy else 'https://assets.semantier.cn/'
            assert url.startswith(package_base if url.endswith('.msix') else 'https://cloud.semantier.cn/')
            assert kwargs['allow_redirects'] is False
            status = 200; headers = {'Content-Type':'application/json','Cache-Control':'public,max-age=31536000,immutable'}; content = protocol.canonical_json(manifest)
            if url.endswith('native-graph.json'):
                content = graph if failure != 'wrong-graph' else b'changed'
            if url.endswith('.msix'):
                start, end = map(int, kwargs['headers']['Range'][6:].split('-'))
                content = probe.read_bytes()[start:end+1]
                if failure == 'wrong-bytes': content = b'changed'
                status = 421 if failure == '421' or (legacy and self.version==2) else 206
                headers['Content-Range'] = f'bytes {start}-{end}/{manifest["packages"][0]["artifact"]["size"]}'
            return types.SimpleNamespace(status_code=status, content=content, headers=headers,
                                         http_version=2 if failure == 'http1' else self.version)
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=types.SimpleNamespace(Session=Session), CurlOpt=types.SimpleNamespace(PROXY=1,MAXFILESIZE_LARGE=2)))
    monkeypatch.setattr(sys, 'argv', ['probe', __import__('base64').b64encode(json.dumps(manifest).encode()).decode(), str(probe), 'False', str(legacy)])
    if failure:
        with pytest.raises(AssertionError): exec(code, {})
    else:
        exec(code, {})
        result=json.loads(capsys.readouterr().out)
        assert result['packageBase']==('https://download.semantier.cn' if legacy else 'https://assets.semantier.cn')
        assert {r['origin'] for r in result['responses']}=={'cloud-metadata','legacy-ecs-package' if legacy else 'oss-package'}
        assert ('LEGACY_BASELINE' in result['result'])==legacy


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


def test_capacity_recovery_preserves_links_unknown_and_current(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    old_manifest = root/protocol.build_prefix(f'{121:032x}')/'build.json'
    old_metadata = old_manifest.read_bytes()
    publish(root, tmp_path/'new', 122)
    old_manifest.write_bytes(old_metadata)
    pointer = root/channel1.POINTER
    before = pointer.read_bytes()
    old = root/protocol.build_prefix(f'{121:032x}')/'SemantierAgent.msix'
    # Recovery snapshots share the inode: removing one name earns no capacity.
    import os
    os.link(old, tmp_path/'retained.msix')
    unknown = root/'releases/channel-builds'/'unknown.msix'
    unknown.write_bytes(b'unknown')
    result = channel1.recover_space(root, expected_previous_sha256=channel1.sha(pointer),
                                   required_bytes=100, free_bytes=lambda: 0)
    assert result['result'] == 'BLOCKED_INSUFFICIENT_CAPACITY'
    assert result['reclaimedBytes'] == 0
    assert old.exists() and unknown.exists() and pointer.read_bytes() == before
    assert any(row['reason'] == 'hardlinked' for row in result['retained'])


def test_capacity_recovery_is_idempotent_and_skips_inventory_when_ready(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'new', 122)
    for _ in range(2):
        result = channel1.recover_space(root, expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                       required_bytes=100, free_bytes=lambda: 100)
        assert result['result'] == 'CAPACITY_READY'
        assert result['inventory'] == 'SKIPPED_CAPACITY_SUFFICIENT'
        assert result['reclaimedBytes'] == 0


def test_capacity_recovery_checks_pointer_before_cleanup(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'new', 122)
    with pytest.raises(ValueError, match='Concurrent'):
        channel1.recover_space(root, expected_previous_sha256='0'*64,
                              required_bytes=100, free_bytes=lambda: 0)


def test_capacity_recovery_deletes_only_verified_obsolete_single_link(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 120)
    old_manifest = root/protocol.build_prefix(f'{120:032x}')/'build.json'
    old_metadata = old_manifest.read_bytes()
    publish(root, tmp_path/'new', 122)
    old_manifest.write_bytes(old_metadata)
    old = root/protocol.build_prefix(f'{120:032x}')/'SemantierAgent.msix'
    detached = old.with_suffix('.fixture')
    detached.write_bytes(old.read_bytes()); detached.replace(old)
    size = old.stat().st_size
    result = channel1.recover_space(root, expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                   required_bytes=100, free_bytes=lambda: 0 if old.exists() else 100)
    assert result['result'] == 'CAPACITY_READY'
    assert result['reclaimedBytes'] == 100
    assert result['removed'][0]['logicalBytes'] == size
    assert not old.exists()
    again = channel1.recover_space(root, expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                  required_bytes=100, free_bytes=lambda: 100)
    assert again['removed'] == []


def test_capacity_keeps_rollback_snapshot_and_current_upload(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 121)
    old_pointer = (root/channel1.POINTER).read_bytes()
    old_manifest = root/protocol.build_prefix(f'{121:032x}')/'build.json'
    old_metadata = old_manifest.read_bytes()
    publish(root, tmp_path/'new', 122)
    old_manifest.write_bytes(old_metadata)
    old = old_manifest.with_name('SemantierAgent.msix')
    detached = old.with_suffix('.fixture'); detached.write_bytes(old.read_bytes()); detached.replace(old)
    recovery = tmp_path/('.channel1-recovery-'+'a'*32); recovery.mkdir()
    (recovery/'2').write_bytes(old_pointer)
    (recovery/'recovery.json').write_text(json.dumps({'files':[{'key':channel1.POINTER,'saved':'2','sha256':hashlib.sha256(old_pointer).hexdigest()}]}))
    stage = tmp_path/('.semantier-stage-'+'b'*32); stage.mkdir()
    (stage/'SemantierAgent.msix').write_bytes(b'partial')
    called = []
    result = channel1.recover_space(root,expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                   required_bytes=100,free_bytes=lambda:0,protected_stage=stage,
                                   retire_images=lambda:called.append(True) or {'removed':[]})
    assert result['result'] == 'BLOCKED_INSUFFICIENT_CAPACITY'
    assert old.exists() and (stage/'SemantierAgent.msix').read_bytes() == b'partial'
    assert old.relative_to(root).as_posix() in result['protectedReferences']
    assert called == [True]


def test_capacity_keeps_publicly_referenced_package(tmp_path):
    root = tmp_path/'public'; root.mkdir()
    publish(root, tmp_path/'old', 120)
    manifest = root/protocol.build_prefix(f'{120:032x}')/'build.json'; metadata = manifest.read_bytes()
    publish(root, tmp_path/'new', 122); manifest.write_bytes(metadata)
    old = manifest.with_name('SemantierAgent.msix')
    detached = old.with_suffix('.fixture'); detached.write_bytes(old.read_bytes()); detached.replace(old)
    landing = tmp_path/'landing'; landing.mkdir()
    (landing/'index.html').write_text('https://download.semantier.cn/'+old.relative_to(root).as_posix())
    result = channel1.recover_space(root,expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                   required_bytes=100,free_bytes=lambda:0,reference_roots=[landing])
    assert result['removed'] == [] and old.exists()
    assert any(row['reason']=='release-reference' for row in result['retained'])


@pytest.mark.parametrize('hardlink,live', [(False,False),(True,False),(False,True)])
def test_expired_stage_requires_single_links_and_no_writer(tmp_path,monkeypatch,hardlink,live):
    import stat
    root = tmp_path/'public'; root.mkdir(); publish(root,tmp_path/'new',122)
    stage = tmp_path/('.semantier-stage-'+'c'*32); stage.mkdir()
    payload = stage/'SemantierAgent.msix'; payload.write_bytes(b'expired')
    original_stat, original_glob, original_dir = Path.lstat, Path.glob, Path.is_dir
    def observed_stat(path,*args,**kwargs):
        actual = original_stat(path,*args,**kwargs)
        if path in (stage,payload):
            return types.SimpleNamespace(st_mode=(stat.S_IFDIR|0o700) if path==stage else (stat.S_IFREG|0o600),
                                         st_uid=0,st_mtime=0,st_nlink=2 if hardlink else 1,st_size=actual.st_size)
        return actual
    monkeypatch.setattr(Path,'lstat',observed_stat)
    monkeypatch.setattr(Path,'is_dir',lambda p: True if p==Path('/proc') else original_dir(p))
    monkeypatch.setattr(Path,'glob',lambda p,pattern,**kwargs: [] if p==Path('/proc') else original_glob(p,pattern,**kwargs))
    monkeypatch.setattr(channel1.os,'getuid',lambda:0,raising=False)
    # The selected active stage is retained independently of age and process scans.
    result = channel1.recover_space(root,expected_previous_sha256=channel1.sha(root/channel1.POINTER),
                                   required_bytes=100,free_bytes=lambda:0 if payload.exists() else 100,
                                   protected_stage=stage if live else None,now=100000)
    if hardlink or live:
        assert payload.exists() and result['removed']==[]
    else:
        assert not stage.exists() and result['result']=='CAPACITY_READY'
        assert result['removed'][0]['kind']=='expired-upload'


def retirement_fixture(tmp_path):
    import os
    root=tmp_path/'public';root.mkdir()
    publish(root,tmp_path/'old',121)
    old_manifest=root/protocol.build_prefix(f'{121:032x}')/'build.json'
    old_body=old_manifest.read_bytes()
    old_package=old_manifest.with_name('SemantierAgent.msix')
    old_pointer=(root/channel1.POINTER).read_bytes()
    # Upload-stage alias is independently owned; remove it from this fixture.
    (tmp_path/'old'/'SemantierAgent.msix').unlink()
    publish(root,tmp_path/'new',122);old_manifest.write_bytes(old_body)
    recovery=tmp_path/('.channel1-recovery-'+f'{122:032x}');recovery.mkdir()
    os.link(old_package,recovery/'0')
    (recovery/'2').write_bytes(old_pointer)
    (recovery/'recovery.json').write_text(json.dumps({'root':str(root),
      'afterPointerSha256':channel1.sha(root/channel1.POINTER),
      'afterPackageSha256':channel1.sha(root/'releases/SemantierAgent.msix'),
      'files':[{'key':'releases/SemantierAgent.msix','saved':'0','sha256':channel1.sha(old_package)},
               {'key':channel1.POINTER,'saved':'2','sha256':channel1.sha(recovery/'2')}]}))
    return root,old_package,recovery


def test_explicit_retirement_removes_all_known_links_preserves_current_and_repeats(tmp_path,monkeypatch):
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    root,old,recovery=retirement_fixture(tmp_path)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    kwargs=dict(build_id=f'{121:032x}',package_sha256=channel1.sha(old),
                expected_previous_sha256=channel1.sha(root/channel1.POINTER),assert_inactive=lambda paths:None)
    with channel1._lock(root):
        plan=channel1.retire_package(root,**kwargs)
    assert plan['result']=='RETIREMENT_PLAN' and len(plan['aliases'])==2
    assert all(p.read_bytes()==body for p,body in before.items())
    current=(root/'releases/SemantierAgent.msix').read_bytes()
    result=channel1.retire_package(root,apply=True,**kwargs)
    assert result['result']=='PACKAGE_RETIRED' and not old.exists() and not (recovery/'0').exists()
    assert (recovery/'recovery.retired').exists() and not (recovery/'recovery.json').exists()
    assert (root/'releases/SemantierAgent.msix').read_bytes()==current
    assert channel1.retire_package(root,apply=True,**kwargs)['result']=='ALREADY_RETIRED'


@pytest.mark.parametrize('blocker',['current','pointer','outside-link','live','pending-recovery','unknown-owner'])
def test_explicit_retirement_fails_closed_before_changes(tmp_path,blocker,monkeypatch):
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    import os
    root,old,recovery=retirement_fixture(tmp_path)
    kwargs=dict(build_id=f'{121:032x}',package_sha256=channel1.sha(old),
                expected_previous_sha256=channel1.sha(root/channel1.POINTER),assert_inactive=lambda paths:None)
    if blocker=='current':kwargs['build_id']=f'{122:032x}'
    if blocker=='pointer':kwargs['expected_previous_sha256']='0'*64
    if blocker=='outside-link':os.link(old,tmp_path.parent/(tmp_path.name+'-outside.msix'))
    if blocker=='pending-recovery':
        p=recovery/'recovery.json';value=json.loads(p.read_bytes());value['afterPointerSha256']='0'*64;p.write_text(json.dumps(value))
    if blocker=='unknown-owner':
        unknown=tmp_path/'unowned-cache';unknown.mkdir();os.link(old,unknown/'file.msix')
    if blocker=='live':
        def live(paths):raise ValueError('Live owner')
        kwargs['assert_inactive']=live
    with pytest.raises(ValueError):channel1.retire_package(root,apply=True,**kwargs)
    assert old.exists() and (recovery/'0').exists() and (recovery/'recovery.json').exists()


@pytest.mark.parametrize('actor',[0,1001])
def test_retirement_root_can_maintain_service_owned_directory(tmp_path,monkeypatch,actor):
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    root,old,_=retirement_fixture(tmp_path)
    original=Path.stat
    def observed(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path==root:
            fields={name:getattr(value,name) for name in dir(value) if name.startswith('st_')}
            fields.update(st_uid=912,st_mode=0o40755)
            return types.SimpleNamespace(**fields)
        return value
    monkeypatch.setattr(Path,'stat',observed)
    monkeypatch.setattr(channel1.os,'geteuid',lambda:actor,raising=False)
    kwargs=dict(build_id=f'{121:032x}',package_sha256=channel1.sha(old),
                expected_previous_sha256=channel1.sha(root/channel1.POINTER),assert_inactive=lambda paths:None)
    if actor==0:
        assert channel1.retire_package(root,**kwargs)['result']=='RETIREMENT_PLAN'
    else:
        with pytest.raises(ValueError,match='actor'):channel1.retire_package(root,apply=True,**kwargs)


def test_oss_stage_keeps_native_verified_metadata_snapshot(tmp_path,monkeypatch):
    """A file replacement after signature verification cannot switch release pins."""
    from agents.client_operations import oss_release
    from types import SimpleNamespace
    stage_dir=tmp_path/'stage';package(stage_dir,125)
    metadata=stage_dir/'channel1-metadata.json';original=json.loads(metadata.read_text())
    monkeypatch.setattr(oss_release,'assume_role',lambda _:object())
    monkeypatch.setattr(channel1.subprocess,'check_output',lambda *a,**kw:'D:\\fixture.msix')
    def native_verified(*args,**kwargs):
        replacement=json.loads(metadata.read_text())
        replacement['manifest']['request']['commit']='b'*40
        metadata.write_text(json.dumps(replacement))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(channel1.subprocess,'run',native_verified)
    monkeypatch.setattr(oss_release,'Objects',lambda _:object())
    seen=[]
    monkeypatch.setattr(oss_release,'stage',lambda p,c,o,obj:seen.append(c) or c)
    result=channel1.stage_oss(stage_dir/'SemantierAgent.msix',metadata,tmp_path/'candidate.json')
    assert result==oss_release.candidate_from_manifest(original['manifest'])
    assert seen==[result]


def test_oss_package_origin_is_bound_separately_from_metadata(tmp_path):
    package(tmp_path/'candidate', 131)
    manifest = json.loads((tmp_path/'candidate/channel1-metadata.json').read_bytes())['manifest']
    manifest['request']['publicBase'] = 'https://cloud.semantier.cn'
    native = manifest['packages'][0]['native']
    native.update(schema=2, packageBase='https://assets.semantier.cn')
    record = channel1._record(manifest, None, protocol)
    assert protocol.validate_manifest(manifest, record, 'https://cloud.semantier.cn', allow_native=True) == manifest


@pytest.mark.parametrize('origin', [
    'https://evil.example', 'http://assets.semantier.cn', 'https://assets.semantier.cn.evil.example',
    'https://assets.semantier.cn/path', 'https://assets.semantier.cn?token=secret',
    'https://user@assets.semantier.cn', 'https://assets.semantier.cn:443',
])
def test_oss_package_origin_rejects_noncanonical_hosts(tmp_path, origin):
    package(tmp_path/'candidate', 131)
    manifest = json.loads((tmp_path/'candidate/channel1-metadata.json').read_bytes())['manifest']
    manifest['request']['publicBase'] = 'https://cloud.semantier.cn'
    manifest['packages'][0]['native'].update(schema=2, packageBase=origin)
    record = channel1._record(manifest, None, protocol)
    with pytest.raises(protocol.ChannelError, match='package origin'):
        protocol.validate_manifest(manifest, record, 'https://cloud.semantier.cn', allow_native=True)


def test_legacy_native_schema_cannot_smuggle_package_origin(tmp_path):
    package(tmp_path/'candidate', 131)
    manifest = json.loads((tmp_path/'candidate/channel1-metadata.json').read_bytes())['manifest']
    manifest['packages'][0]['native']['schema'] = 1
    manifest['packages'][0]['native']['packageBase'] = 'https://evil.example'
    record = channel1._record(manifest, None, protocol)
    with pytest.raises(protocol.ChannelError, match='package origin'):
        protocol.validate_manifest(manifest, record, channel1.BASE, allow_native=True)


def test_prepared_release_emits_cloud_metadata_and_separate_oss_origin(tmp_path):
    package(tmp_path/'candidate', 131)
    manifest = json.loads((tmp_path/'candidate/channel1-metadata.json').read_bytes())['manifest']
    assert manifest['request']['publicBase'] == 'https://cloud.semantier.cn'
    assert manifest['packages'][0]['native']['schema'] == 2
    assert manifest['packages'][0]['native']['packageBase'] == 'https://assets.semantier.cn'


def test_oss_publication_requires_only_metadata_on_ecs_and_commits_pointer_last(tmp_path, monkeypatch):
    root=tmp_path/'public';root.mkdir()
    request,digest,metadata=package(tmp_path/'upload',131)
    (tmp_path/'upload/SemantierAgent.msix').unlink()
    checked=[]
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:checked.append(row['artifact']['sha256']),raising=False)
    real_atomic=channel1._atomic;writes=[]
    def atomic(path,body):
        writes.append(path.relative_to(root).as_posix())
        real_atomic(path,body)
    monkeypatch.setattr(channel1,'_atomic',atomic)
    result=channel1.publish(root,tmp_path/'upload',metadata_sha256=metadata,package_sha256=digest,
                            protocol=protocol,metadata_only=True)
    assert checked==[digest] and result['buildId']==request['buildId']
    assert writes[-1]==channel1.POINTER
    assert not list(root.rglob('*.msix'))
    assert not (root/'releases/SemantierAgent.msix.sha256').exists()


def test_missing_oss_bytes_refuses_pointer_change_and_metadata_only_rollback_restores_old_closure(tmp_path, monkeypatch):
    root=tmp_path/'public';root.mkdir()
    publish(root,tmp_path/'old',130)
    before=(root/channel1.POINTER).read_bytes()
    stable=(root/'releases/SemantierAgent.msix').read_bytes()
    _,digest,metadata=package(tmp_path/'upload',131)
    (tmp_path/'upload/SemantierAgent.msix').unlink()
    def refused(row):raise ValueError('OSS bytes unavailable')
    monkeypatch.setattr(channel1,'_verify_oss_head',refused,raising=False)
    args=dict(metadata_sha256=metadata,package_sha256=digest,protocol=protocol,metadata_only=True,
              expected_previous_sha256=hashlib.sha256(before).hexdigest(),recovery=root.parent/'.channel1-recovery-oss')
    with pytest.raises(ValueError,match='OSS bytes unavailable'):
        channel1.publish(root,tmp_path/'upload',**args)
    assert (root/channel1.POINTER).read_bytes()==before
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    channel1.publish(root,tmp_path/'upload',**args)
    assert (root/'releases/SemantierAgent.msix').read_bytes()==stable
    assert not (root/'releases/channel-builds'/f'{131:032x}'/'SemantierAgent.msix').exists()
    channel1.restore(root,args['recovery'])
    assert (root/channel1.POINTER).read_bytes()==before
    assert (root/'releases/SemantierAgent.msix').read_bytes()==stable


def test_oss_metadata_staging_preserves_head_and_caddy_without_ecs_package(tmp_path, monkeypatch):
    root=tmp_path/'public';root.mkdir();publish(root,tmp_path/'old',130)
    previous=(root/channel1.POINTER).read_bytes()
    _,digest,metadata=package(tmp_path/'upload',131)
    (tmp_path/'upload/SemantierAgent.msix').unlink()
    caddy=tmp_path/'Caddyfile';caddy.write_bytes(b'qualified cloud metadata route fixture')
    real_path=Path
    monkeypatch.setattr(channel1,'Path',lambda p: caddy if str(p)=='/opt/semantier-cloud-live-v1/Caddyfile' else real_path(p))
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    def no_route_change(*args):raise AssertionError('Metadata staging must not mutate the qualified Caddy route')
    monkeypatch.setattr(channel1,'_apply_route',no_route_change)
    result=channel1.stage_native(root,tmp_path/'upload',metadata_sha256=metadata,package_sha256=digest,
        protocol=protocol,expected_previous_sha256=hashlib.sha256(previous).hexdigest(),
        recovery=root.parent/'.channel1-recovery-oss-stage',caddy_sha256=hashlib.sha256(caddy.read_bytes()).hexdigest(),metadata_only=True)
    assert result['result']=='IMMUTABLE_METADATA_STAGED_POINTER_UNCHANGED'
    assert (root/channel1.POINTER).read_bytes()==previous
    prefix=root/'releases/channel-builds'/f'{131:032x}'
    assert (prefix/'build.json').is_file() and (prefix/'native-graph.json').is_file()
    assert not (prefix/'SemantierAgent.msix').exists()
    assert caddy.read_bytes()==b'qualified cloud metadata route fixture'


def test_public_oss_stage_uses_verified_snapshot_without_full_readback(tmp_path,monkeypatch):
    from agents.client_operations import oss_release
    from types import SimpleNamespace
    stage_dir=tmp_path/'stage';package(stage_dir,131)
    metadata=stage_dir/'channel1-metadata.json';original=json.loads(metadata.read_text())
    roles=[];monkeypatch.setattr(oss_release,'assume_role',lambda role:roles.append(role) or object())
    monkeypatch.setattr(channel1.subprocess,'check_output',lambda *a,**kw:'D:\\fixture.msix')
    monkeypatch.setattr(channel1.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    monkeypatch.setattr(oss_release,'Objects',lambda _:object())
    monkeypatch.setattr(oss_release,'ReleaseObjects',lambda _:object())
    seen=[]
    def stage_bytes(path,artifact,private,public):
        seen.append(artifact)
        return {**artifact,'state':'OSS_BYTES_STAGED_NOT_ACTIVE','fullDownloadVerification':'PENDING'}
    monkeypatch.setattr(oss_release,'stage_release_bytes',stage_bytes)
    def full_readback(*args):raise AssertionError('Private full-readback mode must not be selected')
    monkeypatch.setattr(oss_release,'stage',full_readback)
    output=tmp_path/'public-staged.json'
    result=channel1.stage_oss(stage_dir/'SemantierAgent.msix',metadata,output,release_bytes=True)
    assert roles==['publish']
    assert seen==[{'buildId':original['manifest']['request']['buildId'],**original['manifest']['packages'][0]['artifact']}]
    assert result==json.loads(output.read_bytes()) and result['fullDownloadVerification']=='PENDING'


def test_metadata_only_pointer_failure_restores_old_head_and_preserves_unowned_binary_residue(tmp_path,monkeypatch):
    root=tmp_path/'public';root.mkdir();publish(root,tmp_path/'old',130)
    previous=(root/channel1.POINTER).read_bytes()
    _,digest,metadata=package(tmp_path/'upload',131)
    (tmp_path/'upload/SemantierAgent.msix').unlink()
    residue=root/'releases/.channel1-package.tmp';residue.write_bytes(b'not owned by metadata transaction')
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    original=channel1._atomic
    def fail_after_pointer(path,body):
        original(path,body)
        if path==root/channel1.POINTER:raise OSError('injected commit failure')
    monkeypatch.setattr(channel1,'_atomic',fail_after_pointer)
    with pytest.raises(OSError,match='injected'):
        channel1.publish(root,tmp_path/'upload',metadata_sha256=metadata,package_sha256=digest,
                          protocol=protocol,metadata_only=True)
    assert (root/channel1.POINTER).read_bytes()==previous
    assert residue.read_bytes()==b'not owned by metadata transaction'
    assert not (root/'releases/channel-builds'/f'{131:032x}').exists()


@pytest.mark.parametrize('status,digest,refused',[(200,'b'*64,False),(301,'b'*64,True),(200,'c'*64,True)])
def test_oss_publication_head_never_reads_body_or_follows_redirect(tmp_path,monkeypatch,status,digest,refused):
    import http.client
    requests=[]
    class Reply:
        def __init__(self):self.status=status
        def getheader(self,name,default=None):
            return {'Content-Length':'17','x-oss-meta-sha256':digest,'Content-Type':'application/msix'}.get(name,default)
        def read(self,*args):raise AssertionError('HEAD guard must never read a package body')
        def close(self):pass
    class Connection:
        def request(self,method,path,headers):requests.append((method,path))
        def getresponse(self):return Reply()
        def close(self):pass
    monkeypatch.setattr(http.client,'HTTPSConnection',lambda *args,**kw:Connection())
    row={'artifact':{'key':'releases/channel-builds/'+'a'*32+'/SemantierAgent.msix','size':17,'sha256':'b'*64},
         'native':{'schema':2,'packageBase':'https://assets.semantier.cn'}}
    if refused:
        with pytest.raises(ValueError):channel1._verify_oss_head(row)
    else:channel1._verify_oss_head(row)
    assert requests==[('HEAD','/'+row['artifact']['key'])]


@pytest.mark.parametrize('public_ready',[True,False])
def test_retirement_after_oss_cutover_removes_old_ecs_stable_without_touching_active_head(tmp_path,monkeypatch,public_ready):
    root,_,_=retirement_fixture(tmp_path)
    previous=channel1.sha(root/channel1.POINTER)
    old=root/protocol.build_prefix(f'{122:032x}')/'SemantierAgent.msix'
    old_sha=channel1.sha(old)
    (tmp_path/'new/SemantierAgent.msix').unlink()
    stage=tmp_path/'oss';request,new_sha,metadata_sha=package(stage,123)
    recovery=tmp_path/('.channel1-recovery-'+request['buildId'])
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    channel1.publish(root,stage,metadata_sha256=metadata_sha,package_sha256=new_sha,protocol=protocol,
        expected_previous_sha256=previous,recovery=recovery,metadata_only=True)
    current=channel1.sha(root/channel1.POINTER)
    assert not (root/protocol.build_prefix(request['buildId'])/'SemantierAgent.msix').exists()
    if not public_ready:
        def unavailable(row):raise ValueError('Active OSS bytes unavailable')
        monkeypatch.setattr(channel1,'_verify_oss_head',unavailable)
    args=dict(build_id=f'{122:032x}',package_sha256=old_sha,expected_previous_sha256=current,assert_inactive=lambda paths:None)
    if public_ready:
        result=channel1.retire_package(root,**args,apply=True)
        assert result['result']=='PACKAGE_RETIRED'
        assert not old.exists() and not (root/'releases/SemantierAgent.msix').exists()
        assert (recovery/'recovery.retired').is_file()
        assert channel1.retire_package(root,**args,apply=True)['result']=='ALREADY_RETIRED'
        with pytest.raises(ValueError,match='Active release'):
            channel1.retire_package(root,build_id=request['buildId'],package_sha256=new_sha,expected_previous_sha256=current,apply=True)
    else:
        with pytest.raises(ValueError,match='unavailable'):channel1.retire_package(root,**args,apply=True)
        assert old.exists() and (root/'releases/SemantierAgent.msix').is_file()
        assert (recovery/'recovery.json').is_file()
    assert channel1.sha(root/channel1.POINTER)==current


def test_direct_oss_version_path_uses_real_prepared_metadata_without_private_role(tmp_path,monkeypatch):
    from agents.client_operations import oss_release
    stage=tmp_path/'stage';package(stage,132,version='1.2.132.4')
    roles=[];calls=[]
    def assume(kind):
        assert kind=='publish'
        roles.append(kind);return object()
    monkeypatch.setattr(oss_release,'assume_role',assume)
    monkeypatch.setattr(channel1.subprocess,'check_output',lambda *a,**k:'D:\\fixture.msix')
    monkeypatch.setattr(channel1.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0))
    class Public:
        def __init__(self,credentials):pass
        def put_once(self,key,path,digest):calls.append(('PUT',key))
        def head_identity(self,key,digest,size):calls.append(('HEAD',key))
    monkeypatch.setattr(oss_release,'ReleaseObjects',Public)
    result=channel1.stage_oss(stage/'SemantierAgent.msix',stage/'channel1-metadata.json',tmp_path/'direct.json',release_bytes=True)
    assert result['key']=='releases/channel-builds/1.2.132.4/SemantierAgent.msix'
    assert result['storagePolicy']=='DIRECT_PUBLISH' and roles==['publish']
    assert [method for method,key in calls]==['PUT','HEAD']


def test_post_success_oss_retention_keeps_active_and_deletes_only_bound_previous(tmp_path,monkeypatch):
    from agents.client_operations import oss_release
    from contextlib import contextmanager
    old=tmp_path/'old';package(old,131,version='1.2.131.4')
    current=tmp_path/'current';request,_,_=package(current,132,version='1.2.132.4')
    data=json.loads((current/'channel1-metadata.json').read_bytes());manifest=data['manifest'];active=manifest['packages'][0]['artifact']
    prior=json.loads((old/'channel1-metadata.json').read_bytes())['manifest']
    record=channel1._record(prior,None,protocol)
    proof={'status':'DOWNLOADED_VERIFIED_NOT_INSTALLED','sha256':active['sha256'],
           'manifestSha256':hashlib.sha256(protocol.canonical_json(manifest)).hexdigest(),'buildId':request['buildId']}
    receipt={'publicationVerification':'PASS','ossFullTransferCount':1,'ossSetupQualification':proof,
             'nativePreflight':{'remote':{'manifest':prior,'record':record}},
             'publication':json.dumps({'buildId':request['buildId'],'manifestSha256':proof['manifestSha256'],'channelSha256':'a'*64}),
             'plan':{'ssh':'root@123.57.184.194','remoteRoot':'/opt/semantier-client-distribution/channel1-storage-root/releases'}}
    (current/'result.json').write_text(json.dumps(receipt))
    obsolete=prior['packages'][0]['artifact'];objects={obsolete['key']:obsolete,active['key']:active}
    calls=[];locked=[];account=[oss_release.ACCOUNT];unknown=[]
    @contextmanager
    def guard(target,root,head):
        assert head['buildId']==request['buildId'];locked.append(True)
        try:yield types.SimpleNamespace(poll=lambda:None)
        finally:locked.pop()
    monkeypatch.setattr(channel1,'_oss_retention_lock',guard)
    monkeypatch.setattr(channel1,'_verify_oss_head',lambda row:None)
    def api(args,**kwargs):
        if args[1:]==['sts','GetCallerIdentity']:
            return types.SimpleNamespace(returncode=0,stdout=json.dumps({'AccountId':account[0]}),stderr='')
        assert locked,'Delete/readback outside publisher lock'
        if 'get-bucket-versioning' in args:
            return types.SimpleNamespace(returncode=0,stdout='<VersioningConfiguration/>',stderr='')
        if 'list-multipart-uploads' in args:
            return types.SimpleNamespace(returncode=0,stdout='<ListMultipartUploadsResult><Bucket>'+oss_release.BUCKET+'</Bucket><IsTruncated>false</IsTruncated></ListMultipartUploadsResult>',stderr='')
        key=args[-1]
        if 'head-object' in args:
            row=objects.get(key)
            if row is None:return types.SimpleNamespace(returncode=1,stdout='',stderr='NoSuchKey')
            return types.SimpleNamespace(returncode=0,stdout=json.dumps({'Header':{'X-Oss-Meta-Sha256':[row['sha256']],'Content-Length':[str(row['size'])]}}),stderr='')
        if 'delete-object' in args:
            calls.append(key);objects.pop(key)
            return types.SimpleNamespace(returncode=0,stdout='{}',stderr='')
        return types.SimpleNamespace(returncode=0,stdout=json.dumps({'complete':True,'items':[{'key':active['key']},{'key':'releases/installers/semantier-setup-ch1.exe'},{'key':'skills/v1/catalog.json'},*unknown]}),stderr='')
    monkeypatch.setattr(channel1.subprocess,'run',api)
    output=current/'oss-retention.json'
    account[0]='foreign'
    with pytest.raises(ValueError,match='account'):channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)
    account[0]=oss_release.ACCOUNT
    objects[obsolete['key']]={**obsolete,'sha256':'c'*64}
    with pytest.raises(ValueError,match='identity'):channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)
    assert not calls and active['key'] in objects
    objects[obsolete['key']]=obsolete
    value=channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)
    assert value['result']=='ONE_ACTIVE_MSIX_AND_SETUP' and calls==[obsolete['key']]
    assert active['key'] in objects
    receipt['publication']=['sftp> progress',receipt['publication']]
    (current/'result.json').write_text(json.dumps(receipt))
    assert channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)==value
    assert calls==[obsolete['key']]
    unknown.append({'key':'staging/windows-msix/unknown.msix'})
    with pytest.raises(ValueError,match='inventory'):channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)
    assert calls==[obsolete['key']]
    receipt['publicationVerification']='PENDING';(current/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='qualification'):channel1.retire_oss_previous(current/'SemantierAgent.msix',current/'channel1-metadata.json',output)
    assert calls==[obsolete['key']]


def test_successive_oss_releases_recover_without_any_ecs_msix(tmp_path, monkeypatch):
    root = tmp_path / 'public'; root.mkdir()
    monkeypatch.setattr(channel1, '_verify_oss_head', lambda row: None)
    _, old_digest, old_metadata = package(tmp_path / 'old', 132, version='1.2.132.4')
    channel1.publish(root, tmp_path / 'old', metadata_sha256=old_metadata,
                     package_sha256=old_digest, protocol=protocol, metadata_only=True)
    previous = (root / channel1.POINTER).read_bytes()
    _, digest, metadata = package(tmp_path / 'new', 133, version='2.3.133.5')
    (tmp_path / 'new/SemantierAgent.msix').unlink()
    recovery = root.parent / '.channel1-recovery-oss-only'
    channel1.publish(root, tmp_path / 'new', metadata_sha256=metadata,
                     package_sha256=digest, protocol=protocol, metadata_only=True,
                     expected_previous_sha256=hashlib.sha256(previous).hexdigest(), recovery=recovery)
    saved = json.loads((recovery / 'recovery.json').read_bytes())
    assert saved['schema'] == 2 and [row['key'] for row in saved['files']] == [channel1.POINTER]
    for _ in range(3):
        assert channel1.restore(root, recovery)['result'] == 'POINTER_RESTORED'
        assert (root / channel1.POINTER).read_bytes() == previous
    assert not list(root.rglob('*.msix')) and not list(recovery.rglob('*.msix'))


def test_legacy_rollback_replay_does_not_leave_a_same_inode_temporary_alias(tmp_path):
    root = tmp_path / 'public'; root.mkdir()
    publish(root, tmp_path / 'old', 130)
    previous = channel1.sha(root / channel1.POINTER)
    _, digest, metadata = package(tmp_path / 'new', 131)
    recovery = root.parent / '.channel1-recovery-legacy'
    channel1.publish(root, tmp_path / 'new', metadata_sha256=metadata,
                     package_sha256=digest, protocol=protocol,
                     expected_previous_sha256=previous, recovery=recovery)
    for _ in range(3):
        channel1.restore(root, recovery)
        assert not (root / 'releases/.channel1-restore.tmp').exists()
    temporary = root / 'releases/.channel1-restore.tmp'
    temporary.hardlink_to(recovery / '0')
    channel1.restore(root, recovery)
    assert not temporary.exists()
    temporary.write_bytes(b'unknown-owner')
    with pytest.raises(ValueError, match='Unreconciled rollback'):
        channel1.restore(root, recovery)
    assert temporary.read_bytes() == b'unknown-owner' and channel1.sha(root / channel1.POINTER) == previous


@pytest.mark.parametrize('valid', [True, False])
def test_retirement_validates_superseded_recovery_before_retiring_its_old_body(tmp_path, monkeypatch, valid):
    root = tmp_path / 'public'; root.mkdir()
    _, old_sha, _ = publish(root, tmp_path / 'old', 130)
    (tmp_path / 'old/SemantierAgent.msix').unlink()
    before = json.loads((root / channel1.POINTER).read_bytes())
    request, digest, _ = package(tmp_path / 'pending', 131)
    metadata = json.loads((tmp_path / 'pending/channel1-metadata.json').read_bytes())
    recovery = root.parent / ('.channel1-recovery-' + request['buildId'])
    channel1._save_recovery(root, recovery, channel1._record(metadata['manifest'], before, protocol), digest, protocol)
    prefix = root / protocol.build_prefix(request['buildId']); prefix.mkdir(parents=True)
    (prefix / 'build.json').write_bytes(protocol.canonical_json(metadata['manifest']))
    (prefix / 'native-graph.json').write_bytes(protocol.canonical_json(metadata['graph']))
    monkeypatch.setattr(channel1, '_verify_oss_head', lambda row: None)
    _, current_sha, current_metadata = package(tmp_path / 'current', 132)
    channel1.publish(root, tmp_path / 'current', metadata_sha256=current_metadata,
                     package_sha256=current_sha, protocol=protocol, metadata_only=True,
                     expected_previous_sha256=channel1.sha(root / channel1.POINTER),
                     recovery=root.parent / ('.channel1-recovery-' + f'{132:032x}'))
    pointer = channel1.sha(root / channel1.POINTER)
    if not valid:
        saved = json.loads((recovery / 'recovery.json').read_bytes())
        saved['afterPointerSha256'] = 'f' * 64
        (recovery / 'recovery.json').write_bytes(protocol.canonical_json(saved))
    arguments = dict(build_id=f'{130:032x}', package_sha256=old_sha,
                     expected_previous_sha256=pointer, apply=True, assert_inactive=lambda paths: None)
    if valid:
        assert channel1.retire_package(root, **arguments)['result'] == 'PACKAGE_RETIRED'
        assert (recovery / 'recovery.retired').is_file()
    else:
        with pytest.raises(ValueError, match='Active or ambiguous recovery'):
            channel1.retire_package(root, **arguments)
        assert (root / 'releases/SemantierAgent.msix').exists()
        assert (recovery / 'recovery.json').exists()
    assert channel1.sha(root / channel1.POINTER) == pointer
