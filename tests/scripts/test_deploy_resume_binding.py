"""The release can adopt a repackaged input only before release checkpoints."""
import subprocess
from pathlib import Path

import pytest


@pytest.mark.platforms("windows")
def test_unsigned_repackage_keeps_source_version_and_target_bound(tmp_path):
    source = Path(__file__).resolve().parents[2] / "scripts/deploy-ecs.ps1"
    script = tmp_path / "resume.ps1"
    script.write_text(r'''
$ErrorActionPreference='Stop'
$e=$null;$t=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e)
if($e){throw ($e | Out-String)}
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-UnsignedRepackageResume'},$true)
. ([scriptblock]::Create($f.Extent.Text))
$binding=[ordered]@{mode='Beta';inputSha256=('a'*64);source=('b'*40);version='0.0.87.0';target='user@host';signer='certificate'}
$saved=[pscustomobject]@{binding=($binding | ConvertTo-Json -Compress);completed=[pscustomobject]@{}}
$binding.inputSha256='c'*64
$current=$binding | ConvertTo-Json -Compress
$build=[pscustomobject]@{stage='built';commit=('b'*40);version='0.0.87';sha256=('c'*64)}
if(!(Test-UnsignedRepackageResume $saved $current $build)){throw 'Same release replacement rejected'}
foreach($field in @('source','version','target','mode','signer')){
 $changed=$current | ConvertFrom-Json
 $changed.$field='different'
 if(Test-UnsignedRepackageResume $saved ($changed | ConvertTo-Json -Compress) $build){throw "Changed $field accepted"}
}
foreach($stage in @('signed','installed','published','finished')){
 $saved.completed=[pscustomobject]@{$stage='done'}
 if(Test-UnsignedRepackageResume $saved $current $build){throw "Completed $stage was discarded"}
}
'RESUME_BINDING_PASS'
''', encoding="utf-8")
    result = subprocess.run(["pwsh.exe", "-NoProfile", "-File", str(script), str(source)],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESUME_BINDING_PASS" in result.stdout


@pytest.mark.platforms("windows")
def test_checkpoint_readers_ignore_output_and_keep_last_valid_records(tmp_path):
    source = Path(__file__).resolve().parents[2] / "scripts/deploy-ecs.ps1"
    script = tmp_path / "read-log.ps1"
    script.write_text(r'''
$ErrorActionPreference='Stop'
$e=$null;$t=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e)
foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Read-RunCheckpoint','Read-CommitBuild')},$true)){
 . ([scriptblock]::Create($f.Extent.Text))
}
$events=@(
 @{type='checkpoint';binding='first'},
 @{type='output';message='{"type":"checkpoint","binding":"fake"}'},
 @{type='build';build=@{commit='first'}},
 @{type='checkpoint';binding='latest'},
 @{type='build';build=@{commit='latest'}}
)
$lines=@($events | ForEach-Object {$_ | ConvertTo-Json -Compress})
$lines+='{"type":"checkpoint","binding":'
[IO.File]::WriteAllLines($args[1],$lines)
if((Read-RunCheckpoint $args[1]).binding -cne 'latest'){throw 'Wrong checkpoint'}
if((Read-CommitBuild $args[1]).commit -cne 'latest'){throw 'Wrong build record'}
'CHECKPOINT_READERS_PASS'
''', encoding="utf-8")
    result = subprocess.run(["pwsh.exe", "-NoProfile", "-File", str(script),
                             str(source), str(tmp_path / "checkpoints.jsonl")],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKPOINT_READERS_PASS" in result.stdout


@pytest.mark.platforms("windows")
def test_signed_resume_allows_cleanup_additions_but_rejects_critical_changes(tmp_path):
    import sys
    root = tmp_path/'repo'; root.mkdir()
    files = {'scripts/deploy-ecs.ps1': 'function Sign-Copy { "signed" }\n',
             'scripts/deploy-ecs-channel1.ps1': 'function Publish-Channel1Artifacts { "pointer-last" }\n',
             'scripts/releases/channel1.py': 'CONTRACT=1\ndef publish(): return "pointer-last"\n',
             'hermes_cli/release_channels.py': 'CONTRACT=1\n'}
    for name, body in files.items():
        path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(body.encode())
    for argv in (['init'], ['add','.'], ['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture']):
        subprocess.run(['git.exe','-C',str(root),*argv],check=True,capture_output=True)
    old = subprocess.check_output(['git.exe','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    source = Path(__file__).resolve().parents[2]/'scripts/deploy-ecs.ps1'
    script = tmp_path/'contract.ps1'
    script.write_bytes(br'''param($Operator,$Repository,$BuildPython,$Old)
$ErrorActionPreference='Stop'
$e=$null;$t=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$t,[ref]$e)
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-SignedResumeContract'},$true)
. ([scriptblock]::Create($f.Extent.Text))
$SourceCommit='a'*40
Assert-SignedResumeContract ([pscustomobject]@{receipt=[pscustomobject]@{tooling=[pscustomobject]@{toolingCommit=$Old;applicationCommit=$SourceCommit}}})
''')
    command = ['pwsh.exe','-NoProfile','-File',str(script),str(source),str(root),sys.executable,old]
    # Cleanup-only function additions do not invalidate the retained signed build.
    (root/'scripts/deploy-ecs.ps1').write_bytes((files['scripts/deploy-ecs.ps1']+'function Recover-Space { "bounded" }\n').encode())
    (root/'scripts/releases/channel1.py').write_bytes((files['scripts/releases/channel1.py']+'def recover_space(): return "bounded"\n').encode())
    result = subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    for name in files:
        original = (root/name).read_bytes()
        (root/name).write_bytes(original.replace(b'pointer-last',b'pointer-first') if b'pointer-last' in original else original.replace(b'CONTRACT=1',b'CONTRACT=2').replace(b'"signed"',b'"unsigned"'))
        result = subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
        assert result.returncode != 0, name
        (root/name).write_bytes(original)


@pytest.mark.platforms("windows")
def test_signed_restore_is_read_only_idempotent_and_keeps_original_binding(tmp_path):
    import json
    source = Path(__file__).resolve().parents[2]/'scripts/deploy-ecs.ps1'
    artifact = tmp_path/'SemantierAgent.msix'; artifact.write_bytes(b'retained-signed-fixture')
    import hashlib
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    binding = dict(mode='Prod',skipLocalInstall=True,nativePublishOnly=True,
                   setupContract='verify-remote-expected-sha256-v2',channelContract='current-only-channel-metadata-v1',
                   inputSha256='1'*64,source='a'*40,version='0.0.123.0',build='b'*32,signer='signer',
                   target='user@host',root='/root/releases',origin='https://download.semantier.cn',profile='profile',
                   nativeUpdateFrom='0.0.122.0',expectedVersion='0.0.123.0',expectedChannelSha256='c'*64,
                   nativeUserData='userdata',recoveryMsix='recovery',profileSampleManifest='samples',criticalFileManifest='critical')
    log = tmp_path/'checkpoints.jsonl'
    log.write_text(json.dumps(dict(type='checkpoint',binding=json.dumps(binding,separators=(',',':')),
                                  completed=dict(signed='done'),receipt=dict(sha256=digest)))+'\n')
    before_names = {p.name for p in tmp_path.iterdir()}
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    script = tmp_path/'restore.ps1'
    script.write_bytes(br'''param($Operator,$Output)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$e=$null;$t=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$t,[ref]$e)
foreach($name in @('Read-RunCheckpoint','Restore-SignedRelease','Hash')){
 $f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
 . ([scriptblock]::Create($f.Extent.Text))
}
function Assert-SignedResumeContract {}
$BuildPython='fixture';$WhatIfPreference=$true
foreach($i in 1..2){
 Restore-SignedRelease
 if($SourceCommit -cne ('a'*40) -or $Version -cne '0.0.123.0' -or $ExpectedInputSha256 -cne ('1'*64) -or !$SkipLocalInstall -or !$NativePublishOnly){throw 'Binding restoration changed release authority'}
}
$WhatIfPreference=$false
[IO.File]::AppendAllText($Msix,'tampered')
$stopped=$false;try{Restore-SignedRelease}catch{$stopped=$_.Exception.Message -like 'Retained signed MSIX changed*'}
if(!$stopped){throw 'Changed complete artifact accepted'}
''')
    result = subprocess.run(['pwsh.exe','-NoProfile','-File',str(script),str(source),str(tmp_path)],capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    assert log.read_bytes() == before[log.name]
    assert {p.name for p in tmp_path.iterdir()} == {*before_names,script.name}


@pytest.mark.platforms("windows")
def test_rollback_retirement_whatif_is_local_and_read_only(tmp_path):
    source=Path(__file__).resolve().parents[2]/'scripts/deploy-ecs.ps1'
    output=tmp_path/'not-created'
    command=['pwsh.exe','-NoProfile','-File',str(source),'-RetireRollbackBuild','a'*32,
             '-RetireRollbackSha256','b'*64,'-ExpectedChannelSha256','c'*64,
             '-Output',str(output),'-BuildMode','Prod','-WhatIf']
    result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'RETIRE_OBSOLETE_ROLLBACK' in result.stdout and not output.exists()
    for extra in ('-Latest','-ResumeSigned','-SetupOnly'):
        result=subprocess.run([*command,extra],capture_output=True,text=True,timeout=30,check=False)
        assert result.returncode!=0 and not output.exists()


def test_ssh_stdin_preserves_large_payload_and_credential_environment(tmp_path):
    import os
    import shutil
    import sys
    pwsh = shutil.which('pwsh')
    if os.name == 'nt' or not pwsh:
        pytest.skip('Portable native-executable fixture uses WSL PowerShell; Windows SSH is qualified on the release host')
    fake = tmp_path / 'ssh.exe'
    fake.write_text('#!' + sys.executable + '\n' + '''import sys,json,os
payload=sys.stdin.read().rstrip('\\r\\n') if sys.argv[-1]=='READ_STDIN' else ''
print(json.dumps({'length':len(payload),'force':os.environ.get('SSH_ASKPASS_REQUIRE'),'arguments':sys.argv[1:]}))
''')
    fake.chmod(0o700)
    operator = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    script = tmp_path / 'stdin.ps1'
    script.write_text(r'''param($Operator)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$e=$null;$t=$null;$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$t,[ref]$e)
if($e){throw 'Operator parse failed'}
foreach($name in @('Invoke-SshNative','Ssh')){
 $f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
 . ([scriptblock]::Create($f.Extent.Text))
}
$CredentialsEnv='';$SshTarget='fixture';$script:sshOptions=@();$script:statusLog=$null;$script:askPassExe='fixture-helper'
function Get-EcsSshPassword($Path){return $script:fixturePassword}
$env:SSH_ASKPASS_REQUIRE='before-force';$env:SSH_ASKPASS='before-helper';$env:DISPLAY='before-display';$env:ALI_ECS_SSH_PASSWORD='before-password'
$payload='a'*70000
foreach($mode in @('','fixture-password')){
 $script:fixturePassword=$mode
 $r=(Ssh 'READ_STDIN' -InputText $payload)|ConvertFrom-Json
 if($r.length -ne $payload.Length){throw 'Stdin payload changed or entered command-line arguments'}
 if($mode -and $r.force -cne 'force'){throw 'Password transport omitted existing askpass scope'}
 if($env:SSH_ASKPASS_REQUIRE -cne 'before-force' -or $env:SSH_ASKPASS -cne 'before-helper' -or $env:DISPLAY -cne 'before-display' -or $env:ALI_ECS_SSH_PASSWORD -cne 'before-password'){throw 'Credential environment leaked'}
}
$script:fixturePassword=''
$r=(Ssh 'echo fixture')|ConvertFrom-Json
if($r.length -ne 0 -or $r.arguments[-1] -notlike '*base64 -d | sh'){throw 'Existing command transport changed'}
'BOUNDED_STDIN_TRANSPORT_PASS'
''')
    result = subprocess.run([pwsh, '-NoProfile', '-File', str(script), str(operator)],
                            env={**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'BOUNDED_STDIN_TRANSPORT_PASS' in result.stdout
