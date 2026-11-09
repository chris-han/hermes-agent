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
