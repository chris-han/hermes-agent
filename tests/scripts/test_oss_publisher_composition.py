"""Existing PowerShell publisher ordering with only OS/cloud boundaries injected."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_oss_publisher_stages_metadata_and_qualifies_setup_before_pointer(tmp_path):
    pwsh=shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell required')
    fake=tmp_path/'source'
    (fake/'scripts/releases').mkdir(parents=True)
    (fake/'apps/desktop/scripts').mkdir(parents=True)
    publisher=fake/'scripts/releases/channel1.py';publisher.write_text('# external fixture')
    (fake/'apps/desktop/scripts/native-package-digest.ps1').write_text("function Get-NativePackageDigest($p){return 'AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE='}")
    script=tmp_path/'check.ps1'
    script.write_text(r'''
param($Source,$Directory,$Publisher)
$ErrorActionPreference='Stop'
. $Source
$BuildPython='fixture-python';$SourceCommit='b'*40;$Version='0.0.131.0';$BuildId='c'*32
$NativeUpdateFrom='0.0.130.0';$NativePublishOnly=$true;$SkipLocalInstall=$true;$BuildMode='Prod';$OssSetupQualification=$true
$ExpectedChannelSha256='a'*64;$Output=$Directory;$receipt=[ordered]@{};$script:checkpoints=@{}
$script:nativePreflight=@{remote=@{caddySha256=('d'*64)}}
$script:events=[Collections.Generic.List[string]]::new()
function Hash($Path){return 'e'*64}
function Report-Status($Message){}
function Run($Exe,$Arguments){
 $script:events.Add('prepare')
 @{manifest=@{request=@{buildId=$BuildId;sequence=131};packages=@(@{artifact=@{sha256=('f'*64)}})}}|ConvertTo-Json -Depth 10|Set-Content $Metadata
}
function Upload($Path,$Destination){if($Destination.EndsWith('.msix')){throw 'MSIX body uploaded to ECS'};$script:events.Add('metadata-upload')}
function Save-Json($Value,$Path){}
function Invoke-OssByteOperation($Mode,$Artifact,$Metadata,$ResultPath){$script:events.Add('oss-byte-stage');return @{state='OSS_BYTES_STAGED_NOT_ACTIVE'}}
function Test-OssSetupQualification($Manifest,$Package){$script:events.Add('single-setup-qualification');return @{status='PASS'}}
function Test-NativePublic($Manifest,$Package){$script:events.Add('bounded-public-probe');return @{result='PASS'}}
function Ssh($Command){
 if($Command -notmatch '--metadata-only'){throw 'ECS publication omitted metadata-only mode'}
 if($Command -match 'stage-native'){$script:events.Add('metadata-stage');return '{}'}
 if($Command -match ' publish '){
  if(!$script:events.Contains('single-setup-qualification')){throw 'Head activated before Setup qualification'}
  $script:events.Add('pointer-publish');return '{}'
 }
 throw 'Unexpected SSH operation'
}
$Metadata=Join-Path $Directory 'channel1-metadata.json'
Publish-Channel1Artifacts (Join-Path $Directory 'signed.msix') '/opt/fixture/releases' '/opt/fixture/upload' ('f'*64) $Metadata $Publisher|Out-Null
if($script:events[-1] -cne 'pointer-publish' -or ($script:events|Where-Object{$_ -ceq 'single-setup-qualification'}).Count -ne 1){throw 'Wrong publication sequence'}
if($script:events.IndexOf('oss-byte-stage') -gt $script:events.IndexOf('metadata-stage')){throw 'Metadata exposed before OSS bytes staged'}
'PASS: OSS bytes -> metadata-only stage -> bounded probes -> single Setup qualification -> pointer'
''')
    result=subprocess.run([pwsh,'-NoProfile','-NonInteractive','-File',str(script),
                           str(ROOT/'scripts/deploy-ecs-channel1.ps1'),str(tmp_path),str(publisher)],
                          capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr
    assert result.stdout.startswith('PASS: OSS bytes')


def test_one_setup_attempt_is_checkpointed_and_never_replayed(tmp_path):
    import hashlib
    import json
    import sys
    pwsh=shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell required')
    package=tmp_path/'original.msix';package.write_bytes(b'owned signed-package fixture; no installation')
    sha=hashlib.sha256(package.read_bytes()).hexdigest()
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'manifest':{'request':{'buildId':'c'*32,'sequence':131,'windowsVersion':'0.0.131.0','commit':'b'*40},
        'packages':[{'artifact':{'sha256':sha,'size':package.stat().st_size}}]}}))
    script=tmp_path/'qualify.ps1'
    script.write_text(r'''
param($Source,$Directory,$Metadata,$Package,$Python)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
. $Source
$BuildPython=$Python;$Output=$Directory;$SetupExe='injected-signed-setup.exe'
$OssSetupQualification=$true;$NativePublishOnly=$true;$SkipLocalInstall=$true;$BuildMode='Prod';$ExpectedChannelSha256='a'*64
$script:checkpoints=@{};$receipt=[ordered]@{};$script:downloads=0
function Require-Signature($Path){}
function Hash($Path){return (Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()}
function Complete-Checkpoint($Name){$script:checkpoints[$Name]=$true}
function Run-Setup($Exe,$Arguments){
 if(!$script:checkpoints.ContainsKey('oss-setup-transfer-started')){throw 'Network invoked before budget checkpoint'}
 $script:downloads++
 $folder=$Arguments[-1];New-Item -ItemType Directory $folder|Out-Null
 Copy-Item $Package (Join-Path $folder 'SemantierAgent.msix')
 @{status='DOWNLOADED_VERIFIED_NOT_INSTALLED';selection='PINNED_STAGED_CANDIDATE';
   buildId=$Manifest.request.buildId;version=$Manifest.request.windowsVersion;sha256=$Manifest.packages[0].artifact.sha256;
   manifestSha256=(Get-OssManifestHash $Metadata);channelSha256=$ExpectedChannelSha256;origin='https://assets.semantier.cn';
   ossResponseBytes=$Manifest.packages[0].artifact.size;signature='PINNED_SIGNER_VALID';sourceCommit=$Manifest.request.commit;
   uri=('https://assets.semantier.cn/releases/channel-builds/'+$Manifest.request.buildId+'/SemantierAgent.msix')
 }|ConvertTo-Json|Set-Content (Join-Path $folder 'download-receipt.json')
}
$Manifest=(Get-Content $Metadata -Raw|ConvertFrom-Json).manifest
$first=Test-OssSetupQualification $Manifest $Package $Metadata
$second=Test-OssSetupQualification $Manifest $Package $Metadata
if($script:downloads -ne 1 -or $receipt.ossFullTransferCount -ne 1){throw 'Qualified bytes redownloaded'}
$script:checkpoints.Remove('oss-setup-qualified')
$refused=$false
try{Test-OssSetupQualification $Manifest $Package $Metadata|Out-Null}catch{if($_.Exception.Message -match 'consumed'){$refused=$true}else{throw}}
if(!$refused -or $script:downloads -ne 1){throw 'Interrupted one-shot was replayed'}
'PASS: attempt checkpoint precedes network; successful rerun uses retained bytes; interrupted attempt refuses replay'
''')
    result=subprocess.run([pwsh,'-NoProfile','-NonInteractive','-File',str(script),str(ROOT/'scripts/deploy-ecs-channel1.ps1'),
                           str(tmp_path),str(metadata),str(package),sys.executable],capture_output=True,text=True,timeout=40,check=False)
    assert result.returncode==0,result.stderr
    assert result.stdout.startswith('PASS: attempt checkpoint')


def test_signed_resume_preserves_oss_owner_and_one_shot_contract(tmp_path):
    pwsh=shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell required')
    script=tmp_path/'resume.ps1'
    script.write_text(r'''
param($Source,$Directory)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$null,[ref]$null)
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Restore-SignedRelease'},$true)
. ([scriptblock]::Create($f.Extent.Text))
$BuildPython='fixture-python';$Output=$Directory
$binding=@{mode='Prod';skipLocalInstall=$true;nativePublishOnly=$true;setupContract='cloud-channel-oss-native-v3';channelContract='current-only-channel-metadata-v1';
 inputSha256=('a'*64);source=('b'*40);version='0.0.131.0';build=('c'*32);signer='fixture';target='fixture';root='fixture';origin='https://cloud.semantier.cn';profile='fixture';
 nativeUpdateFrom='0.0.125.0';expectedVersion='0.0.131.0';expectedChannelSha256=('d'*64);nativeUserData='fixture';recoveryMsix='fixture';profileSampleManifest='fixture';criticalFileManifest='fixture';
 ossRuntimeRoot='/home/fixture/selected';ossWslDistro='selected-distro';ossSetupQualification=$true}
function Read-RunCheckpoint($Path){return [pscustomobject]@{binding=($binding|ConvertTo-Json -Compress);completed=[pscustomobject]@{signed=$true};receipt=[pscustomobject]@{sha256=('e'*64)}}}
function Hash($Path){return 'e'*64}
function Assert-SignedResumeContract($Saved){}
Restore-SignedRelease
if($OssRuntimeRoot -cne '/home/fixture/selected' -or $OssWslDistro -cne 'selected-distro' -or !$OssSetupQualification){throw 'OSS resume owner changed'}
$binding.Remove('ossSetupQualification')
$refused=$false
try{Restore-SignedRelease}catch{$refused=$true}
if(!$refused){throw 'Historical signed run silently adopted the OSS contract'}
'PASS: bound OSS resume restored; historical contract refused'
''')
    result=subprocess.run([pwsh,'-NoProfile','-NonInteractive','-File',str(script),str(ROOT/'scripts/deploy-ecs.ps1'),str(tmp_path)],
                          capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr
    assert result.stdout.startswith('PASS: bound OSS resume')


def test_bridge_selects_existing_cli_without_interactive_shell_path(tmp_path,monkeypatch):
    import hashlib
    import sys
    from agents.client_operations import oss_release
    runtime=Path(oss_release.__file__).resolve().parents[3]
    home=tmp_path/'home';cli=home/'.local/bin/aliyun';cli.parent.mkdir(parents=True)
    cli.write_text('#!/bin/sh\nexit 99\n');cli.chmod(0o700)
    monkeypatch.setenv('HOME',str(home));monkeypatch.setenv('PATH','/usr/bin')
    package=tmp_path/'signed-setup.exe';package.write_bytes(b'bounded signed-Setup fixture')
    result=tmp_path/'result.json';calls=[]
    class Objects:
        def __init__(self,credentials):pass
        def put_once(self,key,path,sha,**kwargs):calls.append(('put',key))
        def head_identity(self,key,sha,size):calls.append(('head',key))
        def verify(self,key,sha,size):calls.append(('small-get',key))
    monkeypatch.setattr(oss_release,'assume_role',lambda role:None)
    monkeypatch.setattr(oss_release,'ReleaseObjects',Objects)
    files=['src/agents/client_operations/oss_release.py','hermes-agent/scripts/releases/channel1.py','hermes-agent/hermes_cli/release_channels.py']
    hashes=[hashlib.sha256((runtime/f).read_text(encoding='utf-8-sig').encode()).hexdigest() for f in files]
    monkeypatch.setattr(sys,'argv',['bridge',str(runtime),'setup',str(package),'-',str(result),*hashes])
    source=(ROOT/'scripts/deploy-ecs-channel1.ps1').read_text()
    code=source.split('function Invoke-OssByteOperation(',1)[1].split("$code=@'\n",1)[1].split("\n'@",1)[0]
    previous=sys.path[:]
    try:exec(code,{})
    finally:sys.path[:]=previous
    assert shutil.which('aliyun')==str(cli)
    assert [c[0] for c in calls]==['put','head','small-get']
    assert result.is_file()


def test_oss_publish_only_does_not_require_profile_mutation_prerequisites(tmp_path):
    pwsh=shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell required')
    script=tmp_path/'options.ps1'
    script.write_text(r'''
param($Source)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$null,[ref]$null)
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-NativeReleaseOptions'},$true)
. ([scriptblock]::Create($f.Extent.Text))
$NativeUpdateFrom='0.0.130.0';$NativePublishOnly=$true;$BuildMode='Prod';$SkipLocalInstall=$true;$OssSetupQualification=$true
$Latest=$false;$BuildOnly=$false;$LocalOnly=$false;$DesktopOnly=$false;$SetupOnly=$false;$LocalSetup=$false;$UpdateSetup=$false;$PreflightOnly=$true
$Commit='b'*40;$SourceCommit='b'*40;$ExpectedChannelSha256='a'*64;$ExpectedVersion='0.0.131.0';$RecoveryMsix='pinned-signed-baseline.msix'
$NativeUserData='';$ProfileSampleManifest='';$CriticalFileManifest='';$ChannelRecoveryMsix=''
Assert-NativeReleaseOptions
# ENTRY_GUARD
$PreflightOnly=$false;Assert-NativeReleaseOptions
foreach($negative in @('install','no-recovery','legacy')){
 $SkipLocalInstall=$negative -cne 'install';$RecoveryMsix=if($negative -ceq 'no-recovery'){''}else{'pinned-signed-baseline.msix'};$OssSetupQualification=$negative -cne 'legacy'
 $refused=$false;try{Assert-NativeReleaseOptions}catch{$refused=$true}
 if(!$refused){throw ('Unsafe release options accepted: '+$negative)}
}
'PASS: read-only OSS preflight and no-install publication accepted; installation, missing recovery and legacy profile bypass refused'
''')
    entry=(ROOT/'scripts/deploy-ecs.ps1').read_text().split('if($NativeUpdateFrom){Assert-NativeReleaseOptions}',1)[1].split('Initialize-ReleasePaths',1)[0]
    script.write_text(script.read_text().replace('# ENTRY_GUARD',entry))
    result=subprocess.run([pwsh,'-NoProfile','-NonInteractive','-File',str(script),str(ROOT/'scripts/deploy-ecs.ps1')],capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr
    assert result.stdout.startswith('PASS: read-only OSS preflight')
