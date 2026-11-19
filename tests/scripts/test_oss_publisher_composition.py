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
