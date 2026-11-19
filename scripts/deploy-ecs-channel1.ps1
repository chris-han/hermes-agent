# Channel 1 metadata publication is composed by deploy-ecs.ps1. No standalone execution.
function Get-PackageSignatureHash([string]$Package) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip=[IO.Compression.ZipFile]::OpenRead($Package)
  try {
    $stream=$zip.GetEntry('AppxSignature.p7x').Open()
    try{return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant()}finally{$stream.Dispose()}
  } finally {$zip.Dispose()}
}
function Get-NativeProfileSamples {
  $samples=Get-Content -LiteralPath $ProfileSampleManifest -Raw | ConvertFrom-Json
  $critical=Get-Content -LiteralPath $CriticalFileManifest -Raw | ConvertFrom-Json
  if($samples.files.Count -ne 81 -or $critical.critical.Count -ne 7){throw 'Expected 81 profile samples and seven protected critical files.'}
  $files=@($samples.files | ForEach-Object {@{path=$_.path;sha256=(Hash $_.path)}})
  foreach($file in $critical.critical){
    $owner=@($critical.roots | Where-Object name -CEQ $file.root)
    if($owner.Count -ne 1){throw 'Ambiguous critical profile owner'}
    $path=Join-Path $owner[0].source $file.relative
    $backup=Join-Path $owner[0].backup $file.relative
    if((Hash $backup) -cne $file.sha256.ToLowerInvariant()){throw 'Protected recovery backup changed'}
    $files+=@{path=$path;sha256=(Hash $path)}
  }
  return $files
}
function Test-NativeGateway($Package,[string]$ProfileHome) {
  $state=Get-Content -LiteralPath (Join-Path $ProfileHome 'gateway.pid') -Raw | ConvertFrom-Json
  $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
  $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  if(!$process -or !$process.ExecutablePath.StartsWith($Package.InstallLocation+'\',[StringComparison]::OrdinalIgnoreCase) -or
    (Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid).Sid -cne $sid -or
    $process.SessionId -ne (Get-Process -Id $PID).SessionId -or
    [IO.Path]::GetFullPath($state.hermes_home) -ine [IO.Path]::GetFullPath($ProfileHome) -or
    $state.argv[0] -ine (Join-Path $Package.InstallLocation 'app/resources/agent-payload/hermes-agent/hermes_cli/main.py') -or
    $process.CommandLine -notmatch '\s-m hermes_cli\.main --profile default gateway run --quiet$'){
    throw 'Unknown or wrong-version Gateway owner; native release stopped.'
  }
  return @{pid=$process.ProcessId;birth=$process.CreationDate.ToUniversalTime().ToString('o');home=$ProfileHome;executable=$process.ExecutablePath;sid=$sid;session=$process.SessionId}
}
function Test-NativeChannelRecovery($Remote,[string]$Path) {
  if($Path -ine $RecoveryMsix){Require-Signature $Path}
  if((Hash $Path) -cne $Remote.packageSha256){throw 'Signed channel recovery and ECS baseline package mismatch'}
  $archive=[IO.Compression.ZipFile]::OpenRead($Path)
  try{[xml]$manifest=Read-ZipText $archive 'AppxManifest.xml'}finally{$archive.Dispose()}
  $native=$manifest.Package.Identity
  $version=$Remote.manifest.request.windowsVersion
  if($native.Name -cne $identity -or $native.Publisher -cne $publisher -or $native.Version -cne $version -or $native.ProcessorArchitecture -cne 'x64'){
    throw 'Signed channel recovery identity/publisher/version mismatch'
  }
  if([Version]$ExpectedVersion -le [Version]$version){throw 'Native target must advance both installed and published baselines'}
  return @{path=$Path;version=$version;sha256=$Remote.packageSha256}
}
function Test-NativeReleasePreflight {
  if($RemoteRoot -cnotmatch '^/opt/[A-Za-z0-9_/-]+/releases$' -or $RemoteRoot.Contains('/../') -or
    $SshTarget -cnotmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$' -or $PublicBase -cne 'https://cloud.semantier.cn') {throw 'Invalid native release destination'}
  # Refuse prompts: existing process credentials or noninteractive SSH must work.
  if(!(Get-EcsSshPassword $CredentialsEnv)){$script:sshOptions+=@('-o','BatchMode=yes')}
  $package=@(Get-AppxPackage -Name $identity)
  if($package.Count -ne 1 -or $package[0].Version.ToString() -cne $NativeUpdateFrom -or
    $package[0].Publisher -cne $publisher -or $package[0].Status.ToString() -cne 'Ok' -or
    $package[0].Architecture.ToString() -ine 'X64'){throw 'Installed baseline identity/version/status mismatch'}
  $certificate=Get-Item ('Cert:/CurrentUser/My/'+$Thumbprint)
  if(!$certificate.HasPrivateKey -or $certificate.Subject -cne $publisher -or $certificate.NotAfter.ToUniversalTime() -le [DateTime]::UtcNow){throw 'Native release signer unavailable'}
  Require-Signature $RecoveryMsix
  $signature=Get-PackageSignatureHash $RecoveryMsix
  if((Hash (Join-Path $package[0].InstallLocation 'AppxSignature.p7x')) -cne $signature){throw 'Recovery package does not match installed baseline signature'}
  if($OssSetupQualification -and $NativePublishOnly -and $SkipLocalInstall){
    # This path never starts a package deployment or touches the installed profile.
    $journalReadiness=@{result='NOT_RUN_PUBLISH_ONLY_NO_INSTALL'}
    $samples=@();$gateway=@{result='NOT_RUN_PUBLISH_ONLY_NO_INSTALL'};$window=$null;$profileManifests=$null
  } else {
    . (Join-Path $Repository 'apps/desktop/scripts/native-package-update.ps1')
    $journalReadiness=Get-NativeUpdateJournalReadiness -Directory (Join-Path $NativeUserData 'updates') -ExpectedPackageFullName $package[0].PackageFullName -ExpectedSignatureSha256 $signature
    if($journalReadiness.result -cne 'PASS'){throw ('Local native journal readiness STOP: '+$journalReadiness.reason)}
    $samples=Get-NativeProfileSamples
    $gateway=Test-NativeGateway $package[0] $UserProfile
    $window=Get-NativeWindow $package[0] -AllowStopped:$NativePublishOnly
    $profileManifests=@{samples=(Hash $ProfileSampleManifest);critical=(Hash $CriticalFileManifest)}
  }
  foreach($tool in @($SignTool,$BuildPython,$Node)){if(!(Test-Path -LiteralPath $tool -PathType Leaf)){throw 'Required native release tool unavailable'}}
  & $BuildPython -B -c 'import curl_cffi'
  if($LASTEXITCODE -ne 0){throw 'Existing curl_cffi environment unavailable'}
  $dirty=& git.exe -C $Repository status --porcelain
  if($LASTEXITCODE -ne 0 -or $dirty){throw 'Native release source must be committed and clean'}
  $applicationCommit=if($Commit){$Commit}else{$SourceCommit}
  if($script:SourceBinding.hermesGitlink -cne $script:SourceBinding.toolingHermesHead -or
    ($Commit -and $script:SourceBinding.hermesGitlink -cne $Commit)){throw 'Native source pin mismatch'}
  $saved=Read-CommitBuild (Join-Path $Output 'checkpoints.jsonl')
  $ledger=Get-Content -LiteralPath $VersionState -Raw | ConvertFrom-Json
  if(!$Commit){
    if(!$NativePublishOnly -or $Version -cne $ExpectedVersion -or ($ledger.version+'.0') -cne $ExpectedVersion -or
      $ledger.sourceCommit -cne $applicationCommit){throw 'Fixed native publication requires the exact reserved version and application source'}
  }
  elseif($saved){if(($saved.version+'.0') -cne $ExpectedVersion){throw 'Saved reservation differs from requested version'}}
  elseif(([Version]($ledger.version+'.0')).Build+1 -ne ([Version]$ExpectedVersion).Build){throw 'Next version reservation differs from explicit target'}
  if($SupersededBuildOutput -and !$PreflightOnly){$null=Remove-SupersededUnsignedInput $SupersededBuildOutput}
  if((Get-PSDrive ([IO.Path]::GetPathRoot($Output).Substring(0,1))).Free -lt 12GB){throw 'Insufficient native build space'}
  $code=@'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];identity=sys.argv[3]
p=root/'releases/channels/semantier-acceptance.json'
body=p.read_bytes();assert hashlib.sha256(body).hexdigest()==expected,'Concurrent channel pointer'
r=json.loads(body);h=r['nativeHead'];m=root/h['manifestKey'];b=m.read_bytes()
assert hashlib.sha256(b).hexdigest()==h['sha256'];manifest=json.loads(b)
assert manifest['request']['identity']['msixAppIdWithOrg']==identity and r['state']=='active'
row=manifest['packages'][0];a=root/row['artifact']['key']
with a.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['artifact']['sha256']
assert a.samefile(root/'releases/SemantierAgent.msix')
assert (root/row['feed']['key']).is_file()
assert (root/'releases/installers/semantier-setup-ch1.exe').is_file()
print(json.dumps({'pointerSha256':expected,'record':r,'manifest':manifest,'packageSha256':row['artifact']['sha256'],
 'caddySha256':hashlib.sha256(pathlib.Path('/opt/semantier-cloud-live-v1/Caddyfile').read_bytes()).hexdigest()}))
'@
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
  $remote=Ssh "printf '%s' '$encoded' | base64 -d | python3 - '$RemoteRoot' '$ExpectedChannelSha256' '$identity'"
  $remote=$remote | ConvertFrom-Json
  $channelRecovery=Test-NativeChannelRecovery $remote $(if($ChannelRecoveryMsix){$ChannelRecoveryMsix}else{$RecoveryMsix})
  $public=Test-NativePublic $remote.manifest $channelRecovery.path -Active -LegacyBaseline
  return @{result='PASS';stage='READ_ONLY_PREFLIGHT';producedAt=[DateTime]::UtcNow.ToString('o');source=$script:SourceBinding;
    journalReadiness=$journalReadiness;baseline=$NativeUpdateFrom;target=$ExpectedVersion;family=$package[0].PackageFamilyName;signature=$signature;
    recoveryMsix=$RecoveryMsix;recoverySha256=(Hash $RecoveryMsix);channelRecovery=$channelRecovery;remote=$remote;gateway=$gateway;files=$samples;
    visibleWindow=$(if($window){'PRESENT'}else{'NOT_OBSERVED_NO_UI_ACCEPTANCE'});
    public=$public;profileManifests=$profileManifests;llmCalls=0}
}
function Test-NativeCompletedRelease {
  $path=Join-Path $Output 'result.json'
  if(!(Test-Path -LiteralPath $path)){return $false}
  $saved=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
  if($NativePublishOnly -and $saved.result -cin @('OSS_PUBLISHED_SETUP_VERIFIED_NATIVE_NOT_RUN','OSS_PUBLISHED_RETENTION_PENDING')){
    $applicationCommit=if($Commit){$Commit}else{$SourceCommit}
    if($saved.plan.sourceCommit -cne $applicationCommit -or $saved.plan.version -cne $ExpectedVersion -or
      $saved.signedMsix -ine (Join-Path $Output 'SemantierAgent.msix') -or (Hash $saved.signedMsix) -cne $saved.sha256){throw 'Completed publication binding changed'}
    Require-Signature $saved.signedMsix
    $null=Ssh (Get-Channel1ProofCommand -Root $RemoteRoot -PackageSha256 $saved.sha256 -Build $saved.plan.buildId)
    $metadata=Get-Content -LiteralPath (Join-Path $Output 'channel1-metadata.json') -Raw | ConvertFrom-Json
    $cached=Join-Path $Output 'setup-download/SemantierAgent.msix'
    $proof=Get-Content -LiteralPath (Join-Path $Output 'setup-download/download-receipt.json') -Raw|ConvertFrom-Json
    if($saved.ossFullTransferCount -ne 1 -or $proof.status -cne 'DOWNLOADED_VERIFIED_NOT_INSTALLED' -or
       $proof.selection -cne 'PINNED_STAGED_CANDIDATE' -or $proof.signature -cne 'PINNED_SIGNER_VALID' -or
       $proof.sha256 -cne $saved.sha256 -or $proof.buildId -cne $saved.plan.buildId -or
       $proof.manifestSha256 -cne (Get-OssManifestHash (Join-Path $Output 'channel1-metadata.json')) -or
       (Hash $cached) -cne $saved.sha256){throw 'Retained one-shot Setup qualification changed; no redownload permitted'}
    $public=Test-NativePublic $metadata.manifest $saved.signedMsix -Active
    $saved | Add-Member -NotePropertyName publicationVerification -NotePropertyValue 'PASS' -Force
    $saved | Add-Member -NotePropertyName nativeAcceptance -NotePropertyValue 'NOT_RUN_OWNER_MANUAL' -Force
    $saved | Add-Member -NotePropertyName nativePublicAfterPointer -NotePropertyValue $public -Force
    $saved | Add-Member -NotePropertyName revalidatedAt -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
    Save-Json $saved $path
    if(!$SupersededBuildOutput -and $saved.plan.PSObject.Properties['supersededBuildOutput']){$SupersededBuildOutput=[string]$saved.plan.supersededBuildOutput}
    try {
      $retention=Invoke-OssByteOperation 'retention' $saved.signedMsix (Join-Path $Output 'channel1-metadata.json') (Join-Path $Output 'oss-retention.json')
      $saved | Add-Member -NotePropertyName ossRetention -NotePropertyValue $retention -Force
      $saved.result='OSS_PUBLISHED_SETUP_VERIFIED_NATIVE_NOT_RUN'
    } catch {
      $saved.result='OSS_PUBLISHED_RETENTION_PENDING'
      $saved | Add-Member -NotePropertyName retentionError -NotePropertyValue $_.Exception.Message -Force
      Save-Json $saved $path
      Write-Warning 'Verified active head preserved; OSS retention pending. No full transfer retry.'
      return $true
    }
    Save-Json $saved $path
    Report-Status 'PASS: ECS publication revalidated; native update remains manual. No allocation, signing, upload or installation.'
    return $true
  }
  if($saved.result -cne 'PASS'){return $false}
  if(!$saved.PSObject.Properties['nativeAcceptance'] -or $saved.nativeAcceptance.result -cne 'PASS' -or
    $saved.nativeAcceptance.from -cne $NativeUpdateFrom -or $saved.nativeAcceptance.to -cne $ExpectedVersion -or
    $saved.plan.sourceCommit -cne $Commit -or $saved.signedMsix -ine (Join-Path $Output 'SemantierAgent.msix') -or
    (Hash $saved.signedMsix) -cne $saved.sha256 -or $saved.nativeAcceptance.sha256 -cne $saved.sha256){throw 'Completed native release binding changed'}
  Require-Signature $saved.signedMsix
  $package=Get-AppxPackage -Name $identity
  if($package.Version.ToString() -cne $ExpectedVersion -or $package.Status.ToString() -cne 'Ok' -or
    (Hash (Join-Path $package.InstallLocation 'AppxSignature.p7x')) -cne (Get-PackageSignatureHash $saved.signedMsix)){throw 'Completed installed release changed'}
  foreach($file in $saved.nativePreflight.files){if((Hash $file.path) -cne $file.sha256){throw 'Completed release profile samples changed'}}
  $null=Test-NativeGateway $package $UserProfile
  $null=Ssh (Get-Channel1ProofCommand -Root $RemoteRoot -PackageSha256 $saved.sha256 -Build $saved.plan.buildId)
  Report-Status 'PASS: unchanged completed native release revalidated; no build, version allocation, signing, upload or update.'
  return $true
}
function Test-NativePublic {
  param($Manifest,[string]$Package,[switch]$Active,[switch]$LegacyBaseline)
  $code=@'
import base64,hashlib,json,sys
from curl_cffi import requests,CurlOpt
m=json.loads(base64.b64decode(sys.argv[1]));row=m['packages'][0];a=row['artifact'];rows=[]
canonical=(json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+chr(10)).encode()
legacy=sys.argv[4]=='True'
if legacy:
 assert m['request']['publicBase']=='https://download.semantier.cn' and row['native']['schema']==1
 package_base='https://download.semantier.cn'
else:
 assert m['request']['publicBase']=='https://cloud.semantier.cn' and row['native']['packageBase']=='https://assets.semantier.cn'
 package_base='https://assets.semantier.cn'
for mode,negotiated in [('v2',3),('v3only',30)]:
 with requests.Session(http_version=mode,trust_env=False,curl_options={CurlOpt.PROXY:b'',CurlOpt.MAXFILESIZE_LARGE:4194304}) as c:
  if sys.argv[3]=='True':
   r=c.get('https://cloud.semantier.cn/releases/channels/semantier-acceptance.json',allow_redirects=False,timeout=25)
   assert r.http_version==negotiated and r.status_code==200 and 'no-store' in r.headers.get('Cache-Control','')
   head=json.loads(r.content)['nativeHead'];assert head['buildId']==m['request']['buildId'] and head['sha256']==hashlib.sha256(canonical).hexdigest()
  for key in (row['feed']['key'],'releases/channel-builds/'+m['request']['buildId']+'/build.json'):
   r=c.get('https://cloud.semantier.cn/'+key,allow_redirects=False,timeout=25)
   assert r.http_version==negotiated and r.status_code==200 and 'application/json' in r.headers.get('Content-Type','')
   assert 'immutable' in r.headers.get('Cache-Control','')
   if key.endswith('build.json'):assert r.content==canonical
   else:assert hashlib.sha256(r.content).hexdigest()==row['native']['graph']['sha256']
   rows.append({'origin':'cloud-metadata','protocol':r.http_version,'status':r.status_code,'bytes':len(r.content)})
with requests.Session(http_version='v2' if legacy else 'v1',trust_env=False,curl_options={CurlOpt.PROXY:b'',CurlOpt.MAXFILESIZE_LARGE:65536}) as c:
 for start,end in ((0,65535),(a['size']-65536,a['size']-1)):
  r=c.get(package_base+'/'+a['key'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},allow_redirects=False,timeout=30)
  with open(sys.argv[2],'rb') as f:f.seek(start);expected=f.read(end-start+1)
  assert r.status_code==206 and len(r.content)==65536 and r.content==expected,(r.status_code,r.http_version)
  assert not legacy or r.http_version==3
  assert r.headers['Content-Range']==f'bytes {start}-{end}/{a["size"]}' and not r.headers.get('Content-Encoding')
  rows.append({'origin':'legacy-ecs-package' if legacy else 'oss-package','protocol':r.http_version,'status':r.status_code,'bytes':len(r.content),'range':r.headers['Content-Range']})
print(json.dumps({'result':'CLOUD_METADATA_LEGACY_BASELINE_BOUNDED_RANGE_PASS' if legacy else 'CLOUD_METADATA_OSS_BOUNDED_RANGE_PASS','packageBase':package_base,'fullMsixTransfers':0,'responses':rows}))
'@
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Manifest|ConvertTo-Json -Depth 30 -Compress)))
  $result=& $BuildPython -B -c $code $encoded $Package ([bool]$Active).ToString() ([bool]$LegacyBaseline).ToString()
  if($LASTEXITCODE -ne 0){throw 'Cloud metadata or bounded OSS Range check failed; no full-file fallback'}
  return ($result|ConvertFrom-Json)
}
function Get-NativeWindow($Package,[switch]$AllowStopped) {
  $windows=@(Get-Process | Where-Object {$_.Path -ieq (Join-Path $Package.InstallLocation 'app/Semantier Agent.exe') -and $_.MainWindowHandle -ne 0})
  if($windows.Count -eq 0 -and $AllowStopped){return $null}
  if($windows.Count -ne 1){throw 'Expected exactly one real installed application window'}
  return $windows[0]
}
function Test-OssSetupTransport([string]$Setup) {
  Require-Signature $Setup
  $size=(Get-Item -LiteralPath $Setup).Length
  if($size -le 0 -or $size -gt 2097152){throw 'Small Setup size limit exceeded'}
  $proof=Join-Path $Output 'setup-oss-transport.json'
  Run-Setup $Setup @('--probe-oss-transport',(Hash $Setup),[string]$size,$proof)
  $result=Get-Content -LiteralPath $proof -Raw|ConvertFrom-Json
  if($result.status -cne 'PASS' -or $result.scope -cne 'SMALL_EXE_TRANSPORT_ONLY_NO_INSTALL' -or $result.sha256 -cne (Hash $Setup) -or $result.bytes -lt $size){throw 'Native OSS small Setup transport failed'}
  return $result
}
function Get-OssSourceHash([string]$Path) {
  $text=[IO.File]::ReadAllText($Path).Replace("`r`n","`n")
  $sha=[Security.Cryptography.SHA256]::Create()
  try{return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text)))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
}
function Invoke-OssByteOperation([string]$Mode,[string]$Artifact,[string]$Metadata,[string]$ResultPath) {
  if($Mode -cnotin @('msix','setup','retention') -or !$script:RuntimeRoot -or
     $OssRuntimeRoot -cnotmatch '^/home/[A-Za-z0-9_./-]+$' -or $OssRuntimeRoot.Contains('/../') -or
     $OssWslDistro -cnotmatch '^[A-Za-z0-9_.-]+$'){throw 'Invalid pinned WSL OSS owner configuration'}
  $wsl=Join-Path $env:SystemRoot 'System32/wsl.exe'
  # --exec bypasses the default Linux shell, preserving Windows backslashes
  # and the Python program as literal argv rather than reparsing shell text.
  $artifactOutput=@(& $wsl -d $OssWslDistro --exec wslpath -u $Artifact)
  if($LASTEXITCODE -ne 0 -or $artifactOutput.Count -ne 1){throw 'Cannot resolve signed artifact in the selected WSL owner'}
  $linuxArtifact=([string]$artifactOutput[0]).Trim()
  $resultOutput=@(& $wsl -d $OssWslDistro --exec wslpath -u $ResultPath)
  if($LASTEXITCODE -ne 0 -or $resultOutput.Count -ne 1){throw 'Cannot resolve OSS result path'}
  $linuxResult=([string]$resultOutput[0]).Trim()
  $linuxMetadata='-'
  if($Metadata){
    $metadataOutput=@(& $wsl -d $OssWslDistro --exec wslpath -u $Metadata)
    if($LASTEXITCODE -ne 0 -or $metadataOutput.Count -ne 1){throw 'Cannot resolve pinned metadata path'}
    $linuxMetadata=([string]$metadataOutput[0]).Trim()
  }
  $linuxSuperseded='-'
  if($Mode -ceq 'retention' -and $SupersededBuildOutput){
    $converted=@(& $wsl -d $OssWslDistro --exec wslpath -u $SupersededBuildOutput)
    if($LASTEXITCODE -ne 0 -or $converted.Count -ne 1){throw 'Cannot bind superseded release for retention'}
    $linuxSuperseded=([string]$converted[0]).Trim()
  }
  $core=Join-Path $script:RuntimeRoot 'src/agents/client_operations/oss_release.py'
  $publisher=Join-Path $Repository 'scripts/releases/channel1.py'
  $protocol=Join-Path $Repository 'hermes_cli/release_channels.py'
  $code=@'
import hashlib,importlib.util,json,os,sys
from pathlib import Path
root,mode,artifact,metadata,output,core_sha,publisher_sha,protocol_sha,previous_setup_sha,superseded=sys.argv[1:]
root=Path(root)
cli=Path.home()/'.local/bin/aliyun'
if not cli.is_file() or not os.access(cli,os.X_OK):raise ValueError('Existing WSL account CLI unavailable')
os.environ['PATH']=str(cli.parent)+os.pathsep+os.environ.get('PATH','')
def normalized(path):return hashlib.sha256(path.read_text(encoding='utf-8-sig').encode()).hexdigest()
for path,wanted in [(root/'src/agents/client_operations/oss_release.py',core_sha),(root/'hermes-agent/scripts/releases/channel1.py',publisher_sha),(root/'hermes-agent/hermes_cli/release_channels.py',protocol_sha)]:
 if normalized(path)!=wanted:raise ValueError('WSL OSS owner source hash differs from the native pinned source')
sys.path[:0]=[str(root/'src'),str(root/'hermes-agent')]
from agents.client_operations import oss_release
if Path(oss_release.__file__).resolve()!=root/'src/agents/client_operations/oss_release.py':raise ValueError('Unexpected OSS module owner')
path=root/'hermes-agent/scripts/releases/channel1.py'
spec=importlib.util.spec_from_file_location('oss_channel1_owner',path);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
artifact=Path(artifact);output=Path(output)
if mode=='msix':result=owner.stage_oss(artifact,metadata,output,release_bytes=True)
elif mode=='retention':
 try:result=owner.retire_oss_previous(artifact,metadata,output,None if superseded=='-' else superseded)
 except ValueError as error:
  if str(error)=='Verified active Setup qualification required before retention':
   print(json.dumps({'state':'OSS_RETENTION_UNQUALIFIED'},separators=(',',':')));raise SystemExit(2)
  raise
else:
 if not artifact.is_file() or artifact.is_symlink() or not 0<artifact.stat().st_size<=2097152:raise ValueError('Small signed Setup required')
 with artifact.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 objects=oss_release.ReleaseObjects(oss_release.assume_role('publish'));key='releases/installers/semantier-setup-ch1.exe'
 objects.put_once(key,artifact,sha,content_type='application/octet-stream',expected_previous_sha256=None if previous_setup_sha=='-' else previous_setup_sha);objects.head_identity(key,sha,artifact.stat().st_size)
 objects.verify(key,sha,artifact.stat().st_size)
 result={'state':'OSS_SETUP_OBJECT_VERIFIED_NOT_SELECTED','key':key,'sha256':sha,'size':artifact.stat().st_size}
 encoded=(json.dumps(result,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if output.exists() and output.read_bytes()!=encoded:raise ValueError('Setup result collision')
 owner._atomic(output,encoded)
print(json.dumps(result,separators=(',',':')))
'@
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
  $entry="import base64;exec(base64.b64decode('$encoded'))"
  Report-Status ('Scoped WSL OSS operation: '+$Mode+'; no active pointer write')
  $previousSetup=if($Mode -ceq 'setup' -and $OssPreviousSetupSha256){$OssPreviousSetupSha256}else{'-'}
  $body=& $wsl -d $OssWslDistro --exec ("$OssRuntimeRoot/.venv/bin/python") -B -c $entry $OssRuntimeRoot $Mode $linuxArtifact $linuxMetadata $linuxResult (Get-OssSourceHash $core) (Get-OssSourceHash $publisher) (Get-OssSourceHash $protocol) $previousSetup $linuxSuperseded
  if($LASTEXITCODE -ne 0){
    if($Mode -ceq 'retention' -and ($body -join '') -ceq '{"state":"OSS_RETENTION_UNQUALIFIED"}'){throw 'OSS_RETENTION_UNQUALIFIED'}
    throw 'Scoped WSL OSS byte operation failed; no credential or authority fallback'
  }
  $result=($body -join "`n")|ConvertFrom-Json
  if($Mode -ceq 'retention'){
    if($result.result -cne 'ONE_ACTIVE_MSIX_AND_SETUP'){throw 'OSS retention is incomplete'}
  }elseif($result.state -cnotin @('OSS_BYTES_STAGED_NOT_ACTIVE','OSS_SETUP_OBJECT_VERIFIED_NOT_SELECTED')){throw 'Unexpected OSS byte operation result'}
  return $result
}
function Get-OssManifestHash([string]$Metadata) {
  $code="import hashlib,json,sys; m=json.load(open(sys.argv[1],encoding='utf-8-sig'))['manifest']; print(hashlib.sha256((json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+chr(10)).encode()).hexdigest())"
  $hash=& $BuildPython -B -c $code $Metadata
  if($LASTEXITCODE -ne 0 -or $hash -cnotmatch '^[a-f0-9]{64}$'){throw 'Cannot bind the immutable candidate manifest'}
  return [string]$hash
}
function Test-OssSetupQualification($Manifest,[string]$Package,[string]$Metadata) {
  if(!$OssSetupQualification -or !$NativePublishOnly -or !$SkipLocalInstall -or $BuildMode -cne 'Prod'){throw 'OSS activation requires explicit one-shot Setup qualification in Prod publish-only mode'}
  Require-Signature $SetupExe
  $manifestHash=Get-OssManifestHash $Metadata
  $folder=Join-Path $Output 'setup-download'
  $download=Join-Path $folder 'SemantierAgent.msix'
  $proof=Join-Path $folder 'download-receipt.json'
  if(!$script:checkpoints.ContainsKey('oss-setup-qualified')){
    if($script:checkpoints.ContainsKey('oss-setup-transfer-started')){throw 'The one Setup transfer attempt was consumed; preserve its state and do not automatically retry'}
    $receipt.ossFullTransferCount=1
    Complete-Checkpoint 'oss-setup-transfer-started'
    Run-Setup $SetupExe @('--download-candidate',[string]$Manifest.request.buildId,$manifestHash,[string]$Manifest.request.sequence,[string]$Manifest.packages[0].artifact.sha256,$folder)
    $receipt.ossSetupQualification=Get-Content -LiteralPath $proof -Raw|ConvertFrom-Json
  }
  if(!(Test-Path -LiteralPath $proof -PathType Leaf) -or !(Test-Path -LiteralPath $download -PathType Leaf)){throw 'Retained Setup download evidence is incomplete'}
  $result=Get-Content -LiteralPath $proof -Raw|ConvertFrom-Json
  if($result.status -cne 'DOWNLOADED_VERIFIED_NOT_INSTALLED' -or $result.selection -cne 'PINNED_STAGED_CANDIDATE' -or
     $result.buildId -cne $Manifest.request.buildId -or $result.version -cne $Manifest.request.windowsVersion -or
     $result.sha256 -cne $Manifest.packages[0].artifact.sha256 -or $result.manifestSha256 -cne $manifestHash -or
     $result.channelSha256 -cne $ExpectedChannelSha256 -or $result.origin -cne 'https://assets.semantier.cn' -or $result.signature -cne 'PINNED_SIGNER_VALID' -or
     $result.sourceCommit -cne $Manifest.request.commit -or
     $result.uri -cne ('https://assets.semantier.cn/releases/channel-builds/'+$Manifest.request.buildId+'/SemantierAgent.msix') -or
     $result.ossResponseBytes -lt $Manifest.packages[0].artifact.size -or (Hash $download) -cne $result.sha256 -or
     (Hash $Package) -cne $result.sha256){throw 'Setup full-download evidence does not match the pinned signed candidate'}
  $receipt.ossSetupQualification=$result
  Complete-Checkpoint 'oss-setup-qualified'
  return $result
}
function Publish-Channel1Artifacts([string]$Signed, [string]$Root, [string]$Stage, [string]$PackageSha256, [string]$Metadata, [string]$Publisher = (Join-Path $PSScriptRoot 'releases/channel1.py')) {
  if($Root -cnotmatch '^[A-Za-z0-9_:/.-]+$' -or $Stage -cnotmatch '^[A-Za-z0-9_:/.-]+$' -or $PackageSha256 -cnotmatch '^[a-f0-9]{64}$') {
    throw 'Invalid Channel 1 publication binding.'
  }
  $protocol=Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Publisher))) 'hermes_cli/release_channels.py'
  . (Join-Path (Split-Path -Parent $Publisher) '../../apps/desktop/scripts/native-package-digest.ps1')
  $nativeDigest=Get-NativePackageDigest $Signed
  Run $BuildPython @('-B',$publisher,'prepare','--package',$Signed,'--output',$Metadata,
    '--expected-sha256',$PackageSha256,'--source-commit',$SourceCommit,'--windows-version',$Version,'--build-id',$BuildId,'--native-digest',$nativeDigest)
  if(!$NativeUpdateFrom -or !$NativePublishOnly -or !$SkipLocalInstall -or $BuildMode -cne 'Prod' -or !$OssSetupQualification){throw 'OSS publication requires the pinned Prod publish-only one-shot flow'}
  $preparedMetadataSha=Hash $Metadata
  $receipt.ossStaging=Invoke-OssByteOperation 'msix' $Signed $Metadata (Join-Path $Output 'oss-staged.json')
  if((Hash $Metadata) -cne $preparedMetadataSha){throw 'Prepared release metadata changed during OSS staging'}
  Upload $Metadata "$Stage/channel1-metadata.json"
  Upload $publisher "$Stage/channel1.py"
  Upload $protocol "$Stage/channel1-protocol.py"
  $arguments="--metadata-only --root '$Root' --stage '$Stage' --metadata-sha256 '$preparedMetadataSha' --package-sha256 '$PackageSha256' --protocol '$Stage/channel1-protocol.py' --protocol-sha256 '$(Hash $protocol)'"
  if($NativeUpdateFrom){
    $recovery=$Root.Substring(0,$Root.LastIndexOf('/'))+'/.channel1-recovery-'+$BuildId
    $receipt.nativeRecovery=$recovery
    Save-Json $receipt (Join-Path $Output 'result.json')
    $guard=" --expected-previous-sha256 '$ExpectedChannelSha256' --recovery '$recovery'"
    $null=Ssh "python3 -B '$Stage/channel1.py' stage-native $arguments $guard --caddy-sha256 '$($script:nativePreflight.remote.caddySha256)'"
    $candidate=Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json
    $receipt.nativePublicBeforePointer=Test-NativePublic $candidate.manifest $Signed
    $receipt.ossSetupQualification=Test-OssSetupQualification $candidate.manifest $Signed $Metadata
    $receipt.result='BUILT_SIGNED_PACKAGE';Save-Json $receipt (Join-Path $Output 'result.json')
    $arguments+=$guard
  }
  return Ssh "python3 -B '$Stage/channel1.py' publish $arguments"
}
function Get-Channel1ProofCommand([string]$Root, [string]$PackageSha256, [string]$Build) {
  $command=@'
set -eu
python3 - '__ROOT__' '__HASH__' '__BUILD__' <<'CHANNEL1_PROOF'
import hashlib,json,pathlib,sys
root,wanted,build=sys.argv[1:];root=pathlib.Path(root)
record=json.loads((root/'releases/channels/semantier-acceptance.json').read_bytes())
head=record['nativeHead'];assert head['buildId']==build
body=(root/head['manifestKey']).read_bytes()
assert hashlib.sha256(body).hexdigest()==head['sha256']
manifest=json.loads(body);pkg=manifest['packages'][0]
assert pkg['artifact']['sha256']==wanted
assert manifest['request']['publicBase']=='https://cloud.semantier.cn'
assert pkg['native']['packageBase']=='https://assets.semantier.cn'
graph=(root/pkg['feed']['key']).read_bytes()
assert hashlib.sha256(graph).hexdigest()==pkg['native']['graph']['sha256']
from urllib.request import Request,urlopen
url='https://assets.semantier.cn/'+pkg['artifact']['key']
with urlopen(Request(url,method='HEAD'),timeout=30) as response:
 assert response.status==200 and response.url==url
 assert response.headers['Content-Length']==str(pkg['artifact']['size'])
 assert response.headers['x-oss-meta-sha256']==wanted and not response.headers.get('Content-Encoding')
print('CHANNEL1_CURRENT_CLOSURE_VERIFIED')
CHANNEL1_PROOF
'@
  return $command.Replace('__ROOT__',$Root).Replace('__HASH__',$PackageSha256).Replace('__BUILD__',$Build)
}
function Get-Channel1CleanupCommand([string]$Root, [string]$Build) {
  $command=@'
set -eu
root='__ROOT__'
current='__BUILD__'
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
old="$root/releases/channel-builds"
if test -d "$old" && ! find "$old" -name "*.appinstaller" -type f -print -quit | grep -q .; then
  find "$old" -mindepth 2 -maxdepth 5 -type f ! -path "$old/$current/*" \( -name '*.msix' -o -name '*.msixbundle' -o -name '*.zip' -o -name '*.msix.sha256' -o -name '*.msixbundle.sha256' -o -name '*.zip.sha256' \) -print -delete
fi
# Existing descriptor/catalog readers remain supported until separately qualified.
# Their retained closure also protects older packages from routine retirement.
printf 'OLD_PACKAGE_CLEANUP_PASS\n'
'@
  return $command.Replace('__ROOT__',$Root).Replace('__BUILD__',$Build)
}
