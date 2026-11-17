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
function Test-NativeReleasePreflight {
  if($RemoteRoot -cnotmatch '^/opt/[A-Za-z0-9_/-]+/releases$' -or $RemoteRoot.Contains('/../') -or
    $SshTarget -cnotmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$' -or $PublicBase -cne 'https://download.semantier.cn') {throw 'Invalid native release destination'}
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
  foreach($tool in @($SignTool,$BuildPython,$Node)){if(!(Test-Path -LiteralPath $tool -PathType Leaf)){throw 'Required native release tool unavailable'}}
  & $BuildPython -B -c 'import curl_cffi'
  if($LASTEXITCODE -ne 0){throw 'Existing curl_cffi environment unavailable'}
  $samples=Get-NativeProfileSamples
  $gateway=Test-NativeGateway $package[0] $UserProfile
  $null=Get-NativeWindow $package[0]
  $dirty=& git.exe -C $Repository status --porcelain
  if($LASTEXITCODE -ne 0 -or $dirty){throw 'Native release source must be committed and clean'}
  if($script:SourceBinding.hermesGitlink -cne $Commit){throw 'Native source pin mismatch'}
  $saved=Read-CommitBuild (Join-Path $Output 'checkpoints.jsonl')
  $ledger=Get-Content -LiteralPath $VersionState -Raw | ConvertFrom-Json
  if($saved){if(($saved.version+'.0') -cne $ExpectedVersion){throw 'Saved reservation differs from requested version'}}
  elseif(([Version]($ledger.version+'.0')).Build+1 -ne ([Version]$ExpectedVersion).Build){throw 'Next version reservation differs from explicit target'}
  if((Get-PSDrive ([IO.Path]::GetPathRoot($Output).Substring(0,1))).Free -lt 12GB){throw 'Insufficient native build space'}
  $code=@'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];version=sys.argv[3]
p=root/'releases/channels/semantier-acceptance.json'
body=p.read_bytes();assert hashlib.sha256(body).hexdigest()==expected,'Concurrent channel pointer'
r=json.loads(body);h=r['nativeHead'];m=root/h['manifestKey'];b=m.read_bytes()
assert hashlib.sha256(b).hexdigest()==h['sha256'];manifest=json.loads(b)
assert manifest['request']['windowsVersion']==version and r['state']=='active'
row=manifest['packages'][0];a=root/row['artifact']['key']
with a.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['artifact']['sha256']
assert a.samefile(root/'releases/SemantierAgent.msix')
assert (root/row['feed']['key']).is_file()
assert (root/'releases/installers/semantier-setup-ch1.exe').is_file()
print(json.dumps({'pointerSha256':expected,'record':r,'manifest':manifest,'packageSha256':row['artifact']['sha256'],
 'caddySha256':hashlib.sha256(pathlib.Path('/opt/semantier-cloud-live-v1/Caddyfile').read_bytes()).hexdigest()}))
'@
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
  $remote=Ssh "printf '%s' '$encoded' | base64 -d | python3 - '$RemoteRoot' '$ExpectedChannelSha256' '$NativeUpdateFrom'"
  $remote=$remote | ConvertFrom-Json
  if((Hash $RecoveryMsix) -cne $remote.packageSha256){throw 'Local recovery and ECS baseline package mismatch'}
  $public=Test-NativePublic $remote.manifest $RecoveryMsix -Active
  return @{result='PASS';stage='READ_ONLY_PREFLIGHT';producedAt=[DateTime]::UtcNow.ToString('o');source=$script:SourceBinding;
    baseline=$NativeUpdateFrom;target=$ExpectedVersion;family=$package[0].PackageFamilyName;signature=$signature;
    recoveryMsix=$RecoveryMsix;recoverySha256=$remote.packageSha256;remote=$remote;gateway=$gateway;files=$samples;
    public=$public;profileManifests=@{samples=(Hash $ProfileSampleManifest);critical=(Hash $CriticalFileManifest)};llmCalls=0}
}
function Test-NativeCompletedRelease {
  $path=Join-Path $Output 'result.json'
  if(!(Test-Path -LiteralPath $path)){return $false}
  $saved=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
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
  param($Manifest,[string]$Package,[switch]$Active)
  $code=@'
import hashlib,json,sys
from curl_cffi import requests,CurlOpt
m=json.loads(sys.argv[1]);row=m['packages'][0];a=row['artifact'];rows=[]
for mode,negotiated in [('v2',3),('v3only',30)]:
 with requests.Session(http_version=mode,trust_env=False,curl_options={CurlOpt.PROXY:b''}) as c:
  if sys.argv[3]=='True':
   r=c.get('https://download.semantier.cn/releases/channels/semantier-acceptance.json',timeout=25)
   assert r.http_version==negotiated and r.status_code==200
   head=json.loads(r.content)['nativeHead']
   assert head['buildId']==m['request']['buildId'] and head['sha256']==hashlib.sha256((json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
  for key in (row['feed']['key'],'releases/channel-builds/'+m['request']['buildId']+'/build.json'):
   r=c.get('https://download.semantier.cn/'+key,timeout=25)
   assert r.http_version==negotiated and r.status_code==200,(key,r.status_code,r.http_version)
   if key.endswith('build.json'):assert r.content==(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()
   else:assert hashlib.sha256(r.content).hexdigest()==row['native']['graph']['sha256']
  for start,end in ((0,65535),(a['size']-65536,a['size']-1)):
   r=c.get('https://download.semantier.cn/'+a['key'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=30)
   with open(sys.argv[2],'rb') as f:f.seek(start);expected=f.read(end-start+1)
   assert r.http_version==negotiated and r.status_code==206 and r.content==expected,(r.status_code,r.http_version)
   assert r.headers['Content-Range']==f'bytes {start}-{end}/{a["size"]}'
   rows.append({'protocol':r.http_version,'status':r.status_code,'bytes':len(r.content),'range':r.headers['Content-Range']})
print(json.dumps({'result':'H2_H3_RANGE_PASS','responses':rows}))
'@
  $result=& $BuildPython -B -c $code ($Manifest | ConvertTo-Json -Depth 30 -Compress) $Package ([bool]$Active).ToString()
  if($LASTEXITCODE -ne 0){throw 'Native public H2/H3 Range or immutable metadata check failed'}
  return ($result | ConvertFrom-Json)
}
function Get-NativeWindow($Package) {
  $windows=@(Get-Process | Where-Object {$_.Path -ieq (Join-Path $Package.InstallLocation 'app/Semantier Agent.exe') -and $_.MainWindowHandle -ne 0})
  if($windows.Count -ne 1){throw 'Expected exactly one real installed application window'}
  return $windows[0]
}
function Get-NativeButton($Root,[string[]]$Names) {
  foreach($name in $Names){
    $condition=[Windows.Automation.AndCondition]::new(
      [Windows.Automation.PropertyCondition]::new([Windows.Automation.AutomationElement]::NameProperty,$name),
      [Windows.Automation.PropertyCondition]::new([Windows.Automation.AutomationElement]::ControlTypeProperty,[Windows.Automation.ControlType]::Button))
    $button=$Root.FindFirst([Windows.Automation.TreeScope]::Descendants,$condition)
    if($button){return $button}
  }
  return $null
}
function Invoke-NativeButton($Button) {
  if(!$Button -or !$Button.Current.IsEnabled){throw 'Required native UI control unavailable'}
  $Button.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke()
}
function Get-NativeModelButton($Window) {
  $root=[Windows.Automation.AutomationElement]::FromHandle($Window.MainWindowHandle)
  $buttons=$root.FindAll([Windows.Automation.TreeScope]::Descendants,
    [Windows.Automation.PropertyCondition]::new([Windows.Automation.AutomationElement]::ControlTypeProperty,[Windows.Automation.ControlType]::Button))
  $selected=@($buttons | Where-Object {$_.Current.Name -match '^(模型|Model) · '})
  if($selected.Count -ne 1){throw 'Current model selection UI unavailable'}
  return $selected[0]
}
function Open-NativeAbout($Window) {
  $root=[Windows.Automation.AutomationElement]::FromHandle($Window.MainWindowHandle)
  $settings=Get-NativeButton $root @('打开设置','Open settings')
  if($settings){Invoke-NativeButton $settings}
  $deadline=[DateTime]::UtcNow.AddSeconds(15)
  do {
    $about=Get-NativeButton $root @('关于','About')
    if($about){Invoke-NativeButton $about;return $root}
    Start-Sleep -Milliseconds 250
  } while([DateTime]::UtcNow -lt $deadline)
  throw 'Native About control unavailable'
}
function Invoke-NativeUpdateAcceptance([string]$Signed,[string]$PackageHash) {
  Add-Type -AssemblyName UIAutomationClient
  Add-Type -AssemblyName UIAutomationTypes
  $old=Get-AppxPackage -Name $identity
  if($old.Version.ToString() -cne $NativeUpdateFrom){throw 'Baseline changed before native apply'}
  $window=Get-NativeWindow $old
  $selectedModel=(Get-NativeModelButton $window).Current.Name
  $root=Open-NativeAbout $window
  $deadline=[DateTime]::UtcNow.AddSeconds(30)
  do {
    $check=Get-NativeButton $root @('立即检查','Check now')
    if($check){break};Start-Sleep -Milliseconds 250
  } while([DateTime]::UtcNow -lt $deadline)
  Invoke-NativeButton $check
  $deadline=[DateTime]::UtcNow.AddSeconds(60)
  do {
    $apply=Get-NativeButton $root @('立即更新','Update now')
    if($apply -and $apply.Current.IsEnabled){break};Start-Sleep -Milliseconds 500
  } while([DateTime]::UtcNow -lt $deadline)
  if(!$apply -or !$apply.Current.IsEnabled){throw 'Real in-app native update is not available'}
  $metadata=Get-Content -LiteralPath (Join-Path $Output 'channel1-metadata.json') -Raw | ConvertFrom-Json
  $manifestSha=Hash-CanonicalManifest $metadata.manifest
  # Offset and inode are captured immediately before the real UI apply, excluding probes.
  $boundary=Ssh "stat -c '%i %s' /opt/semantier-client-distribution/data/msix-h2h3-native-20261117/access.jsonl"
  if($boundary.Trim() -cnotmatch '^(\d+) (\d+)$'){throw 'Native transfer log boundary unavailable'}
  $receipt.nativeTransferBoundary=@{inode=$Matches[1];offset=$Matches[2];at=[DateTime]::UtcNow.ToString('o')}
  $receipt.nativeApply=@{parentPid=$window.Id;parentBirth=$window.StartTime.ToUniversalTime().ToString('o');manifestSha256=$manifestSha;sha256=$PackageHash;status='UI_APPLY_INTENT'}
  Save-Json $receipt (Join-Path $Output 'result.json')
  Invoke-NativeButton $apply
  $deadline=[DateTime]::UtcNow.AddMinutes(15)
  $completed=$null;$journal=$null
  do {
    $journals=@(Get-ChildItem -LiteralPath (Join-Path $NativeUserData 'updates') -Directory -Filter ('native-'+$manifestSha+'-*'))
    if($journals.Count -gt 1){throw 'Ambiguous native release journals'}
    if($journals.Count -eq 1){
      $journal=$journals[0].FullName
      if(Test-Path -LiteralPath (Join-Path $journal 'failed.json')){throw 'Native waiter reported failure; preserve journal'}
      if(Test-Path -LiteralPath (Join-Path $journal 'completed.json')){
        $completed=Get-Content -LiteralPath (Join-Path $journal 'completed.json') -Raw | ConvertFrom-Json;break
      }
    }
    Start-Sleep -Seconds 2
  } while([DateTime]::UtcNow -lt $deadline)
  if(!$completed -or $completed.status -cne 'TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED' -or $completed.manifestSha256 -cne $manifestSha){throw 'Native update/waiter completion timeout or mismatch'}
  $intent=Get-Content -LiteralPath (Join-Path $journal 'intent.json') -Raw | ConvertFrom-Json
  $deployment=Get-Content -LiteralPath (Join-Path $journal 'deployment.json') -Raw | ConvertFrom-Json
  if($intent.currentVersion -cne $NativeUpdateFrom -or $intent.version -cne $Version -or $intent.targetBuildId -cne $BuildId -or
    $intent.graph.objects[0].sha256 -cne $PackageHash -or $intent.graph.objects[0].nativeDigest -cne $metadata.graph.objects[0].nativeDigest -or
    $deployment.status -cne 'OS_SUCCESS'){throw 'Native ExpectedDigests/OS result binding mismatch'}
  $registered=Get-AppxPackage -Name $identity
  if($registered.Version.ToString() -cne $Version -or $registered.Status.ToString() -cne 'Ok' -or
    $registered.PackageFamilyName -cne $script:nativePreflight.family -or
    (Hash (Join-Path $registered.InstallLocation 'AppxSignature.p7x')) -cne (Get-PackageSignatureHash $Signed)){throw 'Installed signed target mismatch'}
  $successor=Get-Process -Id $completed.pid
  if($successor.Path -ine $completed.exe -or $successor.StartTime.ToUniversalTime() -ne [DateTime]::Parse($completed.birth).ToUniversalTime()){throw 'Native successor process identity mismatch'}
  $deadline=[DateTime]::UtcNow.AddMinutes(3)
  do {
    try{$gateway=Test-NativeGateway $registered $UserProfile;break}catch{$gateway=$null}
    Start-Sleep -Seconds 2
  } while([DateTime]::UtcNow -lt $deadline)
  if(!$gateway -or $gateway.pid -eq $script:nativePreflight.gateway.pid){throw 'Native successor Gateway ownership not established'}
  $stale=@(Get-Process | Where-Object {$_.Path -and $_.Path.StartsWith($old.InstallLocation+'\',[StringComparison]::OrdinalIgnoreCase)})
  if($stale.Count){throw 'Old package processes remain after native update'}
  $model=Get-NativeModelButton $successor
  if($model.Current.Name -cne $selectedModel){throw 'User model selection changed during native update'}
  Invoke-NativeButton $model
  $modelRoot=[Windows.Automation.AutomationElement]::FromHandle($successor.MainWindowHandle)
  $deadline=[DateTime]::UtcNow.AddSeconds(20)
  do {
    $cancel=Get-NativeButton $modelRoot @('取消','Cancel')
    if($cancel){break};Start-Sleep -Milliseconds 250
  } while([DateTime]::UtcNow -lt $deadline)
  Invoke-NativeButton $cancel
  $root=Open-NativeAbout $successor
  $check=Get-NativeButton $root @('立即检查','Check now');Invoke-NativeButton $check
  $deadline=[DateTime]::UtcNow.AddSeconds(60)
  do {
    $texts=@($root.FindAll([Windows.Automation.TreeScope]::Descendants,[Windows.Automation.Condition]::TrueCondition) | ForEach-Object {$_.Current.Name})
    $body=$texts -join "`n"
    if($body -match '你正在运行最新版本|running the latest|up to date'){break};Start-Sleep -Milliseconds 500
  } while([DateTime]::UtcNow -lt $deadline)
  if($body -notmatch [Regex]::Escape(([Version]$Version).ToString(3)) -or $body -notmatch '你正在运行最新版本|running the latest|up to date' -or
    $body -notmatch '通道 ?1|Channel 1'){throw 'Native successor version/channel/latest UI not established'}
  foreach($file in $script:nativePreflight.files){if((Hash $file.path) -cne $file.sha256){throw 'Protected profile sample changed'}}
  $pty=Invoke-InstalledPtyAcceptance $registered $PackageHash
  $transfer=Get-NativeTransfer $metadata.manifest
  $receipt.nativeAcceptance=@{result='PASS';from=$NativeUpdateFrom;to=$Version;sha256=$PackageHash;journal=$journal;completed=$completed;
    expectedDigests=$intent.graph.objects[0].nativeDigest;gateway=$gateway;profileSamples=81;criticalFiles=7;modelSelection=$selectedModel;modelPicker='OPENED_AND_CANCELLED_WITH_SELECTION_PRESERVED';ui='NATIVE_SUCCESSOR_LATEST';pty=$pty.ui.result;transfer=$transfer;llmCalls=0}
}
function Get-NativeTransfer($Manifest) {
  $code=@'
import json,os,pathlib,sys
path=pathlib.Path('/opt/semantier-client-distribution/data/msix-h2h3-native-20261117/access.jsonl')
inode,offset=int(sys.argv[1]),int(sys.argv[2]);key=sys.argv[3];size=int(sys.argv[4])
stat=path.stat();assert stat.st_ino==inode and stat.st_size>=offset,'Transfer log rotated or truncated'
rows=[]
with path.open('rb') as f:
 f.seek(offset)
 for line in f:
  r=json.loads(line);q=r.get('request',{})
  if q.get('uri')!='/'+key or q.get('method')!='GET':continue
  assert q['proto']=='HTTP/2.0' and r['status']==206,'Non-H2 Range native response'
  rows.append({'protocol':q['proto'],'status':r['status'],'bytes':r['size'],'range':r.get('content_range'),'requestRange':q.get('headers',{}).get('Range')})
assert rows and sum(r['bytes'] for r in rows)>0,'No real native package transfer observed'
total=sum(r['bytes'] for r in rows)
print(json.dumps({'result':'PASS','transferredBodyBytes':total,'packageBytes':size,'responses':len(rows),
 'savingsPercent':(1-total/size)*100,'scope':'All exact candidate URL GET bodies in bounded log window, including retries','rows':rows}))
'@
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
  $row=$Manifest.packages[0].artifact
  return ((Ssh "printf '%s' '$encoded' | base64 -d | python3 - '$($receipt.nativeTransferBoundary.inode)' '$($receipt.nativeTransferBoundary.offset)' '$($row.key)' '$($row.size)'") | ConvertFrom-Json)
}
function Hash-CanonicalManifest($Manifest) {
  $value=& $BuildPython -B -c 'import hashlib,json,sys; print(hashlib.sha256((json.dumps(json.loads(sys.argv[1]),sort_keys=True,separators=(",", ":"))+"\n").encode()).hexdigest())' ($Manifest | ConvertTo-Json -Depth 30 -Compress)
  if($LASTEXITCODE -ne 0 -or $value -cnotmatch '^[a-f0-9]{64}$'){throw 'Invalid manifest digest'}
  return $value
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
  Upload $Signed "$Stage/SemantierAgent.msix"
  Upload $Metadata "$Stage/channel1-metadata.json"
  Upload $publisher "$Stage/channel1.py"
  Upload $protocol "$Stage/channel1-protocol.py"
  $arguments="--root '$Root' --stage '$Stage' --metadata-sha256 '$(Hash $Metadata)' --package-sha256 '$PackageSha256' --protocol '$Stage/channel1-protocol.py' --protocol-sha256 '$(Hash $protocol)'"
  if($NativeUpdateFrom){
    $recovery=$Root.Substring(0,$Root.LastIndexOf('/'))+'/.channel1-recovery-'+$BuildId
    $receipt.nativeRecovery=$recovery
    Save-Json $receipt (Join-Path $Output 'result.json')
    $guard=" --expected-previous-sha256 '$ExpectedChannelSha256' --recovery '$recovery'"
    $null=Ssh "python3 -B '$Stage/channel1.py' stage-native $arguments $guard --caddy-sha256 '$($script:nativePreflight.remote.caddySha256)'"
    $candidate=Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json
    $receipt.nativePublicBeforePointer=Test-NativePublic $candidate.manifest $Signed
    $receipt.result='BUILT_SIGNED_122';Save-Json $receipt (Join-Path $Output 'result.json')
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
assert (root/'releases/installers/semantier-setup-ch1.exe').is_file()
with (root/'releases/SemantierAgent.msix').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==wanted
record=json.loads((root/'releases/channels/semantier-acceptance.json').read_bytes())
head=record['nativeHead'];assert head['buildId']==build
body=(root/head['manifestKey']).read_bytes()
assert hashlib.sha256(body).hexdigest()==head['sha256']
manifest=json.loads(body);pkg=manifest['packages'][0]
assert pkg['artifact']['sha256']==wanted
assert (root/pkg['artifact']['key']).samefile(root/'releases/SemantierAgent.msix')
assert (root/pkg['feed']['key']).is_file()
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
