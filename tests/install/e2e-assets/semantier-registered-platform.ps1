# Native boundary for semantier_registered_update.py. No authority from markers.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$r=[Console]::In.ReadToEnd() | ConvertFrom-Json
if ($env:OS -cne 'Windows_NT') { throw 'Windows required' }
$who=[Security.Principal.WindowsIdentity]::GetCurrent()
$sid=$who.User.Value
$session=(Get-Process -Id $PID).SessionId
if ($sid -cne $r.authorized_sid -or $sid -ceq 'S-1-5-21-1928370875-4287869425-2092521300-1003' -or $who.Name.Split('\')[-1] -ieq 'ChrisH' -or $session -le 0 -or -not [Environment]::UserInteractive) { throw 'Authorized disposable interactive user required; normal ChrisH refused' }
if ($r.expected_language -cnotin @('en','zh')) { throw 'Expected language must be en or zh' }
$cpus=@(Get-CimInstance Win32_Processor)
if(-not $cpus.Count -or @($cpus|Where-Object{$_.Architecture -ne 9}).Count){throw 'Actual native x64 required'}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class SemantierNativeFacts {
 [DllImport("wtsapi32.dll",SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr s,int id,int info,out IntPtr b,out int n);
 [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr p);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern SafeFileHandle CreateFile(string n,uint access,uint share,IntPtr sa,uint disposition,uint flags,IntPtr template);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern uint GetFinalPathNameByHandle(SafeFileHandle h,System.Text.StringBuilder p,uint n,uint flags);
 [StructLayout(LayoutKind.Sequential)] public struct Info { public uint attr; public System.Runtime.InteropServices.ComTypes.FILETIME created,accessed,written; public uint volume,high,low,links,indexHigh,indexLow; }
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
}
'@
$b=[IntPtr]::Zero;$n=0
if (-not [SemantierNativeFacts]::WTSQuerySessionInformation([IntPtr]::Zero,$session,8,[ref]$b,[ref]$n)) { throw 'WTS state unavailable' }
try { if($n -lt 4 -or [Runtime.InteropServices.Marshal]::ReadInt32($b) -ne 0){throw 'Session not active'} } finally {[SemantierNativeFacts]::WTSFreeMemory($b)}
function Physical([string]$path) {
 $item=Get-Item -LiteralPath $path -Force
 $cursor=$item
 while($cursor){if(($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'Reparse path refused'};if($cursor -is [IO.DirectoryInfo]){$cursor=$cursor.Parent}else{$cursor=$cursor.Directory}}
 $handle=[SemantierNativeFacts]::CreateFile($item.FullName,0,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
 if($handle.IsInvalid){throw 'Cannot hold physical path against replacement'}
 $final=[Text.StringBuilder]::new(32768)
 $length=[SemantierNativeFacts]::GetFinalPathNameByHandle($handle,$final,32768,0)
 $canonical=$final.ToString()
 if($canonical.StartsWith('\\?\UNC\')){$canonical='\\'+$canonical.Substring(8)}elseif($canonical.StartsWith('\\?\')){$canonical=$canonical.Substring(4)}
 if($length -eq 0 -or $length -ge 32768 -or $canonical.TrimEnd('\') -ine $item.FullName.TrimEnd('\')){$handle.Dispose();throw 'Opened physical path escaped inspected path'}
 $info=New-Object SemantierNativeFacts+Info
 if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info)){$handle.Dispose();throw 'Physical identity unavailable'}
 return @{handle=$handle;id=('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow);path=$item.FullName}
}
$profile=[Environment]::GetFolderPath('UserProfile').TrimEnd('\')
$root=[IO.Path]::GetFullPath($r.root).TrimEnd('\')
if(-not $root.StartsWith($profile+'\semantier-acceptance\',[StringComparison]::OrdinalIgnoreCase)){throw 'Root outside disposable profile'}
$held=Physical $root
try {
 $acl=Get-Acl -LiteralPath $root
 if($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Root owner differs from actual user'}
 foreach($ace in $acl.Access){
  $principal=$ace.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
  if($ace.AccessControlType -eq 'Allow' -and $principal -notin @($sid,'S-1-5-18','S-1-5-32-544') -and ($ace.FileSystemRights -band ([Security.AccessControl.FileSystemRights]::Write -bor [Security.AccessControl.FileSystemRights]::Delete -bor [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor [Security.AccessControl.FileSystemRights]::ChangePermissions -bor [Security.AccessControl.FileSystemRights]::TakeOwnership))){throw 'Root writable by unrelated principal'}
 }
 . (Join-Path $PSScriptRoot 'windows-bundle-metadata.ps1')
 . (Join-Path $PSScriptRoot 'desktop-smoke-windows.ps1')
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 function Artifact($side){
  $spec=$r.$side
  $physical=Physical $spec.path
  try {
   $hash=(Get-FileHash -LiteralPath $spec.path -Algorithm SHA256).Hash.ToLowerInvariant()
   if($hash -cne $spec.sha256){throw 'Artifact digest mismatch'}
   if($side -ceq 'baseline' -and $hash -cne '55be24a18b1827cabfd951e30645a69e9cfcd0d1b07729e1b1c8f3483b079ec9'){throw 'Retained baseline mismatch'}
   $sig=Get-AuthenticodeSignature -LiteralPath $spec.path
   if($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -cne 'CN=Semantier Technology, O=Semantier Technology'){throw 'Native trust/publisher mismatch'}
   $zip=[IO.Compression.ZipFile]::OpenRead($spec.path)
   try {
    $entry=$zip.GetEntry('app/resources/install-stamp.json');if(-not $entry){throw 'Install stamp missing'}
    $reader=[IO.StreamReader]::new($entry.Open());try{$stamp=$reader.ReadToEnd()|ConvertFrom-Json}finally{$reader.Dispose()}
   }finally{$zip.Dispose()}
   if($stamp.dirty -or $stamp.source -cne 'channel-build' -or $stamp.payload -cne 'bundled' -or $stamp.channelBuild.semantierScope -cne 'semantier-windows-x64/v1'){throw 'Wrong source/channel stamp'}
   $expected=[pscustomobject]@{msixIdentity='Semantier.Agent.Acceptance';publisher='CN=Semantier Technology, O=Semantier Technology';applicationId=$stamp.channelBuild.identity.appNamePascal;windowsVersion=$stamp.channelBuild.windowsVersion}
   $metadata=Read-BundleSmokeMetadata $spec.path 'x64' $expected
   if($metadata.Slice){throw 'This entrypoint requires signed MSIX'}
   return @{name=$expected.msixIdentity;publisher=$expected.publisher;version=$metadata.Version;arch='X64';commit=$stamp.commit;build_id=$stamp.channelBuild.buildId}
  }finally{$physical.handle.Dispose()}
 }
 function ExactBirth($proc) {
  $handle=Get-Process -Id $proc.ProcessId -ErrorAction Stop
  try{
   $null=$handle.Handle
   $birth=[DateTimeOffset]$handle.StartTime
   if([Math]::Abs(($birth.ToUniversalTime()-([DateTimeOffset]$proc.CreationDate).ToUniversalTime()).Ticks) -ge 10 -or $handle.Path -ine $proc.ExecutablePath){throw 'Process changed during observation'}
   return $birth.ToUniversalTime().ToString('o')
  }finally{$handle.Dispose()}
 }
 function GatewayIdentity($package) {
  $file=Join-Path $root 'userdata\hermes-home\state\gateway.lifecycle.json'
  if(-not (Test-Path -LiteralPath $file)){return $null}
  $fileHold=Physical $file
  try{$life=Get-Content -LiteralPath $file -Raw|ConvertFrom-Json}finally{$fileHold.handle.Dispose()}
  if($life.phase -cne 'running'){return $null}
  $proc=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$life.pid)"
  if(-not $proc){return $null}
  $prefix=Join-Path $package.install_path 'app\resources\agent-payload\'
  if($proc.SessionId -ne $session -or (Invoke-CimMethod -InputObject $proc -MethodName GetOwnerSid).Sid -cne $sid -or -not $proc.ExecutablePath.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Gateway lifecycle ownership mismatch'}
  $epoch=([DateTimeOffset]$proc.CreationDate).ToUnixTimeMilliseconds()/1000.0
  if(-not $life.create_time -or [Math]::Abs($epoch-$life.create_time) -ge 2){throw 'Gateway lifecycle incarnation mismatch'}
  $exeHold=Physical $proc.ExecutablePath
  try{return @{pid=$proc.ProcessId;birth=(ExactBirth $proc);exe=$proc.ExecutablePath;exe_id=$exeHold.id;packageFullName=$package.full}}finally{$exeHold.handle.Dispose()}
 }
 function VerifyDownload {
  $target=$r.target
  if($target.manifest_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $target.chunk_descriptor_sha256 -cnotmatch '^[a-f0-9]{64}$'){throw 'Qualified target manifest/chunk identities required'}
  $directory=Join-Path $root 'userdata';$queue=[Collections.Generic.Queue[string]]::new();$queue.Enqueue($directory);$records=@();$matches=@()
  while($queue.Count){
   foreach($entry in @(Get-ChildItem -LiteralPath $queue.Dequeue() -Force)){
    if(($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'Reparse in updater custody'}
    if($entry.PSIsContainer){$queue.Enqueue($entry.FullName)}
    elseif($entry.Name -like '.channel-artifact-*.json'){
     $record=Get-Content -LiteralPath $entry.FullName -Raw|ConvertFrom-Json
     if($record.artifactSha256 -ceq $target.sha256 -and $record.manifestSha256 -ceq $target.manifest_sha256 -and $record.phase -ceq 'handoff'){$records+= $record}
    }elseif($entry.Extension -ceq '.msix'){
     $physical=Physical $entry.FullName
     try{if((Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ceq $target.sha256){$sig=Get-AuthenticodeSignature -LiteralPath $entry.FullName;if($sig.Status -ne 'Valid'){throw 'Downloaded signature invalid'};$matches+= $entry.FullName}}finally{$physical.handle.Dispose()}
    }
   }
  }
  if($records.Count -ne 1 -or $matches.Count -ne 1){throw 'Exact production download custody not established'}
  return @{manifestSha256=$target.manifest_sha256;artifactSha256=$target.sha256;expectedChunkDescriptorSha256=$target.chunk_descriptor_sha256;chunkObservation='NOT_RUN';scope='full package hash/signature and production manifest cache record';file=$matches[0]}
 }
 function Observe {
  $packages=@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance');if($packages.Count -gt 1){throw 'Ambiguous registration'}
  $package=$null;$process=$null
  if($packages.Count){
   $p=$packages[0]
   if($p.Publisher -cne 'CN=Semantier Technology, O=Semantier Technology' -or $p.Status.ToString() -cne 'Ok' -or $p.SignatureKind.ToString() -cne 'Developer' -or $p.IsDevelopmentMode){throw 'Unexpected registered package'}
   $manifest=Get-AppxPackageManifest -Package $p.PackageFullName
   $apps=@($manifest.Package.Applications.Application);if($apps.Count -ne 1){throw 'Ambiguous application'}
   $exe=Join-Path $p.InstallLocation $apps[0].Executable
   $physical=Physical $exe
   try {
    $stamp=Get-Content -LiteralPath (Join-Path $p.InstallLocation 'app\resources\install-stamp.json') -Raw|ConvertFrom-Json
    $package=@{name=$p.Name;publisher=$p.Publisher;version=$p.Version.ToString();arch=$p.Architecture.ToString();commit=$stamp.commit;build_id=$stamp.channelBuild.buildId;full=$p.PackageFullName;family=$p.PackageFamilyName;install_path=$p.InstallLocation;exe=$exe;exe_id=$physical.id}
    $rows=Main-Processes $exe
    if($rows.Count -gt 1){throw 'Ambiguous installed main process'}
    if($rows.Count){$row=$rows[0];$owner=Invoke-CimMethod -InputObject $row -MethodName GetOwnerSid;if($owner.Sid -cne $sid -or $row.SessionId -ne $session){throw 'Process belongs to different user/session'};$process=@{pid=$row.ProcessId;birth=(ExactBirth $row);exe=$exe;exe_id=$physical.id}}
   }finally{$physical.handle.Dispose()}
  }
  return @{sid=$sid;session=$session;root_id=$held.id;package=$package;process=$process}
 }
 $result=$null
 if($r.action -in @('register','launch','update','remove')){
  if(-not $r.PSObject.Properties['expected_before']){throw 'Durable intent precondition required'}
  $actual=Observe;$expected=$r.expected_before
  if($actual.sid -cne $expected.sid -or $actual.session -ne $expected.session -or $actual.root_id -cne $expected.root_id){throw 'Intent context changed'}
  foreach($kind in @('package','process')){
   if(($null -eq $actual[$kind]) -ne ($null -eq $expected.$kind)){throw 'Intent OS state changed'}
   if($actual[$kind]){foreach($property in $expected.$kind.PSObject.Properties){if($actual[$kind][$property.Name] -cne $property.Value){throw 'Intent identity changed'}}}
  }
 }
 switch($r.action){
  observe {$result=Observe}
  artifact {$result=Artifact $r.side}
  download {$result=VerifyDownload}
  register {
   $sourceHold=Physical $r.baseline.path
   try{
    $expected=Artifact 'baseline';$before=Observe;if($before.package){throw 'Existing package conflict'}
    Add-AppxPackage -Path $r.baseline.path -ErrorAction Stop
    $result=Observe
   }finally{$sourceHold.handle.Dispose()}
  }
  launch {
   $before=Observe;if(-not $before.package -or $before.process){throw 'Launch precondition failed'}
   $env:HERMES_DESKTOP_USER_DATA_DIR=Join-Path $root 'userdata';$env:HERMES_HOME=Join-Path $root 'userdata\hermes-home'
   $env:HERMES_DESKTOP_IGNORE_EXISTING='1';$env:HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION='1'
   $exeHold=Physical $before.package.exe
   try{
    if($exeHold.id -cne $before.package.exe_id){throw 'Executable replaced before launch'}
    Start-Process -FilePath $before.package.exe -WorkingDirectory $root | Out-Null
   }finally{$exeHold.handle.Dispose()}
   $deadline=(Get-Date).AddMinutes(3);do{Start-Sleep -Seconds 1;$result=Observe}while(-not $result.process -and (Get-Date)-lt $deadline)
   if(-not $result.process){throw 'Registered launch failed'}
  }
  update {
   $target=Artifact 'target';$before=Observe
   if(-not $before.process -or [version]$target.version -le [version]$before.package.version){throw 'Update precondition failed'}
   $proof=Join-Path $root 'proof';$log=Join-Path $root 'ui-driver.log'
   if((Test-Path -LiteralPath $proof) -or (Test-Path -LiteralPath $log)){throw 'Existing UI evidence requires inspection; no overwrite'}
   New-Item -ItemType Directory -Path $proof -ErrorAction Stop|Out-Null
   $proofHold=Physical $proof
   try{
    $uiOutput=& (Join-Path $PSScriptRoot 'windows-bundled-drive-update.ps1') -OldProcessId $before.process.pid -ProofDir $proof -SemantierIsolated -ExpectedSid $sid -ExpectedExecutable $before.package.exe -ExpectedBirth $before.process.birth -ExpectedTargetVersion $target.version *>&1
    $uiExit=$LASTEXITCODE
    $logHandle=[IO.File]::Open($log,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{$bytes=[Text.Encoding]::UTF8.GetBytes(($uiOutput|Out-String));$logHandle.Write($bytes,0,$bytes.Length);$logHandle.Flush($true)}finally{$logHandle.Dispose()}
    if($uiExit -ne 0){throw 'Production UI update failed'}
   }finally{$proofHold.handle.Dispose()}
   $deadline=(Get-Date).AddMinutes(15)
   do{Start-Sleep -Seconds 2;$result=Observe}while(((-not $result.process) -or $result.package.version -cne $target.version) -and (Get-Date)-lt $deadline)
   if(-not $result.process -or $result.package.version -cne $target.version){throw 'Successor did not automatically relaunch'}
   $oldPayload=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and $_.ExecutablePath.StartsWith($before.package.install_path+'\',[StringComparison]::OrdinalIgnoreCase)})
   if($oldPayload.Count){throw 'Old payload processes remain; preserve update state'}
  }
  ready {
   $state=Observe;if(-not $state.process){throw 'Registered application not running'}
   $payload=Join-Path $state.package.install_path 'app\resources\agent-payload\'
   $deadline=(Get-Date).AddMinutes(3);$healthy=$false
   do{
    foreach($proc in @(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and $_.ExecutablePath.StartsWith($payload,[StringComparison]::OrdinalIgnoreCase) -and $_.SessionId -eq $session})){
     foreach($port in @(Get-NetTCPConnection -State Listen -OwningProcess $proc.ProcessId -ErrorAction SilentlyContinue)){
      try{$reply=Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($port.LocalPort)/api/health" -TimeoutSec 2;if($reply.StatusCode -eq 200){$healthy=$true}}catch{}
     }
    };if(-not $healthy){Start-Sleep -Seconds 2}
   }while(-not $healthy -and (Get-Date)-lt $deadline)
   if(-not $healthy){throw 'Standalone backend readiness failed'}
   Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
   $condition=[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ProcessIdProperty,[int]$state.process.pid)
   $window=[System.Windows.Automation.AutomationElement]::RootElement.FindFirst([System.Windows.Automation.TreeScope]::Children,$condition)
   if(-not $window){throw 'Registered application window unavailable'}
   $names=@($window.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|ForEach-Object{$_.Current.Name})
   $newSession=if($r.expected_language -ceq 'zh'){([char]0x65b0+[char]0x5efa+[char]0x4f1a+[char]0x8bdd)}else{'New session'}
   if($names -notcontains $newSession){throw 'Expected production language control unavailable'}
   $preference=Get-Content -LiteralPath (Join-Path $root 'userdata\active-profile.json') -Raw|ConvertFrom-Json
   if($preference.profile -cne 'semantier-local' -or $preference.defaultRoute.profile -cne 'semantier-local'){throw 'Unexpected active desktop profile'}
   $profiles=Join-Path $root 'userdata\hermes-home\profiles'
   $actualNames=@(Get-ChildItem -LiteralPath $profiles -Directory|Where-Object{-not $_.Name.StartsWith('.')}|ForEach-Object{$_.Name}|Sort-Object)
   $expectedNames=@($r.expected_profiles|Sort-Object)
   if(($actualNames -join '|') -cne ($expectedNames -join '|') -or $actualNames -contains 'tender-runtime-expert'){throw 'Desktop profile namespace mismatch'}
   $result=@{healthy=$true;languageControl=$r.expected_language;activeProfile='semantier-local';profiles=$actualNames;gateway=(GatewayIdentity $state.package);themeRendering='retained exact-package evidence only; current rendering NOT_RUN'}
  }
  conservation {
   $result=@{}
   foreach($category in @('chats','credentials','providers','memories','skills','language','theme','profiles','legacy_backup')){
    $property=$r.sentinels.PSObject.Properties[$category]
    if(-not $property -or -not @($property.Value).Count){throw 'Missing required conservation category'}
    foreach($relative in @($property.Value)){Assert-SmokeRelativePath $relative;$file=Join-Path $root $relative;$physical=Physical $file;try{$result[$category+':'+$relative]=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()}finally{$physical.handle.Dispose()}}
   }
  }
  remove {
   $before=Observe;if(-not $before.package){throw 'No registered package'}
   $owned=@($r.owned_packages|Where-Object{$_.full -ceq $before.package.full -and $_.exe_id -ceq $before.package.exe_id -and $_.family -ceq $before.package.family})
   if(-not $owned.Count){throw 'Unowned registration: preserve it'}
   if($before.process){
    $ownedProcess=@($r.owned_processes|Where-Object{$_.pid -eq $before.process.pid -and $_.birth -ceq $before.process.birth -and $_.exe_id -ceq $before.process.exe_id})
    if(-not $ownedProcess.Count){throw 'Unowned process: preserve it'}
    $mainHandle=Get-Process -Id $before.process.pid -ErrorAction Stop
    try{
     $null=$mainHandle.Handle
     if(([DateTimeOffset]$mainHandle.StartTime).ToUniversalTime().ToString('o') -cne $before.process.birth){throw 'Main incarnation changed before close'}
     if(-not $mainHandle.CloseMainWindow() -or -not $mainHandle.WaitForExit(30000)){throw 'Owned main did not close normally'}
    }finally{$mainHandle.Dispose()}
   }
   $gateway=GatewayIdentity $before.package
   if($gateway){
    $ownedGateway=@($r.owned_gateways|Where-Object{$_.pid -eq $gateway.pid -and $_.birth -ceq $gateway.birth -and $_.exe -ceq $gateway.exe -and $_.exe_id -ceq $gateway.exe_id -and $_.packageFullName -ceq $before.package.full})
    if(-not $ownedGateway.Count){throw 'Unrecorded gateway incarnation: preserve it'}
    $freshGateway=GatewayIdentity $before.package
    if(-not $freshGateway -or $freshGateway.pid -ne $gateway.pid -or $freshGateway.birth -cne $gateway.birth -or $freshGateway.exe_id -cne $gateway.exe_id){throw 'Gateway changed before cleanup'}
    $gatewayHandle=Get-Process -Id $gateway.pid -ErrorAction Stop
    try{
     $null=$gatewayHandle.Handle
     if(([DateTimeOffset]$gatewayHandle.StartTime).ToUniversalTime().ToString('o') -cne $gateway.birth){throw 'Gateway incarnation changed before stop'}
     $gatewayHandle.Kill()
     if(-not $gatewayHandle.WaitForExit(60000)){throw 'Owned gateway did not stop'}
    }finally{$gatewayHandle.Dispose()}
   }
   $remaining=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and $_.ExecutablePath.StartsWith($before.package.install_path+'\',[StringComparison]::OrdinalIgnoreCase)})
   if($remaining.Count){throw 'Package processes remain; preserve registration for recovery'}
   $again=Observe;if($again.package.full -cne $before.package.full -or $again.package.exe_id -cne $before.package.exe_id){throw 'Registration changed before cleanup'}
   Remove-AppxPackage -Package $before.package.full -ErrorAction Stop;$result=Observe
  }
  default {throw 'Unknown native operation'}
 }
 $result|ConvertTo-Json -Compress -Depth 20
}finally{$held.handle.Dispose()}
