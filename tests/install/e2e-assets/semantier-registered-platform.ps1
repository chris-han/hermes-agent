# Native boundary for semantier_registered_update.py. No authority from markers.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$r=[Console]::In.ReadToEnd() | ConvertFrom-Json
if ($env:OS -cne 'Windows_NT') { throw 'Windows required' }
$who=[Security.Principal.WindowsIdentity]::GetCurrent()
$sid=$who.User.Value
$session=(Get-Process -Id $PID).SessionId
. (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
$localWD1=$false
if($r.PSObject.Properties['purpose']){$localWD1=Test-WD1LocalActor $sid $r.authorized_sid $r.purpose}
if ($sid -cne $r.authorized_sid -or (($sid -ceq 'S-1-5-21-1928370875-4287869425-2092521300-1003' -or $who.Name.Split('\')[-1] -ieq 'ChrisH') -and -not $localWD1) -or $session -le 0 -or -not [Environment]::UserInteractive) { throw 'Authorized interactive user required; ChrisH limited to explicit local WD1 purpose' }
if ($r.expected_language -cnotin @('en','zh')) { throw 'Expected language must be en or zh' }
$cpus=@(Get-CimInstance Win32_Processor)
if(-not $cpus.Count -or @($cpus|Where-Object{$_.Architecture -ne 9}).Count){throw 'Actual native x64 required'}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class SemantierNativeFacts {
 [DllImport("wtsapi32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr s,int id,int info,out IntPtr b,out int n);
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
function Wts-Text([int]$kind) {
 $buffer=[IntPtr]::Zero;$length=0
 if(-not [SemantierNativeFacts]::WTSQuerySessionInformation([IntPtr]::Zero,$session,$kind,[ref]$buffer,[ref]$length)){throw 'WTS account unavailable'}
 try{if($length -lt 2){throw 'Empty WTS account'};return [Runtime.InteropServices.Marshal]::PtrToStringUni($buffer)}finally{[SemantierNativeFacts]::WTSFreeMemory($buffer)}
}
# WTS_INFO_CLASS: 6 is WTSWinStationName; domain is 7.
$wtsUser=Wts-Text 5;$wtsDomain=Wts-Text 7
if(-not $wtsUser -or -not $wtsDomain){throw 'Signed-in WTS account unavailable'}
$sessionSid=([Security.Principal.NTAccount]::new($wtsDomain,$wtsUser)).Translate([Security.Principal.SecurityIdentifier]).Value
if($sessionSid -cne $sid){throw 'Actual signed-in WTS account differs from process account'}
function Physical([string]$path) {
 $item=Get-Item -LiteralPath $path -Force
 $cursor=$item
 while($cursor){if(($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'Reparse path refused'};if($cursor -is [IO.DirectoryInfo]){$cursor=$cursor.Parent}else{$cursor=$cursor.Directory}}
 $handle=[SemantierNativeFacts]::CreateFile($item.FullName,0,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
 if($handle.IsInvalid){throw ('Cannot hold physical path against replacement: {0}, Win32={1}' -f $path,[Runtime.InteropServices.Marshal]::GetLastWin32Error())}
 $final=[Text.StringBuilder]::new(32768)
 $length=[SemantierNativeFacts]::GetFinalPathNameByHandle($handle,$final,32768,0)
 $canonical=$final.ToString()
 if($canonical.StartsWith('\\?\UNC\')){$canonical='\\'+$canonical.Substring(8)}elseif($canonical.StartsWith('\\?\')){$canonical=$canonical.Substring(4)}
 if($length -eq 0 -or $length -ge 32768 -or $canonical.TrimEnd('\') -ine $item.FullName.TrimEnd('\')){$handle.Dispose();throw 'Opened physical path escaped inspected path'}
 $info=New-Object SemantierNativeFacts+Info
 if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info)){$handle.Dispose();throw 'Physical identity unavailable'}
 return @{handle=$handle;id=('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow);path=$item.FullName}
}
# Only an OS-registered package root may cross a package-volume redirect.
# Fixture/artifact/process paths continue through strict Physical unchanged.
function Registered-Root($package) {
 $logical=[IO.Path]::GetFullPath($package.InstallLocation).TrimEnd('\')
 $full=$package.PackageFullName
 if(-not $full -or [IO.Path]::GetFileName($logical) -cne $full){throw 'Registered package root/name mismatch'}
 $stores=@(Get-AppxVolume|Where-Object{-not $_.IsOffline}|ForEach-Object{[IO.Path]::GetFullPath($_.PackageStorePath).TrimEnd('\')})
 $parent=Split-Path -Parent $logical
 if($parent -notin $stores){throw 'Registered package root outside OS package stores'}
 $cursor=Get-Item -LiteralPath $parent -Force
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse package-store ancestor refused'};$cursor=$cursor.Parent}
 $handle=$null;$physical=$null
 try {
  $handle=[SemantierNativeFacts]::CreateFile($logical,0,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
  if($handle.IsInvalid){throw 'Registered package root unavailable'}
  $final=[Text.StringBuilder]::new(32768)
  $length=[SemantierNativeFacts]::GetFinalPathNameByHandle($handle,$final,32768,0)
  if($length -eq 0 -or $length -ge 32768){throw 'Registered package root resolution failed'}
  $resolved=$final.ToString()
  if($resolved.StartsWith('\\?\')){$resolved=$resolved.Substring(4)}
  $matches=@($stores|ForEach-Object{Join-Path $_ $full}|Where-Object{$_ -ieq $resolved})
  if($matches.Count -ne 1){throw 'Registered package redirect outside exact OS package store'}
  $physical=Physical $matches[0]
  $info=New-Object SemantierNativeFacts+Info
  if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info)){throw 'Registered package identity unavailable'}
  if(('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow) -cne $physical.id){throw 'Registered package redirect changed'}
  return $physical.path
 } finally {if($physical){$physical.handle.Dispose()};if($handle){$handle.Dispose()}}
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
 function Read-InstallStamp([string]$file) {
  $zip=[IO.Compression.ZipFile]::OpenRead($file)
  try{$entry=$zip.GetEntry('app/resources/install-stamp.json');if(-not $entry){throw 'Install stamp missing'};$reader=[IO.StreamReader]::new($entry.Open());try{return $reader.ReadToEnd()|ConvertFrom-Json}finally{$reader.Dispose()}}finally{$zip.Dispose()}
 }
 function Native-Artifact {
  $spec=$r.target;$binding=$spec.native_graph;$g=$r.native_graph_value
  $graphHold=Physical $binding.path
  try{
   if((Get-FileHash -LiteralPath $binding.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $binding.sha256){throw 'Native graph file changed'}
   $readback=Get-Content -LiteralPath $binding.path -Raw|ConvertFrom-Json
   if(($readback|ConvertTo-Json -Depth 16 -Compress) -cne ($g|ConvertTo-Json -Depth 16 -Compress)){throw 'Native graph differs from validated request'}
  }finally{$graphHold.handle.Dispose()}
  . (Join-Path $PSScriptRoot '..\..\..\apps\desktop\scripts\native-package-digest.ps1')
  $directory=Split-Path -Parent $spec.path;$prefix='releases/channel-builds/'+$spec.build_id+'/'
  $rows=@($g.objects);$main=@($rows|Where-Object{$_.kind -ceq 'main-package'});$rootRow=@($rows|Where-Object{$_.key -ceq $g.root})
  if($main.Count -ne 1 -or $rootRow.Count -ne 1 -or $rootRow[0].kind -cne 'main-bundle' -or $rows.Count -lt 3){throw 'Expected complete folded asset bundle'}
  $related=@();$family=$null;$mainIdentity=$null
  foreach($row in $rows){
   $name=$row.key.Substring($prefix.Length)
   if(-not $row.key.StartsWith($prefix,[StringComparison]::Ordinal) -or $name -match '[\\/]' -or $row.kind -cnotin @('main-bundle','main-package','asset-package')){throw 'Unsupported native object path/role'}
   $file=Join-Path $directory $name;$hold=Physical $file
   try{
    if((Get-Item -LiteralPath $file).Length -ne $row.size -or $row.size -gt 480000000 -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -cne $row.sha256){throw 'Native object bytes mismatch'}
    $sig=Get-AuthenticodeSignature -LiteralPath $file
    if($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -cne 'CN=Semantier Technology, O=Semantier Technology'){throw 'Native object trust/publisher mismatch'}
    if((Get-NativePackageDigest $file) -cne $row.nativeDigest){throw 'Native object SDK digest mismatch'}
    if($row.kind -ceq 'main-bundle'){continue}
    $id=Get-NativePackageIdentity $file
    if($id.Name -cne 'Semantier.Agent.Acceptance' -or $id.Publisher -cne $row.publisher -or $id.Version -cne $row.version){throw 'Native SDK identity mismatch'}
    if($family -and $family -cne $id.Family){throw 'Native package family discontinuity'};$family=$id.Family
    $z=[IO.Compression.ZipFile]::OpenRead($file)
    try{
     $xml=Read-SmokeXmlEntry $z 'AppxManifest.xml'
     if(@($xml.SelectNodes('/*[local-name()="Package"]/*[local-name()="Dependencies"]/*[local-name()!="TargetDeviceFamily"]')).Count){throw 'Unbound native dependency'}
     if($row.kind -ceq 'asset-package'){
      $resource=$xml.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Properties"]/*[local-name()="ResourcePackage"]')
      $execution=$xml.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Properties"]/*[local-name()="AllowExecution"]')
      if($id.Architecture -ne 11 -or $id.ResourceId -cne $row.resourceId -or !$resource -or $resource.InnerText -cne 'true' -or ($execution -and $execution.InnerText -cne 'true')){throw 'Native asset capability mismatch'}
      $related+=@{name=$id.Name;publisher=$id.Publisher;version=$id.Version;arch='Neutral';resource_id=$id.ResourceId;full=$id.FullName;family=$id.Family}
     }else{
      if($id.Architecture -ne 9 -or $id.ResourceId){throw 'Native main architecture/resource mismatch'};$mainIdentity=$id
     }
    }finally{$z.Dispose()}
   }finally{$hold.handle.Dispose()}
  }
  $bundle=[IO.Compression.ZipFile]::OpenRead($spec.path)
  try{
   if(@($bundle.Entries|Where-Object{$_.FullName -match '\.(msix|appx)$'}).Count){throw 'Expected external native bundle'}
   $xml=Read-SmokeXmlEntry $bundle 'AppxMetadata/AppxBundleManifest.xml'
   if(@($xml.SelectNodes('/*[local-name()="Bundle"]/*[local-name()!="Identity" and local-name()!="Packages"]')).Count){throw 'Unbound related native bundle'}
   $entries=@($xml.SelectNodes('/*[local-name()="Bundle"]/*[local-name()="Packages"]/*[local-name()="Package"]'))
   if($entries.Count -ne ($rows.Count - 1) -or @($rootRow[0].references).Count -ne $entries.Count){throw 'Incomplete native bundle references'}
   $seen=@{}
   foreach($entry in $entries){
    $key=$prefix+$entry.GetAttribute('FileName');$match=@($rows|Where-Object{$_.key -ceq $key})
    if($seen.ContainsKey($key) -or $match.Count -ne 1 -or $rootRow[0].references -cnotcontains $key -or $entry.GetAttribute('Version') -cne $spec.version){throw 'Native bundle reference mismatch'};$seen[$key]=$true
    if($match[0].kind -ceq 'asset-package'){
     if($entry.GetAttribute('Type') -cne 'resource' -or $entry.GetAttribute('ResourceId') -cne $match[0].resourceId -or @($entry.SelectNodes('*[local-name()="Resources"]/*/@*')).Count){throw 'Required native asset can be omitted'}
    }elseif($entry.GetAttribute('Type') -cne 'application' -or $entry.GetAttribute('Architecture') -cne 'x64'){throw 'Native main bundle entry mismatch'}
   }
  }finally{$bundle.Dispose()}
  $stamp=Read-InstallStamp (Join-Path $directory (($main[0].key -split '/')[-1]))
  if($stamp.dirty -or $stamp.source -cne 'channel-build' -or $stamp.channelBuild.semantierScope -cne 'semantier-windows-x64/v1' -or $stamp.commit -cne $spec.commit -or $stamp.channelBuild.buildId -cne $spec.build_id){throw 'Native payload source/build mismatch'}
  $expected=[pscustomobject]@{msixIdentity='Semantier.Agent.Acceptance';publisher='CN=Semantier Technology, O=Semantier Technology';applicationId=$stamp.channelBuild.identity.appNamePascal;windowsVersion=$spec.version}
  $null=Read-BundleSmokeMetadata $spec.path 'x64' $expected $main[0]
  return @{name=$mainIdentity.Name;publisher=$mainIdentity.Publisher;version=$mainIdentity.Version;arch='X64';commit=$stamp.commit;build_id=$stamp.channelBuild.buildId;full=$mainIdentity.FullName;family=$mainIdentity.Family;related=@($related|Sort-Object {$_['full']})}
 }
 function Artifact($side){
  $spec=$r.$side
  if($side -ceq 'target' -and $spec.PSObject.Properties['native_graph']){return Native-Artifact}
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
 function Installed-File($pkg,[string]$path) {
  $physicalRoot=Registered-Root $pkg
  $logicalRoot=[IO.Path]::GetFullPath($pkg.InstallLocation).TrimEnd('\')
  $full=[IO.Path]::GetFullPath($path)
  if($full.StartsWith($physicalRoot+'\',[StringComparison]::OrdinalIgnoreCase)){return Physical $full}
  if(-not $full.StartsWith($logicalRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Process file outside registered package'}
  $physical=Physical (Join-Path $physicalRoot $full.Substring($logicalRoot.Length+1))
  $handle=$null
  try {
   $handle=[SemantierNativeFacts]::CreateFile($full,0,1,[IntPtr]::Zero,3,0x02000000,[IntPtr]::Zero)
   if($handle.IsInvalid){throw 'Registered process file unavailable'}
   $info=New-Object SemantierNativeFacts+Info
   if(-not [SemantierNativeFacts]::GetFileInformationByHandle($handle,[ref]$info) -or ('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow) -cne $physical.id){throw 'Registered process file physical identity mismatch'}
   return $physical
  }catch{$physical.handle.Dispose();throw}finally{if($handle){$handle.Dispose()}}
 }
 function GatewayIdentity($package) {
  $file=Join-Path $root 'userdata\hermes-home\state\gateway.lifecycle.json'
  if(-not (Test-Path -LiteralPath $file)){return $null}
  $fileHold=Physical $file
  try{$life=Get-Content -LiteralPath $file -Raw|ConvertFrom-Json}finally{$fileHold.handle.Dispose()}
  if($life.phase -cne 'running'){return $null}
  $proc=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$life.pid)"
  if(-not $proc){return $null}
  $pkg=@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance'|Where-Object{$_.PackageFullName -ceq $package.full});if($pkg.Count -ne 1){throw 'Gateway package changed'}
  $prefix=Join-Path $package.install_path 'app\resources\agent-payload\'
  if($proc.SessionId -ne $session -or (Invoke-CimMethod -InputObject $proc -MethodName GetOwnerSid).Sid -cne $sid ){throw 'Gateway lifecycle ownership mismatch'}
  $epoch=([DateTimeOffset]$proc.CreationDate).ToUnixTimeMilliseconds()/1000.0
  if(-not $life.create_time -or [Math]::Abs($epoch-$life.create_time) -ge 2){throw 'Gateway lifecycle incarnation mismatch'}
  $exeHold=Installed-File $pkg[0] $proc.ExecutablePath
  if(-not $exeHold.path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){$exeHold.handle.Dispose();throw 'Gateway outside installed payload'}
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
 function Related-Packages {
  $result=@()
  $optional=@(Get-AppxPackage -PackageTypeFilter Optional -Name 'Semantier.Agent.Acceptance.*')
  $assets=@(Get-AppxPackage -PackageTypeFilter Resource -Name 'Semantier.Agent.Acceptance')
  foreach($child in @($optional)+@($assets)){
   if($child.Publisher -cne 'CN=Semantier Technology, O=Semantier Technology' -or $child.Status.ToString() -cne 'Ok' -or $child.SignatureKind.ToString() -cne 'Developer' -or $child.IsDevelopmentMode){throw 'Unexpected related package; preserve it'}
   $xml=Get-AppxPackageManifest -Package $child.PackageFullName
   if($child.Name -ceq 'Semantier.Agent.Acceptance'){
    $ident=$xml.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Identity"]')
    $props=$xml.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Properties"]')
    $resource=$props.SelectSingleNode('*[local-name()="ResourcePackage"]')
    $execution=$props.SelectSingleNode('*[local-name()="AllowExecution"]')
    if($ident.GetAttribute('ResourceId') -cnotmatch '^runtime[0-9]{4}$' -or !$resource -or $resource.InnerText -cne 'true' -or ($execution -and $execution.InnerText -cne 'true')){throw 'Unexpected same-family resource package; preserve it'}
   }else{
    $parents=@($xml.SelectNodes('/*[local-name()="Package"]/*[local-name()="Dependencies"]/*[local-name()="MainPackageDependency"]'))
    if($parents.Count -ne 1 -or $parents[0].GetAttribute('Name') -cne 'Semantier.Agent.Acceptance'){throw 'Related package parent mismatch; preserve it'}
    $publisher=$parents[0].GetAttribute('Publisher')
    if($publisher -and $publisher -cne 'CN=Semantier Technology, O=Semantier Technology'){throw 'Related package parent publisher mismatch'}
   }
   $physical=Physical (Registered-Root $child)
   try{$result+=[ordered]@{name=$child.Name;publisher=$child.Publisher;version=$child.Version.ToString();arch=$child.Architecture.ToString();resource_id=$child.ResourceId;full=$child.PackageFullName;family=$child.PackageFamilyName;install_path=$physical.path;root_id=$physical.id}}finally{$physical.handle.Dispose()}
  }
  return ,@($result|Sort-Object {$_['full']})
 }
 function Observe {
  $packages=@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance');if($packages.Count -gt 1){throw 'Ambiguous registration'}
  $package=$null;$process=$null
  if($packages.Count){
   $p=$packages[0]
   if($p.Publisher -cne 'CN=Semantier Technology, O=Semantier Technology' -or $p.Status.ToString() -cne 'Ok' -or $p.SignatureKind.ToString() -cne 'Developer' -or $p.IsDevelopmentMode){throw 'Unexpected registered package'}
   $manifest=Get-AppxPackageManifest -Package $p.PackageFullName
   $expected=[pscustomobject]@{msixIdentity='Semantier.Agent.Acceptance';publisher='CN=Semantier Technology, O=Semantier Technology';windowsVersion=$p.Version.ToString();applicationId='SemantierAgentAcceptance'}
   $desktop=Read-SmokePackageManifest $manifest 'x64' $expected
   $install=Registered-Root $p
   $exe=Join-Path $install $desktop.Executable
   $physical=Physical $exe
   try {
    $stamp=Get-Content -LiteralPath (Join-Path $install 'app\resources\install-stamp.json') -Raw|ConvertFrom-Json
    $package=@{name=$p.Name;publisher=$p.Publisher;version=$p.Version.ToString();arch=$p.Architecture.ToString();commit=$stamp.commit;build_id=$stamp.channelBuild.buildId;full=$p.PackageFullName;family=$p.PackageFamilyName;install_path=$install;exe=$exe;exe_id=$physical.id}
    $logicalExe=Join-Path $p.InstallLocation $desktop.Executable
    $rows=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and $_.ExecutablePath -iin @($exe,$logicalExe) -and $_.CommandLine -notmatch '(?:^|\s)--type(?:=|\s)'})
    if($rows.Count -gt 1){throw 'Ambiguous installed main process'}
    if($rows.Count){
     $row=$rows[0];$owner=Invoke-CimMethod -InputObject $row -MethodName GetOwnerSid
     if($owner.Sid -cne $sid -or $row.SessionId -ne $session){throw 'Process belongs to different user/session'}
     $processHold=Installed-File $p $row.ExecutablePath
     try{if($processHold.id -cne $physical.id){throw 'Observed process executable differs from package'};$process=@{pid=$row.ProcessId;birth=(ExactBirth $row);exe=$row.ExecutablePath;exe_id=$processHold.id}}finally{$processHold.handle.Dispose()}
    }
   }finally{$physical.handle.Dispose()}
  }
  return @{sid=$sid;session_sid=$sessionSid;session=$session;root_id=$held.id;package=$package;process=$process;related=(Related-Packages)}
 }
 $result=$null
 if($r.action -in @('register','launch','update','remove')){
  if(-not $r.PSObject.Properties['expected_before']){throw 'Durable intent precondition required'}
  $actual=Observe;$expected=$r.expected_before
  if($actual.sid -cne $expected.sid -or $actual.session_sid -cne $expected.session_sid -or $actual.session -ne $expected.session -or $actual.root_id -cne $expected.root_id){throw 'Intent context changed'}
  $expectedRelated=@()
  if($expected.PSObject.Properties['related']){$expectedRelated=@($expected.related)}
  if(@($actual.related).Count -ne $expectedRelated.Count){throw 'Intent related package set changed'}
  foreach($child in $expectedRelated){
   $matches=@($actual.related|Where-Object{$_['full'] -ceq $child.full})
   if($matches.Count -ne 1){throw 'Intent related package identity changed'}
   foreach($property in $child.PSObject.Properties){if($matches[0][$property.Name] -cne $property.Value){throw 'Intent related package incarnation changed'}}
  }
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
   $side=if($r.PSObject.Properties['purpose'] -and $r.purpose -ceq 'wd1-native-layout-qualification'){'target'}else{'baseline'}
   if($side -ceq 'target' -and -not $localWD1){throw 'Explicit authorized WD1 local actor required'}
   $source=$r.$side;$holds=@()
   try{
    $holds+=Physical $source.path
    if($side -ceq 'target'){
     $holds+=Physical $source.native_graph.path
     foreach($row in $r.native_graph_value.objects){if($row.key -cne $r.native_graph_value.root){$holds+=Physical (Join-Path (Split-Path -Parent $source.path) (($row.key -split '/')[-1]))}}
    }
    $expected=Artifact $side;$before=Observe;if($before.package -or @($before.related).Count){throw 'Existing package/related conflict'}
    if($r.PSObject.Properties['native_origin']){
     if($side -cne 'target' -or $r.purpose -cne 'wd1-native-layout-qualification'){throw 'Native URI proof requires WD1 target purpose'}
     $attempt=Join-Path $root ('native-deployment-'+[guid]::NewGuid().ToString('N'))
     New-Item -ItemType Directory -Path $attempt -ErrorAction Stop|Out-Null
     . (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
     Invoke-WD1NativeUri $r.native_graph_value $r.native_origin $attempt
    }else{Add-AppxPackage -Path $source.path -ErrorAction Stop}
    $result=Observe
   }finally{foreach($hold in $holds){$hold.handle.Dispose()}}
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
    $uiOutput=& (Join-Path $PSScriptRoot 'windows-bundled-drive-update.ps1') -OldProcessId $before.process.pid -ProofDir $proof -SemantierIsolated -ExpectedSid $sid -ExpectedExecutable $before.process.exe -ExpectedBirth $before.process.birth -ExpectedTargetVersion $target.version *>&1
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
  native-proof {
   if(-not $localWD1){throw 'WD1 local qualification context required'}
   $state=Observe;$target=Native-Artifact
   if(!$state.package -or $state.package.full -cne $target.full){throw 'Exact registered native package required'}
   $attempt=Join-Path $root ('native-proof-'+[guid]::NewGuid().ToString('N'))
   New-Item -ItemType Directory -Path $attempt -ErrorAction Stop|Out-Null
   $uriProof=$null
   if($r.PSObject.Properties['native_origin']){
    $proofFiles=@(Get-ChildItem -LiteralPath $root -Directory -Filter 'native-deployment-*'|ForEach-Object{Get-ChildItem -LiteralPath $_.FullName -File -Filter 'native-uri-proof.json'})
    if($proofFiles.Count -gt 1){throw 'Ambiguous completed native URI proofs; preserve attempts'}
    if($proofFiles.Count -eq 1){
     $uriProof=Get-Content -LiteralPath $proofFiles[0].FullName -Raw|ConvertFrom-Json
     if($uriProof.positive.status -cne 'SUCCESS'){throw 'Original native proof status invalid'}
    }else{
     . (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
     $uriProof=Recover-WD1NativeUri $r.native_graph_value $r.native_origin $root $state $target $attempt
    }
    $expectedUris=@($r.native_graph_value.objects|ForEach-Object{$r.native_origin+'/'+$_.key})
    $expectedDigests=@($r.native_graph_value.objects|ForEach-Object{$_.nativeDigest})
    if($uriProof.root -cne $r.native_graph_value.root -or $uriProof.refusal.hresult -cne '80080219' -or $uriProof.positive.status -cnotin @('SUCCESS','RECONCILED_OS_SUCCESS') -or ($uriProof.uris -join '|') -cne ($expectedUris -join '|') -or ($uriProof.digests -join '|') -cne ($expectedDigests -join '|')){throw 'Native URI proof differs from pinned target'}
   }
   $node=$r.native_node;$hold=Physical $node.path
   try{if((Get-FileHash -LiteralPath $node.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $node.sha256){throw 'Native Node identity mismatch'}}finally{$hold.handle.Dispose()}
   $layout=$r.target.native_graph.layout_plan;$hold=Physical $layout.path
   try{if((Get-FileHash -LiteralPath $layout.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $layout.sha256){throw 'Native layout identity mismatch'}}finally{$hold.handle.Dispose()}
   $locations=Join-Path $attempt 'native-payload-locations.json'
   & $node.path (Join-Path $PSScriptRoot 'native-payload-locations.mjs') --root $state.package.install_path --plan $layout.path --graph $r.target.native_graph.path --out $locations
   if($LASTEXITCODE -ne 0){throw 'Installed native payload location/conservation failed'}
   . (Join-Path $PSScriptRoot 'windows-bundle-plugin-smoke.ps1')
   $pkg=@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance'|Where-Object{$_.PackageFullName -ceq $state.package.full})
   if($pkg.Count -ne 1){throw 'Native qualification package changed'}
   Test-BundleNativePayload (Join-Path $state.package.install_path 'app\resources\agent-payload') $pkg[0] $target.commit $target.build_id $attempt
   $result=@{scope='LOCAL_NATIVE_LAYOUT_RUNTIME_ONLY_NOT_CI_OR_MIGRATION';nativeUriProof=$uriProof;package=$state.package;related=$state.related;locationProof=(Get-Content -LiteralPath $locations -Raw|ConvertFrom-Json);runtimeProof=(Get-Content -LiteralPath (Join-Path $attempt 'native-payload-runtime.json') -Raw|ConvertFrom-Json);evidence=$attempt}
  }
  ready {
   $state=Observe;if(-not $state.process){throw 'Registered application not running'}
   $pkg=@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance'|Where-Object{$_.PackageFullName -ceq $state.package.full});if($pkg.Count -ne 1){throw 'Ready package changed'}
   $payload=Join-Path $state.package.install_path 'app\resources\agent-payload\'
   $logicalPayload=Join-Path $pkg[0].InstallLocation 'app\resources\agent-payload\'
   $deadline=(Get-Date).AddMinutes(3);$healthy=$false
   do{
    foreach($proc in @(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and ($_.ExecutablePath.StartsWith($payload,[StringComparison]::OrdinalIgnoreCase) -or $_.ExecutablePath.StartsWith($logicalPayload,[StringComparison]::OrdinalIgnoreCase)) -and $_.SessionId -eq $session})){
     if((Invoke-CimMethod -InputObject $proc -MethodName GetOwnerSid).Sid -cne $sid){throw 'Backend user changed'}
     $backendHold=Installed-File $pkg[0] $proc.ExecutablePath
     try{if(-not $backendHold.path.StartsWith($payload,[StringComparison]::OrdinalIgnoreCase)){throw 'Backend outside installed payload'}}finally{$backendHold.handle.Dispose()}
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
