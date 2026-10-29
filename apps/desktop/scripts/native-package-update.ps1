# Production native branch of the existing external relaunch waiter.
# Windows owns all package transfers. No package bytes are downloaded here.
. (Join-Path $PSScriptRoot 'native-package-digest.ps1')
function Initialize-NativeUpdate($RequestPath, [int]$ParentId, [long]$ParentBirth, $Package) {
 $ErrorActionPreference='Stop'
 if((Get-Item -LiteralPath $RequestPath).Length -gt 4194304){throw 'Native request exceeds metadata limit'}
 $request=Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8|ConvertFrom-Json
 if($request.schema -ne 1 -or $request.identity -cne $Package.Name -or $request.fullName -cne $Package.PackageFullName -or $request.familyName -cne $Package.PackageFamilyName -or $request.publisher -cne $Package.Publisher -or $request.currentVersion -cne $Package.Version.ToString() -or [Version]$request.version -le $Package.Version -or $request.manifestSha256 -cnotmatch '^[a-f0-9]{64}$'){throw 'Native current package binding mismatch'}
 if($Package.Status.ToString() -cne 'Ok' -or $Package.SignatureKind.ToString() -cne 'Developer' -or $Package.IsDevelopmentMode -or $Package.NonRemovable){throw 'Native current package trust mismatch'}
 Add-Type -AssemblyName System.Runtime.WindowsRuntime
 $null=[Windows.Management.Deployment.PackageManager,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $null=[Windows.Management.Deployment.AddPackageOptions,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $manager=[Windows.Management.Deployment.PackageManager]::new()
 $method=[Windows.Management.Deployment.PackageManager].GetMethod('AddPackageByUriAsync')
 if(!$method -or ![Windows.Foundation.Metadata.ApiInformation]::IsPropertyPresent('Windows.Management.Deployment.AddPackageOptions','ExpectedDigests')){throw 'Native API unavailable'}
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $self=Get-Process -Id $PID
 $parent=Get-CimInstance Win32_Process -Filter ('ProcessId='+$ParentId)
 if(!$parent -or $parent.SessionId -le 0 -or $parent.SessionId -ne $self.SessionId -or (Invoke-CimMethod -InputObject $parent -MethodName GetOwnerSid).Sid -cne $sid -or [Math]::Abs(([DateTimeOffset]$parent.CreationDate).ToUnixTimeMilliseconds()-$ParentBirth) -gt 3000){throw 'Native parent actor/incarnation mismatch'}
 if(-not ('SemantierNativeUpdateFacts' -as [type])){Add-Type -TypeDefinition @'
using System;using System.Runtime.InteropServices;using Microsoft.Win32.SafeHandles;
public static class SemantierNativeUpdateFacts {
 [StructLayout(LayoutKind.Sequential)] public struct Info {public uint attr;public System.Runtime.InteropServices.ComTypes.FILETIME created,accessed,written;public uint volume,high,low,links,indexHigh,indexLow;}
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
 [DllImport("wtsapi32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr s,int id,int info,out IntPtr b,out int n);
 [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr p);
}
'@}
 $buffer=[IntPtr]::Zero;$length=0
 if(![SemantierNativeUpdateFacts]::WTSQuerySessionInformation([IntPtr]::Zero,$self.SessionId,8,[ref]$buffer,[ref]$length)){throw 'Native WTS state unavailable'}
 try{if($length -lt 4 -or [Runtime.InteropServices.Marshal]::ReadInt32($buffer) -ne 0){throw 'Native interactive session is not active'}}finally{[SemantierNativeUpdateFacts]::WTSFreeMemory($buffer)}
 $sessionAccount=@()
 foreach($kind in @(7,5)){
  $buffer=[IntPtr]::Zero;$length=0
  if(![SemantierNativeUpdateFacts]::WTSQuerySessionInformation([IntPtr]::Zero,$self.SessionId,$kind,[ref]$buffer,[ref]$length)){throw 'Native WTS identity unavailable'}
  try{if($length -lt 2){throw 'Empty native WTS identity'};$sessionAccount+=[Runtime.InteropServices.Marshal]::PtrToStringUni($buffer)}finally{[SemantierNativeUpdateFacts]::WTSFreeMemory($buffer)}
 }
 if(([Security.Principal.NTAccount]::new(($sessionAccount -join '\'))).Translate([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native signed-in session identity mismatch'}
 $manifest=Get-AppxPackageManifest -Package $Package.PackageFullName
 $application=@($manifest.Package.Applications.Application|Where-Object{$_.Id -ceq $request.applicationId})
 if($application.Count -ne 1){throw 'Native application ID ambiguous'}
 $expected=[IO.File]::OpenRead((Join-Path $Package.InstallLocation $application[0].Executable));$actual=[IO.File]::OpenRead($parent.ExecutablePath)
 try{
  $left=New-Object SemantierNativeUpdateFacts+Info;$right=New-Object SemantierNativeUpdateFacts+Info
  if(![SemantierNativeUpdateFacts]::GetFileInformationByHandle($expected.SafeFileHandle,[ref]$left) -or ![SemantierNativeUpdateFacts]::GetFileInformationByHandle($actual.SafeFileHandle,[ref]$right) -or $left.volume -ne $right.volume -or $left.indexHigh -ne $right.indexHigh -or $left.indexLow -ne $right.indexLow){throw 'Native parent executable physical identity mismatch'}
 }finally{$expected.Dispose();$actual.Dispose()}
 $current=$manager.FindPackageForUser($sid,$request.fullName)
 if(!$current){throw 'Native current-user package unavailable'}
 $association=$current.GetAppInstallerInfo()
 if($association -and $association.Uri){throw 'Native association migration is not qualified'}
 $rows=@(Read-NativeUpdateGraph $request)
 $directory=[IO.Path]::GetFullPath($request.journalDirectory)
 if($request.targetBuildId -cnotmatch '^[a-f0-9]{32}$' -or $request.targetCommit -cnotmatch '^[a-f0-9]{40}$' -or @($rows|Where-Object{($_.key -split '/')[2] -cne $request.targetBuildId}).Count){throw 'Native target source/build binding mismatch'}
 if([IO.Path]::GetFileName($directory) -cnotmatch ('^native-'+$request.manifestSha256+'-[a-f0-9]{32}$')){throw 'Native journal binding mismatch'}
 $cursor=Get-Item -LiteralPath (Split-Path -Parent $directory) -Force
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native journal reparse path refused'};$cursor=$cursor.Parent}
 foreach($other in Get-ChildItem -LiteralPath (Split-Path -Parent $directory) -Directory -Filter 'native-*'){
  if($other.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native transaction reparse directory refused'}
  if((Test-Path -LiteralPath (Join-Path $other.FullName 'intent.json')) -and !(Test-Path -LiteralPath (Join-Path $other.FullName 'completed.json')) -and !(Test-Path -LiteralPath (Join-Path $other.FullName 'reconciled.json'))){
   $reconciliation=Get-NativeUpdateReconciliation $other.FullName
   if($reconciliation.status -cnotin @('TARGET_GRAPH_REGISTERED','NOT_STARTED')){throw 'Native unresolved transaction exists; preserve its UNKNOWN hold'}
   Write-NativeUpdateReceipt @{directory=$other.FullName} 'reconciled.json' $reconciliation
  }
 }
 if(Test-Path -LiteralPath $directory){throw 'Native transaction already exists; reconcile before retry'}
 $null=[IO.Directory]::CreateDirectory($directory)
 $acl=[Security.AccessControl.DirectorySecurity]::new();$acl.SetAccessRuleProtection($true,$false);$acl.SetOwner([Security.Principal.SecurityIdentifier]::new($sid))
 foreach($principal in @($sid,'S-1-5-18','S-1-5-32-544')){$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($principal),'FullControl','ContainerInherit,ObjectInherit','None','Allow'))}
 [IO.Directory]::SetAccessControl($directory,$acl)
 $request|Add-Member -NotePropertyName waiterPid -NotePropertyValue $PID
 $request|Add-Member -NotePropertyName waiterBirth -NotePropertyValue ([DateTimeOffset]$self.StartTime).ToUnixTimeMilliseconds()
 $request|Add-Member -NotePropertyName sourceSha256 -NotePropertyValue (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
 $request|Add-Member -NotePropertyName authorizedSid -NotePropertyValue $sid
 $request|Add-Member -NotePropertyName session -NotePropertyValue $self.SessionId
 $intent=Join-Path $directory 'intent.json'
 $bytes=[Text.Encoding]::UTF8.GetBytes(($request|ConvertTo-Json -Depth 16 -Compress))
 $file=[IO.File]::Open($intent,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
 return @{request=$request;manager=$manager;method=$method;directory=$directory;rows=$rows}
}

function Invoke-NativePackageUpdate($Context) {
 $request=$Context.request
 $before=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
 if($before.Count -ne 1 -or $before[0].PackageFullName -cne $request.fullName -or $before[0].Publisher -cne $request.publisher){throw 'Native installed baseline changed before deployment'}
 $options=[Windows.Management.Deployment.AddPackageOptions]::new()
 $uris=@($Context.rows|ForEach-Object{$request.base.TrimEnd('/')+'/'+$_.key})
 $digests=@($Context.rows|ForEach-Object{[string]$_.nativeDigest})
 $null=Set-NativeExpectedDigests $options $uris $digests
 $asTask=@([System.WindowsRuntimeSystemExtensions].GetMethods()|Where-Object{$_.Name -ceq 'AsTask' -and $_.IsGenericMethodDefinition -and $_.GetGenericArguments().Count -eq 2 -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -ceq 'IAsyncOperationWithProgress`2'})
 if($asTask.Count -ne 1){throw 'Native asynchronous binding unavailable'}
 $binding=$asTask[0].MakeGenericMethod($Context.method.ReturnType.GetGenericArguments())
 $start=[DateTime]::UtcNow.ToString('o')
 Write-NativeUpdateReceipt $Context 'deployment-intent.json' @{schema=1;status='DEPLOYMENT_INTENT';producedAt=$start;manifestSha256=$request.manifestSha256;uri=($request.base.TrimEnd('/')+'/'+$request.graph.root);sid=$request.authorizedSid;session=$request.session}
 $operation=$Context.manager.AddPackageByUriAsync([Uri]($request.base.TrimEnd('/')+'/'+$request.graph.root),$options)
 $task=$binding.Invoke($null,@($operation))
 # A timeout is UNKNOWN: Windows may complete later. Never cancel/redeploy/delete intent.
 if(!$task.Wait([TimeSpan]::FromMinutes(25))){throw 'Native deployment UNKNOWN; retain intent and reconcile OS state'}
 $result=$task.Result
 if($result.ExtendedErrorCode){throw $result.ExtendedErrorCode}
 $receipt=@{schema=1;startedAt=$start;producedAt=[DateTime]::UtcNow.ToString('o');status='OS_SUCCESS';activityId=[string]$result.ActivityId;manifestSha256=$request.manifestSha256}
 Write-NativeUpdateReceipt $Context 'deployment.json' $receipt
}

function Test-NativeUpdateInstalled($Context, $Package) {
 $r=$Context.request
 if($Package.PackageFamilyName -cne $r.familyName -or $Package.Publisher -cne $r.publisher -or $Package.Version.ToString() -cne $r.version -or $Package.Architecture.ToString() -cne 'X64' -or $Package.Status.ToString() -cne 'Ok'){return $false}
 $stampPath=Join-Path $Package.InstallLocation 'app\resources\install-stamp.json'
 $stamp=Get-Content -LiteralPath $stampPath -Raw -Encoding UTF8|ConvertFrom-Json
 if($stamp.commit -cne $r.targetCommit -or $stamp.channelBuild.buildId -cne $r.targetBuildId){return $false}
 $assets=@(Get-AppxPackage -PackageTypeFilter Resource|Where-Object{$_.PackageFamilyName -ceq $r.familyName -and $_.Version.ToString() -ceq $r.version})
 $required=@($Context.rows|Where-Object{$_.kind -ceq 'asset-package'})
 if($assets.Count -ne $required.Count){return $false}
 foreach($row in $required){$matching=@($assets|Where-Object{$_.ResourceId -ceq $row.resourceId -and $_.Publisher -ceq $r.publisher -and $_.Architecture.ToString() -ceq 'Neutral' -and $_.Status.ToString() -ceq 'Ok'});if($matching.Count -ne 1){return $false}}
 return $true
}

function Read-NativeUpdateGraph($request) {
 $base=[Uri]$request.base
 if(!$base.IsAbsoluteUri -or $base.UserInfo -or $base.Query -or $base.Fragment -or $request.base -match '[%\\\s;]' -or ($base.Scheme -cne 'https' -and !($base.Scheme -ceq 'http' -and $base.Host -cin @('127.0.0.1','localhost','[::1]')))){throw 'Native origin invalid'}
 $graph=$request.graph;$rows=@($graph.objects);$byKey=@{};$main=0;$bundle=0;$identities=@{}
 if($graph.schema -ne 1 -or $graph.transport -cne 'windows-package-uri' -or $graph.payloadInventorySha256 -cnotmatch '^[a-f0-9]{64}$' -or !$rows.Count -or $rows.Count -gt 64){throw 'Native graph invalid'}
 foreach($row in $rows){
  if($row.key -cnotmatch '^releases/channel-builds/[a-f0-9]{32}/[A-Za-z0-9_./+-]+\.(msix|msixbundle)$' -or $row.key -match '(^|/)\.{1,2}(/|$)' -or $byKey.ContainsKey($row.key) -or ($row.size -isnot [int] -and $row.size -isnot [long]) -or $row.size -le 0 -or $row.size -gt 480000000 -or $row.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $row.identity -cne $request.identity -or $row.publisher -cne $request.publisher -or $row.version -cne $request.version -or $row.kind -cnotin @('main-bundle','main-package','asset-package')){throw 'Native object binding mismatch'}
  $digest=[Convert]::FromBase64String($row.nativeDigest)
  if($digest.Length -ne 32 -or [Convert]::ToBase64String($digest) -cne $row.nativeDigest){throw 'Native digest invalid'}
  if($row.kind -ceq 'main-package'){$main++;if($row.arch -cne 'x64' -or @($row.references).Count){throw 'Native main invalid'}}
  elseif($row.arch -cne 'neutral'){throw 'Native neutral architecture invalid'}
  if($row.kind -ceq 'main-bundle'){$bundle++}
  if($row.kind -ceq 'asset-package' -and ($row.resourceId -cnotmatch '^runtime[0-9]{4}$' -or ($row.allowExecution -isnot [bool] -or $row.allowExecution -ne $true) -or @($row.references).Count)){throw 'Native asset invalid'}
  $id=$row.kind+'/'+$row.resourceId
  if($identities.ContainsKey($id)){throw 'Native identity ambiguous'}
  $identities[$id]=$true;$byKey[$row.key]=$row
 }
 if($main -ne 1 -or $bundle -ne 1 -or !$byKey.ContainsKey($graph.root) -or $byKey[$graph.root].kind -cne 'main-bundle'){throw 'Native flat bundle root invalid'}
 $references=@($byKey[$graph.root].references)
 if($references.Count -ne ($rows.Count-1) -or @($references|Select-Object -Unique).Count -ne $references.Count -or @($references|Where-Object{!$byKey.ContainsKey($_) -or $_ -ceq $graph.root}).Count){throw 'Native graph incomplete'}
 $prefix=($graph.root -split '/')[2]
 if(@($rows|Where-Object{($_.key -split '/')[2] -cne $prefix}).Count){throw 'Native graph build mismatch'}
 return $rows
}

function Write-NativeUpdateReceipt($Context, [string]$Name, $Value) {
 $bytes=[Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 16 -Compress))
 $stream=[IO.File]::Open((Join-Path $Context.directory $Name),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}

# Read-only OS reconciliation; old intent/timeout history is never rewritten.
function Get-NativeUpdateReconciliation([string]$Directory) {
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 if((Get-Acl -LiteralPath $Directory).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native journal owner mismatch'}
 $intent=Join-Path $Directory 'intent.json'
 if((Get-Item -LiteralPath $intent).Length -gt 4194304){throw 'Native journal metadata oversized'}
 $request=Get-Content -LiteralPath $intent -Raw -Encoding UTF8|ConvertFrom-Json
 $result=@{schema=1;status='UNKNOWN';producedAt=[DateTime]::UtcNow.ToString('o');manifestSha256=$request.manifestSha256}
 if($request.authorizedSid -cne $sid){throw 'Native journal actor mismatch'}
 $rows=@(Read-NativeUpdateGraph $request)
 $packages=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
 if($packages.Count -ne 1){return $result}
 if(Test-NativeUpdateInstalled @{request=$request;rows=$rows} $packages[0]){
  $result.status='TARGET_GRAPH_REGISTERED';$result.fullName=$packages[0].PackageFullName
  return $result
 }
 if($packages[0].PackageFullName -cne $request.fullName -or $packages[0].Publisher -cne $request.publisher -or (Test-Path -LiteralPath (Join-Path $Directory 'deployment-intent.json'))){return $result}
 if($request.sourceSha256 -cne (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()){return $result}
 $waiter=Get-Process -Id $request.waiterPid -ErrorAction SilentlyContinue
 if($waiter -and [Math]::Abs(([DateTimeOffset]$waiter.StartTime).ToUnixTimeMilliseconds()-$request.waiterBirth) -le 3000){return $result}
 $result.status='NOT_STARTED';$result.fullName=$packages[0].PackageFullName
 return $result
}
