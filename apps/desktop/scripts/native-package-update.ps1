# Production native branch of the existing external relaunch waiter.
# Windows owns all package transfers. No package bytes are downloaded here.
. (Join-Path $PSScriptRoot 'native-package-digest.ps1')
function Invoke-NativeWaiterBreakaway([string]$Script, [string[]]$Arguments) {
 # MSIX's job permits explicit breakaway, not silent inheritance. Keep a real
 # hidden console for PowerShell 5.1; DETACHED_PROCESS can exit before -File.
 if(-not ('SemantierNativeWaiterProcess' -as [type])){Add-Type -TypeDefinition @'
using System;using System.ComponentModel;using System.IO;using System.Text;using System.Runtime.InteropServices;
public static class SemantierNativeWaiterProcess {
 [StructLayout(LayoutKind.Sequential,CharSet=CharSet.Unicode)] struct Startup {
  public uint cb;public IntPtr reserved,desktop,title;
  public uint x,y,xSize,ySize,xChars,yChars,fill,flags;
  public ushort show,reservedSize;public IntPtr reservedData,input,output,error;
 }
 [StructLayout(LayoutKind.Sequential)] struct Created {public IntPtr process,thread;public uint pid,tid;}
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool CreateProcessW(string application,StringBuilder command,IntPtr processSecurity,IntPtr threadSecurity,bool inherit,uint flags,IntPtr environment,string directory,ref Startup startup,out Created created);
 [DllImport("kernel32.dll",SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle,uint milliseconds);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process,out uint code);
 [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
 static string Quote(string value) {
  if(value==null || value.IndexOfAny(new char[]{'\0','\r','\n'})>=0)throw new ArgumentException("Invalid native waiter argument");
  StringBuilder result=new StringBuilder("\"");int slashes=0;
  foreach(char c in value){
   if(c=='\\'){slashes++;continue;}
   result.Append('\\',c=='"'?slashes*2+1:slashes);result.Append(c);slashes=0;
  }
  return result.Append('\\',slashes*2).Append('"').ToString();
 }
 public static int Run(string script,string[] arguments) {
  string exe=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),@"WindowsPowerShell\v1.0\powershell.exe");
  StringBuilder command=new StringBuilder(Quote(exe));
  foreach(string value in new string[]{"-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",script})command.Append(' ').Append(Quote(value));
  foreach(string value in arguments)command.Append(' ').Append(Quote(value));
  if(command.Length>32000)throw new ArgumentException("Native waiter command exceeds limit");
  Startup startup=new Startup();startup.cb=(uint)Marshal.SizeOf(typeof(Startup));startup.flags=1;startup.show=0;
  Created child;
  // Same token/environment; no inherited handles or privilege changes. A job
  // that forbids breakaway rejects creation rather than invoking a fallback.
  if(!CreateProcessW(exe,command,IntPtr.Zero,IntPtr.Zero,false,0x01000010,IntPtr.Zero,Path.GetTempPath(),ref startup,out child))throw new Win32Exception(Marshal.GetLastWin32Error(),"Native waiter breakaway refused");
  CloseHandle(child.thread);
  try {
   if(WaitForSingleObject(child.process,0xffffffff)!=0)throw new Win32Exception(Marshal.GetLastWin32Error(),"Native waiter wait failed");
   uint code;if(!GetExitCodeProcess(child.process,out code))throw new Win32Exception(Marshal.GetLastWin32Error());
   return unchecked((int)code);
  } finally {CloseHandle(child.process);}
 }
}
'@}
 return [SemantierNativeWaiterProcess]::Run($Script,$Arguments)
}
function New-NativeUpdateJournalDirectory([string]$directory,[string]$sid) {
 if(Test-Path -LiteralPath $directory){throw 'Native transaction already exists; reconcile before retry'}
 $null=[IO.Directory]::CreateDirectory($directory)
 # The creator already owns the directory. Setting that owner again requires WRITE_OWNER,
 # which ordinary Modify access does not grant; ownership itself permits the private DACL.
 if([IO.Directory]::GetAccessControl($directory,[Security.AccessControl.AccessControlSections]::Owner).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native journal creator ownership mismatch'}
 $acl=[Security.AccessControl.DirectorySecurity]::new();$acl.SetAccessRuleProtection($true,$false)
 foreach($principal in @($sid,'S-1-5-18','S-1-5-32-544')){$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($principal),'FullControl','ContainerInherit,ObjectInherit','None','Allow'))}
 [IO.Directory]::SetAccessControl($directory,$acl)
}
function Initialize-NativeUpdateFacts {
 if(-not ('SemantierNativeUpdateFacts' -as [type])){Add-Type -TypeDefinition @'
using System;using System.Runtime.InteropServices;using Microsoft.Win32.SafeHandles;
public static class SemantierNativeUpdateFacts {
 [StructLayout(LayoutKind.Sequential)] public struct Info {public uint attr;public System.Runtime.InteropServices.ComTypes.FILETIME created,accessed,written;public uint volume,high,low,links,indexHigh,indexLow;}
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetFileInformationByHandle(SafeFileHandle h,out Info i);
 [DllImport("wtsapi32.dll",CharSet=CharSet.Unicode,SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr s,int id,int info,out IntPtr b,out int n);
 [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr p);
}
'@}
}

function Assert-NativeContinuityDirectory([string]$Directory,[string]$Sid) {
 if(!$Directory -or ![IO.Path]::IsPathRooted($Directory) -or $Directory -match '["\x00\r\n]' -or $Directory.StartsWith('\\')){throw 'Native continuity requires an absolute local directory'}
 $item=Get-Item -LiteralPath $Directory -Force -ErrorAction Stop
 if(!$item.PSIsContainer -or $item.FullName.TrimEnd('\') -ine $Directory.TrimEnd('\')){throw 'Native continuity directory is not canonical'}
 if((Get-Acl -LiteralPath $item.FullName).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $Sid){throw 'Native continuity directory owner mismatch'}
 $allowed=@($Sid,'S-1-5-18','S-1-5-32-544')
 foreach($rule in (Get-Acl -LiteralPath $item.FullName).GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])){
  if($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -cnotin $allowed -and ($rule.FileSystemRights -band 0xD0156)){throw 'Native continuity directory has a foreign writer'}
 }
 $cursor=$item
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native continuity reparse path refused'};$cursor=$cursor.Parent}
 return $item.FullName
}

function Read-NativeContinuityJson([string]$File,[string]$Sid) {
 $item=Get-Item -LiteralPath $File -Force -ErrorAction Stop
 if($item.PSIsContainer -or $item.Attributes -band [IO.FileAttributes]::ReparsePoint -or $item.Length -gt 4194304 -or
    (Get-Acl -LiteralPath $File).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $Sid){throw 'Native continuity metadata owner/path/size invalid'}
 return (Get-Content -LiteralPath $File -Raw -Encoding UTF8|ConvertFrom-Json)
}

function Assert-NativeUpdateContinuity($Request,[string]$Sid) {
 $backendRoot=Assert-NativeContinuityDirectory $Request.backendHome $Sid
 $data=Assert-NativeContinuityDirectory $Request.userDataDirectory $Sid
 if($Request.activeProfile -cnotmatch '^[a-z0-9][a-z0-9_-]{0,63}$'){throw 'Native continuity profile invalid'}
 $scope=if($Request.activeProfile -ceq 'default'){$backendRoot}else{Join-Path $backendRoot ('profiles\'+$Request.activeProfile)}
 if($Request.credentialScope -ine $scope){throw 'Native continuity credential scope mismatch'}
 $null=Assert-NativeContinuityDirectory $scope $Sid
 if([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Request.journalDirectory)) -ine (Join-Path $data 'updates')){throw 'Native continuity journal/userData mismatch'}
 return $backendRoot
}

# Read-only target-side compatibility for the already-shipped waiter. Never run
# a historical script, infer another user's home, or touch credential contents.
function Get-NativeLegacyRelaunchHome([string]$UserData,$Build,[int]$ProcessId,[string]$RegistryHome) {
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $data=Assert-NativeContinuityDirectory $UserData $sid
 $updates=Join-Path $data 'updates'
 if(!(Test-Path -LiteralPath $updates)){return $null}
 $null=Assert-NativeContinuityDirectory $updates $sid
 $matches=@()
 $journals=@(Get-ChildItem -LiteralPath $updates -Directory -Filter 'native-*')
 if($journals.Count -gt 64){throw 'Native continuity journal inventory exceeds limit'}
 foreach($journal in $journals){
  $intent=Join-Path $journal.FullName 'intent.json'
  if(!(Test-Path -LiteralPath $intent)){continue}
  $null=Assert-NativeContinuityDirectory $journal.FullName $sid
  $r=Read-NativeContinuityJson $intent $sid
  if($r.targetBuildId -cne $Build.buildId -or $r.targetCommit -cne $Build.commit -or $r.version -cne $Build.windowsVersion){continue}
  $matches+=@{request=$r;directory=$journal.FullName}
 }
 if(!$matches.Count){return $null}
 if($matches.Count -ne 1){throw 'Native continuity journal ambiguous; relaunch with original HERMES_HOME and --profile'}
 $context=$matches[0];$r=$context.request
 if($r.schema -notin @(1,2) -or ($r.schema -eq 2 -and !($r.PSObject.Properties.Name -contains 'backendHome')) -or $r.authorizedSid -cne $sid -or $r.userDataDirectory -ine $data -or $r.journalDirectory -ine $context.directory -or
    $r.identity -cne $Build.identity.msixAppIdWithOrg -or $r.manifestSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    [IO.Path]::GetFileName($context.directory) -cnotmatch ('^native-'+$r.manifestSha256+'-[a-f0-9]{32}$')){throw 'Native continuity journal binding mismatch'}
 $context.rows=@(Read-NativeUpdateGraph $r)
 $packages=@(Get-AppxPackage -Name $r.identity|Where-Object{$_.Name -ceq $r.identity})
 if($packages.Count -ne 1 -or !(Test-NativeUpdateInstalled $context $packages[0]) -or $packages[0].SignatureKind.ToString() -cne 'Developer' -or $packages[0].IsDevelopmentMode){throw 'Native continuity target registration mismatch'}
 $package=$packages[0]
 $row=Get-CimInstance Win32_Process -Filter ('ProcessId='+$ProcessId)
 if(!$row -or $row.SessionId -ne $r.session -or (Invoke-CimMethod -InputObject $row -MethodName GetOwnerSid).Sid -cne $sid){throw 'Native continuity successor actor/session mismatch'}
 $launchPath=Join-Path $context.directory 'relaunch-intent.json'
 $launch=Read-NativeContinuityJson $launchPath $sid
 $manifest=Get-AppxPackageManifest -Package $package.PackageFullName
 $apps=@($manifest.Package.Applications.Application|Where-Object{$_.Id -ceq $r.applicationId})
 if($apps.Count -ne 1 -or $row.ExecutablePath -ine (Join-Path $package.InstallLocation $apps[0].Executable) -or
    $launch.schema -ne 1 -or $launch.manifestSha256 -cne $r.manifestSha256 -or $launch.fullName -cne $package.PackageFullName -or $launch.applicationId -cne $r.applicationId){throw 'Native continuity successor executable/launch mismatch'}
 Initialize-NativeUpdateFacts
 $executable=[IO.File]::OpenRead($row.ExecutablePath)
 try{if((Get-NativeUpdateFileId $executable) -cne $launch.exeId){throw 'Native continuity physical executable identity mismatch'}}finally{$executable.Dispose()}
 $birth=[DateTimeOffset]$row.CreationDate
 $started=ConvertTo-NativeWaiterBirth $launch.producedAt
 if($birth -lt $started -or ($birth-$started).TotalSeconds -gt 30){throw 'Native continuity stale relaunch; relaunch with original HERMES_HOME and --profile'}
 # Parent PID plus birth binds a still-running waiter. Once it exits, its
 # append-only completed receipt must identify this exact successor incarnation.
 $parent=Get-Process -Id $r.waiterPid -ErrorAction SilentlyContinue
 $live=$parent -and $row.ParentProcessId -eq $r.waiterPid -and ([DateTimeOffset]$parent.StartTime).ToUniversalTime() -eq (ConvertTo-NativeWaiterBirth $r.waiterBirth).ToUniversalTime()
 if(!$live){
  $completed=Read-NativeContinuityJson (Join-Path $context.directory 'completed.json') $sid
  if($completed.status -cne 'TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED' -or $completed.pid -ne $ProcessId -or
     [Math]::Abs(((ConvertTo-NativeWaiterBirth $completed.birth).ToUniversalTime()-$birth.ToUniversalTime()).Ticks) -ge 10 -or $completed.sid -cne $sid -or
     $completed.session -ne $row.SessionId -or $completed.fullName -cne $package.PackageFullName -or $completed.manifestSha256 -cne $r.manifestSha256){throw 'Native continuity successor incarnation unproved'}
 }
 if($r.PSObject.Properties.Name -contains 'backendHome'){
  return (Assert-NativeUpdateContinuity $r $sid)
 }
 $suffix=[string]$Build.bundleEnv.HERMES_DATA_DIR_SUFFIX
 if($suffix -cnotmatch '^(?:-[a-z0-9][a-z0-9_-]{0,63})?$' -or [string]$env:HERMES_DATA_DIR_SUFFIX -cne $suffix -or $Build.bundleEnv.HERMES_HOME){throw 'Native legacy bundle home unproved; relaunch with original HERMES_HOME and --profile'}
 $expectedData=Join-Path ([Environment]::GetFolderPath('ApplicationData')) ($Build.identity.appNamePascal+$suffix)
 if($data -ine $expectedData){throw 'Native legacy isolated/custom home unproved; relaunch with original HERMES_HOME and --profile'}
 if($RegistryHome){return (Assert-NativeContinuityDirectory $RegistryHome $sid)}
 $backendRoot=(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'hermes')+$suffix
 if(!$suffix -and !(Test-Path -LiteralPath $backendRoot)){$backendRoot=Join-Path ([Environment]::GetFolderPath('UserProfile')) '.hermes'}
 return (Assert-NativeContinuityDirectory $backendRoot $sid)
}

function Initialize-NativeUpdate($RequestPath, [int]$ParentId, [long]$ParentBirth, $Package) {
 $ErrorActionPreference='Stop'
 if((Get-Item -LiteralPath $RequestPath).Length -gt 4194304){throw 'Native request exceeds metadata limit'}
 $request=Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8|ConvertFrom-Json
 if($request.schema -notin @(1,2) -or $request.identity -cne $Package.Name -or $request.fullName -cne $Package.PackageFullName -or $request.familyName -cne $Package.PackageFamilyName -or $request.publisher -cne $Package.Publisher -or $request.currentVersion -cne $Package.Version.ToString() -or [Version]$request.version -le $Package.Version -or $request.manifestSha256 -cnotmatch '^[a-f0-9]{64}$' -or $request.graphSha256 -cnotmatch '^[a-f0-9]{64}$'){throw 'Native current package binding mismatch'}
 if($Package.Status.ToString() -cne 'Ok' -or $Package.SignatureKind.ToString() -cne 'Developer' -or $Package.IsDevelopmentMode -or $Package.NonRemovable){throw 'Native current package trust mismatch'}
 $cpus=@(Get-CimInstance Win32_Processor)
 if(!$cpus.Count -or @($cpus|Where-Object{$_.Architecture -ne 9}).Count){throw 'Actual native x64 host required'}
 Add-Type -AssemblyName System.Runtime.WindowsRuntime
 $null=[Windows.Management.Deployment.PackageManager,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $null=[Windows.Management.Deployment.AddPackageOptions,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $manager=[Windows.Management.Deployment.PackageManager]::new()
 $method=[Windows.Management.Deployment.PackageManager].GetMethod('AddPackageByUriAsync')
 if(!$method -or ![Windows.Foundation.Metadata.ApiInformation]::IsPropertyPresent('Windows.Management.Deployment.AddPackageOptions','ExpectedDigests')){throw 'Native API unavailable'}
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $null=Assert-NativeUpdateContinuity $request $sid
 $self=Get-Process -Id $PID
 $parent=Get-CimInstance Win32_Process -Filter ('ProcessId='+$ParentId)
 if(!$parent -or $parent.SessionId -le 0 -or $parent.SessionId -ne $self.SessionId -or (Invoke-CimMethod -InputObject $parent -MethodName GetOwnerSid).Sid -cne $sid -or [Math]::Abs(([DateTimeOffset]$parent.CreationDate).ToUnixTimeMilliseconds()-$ParentBirth) -gt 3000){throw 'Native parent actor/incarnation mismatch'}
 Initialize-NativeUpdateFacts
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
 if($request.targetBuildId -cnotmatch '^(?:[a-f0-9]{32}|(?:0|[1-9][0-9]{0,4})(?:\.(?:0|[1-9][0-9]{0,4})){3})$' -or $request.targetCommit -cnotmatch '^[a-f0-9]{40}$' -or @($rows|Where-Object{($_.key -split '/')[2] -cne $request.targetBuildId}).Count){throw 'Native target source/build binding mismatch'}
 if([IO.Path]::GetFileName($directory) -cnotmatch ('^native-'+$request.manifestSha256+'-[a-f0-9]{32}$')){throw 'Native journal binding mismatch'}
 $cursor=Get-Item -LiteralPath (Split-Path -Parent $directory) -Force
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native journal reparse path refused'};$cursor=$cursor.Parent}
 foreach($other in Get-ChildItem -LiteralPath (Split-Path -Parent $directory) -Directory -Filter 'native-*'){
  if($other.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native transaction reparse directory refused'}
  if((Test-Path -LiteralPath (Join-Path $other.FullName 'intent.json')) -and !(Test-Path -LiteralPath (Join-Path $other.FullName 'completed.json')) -and !(Test-Path -LiteralPath (Join-Path $other.FullName 'reconciled.json'))){
   $reconciliation=Get-NativeUpdateReconciliation $other.FullName
   if($reconciliation.status -cnotin @('TARGET_GRAPH_REGISTERED','NOT_STARTED','BASELINE_INTACT_FAILED')){throw 'Native unresolved transaction exists; preserve its UNKNOWN hold'}
   Write-NativeUpdateReceipt @{directory=$other.FullName} 'reconciled.json' $reconciliation
  }
 }
 New-NativeUpdateJournalDirectory $directory $sid
 $resources=@(Get-AppxPackage -PackageTypeFilter Resource|Where-Object{$_.PackageFamilyName -ceq $request.familyName}|ForEach-Object{$_.PackageFullName}|Sort-Object)
 $request|Add-Member -NotePropertyName baselineResources -NotePropertyValue $resources
 $request|Add-Member -NotePropertyName producedAt -NotePropertyValue ([DateTime]::UtcNow.ToString('o'))
 $request|Add-Member -NotePropertyName waiterPid -NotePropertyValue $PID
 $request|Add-Member -NotePropertyName waiterBirth -NotePropertyValue ([DateTimeOffset]$self.StartTime).ToUniversalTime().ToString('o')
 $request|Add-Member -NotePropertyName sourceSha256 -NotePropertyValue (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
 $request|Add-Member -NotePropertyName reconciliationDirectory -NotePropertyValue $PSScriptRoot
 $request|Add-Member -NotePropertyName digestSourceSha256 -NotePropertyValue (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-digest.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
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
 $packageBase=if($request.schema -eq 2){$request.packageBase}else{$request.base}
 $uris=@($Context.rows|ForEach-Object{$packageBase.TrimEnd('/')+'/'+$_.key})
 $digests=@($Context.rows|ForEach-Object{[string]$_.nativeDigest})
 $start=[DateTime]::UtcNow.ToString('o')
 $uri=$packageBase.TrimEnd('/')+'/'+$request.graph.root
 Write-NativeUpdateReceipt $Context 'deployment-intent.json' @{schema=1;status='DEPLOYMENT_INTENT';producedAt=$start;manifestSha256=$request.manifestSha256;uri=$uri;sid=$request.authorizedSid;session=$request.session}
 $deployment=Invoke-BoundNativePackageUri $uris $digests $uri 25 $true
 $deployment.manifestSha256=$request.manifestSha256
 if($deployment.status -cne 'SUCCESS'){
  Write-NativeUpdateReceipt $Context 'deployment-unresolved.json' $deployment
  throw 'Native deployment unresolved; preserve intent and reconcile OS state'
 }
 $receipt=@{schema=1;startedAt=$deployment.startedAt;producedAt=$deployment.producedAt;status='OS_SUCCESS';activityId=$deployment.activityId;manifestSha256=$request.manifestSha256}
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
 if($request.schema -eq 2){
  if($request.base -cne 'https://cloud.semantier.cn' -or $request.packageBase -cne 'https://assets.semantier.cn'){throw 'Native package origin invalid'}
 }elseif($request.PSObject.Properties['packageBase']){throw 'Legacy native request cannot select package origin'}
 if($request.targetBuildId -match '\.' -and ($request.targetBuildId -cne $request.version -or @($request.targetBuildId.Split('.')|Where-Object{[int]$_ -gt 65535}).Count)){throw 'Version directory does not match native target'}
 $graph=$request.graph;$rows=@($graph.objects);$byKey=@{};$main=0;$bundle=0;$identities=@{}
 $single=$rows.Count -eq 1 -and $rows[0].kind -ceq 'main-package'
 $maximum=if($single){68719476736L}else{480000000L}
 if($graph.schema -ne 1 -or $graph.transport -cne 'windows-package-uri' -or $graph.payloadInventorySha256 -cnotmatch '^[a-f0-9]{64}$' -or !$rows.Count -or $rows.Count -gt 64){throw 'Native graph invalid'}
 foreach($row in $rows){
  if($row.key -cnotmatch '^releases/channel-builds/(?:[a-f0-9]{32}|(?:0|[1-9][0-9]{0,4})(?:\.(?:0|[1-9][0-9]{0,4})){3})/[A-Za-z0-9_./+-]+\.(msix|msixbundle)$' -or $row.key -match '(^|/)\.{1,2}(/|$)' -or $byKey.ContainsKey($row.key) -or ($row.size -isnot [int] -and $row.size -isnot [long]) -or $row.size -le 0 -or $row.size -gt $maximum -or $row.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $row.identity -cne $request.identity -or $row.publisher -cne $request.publisher -or $row.version -cne $request.version -or $row.kind -cnotin @('main-bundle','main-package','asset-package')){throw 'Native object binding mismatch'}
  $digest=[Convert]::FromBase64String($row.nativeDigest)
  if($digest.Length -ne 32 -or [Convert]::ToBase64String($digest) -cne $row.nativeDigest){throw 'Native digest invalid'}
  if($row.kind -ceq 'main-package'){$main++;if($row.arch -cne 'x64' -or @($row.references).Count){throw 'Native main invalid'}}
  elseif($row.arch -cne 'neutral'){throw 'Native neutral architecture invalid'}
  if($row.kind -ceq 'main-bundle'){$bundle++}
  if($row.kind -ceq 'asset-package' -and ($row.resourceId -cnotmatch '^runtime[0-9]{4}$' -or ($row.allowExecution -isnot [bool] -or $row.allowExecution -ne $true) -or @($row.references).Count)){throw 'Native asset invalid'}
  $id=$row.kind+'/'+$(if($row.kind -ceq 'asset-package'){$row.resourceId}else{''})
  if($identities.ContainsKey($id)){throw 'Native identity ambiguous'}
  $identities[$id]=$true;$byKey[$row.key]=$row
 }
 if($main -ne 1 -or !$byKey.ContainsKey($graph.root) -or (!$single -and ($bundle -ne 1 -or $byKey[$graph.root].kind -cne 'main-bundle')) -or ($single -and $byKey[$graph.root].kind -cne 'main-package')){throw 'Native package root invalid'}
 $references=@($byKey[$graph.root].references)
 if($references.Count -ne ($rows.Count-1) -or @($references|Select-Object -Unique).Count -ne $references.Count -or @($references|Where-Object{!$byKey.ContainsKey($_) -or $_ -ceq $graph.root}).Count){throw 'Native graph incomplete'}
 $prefix=($graph.root -split '/')[2]
 if(@($rows|Where-Object{($_.key -split '/')[2] -cne $prefix}).Count){throw 'Native graph build mismatch'}
 if($request.schema -eq 2 -and (!$single -or $prefix -cne $request.targetBuildId -or $graph.root -cne ('releases/channel-builds/'+$request.targetBuildId+'/SemantierAgent.msix'))){throw 'Native OSS build binding mismatch'}
 return $rows
}

function Write-NativeUpdateReceipt($Context, [string]$Name, $Value) {
 $bytes=[Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 16 -Compress))
 $stream=[IO.File]::Open((Join-Path $Context.directory $Name),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}

# Read-only OS reconciliation; old intent/timeout history is never rewritten.
function ConvertTo-NativeWaiterBirth($Value) {
 if($Value -is [DateTime]){
  if($Value.Kind -eq [DateTimeKind]::Unspecified){throw 'Native waiter birth lacks a timezone'}
  return [DateTimeOffset]$Value
 }
 if([string]$Value -notmatch '(?:Z|[+-]\d{2}:\d{2})$'){throw 'Native waiter birth lacks a timezone'}
 return [DateTimeOffset]::Parse([string]$Value)
}

function Get-NativeUpdateReconciliation([string]$Directory) {
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 if((Get-Acl -LiteralPath $Directory).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native journal owner mismatch'}
 $intent=Join-Path $Directory 'intent.json'
 if((Get-Item -LiteralPath $intent).Length -gt 4194304){throw 'Native journal metadata oversized'}
 $request=Get-Content -LiteralPath $intent -Raw -Encoding UTF8|ConvertFrom-Json
 $result=@{schema=1;status='UNKNOWN';producedAt=[DateTime]::UtcNow.ToString('o');manifestSha256=$request.manifestSha256}
 if($request.authorizedSid -cne $sid){throw 'Native journal actor mismatch'}
 if($request.sourceSha256 -cne (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()){return $result}
 $rows=@(Read-NativeUpdateGraph $request)
 $waiter=Get-Process -Id $request.waiterPid -ErrorAction SilentlyContinue
 if($waiter -and ([DateTimeOffset]$waiter.StartTime).ToUniversalTime().ToString('o') -ceq (ConvertTo-NativeWaiterBirth $request.waiterBirth).ToUniversalTime().ToString('o')){return $result}
 $packages=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
 if($packages.Count -ne 1){return $result}
 if(Test-NativeUpdateInstalled @{request=$request;rows=$rows} $packages[0]){
  $result.status='TARGET_GRAPH_REGISTERED';$result.fullName=$packages[0].PackageFullName
  return $result
 }
 if($packages[0].PackageFullName -cne $request.fullName -or $packages[0].Publisher -cne $request.publisher -or $packages[0].PackageFamilyName -cne $request.familyName -or $packages[0].Version.ToString() -cne $request.currentVersion -or $packages[0].Architecture.ToString() -cne 'X64' -or $packages[0].Status.ToString() -cne 'Ok'){return $result}
 $failureFile=Join-Path $Directory 'deployment-unresolved.json'
 if(Test-Path -LiteralPath $failureFile){
  $failure=Get-Content -LiteralPath $failureFile -Raw -Encoding UTF8|ConvertFrom-Json
  $resources=@(Get-AppxPackage -PackageTypeFilter Resource|Where-Object{$_.PackageFamilyName -ceq $request.familyName}|ForEach-Object{$_.PackageFullName}|Sort-Object)
  if($failure.status -ceq 'FAILURE' -and $failure.operationCompleted -eq $true -and $failure.manifestSha256 -ceq $request.manifestSha256 -and ($resources -join '|') -ceq (@($request.baselineResources) -join '|')){
   $result.status='BASELINE_INTACT_FAILED';$result.fullName=$packages[0].PackageFullName;$result.hresult=$failure.hresult
  }
  return $result
 }
 if(Test-Path -LiteralPath (Join-Path $Directory 'deployment-intent.json')){return $result}
 $result.status='NOT_STARTED';$result.fullName=$packages[0].PackageFullName
 return $result
}


# Explicit operator recovery only. Automatic UNKNOWN/source-pin handling above
# stays unchanged. A newer independently verified single-package registration
# can supersede an old terminal failure without claiming that old update passed.
function Resolve-NativeSupersededFailure {
 [CmdletBinding()]
 param(
  [Parameter(Mandatory=$true)][string]$Directory,
  [Parameter(Mandatory=$true)][string]$ExpectedPackageFullName,
  [Parameter(Mandatory=$true)][string]$ExpectedSignatureSha256,
  [switch]$Apply
 )
 $ErrorActionPreference='Stop'
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $Directory=[IO.Path]::GetFullPath($Directory)
 $cursor=Get-Item -LiteralPath $Directory -Force
 while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native recovery reparse path refused'};$cursor=$cursor.Parent}
 $acl=Get-Acl -LiteralPath $Directory
 if(!$acl.AreAccessRulesProtected -or $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native recovery private owner mismatch'}
 if($ExpectedSignatureSha256 -cnotmatch '^[a-f0-9]{64}$'){throw 'Verified registered signature required'}
 $names=@('intent.json','deployment-intent.json','deployment-unresolved.json','unresolved.json')
 $files=@(Get-ChildItem -LiteralPath $Directory -Force)
 if(@($files|Where-Object{$_.Name -cnotin ($names+@('reconciled.json')) -or $_.PSIsContainer -or ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)}).Count){throw 'Unknown native recovery evidence; preserve hold'}
 $holds=@();$hashes=@{}
 try {
  foreach($name in $names){
   $file=Join-Path $Directory $name
   if((Get-Item -LiteralPath $file).Length -gt 4194304){throw 'Native recovery evidence oversized'}
   $holds+=@([IO.File]::Open($file,'Open','Read','Read'))
   $hashes[$name]=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  $request=Get-Content -LiteralPath (Join-Path $Directory 'intent.json') -Raw -Encoding UTF8|ConvertFrom-Json
  $deployment=Get-Content -LiteralPath (Join-Path $Directory 'deployment-intent.json') -Raw -Encoding UTF8|ConvertFrom-Json
  $failure=Get-Content -LiteralPath (Join-Path $Directory 'deployment-unresolved.json') -Raw -Encoding UTF8|ConvertFrom-Json
  $unresolved=Get-Content -LiteralPath (Join-Path $Directory 'unresolved.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if($request.schema -ne 1 -or $request.authorizedSid -cne $sid -or $deployment.sid -cne $sid -or
    $request.manifestSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    [IO.Path]::GetFullPath($request.journalDirectory) -ine $Directory -or
    [IO.Path]::GetFileName($Directory) -cnotmatch ('^native-'+$request.manifestSha256+'-[a-f0-9]{32}$')){throw 'Native recovery actor/path binding mismatch'}
  foreach($row in @($deployment,$failure,$unresolved)){if($row.manifestSha256 -cne $request.manifestSha256){throw 'Native recovery evidence binding mismatch'}}
  if($deployment.status -cne 'DEPLOYMENT_INTENT' -or $unresolved.status -cne 'UNKNOWN' -or
    $failure.status -cne 'FAILURE' -or $failure.operationCompleted -isnot [bool] -or !$failure.operationCompleted -or
    $failure.hresult -cnotmatch '^[89A-F][A-F0-9]{7}$'){throw 'Native failure is not terminal; preserve UNKNOWN hold'}
  $activity=[guid]::Empty
  if(![guid]::TryParse([string]$failure.activityId,[ref]$activity) -or $activity -eq [guid]::Empty){throw 'Native terminal activity missing'}
  $null=Read-NativeUpdateGraph $request
  $waiterBirth=ConvertTo-NativeWaiterBirth $request.waiterBirth
  $waiter=Get-Process -Id $request.waiterPid -ErrorAction SilentlyContinue
  if($waiter -and ([DateTimeOffset]$waiter.StartTime).ToUniversalTime() -eq $waiterBirth.ToUniversalTime()){throw 'Native historical waiter still alive'}
  foreach($other in @(Get-ChildItem -LiteralPath (Split-Path -Parent $Directory) -Filter 'native-*' -Force)){
   if($other.FullName -ieq $Directory){continue}
   if(!$other.PSIsContainer -or ($other.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
     (!(Test-Path -LiteralPath (Join-Path $other.FullName 'completed.json')) -and !(Test-Path -LiteralPath (Join-Path $other.FullName 'reconciled.json')))){
    throw 'Another native transaction is unresolved; preserve hold'
   }
   foreach($file in @(Get-ChildItem -LiteralPath $other.FullName -Force)){
    if($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Length -gt 4194304){throw 'Other native evidence ambiguous'}
   }
   $otherRequest=Get-Content -LiteralPath (Join-Path $other.FullName 'intent.json') -Raw -Encoding UTF8|ConvertFrom-Json
   if((Get-Acl -LiteralPath $other.FullName).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid -or $otherRequest.authorizedSid -cne $sid){throw 'Other native transaction owner ambiguous'}
   $otherWaiter=Get-Process -Id $otherRequest.waiterPid -ErrorAction SilentlyContinue
   if($otherWaiter -and ([DateTimeOffset]$otherWaiter.StartTime).ToUniversalTime() -eq (ConvertTo-NativeWaiterBirth $otherRequest.waiterBirth).ToUniversalTime()){throw 'Another native waiter is active'}
  }
  $packages=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
  if($packages.Count -ne 1){throw 'Native registered package ambiguous'}
  $package=$packages[0]
  $registeredVersion=[Version]$package.Version
  if($package.PackageFullName -cne $ExpectedPackageFullName -or $package.Publisher -cne $request.publisher -or
    $package.PackageFamilyName -cne $request.familyName -or $package.Status.ToString() -cne 'Ok' -or
    $package.Architecture.ToString() -cne 'X64' -or $package.IsDevelopmentMode -ne $false -or
    $registeredVersion -le [Version]$request.currentVersion -or $registeredVersion -le [Version]$request.version){throw 'Native verified newer registration not established'}
  $resources=@(Get-AppxPackage -PackageTypeFilter Resource|Where-Object{$_.PackageFamilyName -ceq $request.familyName})
  if($resources.Count){throw 'Native recovery requires a complete single-package registration'}
  $signature=Join-Path $package.InstallLocation 'AppxSignature.p7x'
  $holds+=@([IO.File]::Open($signature,'Open','Read','Read'))
  if((Get-FileHash -LiteralPath $signature -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ExpectedSignatureSha256){throw 'Native registered signature binding mismatch'}
  $result=@{schema=1;status='SUPERSEDED_TERMINAL_FAILURE';producedAt=[DateTime]::UtcNow.ToString('o');
   manifestSha256=$request.manifestSha256;authorizedSid=$sid;historicalBaseline=$request.currentVersion;historicalTarget=$request.version;
   historicalSourceSha256=$request.sourceSha256;activityId=$failure.activityId;hresult=$failure.hresult;
   registeredPackage=$package.PackageFullName;registeredVersion=$package.Version.ToString();registeredSignatureSha256=$ExpectedSignatureSha256;
   fileSha256=$hashes;reconcilerSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1')).Hash.ToLowerInvariant();
   installation='NOT_ATTEMPTED';historicalOutcome='FAILURE_PRESERVED';reason='TERMINAL_FAILURE_SUPERSEDED_BY_VERIFIED_NEWER_REGISTRATION'}
  $receiptPath=Join-Path $Directory 'reconciled.json'
  if(Test-Path -LiteralPath $receiptPath){
   if((Get-Item -LiteralPath $receiptPath).Length -gt 4194304){throw 'Native reconciliation oversized'}
   $holds+=@([IO.File]::Open($receiptPath,'Open','Read','Read'))
   $prior=Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8|ConvertFrom-Json
   foreach($field in @('schema','status','manifestSha256','authorizedSid','historicalBaseline','historicalTarget','historicalSourceSha256','activityId','hresult','installation','historicalOutcome','reason')){
    if([string]$prior.$field -cne [string]$result[$field]){throw 'Native prior reconciliation binding mismatch'}
   }
   foreach($name in $names){if($prior.fileSha256.$name -cne $hashes[$name]){throw 'Native reconciled original evidence changed'}}
   $priorVersion=[Version]$prior.registeredVersion
   if($priorVersion -le [Version]$request.version -or $priorVersion -le [Version]$request.currentVersion -or $priorVersion -gt $registeredVersion -or
     $prior.registeredPackage -cne ($request.identity+'_'+$prior.registeredVersion+'_x64__'+($package.PackageFamilyName -split '_')[-1]) -or
     $prior.registeredSignatureSha256 -cnotmatch '^[a-f0-9]{64}$' -or $prior.reconcilerSha256 -cnotmatch '^[a-f0-9]{64}$'){
    throw 'Native prior registration provenance invalid'
   }
   if($priorVersion -eq $registeredVersion -and ($prior.registeredPackage -cne $ExpectedPackageFullName -or $prior.registeredSignatureSha256 -cne $ExpectedSignatureSha256)){throw 'Native prior registered signature changed'}
   return $prior # Historical proof remains immutable after later verified upgrades.
  }
  if($Apply){
   $current=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
   if($current.Count -ne 1 -or $current[0].PackageFullName -cne $ExpectedPackageFullName -or $current[0].Status.ToString() -cne 'Ok'){throw 'Native registration changed during reconciliation'}
   Write-NativeUpdateReceipt @{directory=$Directory} 'reconciled.json' $result
  }
  return $result
 } finally {foreach($hold in $holds){$hold.Dispose()}}
}

function Start-NativeUpdateSuccessor($Context, $Package) {
 $r=$Context.request
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 if($r.authorizedSid -cne $sid -or $r.session -le 0 -or (Get-Process -Id $PID).SessionId -ne $r.session){throw 'Native successor actor/session changed before launch'}
 if($Package.Name -cne $r.identity -or $Package.SignatureKind.ToString() -cne 'Developer' -or $Package.IsDevelopmentMode){throw 'Native successor package trust mismatch'}
 if(!(Test-NativeUpdateInstalled $Context $Package)){throw 'Native successor graph not established'}
 $manifest=Get-AppxPackageManifest -Package $Package.PackageFullName
 $application=@($manifest.Package.Applications.Application|Where-Object{$_.Id -ceq $r.applicationId})
 if($application.Count -ne 1){throw 'Native successor application identity ambiguous'}
 $exe=[IO.Path]::GetFullPath((Join-Path $Package.InstallLocation $application[0].Executable))
 if(!$exe.StartsWith($Package.InstallLocation.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Native successor executable escaped package'}
 $file=[IO.File]::OpenRead($exe)
 try{
  $expected=Get-NativeUpdateFileId $file
  foreach($existing in Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and [IO.Path]::GetFileName($_.ExecutablePath) -ieq [IO.Path]::GetFileName($exe) -and $_.CommandLine -notmatch '(?:^|\s)--type(?:=|\s)'}){
   $probe=[IO.File]::OpenRead($existing.ExecutablePath)
   try{if((Get-NativeUpdateFileId $probe) -ceq $expected){throw 'Unrecorded successor already running; preserve it'}}finally{$probe.Dispose()}
  }
  if(![IO.Path]::IsPathRooted($r.userDataDirectory) -or $r.userDataDirectory -match '["\x00\r\n]'){throw 'Native userData binding invalid'}
  $env:HERMES_DESKTOP_USER_DATA_DIR=$r.userDataDirectory
  $env:HERMES_HOME=Assert-NativeUpdateContinuity $r ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
  $started=[DateTimeOffset]::UtcNow
  Write-NativeUpdateReceipt $Context 'relaunch-intent.json' @{schema=1;producedAt=$started.ToString('o');manifestSha256=$r.manifestSha256;fullName=$Package.PackageFullName;applicationId=$r.applicationId;exeId=$expected}
  $process=Start-Process -FilePath $exe -ArgumentList @('--profile',$r.activeProfile) -WorkingDirectory ([IO.Path]::GetTempPath()) -WindowStyle Hidden -PassThru -ErrorAction Stop
  try{
   $null=$process.Handle
   $row=Get-CimInstance Win32_Process -Filter ('ProcessId='+$process.Id)
   if(!$row -or $row.SessionId -ne $r.session -or (Invoke-CimMethod -InputObject $row -MethodName GetOwnerSid).Sid -cne $r.authorizedSid){throw 'Native successor actor/session unavailable'}
   $birth=[DateTimeOffset]$process.StartTime
   if($birth.ToUniversalTime() -lt $started -or [Math]::Abs(($birth.ToUniversalTime()-([DateTimeOffset]$row.CreationDate).ToUniversalTime()).Ticks) -ge 10){throw 'Native successor incarnation mismatch'}
   $actual=[IO.File]::OpenRead($row.ExecutablePath)
   try{if((Get-NativeUpdateFileId $actual) -cne $expected){throw 'Native successor executable changed'}}finally{$actual.Dispose()}
   return @{schema=1;status='TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED';producedAt=[DateTime]::UtcNow.ToString('o');manifestSha256=$r.manifestSha256;fullName=$Package.PackageFullName;pid=$process.Id;birth=$birth.ToUniversalTime().ToString('o');exe=$row.ExecutablePath;exeId=$expected;sid=$r.authorizedSid;session=$r.session;readiness='NOT_RUN'}
  }finally{$process.Dispose()}
 }finally{$file.Dispose()}
}

function Get-NativeUpdateFileId($Stream) {
 $info=New-Object SemantierNativeUpdateFacts+Info
 if(![SemantierNativeUpdateFacts]::GetFileInformationByHandle($Stream.SafeFileHandle,[ref]$info)){throw 'Native file identity unavailable'}
 return ('{0}:{1}:{2}' -f $info.volume,$info.indexHigh,$info.indexLow)
}

# Read-only qualification of this operator's selected local profile, never customers.
function Get-NativeUpdateJournalReadiness {
 [CmdletBinding()]
 param([Parameter(Mandatory=$true)][string]$Directory,
       [Parameter(Mandatory=$true)][string]$ExpectedPackageFullName,
       [Parameter(Mandatory=$true)][string]$ExpectedSignatureSha256)
 $ErrorActionPreference='Stop'
 $entries=@()
 try {
  if(!(Test-Path -LiteralPath (Split-Path -Parent $Directory) -PathType Container)){throw 'Local qualification profile missing'}
  if(Test-Path -LiteralPath $Directory){
   $cursor=Get-Item -LiteralPath $Directory -Force
   if(!$cursor.PSIsContainer){throw 'Native journal root is not a directory'}
   while($cursor){if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Native journal reparse path refused'};$cursor=$cursor.Parent}
   $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
   foreach($journal in @(Get-ChildItem -LiteralPath $Directory -Filter 'native-*' -Force)){
    if(!$journal.PSIsContainer -or ($journal.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Native journal path invalid'}
    if((Get-Acl -LiteralPath $journal.FullName).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $sid){throw 'Native journal owner mismatch'}
    foreach($file in @(Get-ChildItem -LiteralPath $journal.FullName -Force)){
     if($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Length -gt 4194304){throw 'Native journal evidence invalid'}
    }
    $request=Get-Content -LiteralPath (Join-Path $journal.FullName 'intent.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if($request.schema -ne 1 -or $request.authorizedSid -cne $sid -or
      $request.manifestSha256 -cnotmatch '^[a-f0-9]{64}$' -or $journal.Name -cnotmatch ('^native-'+$request.manifestSha256+'-[a-f0-9]{32}$') -or
      [IO.Path]::GetFullPath($request.journalDirectory) -ine $journal.FullName){throw 'Native journal actor/path mismatch'}
    $waiter=Get-Process -Id $request.waiterPid -ErrorAction SilentlyContinue
    if($waiter -and ([DateTimeOffset]$waiter.StartTime).ToUniversalTime() -eq (ConvertTo-NativeWaiterBirth $request.waiterBirth).ToUniversalTime()){throw 'Native journal waiter active'}
    $receipt=Join-Path $journal.FullName 'reconciled.json'
    if(Test-Path -LiteralPath $receipt){
     $closed=Get-Content -LiteralPath $receipt -Raw -Encoding UTF8|ConvertFrom-Json
     if($closed.status -ceq 'SUPERSEDED_TERMINAL_FAILURE'){
      $closed=Resolve-NativeSupersededFailure -Directory $journal.FullName -ExpectedPackageFullName $ExpectedPackageFullName -ExpectedSignatureSha256 $ExpectedSignatureSha256
     }else{
      $fresh=Get-NativeUpdateReconciliation $journal.FullName
      if($closed.schema -ne 1 -or $closed.manifestSha256 -cne $request.manifestSha256 -or $closed.status -cnotin @('TARGET_GRAPH_REGISTERED','NOT_STARTED','BASELINE_INTACT_FAILED') -or $fresh.status -cne $closed.status){throw 'Historical reconciliation requires explicit supported proof'}
     }
    }elseif(Test-Path -LiteralPath (Join-Path $journal.FullName 'completed.json')){
     if((Test-Path -LiteralPath (Join-Path $journal.FullName 'unresolved.json')) -or (Test-Path -LiteralPath (Join-Path $journal.FullName 'deployment-unresolved.json'))){throw 'Conflicting native terminal evidence'}
     # A closed success is historical evidence, not a claim that today's script ran.
     $closed=Get-Content -LiteralPath (Join-Path $journal.FullName 'completed.json') -Raw -Encoding UTF8|ConvertFrom-Json
     $deployment=Get-Content -LiteralPath (Join-Path $journal.FullName 'deployment.json') -Raw -Encoding UTF8|ConvertFrom-Json
     $null=Read-NativeUpdateGraph $request
     $registered=@(Get-AppxPackage -Name $request.identity|Where-Object{$_.Name -ceq $request.identity})
     if($registered.Count -ne 1){throw 'Native completed registration ambiguous'}
     $package=$registered[0]
     if($closed.schema -ne 1 -or $closed.status -cne 'TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED' -or $closed.sid -cne $sid -or
       $closed.manifestSha256 -cne $request.manifestSha256 -or $deployment.schema -ne 1 -or $deployment.status -cne 'OS_SUCCESS' -or $deployment.manifestSha256 -cne $request.manifestSha256 -or
       $closed.fullName -cne ($request.identity+'_'+$request.version+'_x64__'+($package.PackageFamilyName -split '_')[-1]) -or
       $package.PackageFullName -cne $ExpectedPackageFullName -or $package.PackageFamilyName -cne $request.familyName -or $package.Publisher -cne $request.publisher -or
       $package.Status.ToString() -cne 'Ok' -or $package.Architecture.ToString() -cne 'X64' -or $package.IsDevelopmentMode -ne $false -or [Version]$package.Version -lt [Version]$request.version -or
       $ExpectedSignatureSha256 -cnotmatch '^[a-f0-9]{64}$' -or (Get-FileHash (Join-Path $package.InstallLocation 'AppxSignature.p7x')).Hash.ToLowerInvariant() -cne $ExpectedSignatureSha256){throw 'Native completed historical proof insufficient'}
    }else{
     # Unknown script revisions and ambiguous deployments remain blocking.
     $closed=Get-NativeUpdateReconciliation $journal.FullName
     if($closed.status -cnotin @('TARGET_GRAPH_REGISTERED','NOT_STARTED','BASELINE_INTACT_FAILED')){throw 'Native journal UNKNOWN; preserve hold'}
    }
    $entries+=@{directory=$journal.FullName;status=$closed.status}
   }
  }
  return @{result='PASS';stage='LOCAL_NATIVE_JOURNAL_READINESS';journals=$entries;readOnly=$true;llmCalls=0}
 }catch{
  return @{result='STOP';stage='LOCAL_NATIVE_JOURNAL_READINESS';journals=$entries;reason=$_.Exception.Message;readOnly=$true;llmCalls=0}
 }
}
