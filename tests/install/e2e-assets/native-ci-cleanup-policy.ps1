# This checks ownership only; callers still need the coordinator's execution
# authority and must preserve registrations on any failed or unknown outcome.
function Assert-CINativeCleanupOwnership($Graph,[string]$InstalledFullName,$State=$null,$Verified=$null,$Package=$null,[string]$ExpectedSid='',[string]$GraphPath='',[string]$SourceCommit='',$HeldRoots=$null,[string]$Out='') {
 if($Graph){
  if(!$State -or !$Verified -or !$Package -or !$ExpectedSid -or !$GraphPath -or !$SourceCommit -or !$HeldRoots -or !$Out -or $State.unknown -ne $false -or !$State.nativeSuccess){throw 'Native CI registration ownership unresolved; preserve package and reconcile'}
  if($SourceCommit -cnotmatch '^[a-f0-9]{40}$'){throw 'Native CI source identity unavailable'}
  $success=$State.nativeSuccess;$result=$success.result
  if(!$result -or $result.status -cne 'SUCCESS' -or $result.hresult -cne '00000000' -or $result.operationCompleted -ne $true -or
     $success.operationId -cnotmatch '^[a-f0-9]{32}$' -or $result.operationId -cne $success.operationId){throw 'Native CI operation outcome unbound'}
  $activity=[Guid]::Empty
  if(![Guid]::TryParse([string]$result.activityId,[ref]$activity) -or $activity -eq [Guid]::Empty){throw 'Native CI activity unavailable'}
  $intentPath=Join-Path $Out ('ci-native-intent-'+$success.operationId+'.json')
  $outcomePath=Join-Path $Out ('ci-native-outcome-'+$success.operationId+'.json')
  $records=@()
  foreach($path in @($intentPath,$outcomePath)){
   $physical=Physical $path;$stream=$null;$reader=$null
   try {
    $stream=[IO.FileStream]::new($physical.handle,[IO.FileAccess]::Read)
    if($stream.Length -le 0 -or $stream.Length -gt 4194304){throw 'Native CI operation record unavailable'}
    $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
    $records+=@($reader.ReadToEnd()|ConvertFrom-Json)
   } finally {if($reader){$reader.Dispose()}elseif($stream){$stream.Dispose()};$physical.handle.Dispose()}
  }
  $intent=$records[0];$outcome=$records[1]
  $self=Get-Process -Id $PID
  if($intent.schema -ne 1 -or $outcome.schema -ne 1 -or $intent.operationId -cne $success.operationId -or
     $outcome.operationId -cne $success.operationId -or $intent.operation -cne 'positive' -or
     $intent.sourceCommit -cne $SourceCommit -or $intent.graphSha256 -cne $success.graphSha256 -or
     $intent.actor.sid -cne $ExpectedSid -or $intent.actor.session_sid -cne $ExpectedSid -or $intent.actor.session -ne $success.session -or $intent.actor.wts_state -ne 0 -or
     $intent.actorPid -ne $PID -or $intent.actorBirth -cne $self.StartTime.ToUniversalTime().ToString('o') -or
     $intent.actorExecutable -ine $self.Path -or $intent.root -cne $success.rootUri -or
     $outcome.result.operationId -cne $success.operationId -or $outcome.result.status -cne 'SUCCESS' -or
     $outcome.result.hresult -cne '00000000' -or $outcome.result.operationCompleted -ne $true -or
     $outcome.result.activityId -cne $result.activityId){throw 'Native CI durable operation evidence mismatch'}
  $context=Assert-CISession
  $graphPhysical=Physical $GraphPath;$graphStream=$null;$memory=$null
  try {
   $graphStream=[IO.FileStream]::new($graphPhysical.handle,[IO.FileAccess]::Read)
   if($graphStream.Length -le 0 -or $graphStream.Length -gt 4194304){throw 'Native CI graph bytes unavailable'}
   $memory=[IO.MemoryStream]::new();$graphStream.CopyTo($memory);$bytes=$memory.ToArray()
  } finally {if($memory){$memory.Dispose()};if($graphStream){$graphStream.Dispose()};$graphPhysical.handle.Dispose()}
  $sha=[Security.Cryptography.SHA256]::Create()
  try {$hash=([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()} finally {$sha.Dispose()}
  $recorded=[Text.Encoding]::UTF8.GetString($bytes).TrimStart([char]0xFEFF)|ConvertFrom-Json
  if(($recorded|ConvertTo-Json -Depth 32 -Compress) -cne ($Graph|ConvertTo-Json -Depth 32 -Compress)){throw 'Native CI graph object differs from bound bytes'}
  if($success.rootUri -cnotmatch '^http://127\.0\.0\.1:[1-9][0-9]{0,4}/'){throw 'Native CI origin mismatch'}
  $origin=([Uri]$success.rootUri).GetLeftPart([UriPartial]::Authority)
  if($success.rootUri -cne ($origin+'/'+$Graph.root)){throw 'Native CI root URI mismatch'}
  $rows=@($Graph.objects);$uris=@($success.uris);$digests=@($success.digests);$intentUris=@($intent.uris);$intentDigests=@($intent.digests)
  if(!$rows.Count -or $uris.Count -ne $rows.Count -or $digests.Count -ne $rows.Count -or $intentUris.Count -ne $rows.Count -or $intentDigests.Count -ne $rows.Count){throw 'Native CI URI graph incomplete'}
  for($i=0;$i -lt $rows.Count;$i++){
   if($uris[$i] -cne ($origin+'/'+$rows[$i].key) -or $intentUris[$i] -cne $uris[$i] -or !$digests[$i] -or
      $digests[$i] -cne $rows[$i].nativeDigest -or $intentDigests[$i] -cne $digests[$i]){throw 'Native CI URI graph mismatch'}
  }
  if($context.sid -cne $ExpectedSid -or $context.session_sid -cne $ExpectedSid -or $context.wts_state -ne 0 -or $context.interactive -ne $true -or
     $context.session -ne $success.session -or $success.actorSid -cne $ExpectedSid -or
     $success.sourceCommit -cne $SourceCommit -or $success.graphSha256 -cne $hash -or $success.root -cne $Graph.root -or
     $Verified.registration.graphSha256 -cne $hash -or $Verified.registration.sid -cne $ExpectedSid){throw 'Native CI identity or graph changed'}
  $main=@($Graph.objects|Where-Object{$_.kind -ceq 'main-package'})
  if(@($Graph.objects|Where-Object{$_.kind -cnotin @('main-bundle','main-package','asset-package')}).Count -or
     $main.Count -ne 1 -or !$InstalledFullName -or $Package.PackageFullName -cne $InstalledFullName -or
     $Verified.identity.packageFullName -cne $InstalledFullName -or $Package.Name -cne $main[0].identity -or
     $Package.Publisher -cne $main[0].publisher -or $Package.Version.ToString() -cne $main[0].version -or
     $Package.Architecture.ToString() -ine $main[0].arch -or $Package.Status.ToString() -cne 'Ok'){throw 'Native CI main package mismatch'}
  $freshMain=@(Get-AppxPackage -Name $Package.Name)
  if($freshMain.Count -ne 1 -or $freshMain[0].PackageFullName -cne $InstalledFullName -or
     $freshMain[0].Publisher -cne $Package.Publisher -or $freshMain[0].Version.ToString() -cne $Package.Version.ToString() -or
     $freshMain[0].Architecture.ToString() -ine $Package.Architecture.ToString() -or $freshMain[0].Status.ToString() -cne 'Ok') {throw 'Native CI main registration changed'}
  if(@(Get-AppxPackage -PackageTypeFilter Optional -Name ($Package.Name+'.*')).Count){throw 'Native CI optional package ownership unresolved'}
  $expected=@($Graph.objects|Where-Object{$_.kind -ceq 'asset-package'})
  $observed=@(Get-AppxPackage -PackageTypeFilter Resource -Name $Package.Name)
  $verifiedResources=@($Verified.registration.related)
  if($expected.Count -ne $observed.Count -or $expected.Count -ne $verifiedResources.Count){throw 'Native CI resource set mismatch'}
  foreach($asset in $expected){
   $p=@($observed|Where-Object{$_.ResourceId -ceq $asset.resourceId -and $_.Publisher -ceq $Package.Publisher -and $_.Version.ToString() -ceq $Package.Version.ToString() -and $_.PackageFamilyName -ceq $Package.PackageFamilyName -and $_.Architecture.ToString() -ieq $asset.arch -and $_.Status.ToString() -ceq 'Ok' -and $_.SignatureKind.ToString() -ceq 'Developer' -and !$_.IsDevelopmentMode})
   $v=@($verifiedResources|Where-Object{$_.resource_id -ceq $asset.resourceId})
   if($p.Count -ne 1 -or $v.Count -ne 1 -or $v[0].full -cne $p[0].PackageFullName){throw 'Native CI resource identity mismatch'}
  }
  $roots=@($HeldRoots);$facts=@($Verified.registration.physicalRoots)
  $registered=@($freshMain)+$observed
  if($roots.Count -ne $registered.Count -or $facts.Count -ne $registered.Count){throw 'Native CI physical root set mismatch'}
  foreach($item in $registered){
   $held=@($roots|Where-Object{$_.full -ceq $item.PackageFullName})
   $fact=@($facts|Where-Object{$_.full -ceq $item.PackageFullName})
   if($held.Count -ne 1 -or $fact.Count -ne 1 -or !$held[0].root.handle -or $held[0].root.handle.IsClosed -or $held[0].root.handle.IsInvalid -or $held[0].root.id -cne $fact[0].id){throw 'Native CI held root mismatch'}
   $fresh=Registered-Root $item -Retain
   try {if($fresh.id -cne $held[0].root.id){throw 'Native CI registered physical root changed'}}
   finally {$fresh.handle.Dispose()}
  }
  return
 }
 if(!$InstalledFullName){throw 'No captured package identity; preserve registration'}
}

# Call only from the supported coordinator-owned smoke after its gateway has
# stopped. The execution lock and unique run writer cover the final interval.
function Open-CIOwnedGateway([int]$ProcessId) {
 if(-not ('SemantierCIGatewayHandle' -as [type])) {
  Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
public sealed class SemantierCIGatewayHandle : IDisposable {
 [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
 [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessTimes(IntPtr h, out long created, out long exited, out long kernel, out long user);
 [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] static extern bool QueryFullProcessImageName(IntPtr h, int flags, StringBuilder name, ref int size);
 [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr h, uint code);
 [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(IntPtr h, uint ms);
 [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
 readonly IntPtr handle;
 public SemantierCIGatewayHandle(int pid) { handle=OpenProcess(0x101001, false, pid); if(handle==IntPtr.Zero) throw new Win32Exception(); }
 public double Created { get { long c,e,k,u; if(!GetProcessTimes(handle,out c,out e,out k,out u)) throw new Win32Exception(); return (c-116444736000000000L)/10000000.0; } }
 public string Image { get { var s=new StringBuilder(32768); int n=s.Capacity; if(!QueryFullProcessImageName(handle,0,s,ref n)) throw new Win32Exception(); return s.ToString(); } }
 public void Stop() { if(!TerminateProcess(handle,0)) throw new Win32Exception(); if(WaitForSingleObject(handle,60000)!=0) throw new InvalidOperationException("Owned gateway did not stop"); }
 public void Dispose() { CloseHandle(handle); }
}
'@
 }
 return [SemantierCIGatewayHandle]::new($ProcessId)
}

function Stop-CIOwnedGateway($Record,[string]$PayloadPrefix) {
 if(-not $Record.pid -or -not $Record.create_time){throw 'Refusing cleanup of an unproven gateway incarnation'}
 try {$handle=Open-CIOwnedGateway ([int]$Record.pid)} catch {
  $failure=$_.Exception
  if($failure -is [Management.Automation.MethodInvocationException]){$failure=$failure.InnerException}
  if($failure -is [ComponentModel.Win32Exception] -and $failure.NativeErrorCode -eq 87){return} # PID no longer exists
  throw
 }
 try {
  if([Math]::Abs($handle.Created-[double]$Record.create_time) -ge 0.001 -or
     -not $handle.Image.StartsWith($PayloadPrefix,[StringComparison]::OrdinalIgnoreCase)) {
   throw 'Refusing cleanup of an unproven gateway incarnation'
  }
  $handle.Stop()
 } finally {$handle.Dispose()}
}

function Invoke-CINativeOwnedRemoval($Graph,[string]$InstalledFullName,$State,$Verified,$Package,[string]$ExpectedSid,[string]$GraphPath,[string]$SourceCommit,[ref]$HeldRoots,[string]$Out) {
 Assert-CINativeCleanupOwnership $Graph $InstalledFullName $State $Verified $Package $ExpectedSid $GraphPath $SourceCommit $HeldRoots.Value $Out
 foreach($held in @($HeldRoots.Value)){$held.root.handle.Dispose()}
 $HeldRoots.Value=@();$fresh=@()
 try {
  $current=@(Get-AppxPackage -Name $Package.Name)+@(Get-AppxPackage -PackageTypeFilter Resource -Name $Package.Name)
  foreach($item in $current){$fresh+=@(@{full=$item.PackageFullName;root=(Registered-Root $item -Retain)})}
  Assert-CINativeCleanupOwnership $Graph $InstalledFullName $State $Verified $Package $ExpectedSid $GraphPath $SourceCommit $fresh $Out
  foreach($item in $current){$null=Assert-BundleWinRTReady $item $ExpectedSid}
 } finally {foreach($held in $fresh){$held.root.handle.Dispose()}}
 Remove-AppxPackage -Package $InstalledFullName -ErrorAction Stop
 if(@(Get-AppxPackage -Name $Package.Name).Count -or @(Get-AppxPackage -PackageTypeFilter Resource -Name $Package.Name).Count -or
    @(Get-AppxPackage -PackageTypeFilter Optional -Name ($Package.Name+'.*')).Count){throw 'Native CI package removal incomplete'}
}
