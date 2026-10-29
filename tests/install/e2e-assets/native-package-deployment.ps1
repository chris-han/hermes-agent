# Bounded WD1 native URI proof, called only inside the existing register intent.
# This test adapter never creates CI qualification, publication or update acceptance.
function Get-WD1DeploymentFailure($Operation, [Type]$OperationType, [Exception]$Failure) {
 function HResult-Hex([Exception]$Value){return [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$Value.HResult),0).ToString('X8')}
 $failureException=$Failure.GetBaseException()
 $details=@{hresult=(HResult-Hex $failureException);operationHresult=(HResult-Hex $failureException);error=$failureException.Message;errorSource='async-exception';activityId=$null}
 if($null -eq $Operation){return $details}
 try{
  # AsTask reports the generic operation error; GetResults retains deployment details.
  $deployment=$OperationType.GetMethod('GetResults').Invoke($Operation,@())
  if($deployment -and $deployment.ExtendedErrorCode -is [Exception]){
   $extended=$deployment.ExtendedErrorCode.GetBaseException()
   $details.hresult=HResult-Hex $extended
   $details.error=$deployment.ErrorText
   $details.errorSource='deployment-result'
   $details.activityId=[string]$deployment.ActivityId
   $details.isRegistered=[bool]$deployment.IsRegistered
  }
 }catch{
  $resultException=$_.Exception.GetBaseException()
  $details.hresult=HResult-Hex $resultException
  $details.error=$resultException.Message
  $details.errorSource='get-results-exception'
 }
 return $details
}

# Explicit local WD1 actors only; this is not a CI or generic installer allowance.
function Test-WD1LocalActor([string]$Sid,[string]$AuthorizedSid,[string]$Purpose) {
 return ($Purpose -ceq 'wd1-native-layout-qualification' -and $Sid -ceq $AuthorizedSid -and $Sid -cin @('S-1-5-21-1928370875-4287869425-2092521300-1011','S-1-5-21-1928370875-4287869425-2092521300-1003'))
}

function Invoke-WD1NativeUri($Graph, [string]$Base, [string]$Evidence) {
 $ErrorActionPreference='Stop'
 $context=Get-Variable -Name r -ErrorAction SilentlyContinue
 if(!$context -or $context.Value.purpose -cne 'wd1-native-layout-qualification' -or $context.Value.action -cne 'register' -or !$context.Value.PSObject.Properties['expected_before']){throw 'Existing WD1 register intent context required'}
 if(-not (Test-WD1LocalActor ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) $context.Value.authorized_sid $context.Value.purpose) -or -not [Environment]::UserInteractive -or (Get-Process -Id $PID).SessionId -le 0){throw 'Actual authorized baseline desktop required'}
 if($Base -cnotmatch '^http://127\.0\.0\.1:[1-9][0-9]{0,4}$'){throw 'WD1 requires its exact loopback origin'}
 . (Join-Path $PSScriptRoot '..\..\..\apps\desktop\scripts\native-package-digest.ps1')
 Add-Type -AssemblyName System.Runtime.WindowsRuntime
 $null=[Windows.Management.Deployment.PackageManager,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $null=[Windows.Management.Deployment.AddPackageOptions,Windows.Management.Deployment,ContentType=WindowsRuntime]
 $manager=[Windows.Management.Deployment.PackageManager]::new()
 $method=[Windows.Management.Deployment.PackageManager].GetMethod('AddPackageByUriAsync')
 if(!$method){throw 'AddPackageByUriAsync unavailable'}
 $generic=$method.ReturnType.GetGenericArguments()
 $binding=@([System.WindowsRuntimeSystemExtensions].GetMethods()|Where-Object{$_.Name -ceq 'AsTask' -and $_.IsGenericMethodDefinition -and $_.GetGenericArguments().Count -eq 2 -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -ceq 'IAsyncOperationWithProgress`2'})
 if($binding.Count -ne 1){throw 'Actual native asynchronous binding unavailable'}
 $asTask=$binding[0].MakeGenericMethod($generic)
 $uris=@();$digests=@();$rootIndex=-1
 foreach($row in $Graph.objects){
  if($row.key -cnotmatch '^releases/channel-builds/[0-9a-f]{32}/[A-Za-z0-9.-]+\.(msix|msixbundle)$' -or $row.size -gt 480000000){throw 'Invalid bounded native graph'}
  if($row.key -ceq $Graph.root){$rootIndex=$uris.Count}
  $uris+=($Base+'/'+$row.key);$digests+=[string]$row.nativeDigest
 }
 if($rootIndex -lt 0){throw 'Native root absent'}
 function Deploy([string[]]$Expected, [int]$Minutes) {
  $options=[Windows.Management.Deployment.AddPackageOptions]::new()
  $null=Set-NativeExpectedDigests $options $uris $Expected
  $watch=[Diagnostics.Stopwatch]::StartNew()
  $operation=$null;$task=$null
  try{
   $operation=$manager.AddPackageByUriAsync([Uri]$uris[$rootIndex],$options)
   $task=$asTask.Invoke($null,@($operation))
   if(!$task.Wait([TimeSpan]::FromMinutes($Minutes))){throw [TimeoutException]::new('Native deployment remains unresolved; preserve intent')}
   $deployment=$task.Result
   if($deployment.ExtendedErrorCode){throw $deployment.ExtendedErrorCode}
   return @{producedAt=[DateTime]::UtcNow.ToString('o');status='SUCCESS';hresult='00000000';elapsedSeconds=$watch.Elapsed.TotalSeconds;activityId=[string]$deployment.ActivityId}
  }catch{
   $completed=if($task -and $task.IsCompleted){$operation}else{$null}
   $details=Get-WD1DeploymentFailure $completed $method.ReturnType $_.Exception
   $details.producedAt=[DateTime]::UtcNow.ToString('o');$details.status='FAILURE';$details.elapsedSeconds=$watch.Elapsed.TotalSeconds
   return $details
  }finally{$watch.Stop()}
 }
 # Flip one root SDK-digest bit; every URI and every other expected digest stays fixed.
 $wrong=[string[]]$digests.Clone();$bytes=[Convert]::FromBase64String($wrong[$rootIndex]);$bytes[0]=$bytes[0] -bxor 1;$wrong[$rootIndex]=[Convert]::ToBase64String($bytes)
 $refusal=Deploy $wrong 2
 $refusal|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Evidence 'wrong-digest.json') -Encoding utf8
 $after=Observe
 if($after.package -or @($after.related).Count){throw 'Wrong-digest call changed registration; preserve actual state and pending intent'}
 # SDK26100 winerror.h: APPX_E_DIGEST_MISMATCH 0x80080219. Other failures do not prove binding.
 if($refusal.status -cne 'FAILURE' -or $refusal.hresult -cne '80080219'){throw 'Exact native digest refusal unproved; preserve failure before positive execution'}
 $positive=Deploy ([string[]]$digests) 25
 $positive|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Evidence 'correct-digest.json') -Encoding utf8
 if($positive.status -cne 'SUCCESS'){throw 'Correct native URI deployment failed; preserve pending intent'}
 @{producedAt=[DateTime]::UtcNow.ToString('o');root=$Graph.root;uris=$uris;digests=$digests;refusal=$refusal;positive=$positive;scope='WD1 exact native URI and digest proof only; no migration/differential/cache/relaunch acceptance'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Evidence 'native-uri-proof.json') -Encoding utf8
}

function Find-WD1LateDeployment($Events,[DateTimeOffset]$Started,[DateTimeOffset]$Timeout,[string]$Sid,[string]$Uri,[string]$FullName) {
 $matches=@($Events|Where-Object{$_.id -eq 478 -and $_.operation -ceq '1' -and $_.sid -ceq $Sid -and $_.uri -ceq $Uri -and $_.full -ceq $FullName -and $_.callingProcess -ceq 'powershell.exe' -and $_.time -ge $Timeout}|Where-Object{
  $activity=$_.activity
  @($Events|Where-Object{$_.id -eq 603 -and $_.operation -ceq '1' -and $_.sid -ceq $Sid -and $_.activity -ceq $activity -and $_.callingProcess -ceq 'powershell.exe' -and [Math]::Abs(($_.time-$Started).TotalSeconds) -le 2}).Count -eq 1
 })
 if($matches.Count -ne 1){throw 'Exactly one matching late native deployment required'}
 return $matches[0]
}

function Read-WD1DeploymentEvent($Event) {
 [xml]$xml=$Event.ToXml()
 $system=$xml.SelectSingleNode('/*[local-name()="Event"]/*[local-name()="System"]')
 if($system.SelectSingleNode('*[local-name()="Provider"]').GetAttribute('Name') -cne 'Microsoft-Windows-AppXDeployment-Server'){throw 'Unexpected native event provider'}
 $data=@{}
 foreach($item in $xml.SelectNodes('/*[local-name()="Event"]/*[local-name()="EventData"]/*[local-name()="Data"]')){
  $name=$item.GetAttribute('Name');if(!$name -or $data.ContainsKey($name)){throw 'Ambiguous native event data'}
  $data[$name]=$item.InnerText
 }
 return [pscustomobject]@{id=[int]$Event.Id;operation=$data['DeploymentOperation'];sid=$system.SelectSingleNode('*[local-name()="Security"]').GetAttribute('UserID');activity=[string]$Event.ActivityId;time=[DateTimeOffset]::Parse($system.SelectSingleNode('*[local-name()="TimeCreated"]').GetAttribute('SystemTime'));callingProcess=$data['CallingProcess'];uri=$data['PackageSourceUri'];full=$data['PackageFullName'];recordId=$Event.RecordId}
}
function Recover-WD1NativeUri($Graph,[string]$Base,[string]$Root,$State,$Target,[string]$Evidence) {
 $context=Get-Variable -Name r -ErrorAction SilentlyContinue
 if(!$context -or $context.Value.purpose -cne 'wd1-native-layout-qualification' -or $context.Value.action -cne 'native-proof' -or -not (Test-WD1LocalActor ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) $context.Value.authorized_sid $context.Value.purpose)){throw 'Actual WD1 proof context required'}
 if(!$State.package -or $State.package.full -cne $Target.full -or $State.package.commit -cne $Target.commit -or $State.package.build_id -cne $Target.build_id){throw 'Exact registered native target required'}
 $candidates=@()
 foreach($dir in @(Get-ChildItem -LiteralPath $Root -Directory -Filter 'native-deployment-*')){
  $wrong=Join-Path $dir.FullName 'wrong-digest.json';$correct=Join-Path $dir.FullName 'correct-digest.json'
  if(!(Test-Path -LiteralPath $wrong) -or !(Test-Path -LiteralPath $correct)){continue}
  $a=Physical $wrong;$b=Physical $correct
  try{
   $negative=Get-Content -LiteralPath $wrong -Raw|ConvertFrom-Json;$positive=Get-Content -LiteralPath $correct -Raw|ConvertFrom-Json
   if($negative.hresult -cne '80080219' -or $negative.status -cne 'FAILURE' -or $negative.isRegistered -ne $false -or $positive.hresult -cne '80131505' -or $positive.status -cne 'FAILURE'){continue}
   if($positive.producedAt -notmatch 'Z$' -or $negative.producedAt -notmatch 'Z$' -or $positive.elapsedSeconds -le 0){throw 'Aware native timeout timestamps required'}
   $timeout=[DateTimeOffset]::Parse($positive.producedAt);$started=$timeout.AddSeconds(-[double]$positive.elapsedSeconds);$refused=[DateTimeOffset]::Parse($negative.producedAt)
   if($started -lt $refused -or ($started-$refused).TotalSeconds -gt 5){throw 'Native attempt sequence mismatch'}
   $candidates+=@{negative=$negative;timeout=$positive;started=$started;ended=$timeout;source=$dir.FullName}
  }finally{$a.handle.Dispose();$b.handle.Dispose()}
 }
 if($candidates.Count -ne 1){throw 'Exactly one bounded native timeout attempt required'}
 $candidate=$candidates[0];$events=@()
 foreach($event in @(Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-AppXDeploymentServer/Operational';Id=@(603,478);StartTime=$candidate.started.AddSeconds(-2).LocalDateTime})){
  $events+=Read-WD1DeploymentEvent $event
 }
 $uri=$Base+'/'+$Graph.root
 $success=Find-WD1LateDeployment $events $candidate.started $candidate.ended $State.sid $uri $Target.full
 $proof=@{producedAt=[DateTime]::UtcNow.ToString('o');root=$Graph.root;uris=@($Graph.objects|ForEach-Object{$Base+'/'+$_.key});digests=@($Graph.objects|ForEach-Object{$_.nativeDigest});refusal=$candidate.negative;positive=@{status='RECONCILED_OS_SUCCESS';activityId=$success.activity;completedAt=$success.time.ToString('o');eventRecordId=$success.recordId};originalTimeout=$candidate.timeout;attempt=$candidate.source;scope='late native OS success plus fresh owned registration; original timeout preserved; not CI/migration/differential acceptance'}
 $proof|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $Evidence 'native-uri-reconciled.json') -Encoding utf8
 return $proof
}

# CI-only complete-object digest refusal/registration proof. No publication authority.
function Invoke-CINativeGraphUri($Graph,[string]$GraphPath,[string]$Artifact,[string]$Node,[string]$Out,[string]$ExpectedSid,$State) {
 if($env:GITHUB_ACTIONS -cne 'true' -or !$ExpectedSid -or $ExpectedSid -cin @('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011') -or !(Get-Command Assert-CISession -ErrorAction SilentlyContinue)){throw 'Separate genuine CI session required'}
 $context=Assert-CISession
 if($context.sid -cne $ExpectedSid){throw 'CI URI actor mismatch'}
 . (Join-Path $PSScriptRoot '..\..\..\apps\desktop\scripts\native-package-digest.ps1')
 $ready=Join-Path $Out 'native-origin-ready.json';$log=Join-Path $Out 'native-origin-requests.jsonl'
 $hash=(Get-FileHash -LiteralPath $GraphPath -Algorithm SHA256).Hash.ToLowerInvariant()
 $originScript=Join-Path $PSScriptRoot 'native-package-origin.mjs'
 $argv=@($originScript,'--directory',(Split-Path -Parent $Artifact),'--graph',$GraphPath,'--graph-sha256',$hash,'--ready',$ready,'--log',$log)
 $quoted=@($argv|ForEach-Object{'"'+$_.Replace('"','\"')+'"'})
 $process=Start-Process -FilePath $Node -ArgumentList $quoted -WorkingDirectory $Out -PassThru -RedirectStandardOutput (Join-Path $Out 'native-origin.stdout.log') -RedirectStandardError (Join-Path $Out 'native-origin.stderr.log')
 $null=$process.Handle
 $State.origin=@{pid=$process.Id;birth=([DateTimeOffset]$process.StartTime).ToUniversalTime().ToString('o');exe=$Node;script=$originScript;graphSha256=$hash}
 try{
  $deadline=[DateTimeOffset]::UtcNow.AddMinutes(10)
  while(!(Test-Path -LiteralPath $ready)){
   if($process.HasExited -or [DateTimeOffset]::UtcNow -gt $deadline){throw 'CI pinned native origin unavailable'}
   Start-Sleep -Milliseconds 500
  }
  $origin=Get-Content -LiteralPath $ready -Raw -Encoding UTF8|ConvertFrom-Json
  if($origin.pid -ne $process.Id -or $origin.graphSha256 -cne $hash -or $origin.base -cnotmatch '^http://127\.0\.0\.1:[1-9][0-9]{0,4}$'){throw 'CI native origin binding mismatch'}
  $rows=@($Graph.objects);$uris=@($rows|ForEach-Object{$origin.base+'/'+$_.key});$digests=@($rows|ForEach-Object{[string]$_.nativeDigest});$refusals=@()
  foreach($index in 0..($rows.Count-1)){
   $State.unknown=$true
   $expected=[string[]]$digests.Clone();$bytes=[Convert]::FromBase64String($expected[$index]);$bytes[0]=$bytes[0] -bxor 1;$expected[$index]=[Convert]::ToBase64String($bytes)
   $refusal=Invoke-BoundNativePackageUri $uris $expected ($origin.base+'/'+$Graph.root) 2
   $refusals+=@{key=$rows[$index].key;result=$refusal}
   @{graphSha256=$hash;refusals=$refusals}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $Out 'native-uri-refusals.json') -Encoding UTF8
   $present=@(Get-AppxPackage -Name $rows[$index].identity)+@(Get-AppxPackage -PackageTypeFilter Resource -Name $rows[$index].identity)
   $refusals[-1].registrationAbsentAfterRefusal=($present.Count -eq 0)
   if($refusal.status -cne 'FAILURE' -or $refusal.hresult -cne '80080219' -or $present.Count){throw 'Exact per-object native digest refusal and unchanged registration required'}
   $State.unknown=$false
  }
  $State.unknown=$true
  $positive=Invoke-BoundNativePackageUri $uris $digests ($origin.base+'/'+$Graph.root) 25
  $positive|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $Out 'native-uri-positive.json') -Encoding UTF8
  if($positive.status -cne 'SUCCESS'){throw 'Native URI registration unresolved; preserve CI intent, origin and packages'}
  $State.unknown=$false
  $after=Assert-CISession
  @{scope='CI_NATIVE_URI_GRAPH_ONLY_NOT_DIFFERENTIAL_OR_CACHE';producedAt=[DateTime]::UtcNow.ToString('o');graphSha256=$hash;sourceBuildId=($Graph.root -split '/')[2];contextBefore=$context;contextAfter=$after;uris=$uris;digests=$digests;refusals=$refusals;positive=$positive;nativeRemoteDigestEnforcement='PASS'}|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $Out 'native-uri-proof.json') -Encoding UTF8
 }finally{
  if($State.unknown){$State|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Out 'native-uri-unknown.json') -Encoding UTF8}
  elseif(!$process.HasExited){$process.Kill();if(!$process.WaitForExit(10000)){throw 'Owned CI native origin did not stop'}}
  $process.Dispose()
 }
}
