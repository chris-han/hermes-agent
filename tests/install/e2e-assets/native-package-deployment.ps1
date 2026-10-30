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

function Test-WD1LocalAction([string]$Action) {
 return $Action -cin @('observe','artifact','register','native-registration-proof','launch','native-proof','ready','conservation','remove')
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
   $details.operationCompleted=[bool]($task -and $task.IsCompleted)
   $details.producedAt=[DateTime]::UtcNow.ToString('o');$details.status='FAILURE';$details.elapsedSeconds=$watch.Elapsed.TotalSeconds
   return $details
  }finally{$watch.Stop()}
 }
 $null=Write-NativeRegistrationIntent $Evidence 'intent.json' $context.Value.target.native_graph.sha256 $uris[$rootIndex] $uris $digests
 # Flip one root SDK-digest bit; every URI and every other expected digest stays fixed.
 $wrong=[string[]]$digests.Clone();$bytes=[Convert]::FromBase64String($wrong[$rootIndex]);$bytes[0]=$bytes[0] -bxor 1;$wrong[$rootIndex]=[Convert]::ToBase64String($bytes)
 $refusal=Deploy $wrong 2
 $refusal|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Evidence 'wrong-digest.json') -Encoding utf8
 $after=Observe
 if($after.package -or @($after.related).Count){throw 'Wrong-digest call changed registration; preserve actual state and pending intent'}
 # SDK26100 winerror.h: APPX_E_DIGEST_MISMATCH 0x80080219. Other failures do not prove binding.
 if($refusal.status -cne 'FAILURE' -or $refusal.hresult -cne '80080219'){throw 'Exact native digest refusal unproved; preserve failure before positive execution'}
 $null=Write-NativeRegistrationIntent $Evidence 'positive-intent.json' $context.Value.target.native_graph.sha256 $uris[$rootIndex] $uris $digests
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

# Create-new, flushed evidence immediately before each CI SDK invocation.
# An intent without a terminal outcome is unresolved; it does not grant cleanup.
function Invoke-CIDurableNativeUri([string]$Out,[string]$GraphSha,[string]$SourceCommit,[string]$Operation,$Actor,[string[]]$Uris,[string[]]$Digests,[string]$Root,[int]$Minutes) {
 if($GraphSha -cnotmatch '^[a-f0-9]{64}$' -or $SourceCommit -cnotmatch '^[a-f0-9]{40}$' -or $Operation -cnotin @('positive','wrong-digest') -or $Root -cnotin $Uris -or $Uris.Count -ne $Digests.Count){throw 'Invalid CI native operation binding'}
 if(!$Actor.sid -or $Actor.sid -cne $Actor.session_sid -or $Actor.session -le 0 -or $Actor.wts_state -ne 0){throw 'Invalid CI native actor'}
 $id=[Guid]::NewGuid().ToString('N');$self=Get-Process -Id $PID
 $intent=@{schema=1;operationId=$id;createdAt=[DateTime]::UtcNow.ToString('o');sourceCommit=$SourceCommit;graphSha256=$GraphSha;operation=$Operation;actor=$Actor;actorPid=$PID;actorBirth=$self.StartTime.ToUniversalTime().ToString('o');root=$Root;uris=$Uris;digests=$Digests}
 function Write-CIRecord([string]$Name,$Value) {
  $stream=[IO.File]::Open((Join-Path $Out $Name),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$bytes=[Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 16 -Compress));$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
 }
 Write-CIRecord ('ci-native-intent-'+$id+'.json') $intent
 $result=Invoke-BoundNativePackageUri $Uris $Digests $Root $Minutes
 Write-CIRecord ('ci-native-outcome-'+$id+'.json') @{schema=1;operationId=$id;createdAt=[DateTime]::UtcNow.ToString('o');result=$result}
 return $result
}

# CI-only complete-object digest refusal/registration proof. No publication authority.
function Invoke-CINativeGraphUri($Graph,[string]$GraphPath,[string]$Artifact,[string]$Node,[string]$Out,[string]$ExpectedSid,$State,[string]$SourceCommit) {
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
   $refusal=Invoke-CIDurableNativeUri $Out $hash $SourceCommit 'wrong-digest' $context $uris $expected ($origin.base+'/'+$Graph.root) 2
   $refusals+=@{key=$rows[$index].key;result=$refusal}
   @{graphSha256=$hash;refusals=$refusals}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $Out 'native-uri-refusals.json') -Encoding UTF8
   $present=@(Get-AppxPackage -Name $rows[$index].identity)+@(Get-AppxPackage -PackageTypeFilter Resource -Name $rows[$index].identity)
   $refusals[-1].registrationAbsentAfterRefusal=($present.Count -eq 0)
   if($refusal.status -cne 'FAILURE' -or $refusal.hresult -cne '80080219' -or $present.Count){throw 'Exact per-object native digest refusal and unchanged registration required'}
   $State.unknown=$false
  }
  $State.unknown=$true
  $positive=Invoke-CIDurableNativeUri $Out $hash $SourceCommit 'positive' $context $uris $digests ($origin.base+'/'+$Graph.root) 25
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

# Durable SDK registration provenance. This never grants registration authority;
# the existing platform/register intent supplies the verified actor and graph.
function Write-NativeRegistrationIntent([string]$Directory,[string]$Name,[string]$GraphSha,[string]$RootUri,[string[]]$Uris,[string[]]$Digests) {
 if($Name -cnotin @('intent.json','positive-intent.json') -or $GraphSha -cnotmatch '^[a-f0-9]{64}$' -or $RootUri -cnotin $Uris){throw 'Invalid native registration provenance'}
 $self=Get-Process -Id $PID
 $value=@{schema=1;producedAt=[DateTime]::UtcNow.ToString('o');sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;session=$self.SessionId;actorPid=$PID;actorBirth=$self.StartTime.ToUniversalTime().ToString('o');graphSha256=$GraphSha;root=$RootUri;uris=$Uris;digests=$Digests}
 $file=[IO.File]::Open((Join-Path $Directory $Name),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
 try{$bytes=[Text.Encoding]::UTF8.GetBytes(($value|ConvertTo-Json -Depth 10 -Compress));$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
 return $value
}

function Read-NativeRegistrationOutcome($Spec,$Graph,[string]$Base,[string]$Root,[DateTimeOffset]$IntentAt,[bool]$WD1) {
 $expectedUris=@($Graph.objects|ForEach-Object{$Base+'/'+$_.key});$expectedDigests=@($Graph.objects|ForEach-Object{[string]$_.nativeDigest})
 $pattern=if($WD1){'native-deployment-*'}else{'native-baseline-*'};$found=@()
 foreach($directory in @(Get-ChildItem -LiteralPath $Root -Directory -Filter $pattern)){
  $hold=Physical $directory.FullName
  try{
   $path=Join-Path $directory.FullName 'intent.json';if(!(Test-Path -LiteralPath $path)){continue}
   $file=Physical $path
   try{if((Get-Item -LiteralPath $path).Length -gt 4194304){throw 'Native registration record oversized'};$intent=Get-Content -LiteralPath $path -Raw -Encoding UTF8|ConvertFrom-Json}finally{$file.handle.Dispose()}
   if([DateTimeOffset]::Parse($intent.producedAt) -lt $IntentAt){continue}
   if($intent.sid -cne $sid -or $intent.session -ne $session -or $intent.graphSha256 -cne $Spec.native_graph.sha256 -or $intent.root -cne ($Base+'/'+$Graph.root) -or ($intent.uris -join '|') -cne ($expectedUris -join '|') -or ($intent.digests -join '|') -cne ($expectedDigests -join '|')){throw 'Native registration intent binding changed'}
   $resultName=if($WD1){'correct-digest.json'}else{'result.json'};$resultPath=Join-Path $directory.FullName $resultName
   $result=$null
   if(Test-Path -LiteralPath $resultPath){
    $resultHold=Physical $resultPath
    try{if((Get-Item -LiteralPath $resultPath).Length -gt 4194304){throw 'Native registration result oversized'};$result=Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8|ConvertFrom-Json}finally{$resultHold.handle.Dispose()}
   }
   $found+=@{intent=$intent;result=$result;directory=$directory.FullName}
  }finally{$hold.handle.Dispose()}
 }
 if($found.Count -ne 1){throw 'Unique durable native registration attempt required; preserve UNKNOWN'}
 $candidate=$found[0];$result=$candidate.result
 if($result -and $result.status -ceq 'SUCCESS'){
  if(!$WD1 -and (!$result.PSObject.Properties['operationCompleted'] -or !$result.operationCompleted)){throw 'Native completion not established'}
  return @{status='SUCCESS';operationCompleted=$true;intent=$candidate.intent;sdk=$result;scope='exact native registration only'}
 }
 # Older WD1 callers passed an operation to GetResults only after IsCompleted.
 # Preserve those typed terminal failures; async/text-only errors and an explicit
 # incomplete marker must remain UNKNOWN. The durable graph/actor/URI/digests
 # above still bind the receipt to this one registration attempt.
 $typedWD1Failure=$false;$activity=[Guid]::Empty
 if($WD1 -and $result -and $result.status -ceq 'FAILURE' -and !$result.PSObject.Properties['operationCompleted']){
  $typedWD1Failure=($result.PSObject.Properties['errorSource'] -and $result.errorSource -ceq 'deployment-result' -and
   $result.PSObject.Properties['isRegistered'] -and $result.isRegistered -is [bool] -and !$result.isRegistered -and
   $result.PSObject.Properties['hresult'] -and $result.hresult -cmatch '^[0-9A-F]{8}$' -and $result.hresult -cne '00000000' -and
   $result.PSObject.Properties['activityId'] -and [Guid]::TryParse([string]$result.activityId,[ref]$activity) -and $activity -ne [Guid]::Empty)
 }
 if($result -and $result.status -ceq 'FAILURE' -and (($result.PSObject.Properties['operationCompleted'] -and $result.operationCompleted) -or $typedWD1Failure)){
  return @{status='FAILURE';operationCompleted=$true;intent=$candidate.intent;sdk=$result;scope='completed native failure only'}
 }
 # A missing/live result cannot be closed from registration absence. For an
 # ended caller, reuse exact SDK queue/completion correlation for late success.
 $actor=Get-Process -Id $candidate.intent.actorPid -ErrorAction SilentlyContinue
 if($actor -and $actor.StartTime.ToUniversalTime().ToString('o') -ceq $candidate.intent.actorBirth){throw 'Native registration caller remains live; preserve UNKNOWN'}
 $positivePath=Join-Path $candidate.directory 'positive-intent.json';$started=$candidate.intent.producedAt
 if($WD1){
  if(!(Test-Path -LiteralPath $positivePath)){throw 'Native positive call not established; preserve UNKNOWN'}
  $positiveHold=Physical $positivePath
  try{$positive=Get-Content -LiteralPath $positivePath -Raw -Encoding UTF8|ConvertFrom-Json}finally{$positiveHold.handle.Dispose()}
  if($positive.graphSha256 -cne $Spec.native_graph.sha256 -or $positive.root -cne $candidate.intent.root -or $positive.sid -cne $sid -or $positive.session -ne $session){throw 'Native positive intent changed'}
  $started=$positive.producedAt
 }elseif($result -and $result.PSObject.Properties['startedAt']){$started=$result.startedAt}
 $uri=[string]$candidate.intent.root
 if($uri.Contains("'") -or $sid.Contains("'")){throw 'Unsupported URI for scoped native event query'}
 $stamp=([DateTimeOffset]::Parse($started)).ToUniversalTime().AddSeconds(-2).ToString('o')
 $xpath="*[System[Security[@UserID='$sid'] and TimeCreated[@SystemTime>='$stamp']]] and (*[System[EventID=603]] or (*[System[EventID=478]] and *[EventData[Data[@Name='PackageSourceUri']='$uri']]))"
 $events=@()
 try{$events=@(Get-WinEvent -LogName 'Microsoft-Windows-AppXDeploymentServer/Operational' -FilterXPath $xpath -MaxEvents 513 -ErrorAction Stop)}catch{if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){throw}}
 if($events.Count -gt 512){throw 'Scoped native event window incomplete; preserve UNKNOWN'}
 $parsed=@($events|ForEach-Object{Read-WD1DeploymentEvent $_})
 $timeout=if($result -and $result.PSObject.Properties['producedAt']){[DateTimeOffset]::Parse($result.producedAt)}else{[DateTimeOffset]::Parse($started)}
 $complete=Find-WD1LateDeployment $parsed ([DateTimeOffset]::Parse($started)) $timeout $sid $uri $r.operation.full
 return @{status='RECONCILED_OS_SUCCESS';operationCompleted=$true;intent=$candidate.intent;sdk=$result;activityId=$complete.activity;eventRecordId=$complete.recordId;completedAt=$complete.time.ToString('o');scope='late completed native registration only; no relaunch/differential acceptance'}
}
