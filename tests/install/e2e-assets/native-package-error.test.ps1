# Pure typed-error adapter checks on real PowerShell; no native deployment calls.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
Add-Type -TypeDefinition @'
using System;
public sealed class WD1ResultFixture {
 public object Result; public Exception Failure;
 public object GetResults() { if(Failure!=null)throw Failure; return Result; }
}
'@
$outer=[Runtime.InteropServices.COMException]::new('generic package-open error',[int]0x80073CF0)
$operation=[WD1ResultFixture]::new()
$operation.Result=[pscustomobject]@{ExtendedErrorCode=[Runtime.InteropServices.COMException]::new('specific native digest error',[int]0x80080219);ErrorText='specific native error';ActivityId=[Guid]'11111111-1111-1111-1111-111111111111';IsRegistered=$false}
$result=Get-WD1DeploymentFailure $operation ([WD1ResultFixture]) $outer
if($result.hresult -cne '80080219' -or $result.operationHresult -cne '80073CF0' -or $result.errorSource -cne 'deployment-result'){throw 'Typed extended error was lost'}
$operation.Result=$null
$other=[Runtime.InteropServices.COMException]::new('message mentions 0x80080219',[int]0x80070002)
$result=Get-WD1DeploymentFailure $operation ([WD1ResultFixture]) $other
if($result.hresult -cne '80070002'){throw 'Error text incorrectly became native digest proof'}
$operation.Failure=[Runtime.InteropServices.COMException]::new('native GetResults failed',[int]0x80080219)
$result=Get-WD1DeploymentFailure $operation ([WD1ResultFixture]) $outer
if($result.hresult -cne '80080219' -or $result.errorSource -cne 'get-results-exception'){throw 'Typed GetResults error was lost'}
$result=Get-WD1DeploymentFailure $null ([WD1ResultFixture]) $other
if($result.hresult -cne '80070002'){throw 'Missing operation failed open'}
Write-Output 'PASS: typed extended/native GetResults errors retained; unrelated/text-only errors refused; no SDK deployment'
$begin=[DateTimeOffset]::Parse('2026-10-29T08:58:16.880Z');$timeout=$begin.AddMinutes(12)
$identity='Semantier.Agent.Acceptance_0.0.59.0_x64__p2m4tf4r6sydm';$uri='http://127.0.0.1:2964/releases/channel-builds/646192f4ac6242dc8cbc739ecb04b32e/SemantierAgent.msixbundle'
$start=[pscustomobject]@{id=603;operation='1';sid='fixture-sid';activity='fixture-activity';time=$begin;callingProcess='powershell.exe';uri='';full='';recordId=1}
$finish=[pscustomobject]@{id=478;operation='1';sid='fixture-sid';activity='fixture-activity';time=$timeout.AddMinutes(3);callingProcess='powershell.exe';uri=$uri;full=$identity;recordId=2}
$match=Find-WD1LateDeployment @($start,$finish) $begin $timeout 'fixture-sid' $uri $identity
if($match.recordId -ne 2){throw 'Exact late success not selected'}
foreach($field in @('sid','uri','full','activity')){
 $copy=$finish.PSObject.Copy();$copy.$field='other'
 try{$null=Find-WD1LateDeployment @($start,$copy) $begin $timeout 'fixture-sid' $uri $identity;throw 'Unexpected late success'}catch{if($_.Exception.Message -cne 'Exactly one matching late native deployment required'){throw}}
}
try{$null=Find-WD1LateDeployment @($start,$finish,$finish) $begin $timeout 'fixture-sid' $uri $identity;throw 'Unexpected ambiguous success'}catch{if($_.Exception.Message -cne 'Exactly one matching late native deployment required'){throw}}
Write-Output 'PASS: exact late-result correlation; wrong SID/URI/package/activity and ambiguity refused; fixture only'

# Empty EventData nodes are real Windows event XML, not PS '#text' properties.
Set-StrictMode -Version Latest
$event=[pscustomobject]@{Id=478;ActivityId=[Guid]'11111111-1111-1111-1111-111111111111';RecordId=1;Xml='<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event"><System><Provider Name="Microsoft-Windows-AppXDeployment-Server"/><TimeCreated SystemTime="2026-10-29T09:13:56Z"/><Security UserID="fixture-sid"/></System><EventData><Data Name="DeploymentOperation">1</Data><Data Name="CallingProcess">powershell.exe</Data><Data Name="PackageSourceUri"/><Data Name="PackageFullName"></Data></EventData></Event>'}
$event|Add-Member ScriptMethod ToXml {return $this.Xml}
$parsed=Read-WD1DeploymentEvent $event
if($parsed.uri -cne '' -or $parsed.full -cne '' -or $parsed.sid -cne 'fixture-sid' -or $parsed.operation -cne '1'){throw 'Empty event XML was misread'}
$event.Xml=$event.Xml.Replace('Microsoft-Windows-AppXDeployment-Server','OtherProvider')
try{$null=Read-WD1DeploymentEvent $event;throw 'Wrong provider accepted'}catch{if($_.Exception.Message -cne 'Unexpected native event provider'){throw}}
Write-Output 'PASS: strict-mode real XML empty EventData fields and wrong-provider refusal'

$baseline='S-1-5-21-1928370875-4287869425-2092521300-1011';$chris='S-1-5-21-1928370875-4287869425-2092521300-1003'
foreach($actor in @($baseline,$chris)){if(-not (Test-WD1LocalActor $actor $actor 'wd1-native-layout-qualification')){throw 'Authorized local actor refused'}}
if((Test-WD1LocalActor $chris $baseline 'wd1-native-layout-qualification') -or (Test-WD1LocalActor $chris $chris 'ci') -or (Test-WD1LocalActor 'other' 'other' 'wd1-native-layout-qualification')){throw 'Local WD1 actor scope widened'}
Write-Output 'PASS: explicit local WD1 actor/SID/purpose binding; cross-account, CI and unrelated actors refused'

foreach($action in @('observe','artifact','register','native-registration-proof','launch','native-proof','ready','conservation','remove')){
 if(-not (Test-WD1LocalAction $action)){throw 'Existing WD1 operation refused'}
}
foreach($action in @('update','download','native-reconciled','unknown')){
 if(Test-WD1LocalAction $action){throw 'WD1 purpose improperly authorized production update or unrelated mutation'}
}
Write-Output 'PASS: WD1 operation scope preserves local qualification and refuses production update/apply/ack'

# The SDK wrapper must durably record an actor-bound intent before its call.
$ciTmp=Join-Path ([IO.Path]::GetTempPath()) ('semantier-ci-intent-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $ciTmp -ErrorAction Stop|Out-Null
try {
 $script:seen=$false
 function Invoke-BoundNativePackageUri([string[]]$Uris,[string[]]$Digests,[string]$Root,[int]$Minutes) {
  $files=@(Get-ChildItem -LiteralPath $ciTmp -Filter 'ci-native-intent-*.json')
  if($files.Count -ne 1){throw 'No durable intent at invocation boundary'}
  $i=Get-Content -Raw -LiteralPath $files[0].FullName|ConvertFrom-Json
  if($i.graphSha256 -cne ('a'*64) -or $i.sourceCommit -cne ('b'*40) -or $i.actor.sid -cne 'fixture-sid' -or $i.operation -cne 'positive'){throw 'Incomplete CI intent'}
  $script:seen=$true
  return @{status='SUCCESS';hresult='00000000';activityId='11111111-1111-1111-1111-111111111111';operationCompleted=$true}
 }
 $v=Invoke-CIDurableNativeUri $ciTmp ('a'*64) ('b'*40) 'positive' @{sid='fixture-sid';session_sid='fixture-sid';session=(Get-Process -Id $PID).SessionId;wts_state=0} @('http://127.0.0.1:1/pkg') @('digest') 'http://127.0.0.1:1/pkg' 1
 if(!$script:seen -or $v.status -cne 'SUCCESS' -or @(Get-ChildItem -LiteralPath $ciTmp -Filter 'ci-native-outcome-*.json').Count -ne 1){throw 'CI native durable transaction not completed'}
 $outcome=Get-Content -Raw -LiteralPath (Join-Path $ciTmp ('ci-native-outcome-'+$v.operationId+'.json'))|ConvertFrom-Json
 if($v.operationId -cnotmatch '^[a-f0-9]{32}$' -or $outcome.operationId -cne $v.operationId -or $outcome.result.operationId -cne $v.operationId){throw 'Returned SDK result is not bound to its durable operation'}
 'PASS: simulated SDK result retains its durable operation identity'
} finally {Remove-Item -LiteralPath $ciTmp -Recurse -Force}

. (Join-Path $PSScriptRoot 'windows-physical-path.ps1')
. (Join-Path $PSScriptRoot 'native-ci-cleanup-policy.ps1')
$held=$false;try {Assert-CINativeCleanupOwnership @{root='native'} 'same-full-name'} catch {$held=$true}
if(!$held){throw 'Native cleanup incorrectly trusts a matching full name'}
$held=$false;try {Assert-CINativeCleanupOwnership $null ''} catch {$held=$true}
if(!$held){throw 'Cleanup incorrectly accepts an uncaptured registration'}
'PASS: uncertain CI native cleanup and uncaptured identity held; no package mutation'

# Simulate PID reuse after the original process handle is opened. Stop must act
# on the validated handle, never look the PID up again.
$script:gatewayOriginal=[pscustomobject]@{Created=1234.5;Image='C:\owned\gateway.exe';Stopped=$false;Closed=$false}
$script:gatewayOriginal|Add-Member ScriptMethod Stop {$this.Stopped=$true}
$script:gatewayOriginal|Add-Member ScriptMethod Dispose {$this.Closed=$true}
$script:gatewayReplacement=[pscustomobject]@{Created=1235.5;Image='C:\other\gateway.exe';Stopped=$false}
$script:gatewayCurrent=$script:gatewayOriginal
function Open-CIOwnedGateway([int]$ProcessId) {$opened=$script:gatewayCurrent;$script:gatewayCurrent=$script:gatewayReplacement;return $opened}
Stop-CIOwnedGateway @{pid=100;create_time=1234.5} 'C:\owned\'
if(!$script:gatewayOriginal.Stopped -or !$script:gatewayOriginal.Closed -or $script:gatewayReplacement.Stopped){throw 'PID replacement reached unowned process'}
$script:gatewayCurrent=$script:gatewayReplacement;$script:gatewayReplacement|Add-Member ScriptMethod Dispose {}
$refused=$false;try{Stop-CIOwnedGateway @{pid=100;create_time=1234.5} 'C:\owned\'}catch{$refused=$true}
if(!$refused -or $script:gatewayReplacement.Stopped){throw 'Mismatched gateway incarnation accepted'}
function Open-CIOwnedGateway([int]$ProcessId) {throw [ComponentModel.Win32Exception]::new($ProcessId)}
Stop-CIOwnedGateway @{pid=87;create_time=1234.5} 'C:\owned\'
$refused=$false;try{Stop-CIOwnedGateway @{pid=5;create_time=1234.5} 'C:\owned\'}catch{$refused=$true}
if(!$refused){throw 'Gateway access-denied failure accepted as absence'}
'PASS: simulated PID replacement stops only validated held gateway handle; no process terminated'

# Simulated same-process SDK success and package APIs; never a native deployment.
$fixtureGraph=Join-Path ([IO.Path]::GetTempPath()) ('semantier-ci-graph-'+[Guid]::NewGuid().ToString('N')+'.json')
$fixtureOut=Join-Path ([IO.Path]::GetTempPath()) ('semantier-ci-records-'+[Guid]::NewGuid().ToString('N'))
try {
 New-Item -ItemType Directory -Path $fixtureOut -ErrorAction Stop|Out-Null
 $fixtureSid='S-1-5-21-1-2-3-4';$fixtureSource='b'*40
 $prefix='releases/channel-builds/'+('a'*32)+'/'
 $fixtureGraphObject=@{root=($prefix+'main.msixbundle');objects=@(@{kind='main-bundle';key=($prefix+'main.msixbundle');nativeDigest='digest-bundle'},@{kind='main-package';key=($prefix+'main.msix');nativeDigest='digest-main';identity='fixture-name';publisher='CN=Fixture';version='1.2.3.4';arch='x64'},@{kind='asset-package';key=($prefix+'resource.msix');nativeDigest='digest-resource';resourceId='runtime0001';arch='neutral'})}
 [IO.File]::WriteAllText($fixtureGraph,($fixtureGraphObject|ConvertTo-Json -Depth 32 -Compress))
 $fixtureGraphObject=Get-Content -Raw -LiteralPath $fixtureGraph|ConvertFrom-Json
 $fixtureHash=(Get-FileHash -LiteralPath $fixtureGraph -Algorithm SHA256).Hash.ToLowerInvariant()
 $fixtureUris=@($fixtureGraphObject.objects|ForEach-Object{'http://127.0.0.1:1777/'+$_.key});$fixtureDigests=@($fixtureGraphObject.objects|ForEach-Object{$_.nativeDigest})
 $self=Get-Process -Id $PID
 $fixtureSession=$self.SessionId;$fixtureActor=@{sid=$fixtureSid;session_sid=$fixtureSid;session=$fixtureSession;wts_state=0}
 function Invoke-BoundNativePackageUri([string[]]$Uris,[string[]]$Digests,[string]$Root,[int]$Minutes) {return @{status='SUCCESS';hresult='00000000';operationCompleted=$true;activityId='11111111-1111-1111-1111-111111111111'}}
 $rootUri='http://127.0.0.1:1777/'+$fixtureGraphObject.root
 $produced=Invoke-CIDurableNativeUri $fixtureOut $fixtureHash $fixtureSource 'positive' $fixtureActor $fixtureUris $fixtureDigests $rootUri 1
 $operationId=$produced.operationId
 $state=@{unknown=$false;nativeSuccess=@{actorSid=$fixtureSid;session=$fixtureSession;graphSha256=$fixtureHash;sourceCommit=$fixtureSource;root=$fixtureGraphObject.root;rootUri=$rootUri;uris=$fixtureUris;digests=$fixtureDigests;operationId=$operationId;result=$produced}}
 $intentPath=Join-Path $fixtureOut ('ci-native-intent-'+$operationId+'.json');$outcomePath=Join-Path $fixtureOut ('ci-native-outcome-'+$operationId+'.json')
 $intentText=[IO.File]::ReadAllText($intentPath);$outcomeText=[IO.File]::ReadAllText($outcomePath);$intent=$intentText|ConvertFrom-Json
 $package=[pscustomobject]@{PackageFullName='fixture-main';Name='fixture-name';PackageFamilyName='fixture-family';Publisher='CN=Fixture';Version=[version]'1.2.3.4';Architecture='x64';Status='Ok'}
 $resource=[pscustomobject]@{PackageFullName='fixture-resource';ResourceId='runtime0001';Publisher='CN=Fixture';Version=[version]'1.2.3.4';PackageFamilyName='fixture-family';Architecture='neutral';Status='Ok';SignatureKind='Developer';IsDevelopmentMode=$false}
 $verified=@{identity=@{packageFullName='fixture-main'};registration=@{graphSha256=$fixtureHash;sid=$fixtureSid;related=@(@{full='fixture-resource';resource_id='runtime0001'});physicalRoots=@(@{full='fixture-main';id='id-fixture-main'},@{full='fixture-resource';id='id-fixture-resource'})}}
 $held=@(@{full='fixture-main';root=@{id='id-fixture-main';handle=[pscustomobject]@{IsClosed=$false;IsInvalid=$false}}},@{full='fixture-resource';root=@{id='id-fixture-resource';handle=[pscustomobject]@{IsClosed=$false;IsInvalid=$false}}})
 function Assert-CISession { return @{sid=$fixtureSid;session_sid=$fixtureSid;session=$fixtureSession;wts_state=0;interactive=$true} }
 $script:changedPhysical=$false
 function Registered-Root {param($Package,[switch]$Retain) $id='id-'+$Package.PackageFullName;if($script:changedPhysical){$id='changed'};return @{id=$id;handle=[IO.MemoryStream]::new()}}
 $script:observedMain=$package;$script:observedResources=@($resource);$script:observedOptional=@()
 function Get-AppxPackage { param($PackageTypeFilter,$Name) if($PackageTypeFilter -ceq 'Resource'){return $script:observedResources};if($PackageTypeFilter -ceq 'Optional'){return $script:observedOptional};return @($script:observedMain) }
 Assert-CINativeCleanupOwnership $fixtureGraphObject 'fixture-main' $state $verified $package $fixtureSid $fixtureGraph $fixtureSource $held $fixtureOut
 foreach($damage in @('unknown','activity','operation','actor','graph','graph-object','package','resources','fresh-main','resource-arch','optional','physical','intent','outcome','missing-outcome','root-uri','uris','digests')){
  $copy=@{unknown=$state.unknown;nativeSuccess=$state.nativeSuccess.Clone()};$v=$verified;$p=$package;$g=$fixtureGraphObject
  $script:observedMain=$package;$script:observedResources=@($resource);$script:observedOptional=@();$script:changedPhysical=$false
  [IO.File]::WriteAllText($intentPath,$intentText);[IO.File]::WriteAllText($outcomePath,$outcomeText)
  switch($damage){
   'unknown' {$copy.unknown=$true}
   'activity' {$copy.nativeSuccess.result=@{status='SUCCESS';hresult='00000000';operationCompleted=$true;activityId=''}}
   'operation' {$copy.nativeSuccess.operationId='d'*32}
   'actor' {$copy.nativeSuccess.actorSid='other'}
   'graph' {$copy.nativeSuccess.graphSha256='0'*64}
   'graph-object' {$g=($fixtureGraphObject|ConvertTo-Json -Depth 32 -Compress|ConvertFrom-Json);$g.objects[2].resourceId='changed'}
   'package' {$p=[pscustomobject]@{PackageFullName='other'}}
   'resources' {$v=@{registration=@{graphSha256=$fixtureHash;sid=$fixtureSid;related=@()}}}
   'fresh-main' {$script:observedMain=$package.PSObject.Copy();$script:observedMain.PackageFullName='other'}
   'resource-arch' {$changed=$resource.PSObject.Copy();$changed.Architecture='x64';$script:observedResources=@($changed)}
   'optional' {$script:observedOptional=@([pscustomobject]@{PackageFullName='unowned'})}
   'physical' {$script:changedPhysical=$true}
   'intent' {$changed=$intentText|ConvertFrom-Json;$changed.actorPid=1;[IO.File]::WriteAllText($intentPath,($changed|ConvertTo-Json -Depth 32 -Compress))}
   'outcome' {$changed=@{schema=1;operationId=$operationId;result=@{operationId=$operationId;status='FAILURE';hresult='80000000';operationCompleted=$true;activityId='11111111-1111-1111-1111-111111111111'}};[IO.File]::WriteAllText($outcomePath,($changed|ConvertTo-Json -Depth 32 -Compress))}
   'missing-outcome' {Remove-Item -LiteralPath $outcomePath -Force}
   'root-uri' {$copy.nativeSuccess.rootUri='http://127.0.0.1:1778/'+$fixtureGraphObject.root}
   'uris' {$changed=$intentText|ConvertFrom-Json;$changed.uris=@('http://127.0.0.1:1778/unrelated');[IO.File]::WriteAllText($intentPath,($changed|ConvertTo-Json -Depth 32 -Compress))}
   'digests' {$changed=$intentText|ConvertFrom-Json;$changed.digests=@('wrong');[IO.File]::WriteAllText($intentPath,($changed|ConvertTo-Json -Depth 32 -Compress))}
  }
  $refused=$false;try{Assert-CINativeCleanupOwnership $g 'fixture-main' $copy $v $p $fixtureSid $fixtureGraph $fixtureSource $held $fixtureOut}catch{$refused=$true}
  if(!$refused){throw "CI cleanup accepted changed $damage"}
 }
 'PASS: simulated SDK-bound owned cleanup accepted; unknown and changed evidence refused; no package mutation'
 [IO.File]::WriteAllText($intentPath,$intentText);[IO.File]::WriteAllText($outcomePath,$outcomeText)
 $script:observedMain=$package;$script:observedResources=@($resource);$script:observedOptional=@();$script:changedPhysical=$false
 function New-FixtureRootHandle {$h=[pscustomobject]@{IsClosed=$false;IsInvalid=$false};$h|Add-Member ScriptMethod Dispose {$this.IsClosed=$true};return $h}
 function Registered-Root {param($Package,[switch]$Retain) return @{id=('id-'+$Package.PackageFullName);handle=(New-FixtureRootHandle)}}
 function Assert-BundleWinRTReady {param($Package,$ExpectedSid) return $true}
 $held=@(@{full='fixture-main';root=@{id='id-fixture-main';handle=(New-FixtureRootHandle)}},@{full='fixture-resource';root=@{id='id-fixture-resource';handle=(New-FixtureRootHandle)}})
 $script:removed=@()
 function Remove-AppxPackage {param($Package,$ErrorAction) $script:removed+=@($Package);$script:observedMain=$null;$script:observedResources=@()}
 function Get-AppxPackage {param($PackageTypeFilter,$Name) if($PackageTypeFilter -ceq 'Resource'){return $script:observedResources};if($PackageTypeFilter -ceq 'Optional'){return $script:observedOptional};if($null -ne $script:observedMain){return $script:observedMain}}
 $state.unknown=$true;$refused=$false
 try{Invoke-CINativeOwnedRemoval $fixtureGraphObject 'fixture-main' $state $verified $package $fixtureSid $fixtureGraph $fixtureSource ([ref]$held) $fixtureOut}catch{$refused=$true}
 if(!$refused -or $script:removed.Count -ne 0){throw 'Unresolved simulated operation reached removal'}
 $state.unknown=$false
 Invoke-CINativeOwnedRemoval $fixtureGraphObject 'fixture-main' $state $verified $package $fixtureSid $fixtureGraph $fixtureSource ([ref]$held) $fixtureOut
 if($script:removed.Count -ne 1 -or $script:removed[0] -cne 'fixture-main' -or $held.Count -ne 0){throw 'Owned simulated removal did not complete exactly once'}
 'PASS: production cleanup seam removes only proven fixture package; unresolved operation refuses; Windows package APIs simulated'
} finally {Remove-Item -LiteralPath $fixtureGraph -Force;Remove-Item -LiteralPath $fixtureOut -Recurse -Force}
