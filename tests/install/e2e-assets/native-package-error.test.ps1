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
 $v=Invoke-CIDurableNativeUri $ciTmp ('a'*64) ('b'*40) 'positive' @{sid='fixture-sid';session_sid='fixture-sid';session=7;wts_state=0} @('http://127.0.0.1:1/pkg') @('digest') 'http://127.0.0.1:1/pkg' 1
 if(!$script:seen -or $v.status -cne 'SUCCESS' -or @(Get-ChildItem -LiteralPath $ciTmp -Filter 'ci-native-outcome-*.json').Count -ne 1){throw 'CI native durable transaction not completed'}
} finally {Remove-Item -LiteralPath $ciTmp -Recurse -Force}

. (Join-Path $PSScriptRoot 'native-ci-cleanup-policy.ps1')
$held=$false;try {Assert-CINativeCleanupOwnership @{root='native'} 'same-full-name'} catch {$held=$true}
if(!$held){throw 'Native cleanup incorrectly trusts a matching full name'}
$held=$false;try {Assert-CINativeCleanupOwnership $null ''} catch {$held=$true}
if(!$held){throw 'Cleanup incorrectly accepts an uncaptured registration'}
'PASS: uncertain CI native cleanup and uncaptured identity held; no package mutation'
