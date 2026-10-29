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
