# Read-only observation of this test process; not CI or deployment acceptance.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
# Simulated projection controls distinguish absent members from failed getters.
$simulated=[pscustomobject]@{}
$simulated|Add-Member -MemberType ScriptProperty -Name Servicing -Value {throw 'simulated getter failure'}
$simulated|Add-Member -MemberType ScriptMethod -Name VerifyIsOK -Value {return $true}
$simulatedFields=Read-CIStatusFields $simulated
if($simulatedFields.DeploymentInProgress.state -cne 'UNAVAILABLE' -or $simulatedFields.Servicing.state -cne 'QUERY_ERROR' -or $simulatedFields.VerifyIsOK.state -cne 'VALUE'){throw 'Unavailable and failed status getters were conflated'}
$fixturePackage=@(Get-AppxPackage | Select-Object -First 1)
if($fixturePackage.Count -ne 1){throw 'No current-user package available for read-only WinRT fixture'}
$snapshot=Get-CIWinRTPackageStatus ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) $fixturePackage[0].PackageFullName
if($snapshot.presence -cne 'PRESENT'){throw 'WinRT package lookup did not find current-user fixture'}
if(!$snapshot.startedAt -or !$snapshot.endedAt -or $snapshot.identity.full -cne $fixturePackage[0].PackageFullName){throw 'WinRT snapshot time or exact identity unavailable'}
if($snapshot.installedLocation.state -cne 'VALUE' -or !$snapshot.installedLocation.path){throw 'WinRT installed location query unavailable'}
foreach($name in @('DeploymentInProgress','Servicing','IsPartiallyStaged','VerifyIsOK')){
 if(!$snapshot.status.ContainsKey($name) -or !$snapshot.status[$name].available -or $snapshot.status[$name].value -isnot [bool]){throw "WinRT status property unavailable: $name"}
}
$missing=Get-CIWinRTPackageStatus ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) 'Semantier.NotPresent_1.0.0.0_x64__p2m4tf4r6sydm'
if($missing.presence -cne 'NOT_FOUND'){throw 'WinRT missing package not distinguished'}
$denied=Get-CIWinRTPackageStatus 'S-1-5-21-1-2-3-4' $fixturePackage[0].PackageFullName
if($denied.presence -cne 'QUERY_ERROR'){throw 'Cross-user query mismatch was not refused'}
$self=Get-Process -Id $PID
$intent=@{actor=@{sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;session=$self.SessionId};root='http://127.0.0.1:65534/test';createdAt=[DateTime]::UtcNow.ToString('o');actorPid=$PID;actorBirth=$self.StartTime.ToUniversalTime().ToString('o');actorExecutable=$self.Path;observationExpectedFullNames=@($fixturePackage[0].PackageFullName,'Semantier.NotPresent_1.0.0.0_x64__p2m4tf4r6sydm')}
$result=Get-CINativeObservation $intent
if($result.process -cne 'same-incarnation'){throw 'Current native process was not identified'}
if($result.eventQuery.coverage -cne 'NOT_ESTABLISHED'){throw 'Event query must not claim retained log coverage'}
if($result.processIdentity -cne 'verified'){throw 'Current Windows process owner/session/executable not verified'}
if(@($result.expectedPackages).Count -ne 2 -or $result.expectedPackages[0].presence -cne 'PRESENT' -or $result.expectedPackages[1].presence -cne 'NOT_FOUND'){throw 'Expected package observations did not distinguish missing member'}
function Get-CimInstance { throw 'simulated process query denial' }
try {
 $withoutProcess=Get-CINativeObservation $intent
 if($withoutProcess.process -cne 'unknown' -or $withoutProcess.processIdentity -cne 'QUERY_ERROR' -or $withoutProcess.expectedPackages[0].presence -cne 'PRESENT'){throw 'Process query failure was conflated with absence or discarded package state'}
} finally { Remove-Item Function:\Get-CimInstance }
function Get-WinEvent { throw 'simulated log access failure' }
try {
 $withoutEvents=Get-CINativeObservation $intent
 if($withoutEvents.eventQuery.state -cne 'QUERY_ERROR' -or $withoutEvents.expectedPackages[0].presence -cne 'PRESENT'){throw 'Log failure discarded independent current package state'}
} finally { Remove-Item Function:\Get-WinEvent }
$intent.actorExecutable='C:\not-the-recorded-executable.exe'
$mismatch=Get-CINativeObservation $intent
if($mismatch.processIdentity -cne 'mismatch'){throw 'Changed executable was not refused'}
$intent.actorExecutable=$self.Path
$intent.actor.session=$self.SessionId+1
$mismatchSession=Get-CINativeObservation $intent
if($mismatchSession.processIdentity -cne 'mismatch'){throw 'Changed process session was not refused'}
$intent.actor.session=$self.SessionId
$intent.actor.sid='S-1-5-21-1-2-3-4'
$mismatchSid=Get-CINativeObservation $intent
if($mismatchSid.processIdentity -cne 'mismatch'){throw 'Changed process owner was not refused'}
$intent.actor.sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
if($result.scope -cne 'read-only CI observation; expected names are query inputs, not graph authority'){throw 'Observation scope changed'}
$intent.actorBirth='2000-01-01T00:00:00Z'
$different=Get-CINativeObservation $intent
if($different.process -cne 'different-incarnation'){throw 'PID reuse was not distinguished'}
$intent.actorPid=2147483647
$absent=Get-CINativeObservation $intent
if($absent.process -cne 'absent'){throw 'Missing process was not reported absent'}
$intent.root='https://other.example/object'
$refused=$false
try { Get-CINativeObservation $intent | Out-Null } catch { $refused=$true }
if(!$refused){throw 'Unscoped URI was accepted'}
$directory=Join-Path ([IO.Path]::GetTempPath()) ('semantier-ci-observation-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $directory | Out-Null
try {
 # Substitute only the Windows deployment call; the real CI intent writer runs.
 function Invoke-BoundNativePackageUri { return @{status='SIMULATED';operationCompleted=$false} }
 $actor=@{sid=$intent.actor.sid;session_sid=$intent.actor.sid;session=$self.SessionId;wts_state=0}
 $uri='http://127.0.0.1:65534/test';$digest=[Convert]::ToBase64String([byte[]]::new(32))
 $null=Invoke-CIDurableNativeUri $directory ('a'*64) ('b'*40) 'positive' $actor @($uri) @($digest) $uri 1
 $files=@(Get-ChildItem -LiteralPath $directory -Filter 'ci-native-intent-*.json')
 if($files.Count -ne 1){throw 'Expected exactly one durable CI intent'}
 $written=Get-Content -Raw -LiteralPath $files[0].FullName | ConvertFrom-Json
 if($written.actorExecutable -ine $self.Path -or $written.actor.session -ne $self.SessionId -or $written.actor.sid -cne $actor.sid){throw 'CI producer did not retain actual actor identity'}
} finally { Remove-Item -LiteralPath $directory -Recurse -Force }
Write-Output 'PASS: native process identity observations and simulated deployment intent producer; no actual deployment or package mutation'
