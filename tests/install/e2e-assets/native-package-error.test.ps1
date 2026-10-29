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
