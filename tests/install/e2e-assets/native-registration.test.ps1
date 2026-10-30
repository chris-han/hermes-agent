# Real durable intent/result reader on disposable files. No SDK deployment.
param([Parameter(Mandatory=$true)][string]$PlatformScript)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($PlatformScript,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Platform parse failed'}
$type=@($ast.FindAll({param($n) $n -is [Management.Automation.Language.CommandAst] -and $n.GetCommandName() -ceq 'Add-Type' -and $n.Extent.Text.Contains('public static class SemantierNativeFacts')},$true))
if($type.Count -ne 1){throw 'Native facts declaration ambiguous'}
& ([scriptblock]::Create($type[0].Extent.Text))
$definitions=@($ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -ceq 'Physical'},$true))
if($definitions.Count -ne 1){throw 'Physical function ambiguous'}
. ([scriptblock]::Create($definitions[0].Extent.Text))
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;$session=(Get-Process -Id $PID).SessionId
$root=Join-Path ([IO.Path]::GetTempPath()) ('native-registration-test-'+[guid]::NewGuid().ToString('N'))
$attempt=Join-Path $root ('native-baseline-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($attempt)
$base='https://s3.semantier.com';$graph=@{root='releases/channel-builds/fixture/root.msixbundle';objects=@(@{key='releases/channel-builds/fixture/root.msixbundle';nativeDigest='fixture'})}
$spec=@{native_graph=@{sha256=('a'*64)}};$begin=[DateTimeOffset]::UtcNow.AddSeconds(-1)
try{
 $null=Write-NativeRegistrationIntent $attempt 'intent.json' $spec.native_graph.sha256 ($base+'/'+$graph.root) @($base+'/'+$graph.root) @('fixture')
 $refused=$false;try{$null=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $false}catch{$refused=$true}
 if(!$refused){throw 'Missing live SDK result incorrectly settled'}
 $result=Join-Path $attempt 'result.json'
 @{status='FAILURE';operationCompleted=$false}|ConvertTo-Json|Set-Content -LiteralPath $result -Encoding UTF8
 $refused=$false;try{$null=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $false}catch{$refused=$true}
 if(!$refused){throw 'Incomplete SDK failure incorrectly settled'}
 foreach($status in @('SUCCESS','FAILURE')){
  @{status=$status;operationCompleted=$true}|ConvertTo-Json|Set-Content -LiteralPath $result -Encoding UTF8
  $proof=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $false
  if($proof.status -cne $status -or !$proof.operationCompleted){throw 'Completed SDK result lost'}
 }
 $spec.native_graph.sha256='b'*64
 $refused=$false;try{$null=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $false}catch{$refused=$true}
 if(!$refused){throw 'Changed graph adopted old registration attempt'}
 $spec.native_graph.sha256='a'*64
 $refused=$false;try{$null=Read-NativeRegistrationOutcome $spec $graph $base $root ([DateTimeOffset]::UtcNow.AddMinutes(1)) $false}catch{$refused=$true}
 if(!$refused){throw 'Earlier attempt adopted by later intent'}
 $wd1=Join-Path $root ('native-deployment-'+[guid]::NewGuid().ToString('N'))
 $null=[IO.Directory]::CreateDirectory($wd1)
 $null=Write-NativeRegistrationIntent $wd1 'intent.json' $spec.native_graph.sha256 ($base+'/'+$graph.root) @($base+'/'+$graph.root) @('fixture')
 $legacy=@{status='FAILURE';errorSource='deployment-result';hresult='80070476';activityId='1c15d8c2-6377-0008-3f14-041e7763dd01';isRegistered=$false}
 $legacyPath=Join-Path $wd1 'correct-digest.json'
 $legacy|ConvertTo-Json|Set-Content -LiteralPath $legacyPath -Encoding UTF8
 $proof=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $true
 if($proof.status -cne 'FAILURE' -or !$proof.operationCompleted){throw 'Typed completed WD1 GetResults failure remained UNKNOWN'}
 foreach($change in @(@{errorSource='async-exception'},@{operationCompleted=$false},@{activityId='not-an-activity'})){
  $candidate=$legacy.Clone();foreach($key in $change.Keys){$candidate[$key]=$change[$key]}
  $candidate|ConvertTo-Json|Set-Content -LiteralPath $legacyPath -Encoding UTF8
  $refused=$false;try{$null=Read-NativeRegistrationOutcome $spec $graph $base $root $begin $true}catch{$refused=$true}
  if(!$refused){throw 'Incomplete or unbound WD1 failure incorrectly settled'}
 }
 @{scope='ACTUAL_NATIVE_REGISTRATION_FILES_NO_SDK_DEPLOYMENT';status='PASS';cases=10;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
}finally{Remove-Item -LiteralPath $root -Recurse -Force}
