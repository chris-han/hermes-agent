$ErrorActionPreference='Stop'
$operator=Join-Path $PSScriptRoot '../deploy-ecs.ps1'
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($operator,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
$fn=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-ProfileFileHolders'},$true)
if(!$fn){throw 'Profile holder lookup is missing'}
. ([scriptblock]::Create($fn.Extent.Text))
$stop=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Stop-ProfileBrowserHolders'},$true)
if(!$stop){throw 'Profile browser cleanup is missing'}
. ([scriptblock]::Create($stop.Extent.Text))
function Report-Status([string]$Message) {}
$root=Join-Path ([IO.Path]::GetTempPath()) ('profile-lock-'+[guid]::NewGuid().ToString('N'))
$browser=Join-Path $root 'browser-profile/edge/Default'
$null=New-Item -ItemType Directory $browser -Force
$file=Join-Path $browser 'Cookies'
[IO.File]::WriteAllText($file,'fixture')
$marker=Join-Path $root 'ready'
$child=$null
try {
  $command='$f=[IO.File]::Open($args[0],"Open","ReadWrite","None"); [IO.File]::WriteAllText($args[1],"ready"); Start-Sleep -Seconds 30'
  $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("& { $command } '$file' '$marker'"))
  $child=Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList '-NoProfile','-EncodedCommand',$encoded -PassThru -WindowStyle Hidden
  for($i=0;$i -lt 100 -and !(Test-Path $marker);$i++){Start-Sleep -Milliseconds 100}
  if(!(Test-Path $marker)){throw 'Lock fixture did not start'}
  $holders=@(Get-ProfileFileHolders @($file))
  if($child.Id -notin $holders.Id){throw 'Actual file holder was not identified'}
  if($PID -in $holders.Id){throw 'Unrelated process identified as holder'}
  if(($holders | Where-Object Id -eq $child.Id).Started -ne $child.StartTime.ToUniversalTime()){throw 'Holder start time does not match'}
  $rejected=$false
  try { Stop-ProfileBrowserHolders $root } catch { $rejected=$true }
  if(!$rejected -or $child.HasExited){throw 'Non-browser holder must be reported without terminating it'}
  $script:stopped=@()
  $script:fakeProcess=[pscustomobject]@{Id=$child.Id;ProcessName='msedge';StartTime=$child.StartTime;HasExited=$false}
  $script:fakeProcess | Add-Member ScriptMethod WaitForExit {}
  function Get-Process { param($Id,$ErrorAction) return $script:fakeProcess }
  function Stop-Process { param($Id,[switch]$Force,$ErrorAction) $script:stopped+= $Id }
  $count=Stop-ProfileBrowserHolders $root
  if($count -ne 1 -or $script:stopped.Count -ne 1 -or $script:stopped[0] -ne $child.Id){throw 'Matching browser holder was not stopped'}
  Remove-Item function:Get-Process,function:Stop-Process
  Write-Host 'Profile holder lookup passed.'
} finally {
  Remove-Item function:Get-Process,function:Stop-Process -ErrorAction SilentlyContinue
  if($child -and !$child.HasExited){Stop-Process -Id $child.Id -Force; $child.WaitForExit()}
  Remove-Item -LiteralPath $root -Recurse -Force
}
