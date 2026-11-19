param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
foreach($name in @('Get-ValidatedNativeGit','Initialize-NativeBuildEnvironment','Set-ReleaseEnvironment')){
  $fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
  if(!$fn){throw "Missing function: $name"}
  . ([scriptblock]::Create($fn.Extent.Text))
}
function Require([bool]$Condition,[string]$Message){if(!$Condition){throw $Message}}
function Reject([scriptblock]$Action,[string]$Pattern){
  $message=''
  try{& $Action | Out-Null}catch{$message=$_.Exception.Message}
  Require ($message -match $Pattern) "Expected rejection matching $Pattern; got $message"
}
if(!$IsWindows){Write-Output 'Native Git behavior requires Windows; function parsing PASS';return}
$root=Join-Path ([IO.Path]::GetTempPath()) ('native-git-'+[guid]::NewGuid().ToString('N'))
$script:releaseEnvironment=@{}
try{
  New-Item -ItemType Directory $root | Out-Null
  Reject {Get-ValidatedNativeGit (Join-Path $root 'missing.exe')} 'missing'
  $invalid=Join-Path $root 'invalid.cmd'
  [IO.File]::WriteAllText($invalid,"@echo off`r`necho unexpected executable`r`nexit /b 0`r`n")
  Reject {Get-ValidatedNativeGit $invalid} 'version validation failed'
  $failed=Join-Path $root 'failed.cmd'
  [IO.File]::WriteAllText($failed,"@echo off`r`necho git version 2.56.0.windows.4`r`nexit /b 5`r`n")
  Reject {Get-ValidatedNativeGit $failed} 'exit=5'
  $shadow=Join-Path $root 'git.exe'
  [IO.File]::WriteAllText($shadow,'not executable')
  Set-ReleaseEnvironment 'PATH' ($root+';'+$env:PATH)
  Require ((Get-ValidatedNativeGit) -ceq 'C:\Program Files\Git\cmd\git.exe') 'Canonical selection followed a shadow executable'
  $git=Initialize-NativeBuildEnvironment
  Require ($git -ceq 'C:\Program Files\Git\cmd\git.exe') 'Build environment returned a noncanonical Git'
  Require ((Get-Command git.exe -CommandType Application).Source -ieq $git) 'VS environment shadowed canonical Git'
  Require ($env:PATH -notmatch '(?i)(?:^|;)D:\\Program Files\\Git(?:\\|;)') 'Build environment still depends on D: Git'
  & git.exe --version
  Require ($LASTEXITCODE -eq 0) 'Selected Git failed after VS initialization'
  Write-Output 'Native canonical Git: missing, invalid version, failed exit, PATH shadow and VS selection PASS'
}finally{
  foreach($key in $script:releaseEnvironment.Keys){[Environment]::SetEnvironmentVariable($key,$script:releaseEnvironment[$key],'Process')}
  if(Test-Path -LiteralPath $root){Remove-Item -LiteralPath $root -Recurse -Force}
}
