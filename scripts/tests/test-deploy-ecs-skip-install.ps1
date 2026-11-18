$ErrorActionPreference='Stop'
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot '../deploy-ecs.ps1'),[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
if(!$ast.ParamBlock.Parameters.Where({$_.Name.VariablePath.UserPath -eq 'SkipLocalInstall'})){throw 'Missing explicit skip-install option'}
$gate=$ast.Find({param($n) $n -is [Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -eq '$SkipLocalInstall' -and $n.Extent.Text.Contains("`$receipt.localAcceptance='SKIPPED_USER_REQUEST_NO_LOCAL_INSTALL'")},$true)
if(!$gate){throw 'Missing local-install boundary'}
$receipt=@{}
$script:checkpoints=@{}
$SkipLocalInstall=$true
function Report-Status([string]$Message) {}
function Get-AppxPackage { throw 'LOCAL_INSTALL_PATH_REACHED' }
. ([scriptblock]::Create($gate.Extent.Text))
if($receipt.localAcceptance -ne 'SKIPPED_USER_REQUEST_NO_LOCAL_INSTALL'){throw 'Skipped acceptance was misreported'}
if($script:checkpoints.Count){throw 'Skipping must not forge installation checkpoints'}
$SkipLocalInstall=$false
$hit=$false
try { . ([scriptblock]::Create($gate.Extent.Text)) } catch { if($_.Exception.Message -eq 'LOCAL_INSTALL_PATH_REACHED'){$hit=$true}else{throw} }
if(!$hit){throw 'Default installation path changed'}
Write-Output 'PASS: explicit skip avoids AppX/profile/acceptance operations without forging checkpoints; default still installs'

$dependencies=$ast.Find({param($n) $n -is [Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -eq "`$BuildMode -eq 'Prod'" -and $n.Extent.Text.Contains('Prod acceptance requires -AcceptanceNodeModules')},$true)
if(!$dependencies){throw 'Missing Prod dependency boundary'}
$BuildMode='Prod';$NativePublishOnly=$true;$AcceptanceNodeModules='';$BuildPython='Test-PublicPython'
function Test-PublicPython {$global:LASTEXITCODE=0}
. ([scriptblock]::Create($dependencies.Extent.Text))
$NativePublishOnly=$false;$stopped=$false
try{. ([scriptblock]::Create($dependencies.Extent.Text))}catch{$stopped=$_.Exception.Message -like 'Prod acceptance requires*'}
if(!$stopped){throw 'Local acceptance dependency gate was weakened'}
'PASS: native publish-only needs public-check Python, not local UI node_modules; local acceptance remains gated'
