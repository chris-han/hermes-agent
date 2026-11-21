param([Parameter(Mandatory=$true)][string]$Operator)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Operator parse failed'}
$found=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Install-LocalPilotPackage'},$true))
if($found.Count -ne 1){throw 'Install owner is missing or ambiguous'}
Invoke-Expression $found[0].Extent.Text
$identity='Semantier.Agent.Acceptance';$publisher='CN=Semantier Technology, O=Semantier Technology';$Version='0.0.136.0'
$script:current=@();$script:calls=0
function Get-AppxPackage {param([string]$Name) if($Name -cne $identity){throw 'Wrong current-user identity'};return $script:current}
function Add-AppxPackage {param([string]$Path,$Volume,[switch]$ForceTargetApplicationShutdown)
 if($Path -cne 'D:\fixture\signed.msix' -or $Volume -cne 'system-volume' -or !$ForceTargetApplicationShutdown){throw 'Deployment arguments changed'}
 $script:calls++
}
# Valid artifact/source/signature checks are caller-owned; only external Windows deployment is substituted.
Install-LocalPilotPackage 'D:\fixture\signed.msix' 'system-volume'
if($script:calls -ne 1){throw 'Clean current-user install did not deploy'}
$script:current=@([pscustomobject]@{Publisher=$publisher;Version='0.0.135.0'})
Install-LocalPilotPackage 'D:\fixture\signed.msix' 'system-volume'
if($script:calls -ne 2){throw 'Newer upgrade did not deploy'}
foreach($old in @('0.0.136.0','0.0.137.0')){
 $script:current=@([pscustomobject]@{Publisher=$publisher;Version=$old});$refused=$false
 try{Install-LocalPilotPackage 'D:\fixture\signed.msix' 'system-volume'}catch{$refused=$true}
 if(!$refused -or $script:calls -ne 2){throw 'Same-version/downgrade guard failed'}
}
$script:current=@([pscustomobject]@{Publisher='CN=Other';Version='0.0.135.0'});$refused=$false
try{Install-LocalPilotPackage 'D:\fixture\signed.msix' 'system-volume'}catch{$refused=$true}
if(!$refused -or $script:calls -ne 2){throw 'Foreign publisher guard failed'}
$script:current=@([pscustomobject]@{Publisher=$publisher;Version='0.0.134.0'},[pscustomobject]@{Publisher=$publisher;Version='0.0.135.0'});$refused=$false
try{Install-LocalPilotPackage 'D:\fixture\signed.msix' 'system-volume'}catch{$refused=$true}
if(!$refused -or $script:calls -ne 2){throw 'Ambiguous package guard failed'}
'LOCAL_CLEAN_INSTALL_AND_UPGRADE_GUARDS_PASS'
