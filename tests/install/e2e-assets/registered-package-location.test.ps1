# Read-only physical identity checks against an existing OS package-volume redirect.
param([Parameter(Mandatory)][string]$LogicalRoot,[Parameter(Mandatory)][string]$ExpectedPhysicalRoot)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'semantier-registered-platform.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Platform parser failure'}
$type=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Add-Type' -and $node.Extent.Text.Contains('public static class SemantierNativeFacts')},$true))
if($type.Count -ne 1){throw 'Native physical API definition missing'}
Invoke-Expression $type[0].Extent.Text
foreach($name in @('Physical','Registered-Root','Installed-File')){
 $function=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name},$true))
 if($function.Count -ne 1){throw 'Physical package function missing'}
 Invoke-Expression $function[0].Extent.Text
}
function Reject([scriptblock]$Action){$refused=$false;try{& $Action|Out-Null}catch{$refused=$true};if(-not $refused){throw 'Unsafe package path accepted'}}
$package=[pscustomobject]@{InstallLocation=$LogicalRoot;PackageFullName=[IO.Path]::GetFileName($LogicalRoot.TrimEnd('\'))}
# RED reproducer: generic strict Physical must continue refusing the OS junction.
Reject { Physical $LogicalRoot }
$resolved=Registered-Root $package
if($resolved -ine $ExpectedPhysicalRoot){throw 'Wrong OS package store selected'}
$held=Physical $resolved
try{if(-not $held.id){throw 'Physical identity absent'}}finally{$held.handle.Dispose()}
$wrong=$package.PSObject.Copy();$wrong.PackageFullName='Wrong.Package'
Reject { Registered-Root $wrong }
$outside=$package.PSObject.Copy();$outside.InstallLocation=Join-Path ([IO.Path]::GetTempPath()) $package.PackageFullName
Reject { Registered-Root $outside }
# OS volume records are the external boundary. Ambiguous or offline destinations refuse.
$volumes=@(Get-AppxVolume)
function Get-AppxVolume { return @($volumes)+@($volumes) }
Reject { Registered-Root $package }
Remove-Item Function:\Get-AppxVolume
function Get-AppxVolume { return @($volumes|ForEach-Object{[pscustomobject]@{PackageStorePath=$_.PackageStorePath;IsOffline=$true}}) }
Reject { Registered-Root $package }
Remove-Item Function:\Get-AppxVolume
Write-Output 'PASS: exact OS package-volume redirect and physical identity; arbitrary/name/ambiguous/offline destinations refused; strict Physical still refuses junctions'

$aliasFile=Join-Path $LogicalRoot 'app\Semantier Agent.exe'
$physicalFile=Join-Path $ExpectedPhysicalRoot 'app\Semantier Agent.exe'
$a=Installed-File $package $aliasFile;$b=Physical $physicalFile
try{if($a.id -cne $b.id -or $a.path -ine $physicalFile){throw 'Registered logical process file differs from physical executable'}}finally{$a.handle.Dispose();$b.handle.Dispose()}
Reject { Installed-File $package (Join-Path ([IO.Path]::GetTempPath()) 'other.exe') }
Write-Output 'PASS: logical process executable matches exact physical file ID; unrelated process path refused'
