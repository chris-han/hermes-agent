# Actual process-scope logic; only SDK/CIM facts are fixtures. Never kills or removes.
param([Parameter(Mandatory=$true)][string]$PlatformScript)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$tokens=$null;$errors=$null;$ast=[Management.Automation.Language.Parser]::ParseFile($PlatformScript,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Platform parser failed'}
$fixture=@(
 [pscustomobject]@{ProcessId=1;ExecutablePath='D:\WindowsApps\main\app\renderer.exe'},
 [pscustomobject]@{ProcessId=2;ExecutablePath='C:\Program Files\WindowsApps\main\app\renderer.exe'},
 [pscustomobject]@{ProcessId=3;ExecutablePath='D:\WindowsApps\asset\app\tool.exe'},
 [pscustomobject]@{ProcessId=4;ExecutablePath='C:\Program Files\WindowsApps\asset\app\tool.exe'},
 [pscustomobject]@{ProcessId=5;ExecutablePath='D:\WindowsApps\asset-other\app\tool.exe'},
 [pscustomobject]@{ProcessId=6;ExecutablePath=$null}
)
function Get-CimInstance {param($ClassName) return $fixture}
$state=@{package=@{install_path='D:\WindowsApps\main';logical_install_path='C:\Program Files\WindowsApps\main'};related=@(@{install_path='D:\WindowsApps\asset';logical_install_path='C:\Program Files\WindowsApps\asset'})}
$definitions=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Package-GraphProcesses'},$true))
if($definitions.Count -ne 1){throw 'Missing complete graph process check: old main-only predicate misses logical/asset processes'}
. ([scriptblock]::Create($definitions[0].Extent.Text))
$rows=@(Package-GraphProcesses $state)
if(($rows.ProcessId -join '|') -cne '1|2|3|4'){throw 'Graph process scope missed aliases/assets or included sibling roots'}
$state.related[0].Remove('logical_install_path')
$refused=$false;try{$null=Package-GraphProcesses $state}catch{$refused=$true}
if(!$refused){throw 'Incomplete observed graph root metadata accepted'}
@{scope='ACTUAL_GRAPH_PROCESS_SCOPE_WITH_OS_FACT_FIXTURES_NO_KILL_OR_REMOVAL';status='PASS';cases=6;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
