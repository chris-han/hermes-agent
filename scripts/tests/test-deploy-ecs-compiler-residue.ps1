param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'),[string]$ForeignOwnerFixtureFile='')
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($fn.Extent.Text))}
function Require([bool]$Value,[string]$Message){if(!$Value){throw $Message}}
function Reject([scriptblock]$Action,[string]$Expected='*'){$reason='';try{& $Action}catch{$reason=$_.Exception.Message};Require ($reason -and $reason -like $Expected) ('Expected refusal '+$Expected+'; got '+$reason)}
$root=Join-Path ([IO.Path]::GetTempPath()) ('compiler-residue-'+[guid]::NewGuid().ToString('N'))
$BuildSource=Join-Path $root 'source';$BuildCache=Join-Path $root 'cache'
$target=Join-Path $BuildCache 'react-compiler/desktop';$link=Join-Path $BuildSource 'apps/desktop/node_modules/.cache/hermes-react-compiler'
$generated=Join-Path $BuildSource 'node_modules/generated'
try {
 $null=[IO.Directory]::CreateDirectory((Split-Path $generated -Parent));[IO.File]::WriteAllText($generated,'generated')
 $old=Get-ResidueInventory $BuildSource
 $null=[IO.Directory]::CreateDirectory($target);[IO.File]::WriteAllText((Join-Path $target 'retained'),'shared cache')
 $null=[IO.Directory]::CreateDirectory((Split-Path $link -Parent))
 if($IsWindows){New-Item -ItemType Junction -Path $link -Target $target|Out-Null}
 else {New-Item -ItemType SymbolicLink -Path $link -Target $target|Out-Null}
 if($IsWindows){
  $inventory=Get-ResidueInventory $BuildSource
  Require ($inventory -cne $old) 'Compiler layout omitted from inventory'
  Require ((Get-Content (Join-Path $target 'retained') -Raw) -ceq 'shared cache') 'Inventory altered shared target'
  Require ($ForeignOwnerFixtureFile -and (Test-Path $ForeignOwnerFixtureFile -PathType Leaf)) 'Native suite requires genuine prepared foreign ACL fixture'
  $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $foreign=(Get-Acl $ForeignOwnerFixtureFile).GetOwner([Security.Principal.SecurityIdentifier]).Value
  Require ($foreign -cne $sid) 'Foreign fixture is current owned'
  $foreignHash=Hash $ForeignOwnerFixtureFile
  [IO.File]::Delete($generated);[IO.File]::Move($ForeignOwnerFixtureFile,$generated)
  try {Reject {Get-ResidueInventory $BuildSource} 'Foreign residue ownership.'}
  finally {[IO.File]::Move($generated,$ForeignOwnerFixtureFile);[IO.File]::WriteAllText($generated,'generated')}
  Require ((Hash $ForeignOwnerFixtureFile) -ceq $foreignHash -and (Get-Acl $ForeignOwnerFixtureFile).GetOwner([Security.Principal.SecurityIdentifier]).Value -ceq $foreign) 'Foreign fixture changed'
  [IO.File]::WriteAllText((Join-Path $BuildSource 'apps/unexpected'),'reject')
  Reject {Get-ResidueInventory $BuildSource}
  [IO.File]::Delete((Join-Path $BuildSource 'apps/unexpected'))
  $saved=Join-Path $root 'preserved';[IO.Directory]::Move($BuildSource,$saved)
  Require ((Get-ResidueInventory $saved) -ceq $inventory) 'Preserved inventory changed'
  Require ((Get-Content (Join-Path $target 'retained') -Raw) -ceq 'shared cache') 'Preservation followed or changed shared target'
  [IO.Directory]::Move($saved,$BuildSource)
  [IO.Directory]::Delete($link)
  Reject {Get-ResidueInventory $BuildSource}
  $wrong=Join-Path $root 'foreign-target';$null=[IO.Directory]::CreateDirectory($wrong)
  New-Item -ItemType Junction -Path $link -Target $wrong|Out-Null
  Reject {Get-ResidueInventory $BuildSource}
  [IO.Directory]::Delete($link)
  New-Item -ItemType HardLink -Path $link -Target (Join-Path $target 'retained')|Out-Null
  Reject {Get-ResidueInventory $BuildSource}
  [IO.File]::Delete($link)
  Write-Output 'WINDOWS_NATIVE: compiler junction preservation, genuine foreign ACL, wrong target/type, unexpected file and incomplete scaffold PASS'
 } else {
  Reject {Get-ResidueInventory $BuildSource}
  [IO.File]::Delete($link)
  Write-Output 'WINDOWS_NATIVE: NOT_RUN; portable symbolic-link refusal only'
 }
 [IO.Directory]::Delete((Split-Path $link -Parent));[IO.Directory]::Delete((Split-Path (Split-Path $link -Parent) -Parent));[IO.Directory]::Delete((Join-Path $BuildSource 'apps/desktop'));[IO.Directory]::Delete((Join-Path $BuildSource 'apps'))
 Require ((Get-ResidueInventory $BuildSource) -ceq $old) 'Legacy node_modules inventory hash changed'
 Write-Output 'PASS: compiler residue bounds and unchanged legacy inventory'
} finally {
 if(Test-Path $link){[IO.Directory]::Delete($link)}
 if(Test-Path $root){Remove-Item -LiteralPath $root -Recurse -Force}
}
