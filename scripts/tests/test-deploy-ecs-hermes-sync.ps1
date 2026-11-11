$ErrorActionPreference='Stop'
$operator=Join-Path $PSScriptRoot '../deploy-ecs.ps1'
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($operator,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
foreach($name in @('Run','Get-SourceGit','Save-HermesChanges')) {
  $fn=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
  if(!$fn){throw "Missing function: $name"}
  . ([scriptblock]::Create($fn.Extent.Text))
}
function Report-Status([string]$Message) {}
$root=Join-Path ([IO.Path]::GetTempPath()) ('hermes-sync-'+[guid]::NewGuid().ToString('N'))
$repo=Join-Path $root 'repo'; $remote=Join-Path $root 'remote.git'
try {
  New-Item -ItemType Directory $repo -Force | Out-Null
  & git.exe init --bare --initial-branch=main $remote | Out-Null
  & git.exe -C $repo init --initial-branch=main | Out-Null
  & git.exe -C $repo config user.name Fixture
  & git.exe -C $repo config user.email fixture@example.invalid
  & git.exe -C $repo remote add origin $remote
  [IO.File]::WriteAllText((Join-Path $repo 'tracked.txt'),'before')
  & git.exe -C $repo add tracked.txt
  & git.exe -C $repo commit -qm initial
  & git.exe -C $repo push -q origin main
  [IO.File]::WriteAllText((Join-Path $repo 'tracked.txt'),'after')
  [IO.File]::WriteAllText((Join-Path $repo 'untracked.txt'),'new')
  Save-HermesChanges $repo
  if(Get-SourceGit $repo @('status','--porcelain','--untracked-files=all')){throw 'Hermes remains dirty'}
  if((Get-SourceGit $repo @('rev-parse','HEAD')) -cne (Get-SourceGit $remote @('rev-parse','refs/heads/main'))){throw 'Remote did not receive commit'}
  if((Get-SourceGit $repo @('show','HEAD:tracked.txt')) -cne 'after' -or (Get-SourceGit $repo @('show','HEAD:untracked.txt')) -cne 'new'){throw 'Changes missing from commit'}
  Write-Host 'Hermes synchronization passed.'
} finally {
  if(Test-Path -LiteralPath $root){Remove-Item -LiteralPath $root -Recurse -Force}
}
