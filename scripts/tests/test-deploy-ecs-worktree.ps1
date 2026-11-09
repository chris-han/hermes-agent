param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'),[string]$OuterOperator='')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
  . ([scriptblock]::Create($fn.Extent.Text))
}
$native=$IsWindows
if(!$native){
  # Only the Git algorithm is portable. Windows fixed-drive and process
  # inspection remain native gates; no Linux result qualifies either gate.
  function git.exe { & /usr/bin/git @args }
  function Get-CimInstance { param($ClassName) return @() }
}
function Report-Status([string]$Message) {}
function Require([bool]$Condition,[string]$Message){if(!$Condition){throw $Message}}
function Reject([scriptblock]$Action,[string]$Message){
  $rejected=$false
  try{& $Action}catch{$rejected=$true}
  Require $rejected $Message
}
$root=Join-Path ([IO.Path]::GetTempPath()) ('semantier-worktree-'+[guid]::NewGuid().ToString('N'))
$Repository=Join-Path $root 'hermes-agent'
$Output=Join-Path $root 'run-one'
$BuildSource=Join-Path $Output 'source'
$BuildCache=Join-Path $root 'cache'
$utf8=[Text.UTF8Encoding]::new($false)
$WhatIfPreference=$false
$BuildMode='Prod'
$script:sourceLease=$null;$script:sourceOwner=$null;$script:sourceRecreated=$false
$script:statusLog=$null
$script:SourceBinding=@{runtimeHead='fixture';hermesGitlink='fixture';toolingHermesHead='fixture'}
New-Item -ItemType Directory -Path $Repository,$BuildCache | Out-Null
try {
  [IO.File]::WriteAllText((Join-Path $Repository '.gitignore'),"node_modules/`napps/`n")
  [IO.File]::WriteAllText((Join-Path $Repository 'app.txt'),'pinned')
  Run 'git.exe' @('-C',$Repository,'init','-q')
  Run 'git.exe' @('-C',$Repository,'add','.')
  Run 'git.exe' @('-C',$Repository,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','pinned')
  $Commit=Get-SourceGit $Repository @('rev-parse','HEAD')
  [IO.File]::WriteAllText((Join-Path $Repository 'app.txt'),'moving head')
  Run 'git.exe' @('-C',$Repository,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qam','moving')
  $protected=@{}
  foreach($name in @('cache/input','release/signed.msix','version-ledger.json')){
    $path=Join-Path $root $name
    $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
    [IO.File]::WriteAllText($path,'preserve '+$name)
    $protected[$path]=Hash $path
  }
  $WhatIfPreference=$true
  Initialize-LocalBuildSource $BuildSource $Commit
  Require (!(Test-Path -LiteralPath $Output)) 'WhatIf created output'
  $WhatIfPreference=$false
  Initialize-LocalBuildSource $BuildSource $Commit
  Require ($script:sourceRecreated) 'First stage not created'
  Require (!(Get-Item -Force (Join-Path $BuildSource '.git')).PSIsContainer) '.git is not a file'
  Require ((Get-SourceGit $BuildSource @('rev-parse','HEAD')) -ceq $Commit) 'Moving branch selected'
  $common=Assert-RegisteredSource $BuildSource $Commit
  Require ($common -ceq (Get-SourceGit $Repository @('rev-parse','--path-format=absolute','--git-common-dir'))) 'Objects not shared'
  Write-Output ('COMMON_DIR_EQUAL: '+$common)
  $prepared=Join-Path $Output 'desktop-job/prepared.json'
  $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $prepared))
  [IO.File]::WriteAllText($prepared,'stale paths')
  Reset-StalePreparation $prepared
  Require (!(Test-Path -LiteralPath $prepared)) 'Stale preparation survived reconstruction'
  [IO.File]::WriteAllText($prepared,'valid for this stage')
  $script:sourceRecreated=$false
  Reset-StalePreparation $prepared
  Require (Test-Path -LiteralPath $prepared) 'Warm preparation discarded'
  $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'))
  [IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'build output')
  Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Success cleanup failed'
  Require (!(Test-Path -LiteralPath $BuildSource)) 'Source still exists'
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  # Same run reconstructs the pinned source and invalidates only job receipt.
  Initialize-LocalBuildSource $BuildSource $Commit
  Require ($script:sourceRecreated) 'Resume did not reconstruct source'
  Reset-StalePreparation $prepared
  Require (!(Test-Path -LiteralPath $prepared)) 'Resume retained deleted-source receipt'
  $lease=Join-Path $Output '.source.lock'
  Reject { $other=[IO.File]::Open($lease,'OpenOrCreate','ReadWrite','None');$other.Dispose() } 'Concurrent lease accepted'
  # Interrupted run: release the handle without retirement, then recover it.
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  Initialize-LocalBuildSource $BuildSource $Commit
  Require ((Get-SourceGit $BuildSource @('rev-parse','HEAD')) -ceq $Commit) 'Interrupted recovery changed SHA'
  # Crash during retirement: the exact owned registration survives a missing
  # directory. Recovery removes only that entry and reconstructs the source.
  Remove-Item -LiteralPath $BuildSource -Recurse -Force
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  Initialize-LocalBuildSource $BuildSource $Commit
  Require ((Get-SourceGit $BuildSource @('rev-parse','HEAD')) -ceq $Commit) 'Missing-directory recovery failed'
  # Failure unwind uses the same cleanup as the operator outer finally.
  try{throw 'fixture failure'}catch{}finally{$clean=Remove-OwnedBuildSource}
  Require ($clean -ceq 'REMOVED') 'Failure cleanup failed'
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  # Different runs never share a source path. Foreign source is never adopted.
  $Output=Join-Path $root 'run-two';$BuildSource=Join-Path $Output 'source'
  $null=[IO.Directory]::CreateDirectory($BuildSource)
  [IO.File]::WriteAllText((Join-Path $BuildSource 'user.txt'),'foreign')
  Reject {Initialize-LocalBuildSource $BuildSource $Commit} 'Foreign source accepted'
  Require ((Remove-OwnedBuildSource) -ceq 'NOT_CREATED') 'Foreign source claimed'
  Require (Test-Path -LiteralPath (Join-Path $BuildSource 'user.txt')) 'Foreign content deleted'
  $script:sourceLease.Dispose();$script:sourceLease=$null
  $Output=Join-Path $root 'run-one';$BuildSource=Join-Path $Output 'source'
  Initialize-LocalBuildSource $BuildSource $Commit
  [IO.File]::WriteAllText((Join-Path $BuildSource 'app.txt'),'user edit')
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  Reject {Initialize-LocalBuildSource $BuildSource $Commit} 'Dirty recovered source accepted'
  Require ((Remove-OwnedBuildSource) -ceq 'NOT_CREATED') 'Dirty recovered stage claimed for cleanup'
  Require ([IO.File]::ReadAllText((Join-Path $BuildSource 'app.txt')) -ceq 'user edit') 'User edit lost'
  # Fixture repair, then prove wrong commit and wrong ownership refuse cleanup.
  Run 'git.exe' @('-C',$BuildSource,'restore','app.txt')
  $script:sourceLease.Dispose();$script:sourceLease=$null
  Initialize-LocalBuildSource $BuildSource $Commit
  Run 'git.exe' @('-C',$Repository,'worktree','lock',$BuildSource)
  Require ((Remove-OwnedBuildSource) -like 'CLEANUP_DEFERRED:*') 'Locked registration removed'
  Run 'git.exe' @('-C',$Repository,'worktree','unlock',$BuildSource)
  if($native){
    $busyScript=Join-Path $BuildSource 'wait-child.ps1'
    [IO.File]::WriteAllText($busyScript,'Wait-Event -Timeout 120 | Out-Null')
    $child=Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList @('-NoProfile','-File',$busyScript) -PassThru
    try {Require ((Remove-OwnedBuildSource) -like 'CLEANUP_DEFERRED:*') 'Busy stage removed'}
    finally {Stop-Process -Id $child.Id -Force;$child.WaitForExit()}
  }
  $savedCommit=$Commit;$Commit='a'*40
  Require ((Remove-OwnedBuildSource) -like 'CLEANUP_DEFERRED:*') 'Wrong commit cleanup allowed'
  $Commit=$savedCommit
  $script:sourceOwner.run='foreign'
  Require ((Remove-OwnedBuildSource) -like 'CLEANUP_DEFERRED:*') 'Foreign owner cleanup allowed'
  $script:sourceOwner.run=$Output
  Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Final cleanup failed'
  Require ((Get-SourceGit $Repository @('worktree','list','--porcelain')) -notmatch 'source') 'Registration residue'
  foreach($path in $protected.Keys){Require ((Hash $path) -ceq $protected[$path]) ('Protected input changed: '+$path)}
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  # Reparse targets are rejected before any stage ownership is acquired.
  $Output=Join-Path $root 'reparse';$BuildSource=Join-Path $Output 'source'
  $null=[IO.Directory]::CreateDirectory($Output)
  if($native){New-Item -ItemType Junction -Path $BuildSource -Target $BuildCache | Out-Null}
  else{New-Item -ItemType SymbolicLink -Path $BuildSource -Target $BuildCache | Out-Null}
  Reject {Initialize-LocalBuildSource $BuildSource $Commit} 'Reparse stage accepted'
  Remove-Item -LiteralPath $BuildSource -Force
  if($native){
    $UserProfile='';$CredentialsEnv='';$AcceptanceNodeModules='';$BuildPython='';$Node='';$SignTool=''
    $VersionState=Join-Path $root 'version-ledger.json';$SetupExe=Join-Path $root 'release/setup.exe'
    $Output='D:\repo\deliverables\work\ecs-deploy-20990101-000000-12345678';$BuildSource=''
    Set-RunBuildSource
    $one=$BuildSource
    $Output='D:\repo\deliverables\work\ecs-deploy-20990101-000000-87654321';$BuildSource=''
    Set-RunBuildSource
    Require ($BuildSource -ine $one) 'Run sources collided'
    $BuildSource=$one
    Reject {Set-RunBuildSource} 'Cross-run source accepted'
    $Output='D:\repo\deliverables\work';$BuildSource=''
    Reject {Set-RunBuildSource} 'Work root accepted as run'
  }
  # Exercise the actual outer operator try/finally and unsigned early return.
  # Only unavailable tool discovery and packaging are substituted here.
  $outerAst=$ast
  if($OuterOperator){$outerAst=[Management.Automation.Language.Parser]::ParseFile($OuterOperator,[ref]$null,[ref]$errors);if($errors){throw ($errors | Out-String)}}
  $main=@($outerAst.EndBlock.Statements | Where-Object {$_ -is [Management.Automation.Language.TryStatementAst]})[-1]
  $script:releaseEnvironment=@{}
  $Latest=$false;$BuildOnly=$true;$SetupOnly=$false;$LocalSetup=$false;$UpdateSetup=$false
  $PreflightOnly=$false;$DesktopOnly=$false;$NewRun=$true
  function Initialize-ReleasePaths {}
  function Set-RunBuildSource {}
  function Build-CommitPackage {
    Initialize-LocalBuildSource $BuildSource $Commit
    if($script:failPackaging){throw 'fixture packaging failure'}
    return [pscustomobject]@{version='0.0.99';msix=(Join-Path $Output 'unsigned.msix');sha256=('b'*64);source=$BuildSource}
  }
  function Sign-Copy {throw 'Signing reached in BuildOnly'}
  foreach($failure in @($false,$true)){
    $script:failPackaging=$failure
    $Output=Join-Path $root ('outer-'+$failure);$BuildSource=Join-Path $Output 'source'
    $caught=$false
    try{& ([scriptblock]::Create($main.Extent.Text))}catch{
      if($_.Exception.Message -cne 'fixture packaging failure'){throw}
      $caught=$true
    }
    Require ($caught -eq $failure) 'Packaging failure did not propagate'
    Require (!(Test-Path -LiteralPath $BuildSource)) 'Outer finally left source'
    Require ($null -eq $script:sourceLease) 'Outer finally left lease'
    if($failure){
      $failed=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw | ConvertFrom-Json
      Require ($failed.result -ceq 'FAIL' -and $failed.sourceCleanup -ceq 'REMOVED') 'Failure receipt/cleanup missing'
    }
    if(!$failure){
      $result=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw | ConvertFrom-Json
      Require ($result.result -ceq 'UNSIGNED_LOCAL_BUILD' -and $result.sourceCleanup -ceq 'REMOVED') 'Unsigned receipt/cleanup missing'
      foreach($stage in @('signing','installation','publication')){Require ($result.$stage -ceq 'NOT_RUN') ('Unexpected '+$stage)}
    }
    $script:sourceOwner=$null
  }
  $originalBuild=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Build-CommitPackage'},$true)
  . ([scriptblock]::Create($originalBuild.Extent.Text))
  $Output=Join-Path $root 'outer-whatif';$BuildSource=Join-Path $Output 'source'
  $BuildPython=(Get-Process -Id $PID).Path;$Node=$BuildPython
  $CredentialsEnv='';$LocalOnly=$false;$PublicBase='https://download.semantier.cn'
  $script:SourceBinding=@{runtimeHead='fixture';hermesGitlink=$Commit;toolingHermesHead=$Commit}
  $WhatIfPreference=$true
  & ([scriptblock]::Create($main.Extent.Text)) | Out-Null
  Require (!(Test-Path -LiteralPath $Output)) 'Outer WhatIf created a run'
  Require ($null -eq $script:sourceLease) 'Outer WhatIf acquired a lease'
  foreach($path in $protected.Keys){Require ((Hash $path) -ceq $protected[$path]) ('WhatIf changed protected input: '+$path)}
  $WhatIfPreference=$false
  Write-Output 'PASS: exact SHA, shared objects, linked metadata, unique paths, WhatIf, success/failure cleanup, interrupted resume, foreign/dirty/lease guards, stale preparation, protected cache/release/ledger'
  if(!$native){Write-Output 'WINDOWS_NATIVE: NOT_RUN (portable Git algorithm only)'}
} finally {
  if($script:sourceLease){$script:sourceLease.Dispose()}
  # This fixture owns the entire unique temp root, unlike deployment cleanup.
  & git.exe -C $Repository worktree remove --force (Join-Path $root 'run-one/source') 2>$null
  Remove-Item -LiteralPath $root -Recurse -Force
}
