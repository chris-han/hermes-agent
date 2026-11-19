param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'),[string]$OuterOperator='',[string]$ForeignOwnerFixtureFile='')
$ErrorActionPreference='Stop'
$ResumeSigned=$false
$RetireRollbackBuild='';$RetireRollbackSha256='';$ChannelRecoveryMsix=''
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
  [IO.File]::WriteAllText((Join-Path $Repository 'package-lock.json'),'{"packages":{"node_modules/fixture-workspace":{"link":true,"resolved":"apps/workspace"}}}')
  [IO.File]::WriteAllText((Join-Path $Repository 'MODEL-NOTICE.txt'),"LF notice`n")
  Run 'git.exe' @('-C',$Repository,'init','-q')
  Run 'git.exe' @('-C',$Repository,'add','.')
  Run 'git.exe' @('-C',$Repository,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','pinned')
  $Commit=Get-SourceGit $Repository @('rev-parse','HEAD')
  Run 'git.exe' @('-C',$Repository,'config','core.autocrlf','true')
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
  Require ((Hash (Join-Path $BuildSource 'MODEL-NOTICE.txt')) -ceq (Hash (Join-Path $Repository 'MODEL-NOTICE.txt'))) 'MODEL-NOTICE LF integrity lost'
  Require ((Get-SourceGit $Repository @('config','--local','core.autocrlf')) -ceq 'true') 'Operator changed repository Git configuration'
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
  # An interrupted retirement can leave only generated dependencies, with no
  # registration. The retained reservation/request are authority, never recreated.
  $VersionState=Join-Path $root 'version-ledger.json'
  $RecoveryMsix=Join-Path $root 'release/signed.msix';$ChannelRecoveryMsix='';$ProfileSampleManifest='';$CriticalFileManifest=''
  $BuildId='8b19a2c012444a959b2370160324402a'
  $reservation=@{kind='semantier-windows-build-version-v1';version='0.0.129';sourceCommit=$Commit}
  $reservation | ConvertTo-Json | Set-Content -LiteralPath $VersionState
  $request=Join-Path $Output 'channel-request.json'
  @{commit=$Commit;version='0.0.129';buildId=$BuildId;sequence=129;publicBase='https://download.semantier.cn'} | ConvertTo-Json | Set-Content $request
  $record=@{commit=$Commit;version='0.0.129';source=$BuildSource;cache=$BuildCache;request=$request;
    prepared=(Join-Path $Output 'desktop-job/prepared.json');msix=(Join-Path $Output 'unsigned.msix');stage='reserved';publicBase='https://download.semantier.cn'}
  Write-RunEvent @{type='build';build=$record}
  $protected[$VersionState]=Hash $VersionState
  $protected[$request]=Hash $request
  $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'))
  [IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'retained dependencies')
  $checkpointBefore=[IO.File]::ReadAllText($script:statusLog)
  Initialize-LocalBuildSource $BuildSource $Commit
  $savedResidue=Join-Path $Output 'source-residue/source/node_modules/generated'
  Require (Test-Path -LiteralPath $savedResidue) 'Owned residue not preserved'
  Require ([IO.File]::ReadAllText($savedResidue) -ceq 'retained dependencies') 'Residue changed'
  Require (!(Test-Path -LiteralPath (Join-Path $BuildSource 'node_modules'))) 'Unverified dependencies reused'
  Require ((Get-SourceGit $BuildSource @('rev-parse','HEAD')) -ceq $Commit) 'Residue reconstruction changed source'
  $null=Assert-RegisteredSource $BuildSource $Commit
  Require ([IO.File]::ReadAllText($script:statusLog).StartsWith($checkpointBefore)) 'Historical checkpoints changed'
  $eventsBefore=@(Get-Content $script:statusLog | Where-Object {$_ -match '"type":"source-residue-recovery"'}).Count
  Initialize-LocalBuildSource $BuildSource $Commit
  Require (@(Get-Content $script:statusLog | Where-Object {$_ -match '"type":"source-residue-recovery"'}).Count -eq $eventsBefore) 'Repeated recovery appended duplicate transition'
  Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Recovered cleanup failed'
  $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  function New-ResidueFixture([string]$Name) {
    if($script:sourceLease){$script:sourceLease.Dispose()}
    $script:sourceLease=$null;$script:sourceOwner=$null
    $script:Output=Join-Path $root $Name;$script:BuildSource=Join-Path $Output 'source'
    Initialize-LocalBuildSource $BuildSource $Commit
    Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Fixture retirement failed'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    $script:VersionState=Join-Path $Output 'version-ledger.json'
    $reservation | ConvertTo-Json | Set-Content -LiteralPath $VersionState
    $request=Join-Path $Output 'channel-request.json'
    @{commit=$Commit;version='0.0.129';buildId=$BuildId;sequence=129;publicBase='https://download.semantier.cn'} | ConvertTo-Json | Set-Content $request
    $copy=$record.Clone();$copy.source=$BuildSource;$copy.request=$request
    $copy.prepared=Join-Path $Output 'desktop-job/prepared.json';$copy.msix=Join-Path $Output 'unsigned.msix'
    Write-RunEvent @{type='progress';message='fixture-original'}
    Write-RunEvent @{type='build';build=$copy}
    $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'))
    [IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'retained dependencies')
  }
  foreach($case in @('missing-owner','foreign-owner','run','commit','version','build-id','unexpected','link','registration','cache-lock','stage-process','cache-process')){
    New-ResidueFixture ('reject-'+$case)
    $extraLease=$null
    switch($case){
      'missing-owner' {
        $lines=@(Get-Content $script:statusLog | Where-Object {$_ -notmatch '"type":"source-owner"'})
        [IO.File]::WriteAllLines($script:statusLog,$lines)
      }
      'foreign-owner' {$owner=Read-SourceOwner;$owner.repository='foreign';Write-RunEvent @{type='source-owner';owner=$owner}}
      'run' {$owner=Read-SourceOwner;$owner.run='foreign';Write-RunEvent @{type='source-owner';owner=$owner}}
      'commit' {$owner=Read-SourceOwner;$owner.commit='f'*40;Write-RunEvent @{type='source-owner';owner=$owner}}
      'version' {$bad=$reservation.Clone();$bad.version='0.0.130';$bad | ConvertTo-Json | Set-Content $VersionState}
      'build-id' {$request=Join-Path $Output 'channel-request.json';$bad=Get-Content $request -Raw | ConvertFrom-Json;$bad.buildId='f'*32;$bad | ConvertTo-Json | Set-Content $request}
      'unexpected' {[IO.File]::WriteAllText((Join-Path $BuildSource 'app.txt'),'unregistered tracked source edit')}
      'link' {
        $link=Join-Path $BuildSource 'node_modules/escape'
        if($native){New-Item -ItemType Junction -Path $link -Target $BuildCache | Out-Null}
        else{New-Item -ItemType SymbolicLink -Path $link -Target $BuildCache | Out-Null}
      }
      'registration' {
        # Missing .git plus a surviving exact registration must also refuse.
        [IO.Directory]::Move($BuildSource,($BuildSource+'-fixture'))
        Run 'git.exe' @('-C',$Repository,'worktree','add','--detach',$BuildSource,$Commit)
        Remove-Item -LiteralPath $BuildSource -Recurse -Force
        [IO.Directory]::Move(($BuildSource+'-fixture'),$BuildSource)
      }
      'cache-lock' {$extraLease=[IO.File]::Open((Join-Path $BuildCache '.desktop-build.lock'),'OpenOrCreate','ReadWrite','None')}
      'stage-process' {
        $script:busyPath=$BuildSource
        function Get-CimInstance {param($ClassName) return @([pscustomobject]@{ProcessId=123456;ParentProcessId=0;CreationDate=[datetime]::UtcNow;ExecutablePath='';Name='fixture';CommandLine=$script:busyPath})}
      }
      'cache-process' {
        $script:busyPath=$BuildCache
        function Get-CimInstance {param($ClassName) return @([pscustomobject]@{ProcessId=123456;ParentProcessId=0;CreationDate=[datetime]::UtcNow;ExecutablePath='';Name='fixture';CommandLine=$script:busyPath})}
      }
    }
    $before=[IO.File]::ReadAllText($script:statusLog);$ledgerHash=Hash $VersionState
    try {
      Reject {Initialize-LocalBuildSource $BuildSource $Commit} ('Unsafe recovery accepted: '+$case)
      Require (Test-Path -LiteralPath (Join-Path $BuildSource 'node_modules/generated')) ('Refusal moved residue: '+$case)
      Require (!(Test-Path -LiteralPath (Join-Path $Output 'source-residue'))) ('Refusal preserved unverified residue: '+$case)
      Require ([IO.File]::ReadAllText($script:statusLog) -ceq $before) ('Refusal changed checkpoints: '+$case)
      Require ((Hash $VersionState) -ceq $ledgerHash) ('Refusal changed reservation: '+$case)
    } finally {
      if($extraLease){$extraLease.Dispose()}
      if($script:sourceLease){$script:sourceLease.Dispose();$script:sourceLease=$null}
      $script:sourceOwner=$null
      if($case -in @('stage-process','cache-process')){
        Remove-Item Function:Get-CimInstance
        if(!$native){function Get-CimInstance {param($ClassName) return @()}}
      }
      if($case -eq 'link'){Remove-Item -LiteralPath $link -Force}
      if($case -eq 'registration'){Remove-Item -LiteralPath $BuildSource -Recurse -Force;Run 'git.exe' @('-C',$Repository,'worktree','remove','--force',$BuildSource)}
    }
    Write-Output ('RECOVERY_REFUSED: '+$case)
  }
  if($native){
    New-ResidueFixture 'reject-foreign-acl'
    $path=Join-Path $BuildSource 'node_modules/generated'
    if($ForeignOwnerFixtureFile){
      # A same-volume rename retains the real foreign security descriptor.
      # Preparation alone is elevated; restore the disposable file in finally.
      Assert-NoReparseAncestor $ForeignOwnerFixtureFile
      Require (Test-Path -LiteralPath $ForeignOwnerFixtureFile -PathType Leaf) 'Foreign fixture file missing'
      $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
      $foreignSid=(Get-Acl -LiteralPath $ForeignOwnerFixtureFile).GetOwner([Security.Principal.SecurityIdentifier]).Value
      Require ($foreignSid -cne $sid) 'Prepared ACL fixture is not foreign owned'
      $fixtureHash=Hash $ForeignOwnerFixtureFile
      Remove-Item -LiteralPath $path
      [IO.File]::Move($ForeignOwnerFixtureFile,$path)
      Require ((Get-Acl -LiteralPath $path).GetOwner([Security.Principal.SecurityIdentifier]).Value -ceq $foreignSid) 'Rename lost foreign owner'
      $reason=''
      try {Initialize-LocalBuildSource $BuildSource $Commit}catch{$reason=$_.Exception.Message}
      finally {[IO.File]::Move($path,$ForeignOwnerFixtureFile)}
      Require ($reason -ceq 'Foreign residue ownership.') ('Native ACL guard not exercised: '+$reason)
      Require ((Hash $ForeignOwnerFixtureFile) -ceq $fixtureHash) 'Foreign fixture bytes changed'
      Require ((Get-Acl -LiteralPath $ForeignOwnerFixtureFile).GetOwner([Security.Principal.SecurityIdentifier]).Value -ceq $foreignSid) 'Foreign fixture owner changed'
    } else {
      $acl=Get-Acl -LiteralPath $path
      $foreign=Get-Acl -LiteralPath $path
      $foreign.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))
      # Failure to establish the real foreign owner remains a test failure.
      Set-Acl -LiteralPath $path -AclObject $foreign
      try {Reject {Initialize-LocalBuildSource $BuildSource $Commit} 'Foreign filesystem owner accepted'}
      finally {Set-Acl -LiteralPath $path -AclObject $acl}
    }
    Write-Output 'RECOVERY_REFUSED: foreign-filesystem-owner (native ACL)'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
  }
  if($native){
    New-ResidueFixture 'pinned-junction'
    $target=Join-Path $BuildSource 'apps/workspace'
    $null=[IO.Directory]::CreateDirectory($target)
    New-Item -ItemType Junction -Path (Join-Path $BuildSource 'node_modules/fixture-workspace') -Target $target | Out-Null
    [IO.Directory]::Delete($target);[IO.Directory]::Delete((Split-Path -Parent $target))
    $inventory=Get-ResidueInventory $BuildSource
    Initialize-LocalBuildSource $BuildSource $Commit
    $preserved=Join-Path $Output 'source-residue/source'
    Require ((Get-ResidueInventory $preserved) -ceq $inventory) 'Pinned junction inventory changed during recovery'
    Require ((Get-Item -LiteralPath (Join-Path $preserved 'node_modules/fixture-workspace') -Force).Target -ceq $target) 'Preserved junction target changed'
    Require (!(Test-Path -LiteralPath (Join-Path $BuildSource 'node_modules'))) 'Pinned junction dependencies reused'
    Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Pinned junction stage cleanup failed'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    foreach($case in @('wrong-target','unpinned')){
      New-ResidueFixture ('reject-workspace-'+$case)
      $name=if($case -eq 'wrong-target'){'fixture-workspace'}else{'unknown-workspace'}
      $target=if($case -eq 'wrong-target'){$BuildCache}else{Join-Path $BuildSource 'apps/workspace'}
      if($case -eq 'unpinned'){$null=[IO.Directory]::CreateDirectory($target)}
      New-Item -ItemType Junction -Path (Join-Path $BuildSource ('node_modules/'+$name)) -Target $target | Out-Null
      if($case -eq 'unpinned'){[IO.Directory]::Delete($target);[IO.Directory]::Delete((Split-Path -Parent $target))}
      Reject {Initialize-LocalBuildSource $BuildSource $Commit} ('Unsafe workspace junction accepted: '+$case)
      Require (!(Test-Path -LiteralPath (Join-Path $Output 'source-residue'))) 'Rejected workspace junction moved'
      $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    }
    Write-Output 'PINNED_WORKSPACE_JUNCTION: recovery/integrity PASS; wrong-target/unpinned REFUSED'
  }
  $writeEvent=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Write-RunEvent'},$true)
  foreach($phase in @('intent','preserved-before','preserved')){
    New-ResidueFixture ('interrupt-'+$phase)
    $checkpointBefore=[IO.File]::ReadAllText($script:statusLog)
    $script:stopPhase=$phase
    function Write-RunEvent($Event){
      if($Event.type -eq 'source-residue-recovery' -and $Event.phase -eq 'preserved' -and $script:stopPhase -eq 'preserved-before'){throw 'fixture interrupted before completion append'}
      $json=ConvertTo-Json -InputObject $Event -Depth 32 -Compress
      [IO.File]::AppendAllText($script:statusLog,"`n"+$json+"`n",$utf8)
      if($Event.type -eq 'source-residue-recovery' -and $Event.phase -eq $script:stopPhase){throw 'fixture interrupted'}
    }
    try {Reject {Initialize-LocalBuildSource $BuildSource $Commit} ('Interruption not reached: '+$phase)}
    finally {. ([scriptblock]::Create($writeEvent.Extent.Text))}
    Require ([IO.File]::ReadAllText($script:statusLog).StartsWith($checkpointBefore)) 'Interruption changed history'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    Initialize-LocalBuildSource $BuildSource $Commit
    $null=Assert-RegisteredSource $BuildSource $Commit
    $events=@(Get-Content $script:statusLog | Where-Object {$_ -match '"type":"source-residue-recovery"'})
    Require ($events.Count -eq 2) 'Interruption duplicated recovery events'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    Initialize-LocalBuildSource $BuildSource $Commit
    Require (@(Get-Content $script:statusLog | Where-Object {$_ -match '"type":"source-residue-recovery"'}).Count -eq 2) 'Restart duplicated recovery events'
    Require ((Remove-OwnedBuildSource) -ceq 'REMOVED') 'Interrupted recovery cleanup failed'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    Write-Output ('RECOVERY_INTERRUPTION_PASS: '+$phase)
  }
  foreach($change in @('history','protected-input','signed-input','residue')){
    New-ResidueFixture ('changed-'+$change)
    function Write-RunEvent($Event){
      [IO.File]::AppendAllText($script:statusLog,"`n"+($Event | ConvertTo-Json -Depth 32 -Compress)+"`n",$utf8)
      if($Event.type -eq 'source-residue-recovery' -and $Event.phase -eq 'intent'){throw 'fixture interruption'}
    }
    try {Reject {Initialize-LocalBuildSource $BuildSource $Commit} 'Intent interruption missing'}
    finally {. ([scriptblock]::Create($writeEvent.Extent.Text))}
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    switch($change){
      'history' {$text=[IO.File]::ReadAllText($script:statusLog);[IO.File]::WriteAllText($script:statusLog,$text.Replace('fixture-original','fixture-tampered'))}
      'protected-input' {[IO.File]::AppendAllText($VersionState," `n")}
      'signed-input' {[IO.File]::AppendAllText($RecoveryMsix,'changed')}
      'residue' {[IO.File]::AppendAllText((Join-Path $BuildSource 'node_modules/generated'),'changed')}
    }
    Reject {Initialize-LocalBuildSource $BuildSource $Commit} ('Changed recovery admitted: '+$change)
    Require (Test-Path -LiteralPath (Join-Path $BuildSource 'node_modules/generated')) 'Changed residue moved'
    $script:sourceLease.Dispose();$script:sourceLease=$null;$script:sourceOwner=$null
    if($change -eq 'signed-input'){[IO.File]::WriteAllText($RecoveryMsix,'preserve release/signed.msix')}
    Write-Output ('RECOVERY_TAMPER_REFUSED: '+$change)
  }
  $Output=Join-Path $root 'run-one';$BuildSource=Join-Path $Output 'source'
  $VersionState=Join-Path $root 'version-ledger.json'
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
    $child=Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList @('-NoProfile','-File',$busyScript) -WindowStyle Hidden -PassThru
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
  $NativeUpdateFrom='';$NativePublishOnly=$false;$SkipLocalInstall=$false;$script:nativePreflight=$null
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
