param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'),[string]$BuildOperator='')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
  . ([scriptblock]::Create($fn.Extent.Text.Replace('$PSCommandPath',("'"+$PSCommandPath.Replace("'","''")+"'"))))
}
# A signed synthetic qualification must be accepted by the maintained validator.
$rsa=[Security.Cryptography.RSA]::Create(2048)
$req=[Security.Cryptography.X509Certificates.CertificateRequest]::new('CN=Fixture',$rsa,[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)
$cert=$req.CreateSelfSigned([DateTimeOffset]::UtcNow.AddDays(-1),[DateTimeOffset]::UtcNow.AddDays(1))
$root=Join-Path ([IO.Path]::GetTempPath()) ('tooling-transition-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($root)
try {
  $path=Join-Path $root 'qualification.json'
  [IO.File]::WriteAllText($path,'{"contract":"semantier-tooling-transition-v1"}')
  $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new([IO.File]::ReadAllBytes($path)),$true)
  $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert))
  [IO.File]::WriteAllBytes(($path+'.p7s'),$cms.Encode())
  $q=Read-ToolingQualification $path $cert.Thumbprint
  if($q.contract -cne 'semantier-tooling-transition-v1'){throw 'Exact qualification rejected'}
  function Require([bool]$Ok,[string]$Message){if(!$Ok){throw $Message}}
  function Reject([scriptblock]$Action,[string]$Message){$caught=$false;try{& $Action}catch{$caught=$true};Require $caught $Message}
  Reject {Read-ToolingQualification $path ('0'*40)} 'Wrong signer accepted'
  $sig=[IO.File]::ReadAllBytes(($path+'.p7s'))
  [IO.File]::WriteAllText($path,'{"contract":"unknown"}')
  Reject {Read-ToolingQualification $path $cert.Thumbprint} 'Modified signed payload accepted'
  [IO.File]::WriteAllText($path,'{"contract":"semantier-tooling-transition-v1"}')
  Remove-Item ($path+'.p7s')
  Reject {Read-ToolingQualification $path $cert.Thumbprint} 'Unsigned qualification accepted'
  [IO.File]::WriteAllBytes(($path+'.p7s'),$sig)
  $q.contract='unknown'
  [IO.File]::WriteAllText($path,($q | ConvertTo-Json -Compress))
  $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new([IO.File]::ReadAllBytes($path)),$true)
  $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert))
  [IO.File]::WriteAllBytes(($path+'.p7s'),$cms.Encode())
  Reject {Read-ToolingQualification $path $cert.Thumbprint} 'Unknown signed contract accepted'

  $utf8=[Text.UTF8Encoding]::new($false)
  $ToolingQualification=$path
  $Commit='4891d1b294a8c1d78aa98fc746bbd91d4de2ec6b'
  $BuildId='8b19a2c012444a959b2370160324402a'
  $Repository=Join-Path $root 'repo';$BuildCache=Join-Path $root 'cache'
  $null=[IO.Directory]::CreateDirectory($BuildCache)
  # CIM/process ownership is native-only; portable tests keep the exact owner
  # identity checks and replace only the external process census.
  if(!$IsWindows){function Get-CimInstance {param($ClassName)return @()}}
  function New-Fixture([string]$Name){
    $script:Output=Join-Path $root $Name
    $script:BuildSource=Join-Path $Output 'source'
    $script:VersionState=Join-Path $Output 'version.json'
    $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
    $null=[IO.Directory]::CreateDirectory($Output)
    $script:sourceLease=$null;$script:sourceOwner=$null
    $old=[ordered]@{python=('1'*64);node=('2'*64);driver=('3'*64);runner=('4'*64);
      operator='409eebcb740bd127cd997dc271d920834f70ebc060f1046d002b0f1c9fa50976';lock=('5'*64)}
    $new=[ordered]@{};foreach($key in $old.Keys){$new[$key]=$old[$key]};$new.operator='6'*64
    $script:inputs=$new | ConvertTo-Json -Compress
    $script:record=[pscustomobject]@{inputs=($old | ConvertTo-Json -Compress);commit=$Commit;version='0.0.129';
      source=$BuildSource;cache=$BuildCache;request=(Join-Path $Output 'channel-request.json');
      prepared=(Join-Path $Output 'desktop-job/prepared.json');msix=(Join-Path $Output 'unsigned.msix');
      publicBase='https://download.semantier.cn';stage='reserved'}
    @{kind='semantier-windows-build-version-v1';version='0.0.129';sourceCommit=$Commit} | ConvertTo-Json | Set-Content $VersionState
    @{commit=$Commit;version='0.0.129';buildId=$BuildId;sequence=129;publicBase=$record.publicBase} | ConvertTo-Json | Set-Content $record.request
    Write-RunEvent @{type='build';build=$record}
    Write-RunEvent @{type='source-owner';owner=@{kind='native-git-worktree-v1';run=$Output;source=$BuildSource;
      repository=$Repository;cache=$BuildCache;commit=$Commit;pid=2147483646;started='2026-11-18T00:00:00.0000000Z'}}
    $script:qualifiedTooling=[pscustomobject]@{contract='semantier-tooling-transition-v1';applicationCommit=$Commit;
      applicationRuntime='ea33018bb024f1e445cd592153bfe68a4f90ae2a';run=$Output;version='0.0.129';buildId=$BuildId;
      originalOperatorSha256=$old.operator;toolingCommit=('7'*40);operatorSha256=$new.operator;
      recoveryCommit='07172caece63d2a8fb6e3f6cefcb5b0527e485c7';recoveryOperatorSha256='f1fb1a9cf298ecfc7e789af169f40ecd6c8fd2b37974471d70ceae95a18cd506';
      originalInputs=$record.inputs;reservationSha256=(Hash $VersionState);requestSha256=(Hash $record.request);
      checkpointPrefixLength=(Get-Item $statusLog).Length;checkpointPrefixSha256=(Hash $statusLog)}
    $script:qualificationBytes=[Text.Encoding]::UTF8.GetBytes(($script:qualifiedTooling | ConvertTo-Json -Compress))
    $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new($script:qualificationBytes),$true)
    $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert))
    $script:qualificationSignature=$cms.Encode()
    [IO.File]::WriteAllBytes($path,$script:qualificationBytes)
    [IO.File]::WriteAllBytes(($path+'.p7s'),$script:qualificationSignature)
    # Equivalent to entrypoint authentication; the fixture signer is deliberately
    # never accepted by the production entrypoint's fixed release-signer pin.
    $null=Read-ToolingQualification $path $cert.Thumbprint
  }
  New-Fixture 'positive'
  $prefix=[IO.File]::ReadAllText($statusLog);$ledgerHash=Hash $VersionState;$requestHash=Hash $record.request
  Confirm-ToolingTransition $record $inputs
  $once=[IO.File]::ReadAllText($statusLog)
  Require ($once.StartsWith($prefix)) 'History rewritten'
  Require ((Hash $VersionState) -ceq $ledgerHash -and (Hash $record.request) -ceq $requestHash) 'Reservation/request rewritten'
  Require ($record.inputs.Contains('409eebcb')) 'Original inputs replaced'
  Confirm-ToolingTransition $record $inputs
  Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Retry duplicated transition'
  # A fresh invocation authenticates the same exact qualification, then replays.
  $script:qualifiedTooling=Read-ToolingQualification $path $cert.Thumbprint
  Confirm-ToolingTransition (Read-CommitBuild $statusLog) $inputs
  Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Restart duplicated transition'
  $event=Get-Content $statusLog | Where-Object {$_ -match '"type":"tooling-transition"'} | ConvertFrom-Json
  $evidence=$event.binding | ConvertFrom-Json
  Require ($evidence.originalOperatorSha256 -ceq '409eebcb740bd127cd997dc271d920834f70ebc060f1046d002b0f1c9fa50976') 'Historical operator lost'
  Require ($evidence.qualification -ceq [Convert]::ToBase64String($qualificationBytes)) 'Replay qualification absent'
  foreach($case in @('hash','app','reservation','build-id','missing','conflict','stale-owner','writer','cache-writer','input','prefix','unknown','foreign-run')){
    New-Fixture $case
    $lease=$null
    switch($case){
      'hash' {$inputs=$inputs.Replace(('6'*64),('8'*64))}
      'app' {$script:qualifiedTooling.applicationCommit='f'*40}
      'reservation' {$bad=Get-Content $VersionState -Raw | ConvertFrom-Json;$bad.version='0.0.130';$bad | ConvertTo-Json | Set-Content $VersionState}
      'build-id' {$bad=Get-Content $record.request -Raw | ConvertFrom-Json;$bad.buildId='f'*32;$bad | ConvertTo-Json | Set-Content $record.request}
      'missing' {$script:qualifiedTooling=$null}
      'conflict' {
        Confirm-ToolingTransition $record $inputs;$script:qualifiedTooling.toolingCommit='9'*40
        $script:qualificationBytes=[Text.Encoding]::UTF8.GetBytes(($script:qualifiedTooling | ConvertTo-Json -Compress))
        $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new($script:qualificationBytes),$true)
        $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert));$script:qualificationSignature=$cms.Encode()
      }
      'stale-owner' {$owner=Read-SourceOwner;$owner.commit='f'*40;Write-RunEvent @{type='source-owner';owner=$owner}}
      'writer' {$lease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')}
      'cache-writer' {$lease=[IO.File]::Open((Join-Path $BuildCache '.desktop-build.lock'),'OpenOrCreate','ReadWrite','None')}
      'input' {$inputs=$inputs.Replace(('3'*64),('8'*64))}
      'prefix' {Confirm-ToolingTransition $record $inputs;$text=[IO.File]::ReadAllText($statusLog);[IO.File]::WriteAllText($statusLog,($text.Replace('2147483646','2147483645')))}
      'unknown' {$script:qualifiedTooling.contract='unknown'}
      'foreign-run' {$script:qualifiedTooling.run='foreign'}
    }
    $before=[IO.File]::ReadAllText($statusLog)
    try{Reject {Confirm-ToolingTransition $record $inputs} ('Accepted '+$case)}finally{if($lease){$lease.Dispose()}}
    Require ([IO.File]::ReadAllText($statusLog) -ceq $before) ('Rejection wrote history: '+$case)
  }
  New-Fixture 'interrupted'
  # Preserve a partial append exactly; it cannot authorize tooling execution.
  [IO.File]::AppendAllText($statusLog,'{"type":"tooling-transition","contract":')
  $prefix=[IO.File]::ReadAllText($statusLog)
  Confirm-ToolingTransition $record $inputs
  Require ([IO.File]::ReadAllText($statusLog).StartsWith($prefix)) 'Partial event rewritten'
  $once=[IO.File]::ReadAllText($statusLog)
  Confirm-ToolingTransition $record $inputs
  Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Partial-write replay duplicated'
  $savedWriter=${function:Write-RunEvent}
  foreach($point in @('before-append','after-append')){
    New-Fixture $point
    $before=[IO.File]::ReadAllText($statusLog)
    function Write-RunEvent($Event){
      if($point -ceq 'after-append'){& $savedWriter $Event}
      throw 'INTERRUPTED'
    }
    Reject {Confirm-ToolingTransition $record $inputs} 'Interruption injection did not fire'
    Set-Item Function:Write-RunEvent $savedWriter
    if($point -ceq 'before-append'){Require ([IO.File]::ReadAllText($statusLog) -ceq $before) 'Pre-append interruption wrote event'}
    Confirm-ToolingTransition $record $inputs
    Require (@(Get-Content $statusLog | Where-Object {$_ -match '"type":"tooling-transition"'}).Count -eq 1) 'Interrupted retry duplicated transition'
  }
  New-Fixture 'ordinary'
  $ToolingQualification=''
  $before=[IO.File]::ReadAllText($statusLog)
  Confirm-ToolingTransition $record $record.inputs
  Reject {Confirm-ToolingTransition $record $inputs} 'Ordinary operator mismatch accepted'
  Require ([IO.File]::ReadAllText($statusLog) -ceq $before) 'Ordinary path appended event'
  New-Fixture 'source-binding'
  $ToolingQualification=$path;$script:explicitOutput=$true
  $BuildMode='Prod';$LocalOnly=$true;$NewRun=$false;$Latest=$false;$DesktopOnly=$false
  $SetupOnly=$false;$LocalSetup=$false;$UpdateSetup=$false;$PreflightOnly=$false;$ResumeSigned=$false;$NativeUpdateFrom=''
  Reject {Assert-ApplicationToolingBinding} 'Synthetic signer admitted by production release authority'
  $ToolingQualification=''
  $script:SourceBinding=@{toolingHermesHead=$Commit;hermesGitlink=$Commit}
  Assert-ApplicationToolingBinding
  $script:SourceBinding.toolingHermesHead='f'*40
  Reject {Assert-ApplicationToolingBinding} 'Ordinary source mismatch admitted'

  # Source checks consume the real operator/parser and exact file hashes;
  # only Git's immutable-object responses are synthetic in this isolated fixture.
  $savedGit=${function:Get-SourceGit}
  $Repository=Join-Path $root 'tooling-source';$RuntimeRoot=Join-Path $root 'runtime'
  $sourceOperator=Join-Path $Repository 'scripts/deploy-ecs.ps1'
  $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $sourceOperator))
  Copy-Item $Operator $sourceOperator
  $script:blobs=@{}
  foreach($relative in @('scripts/deploy-ecs.ps1','scripts/bundles/desktop.py','apps/desktop/scripts/run-electron-builder.mjs','pm/lock.json','scripts/deploy-ecs-continuation.ps1')){
    $file=Join-Path $Repository $relative
    if($relative -cne 'scripts/deploy-ecs.ps1'){
      $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $file));[IO.File]::WriteAllText($file,'fixture '+$relative)
    }
    $script:blobs[$relative]=Hash $file
  }
  $script:baseline=[IO.File]::ReadAllText($sourceOperator)
  function Get-SourceGit([string]$Root,[string[]]$Arguments){
    switch($Arguments[0]){
      'rev-parse' {return $script:blobs[$Arguments[1].Substring(5)]}
      'hash-object' {return (Hash $Arguments[3])}
      'ls-tree' {return ('160000 commit '+$Commit+"`thermes-agent")}
      'show' {return $script:baseline}
      default {throw 'Unexpected source query'}
    }
  }
  $script:SourceBinding=@{toolingHermesHead=$script:qualifiedTooling.toolingCommit;hermesGitlink=$script:qualifiedTooling.toolingCommit}
  $script:qualifiedTooling.operatorSha256=Hash $sourceOperator
  Assert-QualifiedToolingSource $script:qualifiedTooling $sourceOperator
  $script:SourceBinding.hermesGitlink='f'*40
  Reject {Assert-QualifiedToolingSource $script:qualifiedTooling $sourceOperator} 'Wrong tooling gitlink accepted'
  $script:SourceBinding.hermesGitlink=$script:qualifiedTooling.toolingCommit
  $script:qualifiedTooling.operatorSha256='0'*64
  Reject {Assert-QualifiedToolingSource $script:qualifiedTooling $sourceOperator} 'Wrong executing tooling hash accepted'
  $script:qualifiedTooling.operatorSha256=Hash $sourceOperator
  $script:baseline=$script:baseline.Replace('function Restore-OwnedSourceResidue','function UnqualifiedRecovery')
  Reject {Assert-QualifiedToolingSource $script:qualifiedTooling $sourceOperator} 'Changed recovery implementation accepted'
  $script:baseline=[IO.File]::ReadAllText($sourceOperator)
  [IO.File]::AppendAllText((Join-Path $Repository 'pm/lock.json'),'changed')
  Reject {Assert-QualifiedToolingSource $script:qualifiedTooling $sourceOperator} 'Dirty tooling lock accepted'
  Set-Item Function:Get-SourceGit $savedGit
  New-Fixture 'failed-preparation-result'
  $ToolingQualification=$path
  $previous=[pscustomobject]@{result='FAIL';commit=$Commit;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN';
    error='original preparation failure';sourceCleanup='DEFERRED'}
  Require (Preserve-ToolingPreparationFailure $previous) 'Original failed preparation cannot resume release stages'
  $once=[IO.File]::ReadAllText($statusLog)
  Require (Preserve-ToolingPreparationFailure $previous) 'Failure preservation replay rejected'
  Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Historical failure duplicated'
  $previous.error='conflicting failure'
  Reject {Preserve-ToolingPreparationFailure $previous} 'Conflicting failure snapshot accepted'
  $previous.commit='f'*40
  Require (!(Preserve-ToolingPreparationFailure $previous)) 'Foreign failed result admitted'
  $previous.commit=$Commit;$previous.signing='PASS'
  Require (!(Preserve-ToolingPreparationFailure $previous)) 'Signed result admitted as preparation failure'
  $previous.signing='NOT_RUN';$ToolingQualification=''
  Require (!(Preserve-ToolingPreparationFailure $previous)) 'Ordinary result validation bypassed'
  # Exercise the real Build-CommitPackage input guard and transition wiring.
  # Stop exactly at the existing native source initialization boundary.
  New-Fixture 'build-entrypoint'
  $ToolingQualification=$path
  $BuildPython=Join-Path $root 'python.exe';$Node=Join-Path $root 'node.exe'
  foreach($file in @($BuildPython,$Node,(Join-Path $Repository 'scripts/bundles/desktop.py'),
    (Join-Path $Repository 'apps/desktop/scripts/run-electron-builder.mjs'),(Join-Path $Repository 'pm/lock.json'))){
    $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $file));[IO.File]::WriteAllText($file,'fixture '+$file)
  }
  $old=[ordered]@{python=(Hash $BuildPython);node=(Hash $Node);driver=(Hash (Join-Path $Repository 'scripts/bundles/desktop.py'));
    runner=(Hash (Join-Path $Repository 'apps/desktop/scripts/run-electron-builder.mjs'));
    operator='409eebcb740bd127cd997dc271d920834f70ebc060f1046d002b0f1c9fa50976';lock=(Hash (Join-Path $Repository 'pm/lock.json'))}
  $record.inputs=$old | ConvertTo-Json -Compress
  Write-RunEvent @{type='build';build=$record}
  $script:qualifiedTooling.originalInputs=$record.inputs
  $script:qualifiedTooling.operatorSha256=Hash $PSCommandPath
  $script:qualifiedTooling.checkpointPrefixLength=(Get-Item $statusLog).Length
  $script:qualifiedTooling.checkpointPrefixSha256=Hash $statusLog
  $script:qualificationBytes=[Text.Encoding]::UTF8.GetBytes(($script:qualifiedTooling | ConvertTo-Json -Compress))
  $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new($script:qualificationBytes),$true)
  $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert))
  $script:qualificationSignature=$cms.Encode()
  function git.exe {$global:LASTEXITCODE=0;return $Commit}
  function Assert-LocalWindowsPath([string]$Path){return $Path}
  function Initialize-LocalBuildSource {throw 'NATIVE_SOURCE_BOUNDARY_REACHED'}
  $PublicBase=$record.publicBase;$NewRun=$false;$WhatIfPreference=$false
  $script:explicitOutput=$true;$script:nativePreflight=$null
  if($BuildOperator){
    $base=[Management.Automation.Language.Parser]::ParseFile($BuildOperator,[ref]$null,[ref]$null)
    $fn=$base.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -ceq 'Build-CommitPackage'},$true)
    . ([scriptblock]::Create($fn.Extent.Text.Replace('$PSCommandPath',("'"+$PSCommandPath.Replace("'","''")+"'"))))
  }
  $failure=''
  try{Build-CommitPackage}catch{$failure=$_.Exception.Message}
  Require ($failure -ceq 'NATIVE_SOURCE_BOUNDARY_REACHED') ('Real build entrypoint rejected exact transition: '+$failure)
  Require (@(Get-Content $statusLog | Where-Object {$_ -match '"type":"tooling-transition"'}).Count -eq 1) 'Build did not append transition'
  Write-Output 'PASS: signed qualification, exact frozen v129 transition, rejection matrix, interruption, replay, unchanged same-operator behavior'
  if(!$IsWindows){Write-Output 'WINDOWS_NATIVE: NOT_RUN (CIM/ACL/native lock interop remain native gates)'}
} finally {Remove-Item $root -Recurse -Force;$cert.Dispose();$rsa.Dispose()}
