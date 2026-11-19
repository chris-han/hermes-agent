param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($fn.Extent.Text))}
. (Join-Path (Split-Path -Parent $Operator) 'deploy-ecs-continuation.ps1')
if(!$IsWindows){function git.exe { & /usr/bin/git @args };function Get-CimInstance {param($ClassName)return @()}}
$script:sourceLease=$null
$utf8=[Text.UTF8Encoding]::new($false)
function Reject([scriptblock]$Action,[string]$Expected){$caught='';try{& $Action}catch{$caught=$_.Exception.Message};if(!$caught -or $caught -notlike $Expected){throw ('Expected refusal '+$Expected+'; got '+$caught)}}
function Require([bool]$Ok,[string]$Message){if(!$Ok){throw $Message}}
function Reject([scriptblock]$Action,[string]$Expected){$reason='';try{& $Action}catch{$reason=$_.Exception.Message};Require ($reason -like $Expected) ('Wrong refusal: '+$reason+'; expected '+$Expected)}
if(!$IsWindows){function git.exe {& /usr/bin/git @args};function Get-CimInstance {param($ClassName)return @()}}
$root=Join-Path ([IO.Path]::GetTempPath()) ('continuation-'+[guid]::NewGuid().ToString('N'))
$Repository=Join-Path $root 'repo';$BuildCache=Join-Path $root 'cache'
$rsa=[Security.Cryptography.RSA]::Create(2048)
$cert=[Security.Cryptography.X509Certificates.CertificateRequest]::new('CN=Fixture',$rsa,[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1).CreateSelfSigned([DateTimeOffset]::UtcNow.AddDays(-1),[DateTimeOffset]::UtcNow.AddDays(1))
$Commit='4891d1b294a8c1d78aa98fc746bbd91d4de2ec6b';$BuildId='8b19a2c012444a959b2370160324402a'
$RecoveryMsix='';$ChannelRecoveryMsix='';$ProfileSampleManifest='';$CriticalFileManifest='';$WhatIfPreference=$false
$null=[IO.Directory]::CreateDirectory($Repository);$null=[IO.Directory]::CreateDirectory($BuildCache)
& git.exe init -q $Repository
if($LASTEXITCODE){throw 'Fixture native Git unavailable'}
function Report-Status([string]$Message){}
function Sign-Fixture([string]$Path,[byte[]]$Bytes){
 $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new($Bytes),$true)
 $cms.ComputeSignature([Security.Cryptography.Pkcs.CmsSigner]::new($cert));[IO.File]::WriteAllBytes($Path,$Bytes);[IO.File]::WriteAllBytes(($Path+'.p7s'),$cms.Encode())
}
function New-ContinuationFixture([string]$Name){
 $script:Output=Join-Path $root $Name;$script:BuildSource=Join-Path $Output 'source';$script:VersionState=Join-Path $Output 'version.json';$script:statusLog=Join-Path $Output 'checkpoints.jsonl'
 $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'));[IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'first')
 $script:sourceLease=$null;$script:sourceOwner=$null;$script:qualifiedContinuation=$null;$script:continuationConfirmed=$false
 $old=[ordered]@{python=('1'*64);node=('2'*64);driver=('3'*64);runner=('4'*64);operator='409eebcb740bd127cd997dc271d920834f70ebc060f1046d002b0f1c9fa50976';lock=('5'*64)}
 $script:inputs=($old|ConvertTo-Json)|ConvertFrom-Json;$script:inputs.operator='6'*64;$script:inputs=$inputs|ConvertTo-Json -Compress
 $script:record=[pscustomobject]@{inputs=($old|ConvertTo-Json -Compress);commit=$Commit;version='0.0.129';source=$BuildSource;cache=$BuildCache;request=(Join-Path $Output 'channel-request.json');prepared=(Join-Path $Output 'desktop-job/prepared.json');msix=(Join-Path $Output 'unsigned.msix');publicBase='https://download.semantier.cn';stage='reserved'}
 @{kind='semantier-windows-build-version-v1';version='0.0.129';sourceCommit=$Commit}|ConvertTo-Json|Set-Content $VersionState
 @{commit=$Commit;version='0.0.129';buildId=$BuildId;sequence=129;publicBase=$record.publicBase}|ConvertTo-Json|Set-Content $record.request
 Write-RunEvent @{type='build';build=$record}
 $script:owner=[pscustomobject]@{kind='native-git-worktree-v1';run=$Output;source=$BuildSource;repository=$Repository;cache=$BuildCache;commit=$Commit;pid=2147483646;started='2026-11-18T00:00:00.0000000Z'}
 Write-RunEvent @{type='source-owner';owner=$owner}
 $script:qualifiedTooling=[pscustomobject]@{contract='semantier-tooling-transition-v1';applicationCommit=$Commit;applicationRuntime='ea33018bb024f1e445cd592153bfe68a4f90ae2a';run=$Output;version='0.0.129';buildId=$BuildId;originalOperatorSha256=$old.operator;toolingCommit=('7'*40);operatorSha256=('6'*64);recoveryCommit='07172caece63d2a8fb6e3f6cefcb5b0527e485c7';recoveryOperatorSha256='f1fb1a9cf298ecfc7e789af169f40ecd6c8fd2b37974471d70ceae95a18cd506';originalInputs=$record.inputs;reservationSha256=(Hash $VersionState);requestSha256=(Hash $record.request);checkpointPrefixLength=(Get-Item $statusLog).Length;checkpointPrefixSha256=(Hash $statusLog)}
 $script:ToolingQualification=Join-Path $Output 'original.json';Sign-Fixture $ToolingQualification (Get-Bytes)
 $script:qualifiedTooling=Read-ToolingQualification $ToolingQualification $cert.Thumbprint
 Confirm-ToolingTransition $record $inputs
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Restore-OwnedSourceResidue $owner $BuildSource $Commit}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'));[IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'second')
 @{result='FAIL';commit=$Commit;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN';error='fixture preparation failure'}|ConvertTo-Json|Set-Content (Join-Path $Output 'result.json')
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Reject {Restore-OwnedSourceResidue $owner $BuildSource $Commit} 'Recovery inputs changed.'}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 $transition=@(Get-Content $statusLog|Where-Object {$_ -match '"type":"tooling-transition"'}|ConvertFrom-Json)[0]
 $script:c=[pscustomobject]@{contract='semantier-tooling-continuation-v1';sequence=1;run=$Output;applicationCommit=$Commit;applicationRuntime=$qualifiedTooling.applicationRuntime;version='0.0.129';buildId=$BuildId;originalInputs=$record.inputs;originalQualificationSha256=(Get-ToolingBytesHash $qualificationBytes);originalSignatureSha256=(Get-ToolingBytesHash $qualificationSignature);reservationSha256=(Hash $VersionState);requestSha256=(Hash $record.request);toolingCommit=('8'*40);operatorSha256=('9'*64);recoveryCommit=('8'*40);recoveryOperatorSha256=('9'*64);checkpointPrefixLength=(Get-Item $statusLog).Length;checkpointPrefixSha256=(Hash $statusLog);originalTransitionBindingSha256=(Get-ToolingBytesHash ([Text.Encoding]::UTF8.GetBytes($transition.binding)));originalResidueInventory=(Get-ResidueInventory (Join-Path $Output 'source-residue/source'));residueInventory=(Get-ResidueInventory $BuildSource);protectedBinding=(Assert-ResidueBindings $BuildSource $Commit)}
 $script:path=Join-Path $Output 'continued.json';Sign-Fixture $path (Get-Bytes $c)
 $script:qualifiedContinuation=Read-ToolingContinuation $path $cert.Thumbprint
 $script:continuedInputs=$inputs|ConvertFrom-Json;$script:continuedInputs.operator=$c.operatorSha256;$script:continuedInputs=$continuedInputs|ConvertTo-Json -Compress
}
function Get-Bytes($Value=$script:qualifiedTooling){return [Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 8 -Compress))}
try{
 New-ContinuationFixture 'positive'
 $before=[IO.File]::ReadAllText($statusLog);$originalHash=Hash $ToolingQualification
 Confirm-ToolingContinuation $record $continuedInputs
 $once=[IO.File]::ReadAllText($statusLog);Confirm-ToolingContinuation $record $continuedInputs
 Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Replay duplicated continuation'
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Restore-ToolingContinuationResidue $owner $BuildSource $Commit}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 Require ([IO.File]::ReadAllText((Join-Path $Output 'source-residue/source/node_modules/generated')) -ceq 'first') 'Original residue changed'
 Require ([IO.File]::ReadAllText((Join-Path $Output 'source-residue/continuation-1/source/node_modules/generated')) -ceq 'second') 'Second residue lost'
 Require ([IO.File]::ReadAllText($statusLog).StartsWith($before)) 'History rewritten'
 Require ((Hash $ToolingQualification) -ceq $originalHash) 'Original qualification changed'
 # Completed recovery may outlive a new result without changing its old snapshot.
 [IO.File]::AppendAllText((Join-Path $Output 'result.json')," `n")
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Restore-ToolingContinuationResidue $owner $BuildSource $Commit}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 # A subsequent preparation failure requires a new signed predecessor-linked snapshot.
 $null=[IO.Directory]::CreateDirectory((Join-Path $BuildSource 'node_modules'))
 [IO.File]::WriteAllText((Join-Path $BuildSource 'node_modules/generated'),'third')
 $previous=@(Get-Content $statusLog|Where-Object {$_ -match '"type":"tooling-continuation"'}|ConvertFrom-Json|Where-Object type -eq 'tooling-continuation')[-1]
 $c.sequence=2
 $c|Add-Member previousContinuationBindingSha256 (Get-ToolingBytesHash ([Text.Encoding]::UTF8.GetBytes($previous.binding)))
 $c.checkpointPrefixLength=(Get-Item $statusLog).Length;$c.checkpointPrefixSha256=Hash $statusLog
 $c.residueInventory=Get-ResidueInventory $BuildSource;$c.protectedBinding=Assert-ResidueBindings $BuildSource $Commit
 $chainHistory=[IO.File]::ReadAllText($statusLog)
 $chainQualification=Get-Bytes $c
 $priorFile=Join-Path $Output 'source-residue/continuation-1/source/node_modules/generated'
 foreach($badChain in @('predecessor','gap','prior-signature','prior-residue','incomplete-recovery','unknown-recovery')){
  $script:c=[Text.Encoding]::UTF8.GetString($chainQualification)|ConvertFrom-Json
  [IO.File]::WriteAllText($statusLog,$chainHistory)
  switch($badChain){
   'predecessor' {$c.previousContinuationBindingSha256='0'*64}
   'gap' {$c.sequence=3}
   'prior-residue' {[IO.File]::WriteAllText($priorFile,'tampered')}
   'prior-signature' {
    $rows=@(Get-Content $statusLog|Where-Object { $_.Trim() }|ConvertFrom-Json)
    $e=@($rows|Where-Object type -eq 'tooling-continuation')[0];$b=$e.binding|ConvertFrom-Json
    $sig=[Convert]::FromBase64String($b.signature);$sig[-1]=$sig[-1] -bxor 1;$b.signature=[Convert]::ToBase64String($sig)
    $e.binding=$b|ConvertTo-Json -Compress
    $c.previousContinuationBindingSha256=Get-ToolingBytesHash ([Text.Encoding]::UTF8.GetBytes($e.binding))
    $lines=$chainHistory -split "`n";for($j=0;$j -lt $lines.Count;$j++){if($lines[$j] -match '"type":"tooling-continuation"'){$lines[$j]=$e|ConvertTo-Json -Depth 10 -Compress}}
    [IO.File]::WriteAllText($statusLog,($lines -join "`n"))
   }
   'incomplete-recovery' {[IO.File]::WriteAllText($statusLog,($chainHistory -replace '(?m)^.*"phase":"preserved".*"type":"source-residue-continuation".*$',''))
    # Hashtable serialization order is unspecified: filter the event semantically.
    $lines=@($chainHistory -split "`n"|Where-Object {if(!$_.Trim()){$true}else{$e=$_|ConvertFrom-Json;!($e.type -eq 'source-residue-continuation' -and $e.phase -eq 'preserved')}})
    [IO.File]::WriteAllText($statusLog,($lines -join "`n"))
   }
   'unknown-recovery' {Write-RunEvent @{type='source-residue-continuation';binding=('0'*64);phase='intent'}}
  }
  $c.checkpointPrefixLength=(Get-Item $statusLog).Length;$c.checkpointPrefixSha256=Hash $statusLog
  Sign-Fixture $path (Get-Bytes $c);$script:qualifiedContinuation=Read-ToolingContinuation $path $cert.Thumbprint
  Reject {Confirm-ToolingContinuation $record $continuedInputs} '*'
  Require (!(Test-Path (Join-Path $Output 'source-residue/continuation-2'))) ('Bad chain moved residue: '+$badChain)
  [IO.File]::WriteAllText($priorFile,'second')
 }
 [IO.File]::WriteAllText($statusLog,$chainHistory)
 $script:c=[Text.Encoding]::UTF8.GetString($chainQualification)|ConvertFrom-Json
 Sign-Fixture $path (Get-Bytes $c);$script:qualifiedContinuation=Read-ToolingContinuation $path $cert.Thumbprint
 Confirm-ToolingContinuation $record $continuedInputs
 [IO.File]::WriteAllText($priorFile,'changed after admission')
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Reject {Restore-ToolingContinuationResidue $owner $BuildSource $Commit} 'Predecessor preserved residue changed before recovery.'}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 Require (Test-Path $BuildSource) 'Changed predecessor allowed current residue move'
 [IO.File]::WriteAllText($priorFile,'second')
 $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try{Restore-ToolingContinuationResidue $owner $BuildSource $Commit}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
 Require ([IO.File]::ReadAllText((Join-Path $Output 'source-residue/continuation-2/source/node_modules/generated')) -ceq 'third') 'Third residue lost'
 Require ([IO.File]::ReadAllText((Join-Path $Output 'source-residue/continuation-1/source/node_modules/generated')) -ceq 'second') 'Predecessor residue changed'
 $once=[IO.File]::ReadAllText($statusLog);Confirm-ToolingContinuation $record $continuedInputs
 Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Chain replay duplicated history'
 if($IsWindows){
  New-ContinuationFixture 'compiler-continuation'
  $shared=Join-Path $BuildCache 'react-compiler/desktop';$null=[IO.Directory]::CreateDirectory($shared)
  [IO.File]::WriteAllText((Join-Path $shared 'retained'),'shared compiler bytes')
  $link=Join-Path $BuildSource 'apps/desktop/node_modules/.cache/hermes-react-compiler'
  $null=[IO.Directory]::CreateDirectory((Split-Path $link -Parent));New-Item -ItemType Junction -Path $link -Target $shared|Out-Null
  $c.residueInventory=Get-ResidueInventory $BuildSource
  Sign-Fixture $path (Get-Bytes $c);$script:qualifiedContinuation=Read-ToolingContinuation $path $cert.Thumbprint
  Confirm-ToolingContinuation $record $continuedInputs
  $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
  try{Restore-ToolingContinuationResidue $owner $BuildSource $Commit}finally{$script:sourceLease.Dispose();$script:sourceLease=$null}
  Require ((Get-ResidueInventory (Join-Path $Output 'source-residue/continuation-1/source')) -ceq $c.residueInventory) 'Authenticated compiler preservation changed inventory'
  Require ([IO.File]::ReadAllText((Join-Path $shared 'retained')) -ceq 'shared compiler bytes') 'Authenticated preservation changed shared cache'
  Write-Output 'WINDOWS_NATIVE: authenticated compiler-junction continuation preservation PASS'
 }
 foreach($case in @('qualification','signature','prefix','inputs','result','old-residue','new-residue','missing-first','wrong-signer','signed-result','cache-lock')){
  New-ContinuationFixture $case
  switch($case){
   'qualification' {$script:qualifiedContinuation.originalInputs='changed'}
   'signature' {$data=[IO.File]::ReadAllBytes(($path+'.p7s'));$data[-1]=$data[-1] -bxor 1;[IO.File]::WriteAllBytes(($path+'.p7s'),$data)}
   'prefix' {$raw=[IO.File]::ReadAllBytes($statusLog);$raw[0]=32;[IO.File]::WriteAllBytes($statusLog,$raw)}
   'inputs' {$script:continuedInputs=$continuedInputs.Replace(('1'*64),('a'*64))}
   'signed-result' {$bad=Get-Content (Join-Path $Output 'result.json') -Raw|ConvertFrom-Json;$bad|Add-Member sha256 ('a'*64);$bad|ConvertTo-Json|Set-Content (Join-Path $Output 'result.json');$c.protectedBinding=Assert-ResidueBindings $BuildSource $Commit;Sign-Fixture $path (Get-Bytes $c);$script:qualifiedContinuation=Read-ToolingContinuation $path $cert.Thumbprint}
   'cache-lock' {$script:extraLease=[IO.File]::Open((Join-Path $BuildCache '.desktop-build.lock'),'OpenOrCreate','ReadWrite','None')}
   'result' {[IO.File]::AppendAllText((Join-Path $Output 'result.json'),' ')}
   'old-residue' {[IO.File]::AppendAllText((Join-Path $Output 'source-residue/source/node_modules/generated'),'changed')}
   'new-residue' {[IO.File]::AppendAllText((Join-Path $BuildSource 'node_modules/generated'),'changed')}
   'missing-first' {$text=[IO.File]::ReadAllText($statusLog);[IO.File]::WriteAllText($statusLog,($text -replace '(?m)^.*"type":"tooling-transition".*$',''))}
  }
  if($case -eq 'signature'){Reject {Read-ToolingContinuation $path $cert.Thumbprint} '*'}
  elseif($case -eq 'wrong-signer'){Reject {Read-ToolingContinuation $path ('0'*40)} '*'}

  else {Reject {Confirm-ToolingContinuation $record $continuedInputs} '*'}
  if($case -eq 'cache-lock'){$script:extraLease.Dispose()}
  Require (!(Test-Path (Join-Path $Output 'source-residue/continuation-1'))) ('Refusal moved residue: '+$case)
 }
  $script:savedWriteEvent=${function:Write-RunEvent}
  foreach($phase in @('intent','preserved-before','preserved')){
   New-ContinuationFixture ('interrupt-'+$phase)
   Confirm-ToolingContinuation $record $continuedInputs
   $script:stopPhase=$phase
   function Write-RunEvent($Event){
    if($Event.type -eq 'source-residue-continuation' -and $Event.phase -eq 'preserved' -and $script:stopPhase -eq 'preserved-before'){throw 'fixture interrupted'}
    & $script:savedWriteEvent $Event
    if($Event.type -eq 'source-residue-continuation' -and $Event.phase -eq $script:stopPhase){throw 'fixture interrupted'}
   }
   $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
   try {Reject {Restore-ToolingContinuationResidue $owner $BuildSource $Commit} 'fixture interrupted'}
   finally {Set-Item Function:Write-RunEvent $script:savedWriteEvent;$script:sourceLease.Dispose();$script:sourceLease=$null}
   $script:sourceLease=[IO.File]::Open((Join-Path $Output '.source.lock'),'OpenOrCreate','ReadWrite','None')
   try {
    Restore-ToolingContinuationResidue $owner $BuildSource $Commit
    $once=[IO.File]::ReadAllText($statusLog)
    Restore-ToolingContinuationResidue $owner $BuildSource $Commit
    Require ([IO.File]::ReadAllText($statusLog) -ceq $once) 'Interrupted replay duplicated history'
   } finally {$script:sourceLease.Dispose();$script:sourceLease=$null}
   Require (@(Get-Content $statusLog|Where-Object {$_ -match '"type":"source-residue-continuation"'}).Count -eq 2) 'Interruption lost or duplicated recovery phases'
  }
 Write-Output 'PASS: legacy repeated-recovery RED; signed continuation, exact-once admission, chained residues, predecessor authentication, replay and rejection matrix'
} finally {if($script:sourceLease){$script:sourceLease.Dispose()};Remove-Item $root -Recurse -Force;$cert.Dispose();$rsa.Dispose()}
