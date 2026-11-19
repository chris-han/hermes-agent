# Same-run v129 continuation of the existing deterministic deployment operator.
# Original qualification, transition and first residue remain immutable.
function Get-ToolingBytesHash([byte[]]$Bytes) {
  return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
function Read-ToolingContinuation([string]$Path,[string]$Signer) {
  Assert-NoReparseAncestor $Path;Assert-NoReparseAncestor ($Path+'.p7s')
  $bytes=[IO.File]::ReadAllBytes($Path);$signature=[IO.File]::ReadAllBytes(($Path+'.p7s'))
  $cms=[Security.Cryptography.Pkcs.SignedCms]::new([Security.Cryptography.Pkcs.ContentInfo]::new($bytes),$true)
  $cms.Decode($signature);$cms.CheckSignature($true)
  if($cms.SignerInfos.Count -ne 1 -or $cms.SignerInfos[0].Certificate.Thumbprint -cne $Signer){throw 'Continuation signer is not the release authority.'}
  $q=[Text.Encoding]::UTF8.GetString($bytes)|ConvertFrom-Json
  if($q.contract -cne 'semantier-tooling-continuation-v1'){throw 'Unknown tooling continuation contract.'}
  $script:continuationBytes=$bytes;$script:continuationSignature=$signature
  return $q
}
function Assert-ToolingContinuationIdentity($Q) {
  $original=$script:qualifiedTooling
  Assert-ToolingQualificationIdentity $original
  if($Q.contract -cne 'semantier-tooling-continuation-v1' -or $Q.sequence -ne 1 -or
    $Q.run -cne $Output -or $Q.applicationCommit -cne $Commit -or $Q.applicationRuntime -cne $original.applicationRuntime -or
    $Q.version -cne $original.version -or $Q.buildId -cne $BuildId -or $Q.originalInputs -cne $original.originalInputs -or
    $Q.originalQualificationSha256 -cne (Get-ToolingBytesHash $script:qualificationBytes) -or
    $Q.originalSignatureSha256 -cne (Get-ToolingBytesHash $script:qualificationSignature) -or
    $Q.reservationSha256 -cne $original.reservationSha256 -or $Q.requestSha256 -cne $original.requestSha256 -or
    $Q.toolingCommit -cnotmatch '^[a-f0-9]{40}$' -or $Q.operatorSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    $Q.recoveryCommit -cne $Q.toolingCommit -or $Q.recoveryOperatorSha256 -cne $Q.operatorSha256 -or
    $Q.checkpointPrefixLength -le 0 -or $Q.checkpointPrefixSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    $Q.originalResidueInventory -cnotmatch '^[A-F0-9]{64}$' -or $Q.residueInventory -cnotmatch '^[A-F0-9]{64}$'){
    throw 'Continuation is outside the authenticated frozen v129 scope.'
  }
}
function Confirm-ToolingContinuation($Record,[string]$Inputs) {
  $q=Get-AuthenticatedToolingContinuation
  Assert-ToolingContinuationIdentity $q
  $authenticated=[Text.Encoding]::UTF8.GetString($script:continuationBytes)|ConvertFrom-Json
  if(($authenticated|ConvertTo-Json -Depth 8 -Compress) -cne ($q|ConvertTo-Json -Depth 8 -Compress)){throw 'Authenticated continuation changed.'}
  $new=$Inputs|ConvertFrom-Json
  if($new.operator -cne $q.operatorSha256){throw 'Continuation executing operator mismatch.'}
  # Validate the immutable first transition using its exact admitted operator.
  # RequireExisting makes this a read-only replay; absence cannot create history.
  $projected=$Inputs|ConvertFrom-Json;$projected.operator=$script:qualifiedTooling.operatorSha256
  Confirm-ToolingTransition $Record ($projected|ConvertTo-Json -Compress) -RequireExisting
  $leases=@()
  try {
    foreach($p in @((Join-Path $Output '.source.lock'),(Join-Path $BuildCache '.desktop-build.lock'))){
      Assert-NoReparseAncestor $p;$lease=[IO.File]::Open($p,'OpenOrCreate','ReadWrite','None');$lease.Lock(0,1);$leases+=,$lease
    }
    $owner=Read-SourceOwner;Assert-SourceOwner $owner $BuildSource $Commit;Assert-StageInactive $owner
    $locked=Read-CommitBuild $script:statusLog
    if(($locked|ConvertTo-Json -Depth 32 -Compress) -cne ($Record|ConvertTo-Json -Depth 32 -Compress)){throw 'Build changed before continuation lock.'}
    $bytes=[IO.File]::ReadAllBytes($script:statusLog)
    if($q.checkpointPrefixLength -gt $bytes.Length){throw 'Continuation history truncated.'}
    $prefix=[byte[]]::new($q.checkpointPrefixLength);[Array]::Copy($bytes,$prefix,$prefix.Length)
    if((Get-ToolingBytesHash $prefix) -cne $q.checkpointPrefixSha256){throw 'Continuation checkpoint prefix changed.'}
    $events=@(foreach($line in [Text.Encoding]::UTF8.GetString($bytes) -split "`n"){
      if(!$line.Trim()){continue};try{$line|ConvertFrom-Json -ErrorAction Stop}catch{continue}
    })
    $first=@($events|Where-Object type -eq 'tooling-transition')
    if($first.Count -ne 1 -or (Get-ToolingBytesHash ([Text.Encoding]::UTF8.GetBytes($first[0].binding))) -cne $q.originalTransitionBindingSha256){throw 'Original transition binding differs.'}
    if((Hash $VersionState) -cne $q.reservationSha256 -or (Hash $Record.request) -cne $q.requestSha256){throw 'Continuation reservation/request changed.'}
    if((Get-ResidueInventory (Join-Path $Output 'source-residue/source')) -cne $q.originalResidueInventory){throw 'Original preserved residue changed.'}
    $oldRecovery=@(foreach($line in [IO.File]::ReadLines($script:statusLog)){
      try{$e=$line|ConvertFrom-Json -ErrorAction Stop}catch{continue}
      if($e -and $e.PSObject.Properties['type'] -and $e.type -ceq 'source-residue-recovery'){$e}
    })
    $originalPath=Join-Path $Output 'source-residue/source'
    if($oldRecovery.Count -ne 2 -or $oldRecovery[0].phase -cne 'intent' -or $oldRecovery[1].phase -cne 'preserved' -or
      $oldRecovery[0].binding -cne $oldRecovery[1].binding -or $oldRecovery[0].destination -cne $originalPath -or
      $oldRecovery[1].destination -cne $originalPath -or $oldRecovery[0].inventory -cne $q.originalResidueInventory -or
      $oldRecovery[1].inventory -cne $q.originalResidueInventory){throw 'Original recovery history is not complete and exact.'}
    $length=[long]$oldRecovery[0].logLength
    if($length -le 0 -or $length -gt $bytes.Length){throw 'Original recovery prefix is invalid.'}
    $oldPrefix=[byte[]]::new($length);[Array]::Copy($bytes,$oldPrefix,$oldPrefix.Length)
    if((Get-ToolingBytesHash $oldPrefix) -cne $oldRecovery[0].logSha256.ToLowerInvariant()){throw 'Original recovery prefix changed.'}

    $binding=[ordered]@{contract=$q.contract;sequence=1;qualification=[Convert]::ToBase64String($script:continuationBytes);signature=[Convert]::ToBase64String($script:continuationSignature)}|ConvertTo-Json -Compress
    $found=@($events|Where-Object type -eq 'tooling-continuation')
    if($found.Count -gt 1 -or ($found.Count -eq 1 -and ($found[0].binding -cne $binding -or $found[0].sequence -ne 1))){throw 'Conflicting tooling continuation.'}
    if(!$found.Count){
      if($Record.stage -cne 'reserved' -or (Assert-ResidueBindings $BuildSource $Commit) -cne $q.protectedBinding -or
        (Get-ResidueInventory $BuildSource) -cne $q.residueInventory){throw 'Continuation snapshot changed before admission.'}
      Assert-NoResidueRegistration $BuildSource
      $result=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw|ConvertFrom-Json
      foreach($name in @('signing','installation','publication')){if($result.$name -cne 'NOT_RUN'){throw 'Continuation requires a preparation-only failure.'}}
      foreach($name in @('sha256','signedMsix','unsignedSha256','plan')){if($result.PSObject.Properties[$name] -and $result.$name){throw 'Package-bearing result cannot enter preparation continuation.'}}
      if((Test-Path -LiteralPath $Record.msix) -or (Test-Path -LiteralPath (Join-Path $Output 'SemantierAgent.msix'))){throw 'Existing package cannot enter preparation continuation.'}
      if($result.result -cnotin @('FAIL','STOP')){throw 'Continuation requires a preparation-only failure.'}
      Write-RunEvent @{type='tooling-continuation';contract=$q.contract;sequence=1;at=[DateTime]::UtcNow.ToString('o');binding=$binding}
      $durable=[IO.File]::Open($script:statusLog,'Open','Write','Read');try{$durable.Flush($true)}finally{$durable.Dispose()}
    }
    $script:continuationConfirmed=$true
  } finally {foreach($l in $leases){$l.Dispose()}}
}
function Restore-ToolingContinuationResidue($Owner,[string]$Source,[string]$Revision) {
  $q=Get-AuthenticatedToolingContinuation
  if(!$script:continuationConfirmed -or !$script:sourceLease){throw 'Continuation recovery requires admitted evidence and source lease.'}
  Assert-SourceOwner $Owner $Source $Revision
  $leases=@()
  try {
    foreach($p in @((Join-Path $Output '.residue-recovery.lock'),(Join-Path $BuildCache '.desktop-build.lock'))){
      Assert-NoReparseAncestor $p;$l=[IO.File]::Open($p,'OpenOrCreate','ReadWrite','None');$l.Lock(0,1);$leases+=,$l
    }
    Assert-StageInactive $Owner
    $history=[IO.File]::ReadAllBytes($script:statusLog)
    if($q.checkpointPrefixLength -gt $history.Length){throw 'Continuation recovery history truncated.'}
    $admitted=[byte[]]::new($q.checkpointPrefixLength);[Array]::Copy($history,$admitted,$admitted.Length)
    if((Get-ToolingBytesHash $admitted) -cne $q.checkpointPrefixSha256){throw 'Continuation recovery history changed.'}
    if($IsWindows -and (Get-Acl -LiteralPath $Output).GetOwner([Security.Principal.SecurityIdentifier]).Value -cne
      [Security.Principal.WindowsIdentity]::GetCurrent().User.Value){throw 'Foreign continuation run ownership.'}
    if((Get-ResidueInventory (Join-Path $Output 'source-residue/source')) -cne $q.originalResidueInventory){throw 'Original preserved residue changed.'}
    $saved=Join-Path $Output 'source-residue/continuation-1/source';Assert-NoReparseAncestor $saved
    $binding=Get-ToolingBytesHash $script:continuationBytes
    $intent=$null;$complete=$false
    foreach($line in [IO.File]::ReadLines($script:statusLog)){
      try{$event=$line|ConvertFrom-Json -ErrorAction Stop}catch{continue}
      if(!$event -or !$event.PSObject.Properties['type'] -or $event.type -cne 'source-residue-continuation'){continue}
      if($event.binding -cne $binding -or $event.destination -cne $saved -or $event.inventory -cne $q.residueInventory){throw 'Conflicting continued residue history.'}
      if($event.phase -ceq 'intent'){if($intent){throw 'Duplicate continued residue intent.'};$intent=$event}
      elseif($event.phase -ceq 'preserved'){if(!$intent -or $complete){throw 'Conflicting continued residue completion.'};$complete=$true}
      else {throw 'Unknown continued residue phase.'}
    }
    if($intent){
      $bytes=[IO.File]::ReadAllBytes($script:statusLog)
      if($intent.logLength -gt $bytes.Length){throw 'Continued residue history truncated.'}
      $prefix=[byte[]]::new($intent.logLength);[Array]::Copy($bytes,$prefix,$prefix.Length)
      if((Get-ToolingBytesHash $prefix) -cne $intent.logSha256){throw 'Continued residue history changed.'}
    } else {
      if(Test-Path -LiteralPath (Split-Path -Parent $saved)){throw 'Unowned continuation preservation directory.'}
      if((Assert-ResidueBindings $Source $Revision) -cne $q.protectedBinding){throw 'Continuation protected inputs changed.'}
      Assert-NoResidueRegistration $Source
      if((Get-ResidueInventory $Source) -cne $q.residueInventory){throw 'Continuation residue changed.'}
      $bytes=[IO.File]::ReadAllBytes($script:statusLog)
      $intent=@{type='source-residue-continuation';phase='intent';at=[DateTime]::UtcNow.ToString('o');binding=$binding;destination=$saved;
        inventory=$q.residueInventory;logLength=$bytes.Length;logSha256=(Get-ToolingBytesHash $bytes)}
      Write-RunEvent $intent
    }
    if(Test-Path -LiteralPath $saved){
      if((Get-ResidueInventory $saved) -cne $q.residueInventory){throw 'Continued preserved residue changed.'}
      if(Test-Path -LiteralPath $Source){
        $null=Assert-RegisteredSource $Source $Revision
        if(Get-SourceGit $Source @('status','--porcelain','--untracked-files=all')){throw 'Continued reconstructed source is dirty.'}
      } elseif(Get-SourceRegistration $Source $Revision){throw 'Conflicting continued registration.'}
    } else {
      if($complete){throw 'Continued recovery state conflicts.'}
      if((Assert-ResidueBindings $Source $Revision) -cne $q.protectedBinding -or (Get-ResidueInventory $Source) -cne $q.residueInventory){throw 'Continuation inputs changed after intent.'}
      Assert-NoResidueRegistration $Source;Assert-StageInactive $Owner
      $null=[IO.Directory]::CreateDirectory((Split-Path -Parent $saved))
      [IO.Directory]::Move($Source,$saved)
    }
    if(!$complete){Write-RunEvent @{type='source-residue-continuation';phase='preserved';at=[DateTime]::UtcNow.ToString('o');binding=$binding;
      destination=$saved;inventory=$q.residueInventory;dependencies='NOT_REUSED_UNVERIFIED'}}
  } finally {foreach($l in $leases){$l.Dispose()}}
}
