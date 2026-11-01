# Deterministic metadata refusal tests; no installation, network, signing or cleanup.
param([string]$GraphPath,[string]$PlatformScript)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-update.ps1')
if($GraphPath){$graph=Get-Content -LiteralPath $GraphPath -Raw -Encoding UTF8|ConvertFrom-Json}
else{
 $prefix='releases/channel-builds/'+('a'*32)+'/'
 $objects=@(
  @{key=$prefix+'app.msixbundle';kind='main-bundle';arch='neutral';references=@($prefix+'main.msix')},
  @{key=$prefix+'main.msix';kind='main-package';arch='x64';references=@()}
 )
 foreach($row in $objects){$row.identity='Semantier.Agent.Acceptance';$row.publisher='CN=Test';$row.version='0.0.59.0';$row.size=3000;$row.sha256='b'*64;$row.nativeDigest=[Convert]::ToBase64String([byte[]](1..32))}
 $graph=@{schema=1;transport='windows-package-uri';root=$prefix+'app.msixbundle';payloadInventorySha256='c'*64;objects=$objects}
}
$root=@($graph.objects|Where-Object{$_.key -ceq $graph.root})[0]
$request=@{base='https://s3.semantier.com';identity=$root.identity;publisher=$root.publisher;version=$root.version;graph=$graph}
$baseline=$request|ConvertTo-Json -Depth 16 -Compress
$valid=@(Read-NativeUpdateGraph ($baseline|ConvertFrom-Json))
if($valid.Count -ne @($graph.objects).Count){throw 'Complete graph not retained'}
$cases=@{
 'outer-cache-cap'={param($r) $r.graph.objects[0].size=480000001}
 'inner-cache-cap'={param($r) $r.graph.objects[-1].size=480000001}
 'missing-reference'={param($r) $r.graph.objects[0].references+=($r.graph.root+'.missing')}
 'wrong-publisher'={param($r) $r.graph.objects[-1].publisher='CN=Other'}
 'full-sha-is-not-native-digest'={param($r) $r.graph.objects[0].nativeDigest='f'*64}
 'mixed-build'={param($r) $parts=$r.graph.objects[-1].key -split '/';$parts[2]=if($parts[2] -ceq ('0'*32)){'1'*32}else{'0'*32};$r.graph.objects[-1].key=$parts -join '/'}
 'unreachable-object'={param($r) $r.graph.objects[0].references=@()}
 'query-authority'={param($r) $r.base+='?other=1'}
}
foreach($case in $cases.GetEnumerator()){
 $copy=$baseline|ConvertFrom-Json
 & $case.Value $copy
 $refused=$false
 try{$null=Read-NativeUpdateGraph $copy}catch{$refused=$true}
 if(!$refused){throw ('Native graph accepted '+$case.Key)}
}
@{scope='DETERMINISTIC_METADATA_ONLY';status='PASS';objects=$valid.Count;refusals=$cases.Count;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress

# Exercise real reconciliation logic; substitute only Windows package/process facts.
$fixture=Join-Path ([IO.Path]::GetTempPath()) ('native-reconcile-unit-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory((Join-Path $fixture 'app\resources'))
$global:nativeUnitResources=@()
$global:nativeUnitWaiter=$null
function Get-AppxPackage {param($Name,$PackageTypeFilter) if($PackageTypeFilter -ceq 'Resource'){return $global:nativeUnitResources};return $global:nativeUnitPackage}
function Get-Process {param($Id,$ErrorAction) return $global:nativeUnitWaiter}
try{
 $state=$baseline|ConvertFrom-Json
 $state|Add-Member -NotePropertyName authorizedSid -NotePropertyValue ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
 $state|Add-Member manifestSha256 ('d'*64)
 $state|Add-Member baselineResources @()
 $state|Add-Member -NotePropertyName fullName -NotePropertyValue 'unit-baseline'
 $state|Add-Member currentVersion '0.0.1.0'
 $state|Add-Member -NotePropertyName familyName -NotePropertyValue 'unit-family'
 $state|Add-Member -NotePropertyName targetBuildId -NotePropertyValue (($state.graph.root -split '/')[2])
 $state|Add-Member -NotePropertyName targetCommit -NotePropertyValue ('e'*40)
 $state|Add-Member -NotePropertyName waiterPid -NotePropertyValue 123
 $state|Add-Member -NotePropertyName waiterBirth -NotePropertyValue '1970-01-01T00:00:00.0010000+00:00'
 $state|Add-Member sourceSha256 (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
 $state|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $fixture 'intent.json') -Encoding UTF8
 $global:nativeUnitPackage=[pscustomobject]@{Name=$state.identity;PackageFullName='unit-baseline';PackageFamilyName='unit-family';Publisher=$state.publisher;Version=[Version]'0.0.1.0';Architecture='X64';Status='Ok';InstallLocation=$fixture}
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'NOT_STARTED'){throw 'Dead waiter without deployment intent not reconciled'}
 $global:nativeUnitWaiter=[pscustomobject]@{StartTime=[DateTimeOffset]::FromUnixTimeMilliseconds(1).UtcDateTime}
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Live waiter was released'}
 $global:nativeUnitWaiter=$null
 [IO.File]::WriteAllText((Join-Path $fixture 'deployment-intent.json'),'{}')
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Unresolved deployment was released'}
 $failure=@{status='FAILURE';operationCompleted=$true;manifestSha256=$state.manifestSha256;hresult='80080219'}
 $failure|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $fixture 'deployment-unresolved.json') -Encoding UTF8
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'BASELINE_INTACT_FAILED'){throw 'Completed failure with intact baseline not distinguished'}
 $failure.operationCompleted=$false
 $failure|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $fixture 'deployment-unresolved.json') -Encoding UTF8
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Pending failure was released'}
 $failure.operationCompleted=$true
 $failure|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $fixture 'deployment-unresolved.json') -Encoding UTF8
 $originalSource=$state.sourceSha256;$state.sourceSha256='0'*64
 $state|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $fixture 'intent.json') -Encoding UTF8
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Failure from unbound source was released'}
 $state.sourceSha256=$originalSource
 $state|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $fixture 'intent.json') -Encoding UTF8
 $global:nativeUnitResources=@([pscustomobject]@{PackageFamilyName=$state.familyName;PackageFullName='unexpected-resource'})
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Changed resource registration was released'}
 $global:nativeUnitResources=@()
 Remove-Item -LiteralPath (Join-Path $fixture 'deployment-unresolved.json')

 $global:nativeUnitPackage.Version=[Version]$state.version
 $global:nativeUnitPackage.PackageFullName='unit-target'
 $global:nativeUnitResources=@($state.graph.objects|Where-Object{$_.kind -ceq 'asset-package'}|ForEach-Object{[pscustomobject]@{ResourceId=$_.resourceId;Publisher=$state.publisher;Version=[Version]$state.version;PackageFamilyName=$state.familyName;Architecture='Neutral';Status='Ok'}})
 @{commit=$state.targetCommit;channelBuild=@{buildId=$state.targetBuildId}}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $fixture 'app\resources\install-stamp.json') -Encoding UTF8
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'TARGET_GRAPH_REGISTERED'){throw 'Complete exact target did not reconcile'}
 @{commit=('f'*40);channelBuild=@{buildId=$state.targetBuildId}}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $fixture 'app\resources\install-stamp.json') -Encoding UTF8
 if((Get-NativeUpdateReconciliation $fixture).status -cne 'UNKNOWN'){throw 'Wrong installed source was released'}
 @{scope='DETERMINISTIC_RECONCILIATION_OS_FACTS_SUBSTITUTED';status='PASS';cases=9;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
}finally{
 Remove-Item -LiteralPath $fixture -Recurse -Force
 Remove-Item Function:Get-AppxPackage,Function:Get-Process
 Remove-Variable nativeUnitPackage,nativeUnitResources,nativeUnitWaiter -Scope Global
}


# Compile the actual production native fact type without initialization/deployment.
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'native-package-update.ps1'),[ref]$tokens,[ref]$errors)
$type=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -ceq 'Add-Type' -and $node.Extent.Text.Contains('public static class SemantierNativeUpdateFacts')},$true))
if($type.Count -ne 1){throw 'Actual native fact declaration ambiguous'}
& ([scriptblock]::Create($type[0].Extent.Text))
$file=Join-Path ([IO.Path]::GetTempPath()) ('native-file-id-unit-'+[guid]::NewGuid().ToString('N'))
$created=[IO.File]::Open($file,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$created.Dispose()
$created=[IO.File]::OpenRead($file)
try{
 $first=Get-NativeUpdateFileId $created
 $reopened=[IO.File]::OpenRead($file)
 try{if((Get-NativeUpdateFileId $reopened) -cne $first){throw 'Native held file identity changed'}}finally{$reopened.Dispose()}
 @{scope='READ_ONLY_NATIVE_FACT_ABI';status='PASS';producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
}finally{$created.Dispose();[IO.File]::Delete($file)}

# Execute the production waiter's actual finally block against owned temp fixtures.
$waiterAst=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'update-relaunch-waiter.ps1'),[ref]$tokens,[ref]$errors)
$finally=@($waiterAst.FindAll({param($node) $node -is [Management.Automation.Language.TryStatementAst] -and $node.Finally -and $node.Finally.Extent.Text.Contains('Remove-Item -LiteralPath $ReadyFile')},$true))
if($finally.Count -ne 1){throw 'Waiter cleanup block ambiguous'}
$text=$finally[0].Finally.Extent.Text
$cleanup=$text.Substring(1,$text.Length-2)
foreach($mode in @('unknown','completed','legacy')){
 $stage=Join-Path ([IO.Path]::GetTempPath()) ('hermes-relaunch-unit-'+[guid]::NewGuid().ToString('N'))
 $journal=Join-Path ([IO.Path]::GetTempPath()) ('native-journal-unit-'+[guid]::NewGuid().ToString('N'))
 $null=[IO.Directory]::CreateDirectory($stage);$null=[IO.Directory]::CreateDirectory($journal)
 $ready=Join-Path $stage 'ready.txt';[IO.File]::WriteAllText($ready,'unit fixture')
 if($mode -ceq 'completed'){[IO.File]::WriteAllText((Join-Path $journal 'completed.json'),'{}')}
 $context=if($mode -ceq 'legacy'){$null}else{@{directory=$journal}}
 try{
  $probe=Join-Path $stage 'cleanup-probe.ps1'
  [IO.File]::WriteAllText($probe,('param($ReadyFile,$nativeContext)'+[Environment]::NewLine+$cleanup))
  & $probe $ready $context
  if((Test-Path -LiteralPath $stage) -ne ($mode -ceq 'unknown')){throw ('Waiter discarded unknown helper source or retained settled stage: '+$mode)}
  if(Test-Path -LiteralPath $ready){throw 'Transient handshake file retained'}
 }finally{
  if(Test-Path -LiteralPath $stage){Remove-Item -LiteralPath $stage -Recurse -Force}
  Remove-Item -LiteralPath $journal -Recurse -Force
 }
}
@{scope='ACTUAL_WAITER_CLEANUP_BLOCK_DISPOSABLE_FILES';status='PASS';cases=3;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress


# Actual protected-file reader and reconciliation writer; package/process facts substituted.
if(!$PlatformScript){$PlatformScript=Join-Path $PSScriptRoot '../../../tests/install/e2e-assets/semantier-registered-platform.ps1'}
$platformAst=[Management.Automation.Language.Parser]::ParseFile($PlatformScript,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Platform source parser failed'}
$nativeType=@($platformAst.FindAll({param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -ceq 'Add-Type' -and $node.Extent.Text.Contains('public static class SemantierNativeFacts')},$true))
if($nativeType.Count -ne 1){throw 'Platform native fact declaration ambiguous'}
& ([scriptblock]::Create($nativeType[0].Extent.Text))
foreach($name in @('Physical','Read-NativeProductionHandoff','Retire-NativeWaiterStage')){
 $definitions=@($platformAst.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name},$true))
 if($definitions.Count -ne 1){throw 'Actual platform function ambiguous'}
 . ([scriptblock]::Create($definitions[0].Extent.Text))
}
$root=Join-Path ([IO.Path]::GetTempPath()) ('native-reader-unit-'+[guid]::NewGuid().ToString('N'))
$manifest='d'*64;$attempt=Join-Path $root ('userdata/updates/native-'+$manifest+'-unit')
$null=[IO.Directory]::CreateDirectory($attempt)
$global:nativeReaderWaiter=$null
$global:nativeReaderState=@{scope='OS_FACTS_SUBSTITUTED';package=$null;process=$null}
function Observe {return $global:nativeReaderState}
function Get-Process {param($Id,$ErrorAction) return $global:nativeReaderWaiter}
try{
 $r=[pscustomobject]@{target=@{manifest_sha256=$manifest};native_intent_at='2026-10-29T00:00:00+00:00';native_directory=$attempt;native_status='BASELINE_INTACT_FAILED';native_handoff_source_sha256=('a'*64);native_digest_source_sha256=('b'*64);native_proof_hashes=$null}
 $intent=@{manifestSha256=$manifest;producedAt='2026-10-29T01:00:00+00:00';waiterPid=123;waiterBirth='1970-01-01T00:00:00.0010000+00:00';sourceSha256=('a'*64);digestSourceSha256=('b'*64)}
 $retained=Join-Path ([IO.Path]::GetTempPath()) ('hermes-relaunch-unit-'+[guid]::NewGuid().ToString('N'))
 $null=[IO.Directory]::CreateDirectory($retained)
 foreach($name in @('native-package-update.ps1','native-package-digest.ps1','update-relaunch-waiter.ps1')){Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $retained $name)}
 $intent.reconciliationDirectory=$retained
 $intent.sourceSha256=(Get-FileHash -LiteralPath (Join-Path $retained 'native-package-update.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
 $intent.digestSourceSha256=(Get-FileHash -LiteralPath (Join-Path $retained 'native-package-digest.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
 $r.native_handoff_source_sha256=$intent.sourceSha256;$r.native_digest_source_sha256=$intent.digestSourceSha256
 $intentFile=Join-Path $attempt 'intent.json';$intent|ConvertTo-Json|Set-Content -LiteralPath $intentFile -Encoding UTF8
 @{status='FAILURE';operationCompleted=$true;hresult='80080219';manifestSha256=$manifest}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $attempt 'deployment-unresolved.json') -Encoding UTF8
 [IO.File]::WriteAllText((Join-Path $attempt 'deployment-intent.json'),'{}')
 $read=Read-NativeProductionHandoff -Reconcile
 if($read.waiterAlive -or !$read.deploymentIntentPresent -or $read.fileSha256.Count -ne 3 -or $read.failure.hresult -cne '80080219'){throw 'Actual reconciliation reader lost failure/source facts'}
 $r.native_proof_hashes=($read.fileSha256|ConvertTo-Json|ConvertFrom-Json)
 $global:nativeReaderWaiter=[pscustomobject]@{StartTime=[DateTimeOffset]::FromUnixTimeMilliseconds(1).UtcDateTime}
 if(!(Read-NativeProductionHandoff -Reconcile).waiterAlive){throw 'Live waiter not exposed'}
 $global:nativeReaderWaiter=$null
 $r.native_intent_at='2026-10-30T00:00:00+00:00'
 $refused=$false;try{$null=Read-NativeProductionHandoff -Reconcile}catch{$refused=$true}
 if(!$refused){throw 'Earlier native attempt adopted by new transaction'}
 $r.native_intent_at='2026-10-29T00:00:00+00:00'
 $blocks=@($platformAst.FindAll({param($node) $node -is [Management.Automation.Language.StatementBlockAst] -and $node.Parent -is [Management.Automation.Language.SwitchStatementAst] -and $node.Extent.Text.Contains('Native reconciliation intent/live waiter mismatch')},$true))
 if($blocks.Count -ne 1){throw 'Actual native reconciliation action ambiguous'}
 $text=$blocks[0].Extent.Text;$write=[scriptblock]::Create($text.Substring(1,$text.Length-2))
 $original=[IO.File]::ReadAllBytes($intentFile)
 [IO.File]::AppendAllText($intentFile,' ')
 $refused=$false;try{& $write}catch{$refused=$true}
 if(!$refused -or (Test-Path -LiteralPath (Join-Path $attempt 'reconciled.json'))){throw 'Changed native proof was acknowledged'}
 [IO.File]::WriteAllBytes($intentFile,$original)
 $unexpected=Join-Path $retained 'user-file.txt';[IO.File]::WriteAllText($unexpected,'preserve this unit fixture')
 $refused=$false;try{& $write}catch{$refused=$true}
 if(!$refused -or [IO.File]::ReadAllText($unexpected) -cne 'preserve this unit fixture'){throw 'Unexpected retained stage contents were deleted'}
 Remove-Item -LiteralPath $unexpected
 & $write
 $receipt=Get-Content -LiteralPath (Join-Path $attempt 'reconciled.json') -Raw|ConvertFrom-Json
 if($receipt.status -cne 'BASELINE_INTACT_FAILED' -or $receipt.manifestSha256 -cne $manifest){throw 'Durable source-bound native reconciliation missing'}
 if(Test-Path -LiteralPath $retained){throw 'Settled exact native helper stage not retired'}
 & $write
 @{scope='ACTUAL_READER_WRITER_FILES_OS_FACTS_SUBSTITUTED';status='PASS';cases=7;producedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress
}finally{
 Remove-Item Function:Observe,Function:Get-Process
 Remove-Variable nativeReaderWaiter,nativeReaderState -Scope Global
 if($retained -and (Test-Path -LiteralPath $retained)){Remove-Item -LiteralPath $retained -Recurse -Force}
 Remove-Item -LiteralPath $root -Recurse -Force
}
