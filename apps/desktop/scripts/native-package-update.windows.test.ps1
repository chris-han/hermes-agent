# Deterministic metadata refusal tests; no installation, network, signing or cleanup.
param([string]$GraphPath)
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
