# Superseded failure recovery: real private NTFS files, mocked OS registration.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-update.ps1')
if(!(Get-Command Resolve-NativeSupersededFailure -ErrorAction SilentlyContinue)){throw 'Missing explicit superseded-failure recovery'}
$testRoot=Join-Path ([IO.Path]::GetTempPath()) ('native-superseded-'+[guid]::NewGuid().ToString('N'))
$directory=Join-Path $testRoot ('native-'+('a'*64)+'-'+('b'*32))
$installed=Join-Path $testRoot 'installed'
[IO.Directory]::CreateDirectory($directory)|Out-Null
[IO.Directory]::CreateDirectory($installed)|Out-Null
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$acl=Get-Acl $directory;$acl.SetAccessRuleProtection($true,$false)
$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($sid),'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
Set-Acl -LiteralPath $directory -AclObject $acl
$signature=Join-Path $installed 'AppxSignature.p7x';[IO.File]::WriteAllText($signature,'fixture signed registration')
$expectedSignature=(Get-FileHash $signature).Hash.ToLowerInvariant()
$fullName='Semantier.Agent.Acceptance_0.0.121.0_x64__fixture'
$script:registered=[pscustomobject]@{Name='Semantier.Agent.Acceptance';PackageFullName=$fullName;PackageFamilyName='Semantier.Agent.Acceptance_fixture';Publisher='CN=Test';Version='0.0.121.0';Architecture='X64';Status='Ok';IsDevelopmentMode=$false;InstallLocation=$installed}
$script:resources=@();$script:waiter=$null
function Get-AppxPackage {param($Name,$PackageTypeFilter);if($PackageTypeFilter){return $script:resources};return $script:registered}
function Get-Process {param($Id,$ErrorAction);return $script:waiter}
$prefix='releases/channel-builds/'+('c'*32)+'/'
$intent=@{schema=1;identity='Semantier.Agent.Acceptance';publisher='CN=Test';familyName='Semantier.Agent.Acceptance_fixture';currentVersion='0.0.79.0';version='0.0.80.0';authorizedSid=$sid;manifestSha256='a'*64;sourceSha256='d'*64;waiterPid=999999;waiterBirth='2026-11-04T01:00:00Z';journalDirectory=$directory;base='https://download.semantier.cn';graph=@{schema=1;transport='windows-package-uri';root=$prefix+'main.msix';payloadInventorySha256='e'*64;objects=@(@{key=$prefix+'main.msix';kind='main-package';identity='Semantier.Agent.Acceptance';publisher='CN=Test';version='0.0.80.0';arch='x64';resourceId='';size=3000;sha256='f'*64;nativeDigest=[Convert]::ToBase64String([byte[]](1..32));references=@()})}}
$failure=@{schema=1;manifestSha256='a'*64;status='FAILURE';operationCompleted=$true;hresult='80072EFD';activityId=[guid]::NewGuid().ToString();producedAt='2026-11-04T01:00:01Z'}
function Save-Fixture {
 [IO.File]::WriteAllText((Join-Path $directory 'intent.json'),($intent|ConvertTo-Json -Depth 12))
 [IO.File]::WriteAllText((Join-Path $directory 'deployment-intent.json'),(@{schema=1;manifestSha256='a'*64;status='DEPLOYMENT_INTENT';sid=$sid}|ConvertTo-Json))
 [IO.File]::WriteAllText((Join-Path $directory 'deployment-unresolved.json'),($failure|ConvertTo-Json))
 [IO.File]::WriteAllText((Join-Path $directory 'unresolved.json'),(@{schema=1;manifestSha256='a'*64;status='UNKNOWN'}|ConvertTo-Json))
}
function Read-Recovery([switch]$Apply){Resolve-NativeSupersededFailure -Directory $directory -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature -Apply:$Apply}
function Refuse([scriptblock]$Action){$script:refusalCase++;$failed=$false;try{& $Action|Out-Null}catch{$failed=$true};if(!$failed){throw ('Unsafe reconciliation accepted case '+$script:refusalCase+': '+$Action.ToString())}}
try {
 Save-Fixture
 $before=@{};Get-ChildItem $directory -File|ForEach-Object{$before[$_.Name]=(Get-FileHash $_.FullName).Hash}
 $proof=Read-Recovery
 if($proof.status -cne 'SUPERSEDED_TERMINAL_FAILURE' -or (Test-Path (Join-Path $directory 'reconciled.json'))){throw 'Read-only proof failed'}
 foreach($name in $before.Keys){if((Get-FileHash (Join-Path $directory $name)).Hash -cne $before[$name]){throw 'Read-only proof changed evidence'}}
 $failure.operationCompleted=$false;Save-Fixture;Refuse {Read-Recovery};$failure.operationCompleted=$true
 $failure.status='UNKNOWN';Save-Fixture;Refuse {Read-Recovery};$failure.status='FAILURE'
 $intent.authorizedSid='S-1-5-18';Save-Fixture;Refuse {Read-Recovery};$intent.authorizedSid=$sid;Save-Fixture
 $script:waiter=[pscustomobject]@{StartTime=[datetime]'2026-11-04T01:00:00Z'}; if(!((Get-Process -Id 999999).StartTime)){throw 'Waiter mock unavailable'};Refuse {Read-Recovery};$script:waiter=$null
 $script:resources=@([pscustomobject]@{PackageFamilyName='Semantier.Agent.Acceptance_fixture'});Refuse {Read-Recovery};$script:resources=@()
 foreach($field in @('PackageFamilyName','Publisher','Status','IsDevelopmentMode')){
  $old=$script:registered.$field;$script:registered.$field=if($field -eq 'IsDevelopmentMode'){$true}else{'changed'}
  Refuse {Read-Recovery};$script:registered.$field=$old
 }
 $script:registered.Version=[version]'0.0.80.0';Refuse {Read-Recovery};$script:registered.Version='0.0.121.0'
 $oldHash=$expectedSignature;$expectedSignature='0'*64;Refuse {Read-Recovery};$expectedSignature=$oldHash
 $before=@{};Get-ChildItem $directory -File|ForEach-Object{$before[$_.Name]=(Get-FileHash $_.FullName).Hash}
 $proof=Read-Recovery -Apply
 $stored=Get-Content (Join-Path $directory 'reconciled.json') -Raw|ConvertFrom-Json
 if($stored.status -cne 'SUPERSEDED_TERMINAL_FAILURE' -or $stored.registeredPackage -cne $fullName){throw 'Append receipt missing'}
 foreach($name in $before.Keys){if((Get-FileHash (Join-Path $directory $name)).Hash -cne $before[$name]){throw ('Original journal bytes changed: '+$name)}}
 $receiptPath=Join-Path $directory 'reconciled.json'
 $receiptHash=(Get-FileHash $receiptPath).Hash
 $null=Read-Recovery -Apply;$null=Read-Recovery
 if((Get-FileHash $receiptPath).Hash -cne $receiptHash){throw 'Idempotent replay changed receipt'}
 $script:waiter=[pscustomobject]@{StartTime=[datetime]'2026-11-04T01:00:00Z'};Refuse {Read-Recovery -Apply};$script:waiter=$null
 $ready=Get-NativeUpdateJournalReadiness -Directory $testRoot -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature
 if($ready.result -cne 'PASS'){throw ('Readiness refused valid historical proof: '+$ready.reason)}
 # A later verified package preserves the original historical 121 receipt.
 $script:registered.Version='0.0.122.0';$fullName='Semantier.Agent.Acceptance_0.0.122.0_x64__fixture';$script:registered.PackageFullName=$fullName
 $null=Read-Recovery -Apply
 if((Get-FileHash $receiptPath).Hash -cne $receiptHash){throw 'Later registration rewrote history'}
 $stored.hresult='80070001';[IO.File]::WriteAllText($receiptPath,($stored|ConvertTo-Json -Depth 12))
 Refuse {Read-Recovery -Apply}
 $ready=Get-NativeUpdateJournalReadiness -Directory $testRoot -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature
 if($ready.result -cne 'STOP'){throw 'Tampered receipt did not stop qualification'}
 $stored.hresult=$failure.hresult;[IO.File]::WriteAllText($receiptPath,($stored|ConvertTo-Json -Depth 12))
 $sibling=Join-Path $testRoot ('native-'+('0'*64)+'-'+('1'*32));[IO.Directory]::CreateDirectory($sibling)|Out-Null
 [IO.File]::WriteAllText((Join-Path $sibling 'intent.json'),($intent|ConvertTo-Json -Depth 12))
 $ready=Get-NativeUpdateJournalReadiness -Directory $testRoot -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature
 if($ready.result -cne 'STOP'){throw 'Ambiguous sibling did not stop qualification'}
 Refuse {Read-Recovery -Apply}
 $siblingIntent=$intent.Clone();$siblingIntent.journalDirectory=$sibling;$siblingIntent.manifestSha256='0'*64;$siblingIntent.version='0.0.122.0'
 [IO.File]::WriteAllText((Join-Path $sibling 'intent.json'),($siblingIntent|ConvertTo-Json -Depth 12))
 $completed=@{schema=1;manifestSha256='0'*64;status='TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED';sid=$sid;fullName=$fullName}
 [IO.File]::WriteAllText((Join-Path $sibling 'completed.json'),($completed|ConvertTo-Json))
 [IO.File]::WriteAllText((Join-Path $sibling 'deployment.json'),(@{schema=1;manifestSha256='0'*64;status='OS_SUCCESS'}|ConvertTo-Json))
 # The graph must match the journal target; a closed receipt is insufficient alone.
 $siblingIntent.graph.objects[0].version='0.0.122.0'
 [IO.File]::WriteAllText((Join-Path $sibling 'intent.json'),($siblingIntent|ConvertTo-Json -Depth 12))
 $ready=Get-NativeUpdateJournalReadiness -Directory $testRoot -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature
 if($ready.result -cne 'PASS'){throw ('Closed success across script revisions refused: '+$ready.reason)}
 $completed.sid='S-1-5-18';[IO.File]::WriteAllText((Join-Path $sibling 'completed.json'),($completed|ConvertTo-Json))
 $ready=Get-NativeUpdateJournalReadiness -Directory $testRoot -ExpectedPackageFullName $fullName -ExpectedSignatureSha256 $expectedSignature
 if($ready.result -cne 'STOP'){throw 'Wrong completed actor accepted'}
 foreach($name in $before.Keys){if((Get-FileHash (Join-Path $directory $name)).Hash -cne $before[$name]){throw 'Original evidence mutated'}}
 'SUPERSEDED_TERMINAL_FAILURE_IDEMPOTENCE_READINESS_AND_REFUSALS_PASS'

}finally{
 $resolved=[IO.Path]::GetFullPath($testRoot)
 if(!$resolved.StartsWith([IO.Path]::GetTempPath(),[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notmatch '^native-superseded-[a-f0-9]{32}$'){throw 'Fixture cleanup boundary refused'}
 Remove-Item -LiteralPath $resolved -Recurse -Force
}
