# Real NTFS owner/reparse/profile checks, without installation or credentials.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-update.ps1')
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$root=Join-Path ([IO.Path]::GetTempPath()) ('native-continuity-'+[guid]::NewGuid().ToString('N'))
New-NativeUpdateJournalDirectory $root $sid
function Refuse($action){$refused=$false;try{& $action}catch{$refused=$true};if(!$refused){throw 'Unsafe continuity binding accepted'}}
try{
 $backendRoot=Join-Path $root 'custom-home';$data=Join-Path $root 'electron'
 $null=[IO.Directory]::CreateDirectory($backendRoot);$null=[IO.Directory]::CreateDirectory($data)
 $null=[IO.Directory]::CreateDirectory((Join-Path $backendRoot 'profiles\work'))
 $r=[pscustomobject]@{backendHome=$backendRoot;activeProfile='work';credentialScope=(Join-Path $backendRoot 'profiles\work');userDataDirectory=$data;journalDirectory=(Join-Path $data 'updates\native-test')}
 $null=Assert-NativeUpdateContinuity $r $sid
 $r.activeProfile='default';$r.credentialScope=$backendRoot
 $null=Assert-NativeUpdateContinuity $r $sid
 $r.backendHome=Join-Path $root 'missing';Refuse {Assert-NativeUpdateContinuity $r $sid};$r.backendHome=$backendRoot
 $r.activeProfile='../foreign';Refuse {Assert-NativeUpdateContinuity $r $sid};$r.activeProfile='default'
 $r.credentialScope=$data;Refuse {Assert-NativeUpdateContinuity $r $sid};$r.credentialScope=$backendRoot
 Refuse {Assert-NativeUpdateContinuity $r 'S-1-5-18'}
 $r.journalDirectory=Join-Path $root 'foreign\native-test';Refuse {Assert-NativeUpdateContinuity $r $sid};$r.journalDirectory=Join-Path $data 'updates\native-test'
 $junction=Join-Path $root 'junction';$null=New-Item -ItemType Junction -Path $junction -Target $backendRoot
 $r.backendHome=$junction;$r.credentialScope=$junction;Refuse {Assert-NativeUpdateContinuity $r $sid}
 $r.backendHome=$backendRoot;$r.credentialScope=$backendRoot
 [IO.Directory]::Delete($junction)
 @{status='PASS';scope='NATIVE_CONTINUITY_PATH_PROFILE_OWNER';installation=$false}|ConvertTo-Json -Compress
}finally{
 if(![IO.Path]::GetFullPath($root).StartsWith([IO.Path]::GetTempPath(),[StringComparison]::OrdinalIgnoreCase)){throw 'Fixture cleanup escaped temp'}
 Remove-Item -LiteralPath $root -Recurse -Force
}

# Reproduce the shipped request (no backendHome/profile), environment and
# successor launch. Only OS package/process discovery is substituted; graph,
# journal, physical file identity, NTFS owner and home resolution are real.
$fixtureId=[guid]::NewGuid().ToString('N')
$suffix='-continuity-'+$fixtureId
$data=Join-Path ([Environment]::GetFolderPath('ApplicationData')) ('SemantierContinuityFixture'+$suffix)
$backendRoot=(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'hermes')+$suffix
$fixture=Join-Path ([IO.Path]::GetTempPath()) ('native-continuity-package-'+$fixtureId)
$savedSuffix=$env:HERMES_DATA_DIR_SUFFIX
$env:HERMES_DATA_DIR_SUFFIX=$suffix
foreach($directory in @($data,$backendRoot,$fixture)){New-NativeUpdateJournalDirectory $directory $sid}
try{
 $null=[IO.Directory]::CreateDirectory((Join-Path $data 'updates'))
 $null=[IO.Directory]::CreateDirectory((Join-Path $fixture 'app\resources'))
 $executable=Join-Path $fixture 'app\fixture.exe';[IO.File]::WriteAllText($executable,'fixture-only')
 $build=[pscustomobject]@{buildId=('a'*32);commit=('b'*40);windowsVersion='0.0.124.0';identity=[pscustomobject]@{msixAppIdWithOrg='Semantier.Fixture';appNamePascal='SemantierContinuityFixture'};bundleEnv=[pscustomobject]@{HERMES_HOME=$null;HERMES_DATA_DIR_SUFFIX=$suffix}}
 @{commit=$build.commit;channelBuild=@{buildId=$build.buildId}}|ConvertTo-Json|Set-Content (Join-Path $fixture 'app\resources\install-stamp.json') -Encoding UTF8
 $journal=Join-Path $data ('updates\native-'+('c'*64)+'-'+('d'*32))
 New-NativeUpdateJournalDirectory $journal $sid
 $key='releases/channel-builds/'+$build.buildId+'/main.msix'
 $graph=@{schema=1;transport='windows-package-uri';root=$key;payloadInventorySha256=('e'*64);objects=@(@{key=$key;kind='main-package';arch='x64';references=@();identity='Semantier.Fixture';publisher='CN=Fixture';version='0.0.124.0';size=3000;sha256=('f'*64);nativeDigest=[Convert]::ToBase64String([byte[]](1..32))})}
 $now=[DateTimeOffset]::UtcNow
 $request=[pscustomobject]@{schema=1;base='https://download.semantier.cn';graph=$graph;manifestSha256=('c'*64);graphSha256=('f'*64);publisher='CN=Fixture';identity='Semantier.Fixture';version='0.0.124.0';currentVersion='0.0.123.0';targetBuildId=$build.buildId;targetCommit=$build.commit;fullName='baseline';familyName='fixture-family';applicationId='Fixture';executable=$executable;userDataDirectory=$data;journalDirectory=$journal;authorizedSid=$sid;session=1;waiterPid=12345;waiterBirth=$now.AddSeconds(-2).ToString('o')}
 $global:continuityPackage=[pscustomobject]@{Name=$request.identity;PackageFamilyName=$request.familyName;PackageFullName='fixed-package';Publisher=$request.publisher;Version=[Version]$request.version;Architecture='X64';Status='Ok';InstallLocation=$fixture;SignatureKind='Developer';IsDevelopmentMode=$false}
 $global:continuityProcess=[pscustomobject]@{SessionId=1;ParentProcessId=12345;ExecutablePath=$executable;CreationDate=$now.UtcDateTime}
 $global:continuityWaiter=[pscustomobject]@{StartTime=$now.AddSeconds(-2).UtcDateTime}
 $global:continuitySid=$sid
 function Get-AppxPackage {param($Name,$PackageTypeFilter) if($PackageTypeFilter -ceq 'Resource'){return @()};return $global:continuityPackage}
 function Get-AppxPackageManifest {param($Package) return [pscustomobject]@{Package=[pscustomobject]@{Applications=[pscustomobject]@{Application=@([pscustomobject]@{Id='Fixture';Executable='app\fixture.exe'})}}}}
 function Get-CimInstance {param($ClassName,$Filter) return $global:continuityProcess}
 function Invoke-CimMethod {param($InputObject,$MethodName) return @{Sid=$global:continuitySid}}
 function Get-Process {param($Id,$ErrorAction) return $global:continuityWaiter}
 Initialize-NativeUpdateFacts
 $stream=[IO.File]::OpenRead($executable);try{$exeId=Get-NativeUpdateFileId $stream}finally{$stream.Dispose()}
 $launch=@{schema=1;producedAt=$now.AddMilliseconds(-100).ToString('o');manifestSha256=$request.manifestSha256;fullName='fixed-package';applicationId='Fixture';exeId=$exeId}
 function Save-Fixture {$request|ConvertTo-Json -Depth 16|Set-Content (Join-Path $journal 'intent.json') -Encoding UTF8;$launch|ConvertTo-Json|Set-Content (Join-Path $journal 'relaunch-intent.json') -Encoding UTF8}
 Save-Fixture
 if((Get-NativeLegacyRelaunchHome $data $build 555 $null) -ine $backendRoot){throw 'Legacy suffix home not recovered'}
 $launch.producedAt=$now.AddMinutes(-5).ToString('o');Save-Fixture;Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null}
 $launch.producedAt=$now.AddMilliseconds(-100).ToString('o');Save-Fixture
 $global:continuitySid='S-1-5-18';Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null};$global:continuitySid=$sid
 $global:continuityProcess.SessionId=2;Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null};$global:continuityProcess.SessionId=1
 $global:continuityPackage.PackageFamilyName='foreign';Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null};$global:continuityPackage.PackageFamilyName=$request.familyName
 $launch.exeId='foreign';Save-Fixture;Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null};$launch.exeId=$exeId;Save-Fixture
 $env:HERMES_DATA_DIR_SUFFIX='-foreign';Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null};$env:HERMES_DATA_DIR_SUFFIX=$suffix
 # If the original home is absent, never create a replacement.
 Rename-Item -LiteralPath $backendRoot -NewName ([IO.Path]::GetFileName($backendRoot)+'.held')
 Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null}
 Rename-Item -LiteralPath ($backendRoot+'.held') -NewName ([IO.Path]::GetFileName($backendRoot))
 # Target-side recovery also accepts a valid completed incarnation once the waiter exits.
 $global:continuityWaiter=$null
 $completed=@{schema=1;status='TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED';pid=555;birth=$now.ToString('o');sid=$sid;session=1;fullName='fixed-package';manifestSha256=$request.manifestSha256}
 $completed|ConvertTo-Json|Set-Content (Join-Path $journal 'completed.json') -Encoding UTF8
 if((Get-NativeLegacyRelaunchHome $data $build 555 $null) -ine $backendRoot){throw 'Completed legacy incarnation not recovered'}
 $completed.pid=556;$completed|ConvertTo-Json|Set-Content (Join-Path $journal 'completed.json') -Encoding UTF8
 Refuse {Get-NativeLegacyRelaunchHome $data $build 555 $null}
 $global:continuityWaiter=[pscustomobject]@{StartTime=$now.AddSeconds(-2).UtcDateTime}
 # A default Windows launch has no suffix environment variable. Use a unique
 # Electron fixture and read the existing default home without changing it.
 $defaultData=Join-Path ([Environment]::GetFolderPath('ApplicationData')) ('SemantierContinuityFixture'+$fixtureId)
 New-NativeUpdateJournalDirectory $defaultData $sid
 $savedData=$data;$savedJournal=$journal
 try{
  $data=$defaultData;$journal=Join-Path $data ('updates\native-'+('c'*64)+'-'+('d'*32))
  $null=[IO.Directory]::CreateDirectory((Join-Path $data 'updates'));New-NativeUpdateJournalDirectory $journal $sid
  $build.identity.appNamePascal='SemantierContinuityFixture'+$fixtureId;$build.bundleEnv.HERMES_DATA_DIR_SUFFIX='';$env:HERMES_DATA_DIR_SUFFIX=$null
  $request.userDataDirectory=$data;$request.journalDirectory=$journal;Save-Fixture
  if((Get-NativeLegacyRelaunchHome $data $build 555 $fixture) -ine $fixture){throw 'Legacy registry home lost for default Windows launch'}
  $defaultRoot=Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'hermes'
  if(Test-Path -LiteralPath $defaultRoot){
   if((Get-NativeLegacyRelaunchHome $data $build 555 $null) -ine $defaultRoot){throw 'Default Windows home lost'}
  }
 }finally{
  $data=$savedData;$journal=$savedJournal;$request.userDataDirectory=$data;$request.journalDirectory=$journal
  $build.identity.appNamePascal='SemantierContinuityFixture';$build.bundleEnv.HERMES_DATA_DIR_SUFFIX=$suffix;$env:HERMES_DATA_DIR_SUFFIX=$suffix
  if(!$defaultData.EndsWith($fixtureId)){throw 'Default fixture cleanup identity mismatch'}
  Remove-Item -LiteralPath $defaultData -Recurse -Force
 }
 # New handoff preserves a custom home and a nondefault credential scope, even
 # with an explicit isolated Electron directory.
 $null=[IO.Directory]::CreateDirectory((Join-Path $fixture 'profiles\work'))
 $request|Add-Member backendHome $fixture;$request|Add-Member activeProfile 'work';$request|Add-Member credentialScope (Join-Path $fixture 'profiles\work');Save-Fixture
 if((Get-NativeLegacyRelaunchHome $data $build 555 $null) -ine $fixture){throw 'Fixed handoff custom home/profile lost'}
 # Exercise the real Windows successor spawn with an owned native probe, never
 # an MSIX install. Package registration is the only remaining substituted fact.
 Remove-Item Function:Get-CimInstance,Function:Invoke-CimMethod,Function:Get-Process
 Remove-Item -LiteralPath $executable
 Add-Type -TypeDefinition @'
using System;using System.IO;using System.Threading;
public static class ContinuityProbe {
 public static void Main(string[] args) {
  File.WriteAllLines(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"continuity-observed.txt"),new string[]{Environment.GetEnvironmentVariable("HERMES_HOME"),Environment.GetEnvironmentVariable("HERMES_DESKTOP_USER_DATA_DIR"),String.Join(" ",args)});
  Thread.Sleep(15000);
 }
}
'@ -OutputAssembly $executable -OutputType ConsoleApplication
 $request.session=(Get-Process -Id $PID).SessionId
 $context=@{request=$request;rows=@(Read-NativeUpdateGraph $request);directory=$journal}
 $request.session=-1;Refuse {Start-NativeUpdateSuccessor $context $global:continuityPackage};$request.session=(Get-Process -Id $PID).SessionId
 $global:continuityPackage.Name='foreign';Refuse {Start-NativeUpdateSuccessor $context $global:continuityPackage};$global:continuityPackage.Name=$request.identity
 # Launch intent is append-only; this new transaction has no old launch receipt.
 Remove-Item -LiteralPath (Join-Path $journal 'relaunch-intent.json')
 $launched=Start-NativeUpdateSuccessor $context $global:continuityPackage
 try{
  $observed=Join-Path $fixture 'app\continuity-observed.txt'
  for($attempt=0;$attempt -lt 50 -and !(Test-Path -LiteralPath $observed);$attempt++){Start-Sleep -Milliseconds 100}
  $lines=[IO.File]::ReadAllLines($observed)
  if($lines[0] -ine $fixture -or $lines[1] -ine $data -or $lines[2] -cne '--profile work'){throw 'Native successor lost effective home/profile/userData'}
  if($launched.sid -cne $sid -or $launched.session -ne $request.session){throw 'Native successor actor binding lost'}
 }finally{
  $owned=Get-Process -Id $launched.pid -ErrorAction SilentlyContinue
  if($owned -and [Math]::Abs((([DateTimeOffset]$owned.StartTime).ToUniversalTime()-(ConvertTo-NativeWaiterBirth $launched.birth).ToUniversalTime()).Ticks) -lt 10){Stop-Process -Id $owned.Id -Force;$owned.WaitForExit()}
 }
 @{status='PASS';scope='LEGACY_AND_FIXED_SUCCESSOR_JOURNAL_PROCESS_PACKAGE';legacyFields='SHIPPED_SCHEMA1_NO_HOME_PROFILE';installation=$false;credentials='NOT_READ'}|ConvertTo-Json -Compress
 @{status='PASS';scope='ACTUAL_WINDOWS_SUCCESSOR_ENV_PROFILE_USERDATA_AND_PROCESS_OWNER';packageRegistration='FIXTURE';installation=$false}|ConvertTo-Json -Compress
}finally{
 $env:HERMES_DATA_DIR_SUFFIX=$savedSuffix
 foreach($directory in @($data,$backendRoot,$fixture)){
  if(![IO.Path]::GetFullPath($directory).Contains($fixtureId)){throw 'Fixture cleanup identity mismatch'}
  if(Test-Path -LiteralPath $directory){Remove-Item -LiteralPath $directory -Recurse -Force}
 }
}
