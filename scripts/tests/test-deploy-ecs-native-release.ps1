$ErrorActionPreference='Stop'
$path=Join-Path $PSScriptRoot '../deploy-ecs.ps1'
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
$fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-NativeReleaseOptions'},$true)
if(!$fn){throw 'Missing deterministic native release option validator'}
. ([scriptblock]::Create($fn.Extent.Text))
$NativePublishOnly=$true;$SourceCommit='a'*40
$NativeUpdateFrom='0.0.121.0';$ExpectedVersion='0.0.122.0';$BuildMode='Prod';$SkipLocalInstall=$true
$Commit='a'*40;$Latest=$false;$BuildOnly=$false;$LocalOnly=$false;$DesktopOnly=$false
$SetupOnly=$false;$LocalSetup=$false;$UpdateSetup=$false;$PreflightOnly=$false
$ExpectedChannelSha256='b'*64;$RecoveryMsix='recovery';$NativeUserData='userdata';$ProfileSampleManifest='samples';$CriticalFileManifest='critical'
Assert-NativeReleaseOptions
foreach($name in @('Latest','BuildOnly','LocalOnly','DesktopOnly','SetupOnly','LocalSetup','UpdateSetup','PreflightOnly')){
 Set-Variable $name $true
 $failed=$false;try{Assert-NativeReleaseOptions}catch{$failed=$true}
 Set-Variable $name $false
 if(!$failed){throw "Unsafe option accepted: $name"}
}
foreach($name in @('ExpectedChannelSha256','RecoveryMsix','NativeUserData','ProfileSampleManifest','CriticalFileManifest')){
 $old=Get-Variable $name -ValueOnly;Set-Variable $name ''
 $failed=$false;try{Assert-NativeReleaseOptions}catch{$failed=$true}
 Set-Variable $name $old
 if(!$failed){throw "Missing binding accepted: $name"}
}
$BuildMode='Beta';$failed=$false;try{Assert-NativeReleaseOptions}catch{$failed=$true}
if(!$failed){throw 'Beta accepted'}
$BuildMode='Prod';$ExpectedVersion='0.0.121.0';$failed=$false;try{Assert-NativeReleaseOptions}catch{$failed=$true}
if(!$failed){throw 'Same-version candidate accepted'}
$ExpectedVersion='0.0.122.0';$NativePublishOnly=$true;$Commit=''
Assert-NativeReleaseOptions
$NativePublishOnly=$true;$Commit='a'*40
'NATIVE_RELEASE_OPTIONS_PASS'
$channel=Join-Path $PSScriptRoot '../deploy-ecs-channel1.ps1'
$ast=[Management.Automation.Language.Parser]::ParseFile($channel,[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
$fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-NativeGateway'},$true)
. ([scriptblock]::Create($fn.Extent.Text))
$nativeSampleHome=Join-Path ([IO.Path]::GetTempPath()) ('native-release-test-'+[Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($nativeSampleHome) | Out-Null
try {
 $package=[pscustomobject]@{InstallLocation='C:\fixture-package'}
 $state=@{pid=123;hermes_home=$nativeSampleHome;argv=@('C:\fixture-package\app\resources\agent-payload\hermes-agent\hermes_cli\main.py')}
 $state | ConvertTo-Json | Set-Content (Join-Path $nativeSampleHome 'gateway.pid')
 $script:fixtureProcess=[pscustomobject]@{ProcessId=123;ExecutablePath='C:\fixture-package\python.exe';SessionId=(Get-Process -Id $PID).SessionId;CommandLine='python -m hermes_cli.main --profile default gateway run --quiet';CreationDate=[DateTime]::UtcNow}
 function Get-CimInstance {param($ClassName,$Filter);return $script:fixtureProcess}
 function Invoke-CimMethod {param($InputObject,$MethodName);return @{Sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}}
 $null=Test-NativeGateway $package $nativeSampleHome
 foreach($field in @('ExecutablePath','SessionId','CommandLine')){
  $old=$script:fixtureProcess.$field
  $script:fixtureProcess.$field=if($field -eq 'SessionId'){-1}else{'unowned'}
  $failed=$false;try{$null=Test-NativeGateway $package $nativeSampleHome}catch{$failed=$true}
  $script:fixtureProcess.$field=$old
  if(!$failed){throw "Unsafe Gateway accepted: $field"}
 }
 'NATIVE_GATEWAY_OWNER_REFUSALS_PASS'
}finally{Remove-Item -LiteralPath (Join-Path $nativeSampleHome 'gateway.pid');Remove-Item -LiteralPath $nativeSampleHome}
$ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$errors)
$branch=$ast.Find({param($n)$n -is [Management.Automation.Language.IfStatementAst] -and
  $n.Clauses[0].Item1.Extent.Text -ceq '$NativeUpdateFrom' -and $n.Extent.Text.Contains('$receipt.nativePublicAfterPointer=')},$true)
if(!$branch -or !$branch.Extent.Text.Contains('NOT_RUN_OWNER_MANUAL')){throw 'Publish-only boundary missing'}
$Output=Join-Path ([IO.Path]::GetTempPath()) ('native-publish-test-'+[Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Output)|Out-Null
try {
 [IO.File]::WriteAllText((Join-Path $Output 'channel1-metadata.json'),'{"manifest":{}}')
 $ExpectedInputSha256='a'*64;$Msix='fixture-unsigned.msix'
 $receipt=[ordered]@{};$NativeUpdateFrom='0.0.121.0';$NativePublishOnly=$true;$signed='fixture.msix';$hash='a'*64
 $script:completedStages=@()
 function Hash {param($Path);return 'a'*64}
 function Save-Json {param($Value,$Path)}
 function Test-NativePublic {param($Manifest,$Package,[switch]$Active);return @{result='PASS'}}
 function Get-AppxPackage {param($Name);return @{Version=[Version]'0.0.121.0'}}
 function Complete-Checkpoint {param($Stage);$script:completedStages+=@($Stage)}
 function Invoke-NativeUpdateAcceptance {throw 'Publish-only invoked the native client updater'}
 & ([scriptblock]::Create($branch.Extent.Text))
 if($receipt.result -cne 'PUBLISHED_122_CLIENT_NOT_YET_VERIFIED' -or $receipt.nativeAcceptance -cne 'NOT_RUN_OWNER_MANUAL' -or
   'native-verified' -in $script:completedStages -or 'finished' -notin $script:completedStages){throw 'False native acceptance or missing publish-only completion'}
 'NATIVE_PUBLISH_ONLY_NO_INSTALL_PASS'
}finally{Remove-Item -LiteralPath (Join-Path $Output 'channel1-metadata.json');Remove-Item -LiteralPath $Output}
$ast=[Management.Automation.Language.Parser]::ParseFile($channel,[ref]$null,[ref]$errors)
$fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-NativeCompletedRelease'},$true)
. ([scriptblock]::Create($fn.Extent.Text))
$Output=Join-Path ([IO.Path]::GetTempPath()) ('native-replay-test-'+[Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Output)|Out-Null
try {
 $Commit='';$SourceCommit='a'*40;$ExpectedVersion='0.0.122.0';$NativePublishOnly=$true;$RemoteRoot='/fixture'
 $saved=@{result='PUBLISHED_122_CLIENT_NOT_YET_VERIFIED';plan=@{sourceCommit=$SourceCommit;version=$ExpectedVersion;buildId='b'*32};signedMsix=(Join-Path $Output 'SemantierAgent.msix');sha256='a'*64}
 [IO.File]::WriteAllText((Join-Path $Output 'result.json'),($saved|ConvertTo-Json))
 [IO.File]::WriteAllText((Join-Path $Output 'channel1-metadata.json'),'{"manifest":{}}')
 $script:fixtureHash='a'*64
 function Hash {param($Path);return $script:fixtureHash}
 function Require-Signature {param($Path)}
 function Get-Channel1ProofCommand {param($Root,$PackageSha256,$Build);return 'proof'}
 function Ssh {param($Command);return 'closure verified'}
 function Report-Status {param($Message)}
 function Save-Json {param($Value,$Path);[IO.File]::WriteAllText($Path,($Value|ConvertTo-Json -Depth 10))}
 function Sign-Copy {throw 'Replay attempted signing'}
 function Build-CommitPackage {throw 'Replay attempted version allocation/build'}
 if(!(Test-NativeCompletedRelease)){throw 'Completed publication did not revalidate'}
 $verified=Get-Content (Join-Path $Output 'result.json') -Raw|ConvertFrom-Json
 if($verified.publicationVerification -cne 'PASS' -or $verified.nativeAcceptance -cne 'NOT_RUN_OWNER_MANUAL'){throw 'Replay false native PASS'}
 $script:fixtureHash='c'*64;$failed=$false
 try{$null=Test-NativeCompletedRelease}catch{$failed=$true}
 if(!$failed){throw 'Changed signed artifact accepted by replay'}
 'NATIVE_PUBLISHED_REPLAY_NO_BUILD_SIGN_INSTALL_PASS'
}finally{Remove-Item -LiteralPath (Join-Path $Output 'channel1-metadata.json');Remove-Item -LiteralPath (Join-Path $Output 'result.json');Remove-Item -LiteralPath $Output}
$ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$errors)
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Transfer-Sftp','Upload')},$true)){
 . ([scriptblock]::Create($fn.Extent.Text))
}
$Output=Join-Path ([IO.Path]::GetTempPath()) ('native-transfer-test-'+[Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Output)|Out-Null
try {
 $utf8=[Text.UTF8Encoding]::new($false);$BuildMode='Prod';$NativeUpdateFrom='0.0.121.0';$script:sshOptions=@();$SshTarget='fixture'
 function Ssh {param($Command);if($Command.StartsWith('sha256sum ')){return ('a'*64)};return ''}
 function Hash {param($Path);return ('a'*64)}
 function Run {param($Exe,$Arguments);$i=[Array]::IndexOf($Arguments,'-b');$script:capturedBatch=[IO.File]::ReadAllText($Arguments[$i+1])}
 foreach($extension in @('msix','py','json')){
  $inputFile=Join-Path $Output ('fixture.'+$extension);[IO.File]::WriteAllText($inputFile,'fixture')
  Upload $inputFile ('/owned-stage/fixture.'+$extension)
  $verb=if($extension -eq 'msix'){'reput'}else{'put'}
  if($script:capturedBatch -notmatch ('(?m)^'+$verb+' ')){throw "Unsafe transfer mode for $extension"}
  Remove-Item -LiteralPath $inputFile
 }
 'NATIVE_METADATA_REPLACE_PACKAGE_RESUME_PASS'
}finally{Remove-Item -LiteralPath $Output}

# Actual preflight must refuse a local UNKNOWN before build/network operations.
& {
 $preflightAst=[Management.Automation.Language.Parser]::ParseFile($channel,[ref]$null,[ref]$errors)
 $fn=$preflightAst.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-NativeReleasePreflight'},$true)
 . ([scriptblock]::Create($fn.Extent.Text))
 $Repository=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
 $NativeUserData=Join-Path ([IO.Path]::GetTempPath()) ('native-preflight-'+[Guid]::NewGuid().ToString('N'))
 $updates=Join-Path $NativeUserData 'updates';$unknown=Join-Path $updates 'native-unknown'
 [IO.Directory]::CreateDirectory($unknown)|Out-Null
 try {
  [IO.File]::WriteAllText((Join-Path $unknown 'intent.json'),'{}')
  $RemoteRoot='/opt/fixture/releases';$SshTarget='fixture@fixture';$PublicBase='https://download.semantier.cn'
  $identity='fixture';$publisher='CN=Test';$NativeUpdateFrom='0.0.121.0';$Thumbprint='fixture';$RecoveryMsix='fixture'
  $script:sshOptions=@()
  function Get-EcsSshPassword {return 'fixture'}
  function Get-AppxPackage {return [pscustomobject]@{Version='0.0.121.0';Publisher='CN=Test';Status='Ok';Architecture='X64';PackageFullName='fixture';InstallLocation='fixture'}}
  function Get-Item {param($LiteralPath,$Path,$Force);if($LiteralPath -like 'Cert:*' -or $Path -like 'Cert:*'){return [pscustomobject]@{HasPrivateKey=$true;Subject='CN=Test';NotAfter=[DateTime]::UtcNow.AddDays(1)}};Microsoft.PowerShell.Management\Get-Item -LiteralPath $LiteralPath -Force:$Force}
  function Require-Signature {}
  function Get-PackageSignatureHash {return ('a'*64)}
  function Hash {return ('a'*64)}
  function Ssh {throw 'FORBIDDEN_NETWORK_REACHED'}
  $failed=$false
  try{Test-NativeReleasePreflight|Out-Null}catch{if($_.Exception.Message -notlike 'Local native journal readiness STOP:*'){throw};$failed=$true}
  if(!$failed){throw 'UNKNOWN local journal did not stop actual preflight'}
  'NATIVE_JOURNAL_PREFLIGHT_STOP_BEFORE_NETWORK_PASS'
 }finally{
  Remove-Item -LiteralPath (Join-Path $unknown 'intent.json')
  Remove-Item -LiteralPath $unknown
  Remove-Item -LiteralPath $updates
  Remove-Item -LiteralPath $NativeUserData
 }
}
