#Requires -Version 5.1
<#
.SYNOPSIS
  Sign, locally accept and publish the prepared 0.0.86 MSIX to ONE fixed ECS URL.
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -WhatIf
.EXAMPLE
  .\scripts\deploy-ecs.ps1
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -Output 'D:\repo\deliverables\work\ecs-deploy-20261105-221733-b7667f08'
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -NewRun
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -UpdateSetup
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -VerifyPublicDownload
.DESCRIPTION
  Run in an interactive Windows PowerShell session as the installing user.
  Native SSH/SFTP first read ALI_ECS_SSH_PASSWORDE from the repository .env,
  otherwise use your SSH key/agent or prompt for the ECS password. Secrets
  are never embedded in this script. No npm/uv install, Python publisher, WSL,
  release lock/database, split packages, bundles, ZIPs, or CH2/R2 operations.
  A signed Setup EXE downloads a single fixed MSIX over HTTP/3, with no
  current-release manifest or catalog. Existing Windows profile is preserved.
  A local result.json is evidence of this invocation, not a publication journal.
  Default reruns resume the latest matching incomplete checkpointed run. Use
  -Output to select a run explicitly, or -NewRun to start fresh. Successful
  stages and progress are recorded in the single append-only checkpoints.jsonl.
#>
[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='Low')]
param(
  [string]$Msix = 'D:\repo\hermes-build\apps\desktop\release\SemantierAgent-0.0.86-win-x64.msix',
  [string]$ExpectedInputSha256 = '416d715b00abc184a670e3c9e789a9baf18bd6996c47c20c67c05a7efb124ac7',
  [string]$SourceCommit = '5e9b6d5bd9025b9c64a28ae14243068e0125c3c0',
  [string]$Version = '0.0.86.0',
  [string]$BuildId = '7a76c0b5c57d4845a29ca56e429ed2c1',
  [string]$Thumbprint = '57BF15739B4AA447842EAF388856A01B4E5029FF',
  [string]$SshTarget = 'root@123.57.184.194',
  [string]$IdentityFile = '',
  [string]$RemoteRoot = '/opt/semantier-client-distribution/channel1-storage-root/releases',
  [string]$PublicBase = 'https://download.semantier.cn',
  [string]$UserProfile = "$env:LOCALAPPDATA\hermes-semantier-acceptance",
  [string]$Output = ('D:\repo\deliverables\work\ecs-deploy-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)),
  [string]$SetupExe = 'D:\repo\deliverables\cache\ecs-setup\semantier-setup-ch1.exe',
  [string]$SignTool = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe',
  [string]$Node = 'D:\repo\deliverables\cache\desktop-inputs\tools\node-26.7.0-win32-x64\node.exe',
  [string]$Repository = 'D:\repo\hermes-build',
  [string]$TimestampUrl = 'http://timestamp.digicert.com',
  [Alias('PrepareOnly')][switch]$LocalOnly,
  [switch]$SkipOldCleanup,
  [switch]$NewRun,
  [switch]$UpdateSetup,
  [switch]$VerifyPublicDownload
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$identity = 'Semantier.Agent.Acceptance'
$publisher = 'CN=Semantier Technology, O=Semantier Technology'
$utf8 = New-Object Text.UTF8Encoding($false)
$script:statusClock = [Diagnostics.Stopwatch]::StartNew()
$script:statusLog = $null
$script:askPassExe = $null
function Report-Status([string]$Message) {
  $line = '[{0}] [+{1:0}s] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $script:statusClock.Elapsed.TotalSeconds, $Message
  Write-Host $line
  if ($script:statusLog) { Write-RunEvent @{type='progress';at=[DateTime]::UtcNow.ToString('o');message=$Message} }
}
function Write-RunEvent($Event) {
  # Leading newline isolates a partial final record left by an interrupted write.
  $json=ConvertTo-Json -InputObject $Event -Depth 32 -Compress
  [IO.File]::AppendAllText($script:statusLog,"`n"+$json+"`n",$utf8)
}
function Read-RunCheckpoint([string]$Path) {
  $last=$null
  foreach($line in [IO.File]::ReadLines($Path)) {
    if(!$line.Trim()){continue}
    try {$entry=$line | ConvertFrom-Json -ErrorAction Stop}
    catch {Write-Warning 'Ignoring an incomplete or malformed run-log record; saved artifacts will still be verified.';continue}
    if($entry.PSObject.Properties['type'] -and $entry.type -eq 'checkpoint') {$last=$entry}
  }
  return $last
}
function Save-Json($Value, [string]$Path) {
  $temporary=$Path+'.tmp'
  [IO.File]::WriteAllText($temporary, (ConvertTo-Json -InputObject $Value -Depth 32), $utf8)
  if ([IO.File]::Exists($Path)) {[IO.File]::Replace($temporary,$Path,$Path+'.previous')} else {[IO.File]::Move($temporary,$Path)}
}
function Complete-Checkpoint([string]$Stage) {
  Save-Json $receipt (Join-Path $Output 'result.json')
  $script:checkpoints[$Stage]=[DateTime]::UtcNow.ToString('o')
  Write-RunEvent @{type='checkpoint';stage=$Stage;status='SUCCESS';at=$script:checkpoints[$Stage];
    binding=$script:runBinding;completed=$script:checkpoints;receipt=$receipt}
  Report-Status "Checkpoint saved: $Stage"
}
function Run([string]$Exe, [string[]]$Arguments) {
  $timer = [Diagnostics.Stopwatch]::StartNew()
  Report-Status ("Running {0} {1}" -f [IO.Path]::GetFileName($Exe), $Arguments[0])
  if([IO.Path]::GetFileName($Exe) -in @('ssh.exe','sftp.exe')) {
    Invoke-SshNative $Exe $Arguments | Out-Host
  } else { & $Exe @Arguments | Out-Host }
  if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
  Report-Status ("Completed {0} ({1:0}s)" -f [IO.Path]::GetFileName($Exe), $timer.Elapsed.TotalSeconds)
}
function Run-Setup([string]$Exe, [string[]]$Arguments) {
  Report-Status ("Setup: {0}; waiting for completion" -f $Arguments[0])
  # GUI-subsystem EXEs can otherwise return control to Windows PowerShell before
  # the command finishes. Pass a quoted command line and explicitly wait.
  $argsLine=($Arguments | ForEach-Object { '"' + $_.Replace('"','') + '"' }) -join ' '
  $p=Start-Process -FilePath $Exe -ArgumentList $argsLine -Wait -PassThru
  if ($p.ExitCode -ne 0) {throw "Setup command failed with exit code $($p.ExitCode): $Exe $($Arguments[0])"}
  Report-Status ("Setup completed: {0}" -f $Arguments[0])
}
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant() }
function Preserve-Databases([string]$Mode) {
  $python=Join-Path $Repository 'apps\desktop\build\agent-payload\venv\Scripts\python.exe'
  $code=@'
import os,sqlite3,sys,time,uuid
from contextlib import closing
from pathlib import Path
mode,home,backup=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
suffixes={'.db','.sqlite','.sqlite3'}
def connect(path,immutable=False):
    return sqlite3.connect(path.resolve().as_uri()+('?mode=ro&immutable=1' if immutable else '?mode=ro'),uri=True,timeout=5)
if mode=='backup':
    for source in home.rglob('*'):
        if not source.is_file() or source.suffix.lower() not in suffixes: continue
        target=backup/source.relative_to(home);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.snapshot-'+uuid.uuid4().hex)
        deadline=time.monotonic()+60
        def progress(status,remaining,total):
            if time.monotonic()>deadline: raise TimeoutError('SQLite snapshot exceeded 60 seconds: '+str(source))
        try:
            with closing(connect(source)) as src,closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst,pages=256,progress=progress,sleep=0.05)
                if dst.execute('pragma integrity_check').fetchone()[0]!='ok': raise RuntimeError('Invalid SQLite snapshot: '+str(source))
            os.replace(temporary,target)
        finally:
            if temporary.exists(): temporary.unlink()
        print('SQLite snapshot verified: '+str(source.relative_to(home)))
else:
    for original in backup.rglob('*'):
        if not original.is_file() or original.suffix.lower() not in suffixes: continue
        current=home/original.relative_to(backup)
        with closing(connect(original,True)) as before,closing(connect(current)) as after:
            if after.execute('pragma integrity_check').fetchone()[0]!='ok': raise RuntimeError('Current database integrity failure: '+str(current))
            for (table,) in before.execute("select name from sqlite_master where type='table' and name in ('sessions','messages')"):
                columns={row[1] for row in before.execute('pragma table_info('+table+')')}
                if 'id' in columns:
                    old={row[0] for row in before.execute('select id from '+table)}
                    new={row[0] for row in after.execute('select id from '+table)}
                    if not old<=new: raise RuntimeError('Existing session/message IDs lost: '+str(current))
        print('SQLite data preserved: '+str(original.relative_to(backup)))
'@
  # Windows PowerShell 5.1 strips embedded quotes in native -c arguments.
  # Standard input preserves the source exactly, including SQL string literals.
  Report-Status ('SQLite '+$Mode+': running snapshot/preservation checks.')
  $code | & $python -B - $Mode $UserProfile $backup | Out-Host
  if($LASTEXITCODE -ne 0){throw "SQLite $Mode failed with exit code $LASTEXITCODE"}
}
function Test-PublicMsix([switch]$FullDownload) {
  $python=Join-Path $Repository 'apps\desktop\build\agent-payload\venv\Scripts\python.exe'
  $code=@'
import hashlib,sys
from curl_cffi import requests,CurlHttpVersion
url,wanted,size,mode=sys.argv[1:];size=int(size)
with requests.Session() as client:
    options=dict(http_version=CurlHttpVersion.V3ONLY,allow_redirects=False,timeout=(15,60),headers={'Cache-Control':'no-cache','Accept-Encoding':'identity'})
    if mode=='head':
        response=client.head(url,**options)
        if response.status_code!=200 or int(response.headers.get('Content-Length','-1'))!=size:
            raise RuntimeError('Public H3 HEAD status/size mismatch')
        if response.headers.get('Accept-Ranges','').lower()!='bytes':
            raise RuntimeError('Public endpoint does not advertise byte-range support')
        print('PUBLIC_H3_HEAD_PASS: size='+str(size)+'; body download skipped')
    else:
        response=client.get(url,stream=True,**options)
        try:
            if response.status_code!=200: raise RuntimeError('Public H3 GET failed')
            digest=hashlib.sha256();total=0
            for chunk in response.iter_content():
                total+=len(chunk)
                if total>size: raise RuntimeError('Public package exceeds expected size')
                digest.update(chunk)
            if total!=size or digest.hexdigest()!=wanted: raise RuntimeError('Public package differs from local signed MSIX')
            print('PUBLIC_H3_FULL_SHA256_PASS')
        finally: response.close()
'@
  $mode=if($FullDownload){'full'}else{'head'}
  $code | & $python -B - "$PublicBase/releases/SemantierAgent.msix" $hash ([string](Get-Item -LiteralPath $signed).Length) $mode | Out-Host
  if($LASTEXITCODE -ne 0){throw "Public H3 $mode verification failed"}
}
function Require-Signature([string]$Path) {
  Run $SignTool @('verify','/pa','/v',$Path)
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -cne $Thumbprint) {
    throw "Signature or signer mismatch: $Path"
  }
}
function Sign-Copy([string]$InputFile, [string]$Destination) {
  Copy-Item -LiteralPath $InputFile -Destination $Destination
  $signature = Get-AuthenticodeSignature -LiteralPath $Destination
  if ($signature.Status -eq 'NotSigned') {
    Run $SignTool @('sign','/fd','SHA256','/sha1',$Thumbprint,'/tr',$TimestampUrl,'/td','SHA256',$Destination)
  }
  Require-Signature $Destination
}
function Read-ZipText($Archive, [string]$Name) {
  $entry = $Archive.GetEntry($Name)
  if (!$entry -or $entry.Length -gt 4194304) { throw "Missing/oversized package entry: $Name" }
  $reader = New-Object IO.StreamReader($entry.Open())
  try { $reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Invoke-SshNative([string]$Exe, [string[]]$Arguments) {
  $password=$null
  $envFile=Join-Path $Repository '.env'
  if(Test-Path -LiteralPath $envFile -PathType Leaf) {
    foreach($line in [IO.File]::ReadLines($envFile)) {
      if($line -match '^\s*(?:export\s+)?ALI_ECS_SSH_PASSWORDE\s*=\s*(.*)$') {
        $value=$Matches[1].Trim()
        if($value.StartsWith('"') -or $value.StartsWith("'")) {
          $quote=$value.Substring(0,1);$end=$value.LastIndexOf($quote)
          if($end -gt 0){$password=$value.Substring(1,$end-1)}
        } else {$password=($value -replace '\s+#.*$','').Trim()}
      }
    }
  }
  if(!$password) { & $Exe @Arguments; return }
  if(!$script:askPassExe) {
    $helperDirectory=Join-Path 'D:\repo\deliverables\work' ('ecs-ssh-askpass-'+[Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $helperDirectory | Out-Null
    $helper=Join-Path $helperDirectory 'askpass.exe'
    $source=Join-Path $helperDirectory 'askpass.cs'
    [IO.File]::WriteAllText($source,@'
using System;
using System.Text;
class AskPass {
  static int Main(string[] args) {
    if(args.Length!=1 || args[0].IndexOf("password",StringComparison.OrdinalIgnoreCase)<0)return 1;
    Console.OutputEncoding=new UTF8Encoding(false);
    Console.WriteLine(Environment.GetEnvironmentVariable("SEMANTIER_SSH_PASSWORD"));return 0;
  }
}
'@,$utf8)
    & "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:exe "/out:$helper" $source | Out-Host
    if($LASTEXITCODE -ne 0){throw 'Cannot compile native SSH password helper.'}
    $script:askPassExe=$helper
  }
  $names=@('SSH_ASKPASS','SSH_ASKPASS_REQUIRE','DISPLAY','SEMANTIER_SSH_PASSWORD')
  $savedEnvironment=@{}
  foreach($name in $names){$savedEnvironment[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try {
    [Environment]::SetEnvironmentVariable('SSH_ASKPASS',$script:askPassExe,'Process')
    [Environment]::SetEnvironmentVariable('SSH_ASKPASS_REQUIRE','force','Process')
    [Environment]::SetEnvironmentVariable('DISPLAY','semantier','Process')
    [Environment]::SetEnvironmentVariable('SEMANTIER_SSH_PASSWORD',$password,'Process')
    $Arguments=@('-o','PreferredAuthentications=password','-o','NumberOfPasswordPrompts=1')+$Arguments
    & $Exe @Arguments
  } finally {
    foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process')}
    $password=$null
  }
}
function Ssh([string]$Command) {
  # Base64 avoids Windows/POSIX quoting differences, including $ and newlines.
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command.Replace("`r",'')))
  $text = Invoke-SshNative 'ssh.exe' ($script:sshOptions + @($SshTarget,"printf '%s' '$encoded' | base64 -d | sh"))
  if ($LASTEXITCODE -ne 0) { throw "ECS SSH command failed ($LASTEXITCODE)" }
  return ($text -join "`n")
}
function Transfer-Sftp([string]$LocalFile, [string]$RemoteFile, [switch]$Download) {
  if($LocalFile -match '[\r\n"]' -or $RemoteFile -match '[\r\n"]'){throw 'Invalid SFTP filename.'}
  if(!$Download) {
    # Windows OpenSSH reput requires the remote file to exist. Create only a
    # missing staging file; never truncate an existing resumable upload.
    if($RemoteFile -match "'"){throw 'Invalid remote SFTP filename.'}
    $null=Ssh "set -eu; if test ! -e '$RemoteFile'; then umask 077; : > '$RemoteFile'; fi; test -f '$RemoteFile'; test ! -L '$RemoteFile'"
  }
  $local=$LocalFile.Replace('\','/').Replace('[','\[').Replace(']','\]').Replace('*','\*').Replace('?','\?')
  $command=if($Download){'get "'+$RemoteFile+'" "'+$local+'"'}else{'reput "'+$local+'" "'+$RemoteFile+'"'}
  $batch=Join-Path $Output ('sftp-'+[Guid]::NewGuid().ToString('N')+'.txt')
  [IO.File]::WriteAllText($batch,$command+"`nbye`n",$utf8)
  try {
    # -N keeps progress enabled in batch mode; BatchMode=no permits ASKPASS
    # or console password authentication. Batch commands fail on transfer errors.
    Run 'sftp.exe' ($script:sshOptions + @('-o','BatchMode=no','-N','-b',$batch,$SshTarget))
  } finally {
    Remove-Item -LiteralPath $batch -Force
  }
}
function Upload([string]$Path, [string]$RemoteFile) {
  if([IO.Path]::GetExtension($Path) -ieq '.exe') {
    Require-Signature $Path
    Report-Status ('Uploading signed Setup: '+[IO.Path]::GetFileName($Path))
    Transfer-Sftp $Path $RemoteFile
    Complete-Checkpoint ('upload-'+[IO.Path]::GetFileName($Path))
    return
  }
  $localHash=Hash $Path
  $remoteHash=Ssh "if test -f '$RemoteFile'; then sha256sum '$RemoteFile' | cut -d' ' -f1; fi"
  if($remoteHash -ceq $localHash) {Report-Status ('Reusing verified upload: '+[IO.Path]::GetFileName($Path));return}
  Report-Status ("Uploading {0} ({1:N1} MiB)" -f [IO.Path]::GetFileName($Path), ((Get-Item -LiteralPath $Path).Length / 1MB))
  Transfer-Sftp $Path $RemoteFile
  $remoteHash=Ssh "sha256sum '$RemoteFile' | cut -d' ' -f1"
  if($remoteHash -cne $localHash){throw 'Uploaded file hash mismatch.'}
  Complete-Checkpoint ('upload-'+[IO.Path]::GetFileName($Path))
}
if ($env:OS -ne 'Windows_NT') { throw 'Run on native Windows.' }
if ($SshTarget -cnotmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$' -or
    $RemoteRoot -cnotmatch '^/opt/[A-Za-z0-9_/-]+/releases$' -or $RemoteRoot.Contains('/../') -or
    $PublicBase -cne 'https://download.semantier.cn' -or $BuildId -cnotmatch '^[a-f0-9]{32}$' -or
    $SourceCommit -cnotmatch '^[a-f0-9]{40}$' -or $ExpectedInputSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    $Version -cnotmatch '^\d+\.\d+\.\d+\.\d+$' -or $Thumbprint -cnotmatch '^[A-F0-9]{40}$') {
  throw 'Invalid release/SSH/path configuration. This script publishes CH1 only.'
}
$script:sshOptions = @('-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
if ($IdentityFile) { $script:sshOptions += @('-i',(Resolve-Path -LiteralPath $IdentityFile).Path) }
$plan = [ordered]@{input=$Msix;version=$Version;sourceCommit=$SourceCommit;ssh=$SshTarget;
  remoteRoot=$RemoteRoot;fixedMsixUrl="$PublicBase/releases/SemantierAgent.msix";
  profileBackup=(Join-Path $Output 'profile-backup');output=$Output;
  stages=@('verify existing input','sign one MSIX','reuse/build fixed-file HTTP3 Setup',
    'back up profile','install and observe Electron/real PTY','SFTP reput upload with progress to unpublished directory',
    'verify remote hash','atomic replace one stable MSIX + Setup',
    $(if($VerifyPublicDownload){'explicit full public H3 download and hash'}else{'lightweight public H3 HEAD; no package body download'}),
    'verify live Landing before old-package cleanup')}
if ($WhatIfPreference) { $plan | ConvertTo-Json -Depth 5; return }
Report-Status 'Validating existing MSIX hash, source, identity and node-pty inputs; no dependency rebuild.'
if (!(Test-Path -LiteralPath $Msix -PathType Leaf) -or (Hash $Msix) -cne $ExpectedInputSha256) { throw 'Existing MSIX integrity check failed; no rebuild attempted.' }
$tools=@($SignTool,$Node)
foreach ($tool in $tools) {
  if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Required existing tool unavailable: $tool" }
}
if (!(Test-Path -LiteralPath (Join-Path $Repository 'node_modules\@playwright\test'))) { throw 'Existing Playwright dependencies unavailable; no npm installation attempted.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($Msix)
try {
  [xml]$manifest = Read-ZipText $archive 'AppxManifest.xml'
  $stamp = Read-ZipText $archive 'app/resources/install-stamp.json' | ConvertFrom-Json
  if ($manifest.Package.Identity.Name -cne $identity -or $manifest.Package.Identity.Version -cne $Version -or
      $manifest.Package.Identity.Publisher -cne $publisher -or $manifest.Package.Identity.ProcessorArchitecture -cne 'x64' -or
      $stamp.commit -cne $SourceCommit -or $stamp.dirty -or $stamp.channelBuild.buildId -cne $BuildId -or
      $stamp.channelBuild.windowsVersion -cne $Version) { throw 'Package identity/version/source stamp mismatch.' }
  $asar = $archive.GetEntry('app/resources/app.asar')
  if (!$asar) { throw 'Electron ASAR missing.' }
  $reader = New-Object IO.BinaryReader($asar.Open())
  try {
    $null=$reader.ReadBytes(12); $length=$reader.ReadUInt32()
    if ($length -le 0 -or $length -gt 33554432) { throw 'Invalid ASAR index size.' }
    $index = [Text.Encoding]::UTF8.GetString($reader.ReadBytes($length)) | ConvertFrom-Json
    $pty = $index.files.dist.files.node_modules.files.'node-pty'.files
    if (!$pty.'package.json' -or !$pty.lib.files.'index.js') { throw 'node-pty package.json/index.js missing.' }
  } finally { $reader.Dispose() }
  foreach ($name in @('pty.node','conpty.node','conpty_console_list.node','winpty-agent.exe','winpty.dll','conpty/OpenConsole.exe','conpty/conpty.dll')) {
    $entry=$archive.GetEntry('app/resources/app.asar.unpacked/dist/node_modules/node-pty/prebuilds/win32-x64/'+$name)
    if (!$entry -or $entry.Length -le 0) { throw "node-pty native closure missing: $name" }
  }
} finally { $archive.Dispose() }
if (!$LocalOnly) {
  $null=Get-Command ssh.exe,sftp.exe -ErrorAction Stop
  Write-Host 'Checking native SSH access (enter the ECS password if prompted).'
  $null=Ssh "set -eu; test -d '$RemoteRoot/releases'; test ! -L '$RemoteRoot'; command -v sha256sum"
}
$script:runBinding=([ordered]@{inputSha256=$ExpectedInputSha256;source=$SourceCommit;version=$Version;build=$BuildId;
  signer=$Thumbprint;target=$SshTarget;root=$RemoteRoot;origin=$PublicBase;profile=$UserProfile;
  setupContract='verify-remote-expected-sha256-v2'} | ConvertTo-Json -Compress)
if (!$NewRun -and !$PSBoundParameters.ContainsKey('Output')) {
  $candidates=Get-ChildItem -LiteralPath (Split-Path -Parent $Output) -Directory -Filter 'ecs-deploy-*' |
    Sort-Object LastWriteTime -Descending
  foreach($candidate in $candidates) {
    $savedPath=Join-Path $candidate.FullName 'checkpoints.jsonl'
    if (!(Test-Path -LiteralPath $savedPath)) {continue}
    $saved=Read-RunCheckpoint $savedPath
    if ($saved -and $saved.binding -ceq $script:runBinding) {
      if (!$saved.completed.PSObject.Properties['finished']) {$Output=$candidate.FullName}
      # A successful latest run starts fresh; do not revive older failed runs.
      break
    }
  }
}
if (!(Test-Path -LiteralPath $Output)) {New-Item -ItemType Directory -Path $Output | Out-Null}
$plan.output=$Output;$plan.profileBackup=Join-Path $Output 'profile-backup'
$script:statusLog = Join-Path $Output 'checkpoints.jsonl'
Report-Status "Input verification passed. Run log: $script:statusLog"
$receipt=[ordered]@{result='IN_PROGRESS';plan=$plan;unsignedSha256=$ExpectedInputSha256;CH2='UNCHANGED';
  reused=@('complete MSIX','npm dependencies','node-pty','Electron inputs','Python runtime/venv','uv cache','.NET/NuGet cache');
  signedMsix=$null;sha256=$null;localAcceptance=$null;publication='NOT_STARTED';
  h3='NOT_RUN';publicReadback='NOT_RUN';cleanup='NOT_STARTED';landingCutover='NOT_VERIFIED'}
$script:checkpoints=@{}
$checkpointPath=Join-Path $Output 'checkpoints.jsonl'
$saved=Read-RunCheckpoint $checkpointPath
if($saved) {
  if($saved.binding -cne $script:runBinding){throw 'Run configuration changed. Use -NewRun or a new -Output.'}
  foreach($property in $saved.completed.PSObject.Properties){$script:checkpoints[$property.Name]=$property.Value}
  $previous=$saved.receipt
  foreach($property in $previous.PSObject.Properties){$receipt[$property.Name]=$property.Value}
  Report-Status ('Resuming run; completed checkpoints: '+($script:checkpoints.Keys -join ', '))
} elseif(Test-Path -LiteralPath (Join-Path $Output 'result.json')) {
  # Explicitly selected older runs may predate checkpoints. Adopt only their
  # verified signed artifact; never infer local acceptance or publication.
  $previous=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw | ConvertFrom-Json
  if($previous.unsignedSha256 -cne $ExpectedInputSha256 -or $previous.plan.sourceCommit -cne $SourceCommit){throw 'Older run source mismatch.'}
  if($previous.sha256 -and (Test-Path -LiteralPath (Join-Path $Output 'SemantierAgent.msix')) -and
    (Hash (Join-Path $Output 'SemantierAgent.msix')) -ceq $previous.sha256) {
    $receipt.sha256=$previous.sha256;$script:checkpoints['signed']='imported-verified-hash'
  }
}
$receipt.result='IN_PROGRESS'
Write-RunEvent @{type='checkpoint';stage='resume-ready';binding=$script:runBinding;completed=$script:checkpoints;receipt=$receipt}
try {
  Save-Json $receipt (Join-Path $Output 'result.json')
  $signed=Join-Path $Output 'SemantierAgent.msix'
  Report-Status 'Signing and verifying the distribution MSIX copy.'
  if($script:checkpoints.ContainsKey('signed')) {
    if((Hash $signed) -cne $receipt.sha256){throw 'Checkpoint signed MSIX hash changed.'}
    Require-Signature $signed
    Report-Status 'Reusing checkpointed signed MSIX; no re-signing.'
  } else {Sign-Copy $Msix $signed}
  $hash=Hash $signed
  $receipt.signedMsix=$signed; $receipt.sha256=$hash
  $sidecar=$signed+'.sha256'
  [IO.File]::WriteAllText($sidecar, "$hash  $([IO.Path]::GetFileName($signed))`n",[Text.Encoding]::ASCII)
  Complete-Checkpoint 'signed'

  # The installer is intentionally version-agnostic: one immutable URL contract
  # identifies the ONE ECS MSIX, with a co-located .sha256 sidecar. No manifests.
  $setup=Join-Path $Output 'semantier-setup-ch1.exe'
  $setupExpected="${PublicBase}/releases/SemantierAgent.msix"
  $setupContractVersion=$setupExpected+'|verify-remote-expected-sha256-v2'
  $remoteSetup="$RemoteRoot/releases/installers/semantier-setup-ch1.exe"
  $uploadSetup=$false
  $reuseRemoteSetup=$false
  if(!$LocalOnly) {
    $remoteSetupState=Ssh "if test -s '$remoteSetup'; then if test -f '$remoteSetup.contract.txt'; then cat '$remoteSetup.contract.txt'; else printf UNKNOWN; fi; else printf MISSING; fi"
    if($UpdateSetup -or $remoteSetupState -ceq 'MISSING') {$uploadSetup=$true}
    elseif($remoteSetupState.Trim() -ceq $setupContractVersion) {$reuseRemoteSetup=$true}
    else {throw 'Existing ECS Setup compatibility is unverified. Run with -UpdateSetup once to install the stable-URL/checksum-validating installer.'}
  }
  if($reuseRemoteSetup) {
    Report-Status 'Reusing the compatible signed Setup already on ECS; no Setup build or upload.'
    $receipt.setupExe=$remoteSetup;$receipt.fixedMsixUrl=$setupExpected
  } else {
  if($UpdateSetup -and !$receipt.Contains('setupExplicitUpdate')){$script:checkpoints.Remove('setup')}
  if($script:checkpoints.ContainsKey('setup')) {
    Require-Signature $setup
    Report-Status 'Reusing checkpointed signed Setup.'
  } else {
  $setupExpected="${PublicBase}/releases/SemantierAgent.msix"
  $setupIsReusable=$false
  # Do not execute an older cached Setup merely to discover its command-line API.
  # Only our own prior fixed-file builds carry this separate contract marker.
  $cachedContract=$SetupExe+'.fixed-msix-url.txt'
  $setupContractVersion=$setupExpected+'|verify-remote-expected-sha256-v2'
  if (!$UpdateSetup -and (Test-Path -LiteralPath $SetupExe -PathType Leaf) -and
      (Test-Path -LiteralPath $cachedContract -PathType Leaf) -and
      (([IO.File]::ReadAllText($cachedContract)).Trim() -ceq $setupContractVersion)) {
    try {
      Require-Signature $SetupExe
      $probe=Join-Path $Output 'setup-contract.txt'
      Run-Setup $SetupExe @('--describe-fixed-download',$probe)
      if ((Test-Path -LiteralPath $probe -PathType Leaf) -and
          (([IO.File]::ReadAllText($probe)).Trim() -ceq $setupExpected)) {
        $setupIsReusable=$true
        Copy-Item -LiteralPath $SetupExe -Destination $setup
      }
    } catch { Write-Host 'Cached Setup does not support fixed-file contract; will build the small Setup only.' }
  }
  if (!$setupIsReusable) {
    Report-Status 'Building a compatible Setup EXE using the existing .NET cache.'
    $dotnet=Join-Path $env:ProgramFiles 'dotnet\dotnet.exe'
    if (!(Test-Path -LiteralPath $dotnet -PathType Leaf)) {
      throw 'SETUP_INCOMPATIBLE: fixed-MSIX Setup is unavailable and .NET SDK is missing.'
    }
    $setupWork=Join-Path $Output 'setup-build'
    New-Item -ItemType Directory -Path $setupWork -Force | Out-Null
    $project=@'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFramework>net8.0-windows</TargetFramework>
    <UseWindowsForms>true</UseWindowsForms>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
    <SelfContained>true</SelfContained>
    <PublishSingleFile>true</PublishSingleFile>
    <IncludeNativeLibrariesForSelfExtract>true</IncludeNativeLibrariesForSelfExtract>
    <AssemblyName>SemantierSetup</AssemblyName>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
</Project>
'@
    [IO.File]::WriteAllText((Join-Path $setupWork 'SemantierSetup.csproj'),$project,$utf8)
    $source=@'
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;

internal static class Program {
  private const string MsixUrl = "https://download.semantier.cn/releases/SemantierAgent.msix";
  private const string PackageName = "Semantier.Agent.Acceptance";
  private const string Publisher = "CN=Semantier Technology, O=Semantier Technology";
  private const long MaxBytes = 12000000000;
  [STAThread]
  private static int Main(string[] args) {
    if (args.Length == 2 && args[0] == "--describe-fixed-download") {
      File.WriteAllText(args[1], MsixUrl, new UTF8Encoding(false));
      return 0;
    }
    if (args.Length == 1 && args[0] == "--probe-h3") {
      try {
        using var client = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseProxy = false });
        var checksum = GetH3BytesAsync(client, MsixUrl + ".sha256", 4096, CancellationToken.None).GetAwaiter().GetResult();
        if (checksum.Length < 64) throw new InvalidDataException("H3 checksum probe failed.");
        return 0;
      } catch (Exception e) { File.WriteAllText(Path.Combine(Path.GetTempPath(), "semantier-setup-last-error.txt"), e.ToString()); return 1; }
    }
    if (args.Length == 2 && args[0] == "--verify-remote") {
      try {
        string expectedHash = args[1].ToLowerInvariant();
        if (expectedHash.Length != 64 || !expectedHash.All(Uri.IsHexDigit))
          throw new ArgumentException("Invalid expected local MSIX SHA-256.");
        var dir = Path.Combine(Path.GetTempPath(), "semantier-verify-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try { DownloadAndVerifyAsync(dir, CancellationToken.None, expectedHash).GetAwaiter().GetResult(); return 0; }
        finally { Directory.Delete(dir, true); }
      } catch (Exception e) { File.WriteAllText(Path.Combine(Path.GetTempPath(), "semantier-setup-last-error.txt"), e.ToString()); return 1; }
    }
    if (args.Length != 0) return 2;
    Application.EnableVisualStyles();
    using var form = new Form { Text = "Semantier Setup", Width = 440, Height = 190,
      StartPosition = FormStartPosition.CenterScreen, FormBorderStyle = FormBorderStyle.FixedDialog,
      MaximizeBox = false, MinimizeBox = false };
    var status = new Label { Text = "Install the latest signed Semantier application.", AutoSize = false,
      Left = 20, Top = 24, Width = 386, Height = 43 };
    var button = new Button { Text = "Download and install", Left = 20, Top = 86, Width = 185, Height = 36 };
    form.Controls.Add(status); form.Controls.Add(button);
    button.Click += async (_, _) => {
      button.Enabled = false;
      string? dir = null;
      try {
        status.Text = "Downloading over HTTP/3 and verifying checksum...";
        dir = Path.Combine(Path.GetTempPath(), "semantier-install-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var package = await DownloadAndVerifyAsync(dir, CancellationToken.None);
        status.Text = "Installing the signed package...";
        await Task.Run(() => InstallMsix(package));
        status.Text = "Installation successful.";
        MessageBox.Show(form, "Semantier is installed.", "Semantier", MessageBoxButtons.OK, MessageBoxIcon.Information);
        form.Close();
      } catch (Exception e) {
        status.Text = "Installation failed. No unsigned package was installed.";
        MessageBox.Show(form, e.Message, "Semantier Setup", MessageBoxButtons.OK, MessageBoxIcon.Error);
      } finally {
        if (dir != null) { try { Directory.Delete(dir, true); } catch { } }
        button.Enabled = true;
      }
    };
    Application.Run(form);
    return 0;
  }
  private static async Task<string> DownloadAndVerifyAsync(string dir, CancellationToken cancel, string? expectedHash = null) {
    using var handler = new SocketsHttpHandler { AllowAutoRedirect = false, UseProxy = false };
    using var client = new HttpClient(handler) { Timeout = TimeSpan.FromMinutes(60) };
    string packagePath = Path.Combine(dir, "SemantierAgent.msix");
    for (int attempt = 0; attempt != 3; attempt++) {
      string sidecar = (await GetH3BytesAsync(client, MsixUrl + ".sha256", 4096, cancel)).Trim();
      string wanted = sidecar.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
      if (wanted.Length != 64 || !wanted.All(Uri.IsHexDigit)) throw new InvalidDataException("Invalid SHA-256 sidecar.");
      // This run's locally signed MSIX is the publication authority; the
      // server's MSIX and sidecar could otherwise both belong to an older release.
      if (expectedHash is not null && !string.Equals(wanted, expectedHash, StringComparison.OrdinalIgnoreCase))
        throw new InvalidDataException("Public checksum is not the locally signed release SHA-256.");
      var observed = await DownloadH3Async(client, MsixUrl, packagePath, cancel);
      if (string.Equals(wanted, observed, StringComparison.OrdinalIgnoreCase) &&
          (expectedHash is null || string.Equals(expectedHash, observed, StringComparison.OrdinalIgnoreCase))) {
        VerifyIdentity(packagePath);
        return packagePath;
      }
      if (attempt == 2) throw new InvalidDataException("MSIX SHA-256 mismatch; installation stopped.");
      await Task.Delay(2000, cancel); // a release may swap file and checksum between two GETs
    }
    throw new InvalidDataException("No matching release.");
  }
  private static async Task<HttpResponseMessage> GetH3Async(HttpClient client, string url, CancellationToken cancel) {
    using var request = new HttpRequestMessage(HttpMethod.Get, url) {
      Version = HttpVersion.Version30,
      VersionPolicy = HttpVersionPolicy.RequestVersionExact
    };
    var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel);
    if (response.Version != HttpVersion.Version30 || !response.IsSuccessStatusCode ||
        response.Headers.Location != null || response.Content.Headers.ContentLength > MaxBytes) {
      response.Dispose(); throw new HttpRequestException("Required HTTP/3 endpoint unavailable.");
    }
    return response;
  }
  private static async Task<string> GetH3BytesAsync(HttpClient client, string url, int maximum, CancellationToken cancel) {
    using var response = await GetH3Async(client, url, cancel);
    using var body = await response.Content.ReadAsStreamAsync(cancel);
    using var memory = new MemoryStream();
    var bytes = new byte[4096]; int n;
    while ((n = await body.ReadAsync(bytes, cancel)) != 0) {
      if (memory.Length + n > maximum) throw new InvalidDataException("Checksum response too large.");
      memory.Write(bytes, 0, n);
    }
    return Encoding.ASCII.GetString(memory.ToArray());
  }
  private static async Task<string> DownloadH3Async(HttpClient client, string url, string file, CancellationToken cancel) {
    using var response = await GetH3Async(client, url, cancel);
    using var body = await response.Content.ReadAsStreamAsync(cancel);
    using var output = new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, true);
    using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    var buffer = new byte[1024 * 1024]; long total = 0; int count;
    while ((count = await body.ReadAsync(buffer, cancel)) != 0) {
      total += count;
      if (total > MaxBytes) throw new InvalidDataException("MSIX exceeds download size limit.");
      digest.AppendData(buffer, 0, count);
      await output.WriteAsync(buffer.AsMemory(0, count), cancel);
    }
    if (total == 0 || (response.Content.Headers.ContentLength is long expected && expected != total))
      throw new InvalidDataException("MSIX transfer incomplete.");
    await output.FlushAsync(cancel);
    return Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
  }
  private static void VerifyIdentity(string file) {
    using var zip = ZipFile.OpenRead(file);
    var entry = zip.GetEntry("AppxManifest.xml") ?? throw new InvalidDataException("Missing AppxManifest.");
    using var stream = entry.Open();
    var xml = XDocument.Load(stream);
    var identity = xml.Descendants().FirstOrDefault(x => x.Name.LocalName == "Identity")
      ?? throw new InvalidDataException("Missing package identity.");
    string Attribute(string key) => identity.Attribute(key)?.Value ?? "";
    if (Attribute("Name") != PackageName || Attribute("Publisher") != Publisher ||
        Attribute("ProcessorArchitecture") != "x64" || !Version.TryParse(Attribute("Version"), out _))
      throw new InvalidDataException("Unexpected MSIX identity / publisher / architecture.");
    if (zip.GetEntry("AppxSignature.p7x") is null) throw new InvalidDataException("Missing MSIX signature.");
    // Windows AppX deployment verifies signature trust and package integrity.
  }
  private static void InstallMsix(string path) {
    var quoted = path.Replace("'", "''");
    var command = "$ErrorActionPreference='Stop'; Add-AppxPackage -Path '" + quoted + "' -ForceApplicationShutdown -ForceUpdateFromAnyVersion";
    var encoded = Convert.ToBase64String(Encoding.Unicode.GetBytes(command));
    using var process = new Process { StartInfo = new ProcessStartInfo("powershell.exe") {
      UseShellExecute = false, CreateNoWindow = true, ArgumentList = { "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded }
    }};
    if (!process.Start()) throw new InvalidOperationException("Cannot launch Windows package installer.");
    process.WaitForExit();
    if (process.ExitCode != 0) throw new InvalidOperationException("Windows rejected the signed MSIX (exit code " + process.ExitCode + ").");
  }
}
'@
    [IO.File]::WriteAllText((Join-Path $setupWork 'Program.cs'),$source,$utf8)
    $env:DOTNET_CLI_HOME='D:\repo\deliverables\cache\dotnet'
    $env:NUGET_PACKAGES='D:\repo\deliverables\cache\nuget'
    Run $dotnet @('publish',(Join-Path $setupWork 'SemantierSetup.csproj'),'-c','Release',
      '-o',(Join-Path $setupWork 'published'),'-p:RestoreIgnoreFailedSources=true')
    $built=Join-Path $setupWork 'published\SemantierSetup.exe'
    if (!(Test-Path -LiteralPath $built -PathType Leaf)) {throw 'Built Setup EXE unavailable.'}
    Sign-Copy $built $setup
    New-Item -ItemType Directory -Path (Split-Path -Parent $SetupExe) -Force | Out-Null
    Copy-Item -LiteralPath $setup -Destination $SetupExe
    [IO.File]::WriteAllText($cachedContract,$setupContractVersion,$utf8)
  }
  Require-Signature $setup
  $probe=Join-Path $Output 'setup-contract-verified.txt'
  Run-Setup $setup @('--describe-fixed-download',$probe)
  if (!(Test-Path -LiteralPath $probe -PathType Leaf) -or
      ([IO.File]::ReadAllText($probe)).Trim() -cne $setupExpected) {
    throw 'SETUP_PROTOCOL_INCOMPATIBLE: signed Setup does not target the one fixed ECS MSIX.'
  }
  $receipt.setupExe=$setup
  $receipt.fixedMsixUrl=$setupExpected
  Save-Json $receipt (Join-Path $Output 'result.json')
  if($UpdateSetup){$receipt.setupExplicitUpdate=$true}
  Complete-Checkpoint 'setup'
  }
  }
  $receipt.setupAction=if($uploadSetup){'UPLOAD'}elseif($reuseRemoteSetup){'REUSE_ECS'}else{'LOCAL_ONLY'}

  # Stop only executables inside this registered package before copying its
  # existing profile. The acceptance app uses isolated state and cannot alter it.
  if(!$script:checkpoints.ContainsKey('local')) {
  $registered=@(Get-AppxPackage -Name $identity)
  Report-Status 'Closing the installed application and backing up the user profile.'
  foreach ($package in $registered) {
    $prefix=$package.InstallLocation.TrimEnd('\')+'\'
    Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force
  }
  $backup=Join-Path $Output 'profile-backup'
  $profileFiles=@()
  if($script:checkpoints.ContainsKey('backup')) {
    # Direct assignment avoids nesting the JSON array in Windows PowerShell 5.1.
    $profileFiles=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $Output 'profile-files.json')))
    foreach($file in $profileFiles){
      if($file.relative -isnot [string]){throw 'Invalid profile inventory entry: expected a single relative path.'}
      if(!(Test-Path -LiteralPath (Join-Path $backup $file.relative) -PathType Leaf)){throw 'Checkpoint profile backup file missing.'}
    }
    Report-Status 'Reusing original verified profile backup.'
  } else {
  if (Test-Path -LiteralPath $UserProfile) {
    $profilePath=(Resolve-Path -LiteralPath $UserProfile).Path.TrimEnd('\')
    # Runtime coordination files are regenerated, not persistent user state.
    $runtimeFiles=@('gateway.lock','gateway.pid','gateway_state.json','cron.pid','processes.json')
    Report-Status 'Backing up profile files, excluding transient gateway/PID state; no per-file hashing.'
    $databasePatterns=@('*.db','*.db-wal','*.db-shm','*.db-journal','*.sqlite','*.sqlite-*','*.sqlite3','*.sqlite3-*')
    & robocopy.exe $profilePath $backup /E /XJ /R:1 /W:1 /NFL /NDL /NP /XF @runtimeFiles @databasePatterns | Out-Host
    if ($LASTEXITCODE -gt 7) { throw 'Profile backup failed.' }
    Report-Status 'Taking consistent SQLite snapshots, including committed WAL data; live databases are not byte-hashed.'
    Preserve-Databases 'backup'
    $profileFiles=@(Get-ChildItem -LiteralPath $profilePath -Recurse -File | Where-Object { $_.Name -notin $runtimeFiles -and $_.Name -notmatch '\.(db|sqlite|sqlite3)(-wal|-shm|-journal)?$' } | ForEach-Object {
      $relative=$_.FullName.Substring($profilePath.Length+1)
      if (!(Test-Path -LiteralPath (Join-Path $backup $relative) -PathType Leaf)) {throw "Profile backup file missing: $relative"}
      @{relative=$relative}
    })
    $profileFiles+=@(Get-ChildItem -LiteralPath $backup -Recurse -File | Where-Object { $_.Extension -in @('.db','.sqlite','.sqlite3') } | ForEach-Object {
      @{relative=$_.FullName.Substring($backup.Length+1)}
    })
  }
  Save-Json @($profileFiles) (Join-Path $Output 'profile-files.json')
  Complete-Checkpoint 'backup'
  }
  Report-Status 'Installing the signed MSIX; preserving user profile.'
  if(!$script:checkpoints.ContainsKey('installed')) {
    $existing=@(Get-AppxPackage -Name $identity)
    $volumes=@(Get-AppxVolume)
    if($existing.Count -eq 1) {
      $installVolume=@($volumes | Where-Object {
        $existing[0].InstallLocation.StartsWith($_.PackageStorePath.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)
      })
    } else {$installVolume=@($volumes | Where-Object {$_.IsSystemVolume})}
    if($installVolume.Count -ne 1 -or $installVolume[0].IsOffline){throw 'Existing/system AppX installation volume is unavailable.'}
    Report-Status ('Installing on explicit AppX volume: '+$installVolume[0].PackageStorePath)
    Add-AppxPackage -Path $signed -Volume $installVolume[0] -ForceApplicationShutdown -ForceUpdateFromAnyVersion
  }
  $registered=@(Get-AppxPackage -Name $identity)
  if ($registered.Count -ne 1 -or $registered[0].Version.ToString() -cne $Version) {throw 'Installed registration mismatch.'}
  $archive=[IO.Compression.ZipFile]::OpenRead($signed)
  try {
    $stream=$archive.GetEntry('AppxSignature.p7x').Open();$sha=[Security.Cryptography.SHA256]::Create()
    try {$expectedSignature=[BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}
    finally {$sha.Dispose();$stream.Dispose()}
  } finally {$archive.Dispose()}
  if ((Hash (Join-Path $registered[0].InstallLocation 'AppxSignature.p7x')) -cne $expectedSignature) {throw 'Windows retained different same-version package bytes.'}
  $receipt.installedSignature=$expectedSignature
  Complete-Checkpoint 'installed'
  $binding=Join-Path $Output 'installed-binding.json';$uiOutput=Join-Path $Output ('acceptance-'+[Guid]::NewGuid().ToString('N'))
  Save-Json @{version=([Version]$Version).ToString(3);sourceCommit=$SourceCommit;artifactSha256=$hash;
    executable=(Join-Path $registered[0].InstallLocation 'app\Semantier Agent.exe');output=$uiOutput;
    playwright=(Join-Path $Repository 'node_modules\@playwright\test')} $binding
  $harness=@'
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const b=JSON.parse(fs.readFileSync(process.argv[2],'utf8').replace(/^\uFEFF/,''));
const {_electron}=require(b.playwright);fs.mkdirSync(b.output);let app;const errors=[];
(async()=>{try{
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>/^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|USERPROFILE|SYSTEMDRIVE|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS|APPDATA|LOCALAPPDATA|TEMP|TMP|PROGRAMFILES|PROGRAMFILES\(X86\))$/i.test(k)));
 app=await _electron.launch({executablePath:b.executable,cwd:path.dirname(b.executable),timeout:120000,
 env:{...env,HERMES_HOME:path.join(b.output,'hermes-home'),HERMES_DESKTOP_USER_DATA_DIR:path.join(b.output,'userdata'),
 HERMES_DESKTOP_IGNORE_EXISTING:'1',HERMES_DESKTOP_SKIP_QUIT_CONFIRM:'1',HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION:'1',HERMES_DESKTOP_CDP_PORT:'off'}});
 app.process().stderr.on('data',x=>{if(String(x).includes('ERR_MODULE_NOT_FOUND'))errors.push(String(x));});
 const facts=await app.evaluate(({app})=>({packaged:app.isPackaged,version:app.getVersion(),executable:process.execPath,platform:process.platform}));
 assert.equal(facts.packaged,true);assert.equal(facts.version,b.version);assert.equal(facts.platform,'win32');assert.equal(path.resolve(facts.executable),path.resolve(b.executable));
 const page=await app.firstWindow({timeout:120000});await page.waitForLoadState('domcontentloaded');assert(await page.isVisible('body'));
 assert(await app.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().some(w=>w.isVisible())));
 await page.waitForFunction(async()=>{try{const p=await window.hermesDesktop.api({path:'/api/profiles/active'});return typeof p.current==='string'&&p.current.length>0;}catch{return false;}},null,{timeout:180000,polling:2000});
 const pty=await page.evaluate(async()=>{const api=window.hermesDesktop.terminal,s=await api.start({cols:100,rows:30});let off,timer;
 try{return await new Promise((resolve,reject)=>{let output='';timer=setTimeout(()=>reject(Error('PTY deadline exceeded')),30000);
 off=api.onData(s.id,data=>{output+=typeof data==='string'?data:JSON.stringify(data);if(output.includes('SEMANTIER_REAL_PTY_AMD64_OK')){clearTimeout(timer);resolve({shell:s.shell,output});}});
 (async()=>{try{await api.attach(s.id);await api.write(s.id,'cmd.exe /d /c "echo SEMANTIER_REAL_PTY_%PROCESSOR_ARCHITECTURE%_OK"\r');}catch(e){clearTimeout(timer);reject(e);}})();});}
 finally{clearTimeout(timer);if(off)off();await api.dispose(s.id);}});
 assert.deepEqual(errors,[]);await page.screenshot({path:path.join(b.output,'window.png')});
 fs.writeFileSync(path.join(b.output,'acceptance.json'),JSON.stringify({result:'WINDOW_PTY_PASS',binding:b,facts,pty,errors},null,2));
 }finally{if(app)await app.close();}})().catch(e=>{fs.writeFileSync(path.join(b.output,'failure.txt'),e.stack);process.exitCode=1;});
'@
  $harnessPath=Join-Path $Output 'accept-installed.cjs';[IO.File]::WriteAllText($harnessPath,$harness,$utf8)
  Report-Status 'Opening Electron and testing a real PTY command; backend startup can take several minutes.'
  Run $Node @($harnessPath,$binding)
  $ui=Get-Content -LiteralPath (Join-Path $uiOutput 'acceptance.json') -Raw | ConvertFrom-Json
  if ($ui.result -cne 'WINDOW_PTY_PASS' -or $ui.binding.artifactSha256 -cne $hash) {throw 'Installed UI/PTY evidence mismatch.'}
  foreach ($file in $profileFiles) {
    if($file.relative -match '\.(db|sqlite|sqlite3)$'){continue}
    if (!(Test-Path -LiteralPath (Join-Path $UserProfile $file.relative) -PathType Leaf)) {throw "User profile file missing: $($file.relative)"}
  }
  if(Test-Path -LiteralPath $backup){Preserve-Databases 'verify'}
  $receipt.localAcceptance='LOCAL_SIGNED_MSIX_ELECTRON_PTY_PASS';$receipt.profileBackup=$backup
  Report-Status 'Local Electron, PTY and profile-preservation acceptance passed.'
  Save-Json $receipt (Join-Path $Output 'result.json')
  $receipt.uiReceipt=Join-Path $uiOutput 'acceptance.json'
  Complete-Checkpoint 'local'
  } else {
    $registered=@(Get-AppxPackage -Name $identity)
    if($registered.Count -ne 1 -or $registered[0].Version.ToString() -cne $Version -or
      (Hash (Join-Path $registered[0].InstallLocation 'AppxSignature.p7x')) -cne $receipt.installedSignature){throw 'Installed package changed since local acceptance; start a new run.'}
    $ui=Get-Content -LiteralPath $receipt.uiReceipt -Raw | ConvertFrom-Json
    if($ui.result -cne 'WINDOW_PTY_PASS' -or $ui.binding.artifactSha256 -cne $hash){throw 'Checkpoint local acceptance mismatch.'}
    Report-Status 'Reusing verified installation and Electron/PTY acceptance.'
  }
  if ($LocalOnly) {
    $receipt.h3='NOT_RUN_LOCAL_ONLY'
    $receipt.publicReadback='NOT_RUN_LOCAL_ONLY'
    $receipt.result='LOCAL_PASS_NOT_PUBLISHED'
    return
  }

  # One public MSIX and one published signed Setup; neither is version-pinned.
  # Upload outside Caddy's document root; mv on the SAME filesystem is atomic.
  $remoteMsix="$RemoteRoot/releases/SemantierAgent.msix"
  $remoteProof="set -eu; test `$(sha256sum '$remoteMsix' | cut -d' ' -f1) = '$hash'; test -s '$RemoteRoot/releases/installers/semantier-setup-ch1.exe'"
  if($script:checkpoints.ContainsKey('published') -and !$uploadSetup) {
    $null=Ssh $remoteProof
    Report-Status 'Remote published hashes still match; skipping upload and activation.'
  } else {
  $remoteChecksum="$remoteMsix.sha256"
  $remoteSetup="$RemoteRoot/releases/installers/semantier-setup-ch1.exe"
  $setupSize=if($uploadSetup){(Get-Item -LiteralPath $setup).Length}else{0}
  $msixSize=(Get-Item -LiteralPath $signed).Length
  # Compute the POSIX parent in PowerShell without Windows Split-Path (which uses backslashes).
  $posixParent=$RemoteRoot.Substring(0,$RemoteRoot.LastIndexOf('/'))
  $stage="$posixParent/.semantier-stage-$([Guid]::NewGuid().ToString('N'))"
  if($receipt.Contains('remoteStage')) {$stage=[string]$receipt.remoteStage}
  if($stage -cnotmatch ('^'+[Regex]::Escape($posixParent)+'/\.semantier-stage-[a-f0-9]{32}$')) {throw 'Invalid saved staging path.'}
  $minimumFree=$msixSize+$setupSize+67108864
  $prepare=@'
set -eu
root='__ROOT__'
stage='__STAGE__'
minimum='__MIN_FREE__'
test -d "$root/releases" && test ! -L "$root"
if test -d "$stage" && test ! -L "$stage"; then exit 0; fi
test ! -e "$stage"
free=$(df -PB1 "$root" | awk 'NR==2 {print $4}')
test "$free" -ge "$minimum" || { echo 'Insufficient ECS staging space' >&2; exit 9; }
mkdir -m 700 "$stage"
'@
  $prepare=$prepare.Replace('__ROOT__',$RemoteRoot).Replace('__STAGE__',$stage).Replace('__MIN_FREE__',[string]$minimumFree)
  Report-Status 'Checking ECS staging capacity and preparing the upload directory.'
  $null=Ssh $prepare
  $receipt.remoteStage=$stage
  Save-Json $receipt (Join-Path $Output 'result.json')
  Upload $signed "$stage/SemantierAgent.msix"
  Upload $sidecar "$stage/SemantierAgent.msix.sha256"
  if($uploadSetup) {
    Upload $setup "$stage/semantier-setup-ch1.exe"
    $contractFile=Join-Path $Output 'setup.contract.txt'
    [IO.File]::WriteAllText($contractFile,$setupContractVersion,$utf8)
    Upload $contractFile "$stage/setup.contract.txt"
  }
  $publish=@'
set -eu
root='__ROOT__'
stage='__STAGE__'
msix="$root/releases/SemantierAgent.msix"
setup="$root/releases/installers/semantier-setup-ch1.exe"
checksum="$root/releases/SemantierAgent.msix.sha256"
test -d "$stage" && test ! -L "$stage"
cd "$stage"
sha256sum -c SemantierAgent.msix.sha256
mkdir -p "$root/releases/installers"
# A reader opening the stable MSIX gets either the old complete inode or the
# new complete inode. There is NO manifest/catalog/version swap.
mv -f SemantierAgent.msix "$msix"
mv -f SemantierAgent.msix.sha256 "$checksum"
if test -f semantier-setup-ch1.exe; then
  mv -f semantier-setup-ch1.exe "$setup"
  mv -f setup.contract.txt "$setup.contract.txt"
  chmod 644 "$setup" "$setup.contract.txt"
fi
chmod 644 "$msix" "$checksum"
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
rmdir "$stage"
printf 'SINGLE_MSIX_ATOMIC_REPLACE_PASS\n'
'@
  $publish=$publish.Replace('__ROOT__',$RemoteRoot).Replace('__STAGE__',$stage)
  Report-Status 'Verifying uploaded hashes on ECS and replacing the public files.'
  $receipt.publication=Ssh $publish
  Save-Json $receipt (Join-Path $Output 'result.json')
  Complete-Checkpoint 'published'
  }
  $null=Ssh $remoteProof
  Report-Status 'Verifying public H3 status and package size with HEAD; no package body download.'
  Test-PublicMsix
  $receipt.h3='PUBLIC_H3_HEAD_STATUS_SIZE_PASS'
  $receipt.publicReadback='LIGHTWEIGHT_HEAD_ONLY; full public content hash NOT verified; ECS disk SHA256 matches local signed MSIX'
  if($VerifyPublicDownload) {
    if(!$script:checkpoints.ContainsKey('public-readback')) {
      Report-Status 'Explicit full public download requested; this incurs ECS outbound traffic.'
      Test-PublicMsix -FullDownload
      Complete-Checkpoint 'public-readback'
    }
    $receipt.publicReadback='FULL_MSIX_H3_DOWNLOAD_MATCHES_LOCAL_SIGNED_SHA256'
  }
  Save-Json $receipt (Join-Path $Output 'result.json')

  # Cleanup is permitted only after the deployed CN Landing stops depending on
  # legacy catalog.json. Check the ACTUAL production page with a browser.
  # Failed/missing Playwright or an old Landing => retain old catalog/packages.
  $landingReady=$false
  if (!$SkipOldCleanup) {
    $landingProbe=Join-Path $Output 'verify-live-landing.cjs'
    $landingProbeSource=@'
const {chromium}=require(process.argv[2]+'/node_modules/@playwright/test');
const fs=require('node:fs');
(async()=>{
  const expected='https://download.semantier.cn/releases/installers/semantier-setup-ch1.exe';
  const browser=await chromium.launch({headless:true});
  const observations=[];
  try {
    for(const host of ['https://semantier.cn/','https://www.semantier.cn/']) {
      const page=await browser.newPage();
      const catalogRequests=[];
      page.on('request',r=>{if(/\/(?:manual\/)?catalog\.json(?:\?|$)/i.test(r.url())) catalogRequests.push(r.url());});
      const response=await page.goto(host,{waitUntil:'domcontentloaded',timeout:30000});
      if(!response || !response.ok()) throw Error('Landing HTTP failure: '+host);
      const link=page.locator('a[href="'+expected+'"]');
      await link.first().waitFor({state:'visible',timeout:15000});
      const href=await link.first().getAttribute('href');
      if(new URL(href,page.url()).href!==expected) throw Error('Landing installer link mismatch: '+host);
      if(catalogRequests.length) throw Error('Landing still calls catalog: '+host);
      observations.push({host,url:page.url(),setupHref:expected,visibleFixedLinks:await link.count(),catalogRequests:0});
      await page.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(process.argv[3],JSON.stringify({result:'LIVE_LANDING_FIXED_SETUP_PASS',observations},null,2));
})().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
'@
    [IO.File]::WriteAllText($landingProbe,$landingProbeSource,$utf8)
    $landingEvidence=Join-Path $Output 'landing-cutover.json'
    try {
      Report-Status 'Checking both live Landing hosts before cleanup.'
      Run $Node @($landingProbe,$Repository,$landingEvidence)
      $landingProof=Get-Content -LiteralPath $landingEvidence -Raw | ConvertFrom-Json
      if ($landingProof.result -cne 'LIVE_LANDING_FIXED_SETUP_PASS' -or $landingProof.observations.Count -ne 2) {
        throw 'Landing browser evidence incomplete.'
      }
      $landingReady=$true
      $receipt.landingCutover='LIVE_CN_LANDING_CUTOVER_VERIFIED'
    } catch {
      $receipt.landingCutover='NOT_VERIFIED: '+$_.Exception.Message
      $receipt.cleanup='DEFERRED_LANDING_CUTOVER_REQUIRED'
      Write-Warning 'MSIX is published and hash verified; legacy cleanup deferred because live Landing is not yet verified on the fixed Setup link.'
    }
    Save-Json $receipt (Join-Path $Output 'result.json')
  }
  if (!$SkipOldCleanup -and $landingReady -and !$script:checkpoints.ContainsKey('cleanup')) {
    # No more version-specific installer/MSIX publication. Keep ALL old Setup
    # EXEs because existing Landing copies may still link to them. Remove ONLY
    # old Semantier release-package formats, never publisher state or R2.
    $cleanup=@'
set -eu
root='__ROOT__'
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
old="$root/releases/channel-builds"
if test -d "$old"; then
  find "$old" -mindepth 2 -maxdepth 5 -type f \( -name '*.msix' -o -name '*.msixbundle' -o -name '*.zip' -o -name '*.msix.sha256' -o -name '*.msixbundle.sha256' -o -name '*.zip.sha256' \) -print -delete
fi
# The running Landing has been browser-verified to use the fixed Setup link.
# Only now may these legacy discovery pointers be removed.
rm -f "$root/releases/current-release.json" "$root/releases/build.json" \
      "$root/releases/channels/semantier-acceptance.json" \
      "$root/releases/manual/catalog.json"
printf 'OLD_PACKAGE_CLEANUP_PASS\n'
'@
    Report-Status 'Removing obsolete ECS packages and legacy discovery files.'
    $receipt.cleanup=Ssh ($cleanup.Replace('__ROOT__',$RemoteRoot))
    Complete-Checkpoint 'cleanup'
  } elseif ($SkipOldCleanup) {
    $receipt.cleanup='SKIPPED_BY_REQUEST'
  }
  if ((Hash $Msix) -cne $ExpectedInputSha256 -or (Hash $signed) -cne $hash) {
    throw 'Local package changed during publication.'
  }
  $receipt.result=if ($landingReady) { 'MSIX_PUBLISHED_VERIFIED_LANDING_CUTOVER_PASS' } else { 'MSIX_PUBLISHED_VERIFIED_CLEANUP_DEFERRED' }
  if($landingReady -and !$SkipOldCleanup){Complete-Checkpoint 'finished'}
} catch {
  $receipt.result='FAIL';$receipt.error=$_.Exception.Message
  Report-Status ('FAILED: '+$_.Exception.Message)
  throw
} finally {
  $receipt.finishedAt=[DateTime]::UtcNow.ToString('o')
  Save-Json $receipt (Join-Path $Output 'result.json')
  Report-Status ('Finished: '+$receipt.result)
  [IO.File]::WriteAllText((Join-Path $Output 'release-note.md'),
    "Result: $($receipt.result)`nVersion: $Version`nSource: $SourceCommit`nSigned MSIX: $($receipt.signedMsix)`nSHA-256: $($receipt.sha256)`nFixed public MSIX: $PublicBase/releases/SemantierAgent.msix`nFixed Setup: $PublicBase/releases/installers/semantier-setup-ch1.exe`nLocal acceptance: $($receipt.localAcceptance)`nPublication: $($receipt.publication)`nH3: $($receipt.h3)`nPublic readback: $($receipt.publicReadback)`nLanding cutover: $($receipt.landingCutover)`nCleanup: $($receipt.cleanup)`nCH2: unchanged`n",$utf8)
  Write-Host "Result and evidence: $Output"
}
