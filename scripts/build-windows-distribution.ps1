<#
.SYNOPSIS
  Native half of scripts/build_hermes_windows.py (Semantier parent repository).
  Retains a dedicated native checkout/cache. Never resets an occupied checkout,
  pushes, installs, creates certificates or changes trust. All output directories are fresh.
#>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$request = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$repo = [IO.Path]::GetFullPath($request.repository).TrimEnd('\')
$work = [IO.Path]::GetFullPath($request.worktree).TrimEnd('\')
$output = [IO.Path]::GetFullPath($request.output).TrimEnd('\')
foreach ($path in @($repo,$work,$output)) {
  if ($path -notmatch '^[A-Za-z]:\\' -or $path.Length -le 3) { throw 'Use Windows-native drive paths, not WSL/UNC roots' }
}
$standalone = $repo -ieq $work
if (!$standalone -and ($repo.StartsWith($work + '\',[StringComparison]::OrdinalIgnoreCase) -or
    $work.StartsWith($repo + '\',[StringComparison]::OrdinalIgnoreCase))) { throw 'Separate repository and worktree paths must not be nested' }
if ($output -eq $work -or $output.StartsWith($work + '\',[StringComparison]::OrdinalIgnoreCase) -or
    $work.StartsWith($output + '\',[StringComparison]::OrdinalIgnoreCase) -or
    $output -eq $repo -or $output.StartsWith($repo + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be outside both checkouts' }
if ($request.commit -notmatch '^[0-9a-f]{40}$') { throw 'Exact complete source commit required' }
if (Test-Path -LiteralPath $output) { throw 'Output already exists; choose a fresh directory' }
$lockPath = $work + '.distribution.lock'
$lock = $null
$transcript = $false
$totalClock = [Diagnostics.Stopwatch]::StartNew()
function Git-Value([string]$Directory, [string[]]$Arguments) {
  $value = & git -C $Directory @Arguments
  if ($LASTEXITCODE -ne 0) { throw "git failed ($LASTEXITCODE): $($Arguments -join ' ')" }
  return $value
}
function Native([string]$Tool, [string[]]$Arguments) {
  $saved = $ErrorActionPreference
  $clock = [Diagnostics.Stopwatch]::StartNew()
  $code = $null
  $global:LASTEXITCODE = $null
  $ErrorActionPreference = 'Continue'
  try { & $Tool @Arguments 2>&1 | Out-Host; $code = $LASTEXITCODE }
  finally {
    $ErrorActionPreference = $saved
    $clock.Stop()
    if ($transcript) {
      [ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');tool=[IO.Path]::GetFileName($Tool);exitCode=$code;elapsedSeconds=$clock.Elapsed.TotalSeconds} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $output 'controller-stages.jsonl') -Encoding utf8
    }
  }
  if ($null -eq $code -or $code -ne 0) { throw "$Tool failed ($code)" }
}
function Json-File($Value, [string]$Path) { $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding utf8 }
function Hash([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Find-Tool([string]$Explicit, [string[]]$Candidates, [string]$Name) {
  if ($Explicit) { if (!(Test-Path -LiteralPath $Explicit -PathType Leaf)) { throw "$Name not found: $Explicit" }; return $Explicit }
  foreach ($candidate in $Candidates) { if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate } }
  throw "$Name unavailable; pass its explicit Windows path"
}
try {
  $lock = [IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  New-Item -ItemType Directory -Path $output | Out-Null
  Start-Transcript -LiteralPath (Join-Path $output 'release.log') | Out-Null
  $transcript = $true
  $cert = Get-Item -LiteralPath ("Cert:\CurrentUser\My\" + $request.thumbprint)
  $now = Get-Date
  $basic = @($cert.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.19' })
  if (!$cert.HasPrivateKey -or $cert.NotBefore -gt $now -or $cert.NotAfter -le $now -or
      $cert.EnhancedKeyUsageList.ObjectId -notcontains '1.3.6.1.5.5.7.3.3' -or
      $basic.Count -ne 1 -or $basic[0].CertificateAuthority) { throw 'Valid end-entity code-signing certificate and private key required' }
  $sdk = @(Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object FullName)
  $signTool = Find-Tool $request.signTool ($sdk + @((Join-Path $repo '.build\desktop-job\packager\winCodeSign\x64\signtool.exe'))) 'SignTool'
  $sevenZip = Find-Tool $request.sevenZip @("$env:ProgramFiles\7-Zip\7z.exe", (Join-Path $repo '.build\desktop-job\packager\sevenZip\bin\7za.exe')) '7-Zip'
  $source = $request.commit
  if (!$request.unsignedReceipt) {
    $marker = $work + '.distribution-owner.json'
    $existing = Test-Path -LiteralPath $work
    if (!$existing -and (Test-Path -LiteralPath $marker)) { throw 'Stale worktree owner marker requires inspection' }
    if ($existing -and !(Test-Path -LiteralPath $marker)) { throw 'Occupied worktree has no ownership marker; choose a different path' }
    if ($standalone -and !$existing) {
      Native 'git' @('init',$work)
    }
    $common = [IO.Path]::GetFullPath([string](Git-Value $repo @('rev-parse','--path-format=absolute','--git-common-dir')))
    if (!$standalone) {
      $originalHead = [string](Git-Value $repo @('rev-parse','HEAD'))
      $originalStatus = @(Git-Value $repo @('status','--porcelain','--untracked-files=all'))
    }
    if ($existing) {
      $owner = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
      if ($owner.repository -cne $repo -or $owner.commonDirectory -cne $common -or $owner.kind -cne 'hermes-windows-distribution-v1') { throw 'Worktree ownership mismatch' }
      if ([IO.Path]::GetFullPath([string](Git-Value $work @('rev-parse','--show-toplevel'))) -ine $work -or
          [IO.Path]::GetFullPath([string](Git-Value $work @('rev-parse','--path-format=absolute','--git-common-dir'))) -ine $common) { throw 'Worktree registration mismatch' }
      if (@(Git-Value $work @('status','--porcelain','--untracked-files=all')).Count) { throw 'Build worktree is dirty; preserve it and resolve changes or choose a fresh path' }
      if (@(Git-Value $work @('branch','--show-current')).Count) { throw 'Owned build worktree must remain detached' }
    }
    $bundle = Join-Path $output 'source.bundle'
    Copy-Item -LiteralPath $request.bundle -Destination $bundle
    Native 'git' @('-C',$repo,'bundle','verify',$bundle)
    Native 'git' @('-C',$repo,'fetch','--no-write-fetch-head',$bundle,$request.bundleRef)
    $transferred = [string](Git-Value $repo @('rev-parse',($request.commit + '^{commit}')))
    if ($transferred -cne $request.commit) { throw 'Transferred commit identity mismatch' }
    if ($existing -or $standalone) {
      Native 'git' @('-C',$work,'checkout','--detach',$source)
    } else {
      Native 'git' @('-C',$repo,'worktree','add','--detach',$work,$source)
    }
    if (!$existing) {
      Json-File ([ordered]@{kind='hermes-windows-distribution-v1';repository=$repo;commonDirectory=$common;createdAt=[DateTime]::UtcNow.ToString('o')}) $marker
    }
    if ([string](Git-Value $work @('rev-parse','HEAD')) -cne $source) { throw 'Build HEAD mismatch' }
    if (!$standalone -and ([string](Git-Value $repo @('rev-parse','HEAD')) -cne $originalHead -or
        (($originalStatus -join "`n") -cne (@(Git-Value $repo @('status','--porcelain','--untracked-files=all')) -join "`n")))) { throw 'Original checkout changed concurrently; build stopped without resetting it' }
    if ($request.preflight) {
      Json-File ([ordered]@{result='PREFLIGHT_PASS';sourceCommit=$source;certificateThumbprint=$cert.Thumbprint;certificateSubject=$cert.Subject;certificateNotBefore=$cert.NotBefore.ToUniversalTime().ToString('o');certificateNotAfter=$cert.NotAfter.ToUniversalTime().ToString('o');signTool=$signTool;sevenZip=$sevenZip;repository=$repo;standalone=$standalone;worktree=$work}) (Join-Path $output 'preflight.json')
      Write-Host 'Preflight passed. No build, signing, installation or trust change.'
      exit 0
    }
    $counter = Join-Path (Split-Path $work -Parent) 'semantier-agent-build-version.json'
    $retryVersion = if ($request.PSObject.Properties.Name -contains 'retryVersion') { [string]$request.retryVersion } else { '' }
    $semantierVersion = & (Join-Path $PSScriptRoot 'next-windows-build-version.ps1') -State $counter -SourceCommit $source -RetryVersion $retryVersion
    if ($retryVersion) {
      Json-File ([ordered]@{kind='semantier-windows-build-version-v1';version=$semantierVersion;sourceCommit=$source;reservedAt=[DateTime]::UtcNow.ToString('o');explicitRetry=$true;counterUnchanged=$true}) (Join-Path $output 'build-version.json')
    } else { Copy-Item -LiteralPath $counter -Destination (Join-Path $output 'build-version.json') }
    $arguments = @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $work 'scripts\build-bundle.ps1'))
    if ($request.python) { $arguments += @('-Python',$request.python) }
    # This controller holds the workspace lock until every native build child exits.
    # Only this serial context may share downloads with incomplete compiler identity.
    $priorSemantierVersion = [Environment]::GetEnvironmentVariable('SEMANTIER_DESKTOP_VERSION','Process')
    $priorSerialCache = [Environment]::GetEnvironmentVariable('HERMES_DESKTOP_SERIAL_CACHE','Process')
    try {
      $env:SEMANTIER_DESKTOP_VERSION = $semantierVersion
      $env:HERMES_DESKTOP_SERIAL_CACHE = Join-Path $work '.cache\desktop-inputs'
      Native "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" $arguments
    } finally {
      [Environment]::SetEnvironmentVariable('SEMANTIER_DESKTOP_VERSION',$priorSemantierVersion,'Process')
      [Environment]::SetEnvironmentVariable('HERMES_DESKTOP_SERIAL_CACHE',$priorSerialCache,'Process')
    }
    if ([string](Git-Value $work @('rev-parse','HEAD')) -cne $source -or
        @(Git-Value $work @('status','--porcelain','--untracked-files=all')).Count) { throw 'Build source changed during build' }
    $packages = @(Get-ChildItem -LiteralPath (Join-Path $work 'apps\desktop\release') -Filter '*.msix' -File)
    if ($packages.Count -ne 1) { throw "Expected exactly one MSIX, found $($packages.Count)" }
    $msix = $packages[0].FullName
    $receipt = [ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');sourceCommit=$source;buildResult='PASS';buildExitCode=0;originalBuildArtifact=$msix;unsignedSha256=(Hash $msix);sizeBytes=$packages[0].Length}
  } else {
    $receipt = Get-Content -LiteralPath $request.unsignedReceipt -Raw | ConvertFrom-Json
    $msix = $receipt.preservedUnsignedArtifact
    $source = $receipt.sourceCommit
    if ($source -notmatch '^[0-9a-f]{40}$' -or $receipt.buildResult -cne 'PASS' -or $receipt.buildExitCode -ne 0 -or
        (Hash $msix) -cne $receipt.unsignedSha256 -or (Get-Item -LiteralPath $msix).Length -ne $receipt.sizeBytes) { throw 'Unsigned build receipt/source/hash mismatch' }
  }
  $unsigned = Join-Path $output 'unsigned'
  New-Item -ItemType Directory -Path $unsigned | Out-Null
  $copy = Join-Path $unsigned ([IO.Path]::GetFileName($msix))
  Copy-Item -LiteralPath $msix -Destination $copy
  if ((Hash $copy) -cne $receipt.unsignedSha256) { throw 'Unsigned copy checksum mismatch' }
  $receipt.preservedUnsignedArtifact = $copy
  Json-File $receipt (Join-Path $unsigned 'unsigned-build-receipt.json')
  $signer = Join-Path $PSScriptRoot '..\apps\desktop\scripts\sign-windows-distribution.ps1'
  # Child shell gives a reliable native exit code even when the signing script throws.
  $signArgs = @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$signer,'-Msix',$copy,'-Thumbprint',$request.thumbprint,'-SourceCommit',$source,'-SignTool',$signTool,'-SevenZip',$sevenZip,'-Output',(Join-Path $output 'signed'))
  if ($request.timestampUrl) { $signArgs += @('-TimestampUrl',$request.timestampUrl) }
  Native "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" $signArgs
  Json-File ([ordered]@{result='PASS';sourceCommit=$source;controllerCommit=$request.commit;certificateThumbprint=$cert.Thumbprint;trustedTimestamp=[bool]$request.timestampUrl;worktree=$work;unsignedReceipt=(Join-Path $unsigned 'unsigned-build-receipt.json');signedReceipt=(Join-Path $output 'signed\receipt.json');zip=(Join-Path (Join-Path $output 'signed') ((Get-Content -LiteralPath (Join-Path $output 'signed\receipt.json') -Raw | ConvertFrom-Json).archiveName))}) (Join-Path $output 'result.json')
  Write-Host "Distribution complete: $output"
} catch {
  if ($transcript) { Json-File ([ordered]@{result='FAIL';producedAt=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message}) (Join-Path $output 'failure.json') }
  throw
} finally {
  $totalClock.Stop()
  if ($transcript) {
    Json-File ([ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');elapsedSeconds=$totalClock.Elapsed.TotalSeconds;sourceCommit=$request.commit;passed=(Test-Path -LiteralPath (Join-Path $output 'result.json'))}) (Join-Path $output 'controller-total.json')
    Stop-Transcript | Out-Null
  }
  if ($lock) { $lock.Dispose(); Remove-Item -LiteralPath $lockPath }
}
