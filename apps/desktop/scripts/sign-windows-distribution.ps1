# Sign only a distribution copy; never modify machine trust or install packages.
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Msix,
  [Parameter(Mandatory=$true)][string]$Thumbprint,
  [Parameter(Mandatory=$true)][string]$SourceCommit,
  [Parameter(Mandatory=$true)][string]$SignTool,
  [string]$MakeAppx = '', # Retained for existing callers; manifest inspection needs no unpacker.
  [Parameter(Mandatory=$true)][string]$SevenZip,
  [Parameter(Mandatory=$true)][string]$Output,
  [string]$TimestampUrl = ''
)
$ErrorActionPreference = 'Stop'
function Run-Native([string]$Tool, [string[]]$Arguments) {
  $watch = [Diagnostics.Stopwatch]::StartNew()
  $savedPreference = $ErrorActionPreference
  $code = $null
  try {
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = $null
    & $Tool @Arguments | Out-Host
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $savedPreference
    $watch.Stop()
    [ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');
      stage=([IO.Path]::GetFileName($Tool)+':'+$Arguments[0]);exitCode=$code;
      elapsedSeconds=$watch.Elapsed.TotalSeconds} | ConvertTo-Json -Compress |
      Add-Content -LiteralPath (Join-Path $Output 'distribution-stages.jsonl') -Encoding utf8
  }
  if ($null -eq $code -or $code -ne 0) { throw "$Tool failed ($code)" }
}
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Complete source commit required' }
if (Test-Path -LiteralPath $Output) { throw 'Use a fresh output directory' }
$Msix = (Resolve-Path -LiteralPath $Msix).Path
$unsignedHash = (Get-FileHash -LiteralPath $Msix -Algorithm SHA256).Hash.ToLowerInvariant()
$cert = Get-Item -LiteralPath "Cert:\CurrentUser\My\$Thumbprint"
$now = Get-Date
if (!$cert.HasPrivateKey -or $cert.NotBefore -gt $now -or $cert.NotAfter -le $now) { throw 'Valid certificate/private key unavailable' }
if ($cert.EnhancedKeyUsageList.ObjectId -notcontains '1.3.6.1.5.5.7.3.3') { throw 'Code-signing EKU required' }
$basic = $cert.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.19' }
if (!$basic -or $basic.CertificateAuthority) { throw 'End-entity Basic Constraints required' }
New-Item -ItemType Directory -Path $Output | Out-Null
$dist = Join-Path $Output 'distribution'
New-Item -ItemType Directory -Path $dist | Out-Null
try {
  # Reading one ZIP entry avoids extracting the complete multi-GB runtime and MAX_PATH failures.
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead($Msix)
  try {
    if (@($archive.Entries | Where-Object { $_.FullName -ieq 'AppxSignature.p7x' }).Count) { throw 'Expected unsigned MSIX input' }
    $entries = @($archive.Entries | Where-Object { $_.FullName -ceq 'AppxManifest.xml' })
    if ($entries.Count -ne 1) { throw 'Expected one MSIX manifest' }
    $reader = New-Object IO.StreamReader($entries[0].Open())
    try { [xml]$manifest = $reader.ReadToEnd() } finally { $reader.Dispose() }
  } finally { $archive.Dispose() }
  $identity = $manifest.Package.Identity
  if ([string]$identity.Publisher -cne $cert.Subject) { throw "Publisher mismatch: $($identity.Publisher) != $($cert.Subject)" }
  $signed = Join-Path $dist ([IO.Path]::GetFileName($Msix))
  Copy-Item -LiteralPath $Msix -Destination $signed
  $signArgs = @('sign','/fd','SHA256','/sha1',$Thumbprint)
  if ($TimestampUrl) { $signArgs += @('/tr',$TimestampUrl,'/td','SHA256') }
  Run-Native $SignTool ($signArgs + $signed)
  # This is deliberately before ZIP assembly; do not import certificates on failure.
  Run-Native $SignTool @('verify','/pa','/v',$signed)
  Export-Certificate -Cert $cert -FilePath (Join-Path $dist 'SemantierAgent-signing.cer') | Out-Null
  $readableVersion = ([string]$identity.Version) -replace '\.0$',''
  $archiveName = "SemantierAgent-$readableVersion-win-$($identity.ProcessorArchitecture).zip"
  $metadata = [ordered]@{
    archiveName = $archiveName; displayName = [string]$manifest.Package.Properties.DisplayName
    producedAt = [DateTime]::UtcNow.ToString('o'); sourceCommit = $SourceCommit
    packageName = [string]$identity.Name; packageVersion = [string]$identity.Version
    architecture = [string]$identity.ProcessorArchitecture; publisher = [string]$identity.Publisher
    certificateThumbprint = $cert.Thumbprint; certificateNotBefore = $cert.NotBefore.ToUniversalTime().ToString('o')
    certificateNotAfter = $cert.NotAfter.ToUniversalTime().ToString('o')
    unsignedMsix = $Msix; unsignedSha256 = $unsignedHash
    signedSha256 = (Get-FileHash -LiteralPath $signed -Algorithm SHA256).Hash.ToLowerInvariant()
    signing = 'PASS'; verification = 'PASS: SignTool verify /pa /v'; trustedTimestamp = [bool]$TimestampUrl; timestampUrl = $TimestampUrl
  }
  # Recipient entry points are bound to the exact post-signing package and public certificate.
  $installer = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'install-windows-distribution.ps1'))
  $installer = $installer.Replace('__CERTIFICATE_THUMBPRINT__',$cert.Thumbprint).
    Replace('__PACKAGE_FILE__',([IO.Path]::GetFileName($signed).Replace("'","''"))).
    Replace('__PACKAGE_SHA256__',$metadata.signedSha256)
  [IO.File]::WriteAllText((Join-Path $dist 'Install.ps1'),$installer,(New-Object Text.UTF8Encoding($true)))
  $batch = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'install-windows-distribution.bat')) -replace '\r?\n',"`r`n"
  [IO.File]::WriteAllText((Join-Path $dist 'Install.bat'),$batch,[Text.Encoding]::ASCII)
  Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'install-windows-distribution.txt') -Destination (Join-Path $dist 'README-Install.txt')
  $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dist 'build-metadata.json') -Encoding utf8
  Get-ChildItem -LiteralPath $dist -File | Sort-Object Name | ForEach-Object {
    "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)"
  } | Set-Content -LiteralPath (Join-Path $dist 'SHA256SUMS.txt') -Encoding ascii
  $zip = Join-Path $Output $archiveName
  Run-Native $SevenZip @('a','-tzip','-mx=0',$zip,"$dist\*")
  Run-Native $SevenZip @('t',$zip)
  $extract = Join-Path $Output 'zip-validation'
  Run-Native $SevenZip @('x',$zip,"-o$extract",'-y')
  $expected = @(Get-ChildItem -LiteralPath $dist -File | Sort-Object Name | ForEach-Object { $_.Name })
  $actual = @(Get-ChildItem -LiteralPath $extract -Recurse -File | Sort-Object Name | ForEach-Object { $_.Name })
  if (Compare-Object $expected $actual) { throw 'ZIP contents mismatch' }
  foreach ($name in $expected) {
    if ($name -match '\.(pfx|key|pem)$') { throw 'Private material forbidden' }
    if ((Get-FileHash (Join-Path $dist $name)).Hash -ne (Get-FileHash (Join-Path $extract $name)).Hash) { throw "ZIP checksum mismatch: $name" }
  }
  foreach ($line in Get-Content (Join-Path $extract 'SHA256SUMS.txt')) {
    if ($line -notmatch '^([0-9a-f]{64})  ([^\\/]+)$') { throw 'Invalid checksum entry' }
    if ((Get-FileHash (Join-Path $extract $Matches[2]) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Matches[1]) { throw 'Manifest checksum mismatch' }
  }
  $zipHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
  "$zipHash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath ($zip + '.sha256') -Encoding ascii
  $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'receipt.json') -Encoding utf8
  Get-Item -LiteralPath $signed,$zip | Select-Object FullName,Length
  Write-Output "ZIP SHA256 $zipHash; trusted timestamp: $([bool]$TimestampUrl)"
} finally {
  if ((Get-FileHash -LiteralPath $Msix -Algorithm SHA256).Hash.ToLowerInvariant() -ne $unsignedHash) { throw 'Unsigned artifact changed' }
  foreach ($dir in @((Join-Path $Output 'zip-validation'))) { if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force } }
}
