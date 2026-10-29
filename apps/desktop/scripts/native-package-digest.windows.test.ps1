param([Parameter(Mandatory=$true)][string]$Package)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-digest.ps1')
$first=Get-NativePackageDigest $Package
$second=Get-NativePackageDigest $Package
if (!$first -or $first -cne $second) { throw 'Native digest must be stable' }
if ($first -ceq (Get-FileHash -LiteralPath $Package -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Raw file hash substituted for native digest' }
try { Get-NativePackageDigest (Join-Path $PSScriptRoot 'native-package-digest.ps1'); throw 'Non-package accepted' } catch { if ($_.Exception.Message -eq 'Non-package accepted') { throw } }
[ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');package=$Package;nativeDigest=$first;repeat='PASS';nonPackageRefusal='PASS';deployment='NOT_RUN'} | ConvertTo-Json -Compress

$null=[Windows.Management.Deployment.AddPackageOptions,Windows.Management.Deployment,ContentType=WindowsRuntime]
$options=New-Object Windows.Management.Deployment.AddPackageOptions
$uri=[uri]'https://example.invalid/releases/channel-builds/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/main.msixbundle'
if ((Set-NativeExpectedDigests $options @($uri.AbsoluteUri) @($first)) -ne 1) { throw 'Native URI digest map failed' }
try { Set-NativeExpectedDigests $options @($uri.AbsoluteUri) @($first); throw 'Duplicate URI accepted' } catch { if ($_.Exception.Message -eq 'Duplicate URI accepted') { throw } }
try { Set-NativeExpectedDigests (New-Object Windows.Management.Deployment.AddPackageOptions) @('https://example.invalid/other.msix') @(('a'*64)); throw 'Raw SHA accepted' } catch { if ($_.Exception.Message -eq 'Raw SHA accepted') { throw } }
Write-Output 'PASS: actual WinRT URI digest map round-trip, duplicate URI and raw SHA refusal; no deployment or HTTP request'
try { Set-NativeExpectedDigests (New-Object Windows.Management.Deployment.AddPackageOptions) @($uri.AbsoluteUri,$uri.AbsoluteUri) @($first,$first); throw 'Repeated URI set accepted' } catch { if ($_.Exception.Message -eq 'Repeated URI set accepted') { throw } }

$multi=New-Object Windows.Management.Deployment.AddPackageOptions
if ((Set-NativeExpectedDigests $multi @($uri.AbsoluteUri,'https://example.invalid/runtime0001.msix') @($first,$first)) -ne 2) { throw 'Complete multi-object digest map missing' }
