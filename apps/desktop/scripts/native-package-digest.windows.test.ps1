param([Parameter(Mandatory=$true)][string]$Package)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-digest.ps1')
$first=Get-NativePackageDigest $Package
$second=Get-NativePackageDigest $Package
if (!$first -or $first -cne $second) { throw 'Native digest must be stable' }
if ($first -ceq (Get-FileHash -LiteralPath $Package -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Raw file hash substituted for native digest' }
try { Get-NativePackageDigest (Join-Path $PSScriptRoot 'native-package-digest.ps1'); throw 'Non-package accepted' } catch { if ($_.Exception.Message -eq 'Non-package accepted') { throw } }
[ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');package=$Package;nativeDigest=$first;repeat='PASS';nonPackageRefusal='PASS';deployment='NOT_RUN'} | ConvertTo-Json -Compress
