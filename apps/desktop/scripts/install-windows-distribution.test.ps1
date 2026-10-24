# Native Windows behavioral tests. No machine trust or app registration is changed.
param([Parameter(Mandatory=$true)][string]$PublicCertificate)
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:TEMP ("Semantier installer's test " + [guid]::NewGuid())
New-Item -ItemType Directory -Path $root | Out-Null
try {
  $source = Join-Path $PSScriptRoot 'install-windows-distribution.ps1'
  $cert = Get-PfxCertificate -FilePath $PublicCertificate
  Copy-Item -LiteralPath $PublicCertificate -Destination (Join-Path $root 'SemantierAgent-signing.cer')
  $package = Join-Path $root 'test package.msix'
  Set-Content -LiteralPath $package -Value 'fixture bytes' -Encoding ascii
  $hash = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash
  $text = [IO.File]::ReadAllText($source).Replace('__CERTIFICATE_THUMBPRINT__',$cert.Thumbprint).Replace('__PACKAGE_FILE__','test package.msix').Replace('__PACKAGE_SHA256__',$hash)
  $installer = Join-Path $root 'Install.ps1'
  [IO.File]::WriteAllText($installer,$text)
  . $installer
  function Test-InstallerAdministrator { return $script:admin }
  function Test-Path {
    param([string]$LiteralPath,[string]$PathType='Any')
    if ($LiteralPath -like 'Cert:\LocalMachine\TrustedPeople\*') { return $script:trusted }
    return Microsoft.PowerShell.Management\Test-Path -LiteralPath $LiteralPath -PathType $PathType
  }
  function Get-AuthenticodeSignature { param($LiteralPath)
    $status=if ($script:trustResolvesSignature -and $script:trusted) { 'Valid' } else { $script:signatureStatus }
    return [pscustomobject]@{Status=$status;SignerCertificate=[pscustomobject]@{Thumbprint=$script:signer}}
  }
  function Import-Certificate { param($FilePath,$CertStoreLocation)
    if ($CertStoreLocation -ne 'Cert:\LocalMachine\TrustedPeople') { throw 'Wrong trust store' }
    $script:imports++; $script:trusted=$true
  }
  function Start-Process { param($FilePath,$ArgumentList,$Verb,[switch]$Wait,[switch]$PassThru)
    if ($Verb -ne 'RunAs' -or !$Wait -or !$PassThru -or $ArgumentList[-1] -ne '-TrustCertificate') { throw 'Wrong elevation command' }
    if ($ArgumentList[-2] -ne ('"{0}"' -f $installer)) { throw 'Installer path lost quoting' }
    if ($ArgumentList -contains 'Add-AppxPackage') { throw 'App installation elevated' }
    $script:elevations++
    if ($script:cancel) { throw 'UAC cancelled' }
    $script:trusted=$true
    return [pscustomobject]@{ExitCode=0}
  }
  function Add-AppxPackage { param($Path)
    if (!$script:trusted -or (!$script:trustResolvesSignature -and $script:signatureStatus -ne 'Valid') -or $Path -ne $package) { throw 'Unsafe installation' }
    $script:installs++
  }
  function Reset-Case {
    $script:trusted=$false; $script:admin=$false; $script:cancel=$false
    $script:imports=0; $script:elevations=0; $script:installs=0
    $script:signatureStatus='Valid'; $script:signer=$cert.Thumbprint; $script:trustResolvesSignature=$false
  }
  function Assert-Refused {
    $refused=$false
    try { Invoke-SemantierInstall } catch { $refused=$true }
    if (!$refused -or $script:installs) { throw 'Expected refusal before app install' }
  }
  Reset-Case; $script:trusted=$true
  Invoke-SemantierInstall
  if ($script:installs -ne 1 -or $script:imports -or $script:elevations) { throw 'Existing trust was not reused' }
  Reset-Case
  Invoke-SemantierInstall
  if ($script:installs -ne 1 -or $script:elevations -ne 1 -or $script:imports) { throw 'Trust-only elevation failed' }
  # Windows PowerShell 5.1 maps CERT_E_CHAINING (0x800B010A) to UnknownError.
  Reset-Case; $script:signatureStatus='UnknownError'; $script:trustResolvesSignature=$true
  Invoke-SemantierInstall
  if ($script:installs -ne 1 -or $script:elevations -ne 1) { throw 'Missing-chain trust repair failed' }
  Reset-Case; $script:signatureStatus='UnknownError'; Assert-Refused
  Reset-Case; $script:admin=$true
  Invoke-SemantierInstall -TrustCertificate
  if ($script:imports -ne 1 -or $script:installs) { throw 'Elevated phase installed app' }
  Reset-Case; $script:cancel=$true; Assert-Refused
  Reset-Case; $script:signer='wrong certificate'; Assert-Refused
  if ($script:elevations -or $script:imports) { throw 'Wrong signer was trusted' }
  Reset-Case; $script:signatureStatus='HashMismatch'; Assert-Refused
  Reset-Case
  Add-Content -LiteralPath $package -Value 'tampered'
  Assert-Refused
  if ($script:elevations -or $script:imports) { throw 'Tampered package was trusted' }
  Write-Output 'PASS: existing trust, trust-only elevation, correct install user, missing chain repair, unresolved error refusal, UAC cancellation, signer mismatch, invalid signature, tamper refusal'
} finally { Remove-Item -LiteralPath $root -Recurse -Force }
