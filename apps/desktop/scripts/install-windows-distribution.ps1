# Recipient installer. The release builder pins these values to this signed package.
[CmdletBinding()]
param([switch]$TrustCertificate)
$ErrorActionPreference = 'Stop'

function Test-InstallerAdministrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  return (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-SemantierInstall([switch]$TrustCertificate) {
  $expectedThumbprint = '__CERTIFICATE_THUMBPRINT__'
  $package = Join-Path $PSScriptRoot '__PACKAGE_FILE__'
  $expectedHash = '__PACKAGE_SHA256__'
  $certificate = Join-Path $PSScriptRoot 'SemantierAgent-signing.cer'
  $cert = Get-PfxCertificate -FilePath $certificate
  if ($cert.Thumbprint -ne $expectedThumbprint) { throw 'The certificate does not match this release.' }
  $now = Get-Date
  if ($cert.NotBefore -gt $now -or $cert.NotAfter -le $now) { throw 'The signing certificate is not currently valid.' }
  $trustedPath = "Cert:\LocalMachine\TrustedPeople\$expectedThumbprint"

  if ($TrustCertificate) {
    if (!(Test-InstallerAdministrator)) { throw 'Administrator permission is required to trust the publisher certificate.' }
    if (!(Test-Path -LiteralPath $trustedPath)) {
      Import-Certificate -FilePath $certificate -CertStoreLocation 'Cert:\LocalMachine\TrustedPeople' | Out-Null
    }
    return
  }

  Write-Host 'Verifying the Semantier Agent package...'
  if ((Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Package checksum mismatch. Extract the complete original ZIP again.'
  }
  # Windows PowerShell 5.1 reports missing trust chains as UnknownError.
  # The pinned file hash and signer identify the package before trust.
  # Add-AppxPackage applies Windows' AppX signature/trust policy during install.
  $signature = Get-AuthenticodeSignature -LiteralPath $package
  if (!$signature.SignerCertificate -or $signature.SignerCertificate.Thumbprint -ne $expectedThumbprint -or
      $signature.Status -notin @('Valid','NotTrusted','UnknownError')) { throw 'Package signature does not match this release.' }

  if (!(Test-Path -LiteralPath $trustedPath)) {
    Write-Host "Windows will request administrator permission to trust this publisher: $($cert.Subject)"
    Write-Host "Certificate thumbprint: $expectedThumbprint"
    # Elevate only certificate trust. Add-AppxPackage stays in the original user's process,
    # including when UAC requires credentials for a different administrator account.
    $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $child = Start-Process -FilePath $powershell -ArgumentList @(
      '-NoProfile','-ExecutionPolicy','Bypass','-File',('"{0}"' -f $PSCommandPath),'-TrustCertificate'
    ) -Verb RunAs -Wait -PassThru
    if ($child.ExitCode -ne 0 -or !(Test-Path -LiteralPath $trustedPath)) {
      throw 'Publisher certificate was not installed. Allow the certificate trust request and retry Install.bat.'
    }
  }

  # AppX validates the signature and TrustedPeople policy itself. Generic
  # Authenticode CA-chain status is not an additional MSIX installation gate.
  Write-Host 'Installing Semantier Agent for the current Windows user...'
  Add-AppxPackage -Path $package
  Write-Host 'Semantier Agent installed. Open it from the Windows Start menu.'
}

# Dot-sourcing supports native behavioral tests without changing machine state.
if ($MyInvocation.InvocationName -ne '.') {
  try { Invoke-SemantierInstall -TrustCertificate:$TrustCertificate; exit 0 }
  catch { Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }
}
