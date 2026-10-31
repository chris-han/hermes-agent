<#
.SYNOPSIS
  Report the Semantier Agent Windows version from both authoritative sources:
  the installed MSIX package and the local build-attempt counter.

.DESCRIPTION
  These two numbers answer different questions and can legitimately differ:
    - Installed version: what is actually on this machine (authoritative for
      "what am I running"). Read from the AppX package identity.
    - Build counter: the last version reserved by a build ATTEMPT via
      scripts/next-windows-build-version.ps1. Monotonic, never rewound, and
      says nothing about whether that build succeeded or was installed.

.PARAMETER PackageName
  AppX package identity name. Defaults to the out-of-store acceptance package.

.PARAMETER CounterPath
  Path to semantier-agent-build-version.json. Defaults to the sibling of this
  repository root, matching scripts/build-windows-distribution.ps1.

.EXAMPLE
  pwsh -File scripts/show-semantier-version.ps1
#>
[CmdletBinding()]
param(
  [string]$PackageName = 'Semantier.Agent.Acceptance',
  [string]$CounterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'semantier-agent-build-version.json')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$installed = $null
if (Get-Command Get-AppxPackage -ErrorAction SilentlyContinue) {
  $pkg = Get-AppxPackage -Name $PackageName -ErrorAction SilentlyContinue
  if ($pkg) { $installed = [string]$pkg.Version }
}

$reserved = $null
$reservedAt = $null
$counterError = $null
if (Test-Path -LiteralPath $CounterPath) {
  try {
    $state = Get-Content -LiteralPath $CounterPath -Raw | ConvertFrom-Json
    if ($state.kind -cne 'semantier-windows-build-version-v1') {
      $counterError = "unrecognized counter kind: $($state.kind)"
    } else {
      $reserved = [string]$state.version
      if ($state.PSObject.Properties.Name -contains 'reservedAt') { $reservedAt = [string]$state.reservedAt }
    }
  } catch {
    $counterError = $_.Exception.Message
  }
}

[PSCustomObject]@{
  InstalledVersion = if ($installed) { $installed } else { '(not installed)' }
  PackageName      = $PackageName
  ReservedBuild    = if ($reserved) { $reserved } elseif ($counterError) { "(error: $counterError)" } else { '(counter not found)' }
  ReservedAt       = if ($reservedAt) { $reservedAt } else { '' }
  CounterPath      = $CounterPath
} | Format-List
