#Requires -Version 5.1
<#
.SYNOPSIS
  Verify an already built Semantier MSIX has a transparent App Installer logo,
  then optionally open its real Windows installation dialog for visual review.
.EXAMPLE
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-msix-installer-logo.ps1 -Msix 'D:\repo\deliverables\work\new-run\SemantierAgent.msix' -Launch
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Msix,
  [switch]$Launch
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing
$resolved = (Resolve-Path -LiteralPath $Msix -ErrorAction Stop).ProviderPath
if (-not $resolved.EndsWith('.msix', [StringComparison]::OrdinalIgnoreCase)) {
  throw 'Provide a .msix package, not an installer EXE or an unpacked directory.'
}
$archive = [IO.Compression.ZipFile]::OpenRead($resolved)
try {
  $manifestEntry = $archive.GetEntry('AppxManifest.xml')
  if ($null -eq $manifestEntry) { throw 'MSIX is missing AppxManifest.xml.' }
  $reader = New-Object IO.StreamReader($manifestEntry.Open())
  try { [xml]$manifest = $reader.ReadToEnd() } finally { $reader.Dispose() }
  $identity = $manifest.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Identity"]')
  $visual = $manifest.SelectSingleNode('/*[local-name()="Package"]/*[local-name()="Applications"]/*[local-name()="Application"][1]/*[local-name()="VisualElements"]')
  if ($null -eq $identity -or $null -eq $visual) { throw 'Missing main MSIX identity or visual elements.' }
  if ($visual.GetAttribute('BackgroundColor') -cne 'transparent') {
    throw ('FAIL: App Installer would render an opaque logo plate: ' + $visual.GetAttribute('BackgroundColor'))
  }
  $logo = $visual.GetAttribute('Square150x150Logo').Replace('\','/')
  if (-not $logo) { throw 'MSIX main application has no Square150x150Logo.' }
  $logoEntry = $null
  foreach ($entry in $archive.Entries) {
    if ($entry.FullName.Replace('\','/') -ceq $logo) { $logoEntry = $entry; break }
  }
  if ($null -eq $logoEntry) { throw ('Missing package logo: ' + $logo) }
  $logoStream = $logoEntry.Open()
  try {
    $bitmap = New-Object System.Drawing.Bitmap($logoStream)
    try {
      if ($bitmap.Width -le 0 -or $bitmap.Height -le 0) { throw 'Invalid logo dimensions.' }
      if ($bitmap.GetPixel(0,0).A -ne 0 -or $bitmap.GetPixel($bitmap.Width-1,$bitmap.Height-1).A -ne 0) {
        throw 'FAIL: the MSIX logo PNG has opaque corners.'
      }
    } finally { $bitmap.Dispose() }
  } finally { $logoStream.Dispose() }
  Write-Host ('PASS: Transparent MSIX background; alpha PNG logo; ' + $identity.GetAttribute('Name') + ' ' + $identity.GetAttribute('Version'))
} finally { $archive.Dispose() }
if ($Launch) {
  Start-Process -FilePath $resolved
  Write-Host 'Opened the Windows App Installer. Visually verify the top-right logo has no gray square.'
}
