# Bounded Windows fixture: tests handle sharing only, not CI recovery or authority.
param([string]$FixtureParent = [IO.Path]::GetTempPath())
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class CIDirectoryHandleTest {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 public static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
}
'@
$root = Join-Path $FixtureParent ('semantier-ci-handle-test-' + [Guid]::NewGuid().ToString('N'))
$parent = Join-Path $root 'parent'
$child = Join-Path $parent 'custody'
New-Item -ItemType Directory -Path $child -Force | Out-Null
$handles = @()
try {
 . (Join-Path $PSScriptRoot 'windows-physical-path.ps1')
 $ownedPhysical = Join-Path $root 'physical-helper'
 New-Item -ItemType Directory -Path $ownedPhysical | Out-Null
 $physical = Physical $ownedPhysical
 try {
  $blocked=$false
  try {[IO.Directory]::Move($ownedPhysical,$ownedPhysical+'-moved')} catch [IO.IOException] {
   if(($_.Exception.GetBaseException().HResult -band 0xffff) -ne 32){throw}
   $blocked=$true
  }
  if(!$blocked){throw 'Shared physical path helper did not prevent replacement'}
 } finally {$physical.handle.Dispose()}
 [IO.Directory]::Move($ownedPhysical,$ownedPhysical+'-moved')
 $metadataOnly = Join-Path $root 'metadata-only'
 New-Item -ItemType Directory -Path $metadataOnly | Out-Null
 $probe = [CIDirectoryHandleTest]::CreateFile($metadataOnly,0,3,[IntPtr]::Zero,3,0x02200000,[IntPtr]::Zero)
 if ($probe.IsInvalid) { throw 'Metadata-only control could not open directory' }
 try {
  [IO.Directory]::Move($metadataOnly, $metadataOnly + '-moved')
 } finally { $probe.Dispose() }
 if (!(Test-Path -LiteralPath ($metadataOnly + '-moved') -PathType Container)) { throw 'Metadata-only rename control did not move' }
 foreach ($path in @($parent,$child)) {
  # Share read/write but not delete; BACKUP_SEMANTICS opens directories and
  # OPEN_REPARSE_POINT prevents following the final path component.
  $handle = [CIDirectoryHandleTest]::CreateFile($path,[uint32]2147483648,3,[IntPtr]::Zero,3,0x02200000,[IntPtr]::Zero)
  if ($handle.IsInvalid) { throw "CreateFile failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
  $handles += $handle
 }
 $fixture = Join-Path $child 'fixture.txt'
 Set-Content -LiteralPath $fixture -Value 'owned fixture'
 Remove-Item -LiteralPath $fixture
 if (Test-Path -LiteralPath $fixture) { throw 'Fixture file was not removed while directory was held' }
 foreach ($path in @($child,$parent)) {
  $blocked = $false
  try { [IO.Directory]::Move($path, $path + '-moved') } catch [IO.IOException] {
   $failure = $_.Exception.GetBaseException()
   if (($failure.HResult -band 0xffff) -ne 32) { throw }
   $blocked = $true
  }
  if (!$blocked) { throw "Directory rename was not refused while held: $path" }
 }
 foreach ($handle in $handles) { $handle.Dispose() }
 $handles = @()
 [IO.Directory]::Move($child, $child + '-moved')
 if (!(Test-Path -LiteralPath ($child + '-moved') -PathType Container)) { throw 'Rename did not succeed after handle release' }
 [IO.Directory]::Move($parent, $parent + '-moved')
 if (!(Test-Path -LiteralPath ($parent + '-moved') -PathType Container)) { throw 'Parent rename did not succeed after handle release' }
 Write-Output 'PASS: metadata-only control moved; held directories resisted rename; a contained fixture file remained removable'
} finally {
 foreach ($handle in $handles) { $handle.Dispose() }
 if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
