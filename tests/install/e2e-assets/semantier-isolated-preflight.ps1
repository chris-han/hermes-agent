# Read-only preflight for an explicitly owned interactive acceptance session.
# This script never registers, removes, launches or modifies a package.
# DIAGNOSTIC ONLY: self-supplied SID/marker and name-based path checks do not
# authorize a registration, update, process termination or cleanup.
param(
    [Parameter(Mandatory=$true)][string]$ExpectedSid,
    [Parameter(Mandatory=$true)][string]$OwnedRoot,
    [Parameter(Mandatory=$true)][string]$BaselineMsix,
    [Parameter(Mandatory=$true)][string]$BaselineSha256
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -cne 'Windows_NT') { throw 'Windows required' }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$sid = $identity.User.Value
$account = $identity.Name.Split('\')[-1]
$sessionId = (Get-Process -Id $PID).SessionId
# The normal user SID is refused even if the account is renamed.
$normalUserSid = 'S-1-5-21-1928370875-4287869425-2092521300-1003'
if ($account -ieq 'ChrisH' -or $sid -ceq $normalUserSid -or $sid -cne $ExpectedSid -or $sid -notmatch '^S-1-5-21-(\d+-){3}\d+$' -or
    -not [Environment]::UserInteractive -or $sessionId -le 0) { throw 'Unqualified disposable interactive Windows identity' }
# WTSConnectState=0 is Active. UserInteractive alone also admits disconnected sessions.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SemantierWtsReadOnly {
  [DllImport("wtsapi32.dll", SetLastError=true)] public static extern bool WTSQuerySessionInformation(IntPtr server, int session, int info, out IntPtr buffer, out int bytes);
  [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr buffer);
}
'@
$buffer = [IntPtr]::Zero; $bytes = 0
if (-not [SemantierWtsReadOnly]::WTSQuerySessionInformation([IntPtr]::Zero, $sessionId, 8, [ref]$buffer, [ref]$bytes)) { throw 'Cannot query Windows session state' }
try {
    if ($bytes -lt 4 -or [Runtime.InteropServices.Marshal]::ReadInt32($buffer) -ne 0) { throw 'Windows session is not active' }
} finally { if ($buffer -ne [IntPtr]::Zero) { [SemantierWtsReadOnly]::WTSFreeMemory($buffer) } }

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -cne 'AMD64') { throw 'Native x64 session required' }
$profile = [IO.Path]::GetFullPath([Environment]::GetFolderPath('UserProfile')).TrimEnd('\')
$root = [IO.Path]::GetFullPath($OwnedRoot).TrimEnd('\')
$parent = Join-Path $profile 'semantier-acceptance'
if (-not $root.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Owned root must be under disposable profile acceptance directory' }
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'Existing owned root required' }
# Refuse every reparse point on the traversed path. This keeps lexical and
# physical containment aligned without following junctions or symbolic links.
$current = [IO.DirectoryInfo]::new($root)
while ($current) {
    if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse point in owned root ancestry' }
    $current = $current.Parent
}
$marker = Join-Path $root 'owner.json'
if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { throw 'Owner marker required' }
$item = Get-Item -LiteralPath $marker -Force
if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse owner marker refused' }
$owner = (Get-Acl -LiteralPath $root).GetOwner([Security.Principal.SecurityIdentifier]).Value
$record = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
if ($owner -cne $sid -or $record.sid -cne $sid -or $record.purpose -cne 'semantier-registered-update-acceptance') { throw 'Owned root identity mismatch' }
$packages = @(Get-AppxPackage -Name 'Semantier.Agent.Acceptance')
if ($packages.Count) { throw 'Existing acceptance package conflicts with baseline' }
if (-not [IO.Path]::IsPathRooted($BaselineMsix) -or -not (Test-Path -LiteralPath $BaselineMsix -PathType Leaf)) { throw 'Absolute baseline package required' }
if ($BaselineSha256 -cne '55be24a18b1827cabfd951e30645a69e9cfcd0d1b07729e1b1c8f3483b079ec9') { throw 'Exact retained baseline SHA256 required' }
$hash = (Get-FileHash -LiteralPath $BaselineMsix -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -cne $BaselineSha256) { throw 'Baseline hash mismatch' }
$sig = Get-AuthenticodeSignature -LiteralPath $BaselineMsix
if ($sig.Status -ne 'Valid' -or -not $sig.SignerCertificate -or $sig.SignerCertificate.Subject -cne 'CN=Semantier Technology, O=Semantier Technology') { throw 'Baseline signature or publisher invalid' }
if ($sig.SignerCertificate.Thumbprint -cne '57BF15739B4AA447842EAF388856A01B4E5029FF') { throw 'Baseline signing certificate mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($BaselineMsix)
try {
    $entry = $archive.GetEntry('AppxManifest.xml')
    if (-not $entry) { throw 'Baseline MSIX manifest missing' }
    $settings = [Xml.XmlReaderSettings]::new()
    $settings.DtdProcessing = [Xml.DtdProcessing]::Prohibit
    $settings.XmlResolver = $null
    $stream = $entry.Open()
    try {
        $reader = [Xml.XmlReader]::Create($stream, $settings)
        try { $xml = [Xml.XmlDocument]::new(); $xml.XmlResolver = $null; $xml.Load($reader) } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
    $manager = [Xml.XmlNamespaceManager]::new($xml.NameTable)
    $manager.AddNamespace('a', 'http://schemas.microsoft.com/appx/manifest/foundation/windows10')
    $native = $xml.SelectSingleNode('/a:Package/a:Identity', $manager)
    if (-not $native -or $native.GetAttribute('Name') -cne 'Semantier.Agent.Acceptance' -or
        $native.GetAttribute('Publisher') -cne 'CN=Semantier Technology, O=Semantier Technology' -or
        $native.GetAttribute('Version') -cne '0.0.58.0' -or $native.GetAttribute('ProcessorArchitecture') -cne 'x64') {
        throw 'Baseline MSIX identity, version or architecture mismatch'
    }
} finally { $archive.Dispose() }
@{ result='PREFLIGHT_ONLY'; sid=$sid; sessionId=$sessionId; ownedRoot=$root; baselineSha256=$hash; signature=$sig.Status.ToString(); registeredPackageCount=$packages.Count } | ConvertTo-Json -Compress
