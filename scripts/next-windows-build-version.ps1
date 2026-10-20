# Reserve one numeric Semantier version per full local build attempt.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$State, [Parameter(Mandatory=$true)][string]$SourceCommit)
$ErrorActionPreference = 'Stop'
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Exact source commit required' }
$State = [IO.Path]::GetFullPath($State)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($State)) | Out-Null
$lock = $null
$temp = $State + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
try {
  $lock = [IO.File]::Open($State+'.lock',[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  $version = '0.0.0'
  if (Test-Path -LiteralPath $State) {
    $previous = Get-Content -LiteralPath $State -Raw | ConvertFrom-Json
    if ($previous.kind -cne 'semantier-windows-build-version-v1') { throw 'Unknown version counter; preserve and inspect it' }
    $version = [string]$previous.version
  }
  if ($version -notmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$') { throw 'Invalid version counter' }
  $parts = @($version.Split('.') | ForEach-Object { [int]$_ })
  if (@($parts | Where-Object { $_ -gt 65535 }).Count) { throw 'Version counter exceeds Windows limits' }
  $parts[2]++
  for ($i=2; $i -gt 0; $i--) { if ($parts[$i] -gt 65535) { $parts[$i]=0; $parts[$i-1]++ } }
  if ($parts[0] -gt 65535) { throw 'Windows version space exhausted' }
  $next = $parts -join '.'
  $record = [ordered]@{kind='semantier-windows-build-version-v1';version=$next;sourceCommit=$SourceCommit;reservedAt=[DateTime]::UtcNow.ToString('o')}
  $record | ConvertTo-Json | Set-Content -LiteralPath $temp -Encoding utf8
  Move-Item -LiteralPath $temp -Destination $State -Force
  Write-Output $next
} finally {
  if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
  if ($lock) { $lock.Dispose(); Remove-Item -LiteralPath ($State+'.lock') }
}
