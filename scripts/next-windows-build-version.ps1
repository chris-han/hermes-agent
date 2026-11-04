# Reserve one numeric Semantier version per full local build attempt.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$State, [Parameter(Mandatory=$true)][string]$SourceCommit, [string]$RetryVersion = '', [string]$ExpectedNextVersion = '', [string]$ReservationId = '', [string]$SourceBinding = '')
$ErrorActionPreference = 'Stop'
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Exact source commit required' }
if ($ExpectedNextVersion -and ($RetryVersion -or $ExpectedNextVersion -notmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')) { throw 'Expected next version requires a numeric version and no retry' }
if (($ReservationId -or $SourceBinding) -and ($ReservationId -cnotmatch '^[0-9a-f]{32}$' -or $SourceBinding -cnotmatch '^[0-9a-f]{64}$')) { throw 'Exact reservation and source binding required together' }
$State = [IO.Path]::GetFullPath($State)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($State)) | Out-Null
$lock = $null
$temp = $State + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
try {
  $lock = [IO.File]::Open($State+'.lock',[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  if ($ReservationId -and !(Test-Path -LiteralPath $State)) { throw 'Existing version ledger required for a bound reservation' }
  $version = '0.0.0'
  if (Test-Path -LiteralPath $State) {
    $previous = Get-Content -LiteralPath $State -Raw | ConvertFrom-Json
    if ($previous.kind -cne 'semantier-windows-build-version-v1') { throw 'Unknown version counter; preserve and inspect it' }
    $version = [string]$previous.version
  }
  if ($version -notmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$') { throw 'Invalid version counter' }
  $parts = @($version.Split('.') | ForEach-Object { [int]$_ })
  if (@($parts | Where-Object { $_ -gt 65535 }).Count) { throw 'Version counter exceeds Windows limits' }
  if ($RetryVersion) {
    if (!(Test-Path -LiteralPath $State) -or $RetryVersion -cne $version) { throw 'Retry requires the exact latest recorded version; counter is never reset or rewound' }
    if ($ReservationId -and ($previous.sourceCommit -cne $SourceCommit -or $previous.reservationId -cne $ReservationId -or $previous.sourceBinding -cne $SourceBinding)) { throw 'Retry does not match the original source-bound reservation' }
    Write-Output $version
    return
  }
  $parts[2]++
  for ($i=2; $i -gt 0; $i--) { if ($parts[$i] -gt 65535) { $parts[$i]=0; $parts[$i-1]++ } }
  if ($parts[0] -gt 65535) { throw 'Windows version space exhausted' }
  $next = $parts -join '.'
  if ($ExpectedNextVersion -and $next -cne $ExpectedNextVersion) { throw 'Next version differs from the frozen expected version; counter left unchanged' }
  $record = [ordered]@{kind='semantier-windows-build-version-v1';version=$next;sourceCommit=$SourceCommit;reservedAt=[DateTime]::UtcNow.ToString('o')}
  if ($ReservationId) { $record['reservationId']=$ReservationId; $record['sourceBinding']=$SourceBinding }
  $record | ConvertTo-Json | Set-Content -LiteralPath $temp -Encoding utf8
  Move-Item -LiteralPath $temp -Destination $State -Force
  Write-Output $next
} finally {
  if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
  if ($lock) { $lock.Dispose(); Remove-Item -LiteralPath ($State+'.lock') }
}
