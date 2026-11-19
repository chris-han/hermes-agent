param([string]$Root,[string]$OutputDirectory,[string]$Original)
$ErrorActionPreference='Stop'
# Cookie is piped in memory from the real WSL password login; never argv/disk/log.
$auth=[Console]::In.ReadToEnd() | ConvertFrom-Json
$source='using System;using System.IO;using System.ComponentModel;using System.Globalization;using System.Runtime.InteropServices;'+[IO.File]::ReadAllText((Join-Path $Root 'hermes-agent/win-native-installer/WinHttp.cs'))+[IO.File]::ReadAllText((Join-Path $Root 'hermes-agent/tests/scripts/oss-staging-live.cs'))
Add-Type -TypeDefinition $source
function Get-Grant([bool]$Expired) {
 $path=if($Expired){$auth.expiredPath}else{$auth.path}
 try{$r=Invoke-RestMethod -Uri ($auth.base+$path) -Headers @{Cookie=$auth.cookie} -UserAgent $auth.userAgent}
 catch{throw 'Runtime authorization request failed; signed details withheld'}
 $d=$r.download
 if($d.headers.'x-oss-range-behavior' -cne 'standard'){throw 'Required signed Range header missing'}
 return [OssStagingLive]::Grant($d.url,$d.key,$d.version,$d.sha256,[long]$d.size)
}
$normal=[Func[object]]{Get-Grant $false}
$expired=[Func[object]]{Get-Grant $true}
$full=[OssStagingLive]::Download($normal,$normal,(Join-Path $OutputDirectory 'native-full.msix'),$false)
$range=[OssStagingLive]::Range((Get-Grant $false),$Original)
$resume=[OssStagingLive]::Download($expired,$normal,(Join-Path $OutputDirectory 'native-resumed.msix'),$true)
if($resume.renewals -lt 1 -or !$resume.interrupted -or $resume.expiredOffset -le 0){throw 'Expiry renewal/interruption were not exercised'}
[ordered]@{produced_at=[DateTimeOffset]::UtcNow.ToString('o');full=$full;range=$range;resumed=$resume;status='PASS';installation='NOT_RUN'} | ConvertTo-Json -Depth 5
