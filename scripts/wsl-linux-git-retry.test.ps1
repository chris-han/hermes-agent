$ErrorActionPreference='Stop'
$env:SEMANTIER_WSL_GIT_DISTRO='fixture'
. (Join-Path $PSScriptRoot 'wsl-linux-git.ps1')
$root=Join-Path ([IO.Path]::GetTempPath()) ('semantier-wsl-retry-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($root) | Out-Null
try {
    $fake=Join-Path $root 'fake.ps1'
    @'
$global:FakeAttempt++
if ($global:FakeAttempt -le $global:FakeFailures) {
    $global:LASTEXITCODE=$global:FakeExit
    Write-Output 'partial'
} else {
    $global:LASTEXITCODE=0
    Write-Output 'complete'
}
'@ | Set-Content -LiteralPath $fake -Encoding utf8
    $script:SemantierWslExe=$fake
    $global:FakeAttempt=0; $global:FakeFailures=2; $global:FakeExit=-1
    $result=@(Invoke-SemantierWslRead @('fixture'))
    if($LASTEXITCODE -ne 0 -or $global:FakeAttempt -ne 3 -or $result.Count -ne 1 -or $result[0] -cne 'complete'){throw 'Service retry or partial-output isolation failed'}
    $global:FakeAttempt=0; $global:FakeFailures=3; $global:FakeExit=-1
    $result=@(Invoke-SemantierWslRead @('fixture'))
    if($LASTEXITCODE -ne -1 -or $global:FakeAttempt -ne 3 -or $result.Count){throw 'Exhausted retries must fail closed'}
    $global:FakeAttempt=0; $global:FakeFailures=3; $global:FakeExit=1
    $result=@(Invoke-SemantierWslRead @('fixture'))
    if($LASTEXITCODE -ne 1 -or $global:FakeAttempt -ne 1 -or $result.Count){throw 'Non-service failures must not retry'}
    @'
if ($args -contains '/usr/bin/wslpath') { $global:LASTEXITCODE=0; Write-Output '/fixture'; return }
$global:FakeAttempt++
$global:LASTEXITCODE=-1
'@ | Set-Content -LiteralPath $fake -Encoding utf8
    $global:FakeAttempt=0
    git fetch fixture | Out-Null
    if($LASTEXITCODE -ne -1 -or $global:FakeAttempt -ne 1){throw 'Mutation was retried'}
    Write-Host 'WSL_READ_RETRY_PASS'
} finally { Remove-Item -LiteralPath $root -Recurse -Force }
