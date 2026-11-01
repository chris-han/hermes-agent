param([Parameter(Mandatory=$true)][string]$Repository,[Parameter(Mandatory=$true)][string]$Bundle,[Parameter(Mandatory=$true)][string]$Commit,[Parameter(Mandatory=$true)][string]$Ref,[Parameter(Mandatory=$true)][string]$Distro)
$ErrorActionPreference='Stop'
$env:SEMANTIER_WSL_GIT_DISTRO=$Distro
. (Join-Path $PSScriptRoot 'wsl-linux-git.ps1')
& git init $Repository
if($LASTEXITCODE -ne 0){throw 'Linux Git init failed'}
& git -C $Repository bundle verify $Bundle
if($LASTEXITCODE -ne 0){throw 'Linux Git bundle verification failed'}
& git -C $Repository fetch --no-write-fetch-head $Bundle $Ref
if($LASTEXITCODE -ne 0){throw 'Linux Git bundle fetch failed'}
& git -C $Repository checkout --detach $Commit
if($LASTEXITCODE -ne 0){throw 'Linux Git detached checkout failed'}
$head=@(git -C $Repository rev-parse HEAD)
if($LASTEXITCODE -ne 0 -or $head.Count -ne 1 -or $head[0] -cne $Commit){throw 'Transferred source identity mismatch'}
Write-Output 'WSL_LINUX_GIT_TRANSFER_TEST_PASS'
