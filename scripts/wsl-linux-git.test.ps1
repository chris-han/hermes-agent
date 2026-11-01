param([Parameter(Mandatory=$true)][string]$Repository,[Parameter(Mandatory=$true)][string]$Distro)
$ErrorActionPreference='Stop'
$env:SEMANTIER_WSL_GIT_DISTRO=$Distro
. (Join-Path $PSScriptRoot 'wsl-linux-git.ps1')
Set-Location -LiteralPath $Repository
$head=@(git rev-parse HEAD)
if($LASTEXITCODE -ne 0 -or $head.Count -ne 1 -or $head[0] -cnotmatch '^[0-9a-f]{40}$'){throw 'Linux Git HEAD failed'}
$top=@(git -C $Repository rev-parse --show-toplevel)
if($LASTEXITCODE -ne 0 -or $top.Count -ne 1 -or $top[0] -notmatch '^/mnt/'){throw 'Linux Git -C translation failed'}
$tool='git';$viaTool=@(& $tool -C $Repository rev-parse HEAD)
if($LASTEXITCODE -ne 0 -or $viaTool.Count -ne 1 -or $viaTool[0] -cne $head[0]){throw 'Controller tool dispatch failed'}
$roundTrip=ConvertFrom-SemantierWslPath $top[0]
if([IO.Path]::GetFullPath($roundTrip) -ine [IO.Path]::GetFullPath($Repository)){throw 'Native path conversion failed'}
$clean=@(git status --porcelain --untracked-files=all)
if($LASTEXITCODE -ne 0 -or $clean.Count){throw 'Clean source was not recognized'}
[IO.File]::AppendAllText((Join-Path $Repository 'tracked file.txt'),'changed')
$dirty=@(git status --porcelain --untracked-files=all)
if($LASTEXITCODE -ne 0 -or $dirty.Count -ne 1 -or $dirty[0] -notmatch 'tracked file.txt'){throw 'Dirty source was not rejected'}
Write-Output 'WSL_LINUX_GIT_BRIDGE_TEST_PASS'
