param([Parameter(Mandatory=$true)][string]$Script,[Parameter(Mandatory=$true)][string]$Repository,[Parameter(Mandatory=$true)][string]$Distro)
$env:SEMANTIER_WSL_GIT_DISTRO=$Distro
& node.exe $Script $Repository
if($LASTEXITCODE -ne 0){throw 'Linux Git Node stamp test failed.'}
