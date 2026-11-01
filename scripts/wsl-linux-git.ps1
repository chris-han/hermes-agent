<# Process-local bridge for the WSL controller. No Windows Git is invoked. #>
if ($env:SEMANTIER_WSL_GIT_DISTRO -cnotmatch '^[A-Za-z0-9_.-]{1,64}$') { throw 'Explicit WSL Git distribution required.' }
$script:SemantierWslExe=Join-Path $env:SystemRoot 'System32\wsl.exe'
if (!(Test-Path -LiteralPath $script:SemantierWslExe -PathType Leaf)) { throw 'WSL executable unavailable.' }
function ConvertTo-SemantierWslPath([string]$Path) {
    $result=@(& $script:SemantierWslExe -d $env:SEMANTIER_WSL_GIT_DISTRO --exec /usr/bin/wslpath -u $Path)
    if($LASTEXITCODE -ne 0 -or $result.Count -ne 1 -or !$result[0].StartsWith('/')){throw "WSL path conversion failed (exit $LASTEXITCODE, output lines $($result.Count), path $Path)."}
    return [string]$result[0]
}
function ConvertFrom-SemantierWslPath([string]$Path) {
    $result=@(& $script:SemantierWslExe -d $env:SEMANTIER_WSL_GIT_DISTRO --exec /usr/bin/wslpath -w $Path)
    if($LASTEXITCODE -ne 0 -or $result.Count -ne 1){throw 'Windows path conversion failed.'}
    return [string]$result[0]
}
function git {
    $converted=@()
    foreach($item in $args){
        $value=[string]$item
        if($value -match '^[A-Za-z]:\\' -or $value -match '^\\\\wsl(?:\.localhost|\$)\\'){$value=ConvertTo-SemantierWslPath $value}
        $converted+=$value
    }
    $cwd=ConvertTo-SemantierWslPath (Get-Location).ProviderPath
    & $script:SemantierWslExe -d $env:SEMANTIER_WSL_GIT_DISTRO --exec /usr/bin/env -u GIT_DIR -u GIT_WORK_TREE -u GIT_COMMON_DIR -u GIT_INDEX_FILE -u GIT_OBJECT_DIRECTORY -u GIT_ALTERNATE_OBJECT_DIRECTORIES -u GIT_CONFIG -u GIT_CONFIG_SYSTEM -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT -u GIT_NAMESPACE GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null /usr/bin/git -C $cwd @converted
    $global:LASTEXITCODE=$LASTEXITCODE
}
