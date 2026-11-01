<# Process-local bridge for the WSL controller. No Windows Git is invoked. #>
if ($env:SEMANTIER_WSL_GIT_DISTRO -cnotmatch '^[A-Za-z0-9_.-]{1,64}$') { throw 'Explicit WSL Git distribution required.' }
$script:SemantierWslExe=Join-Path $env:SystemRoot 'System32\wsl.exe'
if (!(Test-Path -LiteralPath $script:SemantierWslExe -PathType Leaf)) { throw 'WSL executable unavailable.' }
function Invoke-SemantierWslRead([string[]]$Arguments) {
    # A failed service connection may emit partial output. Never expose it as
    # a successful read, and never retry a non-service failure.
    $saved=$ErrorActionPreference
    try {
        $ErrorActionPreference='Continue'
        for($attempt=0;$attempt -lt 3;$attempt++) {
            $result=@(& $script:SemantierWslExe @Arguments 2>&1)
            $code=$LASTEXITCODE
            if($code -eq 0) {
                $global:LASTEXITCODE=0
                return $result | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }
            }
            if($code -ne -1 -or $attempt -eq 2) { $global:LASTEXITCODE=$code; return }
            Start-Sleep -Milliseconds 500
        }
    } finally { $ErrorActionPreference=$saved }
}
function ConvertTo-SemantierWslPath([string]$Path) {
    $result=@(Invoke-SemantierWslRead @('-d',$env:SEMANTIER_WSL_GIT_DISTRO,'--exec','/usr/bin/wslpath','-u',$Path))
    if($LASTEXITCODE -ne 0 -or $result.Count -ne 1 -or !$result[0].StartsWith('/')){throw "WSL path conversion failed (exit $LASTEXITCODE, output lines $($result.Count), path $Path)."}
    return [string]$result[0]
}
function ConvertFrom-SemantierWslPath([string]$Path) {
    $result=@(Invoke-SemantierWslRead @('-d',$env:SEMANTIER_WSL_GIT_DISTRO,'--exec','/usr/bin/wslpath','-w',$Path))
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
    $command=@('-d',$env:SEMANTIER_WSL_GIT_DISTRO,'--exec','/usr/bin/env','-u','GIT_DIR','-u','GIT_WORK_TREE','-u','GIT_COMMON_DIR','-u','GIT_INDEX_FILE','-u','GIT_OBJECT_DIRECTORY','-u','GIT_ALTERNATE_OBJECT_DIRECTORIES','-u','GIT_CONFIG','-u','GIT_CONFIG_SYSTEM','-u','GIT_CONFIG_PARAMETERS','-u','GIT_CONFIG_COUNT','-u','GIT_NAMESPACE','GIT_CONFIG_NOSYSTEM=1','GIT_CONFIG_GLOBAL=/dev/null','/usr/bin/git','-C',$cwd) + $converted
    $operation=if($converted.Count -ge 3 -and $converted[0] -eq '-C'){$converted[2]}elseif($converted.Count){$converted[0]}else{''}
    $operationIndex=if($converted.Count -ge 3 -and $converted[0] -eq '-C'){2}else{0}
    $bundleVerify=$operation -eq 'bundle' -and $converted.Count -gt ($operationIndex+1) -and $converted[$operationIndex+1] -eq 'verify'
    if($operation -in @('status','rev-parse') -or $bundleVerify) { Invoke-SemantierWslRead $command }
    else { & $script:SemantierWslExe @command }
    $global:LASTEXITCODE=$LASTEXITCODE
}
