<# Process-only build cache environment. No installation, trust or user profile changes. #>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Controller,
      [Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference='Stop'
$request=Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$work=[IO.Path]::GetFullPath($request.worktree).TrimEnd('\')
if($work -notmatch '^[A-Za-z]:\\' -or $work.Length -le 3){throw 'Native Windows worktree path required'}
if(!$request.unsignedReceipt){
if(Test-Path -LiteralPath ($work+'.distribution.lock')){throw 'Windows build is active; cache and environment left untouched'}
# A fresh checkout must remain absent until the controller claims it.
$diagnostics=if(Test-Path -LiteralPath $work){Join-Path $work '.cache\desktop-inputs\diagnostics'}else{$work+'.environment-diagnostics'}
[IO.Directory]::CreateDirectory($diagnostics) | Out-Null
try {
# Discover an installed toolchain; import only into this build process.
$bootstrapPhase='vswhere'
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if(!(Test-Path -LiteralPath $vswhere)){throw 'Installed Visual Studio Build Tools discovery (vswhere.exe) is required'}
$installation=@(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if($LASTEXITCODE -ne 0 -or $installation.Count -ne 1){throw 'No installed Visual Studio C++ Build Tools found'}
$bootstrapPhase='developer-command'
$devcmd=Join-Path $installation[0] 'Common7\Tools\VsDevCmd.bat'
if(!(Test-Path -LiteralPath $devcmd)){throw 'Installed VsDevCmd.bat is missing'}
$architecture=if($env:PROCESSOR_ARCHITECTURE -eq 'ARM64'){'arm64'}else{'x64'}
$bootstrapPhase='developer-environment'
$start=New-Object Diagnostics.ProcessStartInfo
$start.FileName=$env:ComSpec
$start.Arguments='/d /s /c "call "'+$devcmd+'" -no_logo -arch='+$architecture+' -host_arch='+$architecture+' >nul && set"'
$start.WorkingDirectory=$env:SystemRoot
$start.UseShellExecute=$false
$start.RedirectStandardOutput=$true
$start.RedirectStandardError=$true
# cmd.exe drops an overlong PATH during expansion; reserve space for the VS
# tools it adds. This affects only the explicitly selected WSL build child.
if (($request.PSObject.Properties.Name -contains 'wslLinuxGitDistro') -and $request.wslLinuxGitDistro) {
  $parts=New-Object 'System.Collections.Generic.List[string]'
  foreach($entry in @((Join-Path $env:SystemRoot 'System32'),$env:SystemRoot) + @($env:PATH -split ';')) {
    if (!$entry -or $parts.Contains($entry)) { continue }
    if (@('git.exe','git.cmd','git.bat') | Where-Object { Test-Path -LiteralPath (Join-Path $entry $_) -PathType Leaf }) { continue }
    if ((($parts -join ';').Length + $entry.Length + 1) -le 3500) { $parts.Add($entry) }
  }
  $start.EnvironmentVariables['PATH']=$parts -join ';'
}
$child=[Diagnostics.Process]::Start($start)
try {
  $environmentText=$child.StandardOutput.ReadToEnd()
  $environmentError=$child.StandardError.ReadToEnd()
  $child.WaitForExit()
  if($child.ExitCode -ne 0){throw 'Visual Studio developer environment initialization failed'}
  $environmentLines=$environmentText -split "`r?`n"
} finally {$child.Dispose()}
foreach($line in $environmentLines){
  if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process')}
}
$compiler=Get-Command cl.exe -ErrorAction Stop
if (($request.PSObject.Properties.Name -contains 'wslLinuxGitDistro') -and $request.wslLinuxGitDistro -and $env:PATH.Length -gt 6000) {
  throw 'VS build PATH leaves insufficient room for Windows npm lifecycle commands'
}
if (($request.PSObject.Properties.Name -contains 'wslLinuxGitDistro') -and $request.wslLinuxGitDistro -and
    @(Get-Command git.exe,git.cmd,git.bat -CommandType Application -All -ErrorAction SilentlyContinue).Count) {
  throw 'Windows Git is reachable from the Linux-Git build environment'
}
if($env:WindowsSDKVersion -notmatch '^\d+\.\d+\.\d+\.\d+\\$' -or !$env:VCToolsVersion -or !$env:UniversalCRTSdkDir -or !$env:UCRTVersion -or
   !(Test-Path -LiteralPath (Join-Path $env:UniversalCRTSdkDir ("Include\"+$env:UCRTVersion+"\ucrt\float.h")) -PathType Leaf)) {
  throw 'VS initialized without a complete Windows SDK/VC version identity'
}
# Allowlist only: never serialize the imported environment or credentials.
[ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');result='initialized';architecture=$architecture;
  compiler=$compiler.Source;windowsSdkVersion=$env:WindowsSDKVersion;vcToolsVersion=$env:VCToolsVersion;
  developerCommand=$devcmd} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $diagnostics 'windows-environment.jsonl') -Encoding utf8
Write-Host "Using installed Visual Studio C++ tools $($env:VCToolsVersion), SDK $($env:WindowsSDKVersion) ($architecture)"
} catch {
  [ordered]@{producedAt=[DateTime]::UtcNow.ToString('o');result='failed';missing=$bootstrapPhase;errorType=$_.Exception.GetType().Name} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $diagnostics 'windows-environment.jsonl') -Encoding utf8
  throw
}
}
$cacheOwner = $work
if (($request.PSObject.Properties.Name -contains 'reuseOwnedWindowsCache') -and $request.reuseOwnedWindowsCache) {
  if (!$request.wslLinuxGitDistro -or $request.repository -ine 'D:\repo\hermes-build' -or $work -ieq $request.repository) { throw 'Invalid shared cache binding' }
  $cacheOwner = $request.repository
}
$localCache=Join-Path $cacheOwner '.cache\desktop-inputs\home\AppData\Local'
# platformdirs prefers Known Folder APIs over LOCALAPPDATA; use its explicit override.
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA=$localCache
if(!$env:electron_config_cache){$env:electron_config_cache=Join-Path $cacheOwner '.cache\desktop-inputs\electron-downloads'}
$rustCache=Join-Path $localCache 'puccinialin\puccinialin\Cache'
$cargoRoot=Join-Path $rustCache 'cargo'
$rustRoot=Join-Path $rustCache 'rustup'
$settings=Join-Path $rustRoot 'settings.toml'
if((Test-Path -LiteralPath (Join-Path $cargoRoot 'bin\cargo.exe')) -and (Test-Path -LiteralPath $settings)){
  $content=Get-Content -LiteralPath $settings -Raw
  if($content -match '(?m)^default_toolchain\s*=\s*"([A-Za-z0-9_.-]+)"'){
    $selectedCompiler=Join-Path $rustRoot ("toolchains\"+$Matches[1]+"\bin\rustc.exe")
    if(Test-Path -LiteralPath $selectedCompiler){
      $env:CARGO_HOME=$cargoRoot
      $env:RUSTUP_HOME=$rustRoot
      $env:PATH=(Join-Path $cargoRoot 'bin')+';'+$env:PATH
      Write-Host 'Reusing existing private Windows Rust toolchain/cache'
    }
  }
}
$global:LASTEXITCODE=0
& $Controller -Config $Config
if($LASTEXITCODE -ne 0){throw "Windows distribution controller failed: $LASTEXITCODE"}
