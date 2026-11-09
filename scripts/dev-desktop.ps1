#Requires -Version 7.6
<#
.SYNOPSIS
Prepare cached Windows desktop dependencies and launch development mode.
.EXAMPLE
.\scripts\dev-desktop.ps1
.EXAMPLE
.\scripts\dev-desktop.ps1 -PrepareOnly
.EXAMPLE
.\scripts\dev-desktop.ps1 -Install
#>
[CmdletBinding()]
param(
  [string]$Repository = (Split-Path -Parent $PSScriptRoot),
  [string]$BuildCache = 'D:\repo\deliverables\cache\desktop-inputs',
  [switch]$PrepareOnly,
  [switch]$Install,
  [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (!$IsWindows -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') { throw 'Run on Windows x64.' }
$Repository = (Resolve-Path -LiteralPath $Repository).ProviderPath
$BuildCache = [IO.Path]::GetFullPath($BuildCache)
$utf8 = [Text.UTF8Encoding]::new($false)
$Commit = ''
function Report-Status($Message) { Write-Host ('[desktop-dev] ' + $Message) }
function Write-RunEvent($Event) { }
function Hash($Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Run([string]$Program, [string[]]$Arguments) {
  & $Program @Arguments | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Development launch stopped." }
}
function Initialize-ElectronPreflight {
  if($Commit){
    $json=& git.exe -C $Repository show ($Commit+':apps/desktop/package.json')
    if($LASTEXITCODE -ne 0){throw 'Cannot read Electron version at requested commit.'}
    $package=($json -join "`n") | ConvertFrom-Json
  } else {$package=Get-Content -LiteralPath (Join-Path $Repository 'apps\desktop\package.json') -Raw | ConvertFrom-Json}
  $version=[string]$package.devDependencies.electron
  if($version -cnotmatch '^\d+\.\d+\.\d+$'){throw 'Electron version must be exactly pinned.'}
  $name="electron-v$version-win32-x64.zip"
  $base="https://github.com/electron/electron/releases/download/v$version"
  $cache=Join-Path $BuildCache 'packager\electron'
  New-Item -ItemType Directory -Path $cache -Force | Out-Null
  $zip=Join-Path $cache $name
  $sums=Join-Path $cache "SHASUMS256-v$version.txt"
  $pattern='(?m)^([a-fA-F0-9]{64})[ \t]+\*?'+[Regex]::Escape($name)+'\r?$'
  $legacy=Join-Path $cache 'SHASUMS256.txt'
  if(!(Test-Path -LiteralPath $sums) -and (Test-Path -LiteralPath $legacy)){
    if([Regex]::IsMatch([IO.File]::ReadAllText($legacy),$pattern)){Copy-Item -LiteralPath $legacy -Destination $sums}
  }
  if(!(Test-Path -LiteralPath $sums)){
    Report-Status "Preflight: downloading official Electron $version checksum list."
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','180',
      '--output',($sums+'.download'),"$base/SHASUMS256.txt")
    if(![Regex]::IsMatch([IO.File]::ReadAllText($sums+'.download'),$pattern)){throw 'Official Electron checksum entry missing.'}
    Move-Item -LiteralPath ($sums+'.download') -Destination $sums -Force
  }
  $match=[Regex]::Match([IO.File]::ReadAllText($sums),$pattern)
  if(!$match.Success){throw 'Cached Electron checksum list does not match requested version.'}
  $expected=$match.Groups[1].Value.ToLowerInvariant()
  if(!(Test-Path -LiteralPath $zip) -or (Hash $zip) -cne $expected){
    Report-Status "Preflight: downloading Electron $version; absent or invalid ZIP cache."
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','900',
      '--output',($zip+'.download'),"$base/$name")
    if((Hash ($zip+'.download')) -cne $expected){throw 'Electron archive failed official SHA-256 verification.'}
    Move-Item -LiteralPath ($zip+'.download') -Destination $zip -Force
  }
  $preload=Join-Path $cache "verified-electron-cache-v$version.mjs"
  $source=@'
import fs from 'node:fs';
import {Readable} from 'node:stream';
const files=new Map([[__SUM_URL__,__SUM_FILE__],[__ZIP_URL__,__ZIP_FILE__]]);
const original=globalThis.fetch;
globalThis.fetch=async(input,options)=>{
 const url=input instanceof Request?input.url:String(input),file=files.get(url);
 if(!file)return original(input,options);
 const method=options?.method||(input instanceof Request?input.method:'GET');
 if(method!=='GET'&&method!=='HEAD')return original(input,options);
 console.error('[VERIFIED LOCAL CACHE]',file);
 return new Response(method==='HEAD'?null:Readable.toWeb(fs.createReadStream(file)),{
  status:200,headers:{'content-length':String(fs.statSync(file).size),'content-type':file.endsWith('.zip')?'application/zip':'text/plain'}
 });
};
'@
  foreach($entry in @(@('__SUM_URL__',"$base/SHASUMS256.txt"),@('__SUM_FILE__',$sums),@('__ZIP_URL__',"$base/$name"),@('__ZIP_FILE__',$zip))){
    $source=$source.Replace($entry[0],(ConvertTo-Json -InputObject $entry[1] -Compress))
  }
  [IO.File]::WriteAllText($preload,$source,$utf8)
  Run $Node @('--check',$preload)
  $option='--import='+([Uri]$preload).AbsoluteUri
  $existing=[Environment]::GetEnvironmentVariable('NODE_OPTIONS','Process')
  if(!$existing -or !$existing.Contains($option)){$env:NODE_OPTIONS=($existing+' '+$option).Trim()}
  Write-RunEvent @{type='preflight-electron';at=[DateTime]::UtcNow.ToString('o');version=$version;sha256=$expected;status='PASS'}
  Report-Status "Preflight: Electron $version official SHA-256 verified; Node will reuse the local archive and checksums."
}
function Initialize-NativeCache {
  $cacheRoot='D:\repo\deliverables\cache\node-pre-gyp\get-windows'
  $archive=Join-Path $cacheRoot 'v9.3.0\napi-9-win32-unknown-x64.tar.gz'
  New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force | Out-Null
  $valid=$false
  if((Test-Path -LiteralPath $archive) -and (Test-Path -LiteralPath ($archive+'.sha256'))){
    $valid=(Hash $archive) -ieq ([IO.File]::ReadAllText($archive+'.sha256')).Trim()
  }
  if(!$valid){
    Report-Status 'Preflight: downloading official get-windows 9.3.0 Windows x64 N-API 9 archive.'
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','180',
      '--output',($archive+'.download'),'https://github.com/sindresorhus/get-windows/releases/download/v9.3.0/napi-9-win32-unknown-x64.tar.gz')
    $entries=& tar.exe -tzf ($archive+'.download')
    if($LASTEXITCODE -ne 0 -or !($entries | Where-Object {$_ -match 'node-get-windows\.node$'})){throw 'Invalid official prebuilt archive.'}
    Move-Item -LiteralPath ($archive+'.download') -Destination $archive -Force
    [IO.File]::WriteAllText(($archive+'.sha256'),(Hash $archive),[Text.Encoding]::ASCII)
  }
  # node-pre-gyp 1.0.11 strips file:// instead of decoding a Windows file URL.
  # Its old url.resolve implementation therefore needs this readable local UNC.
  $mirror='file:////localhost/D$/repo/deliverables/cache/node-pre-gyp/get-windows/'
  $code=@'
const fs=require('node:fs'),url=require('node:url');
const path=url.resolve(url.resolve(process.argv[2],'v9.3.0/'),'napi-9-win32-unknown-x64.tar.gz').slice(7);
const fd=fs.openSync(path,'r');fs.closeSync(fd);console.log('Prebuilt cache readable through node-pre-gyp file path');
'@
  $code | & $Node - $mirror | Out-Host
  if($LASTEXITCODE -ne 0){throw 'Local prebuilt cache is not readable through localhost D$; no permissions were changed.'}
  [Environment]::SetEnvironmentVariable('npm_config_node_get-windows_binary_host_mirror',$mirror,'Process')

}
$names = @('PATH','NODE_OPTIONS','npm_config_node_get-windows_binary_host_mirror','ELECTRON_RUN_AS_NODE','HERMES_DESKTOP_HERMES_ROOT','HERMES_DESKTOP_PYTHON')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
Push-Location -LiteralPath $Repository
try {
  $lock = Get-Content pm/lock.json -Raw | ConvertFrom-Json
  $Node = Join-Path $BuildCache ('tools\node-' + $lock.packages.node.version + '-win32-x64\node.exe')
  $npm = Join-Path $BuildCache ('tools\npm-' + $lock.packages.npm.version + '-win32-x64\node_modules\npm\bin\npm-cli.js')
  foreach ($tool in @($Node,$npm)) {
    if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Provisioned tool missing: $tool" }
  }
  $env:PATH = (Split-Path -Parent $Node) + ';' + $env:PATH
  Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
  Run $Node @('--version')
  Run $Node @($npm,'--version')
  if ($Install -and !$SkipInstall) {
    Initialize-NativeCache
    Initialize-ElectronPreflight
    Report-Status 'Installing locked dependencies. Do not run another install in this checkout.'
    Run $Node @($npm,'ci','--no-audit','--no-fund')
  }
  # Resolve from the consumer workspace: npm may nest dependencies there.
  Run $Node @('-e',"const {createRequire}=require('node:module'); const {resolve}=require('node:path'); const desktopRequire=createRequire(resolve('apps/desktop/package.json')); for(const name of ['concurrently','electron','get-windows']) desktopRequire.resolve(name);")
  if ($PrepareOnly) { Report-Status 'Dependencies prepared.'; return }
  $bootstrapPython = Join-Path $BuildCache ('tools\python-' + $lock.packages.python.version + '-win32-x64\python.exe')
  if (!$env:HERMES_DESKTOP_HERMES_ROOT) { $env:HERMES_DESKTOP_HERMES_ROOT = $Repository }
  if (!$env:HERMES_DESKTOP_PYTHON) {
    if (!(Test-Path -LiteralPath $bootstrapPython)) { throw "Bootstrap Python missing: $bootstrapPython" }
    $selectedPython = & $bootstrapPython -c 'from pathlib import Path; from pm.environments import project_python; import sys; p=project_python(Path(sys.argv[1])); print(p); sys.exit(0 if p.is_file() else 1)' $env:HERMES_DESKTOP_HERMES_ROOT
    if ($LASTEXITCODE -ne 0) { throw 'Local backend environment is not prepared. Run .\setup-hermes.ps1 -RuntimeOnly, then retry.' }
    $env:HERMES_DESKTOP_PYTHON = [string]($selectedPython | Select-Object -Last 1)
  }
  if (!(Test-Path -LiteralPath $env:HERMES_DESKTOP_PYTHON -PathType Leaf)) { throw 'Selected backend Python does not exist.' }
  Report-Status ('Backend source: ' + $env:HERMES_DESKTOP_HERMES_ROOT)
  Report-Status ('Backend Python: ' + $env:HERMES_DESKTOP_PYTHON)
  Report-Status 'Starting desktop development. Press Ctrl+C to stop.'
  Run $Node @($npm,'run','dev','--workspace','apps/desktop')
} finally {
  Pop-Location
  foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$saved[$name],'Process') }
}
