"""Windows bundle entry passes selected job-local output paths to the real driver CLI."""
import base64
from pathlib import Path
import subprocess

import pytest

HOST = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')


@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
def test_bundle_entry_passes_job_paths():
    script = Path(__file__).resolve().parents[2] / 'scripts/build-bundle.ps1'
    native = subprocess.check_output(['wslpath', '-w', str(script)], text=True).strip().replace("'", "''")
    command = r'''
$ErrorActionPreference='Stop'
$root=Join-Path $env:TEMP ('bundle job paths-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $root | Out-Null
try {
 $scripts=Join-Path $root 'scripts';New-Item -ItemType Directory $scripts | Out-Null
 Copy-Item -LiteralPath 'SCRIPT' -Destination (Join-Path $scripts 'build-bundle.ps1')
 $py=Join-Path $root 'python.cmd';$env:OUTPUT_LOG=Join-Path $root 'arguments.txt'
 [IO.File]::WriteAllText($py, "@echo %* > `"%OUTPUT_LOG%`"`r`n@exit /b 0`r`n")
 function global:git { $global:LASTEXITCODE=0; if ($args[0] -eq 'rev-parse') { return ('a'*40) } }
 $work=Join-Path $root 'job';$cache=Join-Path $root 'cache';$output=Join-Path $root 'packages'
 New-Item -ItemType Directory $output | Out-Null
 & (Join-Path $scripts 'build-bundle.ps1') -Python $py -Work $work -Cache $cache -PackageOutput $output -ExternalOutputs
 if ($LASTEXITCODE -ne 0) { throw 'bundle entry failed' }
 $argsText=[IO.File]::ReadAllText($env:OUTPUT_LOG)
 foreach($expected in @('--work', $work, '--cache', $cache, '--external-outputs', ('-c.directories.output='+$output))) {
  if (!$argsText.Contains($expected)) { throw "Missing forwarded argument: $expected" }
 }
 Write-Host 'JOB_PATHS_PASS'
} finally {Set-Location $env:TEMP;Remove-Item Function:global:git -ErrorAction SilentlyContinue;Remove-Item Env:OUTPUT_LOG -ErrorAction SilentlyContinue;Remove-Item -LiteralPath $root -Recurse -Force}
'''.replace('SCRIPT', native)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'JOB_PATHS_PASS' in result.stdout


@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
def test_native_controller_rejects_unowned_job_paths_before_writes():
    script = Path(__file__).resolve().parents[2] / 'scripts/build-windows-distribution.ps1'
    native = subprocess.check_output(['wslpath', '-w', str(script)], text=True).strip().replace("'", "''")
    command = r'''
$ErrorActionPreference='Stop'
$root=Join-Path $env:TEMP ('bundle-guard-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $root | Out-Null
try {
 $request=@{repository='D:\repo\hermes-build';worktree='D:\repo\hermes-build';output=(Join-Path $root 'output');cacheRoot='D:\repo\deliverables\cache\desktop-inputs';commit=('a'*40);controllerCommit=('b'*40);unsignedReceipt='';thumbprint=('c'*40)}
 $config=Join-Path $root 'request.json';$request | ConvertTo-Json | Set-Content -LiteralPath $config
 $failed=$false
 try { & 'SCRIPT' -Config $config } catch { if ($_.Exception.Message -notmatch 'owned Windows build layout|Windows layout output must be in the work root') { throw }; $failed=$true }
 if (!$failed -or (Test-Path -LiteralPath $request.output)) { throw 'Unowned output was accepted or written' }
 Write-Host 'UNOWNED_OUTPUT_REFUSED'
} finally {Remove-Item -LiteralPath $root -Recurse -Force}
'''.replace('SCRIPT', native)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'UNOWNED_OUTPUT_REFUSED' in result.stdout


@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
@pytest.mark.parametrize('mode', ['build', 'preflight', 'repack'])
def test_unseeded_external_build_fails_before_writing_job(mode):
    script = Path(__file__).resolve().parents[2] / 'scripts/build-windows-distribution.ps1'
    native = subprocess.check_output(['wslpath', '-w', str(script)], text=True).strip().replace("'", "''")
    command = r'''
$ErrorActionPreference='Stop'
$root=Join-Path $env:TEMP ('bundle-incomplete-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $root | Out-Null
try {
 $output=Join-Path 'D:\repo\deliverables\work' ('test-not-created-'+[guid]::NewGuid().ToString('N'))
 # An invalid commit safely proves the checkout guard runs before later controller work,
 # including creation of the legacy checkout-adjacent distribution lock.
 $request=@{repository='D:\repo\hermes-build';worktree='D:\repo\hermes-build';output=$output;cacheRoot='D:\repo\deliverables\cache\desktop-inputs';commit='invalid';controllerCommit=('b'*40);unsignedReceipt='';preflight=$false}
 if ('MODE' -eq 'preflight') { $request.preflight=$true }
 if ('MODE' -eq 'repack') { $request.unsignedReceipt='fixture-receipt';$request.commit='invalid' }
 $config=Join-Path $root 'request.json';$request | ConvertTo-Json | Set-Content -LiteralPath $config
 $failed=$false
 try { & 'SCRIPT' -Config $config } catch { if ($_.Exception.Message -notmatch 'Windows layout requires the clean Linux-Git-preseeded checkout|Exact complete source commit required') { throw }; $failed=$true }
 if (!$failed -or (Test-Path -LiteralPath $output)) { throw 'Incomplete build was accepted or wrote a job' }
 Write-Host 'INCOMPLETE_BUILD_REFUSED'
} finally {Remove-Item -LiteralPath $root -Recurse -Force}
'''.replace('SCRIPT', native).replace('MODE', mode)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'INCOMPLETE_BUILD_REFUSED' in result.stdout


@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
def test_preseeded_checkout_binding_does_not_fetch_or_checkout():
    script = Path(__file__).resolve().parents[2] / 'scripts/build-windows-distribution.ps1'
    source = script.read_text(encoding='utf-8')
    block = source[source.index('    $preseed = '):source.index('    if (!$existing) {', source.index('    $preseed = '))]
    command = r'''
$ErrorActionPreference='Stop'
$source='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
$repo='D:\repo\hermes-build';$work=$repo;$bundle='fixture.bundle'
$standalone=$true;$existing=$true;$reuseCache=$false
$env:SEMANTIER_WSL_GIT_DISTRO='Fixture'
$request=[pscustomobject]@{commit=$source;bundleRef='refs/local-windows-build/fixture';preseedOwnedCheckout=$true}
$script:actualHead=$source;$script:actualBundle=$source+' '+$request.bundleRef;$script:nativeCalls=0
function Git-Value($directory,$arguments) {
 if ($arguments[0] -eq 'bundle') { return $script:actualBundle }
 if ($arguments[1] -eq 'HEAD') { return $script:actualHead }
 return $source
}
function Native($tool,$arguments) { $script:nativeCalls++; throw 'unexpected source mutation' }
function Check([bool]$shouldPass) {
 $failed=$false
 try { BLOCK } catch { $failed=$true }
 if ($failed -eq $shouldPass) { throw 'unexpected admission result' }
}
Check $true
if ($script:nativeCalls -ne 0) { throw 'preseed ran a mutation' }
$script:actualHead='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';Check $false
$script:actualHead=$source;$script:actualBundle='conflicting bundle';Check $false
$script:actualBundle=$source+' '+$request.bundleRef;$env:SEMANTIER_WSL_GIT_DISTRO=$null;Check $false
Write-Host 'PRESEED_BINDING_PASS'
'''.replace('BLOCK', block)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRESEED_BINDING_PASS' in result.stdout


@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
def test_external_readiness_is_checked_before_version_reservation():
    script = Path(__file__).resolve().parents[2] / 'scripts/build-windows-distribution.ps1'
    source = script.read_text(encoding='utf-8')
    block = source[source.index('    # Check external-output readiness'):source.index('    $counter = ', source.index('    # Check external-output readiness'))]
    command = r'''
$ErrorActionPreference='Stop'
$request=[pscustomobject]@{python='fixture-python.exe'};$work='D:\repo\hermes-build';$output='D:\repo\deliverables\work\fixture'
$script:called=$false
function Native($tool,$arguments) {
 $script:called=$true
 if ($tool -cne 'fixture-python.exe' -or $arguments[0] -cne '-B' -or $arguments[1] -cne '-c' -or $arguments[3] -cne $work -or
     $arguments[4] -cne ($output+'\desktop-job') -or !$arguments[2].Contains('r(True,sys.argv[1],sys.argv[2],sys.argv[3])')) { throw 'invalid readiness invocation' }
 throw 'fixture external guard refusal'
}
$failed=$false
try { BLOCK } catch { if ($_.Exception.Message -notmatch 'fixture external guard refusal') { throw };$failed=$true }
if (!$failed -or !$script:called) { throw 'external guard was not enforced' }
Write-Host 'PREALLOCATION_READINESS_PASS'
'''.replace('BLOCK', block)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PREALLOCATION_READINESS_PASS' in result.stdout
