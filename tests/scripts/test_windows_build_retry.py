"""Actual host PowerShell version reservations: retry cannot rewind or invent state."""
from pathlib import Path
import base64
import subprocess
import pytest

HOST = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')

@pytest.mark.skipif(not HOST.exists(), reason='requires Windows host PowerShell')
def test_native_retry_preserves_counter_and_rejects_other_versions():
    script = Path(__file__).resolve().parents[2] / 'scripts/next-windows-build-version.ps1'
    native = subprocess.check_output(['wslpath', '-w', str(script)], text=True).strip().replace("'", "''")
    command = r'''
$ErrorActionPreference='Stop'
$root=Join-Path $env:TEMP ('semantier-version-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $root | Out-Null
try {
 $script='SCRIPT';$state=Join-Path $root 'version.json';$source='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
 $v=& $script -State $state -SourceCommit $source
 if($v -cne '0.0.1'){throw 'First version differs'}
 $before=[IO.File]::ReadAllBytes($state)
 $v=& $script -State $state -SourceCommit ('b'*40) -RetryVersion '0.0.1'
 if($v -cne '0.0.1'){throw 'Retry version differs'}
 if([Convert]::ToBase64String([IO.File]::ReadAllBytes($state)) -cne [Convert]::ToBase64String($before)){throw 'Retry mutated counter'}
 foreach($bad in @('0.0.0','0.0.2','bogus')){
  $failed=$false;try{& $script -State $state -SourceCommit $source -RetryVersion $bad | Out-Null}catch{$failed=$true}
  if(!$failed){throw 'Invalid retry accepted'}
 }
 $failed=$false;try{& $script -State (Join-Path $root 'missing.json') -SourceCommit $source -RetryVersion '0.0.1' | Out-Null}catch{$failed=$true}
 if(!$failed){throw 'Missing state retry accepted'}
 $v=& $script -State $state -SourceCommit $source
 if($v -cne '0.0.2'){throw 'Normal increment broken'}
 if(Test-Path ($state+'.lock')){throw 'Leaked lock'}
 Write-Host 'RETRY_PASS'
} finally {Remove-Item -LiteralPath $root -Recurse -Force}
'''.replace('SCRIPT', native)
    result = subprocess.run([str(HOST), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
                             base64.b64encode(command.encode('utf-16le')).decode()], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RETRY_PASS' in result.stdout
