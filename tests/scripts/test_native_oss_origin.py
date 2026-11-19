"""Real native URI handoff with injected OS boundary; no network or installation."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_oss_uri_and_digest_are_bound_and_bad_origins_refused(tmp_path):
    pwsh = shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell required for native handoff contract')
    source = ROOT/'apps/desktop/scripts/native-package-update.ps1'
    request = {
        'schema': 2, 'base': 'https://cloud.semantier.cn', 'packageBase': 'https://assets.semantier.cn',
        'targetBuildId': 'a'*32, 'identity': 'Semantier.Agent.Acceptance',
        'publisher': 'CN=Semantier Technology, O=Semantier Technology', 'version': '0.0.131.0',
        'fullName': 'fixture-v130', 'manifestSha256': 'c'*64,
    }
    key = 'releases/channel-builds/'+request['targetBuildId']+'/SemantierAgent.msix'
    row = {'key': key, 'kind': 'main-package', 'arch': 'x64', 'identity': request['identity'],
           'publisher': request['publisher'], 'version': request['version'], 'size': 1600000000,
           'sha256': 'd'*64, 'nativeDigest': 'AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=', 'references': []}
    request['graph'] = {'schema': 1, 'transport': 'windows-package-uri', 'root': key,
                        'payloadInventorySha256': 'e'*64, 'objects': [row]}
    inputs = tmp_path/'request.json';inputs.write_text(json.dumps(request))
    script = tmp_path/'check.ps1'
    script.write_text(r'''
param($Source,$InputPath,$OutputDirectory)
$ErrorActionPreference='Stop'
. $Source
$request=Get-Content $InputPath -Raw|ConvertFrom-Json
$rows=@(Read-NativeUpdateGraph $request)
function Get-AppxPackage {param($Name) return [pscustomobject]@{Name=$request.identity;PackageFullName=$request.fullName;Publisher=$request.publisher}}
function Invoke-BoundNativePackageUri {param($Uris,$Digests,$Root,$Minutes,$ForceTargetAppShutdown)
 $script:captured=@{uris=@($Uris);digests=@($Digests);root=$Root}; throw 'CAPTURED_NATIVE_BOUNDARY'
}
try{Invoke-NativePackageUpdate @{request=$request;rows=$rows;directory=$OutputDirectory};throw 'Boundary was not called'}
catch{if($_.Exception.Message -cne 'CAPTURED_NATIVE_BOUNDARY'){throw}}
if($captured.root -cne ($request.packageBase+'/'+$request.graph.root)){throw 'OSS URI was not selected'}
$negative=@(
 @{packageBase='https://evil.example'},
 @{packageBase='https://assets.semantier.cn?secret=1'},
 @{base='https://download.semantier.cn'},
 @{schema=1},
 @{targetBuildId=('b'*32)}
)
foreach($mutation in $negative){
 $copy=$request|ConvertTo-Json -Depth 16|ConvertFrom-Json
 foreach($entry in $mutation.GetEnumerator()){$copy.($entry.Key)=$entry.Value}
 $refused=$false
 try{$null=Read-NativeUpdateGraph $copy}catch{$refused=$true}
 if(!$refused){throw 'Untrusted package origin/binding was accepted'}
}
$captured|ConvertTo-Json -Compress
''')
    result = subprocess.run([pwsh, '-NoProfile', '-NonInteractive', '-File', str(script),
                             str(source), str(inputs), str(tmp_path)], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert captured == {'uris': ['https://assets.semantier.cn/'+key],
                        'digests': [row['nativeDigest']], 'root': 'https://assets.semantier.cn/'+key}
