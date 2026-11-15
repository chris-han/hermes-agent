"""Execute the release owner's actual remote cleanup against disposable files."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


def test_expired_upload_cleanup_keeps_current_open_recent_and_unknown_files(tmp_path):
    pwsh = shutil.which('pwsh')
    if not pwsh:
        pytest.skip('portable PowerShell unavailable')
    source = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    extract = tmp_path / 'extract.ps1'
    extract.write_text(r'''
$ErrorActionPreference='Stop'
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$null,[ref]$null)
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-ExpiredEcsUploadCleanup'},$true)
if(!$f){throw 'Missing publication cleanup helper'}
. ([scriptblock]::Create($f.Extent.Text))
Get-ExpiredEcsUploadCleanup -Root $args[1] -CurrentStage $args[2]
''')
    root = tmp_path / 'distribution'
    root.mkdir()
    stages = {}
    for i, name in enumerate(['expired', 'current', 'recent', 'unknown', 'open', 'linked']):
        stage = root / ('.semantier-stage-' + str(i) * 32)
        stage.mkdir(mode=0o700)
        package = stage / 'SemantierAgent.msix'
        package.write_bytes(b'protected package')
        os.utime(package, (time.time() - 172800,) * 2)
        stages[name] = stage
    os.utime(stages['recent'] / 'SemantierAgent.msix', None)
    (stages['unknown'] / 'unrelated.txt').write_text('preserve')
    (stages['linked'] / 'SemantierAgent.msix').unlink()
    (stages['linked'] / 'SemantierAgent.msix').symlink_to(stages['current'] / 'SemantierAgent.msix')
    public = root / 'releases'
    public.mkdir()
    (public / 'SemantierAgent.msix').write_bytes(b'live')
    command = subprocess.run([pwsh, '-NoProfile', '-File', str(extract), str(source),
                              str(root), str(stages['current'])], capture_output=True, text=True, check=True)
    # The host process inventory is external/nondeterministic. Limit it to our
    # real fixture owner; keep the actual /proc/fd reads and open-file guard.
    command.stdout = command.stdout.replace('from pathlib import Path',
        'from pathlib import Path\n'
        '_glob=Path.glob\n'
        f'Path.glob=lambda self,pattern: iter([Path("/proc/{os.getpid()}")]) '
        'if self==Path("/proc") else _glob(self,pattern)')
    with (stages['open'] / 'SemantierAgent.msix').open('rb'):
        result = subprocess.run(['bash', '-se'], input=command.stdout,
                                capture_output=True, text=True, check=False)
        again = subprocess.run(['bash', '-se'], input=command.stdout,
                               capture_output=True, text=True, check=True)
    assert json.loads(again.stdout)['removed'] == []
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['removed'] == [str(stages['expired'])]
    assert not stages['expired'].exists()
    assert all(stage.exists() for key, stage in stages.items() if key != 'expired')
    assert (public / 'SemantierAgent.msix').read_bytes() == b'live'


def test_cleanup_and_stage_recovery_are_wired_to_verified_publication():
    source = (Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1').read_text()
    assert 'Complete-Checkpoint \'upload-prepared\'' in source
    call = source.index('$receipt.expiredUploads=Ssh (Get-ExpiredEcsUploadCleanup')
    assert source.index("$receipt.h3='PUBLIC_H3_HEAD_STATUS_SIZE_PASS'") < call
    assert source.index("$receipt.result='BETA_PUBLISHED_UNVERIFIED'") < call
    assert source.index('$posixParent=$RemoteRoot.Substring') < source.index("if($script:checkpoints.ContainsKey('published'))")
