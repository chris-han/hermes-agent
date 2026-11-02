"""The native controller must reject out-of-layout paths before build effects."""
import json
from pathlib import Path
import subprocess
import tempfile

import pytest

ROOT = Path('/mnt/d/repo/deliverables/work')
POWERSHELL = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
CONTROLLER = Path(__file__).resolve().parents[2] / 'scripts/build-windows-distribution.ps1'


@pytest.mark.skipif(not ROOT.is_dir() or not POWERSHELL.is_file(), reason='Windows build host required')
@pytest.mark.parametrize('repository,worktree,output', [
    (r'D:\repo\elsewhere', r'D:\repo\elsewhere', r'D:\repo\deliverables\work\fixture'),
    (r'D:\repo\hermes-build', r'D:\repo\hermes-build', r'C:\outside\fixture'),
    (r'D:\repo\hermes-build', r'D:\repo\hermes-build', r'D:\repo\deliverables\work\..\outside'),
    (r'D:\repo\hermes-build', r'D:\repo\hermes-build', r'D:\repo\deliverables\work\nested\fixture'),
    (r'D:\repo\hermes-build', r'D:\repo\hermes-build', r'D:\repo\deliverables\work\CON.txt'),
    (r'D:\repo\hermes-build', r'D:\repo\hermes-build', r'D:\repo\deliverables\work\uncommitted-layout-controller'),
])
def test_native_controller_rejects_unowned_layout(repository, worktree, output):
    with tempfile.TemporaryDirectory(prefix='controller-guard-', dir=ROOT) as tmp:
        config = Path(tmp) / 'config.json'
        config.write_text(json.dumps({'repository': repository, 'worktree': worktree, 'output': output}))
        def win(path):
            return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
        result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                 '-File', win(CONTROLLER), '-Config', win(config)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert 'Windows layout' in result.stderr
