"""Exercise production PowerShell lifecycle helpers with real disposable Git repos.

The portable lane proves the Git algorithm; native Windows path/process and
BuildOnly qualification remain separate from this result.
"""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_real_git_worktree_lifecycle():
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(root / 'scripts/tests/test-deploy-ecs-worktree.ps1')],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMMON_DIR_EQUAL:' in result.stdout
    assert 'PASS: exact SHA' in result.stdout
