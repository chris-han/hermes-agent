"""WSL Windows builds must not invoke a native Git-backed hook installer."""
import json
import os
from pathlib import Path
import subprocess


def test_prepare_skips_hook_only_for_explicit_wsl_git_build(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = json.loads((root / 'package.json').read_text())['scripts']['prepare']
    (tmp_path / '.git').mkdir()
    marker = tmp_path / 'invoked'
    hook = tmp_path / 'lefthook'
    hook.write_text('#!/bin/sh\nprintf invoked > "$HOOK_MARKER"\n')
    hook.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'], 'HOOK_MARKER': str(marker)}
    env.pop('SEMANTIER_WSL_GIT_DISTRO', None)
    subprocess.run(command, cwd=tmp_path, env=env, shell=True, check=True)
    assert marker.read_text() == 'invoked'
    marker.unlink()
    subprocess.run(command, cwd=tmp_path, env={**env, 'SEMANTIER_WSL_GIT_DISTRO':'fixture'}, shell=True, check=True)
    assert not marker.exists()
