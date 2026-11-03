"""Committed workspace inputs can be staged outside a source checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='fixture creates its repository with Linux Git')


def test_dangling_reparse_entry_is_not_treated_as_missing():
    from types import SimpleNamespace
    from pm.filesystem import is_reparse_point
    class Reparse:
        def lstat(self):return SimpleNamespace(st_file_attributes=0x400)
    class Missing:
        def lstat(self):raise FileNotFoundError
    assert is_reparse_point(Reparse()) and not is_reparse_point(Missing())


def _git(repo, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(repo), *args], text=True).strip()


def _fixture(tmp_path):
    repo = tmp_path / 'source'
    repo.mkdir()
    _git(repo, 'init', '-q')
    files = {
        'package.json': {'name': 'fixture', 'private': True, 'workspaces': ['apps/*']},
        'package-lock.json': {'name': 'fixture', 'lockfileVersion': 3, 'packages': {
            '': {'name': 'fixture'}, 'apps/app': {'name': 'app'}, 'apps/shared': {'name': 'shared'}}},
        'apps/app/package.json': {'name': 'app', 'version': '1.0.0', 'dependencies': {'shared': 'file:../shared'}},
        'apps/app/tsconfig.json': {'compilerOptions': {'strict': True}},
        'apps/shared/package.json': {'name': 'shared', 'version': '1.0.0', 'main': 'index.js'},
    }
    for name, body in files.items():
        file = repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(body))
    (repo / 'apps/shared/index.js').write_text('module.exports = "pinned"')
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'fixture')
    return repo, _git(repo, 'rev-parse', 'HEAD')


def test_stages_committed_manifests_and_local_workspace_inputs(tmp_path):
    from scripts.bundles.desktop_node_workspace import stage_node_workspace
    repo, commit = _fixture(tmp_path)
    (repo / 'apps/shared/index.js').write_text('uncommitted')
    (repo / 'apps/shared/ignored-local.txt').write_text('not tracked')
    out = tmp_path / 'job'
    stage_node_workspace(repo, commit, out)
    assert (out / 'apps/shared/index.js').read_text() == 'module.exports = "pinned"'
    assert not (out / 'apps/shared/ignored-local.txt').exists()
    assert (out / 'apps/app/package.json').is_file()
    assert json.loads((out / 'apps/app/tsconfig.json').read_text())['compilerOptions']['strict'] is True
    assert not (repo / 'node_modules').exists()
    with pytest.raises(ValueError, match='already exists'):
        stage_node_workspace(repo, commit, out)


def test_rejects_symlink_in_committed_workspace(tmp_path):
    from scripts.bundles.desktop_node_workspace import stage_node_workspace
    repo, _ = _fixture(tmp_path)
    (repo / 'apps/shared/escape').symlink_to(tmp_path, target_is_directory=True)
    _git(repo, 'add', 'apps/shared/escape')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'link')
    with pytest.raises(ValueError, match='link'):
        stage_node_workspace(repo, _git(repo, 'rev-parse', 'HEAD'), tmp_path / 'job')
    assert not (tmp_path / 'job').exists()


def test_rejects_redirected_destination_and_unlocked_local_dependency(tmp_path):
    from scripts.bundles.desktop_node_workspace import stage_node_workspace
    repo, commit = _fixture(tmp_path)
    redirected = tmp_path / 'redirected'
    redirected.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='redirected'):
        stage_node_workspace(repo, commit, redirected / 'job')
    manifest = repo / 'apps/app/package.json'
    data = json.loads(manifest.read_text())
    data['dependencies']['shared'] = 'file:../../../outside'
    manifest.write_text(json.dumps(data))
    _git(repo, 'add', 'apps/app/package.json')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'unsafe dependency')
    with pytest.raises(ValueError, match='not a locked workspace'):
        stage_node_workspace(repo, _git(repo, 'rev-parse', 'HEAD'), tmp_path / 'job')
    assert not (tmp_path / 'job').exists()


def test_staged_workspace_installs_and_resolves_offline(tmp_path):
    from scripts.bundles.desktop_node_workspace import stage_node_workspace
    node = os.environ.get('SEMANTIER_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node is required for the offline npm fixture')
    repo, _ = _fixture(tmp_path)
    driver = Path(__file__).resolve().parents[2] / 'scripts/build/node-deps.mjs'
    env = {**os.environ, 'npm_config_offline': 'true', 'npm_config_cache': str(tmp_path / 'npm-cache')}
    subprocess.run([node, str(driver), '--npm', 'install', '--package-lock-only', '--ignore-scripts',
                    '--no-audit', '--no-fund', '--offline'], cwd=repo, env=env, check=True, capture_output=True)
    _git(repo, 'add', 'package-lock.json')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'lock')
    stage = tmp_path / 'job'
    stage_node_workspace(repo, _git(repo, 'rev-parse', 'HEAD'), stage)
    subprocess.run([node, str(driver), '--npm', 'ci', '--workspace', 'apps/app', '--include-workspace-root=true',
                    '--no-audit', '--no-fund', '--offline'], cwd=stage, env=env, check=True, capture_output=True)
    code = 'const {createRequire}=require("module");console.log(createRequire(process.argv[1])("shared"))'
    result = subprocess.run([node, '-e', code, str(stage / 'apps/app/package.json')], cwd=stage,
                            env=env, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'pinned'
    assert not (repo / 'node_modules').exists()
