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


def test_external_node_environment_discards_inherited_loader_options(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.bundles.desktop_inputs import external_node_environment
    source=tmp_path/'source';hook=source/'scripts/build/external-node-resolver.mjs';hook.parent.mkdir(parents=True);hook.write_text('// fixture')
    work=tmp_path/'work';root=work/'node-workspace';root.mkdir(parents=True)
    request=SimpleNamespace(source=source,work=work,cache=tmp_path/'cache',external_outputs=True)
    env=external_node_environment(request,root,{'NODE_OPTIONS':'--require=/tmp/foreign.cjs','Node_Options':'--inspect'})
    assert env['NODE_OPTIONS']=='--import='+hook.as_uri()
    assert 'Node_Options' not in env
    from scripts.bundles import desktop,desktop_node_workspace,desktop_prepare
    (work/'.desktop-preparation').write_text(str(source))
    root.rmdir();calls=[]
    request.commit='a'*40;request.workspaces=lambda:['apps/app']
    monkeypatch.setattr(desktop_node_workspace,'stage_node_workspace',lambda source,commit,destination:destination.mkdir())
    monkeypatch.setattr(desktop,'run',lambda *args,**kwargs:calls.append(kwargs['env']))
    desktop_prepare.prepare_external_node_dependencies(request,Path('/fixture/node'),'fixture',{'Node_Options':'--require=/tmp/foreign.cjs'})
    assert all(key.upper()!='NODE_OPTIONS' for key in calls[0])


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


def test_preparation_installs_committed_workspace_with_real_node_driver(tmp_path):
    from types import SimpleNamespace
    from scripts.bundles.desktop_prepare import prepare_external_node_dependencies

    node = os.environ.get('SEMANTIER_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node is required for the offline npm fixture')
    repo, _ = _fixture(tmp_path)
    driver = Path(__file__).resolve().parents[2] / 'scripts/build/node-deps.mjs'
    env = {**os.environ, 'npm_config_offline': 'true', 'npm_config_cache': str(tmp_path / 'cache/npm')}
    subprocess.run([node, str(driver), '--npm', 'install', '--package-lock-only', '--ignore-scripts',
                    '--no-audit', '--no-fund', '--offline'], cwd=repo, env=env, check=True, capture_output=True)
    hook = repo / 'scripts/build/external-node-resolver.mjs'
    hook.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[2] / 'scripts/build/external-node-resolver.mjs', hook)
    _git(repo, 'add', 'package-lock.json', 'scripts')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'lock')
    work = tmp_path / 'job'
    work.mkdir()
    (work / '.desktop-preparation').write_text(str(repo))
    request = SimpleNamespace(source=repo, commit=_git(repo, 'rev-parse', 'HEAD'), work=work,
                              cache=tmp_path / 'cache', external_outputs=True,
                              workspaces=lambda: ['apps/app'])
    root = prepare_external_node_dependencies(request, Path(node), 'fixture', env)
    assert root == work / 'node-workspace'
    assert (root / 'node_modules/.hermes-node-deps').is_file()
    assert not (repo / 'node_modules').exists()
    code = 'const {createRequire}=require("module");console.log(createRequire(process.argv[1])("shared"))'
    result = subprocess.run([node, '-e', code, str(root / 'apps/app/package.json')],
                            cwd=root, env=env, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'pinned'
    from scripts.bundles.desktop_inputs import external_node_environment
    source_module = repo / 'apps/app/entry.cjs'
    source_module.write_text('console.log(require("shared"))')
    poisoned = repo / 'node_modules/shared'
    poisoned.mkdir(parents=True)
    (poisoned / 'index.js').write_text('module.exports = "foreign"')
    bound = external_node_environment(request, root, env)
    result = subprocess.run([node, str(source_module)], cwd=repo, env=bound,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'pinned'
    subprocess.run([node, str(driver), '--source', str(root), '--reuse', '--no-install',
                    '--native-toolchain', 'fixture', '--workspace', 'apps/app'],
                   cwd=repo, env=bound, check=True, capture_output=True, text=True)
    sentinel = root / 'preserve-me'
    sentinel.write_text('owned input')
    with pytest.raises(ValueError, match='already exists'):
        prepare_external_node_dependencies(request, Path(node), 'fixture', env)
    assert sentinel.read_text() == 'owned input'
