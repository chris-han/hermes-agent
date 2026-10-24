"""Disposable Windows proof: real venv + dependency import survives two source revisions.

Run with the cached native interpreter, --source pointing at the candidate checkout.
No application installation, machine trust changes, provider calls, or MSIX build.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from pm.store import current_target
    from pm.lock import SCHEMA
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV
    from scripts.bundles import native, runtime_cache
    from scripts.bundles.native_prepared import publish_prepared, load_prepared
    from scripts.bundles.bytecode import bake_bytecode
    if current_target() != 'win32-x64':
        raise SystemExit('This fixture requires native Windows x64')
    root = Path(tempfile.mkdtemp(prefix='semantier-dependency-reuse-'))
    print('Fixture root:', root, flush=True)
    def run(*command, cwd=None):
        return subprocess.check_output([str(x) for x in command], cwd=cwd, text=True).strip()
    def refuse(*args, **kwargs):
        raise AssertionError('source-only refresh tried to install/copy dependencies')
    try:
        source = root / 'source'; source.mkdir()
        out = root / 'payload'; out.mkdir()
        os.environ['HERMES_HOME'] = str(root / 'home'); (root / 'home').mkdir()
        fact = json.loads((args.tools / 'facts.json').read_text(encoding='utf-8-sig'))['packages']['python']
        tool = out / 'tools' / fact['entry']
        print('Copying the pinned native interpreter for the fixture...', flush=True)
        shutil.copytree(args.tools / fact['entry'], tool)
        python = tool / 'python.exe'
        (out / 'tools/facts.json').write_text(json.dumps({'schema': SCHEMA, 'packages': {'python': fact}}))
        (source / 'pyproject.toml').write_text('[project]\nname="reuse-fixture"\nversion="1.0.0"\ndependencies=["setuptools==83.0.0"]\n')
        (source / 'uv.lock').write_text('version = 1\n')
        (source / 'fixture_backend.py').write_text('VALUE = 1\n')
        (source / 'pm').mkdir()
        shutil.copyfile(args.source / 'pm/lock.json', source / 'pm/lock.json')
        for name in RESOURCE_ENV:
            (source / name).mkdir(); (source / name / '.keep').touch()
        run('git', 'init', source); run('git', 'config', 'user.name', 'Fixture', cwd=source)
        run('git', 'config', 'user.email', 'fixture@example.invalid', cwd=source)
        def commit():
            run('git', 'add', '.', cwd=source); run('git', 'commit', '-m', 'fixture', cwd=source)
            return run('git', 'rev-parse', 'HEAD', cwd=source)
        revision = commit()
        shutil.copytree(source, out / 'hermes-agent', ignore=shutil.ignore_patterns('.git'))
        uv = next(args.tools.glob('uv-*/uv.exe'))
        print('Creating one real fixture environment and installing the pinned dependency from cache...', flush=True)
        run(uv, 'venv', '--python', python, out / 'venv')
        venv_python = out / 'venv/Scripts/python.exe'
        run(uv, '--offline', 'pip', 'install', '--python', venv_python, '--cache-dir', args.wheels, 'setuptools==83.0.0')
        site = out / 'venv/Lib/site-packages'
        sentinel = site / 'reuse-sentinel.txt'; sentinel.write_text('same physical dependency environment')
        inode = sentinel.stat().st_ino
        pm = out / 'pm-runtime'; (pm / 'Lib/site-packages').mkdir(parents=True)
        (pm / 'pm-runtime.json').write_text(json.dumps({'python': '../tools/' + fact['entry'] + '/python.exe', 'sitePackages': 'Lib/site-packages'}))
        (out / 'enabled-features.json').write_text('[]'); (out / 'uv-cache').mkdir()
        inputs = AgentInputs(project=out / 'hermes-agent/pyproject.toml', code=out / 'hermes-agent',
            repo='hermes-agent', placement='contained', target=current_target(), python=python,
            tools=out / 'tools', environment=out / 'venv', site_packages=site, pm_runtime=pm,
            features=out / 'enabled-features.json', resources={n:out / 'hermes-agent' / n for n in RESOURCE_ENV}, ref=revision)
        identity = hashlib.sha256(python.read_bytes()).hexdigest()
        key = runtime_cache.dependency_key(source, revision, identity, current_target())
        bake_bytecode(out, python)  # completed cold preparation includes sealed dependency bytecode
        prepared = publish_prepared(out, source, revision, inputs)
        runtime_cache.publish_dependencies(prepared, key); runtime_cache.retain_runtime(out, source, 'initial')
        observations = []
        with ExitStack() as stack:
            for name in ['prepare_tools', 'stage_tools', 'build_environment']:
                stack.enter_context(patch('pm.' + name, refuse))
            stack.enter_context(patch.object(native, 'stage_pm_runtime', refuse))
            stack.enter_context(patch.object(native, 'verify_offline_cache', refuse))
            for value in [2, 3]:
                print('Refreshing source revision', value, 'without dependency installation...', flush=True)
                (source / 'fixture_backend.py').write_text(f'VALUE = {value}\n'); revision = commit()
                assert runtime_cache.dependency_key(source, revision, identity, current_target()) == key
                entry = root / 'cache/native/runtime' / revision
                prepared = native.prepare_native(out=out, ref=revision, source=source, cache=args.wheels,
                    runtime_cache=(entry, revision), dependency_identity=key)
                assert load_prepared(prepared).ref == revision
                observed = run(venv_python, '-I', '-c', 'import fixture_backend, setuptools, importlib.metadata; print(fixture_backend.VALUE, importlib.metadata.version("setuptools"))')
                assert observed == f'{value} 83.0.0', observed
                assert sentinel.stat().st_ino == inode
                assert not (entry / 'runtime.tar').exists()
                observations.append({'revision': revision, 'import': observed, 'sameEnvironment': True})
        (source / 'uv.lock').write_text('version = 2\n'); changed = commit()
        assert runtime_cache.dependency_key(source, changed, identity, current_target()) != key
        sentinel.write_text('corrupted dependency environment')
        assert not runtime_cache.admit_dependencies(out, source, key)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'result':'PASS','producedAt':datetime.now(timezone.utc).isoformat(),
            'platform':sys.platform,'python':sys.version,'fixtureInitialInstalls':1,'sourceRefreshInstalls':0,
            'observations':observations,'dependencyChangeInvalidates':True,'corruptionRejected':True,
            'producerSha256':{n:hashlib.sha256((args.source/n).read_bytes()).hexdigest() for n in
              ['scripts/bundles/native.py','scripts/bundles/runtime_cache.py','scripts/bundles/desktop_prepare.py',
               'scripts/bundles/payload.py','scripts/bundles/bytecode.py']}},indent=2)+'\n')
        print('PASS: one real environment, two real source refreshes, zero dependency installs; dependency changes and corruption invalidate reuse.', flush=True)
    finally:
        def writable(func, path, error):
            os.chmod(path, stat.S_IWRITE); func(path)
        shutil.rmtree(root, onexc=writable)


if __name__ == '__main__':
    main()
