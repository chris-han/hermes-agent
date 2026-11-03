"""Tagless packaging enters the real builder with an exact source identity."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.bundles import desktop

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _git, _seed_repo


@pytest.mark.parametrize("variant", ["bundled", "light"])
def test_desktop_build_reaches_the_managed_payload_with_commit_ref(tmp_path, monkeypatch, variant):
    from scripts.bundles import desktop_prepare
    from scripts.bundles.desktop_inputs import build_environment
    from scripts.bundles.desktop_prepare import BuildRequest, PreparedDesktop

    _, repo = _seed_repo(tmp_path)
    sha = _git("rev-parse", "HEAD", cwd=repo)
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("BUILD_NUMBER", "123")
    defaults = {"HERMES_PREVIEW_FULL_CONNECTORS": "1", "HERMES_DATA_DIR_SUFFIX": "magic-test", "HERMES_HOME": None}
    monkeypatch.setenv("HERMES_BUNDLE_ENV_JSON", json.dumps(defaults))
    request = BuildRequest.create(repo, tag=None, commit=sha, variant=variant,
                                  work=tmp_path / "work", cache=tmp_path / "cache", bundle_env=defaults)
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import tree_digest
    (repo / "pm").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "pm/lock.json", repo / "pm/lock.json")
    lock = Lockfile(repo / "pm/lock.json")
    store = request.cache / "tools"
    binaries = {}
    for name in ("python", "node", "npm"):
        package = get_package(name)
        version = lock.version(name)
        assert version is not None
        entry = store / package.store_entry(version, request.target)
        binary = package.binary(entry, request.target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"published fixture")
        Facts(store / "facts.json").record(name, version, entry.name, package.env(entry, request.target), store,
                                           target=request.target, artifacts=[a["sha256"] for a in lock.artifacts(name, request.target)],
                                           digest=tree_digest(entry))
        binaries[name] = binary
    prepared = PreparedDesktop(request, binaries["python"], binaries["node"], Path(sys.executable),
                               tmp_path / "native", tmp_path / "packager", None, {}, "fixture-toolchain")
    env = build_environment(prepared, variant, os.environ)
    assert env.get("HERMES_PAYLOAD_TAG", "") == ""
    assert env["HERMES_BUILD_COMMIT"] == sha
    assert env["GITHUB_SHA"] == sha
    assert env["HERMES_PYTHON"] == str(binaries["python"])
    assert env["HERMES_PAYLOAD_VERSION"] == "0.1.2"
    assert json.loads(env["HERMES_BUNDLE_ENV_JSON"]) == defaults
    assert "BUILD_NUMBER" not in env
    shutil.rmtree(repo / "pm")

    # Composition crosses the preparation seam exactly once; native/Node build
    # behavior is exercised through their real provider tests, not command spies.
    calls = []
    def prepare(value):
        calls.append(value)
        return tmp_path / "prepared.json"
    built = []
    monkeypatch.setattr(desktop_prepare, "prepare", prepare)
    monkeypatch.setattr(desktop, "build_prepared", lambda path, args: built.append((path, args)))
    desktop.build(repo, None, variant, ['--publish=never'], commit_build=sha)
    assert len(calls) == 1 and calls[0].commit == sha and calls[0].variant == variant
    assert built == [(tmp_path / "prepared.json", ['--publish=never'])]
    before = len(calls)
    for tag, commit in [('v0.1.2', sha), (None, 'b' * 40), (None, 'short')]:
        with pytest.raises(ValueError):
            desktop.build(repo, tag, variant, [], commit_build=commit)
        assert len(calls) == before



@pytest.mark.parametrize("tag,commit", [(None, "a" * 40), ("v1.2.3+canary.20260911T120000Z", None)])
def test_store_build_rejects_nonstable_before_reading_or_preparing_repo(tmp_path, tag, commit):
    absent = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="Store.*stable"):
        desktop.build(absent, tag, "store", [], commit_build=commit)
    assert not absent.exists()


def test_termux_commit_args_reach_prerequisite_checks_without_mutation(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "must-not-be-written"
    helper = tmp_path / "bin"
    helper.mkdir()
    # Empty prerequisite commands are not build substitutes: stop at the first
    # actual prerequisite check, before any payload or output is created.
    env = _child_env(PATH=str(helper), HERMES_PAYLOAD_TAG="")
    scripts = [repo / "scripts/termux/termux_build.sh", repo / "scripts/termux/build_deb.sh"]
    for script in scripts:
        args = ["--repo", str(repo), "--commit", "a" * 40, "--out", str(out)]
        if script.name == "build_deb.sh":
            args += ["--payload", str(tmp_path / "absent")]
        result = subprocess.run([_BASH, str(script), *args], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "usage:" not in result.stderr
        assert not out.exists()


def test_external_job_driver_routes_prepared_products_outside_checkout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.bundles import desktop_inputs, desktop_prepare, native

    source = tmp_path / 'source'
    build = tmp_path / 'job' / 'desktop-build'
    request = SimpleNamespace(source=source, target='linux-x64', release_epoch=None, version='1.2.3',
                              tag=None, channel_request=None, external_outputs=True, desktop_build_dir=build,
                              workspaces=list, commit='a' * 40)
    prepared = SimpleNamespace(request=request, validate=lambda: None, node=Path('/node'),
                               icon_python=Path('/python'), native=tmp_path / 'native',
                               packager=tmp_path / 'packager', payload=tmp_path / 'payload', native_toolchain='fixture')
    monkeypatch.setattr(desktop_inputs, 'validate_builder_identity', lambda *args: None)
    monkeypatch.setattr(desktop_inputs, 'select_variant', lambda *args: 'bundled')
    monkeypatch.setattr(desktop_inputs, 'build_environment', lambda *args: {'HERMES_DESKTOP_BUILD_DIR': str(build)})
    monkeypatch.setattr(desktop_prepare, 'require_source', lambda *args: None)
    monkeypatch.setattr(native, 'finish_native', lambda *args: False)
    monkeypatch.setattr(desktop, 'flavored_assets', lambda *args: pytest.fail('checkout assets must not be changed'))
    assets = []
    monkeypatch.setattr(desktop, 'stage_external_assets', lambda *args: assets.append(args))
    calls = []
    monkeypatch.setattr(desktop, 'run', lambda argv, **kwargs: calls.append((argv, kwargs)))
    desktop._build_prepared(prepared, [], None)
    stamp = next(args for args, _ in calls if any(str(arg).endswith('write-build-stamp.mjs') for arg in args))
    renderer = next(args for args, _ in calls if any(str(arg).endswith('scripts/build/desktop.mjs') for arg in args))
    assert assets == [(source / 'apps/desktop/assets', build / 'products/icons/apps/desktop/assets', build / 'assets')]
    assert stamp[stamp.index('--output-dir') + 1] == str(build)
    assert renderer[renderer.index('--stamp') + 1] == str(build / 'install-stamp.json')
    assert renderer[renderer.index('--out') + 1] == str(build / 'dist')


def test_external_assets_overlay_preserves_checkout_and_refuses_existing_output(tmp_path):
    source = tmp_path / 'source-assets'
    rendered = tmp_path / 'rendered-assets'
    output = tmp_path / 'job-assets'
    source.mkdir()
    rendered.mkdir()
    (source / 'icon.ico').write_text('source')
    (source / 'manifest.xml').write_text('template')
    (rendered / 'icon.ico').write_text('rendered')
    desktop.stage_external_assets(source, rendered, output)
    assert (output / 'icon.ico').read_text() == 'rendered'
    assert (output / 'manifest.xml').read_text() == 'template'
    assert (source / 'icon.ico').read_text() == 'source'
    with pytest.raises(ValueError, match='already exists'):
        desktop.stage_external_assets(source, rendered, output)
    (rendered / 'link').symlink_to(source / 'manifest.xml')
    with pytest.raises(ValueError, match='redirected'):
        desktop.stage_external_assets(source, rendered, tmp_path / 'other-output')
    assert not (tmp_path / 'other-output').exists()


def test_external_job_cli_binds_mode_and_forwards_packager_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.bundles import desktop_prepare

    seen = {}
    def create(source, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(desktop_prepare.BuildRequest, 'create', create)
    monkeypatch.setattr(desktop_prepare, 'prepare', lambda request, clean=False: tmp_path / 'prepared.json')
    monkeypatch.setattr(desktop, 'build_prepared', lambda path, args: seen.update(builder_args=args))
    monkeypatch.setattr(sys, 'argv', ['desktop.py', '--repo', str(tmp_path / 'source'), '--commit', 'a' * 40,
                                     '--work', str(tmp_path / 'job'), '--cache', str(tmp_path / 'cache'),
                                     '--external-outputs', '--', f'-c.directories.output={tmp_path / "release"}'])
    desktop.main()
    assert seen['external_outputs'] is True
    assert seen['work'] == tmp_path / 'job'
    assert seen['builder_args'] == [f'-c.directories.output={tmp_path / "release"}']


def test_windows_external_job_refuses_before_preparation_until_layout_complete(tmp_path, monkeypatch, capsys):
    from scripts.bundles import desktop_prepare

    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(desktop_prepare.BuildRequest, 'create', lambda *args, **kwargs: pytest.fail('preparation must not start'))
    monkeypatch.setattr(sys, 'argv', ['desktop.py', '--repo', str(tmp_path / 'source'), '--commit', 'a' * 40,
                                     '--work', str(tmp_path / 'job'), '--cache', str(tmp_path / 'cache'),
                                     '--external-outputs'])
    with pytest.raises(SystemExit) as error:
        desktop.main()
    assert error.value.code == 1
    assert 'external Windows build path migration is incomplete' in capsys.readouterr().err
    assert not (tmp_path / 'job').exists()


def test_windows_prepared_external_job_refuses_before_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.bundles import desktop_inputs, desktop_prepare

    monkeypatch.setattr(sys, 'platform', 'win32')
    request = SimpleNamespace(external_outputs=True, source=tmp_path / 'source')
    monkeypatch.setattr(desktop_prepare.PreparedDesktop, 'load', lambda path: SimpleNamespace(request=request))
    monkeypatch.setattr(desktop_inputs, 'build_lock', lambda *args: pytest.fail('source lock must not be touched'))
    with pytest.raises(ValueError, match='external Windows build path migration is incomplete'):
        desktop.build_prepared(tmp_path / 'prepared.json', [])
