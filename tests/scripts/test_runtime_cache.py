"""Reuse sealed runtime bytes without relabelling changed backend inputs."""
import json
import subprocess
from pathlib import Path
from scripts.bundles.runtime_cache import runtime_key, publish_cache, restore_cache


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args], text=True).strip()


def test_key_reuses_ui_and_invalidates_backend_recipe_toolchain(tmp_path):
    git(tmp_path,'init'); git(tmp_path,'config','user.name','Fixture'); git(tmp_path,'config','user.email','fixture@example.invalid')
    for name in ['entry.py','uv.lock','apps/desktop/view.tsx','scripts/bundles/native.py']:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('one')
    def key(toolchain='observed'):
        git(tmp_path,'add','.'); git(tmp_path,'commit','--allow-empty','-m','fixture')
        return runtime_key(tmp_path,git(tmp_path,'rev-parse','HEAD'),toolchain,'win32-x64')
    first=key(); (tmp_path/'apps/desktop/view.tsx').write_text('UI/logo/version')
    assert key()==first
    (tmp_path/'entry.py').write_text('backend change'); second=key(); assert second!=first
    assert key('new compiler')!=second
    (tmp_path/'scripts/bundles/native.py').write_text('new recipe'); assert key()!=second


def payload(tmp_path):
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV
    from scripts.bundles.native_prepared import publish_prepared
    from pm.store import current_target
    out=tmp_path/'cold'; out.mkdir(); code=out/'hermes-agent'; code.mkdir()
    (code/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="1.0"\n')
    (code/'entry.py').write_text('unchanged backend')
    for name in RESOURCE_ENV: (code/name).mkdir()
    python=out/'tools/python/python.exe'; python.parent.mkdir(parents=True); python.write_bytes(b'native fixture')
    site=out/'venv/Lib/site-packages'; site.mkdir(parents=True)
    pm=out/'pm-runtime'; pm.mkdir(); (pm/'site').mkdir()
    (pm/'pm-runtime.json').write_text(json.dumps({'python':'../tools/python/python.exe','sitePackages':'site'}))
    (out/'enabled-features.json').write_text('[]'); (out/'uv-cache').mkdir()
    inputs=AgentInputs(project=code/'pyproject.toml',code=code,repo='hermes-agent',placement='contained',
        target=current_target(),python=python,tools=out/'tools',environment=out/'venv',site_packages=site,
        pm_runtime=pm,features=out/'enabled-features.json',resources={n:code/n for n in RESOURCE_ENV},ref='a'*40)
    return out,publish_prepared(out,tmp_path,'a'*40,inputs)


def test_two_warm_preparations_and_corrupt_archive(tmp_path):
    from scripts.bundles.native_prepared import load_prepared
    out,prepared=payload(tmp_path); cache=tmp_path/'native/runtime/key'; publish_cache(cache,'key',out,prepared)
    for name in ['warm-one','warm-two']:
        target=tmp_path/name; target.mkdir(); restored=restore_cache(cache,'key',target,tmp_path,'b'*40)
        inputs=load_prepared(restored)
        assert inputs.python==target/'tools/python/python.exe' and inputs.ref=='b'*40
        assert (inputs.code/'entry.py').read_text()=='unchanged backend'
    (cache/'runtime.tar').write_bytes(b'corrupt'); target=tmp_path/'bad'; target.mkdir()
    assert restore_cache(cache,'key',target,tmp_path,'b'*40) is None
    assert not list(target.iterdir())


def test_wrong_key_and_nonempty_output_are_preserved(tmp_path):
    out,prepared=payload(tmp_path); cache=tmp_path/'native/runtime/key'; publish_cache(cache,'key',out,prepared)
    assert restore_cache(cache,'wrong',tmp_path/'new',tmp_path,'b'*40) is None
    target=tmp_path/'dirty'; target.mkdir(); (target/'user.txt').write_text('preserve')
    assert restore_cache(cache,'key',target,tmp_path,'b'*40) is None
    assert (target/'user.txt').read_text()=='preserve'
