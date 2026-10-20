"""Reuse sealed runtime bytes without relabelling changed backend inputs."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest
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


def payload(tmp_path, location=None):
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV
    from scripts.bundles.native_prepared import publish_prepared
    from pm.store import current_target
    out=location or tmp_path/'cold'; out.mkdir(parents=True); code=out/'hermes-agent'; code.mkdir()
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
        shutil.rmtree(out)
        target=out; restored=restore_cache(cache,'key',target,tmp_path,'b'*40)
        inputs=load_prepared(restored)
        assert inputs.python==target/'tools/python/python.exe' and inputs.ref=='b'*40
        assert (inputs.code/'entry.py').read_text()=='unchanged backend'
    (cache/'runtime.tar').write_bytes(b'corrupt'); shutil.rmtree(out); target=out; target.mkdir()
    assert restore_cache(cache,'key',target,tmp_path,'b'*40) is None
    assert not list(target.iterdir())


def test_wrong_key_and_nonempty_output_are_preserved(tmp_path):
    out,prepared=payload(tmp_path); cache=tmp_path/'native/runtime/key'; publish_cache(cache,'key',out,prepared)
    assert restore_cache(cache,'wrong',tmp_path/'new',tmp_path,'b'*40) is None
    target=tmp_path/'dirty'; target.mkdir(); (target/'user.txt').write_text('preserve')
    assert restore_cache(cache,'key',target,tmp_path,'b'*40) is None
    assert (target/'user.txt').read_text()=='preserve'


def test_archive_paths_are_checked_even_with_a_matching_checksum(tmp_path):
    import hashlib
    import io
    import tarfile
    import pytest
    out,prepared=payload(tmp_path); cache=tmp_path/'native/runtime/key'
    publish_cache(cache,'key',out,prepared)
    archive=cache/'runtime.tar'
    # A correctly hashed but invalid archive must not write outside the output.
    with tarfile.open(archive,'a') as bundle:
        member=tarfile.TarInfo('../escaped.txt'); member.size=1
        bundle.addfile(member,io.BytesIO(b'x'))
    receipt=json.loads((cache/'receipt.json').read_text())
    receipt['archiveSha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    (cache/'receipt.json').write_text(json.dumps(receipt))
    shutil.rmtree(out); target=out; target.mkdir()
    with pytest.raises(tarfile.FilterError):restore_cache(cache,'key',target,tmp_path,'b'*40)
    assert not (tmp_path/'escaped.txt').exists()


def test_cached_metadata_is_inside_the_hashed_archive(tmp_path):
    out,prepared=payload(tmp_path); cache=tmp_path/'native/runtime/key'
    publish_cache(cache,'key',out,prepared)
    receipt=json.loads((cache/'receipt.json').read_text())
    assert 'prepared' not in receipt


def test_failed_extraction_cannot_become_a_partial_cache_miss(tmp_path):
    import hashlib
    import tarfile
    import pytest
    out,prepared=payload(tmp_path);cache=tmp_path/'native/runtime/key';publish_cache(cache,'key',out,prepared)
    archive=cache/'runtime.tar'
    with tarfile.open(archive,'a') as bundle:
        member=tarfile.TarInfo('missing-link');member.type=tarfile.LNKTYPE;member.linkname='missing-target'
        bundle.addfile(member)
    receipt=json.loads((cache/'receipt.json').read_text());receipt['archiveSha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    (cache/'receipt.json').write_text(json.dumps(receipt))
    shutil.rmtree(out)
    with pytest.raises((ValueError,tarfile.TarError,KeyError)):
        restore_cache(cache,'key',out,tmp_path,'b'*40)


def test_native_environment_is_not_silently_relocated(tmp_path):
    out,prepared=payload(tmp_path);cache=tmp_path/'native/runtime/key';publish_cache(cache,'key',out,prepared)
    assert restore_cache(cache,'key',tmp_path/'different-location',tmp_path,'b'*40) is None
    assert not (tmp_path/'different-location').exists()


def test_retained_output_is_bound_to_seal_and_validated_before_use(tmp_path, monkeypatch):
    import tarfile
    from scripts.bundles.native_prepared import load_prepared
    out,prepared=payload(tmp_path);cache=tmp_path/'native/runtime/key';publish_cache(cache,'key',out,prepared)
    inode=(out/'hermes-agent/entry.py').stat().st_ino
    def extraction_forbidden(*a,**kw):raise AssertionError('retained bytes must not be recopied')
    monkeypatch.setattr(tarfile.TarFile,'extractall',extraction_forbidden)
    restored=restore_cache(cache,'key',out,tmp_path,'b'*40)
    assert restored is not None
    inputs=load_prepared(restored)
    assert inputs.ref=='b'*40
    assert (out/'hermes-agent/entry.py').stat().st_ino==inode
    with tarfile.open(cache/'runtime.tar') as archive:
        assert archive.next().name=='.runtime-prepared.json'


def test_corrupt_retained_bytes_revoke_retention_without_admission(tmp_path):
    import os
    import pytest
    from scripts.bundles.native_prepared import load_prepared
    from scripts.bundles.runtime_cache import retained_runtime
    out,prepared=payload(tmp_path);cache=tmp_path/'native/runtime/key';publish_cache(cache,'key',out,prepared)
    code=out/'hermes-agent/entry.py';original=code.stat()
    code.write_bytes(b'X'*original.st_size);os.utime(code,ns=(original.st_atime_ns,original.st_mtime_ns))
    assert retained_runtime(out,tmp_path,'key')
    restored=restore_cache(cache,'key',out,tmp_path,'b'*40)
    with pytest.raises(ValueError,match='prepared bytes changed'):load_prepared(restored)
    assert not retained_runtime(out,tmp_path)
    assert code.read_bytes()==b'X'*original.st_size


def test_key_miss_discards_only_recognized_owned_retained_output(tmp_path,monkeypatch):
    import pytest
    from scripts.bundles import native
    out,prepared=payload(tmp_path);cache=tmp_path/'native/runtime/key';publish_cache(cache,'key',out,prepared)
    class ColdReached(Exception):pass
    def cold(**kwargs):
        assert not list(out.iterdir())
        raise ColdReached
    monkeypatch.setattr(native,'_prepare_native',cold)
    with pytest.raises(ColdReached):
        native.prepare_native(out=out,ref='b'*40,source=tmp_path,cache=tmp_path/'wheels',runtime_cache=(cache,'changed-key'))
    (out/'foreign.txt').write_text('preserve')
    with pytest.raises(ValueError,match='nonempty'):
        native.prepare_native(out=out,ref='b'*40,source=tmp_path,cache=tmp_path/'wheels',runtime_cache=(cache,'key'))
    assert (out/'foreign.txt').read_text()=='preserve'


@pytest.mark.parametrize('guarded,corrupt',[(True,False),(False,False),(True,True)])
def test_clean_preserves_only_guarded_owned_runtime_and_restores_receipt(tmp_path,monkeypatch,guarded,corrupt):
    from types import SimpleNamespace
    from scripts.bundles.desktop_prepare import clean_outputs
    from scripts.bundles.native_prepared import load_prepared
    source=tmp_path/'source';source.mkdir()
    out=source/'apps/desktop/build/agent-payload'
    out,prepared=payload(source,out)
    cache=source/'.cache/desktop-inputs';entry=cache/'native/runtime/key';publish_cache(entry,'key',out,prepared)
    for relative in ['apps/desktop/build/products','apps/desktop/dist','apps/desktop/release']:
        folder=source/relative;folder.mkdir();(folder/'stale').write_text('old build')
    work=source/'.build/desktop-job';work.mkdir(parents=True);(work/'.desktop-preparation').write_text(str(source))
    (source/'user.txt').write_text('preserve')
    Path(str(source)+'.distribution.lock').touch()
    if guarded:monkeypatch.setenv('HERMES_DESKTOP_SERIAL_CACHE',str(cache))
    else:monkeypatch.delenv('HERMES_DESKTOP_SERIAL_CACHE',raising=False)
    inode=(out/'hermes-agent/entry.py').stat().st_ino
    if corrupt:
        (out/'hermes-agent/entry.py').write_text('bad retained bytes')
        with pytest.raises(ValueError,match='prepared bytes changed'):
            load_prepared(restore_cache(entry,'key',out,source,'b'*40))
    request=SimpleNamespace(source=source,work=work,cache=cache,target='win32-x64')
    clean_outputs(request)
    assert (source/'user.txt').read_text()=='preserve'
    assert (entry/'runtime.tar').is_file() and not work.exists()
    assert not (source/'apps/desktop/build/products').exists()
    assert not (source/'apps/desktop/dist').exists() and not (source/'apps/desktop/release').exists()
    assert out.exists()==(guarded and not corrupt)
    restored=restore_cache(entry,'key',out,source,'b'*40)
    assert load_prepared(restored).ref=='b'*40
    assert (out/'hermes-agent/entry.py').read_text()=='unchanged backend'
    if guarded and not corrupt:assert (out/'hermes-agent/entry.py').stat().st_ino==inode


def test_input_change_restores_compatible_seal_without_dependency_acquisition(tmp_path,monkeypatch):
    from scripts.bundles import native
    from scripts.bundles.native_prepared import publish_prepared,load_prepared
    from scripts.build.inputs import AgentInputs
    out,prepared=payload(tmp_path);old=tmp_path/'native/runtime/old';new=tmp_path/'native/runtime/new'
    data=json.loads(prepared.read_text());inputs=AgentInputs.from_dict({**data['inputs'],'ref':'b'*40})
    publish_cache(old,'old',out,prepared)
    (out/'hermes-agent/entry.py').write_text('different backend')
    prepared=publish_prepared(out,tmp_path,'b'*40,inputs);publish_cache(new,'new',out,prepared)
    def cold_forbidden(**kwargs):raise AssertionError('compatible sealed archive should be restored')
    monkeypatch.setattr(native,'_prepare_native',cold_forbidden)
    restored=native.prepare_native(out=out,ref='a'*40,source=tmp_path,cache=tmp_path/'wheels',runtime_cache=(old,'old'))
    assert load_prepared(restored).ref=='a'*40
    assert (out/'hermes-agent/entry.py').read_text()=='unchanged backend'
