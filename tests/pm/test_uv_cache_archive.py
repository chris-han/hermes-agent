"""Windows cache archives preserve offline seeds without thousands of MSIX entries."""
import shutil
import stat
import zipfile

import pytest

import hermes_constants
from pm import packages, paths
from scripts.bundles import native


@pytest.fixture
def seed(tmp_path, monkeypatch):
    payload = tmp_path / 'payload'
    cache = payload / 'uv-cache'
    (cache / 'archive-v0/bucket').mkdir(parents=True)
    (cache / 'archive-v0/bucket/module.py').write_bytes(b'COMPLETE = True\n')
    (cache / 'empty').mkdir()
    home = tmp_path / 'home'
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: home)
    monkeypatch.setattr(paths, 'store_root', lambda: payload / 'tools')
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'uv.lock').write_text('version = 1\n')
    monkeypatch.setattr(paths, 'repo_root', lambda: repo)
    return cache, home / 'cache/uv'


def test_compact_seed_roundtrip_and_warm_no_extraction(seed, monkeypatch):
    cache, machine = seed
    native.compact_uv_cache(cache)
    assert [p.name for p in cache.iterdir()] == ['seed.zip']
    with zipfile.ZipFile(cache / 'seed.zip') as z:
        assert z.testzip() is None
        assert z.read('archive-v0/bucket/module.py') == b'COMPLETE = True\n'
    assert packages.uv_cache_dir() == machine
    assert (machine / 'archive-v0/bucket/module.py').read_bytes() == b'COMPLETE = True\n'
    assert (machine / 'empty').is_dir()
    assert (machine / '.seeded').is_file()
    monkeypatch.setattr(zipfile, 'ZipFile', lambda *a, **k: pytest.fail('warm seed must not reopen ZIP'))
    packages.uv_cache_dir()


def test_archive_atomic_publication_and_retry(seed, monkeypatch):
    cache, machine = seed
    native.compact_uv_cache(cache)
    target = machine / 'archive-v0/bucket/module.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old')
    real = shutil.copyfileobj
    attempts = []

    def interrupted(src, dst, *args, **kwargs):
        assert target.read_bytes() == b'old'
        attempts.append(1)
        if len(attempts) == 1:
            dst.write(b'partial')
            raise OSError('interrupted seed')
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, 'copyfileobj', interrupted)
    packages.uv_cache_dir()
    assert target.read_bytes() == b'old'
    assert not (machine / '.seeded').exists()
    assert not list(machine.rglob('.seed-*'))
    packages.uv_cache_dir()
    assert target.read_bytes() == b'COMPLETE = True\n'
    assert (machine / '.seeded').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/escape', 'dir\\escape', '.seeded', 'a/../../escape', 'file:stream', 'dir/CON', '.SEEDED', 'a//b', 'a/./b', 'a?b'])
def test_unsafe_archive_refused_before_publication(seed, name):
    from pm.package import InstallError
    cache, machine = seed
    shutil.rmtree(cache)
    cache.mkdir()
    with zipfile.ZipFile(cache / 'seed.zip', 'w') as z:
        z.writestr('valid', b'good')
        z.writestr(name, b'bad')
    with pytest.raises(InstallError, match='cache'):
        packages.uv_cache_dir()
    assert not (machine / 'valid').exists()
    assert not (machine / '.seeded').exists()


@pytest.mark.parametrize('kind', ['duplicate', 'link', 'corrupt'])
def test_invalid_archive_not_marked_seeded(seed, kind):
    from pm.package import InstallError
    cache, machine = seed
    shutil.rmtree(cache)
    cache.mkdir()
    archive = cache / 'seed.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('item', b'original')
        if kind == 'duplicate':
            z.writestr('ITEM', b'duplicate')
        elif kind == 'link':
            info = zipfile.ZipInfo('link')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, '../escape')
    if kind == 'corrupt':
        archive.write_bytes(archive.read_bytes().replace(b'original', b'corrupt!'))
    with pytest.raises(InstallError, match='cache'):
        packages.uv_cache_dir()
    assert not (machine / '.seeded').exists()
    assert not list(machine.rglob('.seed-*'))


def test_changed_lock_merges_new_archive(seed):
    cache, machine = seed
    native.compact_uv_cache(cache)
    packages.uv_cache_dir()
    with zipfile.ZipFile(cache / 'seed.zip', 'w') as z:
        z.writestr('archive-v0/bucket/new.py', b'new pin')
    (paths.repo_root() / 'uv.lock').write_text('version = 2\n')
    packages.uv_cache_dir()
    assert (machine / 'archive-v0/bucket/new.py').read_bytes() == b'new pin'


def test_shipped_environment_selection_does_not_extract(seed, monkeypatch):
    import json
    cache, machine = seed
    native.compact_uv_cache(cache)
    payload = cache.parent
    (payload / 'venv').mkdir()
    (payload / 'hermes-agent').mkdir()
    (payload / 'manifest.json').write_text(json.dumps({'repo': 'hermes-agent', 'venv': 'venv'}))
    monkeypatch.setattr(paths, 'repo_root', lambda: payload / 'hermes-agent')
    assert packages.Venv().venv_dir() == payload / 'venv'
    assert not machine.exists()


def test_existing_destination_link_cannot_escape(seed, tmp_path):
    from pm.package import InstallError
    cache, machine = seed
    native.compact_uv_cache(cache)
    outside = tmp_path / 'outside'
    outside.mkdir()
    machine.mkdir(parents=True)
    (machine / 'archive-v0').symlink_to(outside, target_is_directory=True)
    with pytest.raises(InstallError, match='cache'):
        packages.uv_cache_dir()
    assert not list(outside.iterdir())
    assert not (machine / '.seeded').exists()


def test_compaction_preserves_loose_seed_on_archive_failure(seed, monkeypatch):
    cache, _ = seed
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda self: 'bad-member')
    with pytest.raises(ValueError, match='integrity'):
        native.compact_uv_cache(cache)
    assert (cache / 'archive-v0/bucket/module.py').read_bytes() == b'COMPLETE = True\n'
    assert not (cache / 'seed.zip').exists()
    assert not (cache / '.seed.zip.tmp').exists()


def test_corrected_archive_for_same_lock_is_seeded(seed):
    cache, machine = seed
    native.compact_uv_cache(cache)
    packages.uv_cache_dir()
    previous = (machine / '.seeded').read_text()
    with zipfile.ZipFile(cache / 'seed.zip', 'w') as z:
        z.writestr('archive-v0/restored/package.py', b'required wheel restored')
    packages.uv_cache_dir()
    assert (machine / 'archive-v0/restored/package.py').read_bytes() == b'required wheel restored'
    assert (machine / '.seeded').read_text() != previous


def test_legacy_payload_marker_cannot_suppress_corrected_archive(seed):
    cache, machine = seed
    native.compact_uv_cache(cache)
    machine.mkdir(parents=True)
    lock = packages._uv_lock_digest(paths.repo_root() / 'uv.lock').hex()
    (machine / '.seeded').write_text(lock + '+payload')
    packages.uv_cache_dir()
    assert (machine / 'archive-v0/bucket/module.py').read_bytes() == b'COMPLETE = True\n'
    assert '+archive:' in (machine / '.seeded').read_text()


def test_bounded_seed_archives_preserve_complete_offline_cache(seed):
    import json
    cache, machine = seed
    for n in range(4):
        (cache / f'wheel-{n}').write_bytes(bytes(range(256)) * 4)
    native.compact_uv_cache(cache, max_archive_bytes=1500)
    manifest = json.loads((cache / 'seed-manifest.json').read_text())
    assert len(manifest['archives']) > 1
    assert all((cache / row['name']).stat().st_size <= 1500 for row in manifest['archives'])
    packages.uv_cache_dir()
    for n in range(4):
        assert (machine / f'wheel-{n}').read_bytes() == bytes(range(256)) * 4
    assert (machine / 'archive-v0/bucket/module.py').read_bytes() == b'COMPLETE = True\n'
    assert not (machine / 'seed-manifest.json').exists()
    assert not list(machine.glob('seed-*.zip'))


def test_missing_seed_segment_refuses_completion(seed):
    import json
    from pm.package import InstallError
    cache, machine = seed
    for n in range(3):
        (cache / f'wheel-{n}').write_bytes(bytes(range(256)) * 4)
    native.compact_uv_cache(cache, max_archive_bytes=1500)
    manifest = json.loads((cache / 'seed-manifest.json').read_text())
    (cache / manifest['archives'][-1]['name']).unlink()
    with pytest.raises(InstallError, match='cache'):
        packages.uv_cache_dir()
    assert not (machine / '.seeded').exists()


def test_indivisible_seed_member_preserves_loose_cache(seed):
    cache, _ = seed
    (cache / 'large').write_bytes(bytes(range(256)) * 4)
    with pytest.raises(ValueError, match='exceeds'):
        native.compact_uv_cache(cache, max_archive_bytes=200)
    assert (cache / 'large').exists()
    assert not (cache / 'seed-manifest.json').exists()


@pytest.mark.parametrize('damage', ['digest', 'name', 'extra', 'schema'])
def test_damaged_archive_set_refuses_before_seed(seed, damage):
    import json
    from pm.package import InstallError
    cache, machine = seed
    for n in range(2):
        (cache / f'wheel-{n}').write_bytes(bytes(range(256)) * 4)
    native.compact_uv_cache(cache, max_archive_bytes=1500)
    file = cache / 'seed-manifest.json'
    data = json.loads(file.read_text())
    if damage == 'digest':
        data['archives'][0]['sha256'] = '0' * 64
    elif damage == 'name':
        data['archives'][0]['name'] = '../escape.zip'
    elif damage == 'schema':
        data['schema'] = 2
    else:
        (cache / 'seed-9999.zip').write_bytes(b'unlisted')
    file.write_text(json.dumps(data))
    with pytest.raises(InstallError, match='cache'):
        packages.uv_cache_dir()
    assert not machine.exists()
