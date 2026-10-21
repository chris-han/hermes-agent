"""Lock pruning must use the same distribution-name identity as the resolver."""
import pytest

from pm.uv_cache_prune import prune_uv_cache_to_lock


@pytest.mark.parametrize(('locked', 'installed', 'indexed'), [
    ('ruamel-yaml', 'ruamel.yaml', 'ruamel-yaml'),
    ('jaraco.functools', 'jaraco_functools', 'jaraco-functools'),
    ('backports-tarfile', 'backports__tarfile', 'backports.tarfile'),
    ('package-x', 'package_x', 'package---x'),
])
def test_pruning_preserves_equivalent_names_and_still_removes_stale(tmp_path, locked, installed, indexed):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'uv.lock').write_text(f'[[package]]\nname="{locked}"\n')
    cache = tmp_path / 'cache'
    bucket = cache / 'archive-v0/required'
    metadata = bucket / f'{installed}-1.0.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text(f'Name: {installed}\nVersion: 1.0\n')
    index = cache / 'wheels-v6/pypi' / indexed
    index.mkdir(parents=True)
    (index / 'pointer').write_text('required')
    stale = cache / 'archive-v0/stale/obsolete.pkg-1.0.dist-info'
    stale.mkdir(parents=True)
    assert prune_uv_cache_to_lock(cache, repo) == 1
    assert metadata.is_dir()
    assert index.is_dir()
    assert not stale.parent.exists()


def test_declared_core_build_backend_requirements_are_not_pruned(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'uv.lock').write_text('[[package]]\nname="core-runtime"\n')
    (repo / 'pyproject.toml').write_text(
        '[build-system]\nrequires=["setuptools==83.0.0", "wheel>=0.46,<1"]\n')
    cache = tmp_path / 'cache'
    for name in ['setuptools', 'wheel', 'unrelated']:
        (cache / f'archive-v0/{name}/{name}-1.0.dist-info').mkdir(parents=True)
        (cache / f'wheels-v6/pypi/{name}').mkdir(parents=True)
    assert prune_uv_cache_to_lock(cache, repo) == 2
    for name in ['setuptools', 'wheel']:
        assert (cache / f'archive-v0/{name}').is_dir()
        assert (cache / f'wheels-v6/pypi/{name}').is_dir()
    assert not (cache / 'archive-v0/unrelated').exists()
