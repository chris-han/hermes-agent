"""Unsigned integrity fixtures never establish publisher authentication."""
import hashlib
import json
from datetime import UTC, datetime

import pytest

from tools.skills_hub_semantier import SemantierSkillSource, RegistryError, manifest_digest


@pytest.fixture
def registry(tmp_path):
    files = {'SKILL.md': b'---\nname: semantier--demo\ndescription: Fixture skill\n---\nRead references/help.md', 'references/help.md': b'Fixture help'}
    manifest = {p: {'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)} for p, b in files.items()}
    release = {'version': '1.0.0', 'source_commit': 'a' * 40, 'released_at': '2026-11-19T00:00:00Z', 'compatibility': {}, 'manifest': manifest}
    payload = {'schema': 1, 'registry_id': 'semantier', 'sequence': 1, 'issued_at': '2026-11-19T00:00:00Z', 'expires_at': '2026-12-01T00:00:00Z', 'publisher': {'id': 'semantier', 'name': 'Semantier', 'repository': 'https://example.org/source', 'contact': 'fixture@example.org'}, 'skills': [{'slug': 'demo', 'name': 'Demo', 'description': 'Fixture skill', 'tags': ['test'], 'category': 'test', 'license': 'MIT', 'homepage': 'https://example.org', 'source_repository': 'https://example.org/source', 'source_commit': 'a' * 40, 'source_path': 'skills/demo', 'type': 'skill', 'latest_version': '1.0.0', 'versions': [release]}]}
    def catalog():
        return json.dumps(payload).encode()
    def read(url, limit):
        return catalog() if url.endswith('catalog.json') else files[url.split(manifest_digest(manifest) + '/')[1]]
    source = SemantierSkillSource(reader=read, cache_dir=tmp_path, now=lambda: datetime(2026, 11, 20, tzinfo=UTC))
    return source, payload, files


def test_verified_bundle_and_exact_claim(registry):
    source, payload, files = registry
    assert source.inspect('official/demo') is None
    meta = source.search('')[0]
    assert meta.identifier == 'semantier/semantier/demo'
    bundle = source.fetch(meta.identifier)
    assert bundle.files == files
    assert bundle.name == 'semantier--demo'
    assert bundle.trust_level == 'community'
    assert bundle.metadata['publisher_verified'] is False
    assert 'key_fingerprint' not in bundle.metadata
    assert bundle.metadata['integrity_verified'] is True
    assert source.current_revision(meta.identifier) == manifest_digest(payload['skills'][0]['versions'][0]['manifest'])


@pytest.mark.parametrize('mutate,code', [
    (lambda p: p.update(expires_at='2026-11-19T12:00:00Z'), 'expired'),
    (lambda p: p['publisher'].update(id='forged'), 'invalid_catalog'),
    (lambda p: p['skills'].append(p['skills'][0].copy()), 'invalid_catalog'),
    (lambda p: p['skills'][0]['versions'][0]['manifest'].update({'CON.txt': {'size': 0, 'sha256': '0'*64}}), 'invalid_catalog'),
])
def test_rejects_catalog(registry, mutate, code):
    source, payload, _ = registry
    mutate(payload)
    with pytest.raises(RegistryError) as exc:
        source.search('')
    assert exc.value.code == code


def test_tamper_and_equivocation(registry):
    source, payload, files = registry
    source.search('')
    files['references/help.md'] = b'changed'
    with pytest.raises(RegistryError, match='integrity_failed'):
        source.fetch('semantier/semantier/demo')
    payload['skills'][0]['description'] = 'changed'
    with pytest.raises(RegistryError, match='rollback_detected'):
        source.search('')


def test_native_install_update_and_failed_artifact_preserve_legacy(registry, monkeypatch):
    from tools.skills_hub import HubLockFile, _skills_dir
    from hermes_cli import skills_hub as cli
    from tools.skills_tool import _find_all_skills
    import tools.skills_hub_install as install
    source, _payload, files = registry
    monkeypatch.setattr(cli, '_sources', lambda: [source])
    monkeypatch.setattr(install, 'create_source_router', lambda: [source], raising=False)
    legacy = _skills_dir() / 'legacy'
    legacy.mkdir(parents=True)
    (legacy / 'SKILL.md').write_text('---\nname: legacy\ndescription: Legacy\n---\nLegacy content', encoding='utf-8')
    legacy_bytes = (legacy / 'SKILL.md').read_bytes()
    assert cli.do_install('semantier/semantier/demo', skip_confirm=True, invalidate_cache=False) is True
    entry = HubLockFile().get_installed('semantier--demo')
    assert entry['source'] == 'semantier'
    assert entry['metadata']['integrity_verified'] is True
    assert any(s['name'] == 'semantier--demo' for s in _find_all_skills(skip_disabled=True))
    before_lock = HubLockFile().path.read_bytes()
    target = _skills_dir() / entry['install_path'] / 'SKILL.md'
    before = target.read_bytes()
    files['SKILL.md'] = b'Tampered'
    with pytest.raises(RegistryError):
        cli.do_install('semantier/semantier/demo', force=True, skip_confirm=True, invalidate_cache=False)
    assert target.read_bytes() == before
    assert HubLockFile().path.read_bytes() == before_lock
    assert (legacy / 'SKILL.md').read_bytes() == legacy_bytes


@pytest.mark.parametrize('mutation,code', [
    (lambda s,p: p.update(sequence=0), 'invalid_catalog'),
    (lambda s,p: p['skills'][0]['versions'][0]['manifest'].update({'../escape': {'size': 0, 'sha256': '0'*64}}), 'invalid_catalog'),
    (lambda s,p: p['skills'][0]['versions'][0]['manifest'].update({'skill.md': {'size': 0, 'sha256': '0'*64}}), 'invalid_catalog'),
    (lambda s,p: p['skills'][0]['versions'][0].update(compatibility={'runtime': 'unknown'}), 'incompatible'),
])
def test_key_path_and_contract_negatives(registry, mutation, code):
    source, payload, _ = registry
    mutation(source, payload)
    with pytest.raises(RegistryError) as exc:
        source.search('')
    assert exc.value.code == code


@pytest.mark.parametrize('field,value,code', [('schema', 2, 'unsupported_schema'), ('signature_b64', 'obsolete', 'invalid_catalog'), ('key_id', 'obsolete', 'invalid_catalog')])
def test_catalog_contract_tampering(registry, field, value, code):
    source, _, _ = registry
    reader = source.reader
    def tamper(url, limit):
        env = json.loads(reader(url, limit))
        env[field] = value
        return json.dumps(env).encode()
    source.reader = tamper
    with pytest.raises(RegistryError) as exc:
        source.search('')
    assert exc.value.code == code


def test_rollback_and_version_replacement(registry):
    source, payload, _ = registry
    payload['sequence'] = 2
    source.search('')
    payload['sequence'] = 1
    with pytest.raises(RegistryError, match='rollback_detected'):
        source.search('')
    payload['sequence'] = 3
    payload['skills'][0]['versions'][0]['manifest']['SKILL.md']['sha256'] = '0'*64
    with pytest.raises(RegistryError, match='integrity_failed'):
        source.search('')


def test_router_packaged_in_real_process(tmp_path):
    import os
    import subprocess
    import sys
    env = {**os.environ, 'HERMES_HOME': str(tmp_path)}
    result = subprocess.run([sys.executable, '-c', "from tools.skills_hub_search import create_source_router; s=create_source_router(); assert s[0].source_id() == 'semantier'; assert {'official','hermes-index','browse-sh','url'} <= {x.source_id() for x in s}"], env=env, capture_output=True, text=True, encoding="utf-8", check=False, timeout=30)
    assert result.returncode == 0, result.stderr


def test_same_origin_update_and_local_edits(registry, monkeypatch):
    from tools.skills_hub import HubLockFile, _skills_dir
    from hermes_cli import skills_hub as cli
    import tools.skills_hub_search as search
    source, payload, files = registry
    monkeypatch.setattr(cli, '_sources', lambda: [source])
    monkeypatch.setattr(search, 'create_source_router', lambda **kw: [source])
    ident = 'semantier/semantier/demo'
    assert cli.do_install(ident, skip_confirm=True, invalidate_cache=False) is True
    entry = HubLockFile().get_installed('semantier--demo')
    target = _skills_dir() / entry['install_path'] / 'SKILL.md'
    files['SKILL.md'] += b'\nNew fixture version'
    import copy
    release = copy.deepcopy(payload['skills'][0]['versions'][0])
    release.update(version='2.0.0', released_at='2026-11-20T00:00:00Z')
    release['manifest'] = {p: {'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)} for p,b in files.items()}
    payload['sequence'] = 2
    payload['skills'][0]['versions'].append(release)
    payload['skills'][0]['latest_version'] = '2.0.0'
    # Reader returns the prepared immutable B file set; installed A remains on disk.
    def read_b(url, limit):
        if url.endswith('catalog.json'):
            return catalog_reader(url, limit)
        return files[url.split(manifest_digest(release['manifest']) + '/')[1]]
    catalog_reader = source.reader
    source.reader = read_b
    target.write_bytes(target.read_bytes() + b'\nLocal edit')
    edited = target.read_bytes()
    cli.do_update()
    assert target.read_bytes() == edited
    target.write_bytes(edited.removesuffix(b'\nLocal edit'))
    assert cli.do_install(ident, force=True, skip_confirm=True, invalidate_cache=False) is True
    assert target.read_bytes() == files['SKILL.md']
    assert HubLockFile().get_installed('semantier--demo')['metadata']['version'] == '2.0.0'


def test_cross_origin_force_conflict_preserves_bytes(registry, monkeypatch):
    from tools.skills_hub import HubLockFile, _skills_dir
    from hermes_cli import skills_hub as cli
    source, _, _ = registry
    monkeypatch.setattr(cli, '_sources', lambda: [source])
    target = _skills_dir() / 'SEMANTIER--DEMO'
    target.mkdir(parents=True)
    (target/'SKILL.md').write_text('---\nname: SEMANTIER--DEMO\ndescription: Existing owner\n---\nKeep', encoding='utf-8')
    original = (target/'SKILL.md').read_bytes()
    assert cli.do_install('semantier/semantier/demo', force=True, skip_confirm=True, invalidate_cache=False) is False
    assert (target/'SKILL.md').read_bytes() == original
    assert HubLockFile().get_installed('semantier--demo') is None


def test_scanner_denial_retains_installed_version(registry, monkeypatch):
    from hermes_cli import skills_hub as cli
    from tools.skills_hub import HubLockFile, _skills_dir
    import tools.skills_guard as guard
    source, _, _ = registry
    monkeypatch.setattr(cli, '_sources', lambda: [source])
    ident = 'semantier/semantier/demo'
    assert cli.do_install(ident, skip_confirm=True, invalidate_cache=False) is True
    entry = HubLockFile().get_installed('semantier--demo')
    target = _skills_dir()/entry['install_path']/'SKILL.md'
    before, lock = target.read_bytes(), HubLockFile().path.read_bytes()
    monkeypatch.setattr(guard, 'should_allow_install', lambda *a, **kw: (False, 'Fixture high-risk denial'))
    assert cli.do_install(ident, force=True, skip_confirm=True, invalidate_cache=False) is False
    assert target.read_bytes() == before
    assert HubLockFile().path.read_bytes() == lock


def test_off_host_transport_refused_before_request():
    from tools.skills_hub_semantier import _read
    with pytest.raises(RegistryError, match='off-host'):
        _read('https://example.org/skills/v1/catalog.json', 1024)


@pytest.mark.parametrize('status,body,limit,code', [(302, b'', 100, 'unavailable'), (200, b'too long', 2, 'integrity_failed'), (500, b'failed', 100, 'unavailable')])
def test_transport_redirect_and_stream_bounds(monkeypatch, status, body, limit, code):
    import httpx
    import tools.url_safety as safety
    import tools.website_policy as policy
    from tools.skills_hub_semantier import _read, BASE
    monkeypatch.setattr(policy, 'check_website_access', lambda _: None)
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body, headers={'Location': 'http://127.0.0.1/private'}))
    monkeypatch.setattr(safety, 'create_ssrf_safe_client', lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(RegistryError) as exc:
        _read(BASE + 'catalog.json', limit)
    assert exc.value.code == code


def test_previous_signed_sequence_history_survives_upgrade(registry):
    source, payload, _ = registry
    path = source.cache_dir/'semantier-sequences.json'
    path.write_text(json.dumps({'old-key': {'sequence': 9, 'digest': 'a'*64}, 'releases': {}}))
    with pytest.raises(RegistryError, match='rollback_detected'):
        source.search('')
    payload['sequence'] = 10
    source.search('')
    state = json.loads(path.read_text())
    assert state['semantier']['sequence'] == 10
    assert 'old-key' not in state


def test_duplicate_json_member_and_obsolete_signed_envelope_rejected(registry):
    source, _, _ = registry
    source.reader = lambda *args: b'{"schema":1,"schema":1}'
    with pytest.raises(RegistryError, match='invalid_catalog'):
        source.search('')
    source.reader = lambda *args: b'{"schema":1,"key_id":"old","payload_b64":"","signature_b64":""}'
    with pytest.raises(RegistryError, match='invalid_catalog'):
        source.search('')


def test_outage_keeps_installed_skill_loadable(registry, monkeypatch):
    from hermes_cli import skills_hub as cli
    from tools.skills_hub import HubLockFile, _skills_dir
    from tools.skills_tool import _find_all_skills
    source, _, _ = registry
    monkeypatch.setattr(cli, '_sources', lambda: [source])
    assert cli.do_install('semantier/semantier/demo', skip_confirm=True, invalidate_cache=False)
    lock = HubLockFile().path.read_bytes()
    entry = HubLockFile().get_installed('semantier--demo')
    target = _skills_dir()/entry['install_path']/'SKILL.md'
    installed = target.read_bytes()
    def outage(*args):
        raise RegistryError('unavailable', 'fixture provider outage')
    source.reader = outage
    with pytest.raises(RegistryError, match='unavailable'):
        source.fetch('semantier/semantier/demo')
    assert target.read_bytes() == installed
    assert HubLockFile().path.read_bytes() == lock
    assert any(s['name'] == 'semantier--demo' for s in _find_all_skills(skip_disabled=True))
