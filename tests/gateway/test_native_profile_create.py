"""A dedicated same-user grant creates profiles without widening profile HTTP grants."""
import asyncio

import httpx
import pytest

from tests.gateway.test_native_http_auth import daemon, headers, reject_ws, ticket



@pytest.mark.platforms("linux")
def test_native_profile_delete_uses_same_user_route_bound_grant(tmp_path):
    """A confirmed Desktop delete needs a one-use host operation, not a sibling profile ticket."""
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor["api_origin"], trust_env=False, timeout=20) as client:
        target = home / "profiles" / "other"
        assert target.is_dir()
        assert client.delete("/api/profiles/other", headers=headers(ticket(home, descriptor))).status_code == 403
        legacy = home / "profiles" / "semantier-local"
        legacy.mkdir()
        (legacy / "config.yaml").write_text("{}")
        blocked = client.delete("/api/profiles/semantier-local",
                                headers=headers(ticket(home, descriptor, "native-profile-delete")))
        assert blocked.status_code == 409
        assert legacy.exists()
        query_blocked = client.delete("/api/profiles/other?profile=default",
                                      headers=headers(ticket(home, descriptor, "native-profile-delete")))
        assert query_blocked.status_code == 403
        grant = ticket(home, descriptor, "native-profile-delete")
        assert client.get("/api/config", headers=headers(grant)).status_code == 401
        assert client.delete("/api/profiles/default", headers=headers(grant)).status_code == 403
        grant = ticket(home, descriptor, "native-profile-delete")
        assert client.delete("/api/profiles/other?profile=other", headers=headers(grant)).status_code == 403
        assert target.exists()
        grant = ticket(home, descriptor, "native-profile-delete")
        assert client.delete("/api/profiles/other", headers={**headers(grant), "Origin": "null"}).status_code == 401
        reply = client.delete("/api/profiles/other", headers=headers(grant))
        assert reply.status_code == 200, reply.text
        assert not target.exists()
        assert client.delete("/api/profiles/current", headers=headers(grant)).status_code == 401
        assert (home / "config.yaml").exists()

@pytest.mark.platforms('linux')
def test_native_profile_create_grant_is_route_bound_and_single_use(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False, timeout=20) as client:
        body = {'name': 'gui-created', 'clone_from': 'default', 'soul': 'A GUI-created persona'}
        assert client.post('/api/profiles', json=body,
            headers=headers(ticket(home, descriptor))).status_code == 403
        assert not (home / 'profiles' / 'gui-created').exists()
        assert client.get('/api/config?profile=other',
            headers=headers(ticket(home, descriptor))).status_code == 403
        grant = ticket(home, descriptor, 'native-profile-create')
        for method, path in [('GET', '/api/config'), ('GET', '/api/profiles'),
                             ('POST', '/api/profiles/import'), ('PATCH', '/api/profiles/other'),
                             ('DELETE', '/api/profiles/other')]:
            response = client.request(method, path, json=body, headers=headers(grant))
            assert response.status_code == 401, (method, path, response.text)
        asyncio.run(reject_ws(descriptor['api_origin'], grant))
        for origin in ('', 'https://evil.example'):
            response = client.post('/api/profiles', json=body, headers={**headers(grant), 'Origin': origin})
            assert response.status_code == 401
        created = client.post('/api/profiles', json=body, headers=headers(grant))
        assert created.status_code == 200, created.text
        assert created.json()['name'] == 'gui-created'
        assert (home / 'profiles' / 'gui-created' / 'config.yaml').is_file()
        assert (home / 'profiles' / 'gui-created' / 'SOUL.md').read_text() == body['soul']
        assert created.json()['soul_set'] is True
        listed = client.get('/api/profiles', headers=headers(ticket(home, descriptor)))
        assert 'gui-created' in {p['name'] for p in listed.json()['profiles']}
        assert client.post('/api/profiles', json={**body, 'name': 'replay'},
            headers=headers(grant)).status_code == 401
        assert not (home / 'profiles' / 'replay').exists()


@pytest.mark.platforms('linux')
def test_owner_delete_acknowledges_real_legacy_migration_without_widening_profile_grants(tmp_path, monkeypatch):
    from hermes_cli.profiles_desktop import resolve_desktop_startup_profile
    def prepare(home):
        legacy = home / 'profiles/semantier-local'
        legacy.mkdir()
        (legacy / 'config.yaml').write_text('{}')
        (legacy / '.env').write_text('KEEP=synthetic-fixture\n')
        with monkeypatch.context() as scoped:
            scoped.setenv('HERMES_HOME', str(home))
            assert resolve_desktop_startup_profile(home, 'semantier-local')['profile'] == 'default'
    with daemon(tmp_path, prepare=prepare) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False, timeout=20) as client:
        assert client.delete('/api/profiles/semantier-local', headers=headers(ticket(home, descriptor))).status_code == 403
        for _ in range(2):
            response = client.delete('/api/profiles/semantier-local', headers=headers(ticket(home, descriptor, 'native-profile-delete')))
            assert response.status_code == 200, response.text
            assert response.json()['migrated_to'] == 'default'
        assert not (home / 'profiles/semantier-local').exists()
        assert (home / '.desktop-default-migration/legacy-home/.env').read_text() == 'KEEP=synthetic-fixture\n'
