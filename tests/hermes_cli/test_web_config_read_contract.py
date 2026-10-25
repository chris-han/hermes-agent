"""The registered Desktop's real config route must remain readable in both stores."""
import pytest


@pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
@pytest.mark.parametrize('include_defaults', [True, False])
def test_config_read_preserves_selection_and_redacts_hosted_secrets(
    tmp_path, monkeypatch, backend, include_defaults
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.web_routers.config_env import config_router

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('SEMANTIER_STORAGE_BACKEND', backend)
    cloud = {'credential_ref': 'native-pointer', 'base_url': 'https://cloud.invalid/v1'}
    if backend == 'postgresql':
        cloud.update(api_key='hosted-fixture-secret', key_env='HERMES_CUSTOM_SEMANTIER_CLOUD_TEST_API_KEY')
    atomic_config_replace(tmp_path / 'config.yaml', {
        'model': {'provider': 'semantier-cloud', 'default': 'selected-alias'},
        'providers': {'semantier-cloud': cloud}, '_private_marker': 'internal',
    })
    before = (tmp_path / 'config.yaml').read_bytes()
    token = set_hermes_home_override(tmp_path)
    try:
        app = FastAPI()
        app.include_router(config_router)
        with TestClient(app) as client:
            response = client.get('/api/config', params={'include_defaults': str(include_defaults).lower()})
        assert response.status_code == 200
        body = response.json()
        assert body['model'] == 'selected-alias'
        assert '_private_marker' not in body
        assert body['providers']['semantier-cloud']['credential_ref'] == 'native-pointer'
        if backend == 'postgresql':
            assert body['providers']['semantier-cloud']['api_key'] == '[configured]'
            assert body['providers']['semantier-cloud']['key_env'] == cloud['key_env']
    finally:
        reset_hermes_home_override(token)
    assert (tmp_path / 'config.yaml').read_bytes() == before
