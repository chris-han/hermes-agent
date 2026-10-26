"""The retired desktop manager has no HTTP interface in the real application."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(('method', 'path'), [('GET', '/api/local-models/status'), ('GET', '/api/local-models/hardware'), ('GET', '/api/local-models/catalog'), ('POST', '/api/local-models/runtime/install'), ('POST', '/api/local-models/download'), ('POST', '/api/local-models/download/pause'), ('POST', '/api/local-models/download/resume'), ('DELETE', '/api/local-models/models/absent'), ('POST', '/api/local-models/quickstart'), ('POST', '/api/local-models/server'), ('POST', '/api/local-models/eject'), ('POST', '/api/local-models/activate'), ('GET', '/api/local-models/jobs'), ('GET', '/api/local-models/jobs/absent'), ('GET', '/api/local-models/search'), ('GET', '/api/local-models/search/files'), ('POST', '/api/local-models/download-browsed'), ('POST', '/api/local-models/sideload')])
def test_retired_model_routes_are_not_mounted(method, path):
    from hermes_cli import web_server
    assert not any(getattr(route, 'path', '').startswith('/api/local-models') for route in web_server.app.routes)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.request(method, path)
    assert response.status_code in (404, 405)
