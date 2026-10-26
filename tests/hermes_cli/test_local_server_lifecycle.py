"""Shared terminal status reads the live managed runtime without GUI HTTP exposure."""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from tests.hermes_cli._local_runtime_handlers import client as handler_client
    return handler_client()

def test_status_reports_loaded_models_from_live_router(client, monkeypatch):
    """Round-11 regression: the loaded-models read inside the status route
    raised NameError (missing json import), the blanket except swallowed it,
    and {} shipped as truth — 'Not in memory' on a machine with 30 GB of
    VRAM in use. This test exercises the REAL route against a stub router
    and demands the loaded set comes through."""
    import http.server
    import json as _json
    import threading

    class _Router(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps({"data": [
                {"id": "m-loaded", "status": {"value": "loaded"}},
                {"id": "m-loading", "status": {"value": "loading"}},
                {"id": "m-cold", "status": {"value": "unloaded"}},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Router)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        # Patch the ROUTE's binding: local_models binds _state_endpoint via
        # from-import at module load, so patching the endpoint module's
        # attribute never reaches the name the route actually calls.
        monkeypatch.setattr(
            "hermes_cli.web_routers.local_models._state_endpoint",
            lambda: {"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "k"})
        payload = client.get("/api/local-models/status").json()
        assert payload["server_running"] is True
        assert payload["loaded_models"] == {"m-loaded": "loaded", "m-loading": "loading"}
    finally:
        server.shutdown()
