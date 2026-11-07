"""GET /api/model/options must forward ``include_unconfigured`` from the query
string (default true) to ``build_model_options_payload`` instead of hardcoding
it — external API consumers need a way to request the explicit-only view, and the
unconfigured-row path must honor ``model_catalog.excluded_providers``
(#68816)."""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal app exposing only the route under test (no key configured → auth passes)."""
    app = web.Application()
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    return app


async def _captured_include_unconfigured(adapter, query: str) -> tuple[int, bool]:
    app = _create_app(adapter)
    with patch("hermes_cli.inventory.load_picker_context", return_value=object()), \
         patch("hermes_cli.inventory.build_model_options_payload",
               return_value={"providers": []}) as build:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(f"/api/model/options{query}")
    return response.status, bool(build.call_args.kwargs.get("include_unconfigured"))


@pytest.mark.asyncio
async def test_default_keeps_unconfigured_rows():
    status, flag = await _captured_include_unconfigured(_make_adapter(), "")
    assert status == 200
    assert flag is True, "default (no query param) must stay include_unconfigured=True"


@pytest.mark.asyncio
async def test_false_query_param_disables_unconfigured_rows():
    status, flag = await _captured_include_unconfigured(
        _make_adapter(), "?include_unconfigured=false")
    assert status == 200
    assert flag is False, "explicit include_unconfigured=false must reach the payload builder"


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected", "unconfigured"), [
    ("", False, True),
    ("?explicit_only=1", True, False),
    ("?explicit_only=false", False, True),
    ("?explicit_only=1&include_unconfigured=1", True, True),
])
async def test_explicit_only_is_forwarded(query, expected, unconfigured):
    app = _create_app(_make_adapter())
    with patch("hermes_cli.inventory.load_picker_context", return_value=object()), \
         patch("hermes_cli.inventory.build_model_options_payload", return_value={"providers": []}) as build:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(f"/api/model/options{query}")
    assert response.status == 200
    assert build.call_args.kwargs.get("explicit_only", False) is expected
    assert build.call_args.kwargs["include_unconfigured"] is unconfigured


@pytest.mark.asyncio
async def test_query_only_is_principal_bound_and_requires_scope():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.inventory.load_picker_context", return_value=object()), \
         patch("hermes_cli.inventory.build_model_options_payload", side_effect=AssertionError("legacy path")), \
         patch("hermes_cli.inventory.build_desktop_model_options_payload",
               return_value={"providers": [], "discovery_contract": "provider-query-v1"}) as build:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get("/api/model/options?query_only=1&query_scope=connection-a&refresh=1")
            assert response.status == 200
            assert (await response.json())["discovery_contract"] == "provider-query-v1"
            assert build.call_args.kwargs["refresh"] is True
            assert "connection-a" in build.call_args.kwargs["source_scope"]
            assert "principal:" in build.call_args.kwargs["source_scope"]
            response = await cli.get("/api/model/options?query_only=1")
            assert response.status == 400
