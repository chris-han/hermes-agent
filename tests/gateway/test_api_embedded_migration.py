"""The canonical gateway migration retains the fork's embedded execution scope."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.execution_boundary import ExecutionBoundary
from hermes_state import SessionDB
from tests.gateway.test_api_server import _create_app, _make_adapter


@pytest.mark.asyncio
async def test_embedded_database_stays_on_configured_root_with_a_canonical_owner(tmp_path):
    adapter = _make_adapter()
    adapter._semantier_embedded_boundary_required = True
    adapter._semantier_session_db_home = tmp_path / "shared"
    owner = SessionDB(db_path=tmp_path / "owner" / "state.db")
    adapter.gateway_runner = SimpleNamespace(session_authority=SimpleNamespace(db=owner))
    try:
        selected = adapter._ensure_session_db()
        assert selected is not owner
        assert selected.db_path == tmp_path / "shared" / "state.db"
        assert await adapter._ensure_session_db_async() is selected
    finally:
        adapter._close_cached_session_dbs()
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_embedded_responses_keep_workspace_scope_on_retrieval(tmp_path, monkeypatch, stream):
    adapter = _make_adapter()
    adapter._semantier_embedded_boundary_required = True
    adapter._semantier_session_db_home = tmp_path / "shared"
    boundary = ExecutionBoundary(source="api_server", user_id="user", workspace_id="workspace")
    monkeypatch.setattr(adapter, "_resolve_internal_execution_boundary", lambda *_: (boundary, None))
    run = AsyncMock(return_value=({"final_response": "scoped reply", "messages": []}, {}))
    monkeypatch.setattr(adapter, "_run_agent", run)
    client = TestClient(TestServer(_create_app(adapter)))
    await client.start_server()
    try:
        response = await client.post("/v1/responses", json={"input": "hello", "stream": stream})
        assert response.status == 200
        content = await response.text()
        assert "scoped reply" in content
        kwargs = run.call_args.kwargs
        assert kwargs["execution_boundary"] is boundary
        assert kwargs["session_id"].startswith("workspace:")
        if stream:
            events = [json.loads(line[6:]) for line in content.splitlines() if line.startswith("data: ")]
            response_id = next(event["response"]["id"] for event in events
                               if event.get("type") == "response.completed")
        else:
            response_id = json.loads(content)["id"]
        assert (await client.get(f"/v1/responses/{response_id}")).status == 200
        other = ExecutionBoundary(source="api_server", user_id="user", workspace_id="other")
        monkeypatch.setattr(adapter, "_resolve_internal_execution_boundary", lambda *_: (other, None))
        assert (await client.get(f"/v1/responses/{response_id}")).status == 404
    finally:
        await client.close()
        adapter._close_cached_session_dbs()
