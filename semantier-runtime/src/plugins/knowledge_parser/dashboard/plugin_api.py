from __future__ import annotations

from typing import Any

from .. import schemas, service

CONTRACT_VERSION = schemas.DASHBOARD_CONTRACT_VERSION
PLUGIN_VERSION = "0.1.0"

ROUTES = (
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-parser/v1/parse",
        "handler": "parse_source",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-parser/v1/runs/{run_id}",
        "handler": "get_run",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-parser/v1/inputs/{input_id}",
        "handler": "get_input",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-parser/v1/replay/{input_id}/verify",
        "handler": "verify_replay",
        "authorization": "authenticated_tenant_member",
    },
)


def _envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "plugin_id": schemas.TOOLSET_NAME,
        "plugin_version": PLUGIN_VERSION,
        "contract_version": CONTRACT_VERSION,
        **result,
    }


def capabilities() -> dict[str, Any]:
    return _envelope(
        {
            "status": "ok",
            "capabilities": [tool["name"] for tool in schemas.TOOLS],
            "routes": list(ROUTES),
            "redaction_policy": {
                "raw_source_values": "minimized",
                "authority_rows": False,
                "parser_output_authority": False,
            },
        }
    )


def parse_source(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope(service.parse_knowledge_source(args=payload, ctx=ctx))


def get_run(*, ctx: Any, run_id: str) -> dict[str, Any]:
    return _envelope(service.get_knowledge_parse_run(args={"run_id": run_id}, ctx=ctx))


def get_input(*, ctx: Any, input_id: str) -> dict[str, Any]:
    return _envelope(service.get_canonical_classification_input(args={"input_id": input_id}, ctx=ctx))


def verify_replay(*, ctx: Any, input_id: str) -> dict[str, Any]:
    return _envelope(service.verify_knowledge_parse_replay(args={"input_id": input_id}, ctx=ctx))
