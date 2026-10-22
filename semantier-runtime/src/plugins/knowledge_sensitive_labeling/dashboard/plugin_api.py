from __future__ import annotations

from typing import Any

from .. import schemas, service

CONTRACT_VERSION = schemas.DASHBOARD_CONTRACT_VERSION
PLUGIN_VERSION = "0.1.0"

ROUTES = (
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/classifications",
        "handler": "create_classification",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/classifications/{run_id}",
        "handler": "get_classification",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/conflicts/{run_id}",
        "handler": "get_conflicts",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/reports",
        "handler": "create_report",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/reports/{report_id}",
        "handler": "get_report",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/derivatives",
        "handler": "create_derivative",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/derivatives/{artifact_id}",
        "handler": "get_derivative",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/knowledge-sensitive-labeling/v1/replay/{run_id}/verify",
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
                "raw_authority_rows": False,
                "sensitive_values": "minimized",
                "source_format_parsing": False,
            },
        }
    )


def create_classification(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope(service.classify_knowledge_input(args=payload, ctx=ctx))


def get_classification(*, ctx: Any, run_id: str) -> dict[str, Any]:
    return _envelope(service.get_classification_evidence(args={"run_id": run_id}, ctx=ctx))


def get_conflicts(*, ctx: Any, run_id: str) -> dict[str, Any]:
    evidence = service.get_classification_evidence(args={"run_id": run_id}, ctx=ctx)["evidence"]
    return _envelope({"status": "ok", "conflicts": evidence["runtime_conflicts"]})


def create_report(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope(service.generate_knowledge_labeling_report(args=payload, ctx=ctx))


def get_report(*, ctx: Any, report_id: str) -> dict[str, Any]:
    return _envelope(service.generate_knowledge_labeling_report(args={"run_id": report_id}, ctx=ctx))


def create_derivative(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope(service.generate_knowledge_labeling_derivative(args=payload, ctx=ctx))


def get_derivative(*, ctx: Any, artifact_id: str) -> dict[str, Any]:
    return _envelope(service.generate_knowledge_labeling_derivative(args={"run_id": artifact_id}, ctx=ctx))


def verify_replay(*, ctx: Any, run_id: str) -> dict[str, Any]:
    return _envelope(service.verify_knowledge_labeling_replay(args={"run_id": run_id}, ctx=ctx))
