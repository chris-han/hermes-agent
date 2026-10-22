from __future__ import annotations

from fastapi import APIRouter

from fastapi import HTTPException, Request
from fastapi.responses import Response

from agents.route_policy import forwarded_upstream_allowed
from agents.webapi.deps import require_authenticated_context as _require_authenticated_context
from agents.webapi.embedded_proxy_service import _forward_embedded_request

router = APIRouter(tags=["webapi-gateway"])


@router.api_route(
    "/health",
    methods=["GET", "HEAD"],
)
async def health(request: Request) -> Response:
    return await _forward_embedded_request(request, "/health")


@router.api_route(
    "/health/detailed",
    methods=["GET", "HEAD"],
)
async def health_detailed(request: Request) -> Response:
    return await _forward_embedded_request(request, "/health/detailed")


@router.api_route(
    "/v1/{upstream_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_v1(request: Request, upstream_path: str) -> Response:
    target = f"/v1/{upstream_path}"
    if not forwarded_upstream_allowed(target):
        raise HTTPException(status_code=404, detail="Not found")
    _require_authenticated_context(request)
    return await _forward_embedded_request(request, target)


@router.api_route(
    "/api/{upstream_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_api(request: Request, upstream_path: str) -> Response:
    target = f"/api/{upstream_path}"
    if not forwarded_upstream_allowed(target):
        raise HTTPException(status_code=404, detail="Not found")
    _require_authenticated_context(request)
    return await _forward_embedded_request(request, target)
