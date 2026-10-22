"""Authenticated readiness projection for the qualified fictional workshop lane."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization


router = APIRouter(tags=["workshop-classroom"])

_EXPECTED_ORG = "f10_org"
_EXPECTED_WORKSPACE = "f10_workspace"
_REQUIRED_MATERIALS = {
    "resume-sample-zh.md",
    "jd-sample-zh.md",
    "prompt-versions-handout-zh.md",
    "report-template-zh.md",
    "student-records-sample-zh.md",
    "finance-briefing-sample-zh.md",
    "finance-briefing-sample-zh.pdf",
    "suspected-classified-sample-zh.md",
}


def _check(name: str, ready: bool, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "READY" if ready else "NOT_READY",
        "detail": detail,
    }


def classroom_readiness(context) -> dict:
    """Project host-prepared classroom state without granting any new authority."""
    fixture_mode = os.environ.get("SEMANTIER_F10_FIXTURE") == "1"
    identity_ready = (
        fixture_mode
        and context.organization_id == _EXPECTED_ORG
        and context.workspace_id == _EXPECTED_WORKSPACE
    )
    session_id = os.environ.get("SEMANTIER_WORKSHOP_CLASSROOM_SESSION", "")
    uploads = (
        Path(context.workspace_root) / "sessions" / session_id / "uploads"
        if session_id
        else None
    )
    observed = (
        {path.name for path in uploads.iterdir() if path.is_file()}
        if uploads is not None and uploads.is_dir()
        else set()
    )
    # Public teaching inputs are independent of the optional F10 mock lab.
    teaching_root = Path(context.workspace_root) / "demo-materials"
    materials_ready = observed == _REQUIRED_MATERIALS or all(
        (teaching_root / name).is_file() for name in _REQUIRED_MATERIALS
    )
    starter_ready = os.environ.get("SEMANTIER_WORKSHOP_STARTER_READY") == "1"
    mock_ready = os.environ.get("SEMANTIER_WORKSHOP_MOCK_LANE_READY") == "1"
    effects_disabled = (
        os.environ.get("SEMANTIER_WORKSHOP_EXTERNAL_EFFECTS") == "DISABLED"
    )

    checks = [
        _check(
            "governed_identity",
            identity_ready,
            "Disposable F10 organization/workspace identity",
        ),
        _check(
            "business_materials",
            materials_ready,
            "Teaching materials available in the current workspace",
        ),
        _check(
            "learner_starter",
            starter_ready,
            "Qualified learner starter package",
        ),
        _check(
            "mock_transport",
            mock_ready,
            "Mock gateway and sandbox calendar lane",
        ),
        _check(
            "external_effects",
            effects_disabled,
            "Live Feishu/calendar effects disabled",
        ),
    ]
    ready = all(item["status"] == "READY" for item in checks)
    return {
        "schema_version": "workshop_classroom_readiness.v1",
        "mode": "MOCK",
        "disposition": (
            "MOCK_CLASSROOM_READY" if ready else "CLASSROOM_NOT_READY"
        ),
        "external_effects": "DISABLED" if effects_disabled else "UNVERIFIED",
        "checks": checks,
        "launch": {
            "business": "/files",
            "it_labs": "/terminal",
            "decision_planning": "/workflows",
        },
    }


@router.get("/api/workshop/classroom/readiness")
def workshop_classroom_readiness(request: Request):
    context = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.SELF_ONLY,
        route_hint="/api/workshop/classroom/readiness",
    )
    return classroom_readiness(context)


@router.get("/api/workshop/classroom/stream")
def workshop_classroom_stream(request: Request):
    context = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint="/api/workshop/classroom/stream",
    )
    readiness = classroom_readiness(context)
    if readiness["disposition"] != "MOCK_CLASSROOM_READY":
        raise HTTPException(status_code=409, detail=readiness)

    session_id = os.environ.get("SEMANTIER_WORKSHOP_CLASSROOM_SESSION", "")
    if not session_id or Path(session_id).name != session_id:
        raise HTTPException(status_code=409, detail="WORKSHOP_STREAM_SESSION_INVALID")
    stream_path = (
        Path(context.workspace_root)
        / "sessions"
        / session_id
        / "artifacts"
        / "workshop-stream"
        / "stream.v1.json"
    )
    try:
        stream_path.resolve().relative_to(Path(context.workspace_root).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="WORKSHOP_STREAM_PATH_INVALID") from exc
    if not stream_path.is_file():
        raise HTTPException(status_code=409, detail="WORKSHOP_STREAM_NOT_READY")

    try:
        payload = json.loads(stream_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="WORKSHOP_STREAM_INVALID") from exc
    if (
        payload.get("schema_version") != "workshop_material_stream.v1"
        or payload.get("presentation_only") is not True
    ):
        raise HTTPException(status_code=409, detail="WORKSHOP_STREAM_CONTRACT_MISMATCH")
    return payload


@router.post("/api/workshop/classroom/start")
def start_workshop_classroom(request: Request):
    context = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_ADMIN,
        route_hint="/api/workshop/classroom/start",
    )
    readiness = classroom_readiness(context)
    if readiness["disposition"] != "MOCK_CLASSROOM_READY":
        raise HTTPException(status_code=409, detail=readiness)
    return {
        **readiness,
        "started_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
