"""Shared HTTP error helpers for Semantier web API route modules."""

from __future__ import annotations

from fastapi import HTTPException


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def bad_gateway(detail: str) -> HTTPException:
    return HTTPException(status_code=502, detail=detail)


def route_auth_class_not_configured() -> HTTPException:
    return HTTPException(status_code=500, detail="Route auth class not configured")
