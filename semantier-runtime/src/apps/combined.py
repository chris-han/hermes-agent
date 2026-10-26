# ---------------------------------------------------------------------------
# apps/combined.py — canonical combined Semantier HTTP application
#
# Owns the FastAPI app instance and exposes execute() for programmatic use.
# Composes the existing Cloud, runtime and EOS HTTP routers and lifecycles.
# CLI and Hermes gateway startup are exposed from agents.launcher.
# ---------------------------------------------------------------------------

import logging
import os
from contextlib import asynccontextmanager

from eos.runtime_entry import execute_with_governance

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from eos.api import router as eos_router
    from agents.semantic_completion_api import router as semantic_completion_router
    from agents.webapi_gateway import router as webapi_gateway_router

    PUBLIC_INGRESS_ROUTERS = (
        eos_router,
        semantic_completion_router,
        webapi_gateway_router,
    )

    def _cron_ticker_enabled() -> bool:
        value = os.environ.get("SEMANTIER_ENABLE_CRON_TICKER", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _embedded_gateway_enabled() -> bool:
        value = os.environ.get("SEMANTIER_RUNTIME_EMBED_GATEWAY", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cron_service = None
        gateway_service = None
        if _cron_ticker_enabled():
            from agents.hermes_cron_service import get_hermes_cron_ticker_service

            cron_service = get_hermes_cron_ticker_service()
            cron_service.start()
        if _embedded_gateway_enabled():
            from agents.hermes_embedded_gateway import (
                get_embedded_hermes_gateway_service,
            )

            gateway_service = get_embedded_hermes_gateway_service()
            await gateway_service.ensure_started()
        from agents.model_gateway.runtime import start_gateway, stop_gateway

        try:
            await start_gateway(_app)
            from agents.client_operations.runtime import start as start_client_operations
            await start_client_operations(_app)
            yield
        finally:
            from agents.client_operations.runtime import stop as stop_client_operations
            await stop_client_operations(_app)
            await stop_gateway(_app)
            from agents.hermes_embedded_api import get_embedded_hermes_api_service

            if cron_service is not None:
                try:
                    cron_service.shutdown()
                except Exception:
                    logging.getLogger(__name__).exception("cron ticker shutdown failed")
            if gateway_service is not None:
                try:
                    await gateway_service.shutdown()
                except Exception:
                    logging.getLogger(__name__).exception(
                        "embedded gateway shutdown failed"
                    )
            await get_embedded_hermes_api_service().shutdown()

    _DEFAULT_CORS_ALLOWED_ORIGINS = (
        "http://localhost:3300",
        "http://127.0.0.1:3300",
        "http://localhost:3301",
        "http://127.0.0.1:3301",
        "http://localhost:4300",
        "http://127.0.0.1:4300",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    def _cors_allowed_origins() -> tuple[str, ...]:
        override = os.environ.get("SEMANTIER_CORS_ALLOWED_ORIGINS", "").strip()
        if not override:
            return _DEFAULT_CORS_ALLOWED_ORIGINS
        return tuple(origin.strip() for origin in override.split(",") if origin.strip())

    def create_app() -> FastAPI:
        runtime_app = FastAPI(title="Semantier Runtime", lifespan=lifespan)
        if os.environ.get("SEMANTIER_STORAGE_BACKEND") == "postgresql":
            from psycopg import OperationalError
            from fastapi.responses import JSONResponse

            @runtime_app.exception_handler(OperationalError)
            async def authoritative_storage_unavailable(_request, error):
                state = error.sqlstate
                if state is not None and not (state.startswith("08") or state in {"57P01", "57P02", "57P03"}):
                    raise error
                return JSONResponse(status_code=503, content={"detail": {
                    "code": "authoritative_storage_unavailable",
                    "message": "Authoritative storage is temporarily unavailable.",
                }})
        dashboard_dir = os.environ.get("SEMANTIER_MODEL_GATEWAY_DASHBOARD_DIR")
        if dashboard_dir and os.environ.get("SEMANTIER_MODEL_GATEWAY_ENABLED", "").lower() in {"1", "true", "yes", "on"}:
            from fastapi.staticfiles import StaticFiles

            runtime_app.mount("/ui", StaticFiles(directory=dashboard_dir, html=True), name="model-gateway-dashboard")
            runtime_app.mount("/litellm-asset-prefix", StaticFiles(directory=dashboard_dir), name="model-gateway-assets")
        # Browser-driven workflows (dev tooling, e2e gates, embedded
        # workspaces) hit the gateway from a different origin than the
        # Vite/dev server they boot from. Without this middleware, the
        # browser blocks credentialed cross-origin POSTs at the CORS
        # preflight stage, which silently breaks auth-gated flows
        # (`/auth/password/login`, `/auth/context`, ...). The default
        # allowlist is dev-safe (localhost/127.0.0.1 on common dev ports);
        # prod operators must set `SEMANTIER_CORS_ALLOWED_ORIGINS` to the
        # exact workspace origins they intend to serve.
        runtime_app.add_middleware(
            CORSMiddleware,
            allow_origins=list(_cors_allowed_origins()),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "accept",
                "authorization",
                "content-type",
                "cookie",
                "x-requested-with",
            ],
            max_age=600,
        )
        for router in PUBLIC_INGRESS_ROUTERS:
            runtime_app.include_router(router)
        return runtime_app

    app = create_app()

    @app.post("/execute")
    def execute_endpoint(req: dict):
        return execute(req)

except ImportError as exc:
    raise RuntimeError("Semantier API application initialization failed") from exc


def execute(req: dict) -> dict:
    """Programmatic governed execution entrypoint for Semantier adapters."""
    return execute_with_governance(
        org_id=req["org_id"],
        J_t=req["justification"],
        pi_t=req["program"],
        semantic_projection_input=req["semantic_projection_input"],
        ontology=req["ontology"],
        sandbox=req.get("sandbox"),
        idem_store=None,
        post_validator=lambda *args: {"status": "ok"},
    )
