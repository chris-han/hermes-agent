from __future__ import annotations

from fastapi import APIRouter

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from agents.auth_db import save_weixin_runtime_account, update_weixin_runtime_account_state
from agents.auth_session import (
    request_context_from_request,
    VT_SESSION_COOKIE,
    auth_context_payload,
    decode_vt_session_token,
    request_user_agent_hash,
    session_grace_window_seconds,
)
from eos.artifact_hashing import content_hash
from agents.webapi.deps import (
    enforce_rate_limit as _enforce_rate_limit,
    require_authenticated_context as _require_authenticated_context,
)
from agents.webapi.gateway_channel_service import (
    get_active_weixin_correlation_for_user,
    _persist_gateway_reconnect_artifact,
    _restart_embedded_gateway_for_feishu_account,
    GATEWAY_PLATFORM_MAP,
    SUPPORTED_GATEWAYS,
    WEIXIN_RECONNECT_CONTRACT_VERSION,
    _REPO_ROOT,
    _adapter_key,
    _cookie_browser_session_hash,
    _default_weixin_home_channel,
    _ensure_feishu_workspace_pairing_approved,
    _ensure_weixin_workspace_pairing_approved,
    _hash_token,
    _iter_gateway_pairing_stores,
    _find_pending_pairing_by_code,
    _iter_weixin_pairing_stores,
    _messaging_platforms_payload,
    _normalize_weixin_home_channel,
    _read_feishu_bot_config,
    _read_workspace_weixin_account,
    _request_context_from_session_metadata,
    _resolve_governed_weixin_account,
    _runtime_timestamp,
    _sync_weixin_home_channel_env,
    _temporary_embedded_gateway_enabled,
    delete_workspace_feishu_bot_config,
    link_feishu_open_id,
    save_workspace_feishu_bot_config,
)

router = APIRouter(tags=["webapi-gateway"])


@router.get("/gateway/channels")
async def list_gateway_channels():
    return {
        "supported_gateways": list(SUPPORTED_GATEWAYS),
        "gateway_platforms": GATEWAY_PLATFORM_MAP,
        "compatible_api_surfaces": ["/health", "/health/detailed", "/v1/*"],
        "integration_mode": "embedded",
    }


@router.get("/messaging/platforms")
async def messaging_platforms(request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _messaging_platforms_payload(ctx)


@router.post("/messaging/weixin/reconnect")
async def messaging_weixin_reconnect(request: Request):
    _enforce_rate_limit("auth_public", request=request)
    ctx = request_context_from_request(request)
    reconnect_auth_level = "seamless"
    reconnect_auth_reason = "session_valid"
    grace_expires_at_utc: str | None = None
    if not ctx.authenticated:
        raw_meta = decode_vt_session_token(request.cookies.get(VT_SESSION_COOKIE))
        if (
            raw_meta is None
            or ctx.auth_invalid_reason != "weixin_session_expired"
        ):
            raise HTTPException(status_code=401, detail="Authentication required")

        browser_hash = _cookie_browser_session_hash(request)
        if (
            not raw_meta.browser_session_nonce_hash
            or browser_hash != raw_meta.browser_session_nonce_hash
            or (
                raw_meta.user_agent_hash
                and raw_meta.user_agent_hash != request_user_agent_hash(request)
            )
        ):
            return JSONResponse(
                {
                    "contract_version": WEIXIN_RECONNECT_CONTRACT_VERSION,
                    "platform": "weixin",
                    "auth_level": "step_up",
                    "auth_reason": "binding_mismatch",
                    "requires_step_up": True,
                    "grace_expires_at_utc": None,
                    "reconnect_applied": False,
                },
                status_code=401,
            )

        grace_window = session_grace_window_seconds()
        issued_at = raw_meta.issued_at or 0
        grace_expires_at = issued_at + grace_window if issued_at else 0
        grace_expires_at_utc = (
            datetime.fromtimestamp(grace_expires_at, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
            if grace_expires_at
            else None
        )
        if not issued_at or int(time.time()) > grace_expires_at:
            return JSONResponse(
                {
                    "contract_version": WEIXIN_RECONNECT_CONTRACT_VERSION,
                    "platform": "weixin",
                    "auth_level": "step_up",
                    "auth_reason": "expired",
                    "requires_step_up": True,
                    "grace_expires_at_utc": grace_expires_at_utc,
                    "reconnect_applied": False,
                },
                status_code=401,
            )

        ctx = _request_context_from_session_metadata(raw_meta)
        reconnect_auth_level = "grace"
        reconnect_auth_reason = "grace_window"

    _enforce_rate_limit("weixin_reconnect_user", request=request, ctx=ctx)

    weixin_account = _read_workspace_weixin_account(ctx)
    if not isinstance(weixin_account, dict):
        raise HTTPException(status_code=404, detail="Weixin account is not configured")

    weixin_account = _resolve_governed_weixin_account(ctx, weixin_account)
    account_id = str(weixin_account.get("account_id") or "").strip()
    resolved_home_channel = _default_weixin_home_channel(
        home_channel=weixin_account.get("home_channel"),
        fallback_user_id=weixin_account.get("user_id") or ctx.weixin_user_id or account_id,
    )
    if resolved_home_channel != _normalize_weixin_home_channel(weixin_account.get("home_channel")):
        weixin_account = dict(weixin_account)
        weixin_account["home_channel"] = resolved_home_channel
        save_weixin_runtime_account(weixin_account)
    _sync_weixin_home_channel_env(str(weixin_account.get("home_channel") or ""))
    _ensure_weixin_workspace_pairing_approved(
        hermes_home=str(ctx.hermes_home),
        weixin_user_id=str(ctx.weixin_user_id or weixin_account.get("user_id") or account_id),
        user_name=str(ctx.weixin_user_id or weixin_account.get("user_id") or account_id),
    )
    requested_adapter_key: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = {}
    if isinstance(body, dict):
        requested_adapter_key = str(body.get("adapter_key") or "").strip() or None
    adapter_key = _adapter_key(
        platform="weixin",
        workspace_id=ctx.workspace_id,
        account_id=account_id,
    )
    if requested_adapter_key is not None and requested_adapter_key != adapter_key:
        raise HTTPException(status_code=409, detail="adapter_key does not match governed Weixin account binding")

    correlation_id = f"weixin-reconnect-{uuid.uuid4().hex}"
    decision_inputs = {
        "adapter_key": adapter_key,
        "platform": "weixin",
        "workspace_id": ctx.workspace_id,
        "account_id": account_id,
        "token_hash": _hash_token(weixin_account.get("token")),
        "auth_level": reconnect_auth_level,
        "auth_reason": reconnect_auth_reason,
        "requires_step_up": False,
    }

    from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service

    async def _artifact_writer(stage: str, runtime_payload: dict[str, Any]) -> None:
        merged_payload = dict(runtime_payload)
        merged_payload.update(
            {
                "stage": stage,
                "reconnect_scope": "adapter",
                "workspace_id": ctx.workspace_id,
                "account_id": account_id,
                "auth_level": reconnect_auth_level,
                "auth_reason": reconnect_auth_reason,
                "requires_step_up": False,
                "decision_inputs_hash": content_hash(decision_inputs),
            }
        )
        event_type = (
            "gateway.adapter_reconnect_decision"
            if stage == "decision"
            else "gateway.adapter_swap_outcome"
        )
        await _persist_gateway_reconnect_artifact(
            ctx=ctx,
            correlation_id=correlation_id,
            event_type=event_type,
            adapter_key=adapter_key,
            payload=merged_payload,
        )

    _hermes_home_str = str(ctx.hermes_home)
    with _temporary_embedded_gateway_enabled():
        result = await get_embedded_hermes_gateway_service().replace_weixin_adapter(
            workspace_id=ctx.workspace_id,
            hermes_home=_hermes_home_str,
            account=weixin_account,
            artifact_writer=_artifact_writer,
        )

    if not result.get("ok"):
        raise HTTPException(
            status_code=409,
            detail=str(result.get("error") or "Failed to reconnect Weixin gateway"),
        )

    # Persist adapter runtime state to account JSON so frontend reads fresh data
    if result.get("ok") and account_id:
        runtime_state = str(result.get("runtime_session_state") or "active").strip()
        update_weixin_runtime_account_state(
            account_id,
            runtime_session_state=runtime_state,
            runtime_session_updated_at=_runtime_timestamp(),
        )
        persist_ok = False
        
        # Try importing from installed package first
        try:
            from gateway.platforms.weixin import _update_weixin_account_runtime_state
            from agents.runtime_env import get_hermes_home
            
            hermes_home = str(get_hermes_home())
            _update_weixin_account_runtime_state(
                hermes_home,
                account_id=account_id,
                session_state=runtime_state,
            )
            persist_ok = True
        except (ImportError, AttributeError):
            # Fallback: try importing from source hermes-agent directory
            import importlib.util
            weixin_module_path = _REPO_ROOT / "hermes-agent" / "gateway" / "platforms" / "weixin.py"
            if weixin_module_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("_weixin_source_persist", weixin_module_path)
                    weixin_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(weixin_mod)
                    _update_weixin_account_runtime_state = getattr(weixin_mod, "_update_weixin_account_runtime_state", None)
                    
                    if _update_weixin_account_runtime_state:
                        from agents.runtime_env import get_hermes_home
                        hermes_home = str(get_hermes_home())
                        _update_weixin_account_runtime_state(
                            hermes_home,
                            account_id=account_id,
                            session_state=runtime_state,
                        )
                        persist_ok = True
                except Exception as exc:
                    import logging
                    logging.warning(
                        "weixin: failed to persist reconnect runtime state (fallback): %s", exc
                    )
            else:
                import logging
                logging.warning("weixin: source weixin.py not found for fallback import")
        except Exception as exc:
            import logging
            logging.warning(
                "weixin: failed to persist reconnect runtime state: %s", exc
            )

    return {
        "contract_version": WEIXIN_RECONNECT_CONTRACT_VERSION,
        "platform": "weixin",
        "adapter_key": adapter_key,
        "adapter_state": str(result.get("adapter_state") or "active"),
        "reconnect_scope": "adapter",
        "auth_level": reconnect_auth_level,
        "auth_reason": reconnect_auth_reason,
        "requires_step_up": False,
        "grace_expires_at_utc": grace_expires_at_utc,
        "reconnect_applied": True,
        "runtime_session_state": str(result.get("runtime_session_state") or "active"),
    }


@router.post("/messaging/weixin/pairing/approve-self")
async def messaging_weixin_pairing_approve_self(request: Request):
    ctx = _require_authenticated_context(request)
    if not ctx.weixin_user_id:
        raise HTTPException(status_code=409, detail="Authenticated user is not linked to a Weixin identity")

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = str(body.get("code") or "").strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="code required")

    matched_store = None
    pending = None
    for store in _iter_weixin_pairing_stores(ctx):
        resolver = getattr(store, "find_pending_by_code", None)
        pending = resolver("weixin", code) if callable(resolver) else _find_pending_pairing_by_code(store, "weixin", code)
        if pending is not None:
            matched_store = store
            break

    if pending is None or matched_store is None:
        raise HTTPException(status_code=404, detail="Pairing code not found or expired")
    pending_user_id = str(pending.get("user_id") or "").strip()
    if pending_user_id != ctx.weixin_user_id:
        raise HTTPException(status_code=403, detail="This pairing code belongs to a different Weixin user")

    approved = matched_store.approve_code("weixin", code)
    if not approved:
        raise HTTPException(status_code=409, detail="Pairing code not found or expired")

    return {
        "ok": True,
        "platform": "weixin",
        "approved_user_id": ctx.weixin_user_id,
        "code": code,
        "message": "Weixin pairing approved. Your next Weixin message will be authorized.",
    }


@router.post("/messaging/feishu/pairing/approve-self")
async def messaging_feishu_pairing_approve_self(request: Request):
    ctx = _require_authenticated_context(request)
    linked_feishu_open_id = str(getattr(ctx, "feishu_open_id", "") or "").strip()

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = str(body.get("code") or "").strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="code required")

    matched_store = None
    pending = None
    for store in _iter_gateway_pairing_stores(
        hermes_home=str(ctx.hermes_home),
        ctx=ctx,
    ):
        resolver = getattr(store, "find_pending_by_code", None)
        pending = resolver("feishu", code) if callable(resolver) else _find_pending_pairing_by_code(store, "feishu", code)
        if pending is not None:
            matched_store = store
            break

    if pending is None or matched_store is None:
        raise HTTPException(
            status_code=404,
            detail="Pairing code not found or expired",
        )
    pending_user_id = str(pending.get("user_id") or "").strip()
    if not pending_user_id:
        raise HTTPException(
            status_code=409,
            detail="Pairing code has no Feishu user binding",
        )
    if linked_feishu_open_id and pending_user_id != linked_feishu_open_id:
        raise HTTPException(
            status_code=403,
            detail="This pairing code belongs to a different Feishu user",
        )
    if not linked_feishu_open_id:
        try:
            link_feishu_open_id(user_id=ctx.user_id, open_id=pending_user_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Authenticated user not found") from exc

    approved = matched_store.approve_code("feishu", code)
    if not approved:
        raise HTTPException(
            status_code=409,
            detail="Pairing code not found or expired",
        )

    return {
        "ok": True,
        "platform": "feishu",
        "approved_user_id": pending_user_id,
        "code": code,
        "message": "Feishu pairing approved. Your next Feishu message will be authorized.",
    }


@router.post("/messaging/{platform}/validate")
async def messaging_platform_validate(platform: str, request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    normalized = platform.strip().lower()
    if normalized not in {"feishu", "weixin"}:
        raise HTTPException(status_code=404, detail="Messaging platform not found")
    if normalized == "weixin":
        return {
            "platform": "weixin",
            "valid": True,
            "summary": "Weixin configuration is managed by the authenticated workspace session.",
            "details": {
                "configured": bool(_read_workspace_weixin_account(ctx)),
                "auth_provider": ctx.auth_provider,
            },
        }
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    config = body.get("config") if isinstance(body.get("config"), dict) else body
    app_id = str(config.get("app_id") or "").strip()
    app_secret = str(config.get("app_secret") or "").strip()
    domain = str(config.get("domain") or "feishu").strip() or "feishu"
    if not app_id or not app_secret:
        raise HTTPException(status_code=400, detail="app_id and app_secret are required")

    from plugins.platforms.feishu.adapter import probe_bot

    bot_info = probe_bot(app_id, app_secret, domain)
    return {
        "platform": "feishu",
        "valid": bot_info is not None,
        "summary": (
            "Feishu bot credentials validated successfully."
            if bot_info is not None
            else "Feishu bot credentials could not be validated."
        ),
        "details": {
            "domain": domain,
            "bot_name": bot_info.get("bot_name") if bot_info else None,
            "bot_open_id": bot_info.get("bot_open_id") if bot_info else None,
        },
    }


@router.put("/messaging/{platform}")
async def messaging_platform_save(platform: str, request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    normalized = platform.strip().lower()
    if normalized not in {"feishu", "weixin"}:
        raise HTTPException(status_code=404, detail="Messaging platform not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    config = body.get("config") if isinstance(body.get("config"), dict) else body
    if normalized == "weixin":
        existing = _read_workspace_weixin_account(ctx)
        if not isinstance(existing, dict):
            raise HTTPException(status_code=404, detail="Weixin account is not configured")
        updated = dict(existing)
        updated["home_channel"] = _normalize_weixin_home_channel(config.get("home_channel"))
        save_weixin_runtime_account(updated)
        _sync_weixin_home_channel_env(updated["home_channel"])
        payload = _messaging_platforms_payload(ctx)
        for item in payload["platforms"]:
            if item.get("platform") == "weixin":
                return item
        return {
            "platform": "weixin",
            "configured": True,
            "config": {
                "home_channel": updated["home_channel"],
            },
        }
    existing_feishu = _read_feishu_bot_config(ctx)
    if not isinstance(existing_feishu, dict):
        existing_feishu = {}
    raw_app_id = str(config.get("app_id") or "").strip()
    raw_app_secret = str(config.get("app_secret") or "").strip()
    existing_app_id = str(existing_feishu.get("app_id") or "").strip()
    existing_app_secret = str(existing_feishu.get("app_secret") or "").strip()
    app_id = raw_app_id or existing_app_id
    app_secret = raw_app_secret or existing_app_secret
    domain = (
        str(config.get("domain") or existing_feishu.get("domain") or "feishu").strip()
        or "feishu"
    )
    connection_mode = str(
        config.get("connection_mode")
        or existing_feishu.get("connection_mode")
        or "websocket"
    ).strip() or "websocket"
    home_channel = str(
        config.get("home_channel")
        if "home_channel" in config
        else existing_feishu.get("home_channel") or ""
    ).strip()
    if raw_app_id and raw_app_id != existing_app_id and not raw_app_secret:
        raise HTTPException(
            status_code=400,
            detail="app_secret is required when changing app_id",
        )
    if not app_id or not app_secret:
        raise HTTPException(status_code=400, detail="app_id and app_secret are required")

    from plugins.platforms.feishu.adapter import probe_bot

    should_probe = (
        bool(raw_app_secret)
        or (bool(raw_app_id) and raw_app_id != existing_app_id)
        or domain != str(existing_feishu.get("domain") or "feishu")
    )
    bot_info = probe_bot(app_id, app_secret, domain) if should_probe else None
    validated_at = (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if should_probe
        else str(existing_feishu.get("validated_at") or "").strip() or None
    )
    user = auth_context_payload(ctx).get("user") or {}
    workspace_id = str(user.get("workspace_id") or "").strip()
    saved = save_workspace_feishu_bot_config(
        workspace_id,
        app_id=app_id,
        app_secret=app_secret,
        domain=domain,
        connection_mode=connection_mode,
        bot_open_id=(
            str(bot_info.get("bot_open_id") or "")
            if bot_info
            else str(existing_feishu.get("bot_open_id") or "") or None
        ),
        bot_name=(
            str(bot_info.get("bot_name") or "")
            if bot_info
            else str(existing_feishu.get("bot_name") or "") or None
        ),
        home_channel=home_channel,
        validated_at=validated_at,
        last_error=(
            None
            if not should_probe or bot_info is not None
            else "validation_failed"
        ),
        manager_user_id=ctx.user_id,
    )
    feishu_open_id = str(user.get("feishu_open_id") or "").strip()
    if feishu_open_id:
        _ensure_feishu_workspace_pairing_approved(
            hermes_home=str(ctx.hermes_home),
            user_id=ctx.user_id,
            feishu_open_id=feishu_open_id,
            user_name=str(user.get("name") or feishu_open_id),
        )
    
    gateway_applied = False
    try:
        restart_result = await _restart_embedded_gateway_for_feishu_account(saved)
        gateway_applied = bool(restart_result.get("ok"))
    except Exception as exc:
        import logging
        logging.warning("feishu: failed to start bot after config save: %s", exc)
    
    payload = _messaging_platforms_payload(ctx)
    for item in payload["platforms"]:
        if item.get("platform") == "feishu":
            item["gateway_applied"] = gateway_applied
            return item
    return {
        "platform": "feishu",
        "configured": True,
        "config": saved,
        "gateway_applied": gateway_applied,
    }


@router.delete("/messaging/{platform}")
async def messaging_platform_delete(platform: str, request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    normalized = platform.strip().lower()
    if normalized not in {"feishu", "weixin"}:
        raise HTTPException(status_code=404, detail="Messaging platform not found")
    if normalized == "feishu":
        user = auth_context_payload(ctx).get("user") or {}
        workspace_id = str(user.get("workspace_id") or "").strip()
        delete_workspace_feishu_bot_config(workspace_id)
        return {"ok": True}
    raise HTTPException(
        status_code=501,
        detail="Messaging configuration deletion is not implemented in this gateway yet",
    )
