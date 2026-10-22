from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable


RUNTIME_EMBED_GATEWAY_ENV = "SEMANTIER_RUNTIME_EMBED_GATEWAY"
_REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _shared_runtime_root() -> Path:
    """Return the shared Semantier runtime root, ignoring workspace HERMES_HOME."""
    raw = os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or str(_REPO_ROOT / ".semantier-home")
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else _REPO_ROOT / candidate).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").strip().lower()


def _weixin_registered_sender_is_authorized(source: Any) -> bool:
    """Authorize Weixin DMs when the sender matches governed runtime state.

    Hermes gateway allowlists are process/env scoped. Semantier tracks Weixin
    ownership in governed auth state per workspace, so a registered runtime
    account or active correlation should be sufficient to admit the owner DM
    even when the env allowlist is stale or absent.
    """
    if _platform_name(getattr(source, "platform", None)) != "weixin":
        return False

    external_user_id = str(getattr(source, "user_id", "") or "").strip()
    if not external_user_id:
        return False

    chat_type = str(getattr(source, "chat_type", "") or "").strip().lower()
    if chat_type and chat_type != "dm":
        return False

    try:
        from agents.gateway_identity import resolve_weixin_owner_correlation

        correlation = resolve_weixin_owner_correlation(
            external_user_id=external_user_id,
            chat_id=str(getattr(source, "chat_id", "") or "").strip() or None,
        )
    except Exception:
        correlation = None

    if isinstance(correlation, dict):
        owner_user_id = str(correlation.get("owner_user_id") or "").strip()
        owner_workspace_id = str(correlation.get("owner_workspace_id") or "").strip()
        if owner_user_id and owner_workspace_id:
            return True

    try:
        from agents.auth_db import load_weixin_runtime_accounts
    except Exception:
        return False

    for account in load_weixin_runtime_accounts():
        if not isinstance(account, dict):
            continue
        account_external_user_id = str(
            account.get("external_user_id") or account.get("user_id") or ""
        ).strip()
        if account_external_user_id != external_user_id:
            continue
        owner_workspace_id = str(
            account.get("owner_workspace_id") or account.get("workspace_id") or ""
        ).strip()
        if owner_workspace_id:
            return True
    return False


def _resolve_weixin_runtime_adapter_key_for_source(source: Any) -> str | None:
    if _platform_name(getattr(source, "platform", None)) != "weixin":
        return None
    external_user_id = str(getattr(source, "user_id", "") or "").strip()
    if not external_user_id:
        return None
    chat_type = str(getattr(source, "chat_type", "") or "").strip().lower()
    if chat_type and chat_type != "dm":
        return None

    try:
        from agents.auth_db import load_weixin_runtime_accounts
    except Exception:
        return None

    matches: list[str] = []
    for account in load_weixin_runtime_accounts():
        if not isinstance(account, dict):
            continue
        account_external_user_id = str(
            account.get("external_user_id") or account.get("user_id") or ""
        ).strip()
        if account_external_user_id != external_user_id:
            continue
        workspace_id = str(
            account.get("owner_workspace_id") or account.get("workspace_id") or ""
        ).strip()
        account_id = str(account.get("account_id") or "").strip()
        if workspace_id and account_id:
            matches.append(AdapterRuntimeKey("weixin", workspace_id, account_id).as_string())
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _patch_gateway_runner_governed_context(gateway_runner_cls: type[Any]) -> None:
    """Restore parent-owned, per-turn context through Hermes's one-shot sidecar."""
    if getattr(gateway_runner_cls, "_semantier_governed_context_patched", False):
        return
    run = getattr(gateway_runner_cls, "_run_agent_inner", None)
    if not callable(run):
        return

    async def _run(self, message, context_prompt, history, source, session_id, **kwargs):
        from agents.governed_context import build_governed_runtime_context_prompt

        workspace_id = str(getattr(source, "workspace_owner_id", "") or "").strip()
        session_key = kwargs.get("session_key")
        note = None
        if workspace_id and session_key:
            # External transport IDs are not governed user IDs. Resolve through
            # the authoritative workspace owner in the core context builder.
            note = build_governed_runtime_context_prompt(
                workspace_id=workspace_id, user_message=str(message),
            )
        if note:
            notes = self._consume_pending_turn_sidecar_notes(session_key)
            self._set_pending_turn_sidecar_notes(session_key, [*notes, note])
        try:
            return await run(self, message, context_prompt, history, source, session_id, **kwargs)
        finally:
            # Cancellation before the worker consumes the sidecar must not carry
            # this turn's context into a later turn or recovery attempt.
            if note:
                remaining = self._consume_pending_turn_sidecar_notes(session_key)
                self._set_pending_turn_sidecar_notes(
                    session_key, [item for item in remaining if item != note],
                )

    gateway_runner_cls._run_agent_inner = _run
    gateway_runner_cls._semantier_governed_context_patched = True


def _patch_gateway_runner_feishu_ingress(gateway_runner_cls: type[Any]) -> None:
    """Keep governed identity binding outside independently versioned Hermes adapters."""
    if getattr(gateway_runner_cls, "_semantier_feishu_ingress_patched", False):
        return
    wire = getattr(gateway_runner_cls, "_wire_adapter_handlers", None)
    authorize = getattr(gateway_runner_cls, "_is_user_authorized", None)
    commands = getattr(gateway_runner_cls, "_hm_dispatch_idle_commands", None)
    if not callable(wire) or not callable(authorize):
        return

    def _wire(self, adapter, **kwargs):
        wire(self, adapter, **kwargs)
        if _platform_name(getattr(adapter, "platform", None)) != "feishu":
            return
        if getattr(adapter, "_semantier_feishu_ingress_bound", False):
            return
        admit = adapter._admit

        def _admit(sender, message):
            reason = admit(sender, message)
            if reason != "dm_policy_rejected":
                return reason
            from agents.feishu_ingress_identity import (
                FeishuIngressOwnerResolutionError,
                resolve_feishu_ingress_owner,
            )
            sender_id = getattr(sender, "sender_id", None)
            try:
                resolve_feishu_ingress_owner(
                    open_id=getattr(sender_id, "open_id", None),
                    union_id=getattr(sender_id, "union_id", None),
                    chat_id=getattr(message, "chat_id", None),
                    platform_session_key=getattr(message, "message_id", None),
                )
            except FeishuIngressOwnerResolutionError:
                return reason
            return None

        adapter._admit = _admit
        receive = getattr(adapter, "_handle_message_event_data", None)
        if callable(receive):
            async def _receive(data):
                raw = getattr(data, "event", None)
                message = getattr(raw, "message", None)
                sender = getattr(raw, "sender", None)
                try:
                    text = json.loads(getattr(message, "content", "") or "{}").get("text", "")
                except (TypeError, ValueError, AttributeError):
                    text = ""
                if not (isinstance(text, str) and text.startswith("/link ")
                        and getattr(message, "message_type", None) == "text"
                        and getattr(message, "chat_type", None) == "p2p"
                        and getattr(sender, "sender_type", None) == "user"):
                    return await receive(data)
                message_id = getattr(message, "message_id", None)
                if not message_id or await adapter._is_duplicate(message_id):
                    return
                from agents.gateway_identity import consume_feishu_message_link

                identity = getattr(sender, "sender_id", None)
                try:
                    consume_feishu_message_link(
                        state=text.removeprefix("/link ").strip(), app_id=adapter._app_id,
                        open_id=getattr(identity, "open_id", ""),
                        union_id=getattr(identity, "union_id", None),
                    )
                except ValueError as exc:
                    reply = str(exc)
                except KeyError:
                    reply = "Link code is invalid, expired or already used. Generate a new code in Semantier Settings. / 验证码无效、过期或已使用，请在 Semantier 设置中重新生成。"
                else:
                    reply = "Feishu account linked. Send hi to start. / 飞书账号已绑定，发送 hi 即可开始。"
                await adapter.send(message.chat_id, reply)

            adapter._handle_message_event_data = _receive

        def _bound(handle):
            async def _handle(event):
                from agents.feishu_ingress_identity import (
                    FeishuIngressOwnerResolutionError,
                    stamp_feishu_ingress_owner,
                )
                try:
                    stamp_feishu_ingress_owner(event)
                except FeishuIngressOwnerResolutionError as exc:
                    logger.warning("Feishu ingress rejected: %s", exc)
                    return
                await handle(event)
            return _handle

        # Bind before text batching pins RoutingIdentity/session keys. Keep the
        # handle guard too: reaction/card events can bypass normal dispatch.
        adapter._dispatch_inbound_event = _bound(adapter._dispatch_inbound_event)
        adapter.handle_message = _bound(adapter.handle_message)
        adapter._semantier_feishu_ingress_bound = True

    def _authorize(self, source, **kwargs):
        if _platform_name(getattr(source, "platform", None)) == "feishu":
            from agents.feishu_ingress_identity import (
                FeishuIngressOwnerResolutionError,
                resolve_feishu_ingress_owner,
            )
            try:
                owner = resolve_feishu_ingress_owner(
                    open_id=getattr(source, "user_id", None),
                    union_id=getattr(source, "user_id_alt", None),
                    chat_id=getattr(source, "chat_id", None),
                )
            except FeishuIngressOwnerResolutionError:
                return False
            return owner.owner_workspace_id == getattr(source, "workspace_owner_id", None)
        return authorize(self, source, **kwargs)

    gateway_runner_cls._wire_adapter_handlers = _wire
    gateway_runner_cls._is_user_authorized = _authorize
    if callable(commands):
        async def _commands(self, event, source, session_key):
            if _platform_name(getattr(source, "platform", None)) != "feishu":
                return await commands(self, event, source, session_key)
            # Hermes now scans skill commands in an executor before preparing
            # the model turn. That executor needs the same governed session
            # context, even for ordinary text that falls through to the model.
            from gateway.session import build_session_context

            entry = await self.async_session_store.get_or_create_session(source)
            tokens = self._set_session_env(build_session_context(source, self.config, entry))
            try:
                return await commands(self, event, source, session_key)
            finally:
                self._clear_session_env(tokens)

        gateway_runner_cls._hm_dispatch_idle_commands = _commands
    gateway_runner_cls._semantier_feishu_ingress_patched = True


def _patch_gateway_runner_weixin_authorization(gateway_runner_cls: type[Any]) -> None:
    if getattr(gateway_runner_cls, "_semantier_weixin_auth_patched", False):
        return

    original = getattr(gateway_runner_cls, "_is_user_authorized", None)
    if not callable(original):
        return

    wire = getattr(gateway_runner_cls, "_wire_adapter_handlers", None)
    if callable(wire):
        def _wire(self, adapter, **kwargs):
            wire(self, adapter, **kwargs)
            if (_platform_name(getattr(adapter, "platform", None)) != "weixin"
                    or getattr(adapter, "_semantier_weixin_ingress_bound", False)):
                return

            def _bind(event):
                from agents.weixin_ingress_identity import (
                    WeixinIngressOwnerResolutionError,
                    resolve_weixin_ingress_owner,
                )
                source = event.source
                try:
                    owner = resolve_weixin_ingress_owner(
                        account_id=adapter._account_id,
                        external_user_id=source.user_id,
                        chat_id=source.chat_id,
                    )
                except WeixinIngressOwnerResolutionError as exc:
                    logger.warning("Weixin ingress rejected: %s", exc)
                    return False
                # Receiving a bot's message is not evidence of sender ownership.
                if (not source.user_id or owner.external_user_id != source.user_id
                        or (owner.account_id and owner.account_id != adapter._account_id)
                        or (source.workspace_owner_id
                            and source.workspace_owner_id != owner.owner_workspace_id)):
                    logger.warning("Weixin ingress rejected: owner binding mismatch")
                    return False
                source.workspace_owner_id = owner.owner_workspace_id
                return True

            enqueue = adapter._enqueue_text_event
            handle = adapter.handle_message

            def _enqueue(event):
                if _bind(event):
                    return enqueue(event)

            async def _handle(event):
                if _bind(event):
                    return await handle(event)

            # Bind before batching derives RoutingIdentity/session keys, and
            # recheck on dispatch so revocation during batching fails closed.
            adapter._enqueue_text_event = _enqueue
            adapter.handle_message = _handle
            adapter._semantier_weixin_ingress_bound = True

        gateway_runner_cls._wire_adapter_handlers = _wire

    def _wrapped_is_user_authorized(self, source, **kwargs):  # type: ignore[no-untyped-def]
        if _weixin_registered_sender_is_authorized(source):
            adapter_key = _resolve_weixin_runtime_adapter_key_for_source(source)
            if adapter_key:
                if not str(getattr(source, "adapter_key", "") or "").strip():
                    setattr(source, "adapter_key", adapter_key)
                if not str(getattr(source, "delivery_adapter_key", "") or "").strip():
                    setattr(source, "delivery_adapter_key", adapter_key)
            return True
        return original(self, source, **kwargs)

    setattr(gateway_runner_cls, "_semantier_original_is_user_authorized", original)
    setattr(gateway_runner_cls, "_is_user_authorized", _wrapped_is_user_authorized)
    setattr(gateway_runner_cls, "_semantier_weixin_auth_patched", True)


@dataclass(frozen=True)
class EmbeddedGatewayStatus:
    running: bool
    enabled_platforms: tuple[str, ...]


@dataclass(frozen=True)
class AdapterRuntimeKey:
    platform: str
    workspace_id: str
    account_id: str

    def as_string(self) -> str:
        return f"{self.platform}:{self.workspace_id}:{self.account_id}"


@dataclass
class AdapterRegistryEntry:
    key: AdapterRuntimeKey
    state: str
    platform: str
    workspace_id: str
    account_id: str
    adapter_name: str
    updated_at_utc: str


@dataclass
class _RunnerRuntime:
    runner: Any
    watch_task: asyncio.Task[None] | None
    cron_stop: threading.Event | None
    cron_thread: threading.Thread | None
    runtime_key: str | None = None


class _ScopedWeixinAdapterProxy:
    """Runner-visible Weixin adapter that resolves account-scoped transports."""

    def __init__(self, platform: Any, service: "EmbeddedHermesGatewayService") -> None:
        self.platform = platform
        self._service = service
        self._runtime_session_state = "active"
        self._pending_messages: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "Weixin"

    @property
    def is_connected(self) -> bool:
        return self._service.has_active_weixin_adapter()

    async def cancel_background_tasks(self) -> None:
        """Satisfy the runner adapter shutdown contract.

        Managed Weixin adapters are stopped by EmbeddedHermesGatewayService.
        The runner only sees this proxy, so lifecycle calls here are no-ops.
        """
        return None

    async def disconnect(self) -> None:
        """Satisfy the runner adapter shutdown contract."""
        return None

    def resolve_for_source(self, source: Any) -> Any | None:
        return self._service.resolve_delivery_adapter(source)

    def get_pending_message(self, session_key: str) -> Any | None:
        return self._pending_messages.pop(session_key, None)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        meta = dict(metadata or {})
        adapter = self._service.resolve_delivery_adapter(
            SimpleNamespace(
                platform=self.platform,
                adapter_key=meta.get("adapter_key"),
                delivery_adapter_key=meta.get("delivery_adapter_key"),
                chat_id=chat_id,
            )
        )
        if adapter is None:
            raise ValueError("No delivery adapter resolved for Weixin send")
        return await adapter.send(chat_id, content, reply_to=reply_to, metadata=metadata)


class EmbeddedHermesGatewayService:
    """Own an in-process Hermes gateway runtime for `semantier run`."""

    def __init__(self) -> None:
        self._shared_runtime: _RunnerRuntime | None = None
        self._weixin_adapters: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._adapter_registry: dict[str, AdapterRegistryEntry] = {}
        self._adapter_locks: dict[str, asyncio.Lock] = {}

    def resolve_adapter_by_key(self, adapter_key: str, *, platform: Any | None = None) -> Any | None:
        key_text = str(adapter_key or "").strip()
        if not key_text:
            return None
        adapter = self._weixin_adapters.get(key_text)
        if adapter is None:
            return None
        if platform is not None:
            adapter_platform = getattr(adapter, "platform", None)
            if adapter_platform is None:
                entry = self._adapter_registry.get(key_text)
                adapter_platform = entry.platform if entry is not None else None
            if _platform_name(platform) != _platform_name(adapter_platform):
                return None
        return adapter

    def resolve_delivery_adapter(self, target: Any) -> Any | None:
        delivery_key = str(getattr(target, "delivery_adapter_key", "") or "").strip()
        if delivery_key:
            return self.resolve_adapter_by_key(
                delivery_key,
                platform=getattr(target, "platform", None),
            )
        adapter_key = str(getattr(target, "adapter_key", "") or "").strip()
        if adapter_key:
            return self.resolve_adapter_by_key(
                adapter_key,
                platform=getattr(target, "platform", None),
            )
        chat_id = str(getattr(target, "chat_id", "") or "").strip()
        if chat_id and _platform_name(getattr(target, "platform", None)) == "weixin":
            return self.resolve_weixin_adapter_by_external_user_id(chat_id)
        return None

    def resolve_weixin_adapter_by_external_user_id(self, external_user_id: str) -> Any | None:
        external_user_id = str(external_user_id or "").strip()
        if not external_user_id:
            return None

        try:
            from agents.auth_db import load_weixin_runtime_accounts
        except Exception:
            return None

        matching_keys: list[str] = []
        for account in load_weixin_runtime_accounts():
            if not isinstance(account, dict):
                continue
            account_external_user_id = str(
                account.get("external_user_id") or account.get("user_id") or ""
            ).strip()
            if account_external_user_id != external_user_id:
                continue
            workspace_id = str(
                account.get("owner_workspace_id") or account.get("workspace_id") or ""
            ).strip()
            account_id = str(account.get("account_id") or "").strip()
            if workspace_id and account_id:
                matching_keys.append(AdapterRuntimeKey("weixin", workspace_id, account_id).as_string())

        unique_keys = sorted(set(matching_keys))
        if len(unique_keys) != 1:
            return None
        return self.resolve_adapter_by_key(unique_keys[0], platform="weixin")

    def has_active_weixin_adapter(self) -> bool:
        return bool(self._weixin_adapters)

    def is_enabled(self) -> bool:
        value = os.environ.get(RUNTIME_EMBED_GATEWAY_ENV, "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    async def ensure_started(self) -> EmbeddedGatewayStatus:
        if not self.is_enabled():
            return EmbeddedGatewayStatus(running=False, enabled_platforms=())
        if self._shared_runtime is not None or self._weixin_adapters:
            return self._status_from_runtimes()

        async with self._lock:
            if self._shared_runtime is not None or self._weixin_adapters:
                return self._status_from_runtimes()

            from agents.launcher import _ensure_repo_local_runtime
            runtime_root = _shared_runtime_root()
            os.environ.setdefault("SEMANTIER_LOCAL_STATE_DIR", ".semantier-home")
            from agents.auth_db import is_postgresql
            if not is_postgresql():
                os.environ.setdefault(
                    "SEMANTIER_AUTH_DB_PATH",
                    str(Path(os.environ["SEMANTIER_LOCAL_STATE_DIR"]) / "auth.db"),
                )
            _ensure_repo_local_runtime()
            from agents.auth_pairing_postgres import install_hosted_channel_storage
            install_hosted_channel_storage()
            _hydrate_saved_feishu_gateway_env()

            from agents.hermes_boundary_provider import SemantierExecutionBoundaryProvider
            from gateway.config import Platform, load_gateway_config
            from gateway.execution_boundary import replace_execution_boundary_provider
            from gateway.run import GatewayRunner, _start_cron_ticker

            replace_execution_boundary_provider(SemantierExecutionBoundaryProvider())
            _refresh_gateway_runtime_home()
            _patch_gateway_runner_governed_context(GatewayRunner)
            _patch_gateway_runner_weixin_authorization(GatewayRunner)
            _patch_gateway_runner_feishu_ingress(GatewayRunner)

            config = load_gateway_config()
            api_server_config = config.platforms.get(Platform.API_SERVER)
            if api_server_config is not None:
                api_server_config.enabled = False
            weixin_platform = getattr(Platform, "WEIXIN", None)
            weixin_config = config.platforms.get(weixin_platform) if weixin_platform is not None else None
            if weixin_config is not None:
                weixin_config.enabled = False

            self._shared_runtime = await self._start_runner_runtime(
                GatewayRunner=GatewayRunner,
                start_cron_ticker=_start_cron_ticker,
                config=config,
                runtime_key=None,
            )
            self._install_weixin_proxy()
            await self._ensure_saved_weixin_adapters_started(runtime_root=runtime_root)
            return self._status_from_runtimes()

    async def shutdown(self) -> None:
        async with self._lock:
            runtime = self._shared_runtime
            adapters = list(self._weixin_adapters.values())
            self._shared_runtime = None
            self._weixin_adapters = {}
            self._adapter_registry.clear()
            self._adapter_locks.clear()

        for adapter in adapters:
            await self._stop_weixin_adapter(adapter)
        if runtime is not None:
            await self._stop_runner_runtime(runtime)

    def _status_from_runtimes(self) -> EmbeddedGatewayStatus:
        enabled_platforms: set[str] = set()
        running = False
        if self._shared_runtime is not None:
            runtime = self._shared_runtime
            running = bool(getattr(runtime.runner, "_running", False))
            for platform in getattr(runtime.runner, "adapters", {}).keys():
                if _platform_name(platform) == "weixin" and not self._weixin_adapters:
                    continue
                enabled_platforms.add(getattr(platform, "value", str(platform)))
        if self._weixin_adapters:
            enabled_platforms.add("weixin")
        return EmbeddedGatewayStatus(
            running=running,
            enabled_platforms=tuple(sorted(enabled_platforms)),
        )

    async def replace_weixin_adapter(
        self,
        *,
        workspace_id: str,
        hermes_home: str,
        account: dict[str, Any],
        artifact_writer: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        account_id = str(account.get("account_id") or "").strip()
        token = str(account.get("token") or "").strip()
        if not workspace_id or not hermes_home or not account_id or not token:
            return {
                "ok": False,
                "error": "workspace_id, hermes_home, account_id, and token are required",
            }

        await self.ensure_started()
        key = AdapterRuntimeKey(
            platform="weixin",
            workspace_id=workspace_id,
            account_id=account_id,
        )
        key_text = key.as_string()
        lock = self._adapter_locks.setdefault(key_text, asyncio.Lock())

        async with lock:
            old_adapter = self._weixin_adapters.get(key_text)
            old_account: dict[str, Any] | None = None
            if old_adapter is not None:
                old_account = self._account_from_adapter(old_adapter)
            self._sync_registry_entry(
                key=key,
                state="stopping" if old_adapter is not None else "pending",
                adapter=old_adapter,
            )

            if artifact_writer is not None:
                await artifact_writer(
                    "decision",
                    {
                        "adapter_key": key_text,
                        "platform": "weixin",
                        "workspace_id": workspace_id,
                        "account_id": account_id,
                        "candidate_token_present": True,
                    },
                )

            try:
                if old_adapter is not None:
                    await self._stop_weixin_adapter(old_adapter)

                new_adapter = await self._start_weixin_adapter(
                    key=key,
                    hermes_home=hermes_home,
                    account=account,
                )
                self._weixin_adapters[key_text] = new_adapter
                self._install_weixin_proxy()
                self._sync_registry_entry(
                    key=key,
                    state="active",
                    adapter=new_adapter,
                )

                if artifact_writer is not None:
                    try:
                        await artifact_writer(
                            "cutover",
                            {
                                "adapter_key": key_text,
                                "platform": "weixin",
                                "workspace_id": workspace_id,
                                "account_id": account_id,
                                "outcome": "cutover",
                            },
                        )
                    except Exception:
                        await self._stop_weixin_adapter(new_adapter)
                        self._weixin_adapters.pop(key_text, None)
                        self._sync_registry_entry(key=key, state="degraded", adapter=None)
                        raise

                return {
                    "ok": True,
                    "adapter_key": key_text,
                    "adapter_state": "active",
                    "runtime_session_state": self._adapter_runtime_state(new_adapter),
                }
            except Exception as exc:
                restored_adapter = None
                if old_account is not None:
                    try:
                        restored_adapter = await self._start_weixin_adapter(
                            key=key,
                            hermes_home=hermes_home,
                            account=old_account,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to restore previous Weixin adapter for %s after replacement failure",
                            key_text,
                            exc_info=True,
                        )
                    else:
                        self._weixin_adapters[key_text] = restored_adapter
                        self._install_weixin_proxy()
                        self._sync_registry_entry(
                            key=key,
                            state="active",
                            adapter=restored_adapter,
                        )
                if restored_adapter is not None:
                    return {
                        "ok": False,
                        "error": str(exc),
                        "adapter_key": key_text,
                        "adapter_state": "rolled_back",
                        "runtime_session_state": self._adapter_runtime_state(restored_adapter),
                    }
                self._sync_registry_entry(key=key, state="degraded", adapter=None)
                return {
                    "ok": False,
                    "error": str(exc),
                    "adapter_key": key_text,
                    "adapter_state": self._adapter_registry[key_text].state,
                }

    async def _watch_runner_shutdown(self, runtime: _RunnerRuntime) -> None:
        try:
            await runtime.runner.wait_for_shutdown()
        except asyncio.CancelledError:
            raise
        finally:
            if runtime.runtime_key is None and self._shared_runtime is runtime:
                self._shared_runtime = None

    def _build_weixin_platform_config(
        self,
        *,
        base_config: Any | None,
        account: dict[str, Any],
    ):
        from gateway.config import HomeChannel, Platform, PlatformConfig

        config = copy.deepcopy(base_config) if isinstance(base_config, PlatformConfig) else PlatformConfig()
        config.enabled = True
        config.token = str(account.get("token") or "").strip()
        extra = dict(config.extra or {})
        extra["account_id"] = str(account.get("account_id") or "").strip()
        for source_key in (
            "base_url",
            "cdn_base_url",
            "dm_policy",
            "group_policy",
            "allow_from",
            "group_allow_from",
            "home_channel",
        ):
            value = account.get(source_key)
            if value not in (None, ""):
                extra[source_key] = value
        config.extra = extra
        home_channel = str(account.get("home_channel") or "").strip()
        if home_channel:
            config.home_channel = HomeChannel(
                platform=Platform.WEIXIN,
                chat_id=home_channel,
                name="Home",
            )
        elif config.home_channel is not None:
            config.home_channel = copy.deepcopy(config.home_channel)
            config.home_channel.platform = Platform.WEIXIN
        return config

    def _sync_registry_entry(
        self,
        *,
        key: AdapterRuntimeKey,
        state: str,
        adapter: Any | None,
    ) -> None:
        self._adapter_registry[key.as_string()] = AdapterRegistryEntry(
            key=key,
            state=state,
            platform=key.platform,
            workspace_id=key.workspace_id,
            account_id=key.account_id,
            adapter_name=type(adapter).__name__ if adapter is not None else "",
            updated_at_utc=_utc_now(),
        )

    @staticmethod
    def _adapter_runtime_state(adapter: Any) -> str:
        return str(getattr(adapter, "_runtime_session_state", "") or "").strip() or "active"

    @staticmethod
    def _account_from_adapter(adapter: Any) -> dict[str, Any]:
        config = getattr(adapter, "config", None)
        extra = dict(getattr(config, "extra", {}) or {})
        account: dict[str, Any] = {
            "account_id": str(extra.get("account_id") or getattr(adapter, "_account_id", "") or "").strip(),
            "token": str(getattr(config, "token", "") or "").strip(),
        }
        for key in (
            "base_url",
            "cdn_base_url",
            "dm_policy",
            "group_policy",
            "allow_from",
            "group_allow_from",
            "home_channel",
        ):
            value = extra.get(key)
            if value not in (None, ""):
                account[key] = value
        return account

    async def _start_runner_runtime(
        self,
        *,
        GatewayRunner: type[Any],
        start_cron_ticker: Callable[..., Any],
        config: Any,
        runtime_key: str | None,
        hermes_home: str | None = None,
    ) -> _RunnerRuntime:
        loop = asyncio.get_running_loop()
        shared_home = str(_shared_runtime_root())
        with _temporary_platform_hermes_home(shared_home):
            runner = GatewayRunner(config)
        runner._semantier_embedded_boundary_required = True
        if hermes_home:
            with _temporary_hermes_home(hermes_home):
                success = await runner.start()
        else:
            with _temporary_platform_hermes_home(shared_home):
                success = await runner.start()
        if not success:
            raise RuntimeError("failed to start embedded Hermes gateway")

        from agents.cron_home_provider import semantier_cron_homes
        from agents.cron_storage_provider import SemantierCronStorageProvider

        def cron_profile_homes():
            homes = [Path(hermes_home)] if hermes_home else semantier_cron_homes()
            profiles = {}
            for home in homes:
                resolved = Path(home).expanduser().resolve()
                scope = SemantierCronStorageProvider._scope(resolved)
                profiles[scope.removeprefix("workspace:")] = resolved
            return sorted(profiles.items())

        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=start_cron_ticker,
            args=(cron_stop,),
            kwargs={"adapters": runner.adapters, "loop": loop, "profile_homes": cron_profile_homes},
            daemon=True,
            name="semantier-embedded-cron" if runtime_key is None else f"semantier-embedded-cron-{runtime_key}",
        )
        cron_thread.start()
        runtime = _RunnerRuntime(
            runner=runner,
            watch_task=None,
            cron_stop=cron_stop,
            cron_thread=cron_thread,
            runtime_key=runtime_key,
        )
        self._bind_delivery_router(runner)
        runtime.watch_task = asyncio.create_task(self._watch_runner_shutdown(runtime))
        return runtime

    def _bind_delivery_router(self, runner: Any) -> None:
        delivery_router = getattr(runner, "delivery_router", None)
        if delivery_router is None:
            return
        setattr(delivery_router, "adapter_resolver", self.resolve_delivery_adapter)

    def _install_weixin_proxy(self) -> None:
        runtime = self._shared_runtime
        if runtime is None:
            return
        adapters = getattr(runtime.runner, "adapters", None)
        if not isinstance(adapters, dict):
            return
        try:
            from gateway.config import Platform
        except Exception:
            return
        weixin_platform = getattr(Platform, "WEIXIN", None)
        if weixin_platform is None:
            return
        existing = adapters.get(weixin_platform)
        if existing is not None and not isinstance(existing, _ScopedWeixinAdapterProxy):
            return
        adapters[weixin_platform] = _ScopedWeixinAdapterProxy(weixin_platform, self)
        delivery_router = getattr(runtime.runner, "delivery_router", None)
        if delivery_router is not None:
            delivery_router.adapters = adapters

    async def _stop_runner_runtime(self, runtime: _RunnerRuntime) -> None:
        try:
            await runtime.runner.stop()
            await runtime.runner.wait_for_shutdown()
        finally:
            if runtime.cron_stop is not None:
                runtime.cron_stop.set()
            if runtime.cron_thread is not None:
                await asyncio.to_thread(runtime.cron_thread.join, 5)
            if runtime.watch_task is not None and not runtime.watch_task.done():
                runtime.watch_task.cancel()
                try:
                    await runtime.watch_task
                except asyncio.CancelledError:
                    pass

    async def _ensure_saved_weixin_adapters_started(self, *, runtime_root: Path) -> None:
        for hermes_home, account in _discover_saved_weixin_accounts(runtime_root):
            workspace_id = str(
                account.get("owner_workspace_id") or account.get("workspace_id") or ""
            ).strip()
            account_id = str(account.get("account_id") or "").strip()
            token = str(account.get("token") or "").strip()
            if not workspace_id or not account_id or not token:
                continue
            key = AdapterRuntimeKey(
                platform="weixin",
                workspace_id=workspace_id,
                account_id=account_id,
            )
            key_text = key.as_string()
            if key_text in self._weixin_adapters:
                continue
            try:
                self._weixin_adapters[key_text] = await self._start_weixin_adapter(
                    key=key,
                    hermes_home=str(hermes_home),
                    account=account,
                )
                self._install_weixin_proxy()
                self._sync_registry_entry(key=key, state="active", adapter=self._weixin_adapters[key_text])
            except Exception:
                self._sync_registry_entry(key=key, state="degraded", adapter=None)

    async def _start_weixin_adapter(
        self,
        *,
        key: AdapterRuntimeKey,
        hermes_home: str,
        account: dict[str, Any],
    ) -> Any:
        from gateway.config import load_gateway_config
        from agents.auth_pairing_postgres import install_hosted_channel_storage
        install_hosted_channel_storage()
        from gateway.platforms.weixin import WeixinAdapter

        if self._shared_runtime is None:
            raise RuntimeError("shared gateway runtime is not running")

        config = load_gateway_config()
        base_config = None
        for platform, platform_config in config.platforms.items():
            if _platform_name(platform) == "weixin":
                base_config = platform_config
                break
        if base_config is None:
            raise RuntimeError("weixin platform is not configured")

        adapter_config = self._build_weixin_platform_config(
            base_config=base_config,
            account=account,
        )
        with _temporary_hermes_home(hermes_home):
            adapter = WeixinAdapter(adapter_config)
        setattr(adapter, "_semantier_adapter_key", key.as_string())
        runner = self._shared_runtime.runner
        wire = getattr(runner, "_wire_adapter_handlers", None)
        if callable(wire):
            wire(adapter)
        message_handler = getattr(runner, "_handle_message", None)
        if callable(message_handler):
            adapter.set_message_handler(message_handler)
        adapter.set_fatal_error_handler(self._handle_managed_weixin_fatal_error)
        if hasattr(runner, "session_store"):
            adapter.set_session_store(runner.session_store)
        busy_handler = getattr(runner, "_handle_active_session_busy_message", None)
        if callable(busy_handler):
            adapter.set_busy_session_handler(busy_handler)
        sync_voice_mode = getattr(runner, "_sync_voice_mode_state_to_adapter", None)
        if callable(sync_voice_mode):
            sync_voice_mode(adapter)
        connected = await adapter.connect()
        if not connected:
            raise RuntimeError("failed to start managed Weixin adapter")
        return adapter

    async def _stop_weixin_adapter(self, adapter: Any) -> None:
        try:
            await adapter.disconnect()
        finally:
            key_text = str(getattr(adapter, "_semantier_adapter_key", "") or "").strip()
            if key_text and self._weixin_adapters.get(key_text) is adapter:
                self._weixin_adapters.pop(key_text, None)

    async def _handle_managed_weixin_fatal_error(self, adapter: Any) -> None:
        key_text = str(getattr(adapter, "_semantier_adapter_key", "") or "").strip()
        if key_text:
            self._weixin_adapters.pop(key_text, None)
            entry = self._adapter_registry.get(key_text)
            if entry is not None:
                self._sync_registry_entry(key=entry.key, state="degraded", adapter=None)
        await self._stop_weixin_adapter(adapter)


def _discover_saved_weixin_accounts(runtime_root: Path) -> list[tuple[Path, dict]]:
    candidates: list[tuple[Path, dict]] = []
    try:
        from agents.auth_db import load_weixin_runtime_accounts
        from runtime_paths import workspace_hermes_home_path

        for account in load_weixin_runtime_accounts():
            if not isinstance(account, dict):
                continue
            account_id = str(account.get("account_id") or "").strip()
            token = str(account.get("token") or "").strip()
            if not account_id or not token:
                continue
            workspace_id = str(account.get("owner_workspace_id") or account.get("workspace_id") or "").strip()
            hermes_home = workspace_hermes_home_path(workspace_id) if workspace_id else runtime_root
            normalized = dict(account)
            normalized["account_id"] = account_id
            normalized["token"] = token
            candidates.append((hermes_home, normalized))
    except Exception:
        pass
    return candidates


def _weixin_account_rank(candidate: tuple[Path, dict]) -> tuple[int, str, str]:
    _hermes_home, account = candidate
    state = str(account.get("runtime_session_state") or "").strip().lower()
    active_score = 1 if state in {"active", "connected", "ok"} else 0
    updated_at = str(
        account.get("runtime_session_updated_at")
        or account.get("updated_at")
        or account.get("saved_at")
        or ""
    )
    account_id = str(account.get("account_id") or "")
    return active_score, updated_at, account_id


def _feishu_config_rank(config: dict[str, Any]) -> tuple[int, str, str]:
    app_id = str(config.get("app_id") or "").strip()
    app_secret = str(config.get("app_secret") or "").strip()
    valid_score = 1 if app_id and app_secret else 0
    updated_at = str(
        config.get("updated_at")
        or config.get("validated_at")
        or config.get("saved_at")
        or ""
    )
    owner_user_id = str(
        config.get("owner_user_id") or config.get("manager_user_id") or ""
    )
    return valid_score, updated_at, owner_user_id


def _hydrate_saved_feishu_gateway_env() -> None:
    try:
        from agents.auth_db import list_feishu_bot_configs, load_users

        configs = [
            dict(config)
            for config in list_feishu_bot_configs()
            if isinstance(config, dict)
        ]
    except Exception:
        return

    candidates = [
        config
        for config in configs
        if str(config.get("app_id") or "").strip()
        and str(config.get("app_secret") or "").strip()
    ]
    if not candidates:
        return

    env_app_id = str(os.environ.get("FEISHU_APP_ID") or "").strip()
    env_app_secret = str(os.environ.get("FEISHU_APP_SECRET") or "").strip()
    selected = None
    if env_app_id and env_app_secret:
        for config in candidates:
            if (
                str(config.get("app_id") or "").strip() == env_app_id
                and str(config.get("app_secret") or "").strip() == env_app_secret
            ):
                selected = config
                break
    if selected is None:
        selected = max(candidates, key=_feishu_config_rank)

    os.environ["FEISHU_APP_ID"] = str(selected.get("app_id") or "").strip()
    os.environ["FEISHU_APP_SECRET"] = str(selected.get("app_secret") or "").strip()
    for env_key, config_key, default in (
        ("FEISHU_DOMAIN", "domain", "feishu"),
        ("FEISHU_CONNECTION_MODE", "connection_mode", "websocket"),
        ("FEISHU_HOME_CHANNEL", "home_channel", ""),
    ):
        value_text = str(selected.get(config_key) or default).strip()
        if value_text:
            os.environ[env_key] = value_text

    owner_user_id = str(
        selected.get("owner_user_id") or selected.get("manager_user_id") or ""
    ).strip()
    if not owner_user_id:
        return
    try:
        user = load_users().get(owner_user_id)
    except Exception:
        user = None
    if not isinstance(user, dict):
        return
    feishu_open_id = str(user.get("feishu_open_id") or "").strip()
    if not feishu_open_id:
        return
    allowed = {
        item.strip()
        for item in os.environ.get("FEISHU_ALLOWED_USERS", "").split(",")
        if item.strip()
    }
    allowed.add(feishu_open_id)
    os.environ["FEISHU_ALLOWED_USERS"] = ",".join(sorted(allowed))


def _hydrate_saved_weixin_gateway_env(runtime_root: Path) -> None:
    """Load one saved Weixin account into env before Hermes config is parsed.

    The settings reconnect endpoint already does this before restarting the
    embedded gateway. This startup path covers `semantier run --replace`.
    """
    accounts = _discover_saved_weixin_accounts(runtime_root)
    unique: dict[tuple[str, str], tuple[Path, dict]] = {}
    for hermes_home, account in accounts:
        key = (str(account["account_id"]), str(account["token"]))
        unique.setdefault(key, (hermes_home, account))
    env_account_id = str(os.environ.get("WEIXIN_ACCOUNT_ID") or "").strip()
    env_token = str(os.environ.get("WEIXIN_TOKEN") or "").strip()
    if env_account_id and env_token:
        selected = unique.get((env_account_id, env_token))
        if selected is not None:
            hermes_home, account = selected
        elif unique:
            hermes_home, account = max(unique.values(), key=_weixin_account_rank)
        else:
            return
    elif len(unique) == 1:
        hermes_home, account = next(iter(unique.values()))
    elif unique:
        hermes_home, account = max(unique.values(), key=_weixin_account_rank)
    else:
        return

    os.environ["WEIXIN_ACCOUNT_ID"] = str(account["account_id"])
    os.environ["WEIXIN_TOKEN"] = str(account["token"])
    for env_key, account_key in (
        ("WEIXIN_BASE_URL", "base_url"),
        ("WEIXIN_CDN_BASE_URL", "cdn_base_url"),
        ("WEIXIN_DM_POLICY", "dm_policy"),
        ("WEIXIN_GROUP_POLICY", "group_policy"),
        ("WEIXIN_ALLOWED_USERS", "allow_from"),
        ("WEIXIN_GROUP_ALLOWED_USERS", "group_allow_from"),
        ("WEIXIN_HOME_CHANNEL", "home_channel"),
    ):
        value = account.get(account_key)
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item).strip() for item in value if str(item).strip())
        value_text = str(value or "").strip()
        if value_text:
            os.environ[env_key] = value_text
    if not str(account.get("allow_from") or "").strip():
        user_id = str(account.get("user_id") or "").strip()
        if user_id:
            os.environ["WEIXIN_ALLOWED_USERS"] = user_id

    current_home = os.environ.get("HERMES_HOME")
    try:
        current_is_runtime_root = (
            Path(current_home).expanduser().resolve() == runtime_root.resolve()
            if current_home
            else True
        )
    except OSError:
        current_is_runtime_root = False
    if current_is_runtime_root and hermes_home != runtime_root:
        os.environ["HERMES_HOME"] = str(hermes_home)
        os.environ.pop("SEMANTIER_WORKSPACE_RUNS_DIR", None)


def _refresh_gateway_runtime_home() -> None:
    """Rebind cached gateway config/env paths to the shared runtime root.

    The embedded gateway can restart in-process for a workspace-owned Weixin
    account. Request/session state should follow the workspace-scoped
    ``HERMES_HOME``, but provider/model config must continue to resolve from
    the shared launcher-managed runtime root (``SEMANTIER_LOCAL_STATE_DIR`` /
    ``.semantier-home``). ``gateway.run`` caches those config paths at import
    time, so reconnect must refresh them before the next start.
    """

    target_path = _shared_runtime_root()
    gateway_run = sys.modules.get("gateway.run")
    if gateway_run is None:
        return

    setattr(gateway_run, "_hermes_home", target_path)
    setattr(gateway_run, "_env_path", target_path / ".env")
    setattr(gateway_run, "_config_path", target_path / "config.yaml")

    gateway_runner = getattr(gateway_run, "GatewayRunner", None)
    if gateway_runner is not None and hasattr(gateway_runner, "_VOICE_MODE_PATH"):
        setattr(gateway_runner, "_VOICE_MODE_PATH", target_path / "gateway_voice_mode.json")

    reload_runtime_env = getattr(
        gateway_run,
        "_reload_runtime_env_preserving_config_authority",
        None,
    )
    if callable(reload_runtime_env):
        try:
            reload_runtime_env()
        except Exception:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class _HermesHomeContext:
    previous_home: str | None
    context_token: Any
    reset_active_hermes_home: Callable[[Any], None] | None


class _temporary_hermes_home:
    def __init__(self, hermes_home: str):
        self._target_home = str(hermes_home or "").strip()
        self._state: _HermesHomeContext | None = None
        self._env_cm = None

    def __enter__(self):
        if not self._target_home:
            return
        context_token = None
        reset_active_hermes_home = None
        try:
            from hermes_constants import reset_active_hermes_home as _reset_home, set_active_hermes_home

            context_token = set_active_hermes_home(Path(self._target_home))
            reset_active_hermes_home = _reset_home
        except Exception:
            context_token = None
            reset_active_hermes_home = None

        # Cross-gateway workspace identity binding lives in runtime_paths.
        # Session-scoped requests add SEMANTIER_WORKSPACE_RUNS_DIR separately.
        from runtime_paths import bind_workspace_env

        self._env_cm = bind_workspace_env(self._target_home)
        self._env_cm.__enter__()
        self._state = _HermesHomeContext(
            previous_home=None,
            context_token=context_token,
            reset_active_hermes_home=reset_active_hermes_home,
        )

    def __exit__(self, exc_type, exc, tb):
        if self._state is None:
            return False
        if self._state.context_token is not None and self._state.reset_active_hermes_home is not None:
            try:
                self._state.reset_active_hermes_home(self._state.context_token)
            except Exception:
                pass
        if self._env_cm is not None:
            try:
                self._env_cm.__exit__(exc_type, exc, tb)
            finally:
                self._env_cm = None
        return False


class _temporary_platform_hermes_home:
    """Bind Hermes' shared platform home without exporting workspace identity."""

    def __init__(self, hermes_home: str):
        self._target_home = str(hermes_home or "").strip()
        self._previous_home: str | None = None
        self._home_was_set = False
        self._context_tokens: list[tuple[Callable[[Any], None], Any]] = []

    def __enter__(self):
        if not self._target_home:
            return self
        target = Path(self._target_home)
        target.mkdir(parents=True, exist_ok=True)
        self._previous_home = os.environ.get("HERMES_HOME")
        self._home_was_set = "HERMES_HOME" in os.environ
        os.environ["HERMES_HOME"] = str(target)
        try:
            from hermes_constants import (
                reset_active_hermes_home,
                reset_hermes_home_override,
                set_active_hermes_home,
                set_hermes_home_override,
            )

            self._context_tokens.append(
                (reset_active_hermes_home, set_active_hermes_home(target))
            )
            self._context_tokens.append(
                (reset_hermes_home_override, set_hermes_home_override(target))
            )
        except Exception:
            for reset, token in reversed(self._context_tokens):
                try:
                    reset(token)
                except Exception:
                    pass
            self._context_tokens = []
        return self

    def __exit__(self, exc_type, exc, tb):
        for reset, token in reversed(self._context_tokens):
            try:
                reset(token)
            except Exception:
                pass
        self._context_tokens = []
        if self._home_was_set:
            os.environ["HERMES_HOME"] = self._previous_home or ""
        else:
            os.environ.pop("HERMES_HOME", None)
        return False


_SERVICE: EmbeddedHermesGatewayService | None = None


def get_embedded_hermes_gateway_service() -> EmbeddedHermesGatewayService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = EmbeddedHermesGatewayService()
    return _SERVICE
