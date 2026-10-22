from __future__ import annotations

import sys
import types

from agents.auth_session import request_context_from_request
from agents.hermes_embedded_api import get_embedded_hermes_api_service
from agents.webapi import (
    auth_routes as _auth_routes,
    embedded_proxy_routes as _embedded_proxy_routes,
    gateway_channel_routes as _gateway_channel_routes,
    job_routes as _job_routes,
    ontology_building_routes as _ontology_building_routes,
    session_routes as _session_routes,
    system_skill_routes as _system_skill_routes,
)
from agents.webapi import embedded_proxy_service as _embedded_proxy_service
from agents.webapi import deps as _deps
from agents.webapi import gateway_channel_service as _gateway_channel_service
from agents.webapi import inventory_service as _inventory_service
from agents.webapi import job_service as _job_service
from agents.webapi import organization_service as _organization_service
from agents.webapi import system_skill_service as _system_skill_service
from agents.webapi.router import router
from agents.workspace_session_logs import (
    get_workspace_session_detail,
    get_workspace_session_log_payload,
    get_workspace_session_messages,
    resolve_workspace_session_id,
)

httpx = _embedded_proxy_service.httpx

# Private compatibility exports retained for tests and downstream callers during
# the webapi split. Each entry is owner, reason, and removal condition.
COMPATIBILITY_EXPORT_METADATA: dict[str, dict[str, str]] = {
    "request_context_from_request": {
        "owner": "agents.auth_session",
        "reason": "tests patch authenticated route context through the stable gateway module",
        "removal_condition": "tests patch route/service owners directly",
    },
    "get_embedded_hermes_api_service": {
        "owner": "agents.hermes_embedded_api",
        "reason": "embedded proxy tests replace the local service accessor",
        "removal_condition": "embedded proxy tests patch agents.webapi.embedded_proxy_service",
    },
    "httpx": {
        "owner": "agents.webapi.embedded_proxy_service",
        "reason": "embedded proxy tests replace AsyncClient without live network IO",
        "removal_condition": "embedded proxy tests patch agents.webapi.embedded_proxy_service",
    },
    "_compat_hermes_home_bound": {
        "owner": "agents.webapi.embedded_proxy_service",
        "reason": "embedded proxy tests neutralize scoped HERMES_HOME binding",
        "removal_condition": "embedded proxy tests patch agents.webapi.embedded_proxy_service",
    },
    "_forward_embedded_request": {
        "owner": "agents.webapi.embedded_proxy_service",
        "reason": "route allowlist tests replace forwarding at the stable gateway module",
        "removal_condition": "route tests patch agents.webapi.embedded_proxy_routes",
    },
    "_post_embedded_chat_completion": {
        "owner": "agents.webapi.embedded_proxy_service",
        "reason": "session chat tests replace embedded completion transport",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_stream_embedded_chat_completion": {
        "owner": "agents.webapi.embedded_proxy_service",
        "reason": "session stream tests replace embedded completion transport",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_read_workspace_weixin_account": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "messaging reconnect tests replace workspace account lookup",
        "removal_condition": "messaging tests patch agents.webapi.gateway_channel_routes",
    },
    "get_active_weixin_correlation_for_user": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "messaging reconnect tests replace governed correlation lookup",
        "removal_condition": "messaging tests patch agents.webapi.gateway_channel_routes",
    },
    "_persist_gateway_reconnect_artifact": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "messaging reconnect tests suppress artifact writes",
        "removal_condition": "messaging tests patch agents.webapi.gateway_channel_routes",
    },
    "_restart_embedded_gateway_for_feishu_account": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "Feishu route tests replace gateway restart side effects",
        "removal_condition": "Feishu tests patch route/service owners directly",
    },
    "_ensure_weixin_workspace_pairing_approved": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "pairing tests exercise the helper through the stable gateway module",
        "removal_condition": "pairing tests import agents.webapi.gateway_channel_service",
    },
    "_restart_embedded_gateway_for_weixin_account": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "messaging tests exercise Weixin restart through the stable gateway module",
        "removal_condition": "messaging tests import agents.webapi.gateway_channel_service",
    },
    "_workspace_id_from_hermes_home_path": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "embedded proxy and job tests replace workspace-home inference",
        "removal_condition": "tests patch agents.webapi.gateway_channel_service",
    },
    "_remove_workspace_weixin_account_files": {
        "owner": "agents.webapi.gateway_channel_service",
        "reason": "cache cleanup tests exercise the helper through the stable gateway module",
        "removal_condition": "cache cleanup tests import agents.webapi.gateway_channel_service",
    },
    "_repo_local_marketplace_root": {
        "owner": "agents.webapi.system_skill_service",
        "reason": "marketplace install tests replace the repo-local source root",
        "removal_condition": "marketplace tests patch agents.webapi.system_skill_service",
    },
    "list_plugins_inventory": {
        "owner": "agents.webapi.inventory_service",
        "reason": "marketplace install tests replace inventory reads",
        "removal_condition": "marketplace tests patch route/service owners directly",
    },
    "_list_workspace_cron_jobs": {
        "owner": "agents.webapi.job_service",
        "reason": "meeting coordinator tests replace cron persistence reads",
        "removal_condition": "meeting coordinator tests patch agents.webapi.job_service",
    },
    "_create_workspace_cron_job": {
        "owner": "agents.webapi.job_service",
        "reason": "meeting coordinator tests replace cron persistence writes",
        "removal_condition": "meeting coordinator tests patch agents.webapi.job_service",
    },
    "_temporary_hermes_home": {
        "owner": "agents.webapi.job_service",
        "reason": "job and meeting tests neutralize temporary HERMES_HOME binding",
        "removal_condition": "tests patch agents.webapi.job_service",
    },
    "_create_session_record": {
        "owner": "agents.webapi.session_routes",
        "reason": "session route tests replace lightweight record creation",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_update_session_record": {
        "owner": "agents.webapi.session_routes",
        "reason": "session route tests replace lightweight record update",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_ensure_runtime_session_hydrated": {
        "owner": "agents.webapi.session_routes",
        "reason": "session stream tests replace hydration",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_resolve_workspace_runtime_session_id": {
        "owner": "agents.webapi.session_routes",
        "reason": "session hydration tests replace workspace session resolution",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "_resolve_or_create_workspace_runtime_session_id": {
        "owner": "agents.webapi.session_routes",
        "reason": "session lane parity tests exercise workspace session resolution",
        "removal_condition": "session tests patch agents.webapi.session_routes",
    },
    "get_workspace_session_detail": {
        "owner": "agents.workspace_session_logs",
        "reason": "session hydration tests replace workspace log reads",
        "removal_condition": "session tests patch agents.workspace_session_logs or session route owners",
    },
    "get_workspace_session_messages": {
        "owner": "agents.workspace_session_logs",
        "reason": "session hydration tests replace workspace message reads",
        "removal_condition": "session tests patch agents.workspace_session_logs or session route owners",
    },
    "get_workspace_session_log_payload": {
        "owner": "agents.workspace_session_logs",
        "reason": "session log tests replace workspace log reads",
        "removal_condition": "session tests patch agents.workspace_session_logs or session route owners",
    },
    "resolve_workspace_session_id": {
        "owner": "agents.workspace_session_logs",
        "reason": "session resolver tests keep legacy alias fallback reachable",
        "removal_condition": "session tests patch agents.workspace_session_logs or session route owners",
    },
    "_RATE_LIMIT_STATE_LOCK": {
        "owner": "agents.webapi.deps",
        "reason": "multitenant control tests reset shared rate-limit state",
        "removal_condition": "tests import agents.webapi.deps directly",
    },
    "_RATE_LIMIT_STATE": {
        "owner": "agents.webapi.deps",
        "reason": "multitenant control tests reset shared rate-limit state",
        "removal_condition": "tests import agents.webapi.deps directly",
    },
    "_RATE_LIMITS": {
        "owner": "agents.webapi.deps",
        "reason": "multitenant control tests override bucket limits",
        "removal_condition": "tests import agents.webapi.deps directly",
    },
    "_enforce_rate_limit": {
        "owner": "agents.webapi.deps",
        "reason": "multitenant control tests exercise rate-limit scoping directly",
        "removal_condition": "tests import agents.webapi.deps directly",
    },
    "_update_workspace_cron_job": {
        "owner": "agents.webapi.job_service",
        "reason": "meeting coordinator tests replace cron persistence updates",
        "removal_condition": "meeting coordinator tests patch agents.webapi.job_service",
    },
    "list_organization_members": {
        "owner": "agents.webapi.organization_service",
        "reason": "organization payload tests replace governed membership listing",
        "removal_condition": "organization tests patch agents.webapi.organization_service",
    },
    "list_organization_events": {
        "owner": "agents.webapi.organization_service",
        "reason": "organization payload tests replace governed event listing",
        "removal_condition": "organization tests patch agents.webapi.organization_service",
    },
    "get_organization_materialization_policy": {
        "owner": "agents.webapi.organization_service",
        "reason": "organization payload tests replace materialization policy reads",
        "removal_condition": "organization tests patch agents.webapi.organization_service",
    },
    "get_pending_role_notifications_for_user": {
        "owner": "agents.webapi.organization_service",
        "reason": "organization payload tests replace role notification reads",
        "removal_condition": "organization tests patch agents.webapi.organization_service",
    },
    "get_user_organization_context": {
        "owner": "agents.webapi.organization_service",
        "reason": "real-company setup tests replace governed organization context",
        "removal_condition": "real-company tests patch agents.webapi.organization_service",
    },
    "_organization_payload_for_context": {
        "owner": "agents.webapi.organization_service",
        "reason": "real-company setup tests exercise organization payload composition directly",
        "removal_condition": "real-company tests import agents.webapi.organization_service",
    },
}


_EXPORT_OWNERS: dict[str, list[types.ModuleType]] = {
    "request_context_from_request": [
        _auth_routes,
        _gateway_channel_routes,
        _system_skill_routes,
    ],
    "get_embedded_hermes_api_service": [_embedded_proxy_service],
    "httpx": [_embedded_proxy_service],
    "_compat_hermes_home_bound": [_embedded_proxy_service],
    "_forward_embedded_request": [_embedded_proxy_service, _embedded_proxy_routes],
    "_post_embedded_chat_completion": [_embedded_proxy_service, _session_routes],
    "_stream_embedded_chat_completion": [_embedded_proxy_service, _session_routes],
    "_read_workspace_weixin_account": [_gateway_channel_service, _gateway_channel_routes],
    "get_active_weixin_correlation_for_user": [
        _gateway_channel_service,
        _gateway_channel_routes,
    ],
    "_persist_gateway_reconnect_artifact": [_gateway_channel_service, _gateway_channel_routes],
    "_restart_embedded_gateway_for_feishu_account": [
        _gateway_channel_service,
        _gateway_channel_routes,
        _auth_routes,
    ],
    "_ensure_weixin_workspace_pairing_approved": [_gateway_channel_service],
    "_restart_embedded_gateway_for_weixin_account": [_gateway_channel_service],
    "_workspace_id_from_hermes_home_path": [
        _gateway_channel_service,
        _embedded_proxy_service,
        _job_service,
    ],
    "_remove_workspace_weixin_account_files": [_gateway_channel_service],
    "_repo_local_marketplace_root": [_system_skill_service, _system_skill_routes],
    "list_plugins_inventory": [_inventory_service, _system_skill_service, _system_skill_routes],
    "_list_workspace_cron_jobs": [_job_service],
    "_create_workspace_cron_job": [_job_service],
    "_temporary_hermes_home": [_job_service],
    "_create_session_record": [_session_routes],
    "_update_session_record": [_session_routes],
    "_ensure_runtime_session_hydrated": [_session_routes],
    "_resolve_workspace_runtime_session_id": [_session_routes],
    "_resolve_or_create_workspace_runtime_session_id": [_session_routes],
    "get_workspace_session_detail": [_session_routes],
    "get_workspace_session_messages": [_session_routes],
    "get_workspace_session_log_payload": [_session_routes],
    "resolve_workspace_session_id": [_session_routes],
    "_RATE_LIMIT_STATE_LOCK": [_deps],
    "_RATE_LIMIT_STATE": [_deps],
    "_RATE_LIMITS": [_deps],
    "_enforce_rate_limit": [_deps],
    "_update_workspace_cron_job": [_job_service],
    "list_organization_members": [_organization_service],
    "list_organization_events": [_organization_service],
    "get_organization_materialization_policy": [_organization_service],
    "get_pending_role_notifications_for_user": [_organization_service],
    "get_user_organization_context": [_organization_service],
    "_organization_payload_for_context": [_organization_service],
}


for _name in COMPATIBILITY_EXPORT_METADATA:
    if _name not in globals():
        for _module in _EXPORT_OWNERS.get(_name, []):
            if hasattr(_module, _name):
                globals()[_name] = getattr(_module, _name)
                break


class _CompatibilityModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _EXPORT_OWNERS.get(name, []):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _CompatibilityModule

__all__ = ["router", *COMPATIBILITY_EXPORT_METADATA.keys()]
