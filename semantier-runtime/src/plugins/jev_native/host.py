"""Fail-closed binding to the explicitly qualified Hermes native-route policy.

This is host route permission only. It grants no Semantier semantic admission,
paid experiment authorization, credentials, or transport capability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import MappingProxyType

PLUGIN_ID = "jev_native"
PROVIDER = "typesafe"
MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
HOST_COMMIT = "bd7a1ba0087ad4f889840232647d2c05939fd545"
HOST_SOURCE_SHA256 = MappingProxyType(dict((
    ("hermes_constants", "026ce08a0f1287d151a53a6af88c934d85f2bb7ce528d419a5e25742f7f02bb8"),
    ("agent.plugin_llm", "a5efb0ffdbcb57806b9a2b65d1d0f749e06eb7048b3f8beb988d83b1c4dc7f4d"),
    ("hermes_cli.plugins", "0e2d82b1abcbda95d4c96c40daef0887d728bfd12a1e0c9bc155d0ebd335fc70"),
    ("hermes_cli.plugins_discovery", "53455c08695e1422fc53e0240abb0724f8c20571775a0b81033cf647a02f3a1c"),
    ("hermes_cli.config", "676ebcbb384e54f83365c5de49ee9bbb679a61ff8ca4e0635e54d870d64a8fd8"),
    ("agent.secret_scope", "04b68de1d35619dbe98263eb76a8fd5eb605a4a49e69741c72bda1d4b56d472e"),
)))


class HostContractError(PermissionError):
    """The exact qualified host/profile route cannot be established."""


@dataclass(frozen=True)
class RouteBinding:
    profile_home: str
    config_file_sha256: str
    config_policy_sha256: str
    host_source_sha256: tuple[tuple[str, str], ...]
    provider: str = PROVIDER
    model: str = MODEL
    endpoint: str = ENDPOINT
    task: str = PLUGIN_ID


def verify_host_sources() -> None:
    """Qualify the exact host helpers before importing/calling private policy."""
    try:
        for name, expected in HOST_SOURCE_SHA256.items():
            spec = importlib.util.find_spec(name)
            if spec is None or not spec.origin:
                raise HostContractError("HOST_SOURCE_UNAVAILABLE")
            if hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
                raise HostContractError("HOST_SOURCE_MISMATCH")
    except HostContractError:
        raise
    except Exception:
        raise HostContractError("HOST_SOURCE_UNAVAILABLE") from None


def _jsonable(value):
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def resolve_route(ctx, *, task=PLUGIN_ID, expected_profile_home=None) -> RouteBinding:
    """Read current host policy every time; return immutable credential-free data.

    The caller must revalidate this binding immediately before its separately
    authorized dispatch. A binding is evidence, never an executable permit.
    """
    verify_host_sources()
    try:
        plugins = importlib.import_module("hermes_cli.plugins")
        discovery = importlib.import_module("hermes_cli.plugins_discovery")
        policy_api = importlib.import_module("agent.plugin_llm")
        config_api = importlib.import_module("hermes_cli.config")
        constants = importlib.import_module("hermes_constants")

        if type(ctx) is not plugins.PluginContext or ctx.plugin_id != PLUGIN_ID:
            raise HostContractError("HOST_CONTEXT_INVALID")
        home = constants.get_hermes_home().expanduser().resolve()
        manager = ctx._manager
        if (manager is not plugins.get_plugin_manager()
                or Path(manager.scope_key).resolve() != home
                or manager.home_path.resolve() != home):
            raise HostContractError("HOST_PROFILE_MISMATCH")
        if expected_profile_home is not None and Path(expected_profile_home).expanduser().resolve() != home:
            raise HostContractError("HOST_PROFILE_MISMATCH")
        loaded = manager._plugins.get(PLUGIN_ID)
        if (loaded is None or loaded.manifest is not ctx.manifest
                or not loaded.enabled or loaded.error or loaded.module is None):
            raise HostContractError("HOST_PLUGIN_NOT_LOADED")
        config_path = home / "config.yaml"
        if not config_path.is_file() or os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1":
            raise HostContractError("HOST_PROFILE_CONFIG_MISSING")
        before = config_path.read_bytes()
        # Raw-file diagnostics reject malformed/non-mapping current files
        # before merged reads can recover the last-known-good configuration.
        # Actual behavior below still uses only the host's merged config APIs.
        if not config_api.read_user_config_raw(config_path):
            raise HostContractError("HOST_PROFILE_CONFIG_EMPTY")
        gate = discovery.gate_manifest(
            ctx.manifest, discovery._get_disabled_plugins(), discovery._get_enabled_plugins())
        if gate.action != "load":
            raise HostContractError("HOST_PLUGIN_DISABLED")

        connection = ctx.get_config("connection")
        if (not isinstance(connection, dict)
                or connection.get("backend") != PROVIDER
                or connection.get("model") != MODEL
                or set(connection) - {"backend", "model", "endpoint", "requested_endpoint"}
                or connection.get("endpoint", ENDPOINT) != ENDPOINT
                or connection.get("requested_endpoint", ENDPOINT) != ENDPOINT):
            raise HostContractError("HOST_CONNECTION_INVALID")

        policy = policy_api._resolve_trust_policy(PLUGIN_ID)
        resolved = policy_api._check_overrides(
            policy, requested_provider=PROVIDER, requested_model=MODEL,
            requested_agent_id=None, requested_profile=None)
        resolved_task = policy_api._check_task(policy, plugin_id=PLUGIN_ID, requested_task=task)
        if resolved != (PROVIDER, MODEL, None, None) or resolved_task != PLUGIN_ID:
            raise HostContractError("HOST_ROUTE_NOT_FIXED")
        if config_path.read_bytes() != before or constants.get_hermes_home().resolve() != home:
            raise HostContractError("HOST_CONFIG_CHANGED")

        file_hash = hashlib.sha256(before).hexdigest()
        fingerprints = tuple(sorted(HOST_SOURCE_SHA256.items()))
        evidence = {
            "profile_home": str(home), "config_file_sha256": file_hash,
            "host_source_sha256": fingerprints, "policy": _jsonable(asdict(policy)),
            "provider": PROVIDER, "model": MODEL, "endpoint": ENDPOINT, "task": resolved_task,
        }
        policy_hash = hashlib.sha256(json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        return RouteBinding(str(home), file_hash, policy_hash, fingerprints)
    except HostContractError:
        raise
    except Exception:
        # Configuration parse/policy errors may embed arbitrary configuration
        # values. Only a static denial leaves this boundary.
        raise HostContractError("HOST_ROUTE_DENIED") from None
