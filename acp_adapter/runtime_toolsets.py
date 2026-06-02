from __future__ import annotations

from typing import Any, Iterable


_SEMANTIER_ACP_DEFAULT_TOOLSETS = ("hermes-acp",)
_API_SERVER_DEFAULT_TOOLSET = "hermes-api-server"


try:
    from agents.runtime_toolsets import (  # type: ignore
        expand_enabled_toolsets,
        resolve_platform_enabled_toolsets,
    )
except Exception:
    from hermes_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY

    def _normalize_toolset_names(value: Any) -> list[str]:
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                name = str(item or "").strip()
                if name and name not in normalized:
                    normalized.append(name)
            return normalized
        name = str(value or "").strip()
        return [name] if name else []

    def expand_enabled_toolsets(
        toolsets: Iterable[str] | None = None,
        *,
        mcp_server_names: Iterable[str] | None = None,
    ) -> list[str]:
        """Return enabled toolsets plus any explicit MCP-backed toolsets."""
        expanded: list[str] = []

        for name in list(toolsets or []):
            if name and name not in expanded:
                expanded.append(name)

        for server_name in list(mcp_server_names or []):
            toolset_name = f"mcp-{server_name}"
            if server_name and toolset_name not in expanded:
                expanded.append(toolset_name)

        return expanded

    def resolve_platform_enabled_toolsets(
        config: dict[str, Any] | None,
        *,
        runtime_platform: str,
        fallback_toolsets: Iterable[str] | None = None,
        mcp_server_names: Iterable[str] | None = None,
    ) -> list[str]:
        """Resolve the shared runtime tool surface for a target platform."""
        resolved: list[str] = []

        if isinstance(config, dict):
            platform_toolsets = config.get("platform_toolsets")
            if isinstance(platform_toolsets, dict):
                resolved = _normalize_toolset_names(platform_toolsets.get(runtime_platform))

        if not resolved:
            default_platform = _PLATFORMS_REGISTRY.get(runtime_platform)
            default_toolset = getattr(default_platform, "default_toolset", None)
            resolved = list(fallback_toolsets or ([default_toolset] if default_toolset else []))

        return expand_enabled_toolsets(resolved, mcp_server_names=mcp_server_names)


def resolve_semantier_acp_enabled_toolsets(
    config: dict[str, Any] | None,
    *,
    mcp_server_names: Iterable[str] | None = None,
) -> list[str]:
    """Resolve ACP toolsets with an ACP-native default and api_server overrides.

    ACP sessions share the api_server override surface in config, but when no
    explicit override is present they must retain the ACP-specific default
    toolset contract (`hermes-acp`) rather than inheriting `hermes-api-server`.
    """
    platform_toolsets = config.get("platform_toolsets") if isinstance(config, dict) else None
    if not isinstance(platform_toolsets, dict) or "api_server" not in platform_toolsets:
        expanded = list(_SEMANTIER_ACP_DEFAULT_TOOLSETS)
        for server_name in list(mcp_server_names or ()):
            toolset_name = f"mcp-{server_name}"
            if server_name and toolset_name not in expanded:
                expanded.append(toolset_name)
        return expanded

    resolved = resolve_platform_enabled_toolsets(
        config,
        runtime_platform="api_server",
        fallback_toolsets=_SEMANTIER_ACP_DEFAULT_TOOLSETS,
        mcp_server_names=mcp_server_names,
    )
    if _API_SERVER_DEFAULT_TOOLSET not in resolved:
        return resolved

    non_mcp_toolsets = [name for name in resolved if not str(name).startswith("mcp-")]
    mcp_toolsets = [name for name in resolved if str(name).startswith("mcp-")]
    explicit_non_default = [name for name in non_mcp_toolsets if name != _API_SERVER_DEFAULT_TOOLSET]
    return explicit_non_default + [_API_SERVER_DEFAULT_TOOLSET] + mcp_toolsets
