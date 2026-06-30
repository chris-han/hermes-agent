from __future__ import annotations

from typing import Any, Iterable


_SEMANTIER_ACP_DEFAULT_TOOLSETS = ("hermes-acp",)


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
    """Resolve ACP toolsets from the ACP platform profile."""
    return resolve_platform_enabled_toolsets(
        config,
        runtime_platform="acp",
        fallback_toolsets=_SEMANTIER_ACP_DEFAULT_TOOLSETS,
        mcp_server_names=mcp_server_names,
    )
