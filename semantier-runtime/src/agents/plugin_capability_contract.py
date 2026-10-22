"""Machine-verifiable least-privilege plugin capability contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

CAPABILITY_SCHEMA_VERSION = "semantier.plugin_capabilities.v1"
ALLOWED_VALUES = {"allow", "controlled", "deny"}
RUNTIME_FORBIDDEN_TOOLS = {"mutate_candidate", "checkpoint", "submit_governance", "approve", "admit", "change_authority_binding"}
CURATOR_FORBIDDEN_TOOLS = {"approve", "admit", "change_authority_binding"}


class CapabilityContractError(ValueError):
    """Raised when declared and registered plugin capabilities diverge."""


def load_capability_contract(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    contract = payload.get("capability_contract")
    if not isinstance(contract, Mapping):
        raise CapabilityContractError("plugin.yaml capability_contract is required")
    if contract.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
        raise CapabilityContractError("unsupported plugin capability schema")
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, Mapping) or not capabilities:
        raise CapabilityContractError("capability_contract.capabilities is required")
    invalid = {str(value) for value in capabilities.values()} - ALLOWED_VALUES
    if invalid:
        raise CapabilityContractError(f"invalid capability values: {sorted(invalid)}")
    return {"schema_version": contract["schema_version"], "capabilities": dict(capabilities)}


def validate_registered_tools(
    manifest_path: str | Path,
    registered_tools: Iterable[Mapping[str, Any] | str],
    *,
    capability_to_tools: Mapping[str, set[str]],
) -> dict[str, Any]:
    contract = load_capability_contract(manifest_path)
    declared = contract["capabilities"]
    actual = {str(item if isinstance(item, str) else item.get("name") or "") for item in registered_tools}
    if "" in actual:
        raise CapabilityContractError("registered tool inventory contains an unnamed tool")
    undeclared = sorted(
        tool for tool in actual
        if not any(tool in tools and declared.get(capability) in {"allow", "controlled"}
                   for capability, tools in capability_to_tools.items())
    )
    denied = sorted(
        tool for capability, tools in capability_to_tools.items()
        if declared.get(capability) == "deny"
        for tool in actual if tool in tools
    )
    if undeclared or denied:
        raise CapabilityContractError(f"plugin capability mismatch: undeclared={undeclared}, denied={denied}")
    return {
        "schema_version": contract["schema_version"],
        "declared_capabilities": declared,
        "registered_tools": sorted(actual),
    }
