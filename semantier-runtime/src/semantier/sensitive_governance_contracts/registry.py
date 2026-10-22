from __future__ import annotations

CONTRACT_FAMILY = "semantier.sensitive_governance"
CURRENT_CONTRACT_VERSION = "sensitive_governance.v1"
SUPPORTED_CONTRACT_VERSIONS = frozenset({CURRENT_CONTRACT_VERSION})


def require_supported_contract_version(version: str | None) -> str:
    resolved = (version or CURRENT_CONTRACT_VERSION).strip()
    if resolved not in SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported sensitive governance contract version: {resolved}")
    return resolved
