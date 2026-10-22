from __future__ import annotations

from .schemas import SUPPORTED_PROFILES


def require_supported_profile(profile_version: str) -> str:
    if profile_version not in SUPPORTED_PROFILES:
        raise ValueError("PARSER_PROFILE_UNSUPPORTED")
    return profile_version
