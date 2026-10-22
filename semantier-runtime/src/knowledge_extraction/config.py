"""Repo-owned extraction provider configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).parents[2] / "bootstrap/config/knowledge_extraction.json"
SUPPORTED_PROVIDERS = ("semantica", "semantica_service", "langextract", "legacy")


@dataclass(frozen=True)
class KnowledgeExtractionConfig:
    provider: str
    profile: str
    schema_version: str = "knowledge_extraction_config.v1"


def load_knowledge_extraction_config(path: Path | None = None) -> KnowledgeExtractionConfig:
    payload = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
    values = payload.get("knowledge_extraction") or {}
    provider = os.environ.get(
        "SEMANTIER_KNOWLEDGE_EXTRACTION_PROVIDER", values.get("provider", "semantica")
    )
    profile = os.environ.get("SEMANTIER_KNOWLEDGE_EXTRACTION_PROFILE", values.get("profile"))
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "knowledge extraction provider must be semantica, semantica_service, langextract, or legacy"
        )
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("knowledge extraction profile must be a non-empty string")
    return KnowledgeExtractionConfig(provider=provider, profile=profile.strip())
