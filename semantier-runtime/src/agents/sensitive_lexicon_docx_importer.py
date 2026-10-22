from __future__ import annotations

from pathlib import Path
from typing import Any

from plugins.tender_sensitive_term_detection_modeling import compiler

PROFILE_VERSION = compiler.PROFILE_VERSION
NORMALIZATION_POLICY_VERSION = compiler.NORMALIZATION_POLICY_VERSION
DEFAULT_RULE_FAMILY = compiler.DEFAULT_RULE_FAMILY
DEFAULT_ISSUE_TYPE = compiler.DEFAULT_ISSUE_TYPE
DEFAULT_SEVERITY = compiler.DEFAULT_SEVERITY


def normalize_sensitive_term(value: str) -> str:
    return compiler.normalize_sensitive_term(value)


def import_sensitive_lexicon_docx(
    *,
    normalized_document_artifact_ref: str,
    organization_id: str,
    workspace_id: str | None,
    knowledge_source_id: str,
    source_ref: str,
    created_by: str,
    proposed_workspace_scope: str = "workspace",
    proposed_workspace_id: str | None = None,
    source_version: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    return compiler.compile_sensitive_lexicon(
        normalized_document_artifact_ref=normalized_document_artifact_ref,
        organization_id=organization_id,
        workspace_id=workspace_id,
        knowledge_source_id=knowledge_source_id,
        source_ref=source_ref,
        created_by=created_by,
        proposed_workspace_scope=proposed_workspace_scope,
        proposed_workspace_id=proposed_workspace_id,
        source_version=source_version,
        db_path=db_path,
        compiler_profile_version=PROFILE_VERSION,
    )
