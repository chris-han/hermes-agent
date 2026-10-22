from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eos.artifact_hashing import content_hash
from eos.legal_corpus_store import LegalCorpusStore


MANIFEST_SCHEMA_VERSION = 1
PENDING_SEMANTIER_SYNC = "PENDING_SEMANTIER_SYNC"

_AUTHORITY_LEVELS = {
    "national_law",
    "administrative_regulation",
    "departmental_rule",
    "policy_document",
}


def import_procurement_manifest(
    *,
    manifest_path: Path | str,
    store: LegalCorpusStore,
    organization_id: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("procurement manifest must be a mapping")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported procurement manifest schema_version")
    if payload.get("jurisdiction") != "CN":
        raise ValueError("procurement manifest jurisdiction must be CN")
    sync = payload.get("sync") if isinstance(payload.get("sync"), dict) else {}
    if sync.get("github_actions_allowed") is not False:
        raise ValueError("procurement manifest must forbid GitHub authority sync")
    if sync.get("external_scheduler_allowed") is not False:
        raise ValueError("procurement manifest must forbid external schedulers")
    governance = payload.get("governance") if isinstance(payload.get("governance"), dict) else {}
    prepared_by = _required(governance, "prepared_by")
    approved_by = _required(governance, "approved_by")
    approved_at = _required(governance, "approved_at")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("procurement manifest sources must be a non-empty list")

    imported: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    root = path.parent
    for entry in sources:
        if not isinstance(entry, dict):
            raise ValueError("procurement manifest source entry must be a mapping")
        source_id = _required_ascii(entry, "source_id")
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate procurement source_id: {source_id}")
        seen_source_ids.add(source_id)
        canonical_title = _required(entry, "canonical_title")
        document_number = _required(entry, "document_number")
        issuer = _required(entry, "authority")
        authority_tier = _required(entry, "authority_level")
        if authority_tier not in _AUTHORITY_LEVELS:
            raise ValueError(f"unsupported authority_level for {source_id}")
        legal_state = _required(entry, "legal_state").upper()
        if legal_state not in {"ACTIVE", "PENDING_EFFECTIVE", "DRAFT", "CONSULTATION", "REPEALED", "SUPERSEDED"}:
            raise ValueError(f"unsupported legal_state for {source_id}")
        effective_from = _required(entry, "effective_from")
        local_path = _required(entry, "local_path")
        if Path(local_path).is_absolute() or ".." in Path(local_path).parts:
            raise ValueError(f"invalid local_path for {source_id}")
        artifact_path = root / local_path
        retrieval_status = str(entry.get("retrieval_status") or "").strip()
        artifact_exists = artifact_path.is_file()
        if not artifact_exists:
            retrieval_status = retrieval_status or PENDING_SEMANTIER_SYNC
        if retrieval_status and retrieval_status != PENDING_SEMANTIER_SYNC:
            raise ValueError(f"unsupported retrieval_status for {source_id}")

        source = _get_or_register_source(
            store=store,
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_id=source_id,
            canonical_title=canonical_title,
            issuer=issuer,
            authority_tier=authority_tier,
            document_number=document_number,
            legal_state=legal_state,
        )
        version_identity = str(entry.get("version_label") or "bootstrap").strip()
        source_hash = _source_hash(entry=entry, artifact_path=artifact_path)
        try:
            version = store.create_source_version(
                source_id=source["source_id"],
                version_identity=version_identity,
                source_hash=source_hash,
                effective_from=effective_from,
                lifecycle_state="REGISTERED" if artifact_exists else PENDING_SEMANTIER_SYNC,
                source_provenance={
                    "source_type": "repo_bootstrap_manifest",
                    "source_ref": str(entry.get("canonical_uri") or entry.get("downloadable_uri") or local_path),
                    "manifest_path": str(path),
                    "manifest_source_id": source_id,
                    "retrieval_status": retrieval_status or "PRESERVED",
                    "git_role": "governed_snapshot_only",
                    "prepared_by": prepared_by,
                    "approved_by": approved_by,
                    "approved_at": approved_at,
                },
            )
        except ValueError as exc:
            if str(exc) != "duplicate or ambiguous legal source version":
                raise
            version = store.get_source_version(
                source_id=source["source_id"],
                version_identity=version_identity,
                content_hash=source_hash,
            )
        artifact: dict[str, Any] | None = None
        if artifact_exists:
            content = artifact_path.read_bytes()
            artifact = store.preserve_artifact(
                version_id=version.version_id,
                artifact_kind="bootstrap_official_pdf",
                uri=str(entry.get("downloadable_uri") or entry.get("canonical_uri") or local_path),
                content=content,
                mime_type="application/pdf",
                storage_ref=f"bootstrap/legal_corpus/{source_id}/{artifact_path.name}",
                http_metadata={
                    "manifest_path": str(path),
                    "local_path": local_path,
                    "retrieval_status": "PRESERVED",
                },
                redirect_metadata=[],
            )
        imported.append(
            {
                "source_id": source["source_id"],
                "version_id": version.version_id,
                "artifact_id": artifact["artifact_id"] if artifact else None,
                "retrieval_status": retrieval_status or "PRESERVED",
            }
        )
    return {
        "schema": "legal_procurement_manifest_import.v1",
        "manifest_path": str(path),
        "organization_id": organization_id,
        "source_count": len(imported),
        "pending_semantier_sync_count": sum(
            1 for item in imported if item["retrieval_status"] == PENDING_SEMANTIER_SYNC
        ),
        "imported_sources": imported,
        "activation_performed": False,
    }


def _get_or_register_source(
    *,
    store: LegalCorpusStore,
    organization_id: str,
    workspace_id: str | None,
    source_id: str,
    canonical_title: str,
    issuer: str,
    authority_tier: str,
    document_number: str,
    legal_state: str,
) -> dict[str, Any]:
    try:
        source = store.get_source(source_id)
    except KeyError:
        return store.register_source(
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_id=source_id,
            canonical_title=canonical_title,
            issuer=issuer,
            authority_tier=authority_tier,
            jurisdiction="CN",
            document_number=document_number,
            expected_status=legal_state,
            governance_state="LISTED",
        )
    if source["organization_id"] != organization_id:
        raise ValueError(f"bootstrap source organization mismatch for {source_id}")
    expected = {
        "canonical_title": canonical_title,
        "issuer": issuer,
        "authority_tier": authority_tier,
        "jurisdiction": "CN",
        "document_number": document_number,
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(f"bootstrap source identity mismatch for {source_id}: {field}")
    return source


def _source_hash(*, entry: dict[str, Any], artifact_path: Path) -> str:
    if artifact_path.is_file():
        return content_hash({"sha256_bytes": artifact_path.read_bytes().hex()})
    return content_hash(
        {
            "source_id": entry["source_id"],
            "canonical_title": entry["canonical_title"],
            "document_number": entry["document_number"],
            "retrieval_status": PENDING_SEMANTIER_SYNC,
        }
    )


def _required(entry: dict[str, Any], field: str) -> str:
    value = str(entry.get(field) or "").strip()
    if not value:
        raise ValueError(f"procurement manifest source missing {field}")
    return value


def _required_ascii(entry: dict[str, Any], field: str) -> str:
    value = _required(entry, field)
    if not value.isascii():
        raise ValueError(f"procurement manifest {field} must be ASCII-stable")
    return value
