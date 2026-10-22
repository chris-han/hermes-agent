from __future__ import annotations

from pathlib import Path
from typing import Any

from . import schemas
from .storage import (
    BoundSession,
    artifact_ref,
    bound_session,
    media_type_for,
    resolve_session_ref,
    sha256_bytes,
    sha256_payload,
    utc_now_iso,
    write_json_artifact,
)


def _empty_document(media_type: str, languages: list[str]) -> dict[str, Any]:
    return {
        "schema_version": schemas.CANONICAL_DOCUMENT_SCHEMA_VERSION,
        "source_media_type": media_type,
        "languages": languages,
        "pages": [],
        "blocks": [],
        "tables": [],
        "key_value_candidates": [],
        "barcodes": [],
        "qr_codes": [],
        "evidence": [],
        "parser": {"method": "empty", "version": schemas.PARSER_VERSION},
    }


def _parser_for(path: Path, media_type: str):
    suffix = path.suffix.lower()
    if media_type == "application/pdf" or suffix == ".pdf":
        from .parsers import pymupdf_native

        return pymupdf_native
    if media_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        from .parsers import image

        return image
    if suffix == ".docx":
        from .parsers import ooxml

        return ooxml
    if suffix == ".xlsx":
        from .parsers import xlsx

        return xlsx
    if media_type == "text/html" or suffix in {".html", ".htm"}:
        from .parsers import html

        return html
    if media_type.startswith("text/") or suffix in {".md", ".markdown", ".txt"}:
        from .parsers import text

        return text
    if suffix in {".xml", ".json"}:
        from .parsers import structured

        return structured
    return None


def _hydrate_docx_anchor_refs(
    value: Any,
    *,
    normalized_document_artifact_ref: str,
    source_hash: str,
) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("anchor"), dict):
            value["anchor"]["normalized_document_artifact_ref"] = normalized_document_artifact_ref
            value["anchor"]["source_hash"] = source_hash
        for child in value.values():
            _hydrate_docx_anchor_refs(
                child,
                normalized_document_artifact_ref=normalized_document_artifact_ref,
                source_hash=source_hash,
            )
    elif isinstance(value, list):
        for child in value:
            _hydrate_docx_anchor_refs(
                child,
                normalized_document_artifact_ref=normalized_document_artifact_ref,
                source_hash=source_hash,
            )


def _read_json_artifact(path: Path) -> dict[str, Any]:
    import json

    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise RuntimeError("INVALID_JSON_ARTIFACT")
    return artifact


def _verify_artifact_hash(artifact: dict[str, Any], expected_hash: str | None, error_code: str) -> str:
    actual_hash = str(artifact.get("artifact_hash") or "")
    payload = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    recomputed_hash = sha256_payload(payload)
    if not actual_hash or actual_hash != recomputed_hash:
        raise RuntimeError(error_code)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(error_code)
    return actual_hash


def extract_live(args: dict[str, Any], *, trusted_session: BoundSession | None = None) -> dict[str, Any]:
    if trusted_session is not None:
        if not isinstance(trusted_session, BoundSession):
            raise TypeError("TRUSTED_BOUND_SESSION_REQUIRED")
        if args.get("session_id") and args["session_id"] != trusted_session.session_id:
            raise RuntimeError("SESSION_CLAIM_MISMATCH")
    bound = trusted_session if trusted_session is not None else bound_session(args)
    source = resolve_session_ref(bound, str(args.get("document_ref") or args.get("source_artifact_ref") or ""))
    media_type = media_type_for(source, args.get("media_type_hint"))
    languages = [str(item) for item in args.get("language_hints") or []]
    parser = _parser_for(source, media_type)
    source_hash = sha256_bytes(source.read_bytes())
    source_ref = artifact_ref(bound, source)
    if parser and parser.__name__.endswith((".pymupdf_native", ".ooxml")):
        parsed = parser.parse_bound(
            source, artifact_ref=source_ref, source_sha256=source_hash, media_type=media_type,
            byte_length=source.stat().st_size, execution_scope_ref=f"session:{bound.session_id}",
        )
    else:
        parsed = parser.parse(source, allow_ocr=bool(args.get("allow_ocr", False)), language_hints=languages) if parser else _empty_document(media_type, languages)
    document = {
        "schema_version": parsed.get("schema_version") or schemas.CANONICAL_DOCUMENT_SCHEMA_VERSION,
        "source_media_type": media_type,
        "languages": languages,
        "pages": parsed.get("pages", []),
        "elements": parsed.get("elements", []),
        "blocks": parsed.get("blocks", []),
        "tables": parsed.get("tables", []),
        "key_value_candidates": parsed.get("key_value_candidates", []),
        "barcodes": parsed.get("barcodes", []),
        "qr_codes": parsed.get("qr_codes", []),
        "evidence": parsed.get("evidence", []),
        "parser": parsed.get("parser") or {"method": "unknown", "version": schemas.PARSER_VERSION},
    }
    ref, _ = write_json_artifact(
        bound,
        area="document_extraction",
        name=f"{source.stem}-{source_hash.removeprefix('sha256:')[:12]}",
        payload={
            "document": document,
            "source_artifact_ref": source_ref,
            "source_hash": source_hash,
        },
    )
    if document["schema_version"] == schemas.CANONICAL_DOCUMENT_V2_SCHEMA_VERSION:
        _hydrate_docx_anchor_refs(
            document,
            normalized_document_artifact_ref=ref,
            source_hash=source_hash,
        )
        ref, normalized_path = write_json_artifact(
            bound,
            area="document_extraction",
            name=f"{source.stem}-{source_hash.removeprefix('sha256:')[:12]}",
            payload={
                "document": document,
                "source_artifact_ref": source_ref,
                "source_hash": source_hash,
            },
        )
    else:
        normalized_path = resolve_session_ref(bound, ref)
    normalized_artifact = _read_json_artifact(normalized_path)
    normalized_document_artifact_hash = _verify_artifact_hash(
        normalized_artifact,
        None,
        "NORMALIZED_DOCUMENT_ARTIFACT_HASH_MISMATCH",
    )
    provider_refs = []
    provider_hashes = []
    for index, provider_payload in enumerate(parsed.get("provider_artifacts") or [], start=1):
        provider_ref, _ = write_json_artifact(
            bound,
            area="document_extraction/provider",
            name=f"{source.stem}-provider-{index}",
            payload=provider_payload,
        )
        provider_artifact = _read_json_artifact(resolve_session_ref(bound, provider_ref))
        provider_refs.append(provider_ref)
        provider_hashes.append(_verify_artifact_hash(provider_artifact, None, "PROVIDER_ARTIFACT_HASH_MISMATCH"))
    return {
        "status": "completed",
        "document": document,
        "artifacts": {
            "source_artifact_ref": source_ref,
            "source_hash": source_hash,
            "normalized_document_artifact_ref": ref,
            "normalized_document_artifact_hash": normalized_document_artifact_hash,
            "document_schema_version": document["schema_version"],
            "parser_method": document["parser"].get("method"),
            "parser_version": document["parser"].get("version"),
            "provider_artifact_refs": provider_refs,
            "provider_artifact_hashes": provider_hashes,
        },
        "parser": document["parser"],
        "completed_at": utc_now_iso(),
    }


def extract_replay(args: dict[str, Any], *, trusted_session: BoundSession | None = None) -> dict[str, Any]:
    if trusted_session is not None:
        if not isinstance(trusted_session, BoundSession):
            raise TypeError("TRUSTED_BOUND_SESSION_REQUIRED")
        if args.get("session_id") and args["session_id"] != trusted_session.session_id:
            raise RuntimeError("SESSION_CLAIM_MISMATCH")
    bound = trusted_session if trusted_session is not None else bound_session(args)
    ref = str(args.get("normalized_document_artifact_ref") or "")
    if not ref:
        raise RuntimeError("NORMALIZED_DOCUMENT_ARTIFACT_REF_REQUIRED")
    path = resolve_session_ref(bound, ref)
    artifact = _read_json_artifact(path)
    pins = args.get("pins") or {}
    _verify_artifact_hash(
        artifact,
        pins.get("normalized_document_artifact_hash"),
        "NORMALIZED_DOCUMENT_ARTIFACT_HASH_MISMATCH",
    )
    document = artifact.get("document")
    if not isinstance(document, dict):
        raise RuntimeError("INVALID_CANONICAL_DOCUMENT_ARTIFACT")
    expected_schema = pins.get("document_schema_version")
    if expected_schema and expected_schema != document.get("schema_version"):
        raise RuntimeError("DOCUMENT_SCHEMA_PIN_MISMATCH")
    if pins.get("occurrence_level_required") and document.get("schema_version") != schemas.CANONICAL_DOCUMENT_V2_SCHEMA_VERSION:
        raise RuntimeError("CANONICAL_DOCUMENT_V2_REQUIRED")
    expected_source_hash = pins.get("source_hash")
    if expected_source_hash and expected_source_hash != artifact.get("source_hash"):
        raise RuntimeError("SOURCE_ARTIFACT_HASH_MISMATCH")
    provider_refs = list(args.get("provider_artifact_refs") or [])
    provider_hashes = list(args.get("provider_artifact_hashes") or pins.get("provider_artifact_hashes") or [])
    if provider_hashes and len(provider_refs) != len(provider_hashes):
        raise RuntimeError("PROVIDER_ARTIFACT_HASH_MISMATCH")
    for index, provider_ref in enumerate(provider_refs):
        provider_artifact = _read_json_artifact(resolve_session_ref(bound, provider_ref))
        expected_provider_hash = provider_hashes[index] if index < len(provider_hashes) else None
        _verify_artifact_hash(
            provider_artifact,
            expected_provider_hash,
            "PROVIDER_ARTIFACT_HASH_MISMATCH",
        )
    return {
        "status": "completed",
        "document": document,
        "artifacts": {
            "source_artifact_ref": artifact.get("source_artifact_ref"),
            "source_hash": artifact.get("source_hash"),
            "normalized_document_artifact_ref": ref,
            "normalized_document_artifact_hash": artifact.get("artifact_hash"),
            "document_schema_version": document.get("schema_version"),
            "parser_method": (document.get("parser") or {}).get("method"),
            "parser_version": (document.get("parser") or {}).get("version"),
            "provider_artifact_refs": provider_refs,
            "provider_artifact_hashes": provider_hashes,
        },
        "parser": document.get("parser") or {},
        "completed_at": utc_now_iso(),
    }
