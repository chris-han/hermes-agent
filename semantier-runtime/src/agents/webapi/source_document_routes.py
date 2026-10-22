"""Semantier source-document content route.

This module is part of the governed source-content delivery contract
(`docs/plans/2026-08-22-contextgraph-open-source-source-viewer-integration-v1.md`
section C2). It exposes a single authenticated same-origin read-only route that
resolves a governed `sourceIdentityRef` to the original PDF/DOCX bytes of the
underlying governed artifact.

Contract guarantees:

- the route takes only `sourceIdentityRef` as its source selector — no raw
  filesystem path, no absolute host path, no arbitrary path query/body parameter.
  Two identity shapes are accepted: session-relative governed artifact refs
  (resolved via `StoreBackedGovernedArtifactService`) and `sha256:<hex>` content
  identity refs, which resolve to the byte-identical original upload under the
  authenticated workspace's governed `wiki/uploads/` root. A scoped
  `governed-source:` identity may also resolve a completed prepared workspace
  artifact through its immutable claim and byte-owner pin;
- tenant/workspace scope is verified against the authenticated request context
  before any artifact resolution;
- the response carries a strong ETag matching the governed `sourceHash`, an
  explicit `X-Source-Hash` header, a correct `Content-Type` for PDF or DOCX,
  a `Content-Length` derived from the governed artifact byte length, and a
  `Content-Disposition: inline` so the browser can render the source instead
  of downloading it;
- the body is streamed directly from the governed artifact `BinaryIO` without
  buffering the full bytes in worker memory;
- HTTP `Range` requests are out of scope for v1 and produce `416` without
  body iteration;
- request and response bodies are never written to the audit log.
"""

from __future__ import annotations

import logging
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from storage import hosted_postgres as hosted
from hashlib import sha256
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from agents.route_policy import RouteAuthorizationClass, classify_authorization_class, classify_route
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from services.governed_artifact_service import StoreBackedGovernedArtifactService
from semantier.governed_upload_ref import (
    KNOWLEDGE_UPLOAD_REF_PREFIX,
    decode_governed_upload_ref,
)
from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO
from contracts.ontology_workbench_context import WorkspaceArtifactV1, WorkspaceArtifactPreparationClaimV1
from storage.ontology_build_store import ontology_build_store_path
from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
from runtime_paths import sqlite_db_path

_LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["webapi-gateway"])

# MIME types that the route will serve. The `text/plain` enum value defined
# for `SourceIdentity.mediaType` is intentionally excluded — Sources keeps the
# existing `<pre>` text/IR preview for plain text and normalized Markdown.
_SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    "text/html": "text/html",
    "application/pdf": "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# The `:path` convertor is REQUIRED: governed refs such as
# `uploads/<name>.md` contain a `/`, and ASGI servers URL-decode `%2F`
# before route matching, so a single-segment `{sourceIdentityRef}` parameter
# can never match an encoded uploads ref — it returns Starlette's plain 404
# "Not found" before this handler runs. Containment/traversal safety is
# unaffected: `_resolve_source_identity_ref` rejects absolute/URL refs and
# the upload resolvers re-check containment under the governed uploads root.
_ROUTE_PATH_TEMPLATE = "/api/contextgraph/source-documents/{sourceIdentityRef:path}/content"
_DEFAULT_CHUNK_BYTES = 64 * 1024

# `sha256:<hex>` identity refs name an ORIGINAL upload in the governed
# workspace upload root (`<workspace_root>/wiki/uploads/`). The W6 Sources
# preview and the shared SourceEvidenceViewer derive the presentation
# `contentUrl` from the governed source hash, so the route must resolve hash
# identity to the byte-identical governed upload. Resolution is
# deterministic: sorted traversal, hash match by content, and the served
# bytes are verified by construction (the match IS the hash proof).
_UPLOAD_HASH_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_UPLOAD_BARE_HASH_REF = re.compile(r"^[0-9a-f]{64}$")
_UPLOAD_MEDIA_BY_SUFFIX: dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Extraction runs record `source_identity_ref` as a workspace-upload-relative
# ref (`uploads/<name>`). For binary originals (`uploads/<name>.pdf|.docx`)
# the ref names the servable upload directly; for normalized representations
# (`uploads/<name>.md|.txt|.json`) the ref names the normalized sibling and
# the original upload is recovered by stem. Resolution is still strictly
# identity-based: the ref never escapes the governed uploads root and no
# arbitrary host path is accepted.
_UPLOAD_RELATIVE_REF = re.compile(r"^uploads/[^/]+(?:/[^/]+)*$")
_ORIGINAL_UPLOAD_SUFFIXES: tuple[str, ...] = (".pdf", ".docx", ".doc", ".docm")
_NORMALIZED_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".json"})

# `knowledge-upload:v1:<base64url-json>` is the staged-upload identity written
# by the workspace knowledge upload flow
# (`hermes-workspace/src/server/knowledge-files.ts` `encodeStagedUploadRef`).
# The payload carries the governed `relativePath` of the ORIGINAL upload
# (for example `wiki/uploads/<name>.docx`); the route decodes it, enforces
# that the path stays inside the governed upload root, and serves the bytes.
_KNOWLEDGE_UPLOAD_REF_PREFIX = KNOWLEDGE_UPLOAD_REF_PREFIX
_GOVERNED_SOURCE_REF = re.compile(r"^governed-source:([A-Za-z0-9_-]+)$")


def _eos_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


@contextmanager
def _source_read_connection(path: Path, *, owner: str = "eos"):
    """Read the explicitly mapped owner without initialization or fallback."""
    connection = (hosted.connect_authority(owner, path) if hosted.enabled()
                  else sqlite3.connect(f"file:{path}?mode=ro", uri=True))
    try:
        with connection:
            if hosted.enabled():
                connection.execute("SET TRANSACTION READ ONLY")
            yield connection
    finally:
        connection.close()


def _sha256_file(path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_DEFAULT_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve_governed_source_ref(ctx, ref: str) -> str:
    """Dereference a canonical knowledge-source identity within its tenant.

    The knowledge source row is the governed authority for the storage-facing
    source ref.  Opening SQLite read-only prevents this content route from
    initializing or mutating the authoritative store as a side effect.
    """
    match = _GOVERNED_SOURCE_REF.fullmatch(ref)
    if match is None:
        return ref
    db_path = _eos_path()
    if not hosted.enabled() and not db_path.is_file():
        raise ValueError("governed knowledge source store not found")
    with _source_read_connection(db_path) as conn:
        source_id = match.group(1)
        # Canonical SourceIdentity refs are opaque. Older sources without a
        # registry retain their explicit source-row identity contract.
        if conn.execute(hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_builder_source_identities'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='knowledge_builder_source_identities'")).fetchone():
            registered = conn.execute(
                hosted.sql_query("SELECT source_id FROM knowledge_builder_source_identities "
                "WHERE source_identity_ref=? AND organization_id=? AND workspace_id=?", 'SELECT source_id FROM knowledge_builder_source_identities WHERE source_identity_ref=%s AND organization_id=%s AND workspace_id=%s'),
                (ref, ctx.organization_id, ctx.workspace_id),
            ).fetchone()
            if registered is not None:
                source_id = registered[0]
        row = conn.execute(
            hosted.sql_query("""SELECT source_ref
                 FROM knowledge_builder_sources
                WHERE source_id = ?
                  AND organization_id = ?
                  AND COALESCE(workspace_id, '') = COALESCE(?, '')""", "SELECT source_ref\n                 FROM knowledge_builder_sources\n                WHERE source_id = %s\n                  AND organization_id = %s\n                  AND COALESCE(workspace_id, '') = COALESCE(%s, '')"),
            (source_id, str(ctx.organization_id or ""), ctx.workspace_id),
        ).fetchone()
    if row is None or not str(row[0] or "").strip():
        raise ValueError("governed knowledge source not found")
    resolved = str(row[0]).strip()
    if _GOVERNED_SOURCE_REF.fullmatch(resolved):
        raise ValueError("governed knowledge source ref is recursive")
    return resolved


def _serve_registered_original(request: Request, ctx, identity_ref: str) -> StreamingResponse:
    """Serve a registered original only from its governed workspace byte pin."""
    from storage.knowledge_builder_store import KnowledgeBuilderStore

    registry = KnowledgeBuilderStore(_eos_path(), workspace_root=ctx.workspace_root)
    try:
        _, identity = registry.resolve_source_identity_lineage(
            source_identity_ref=identity_ref, organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
        )
        root = Path(ctx.workspace_root).resolve()
        relative = Path(identity.artifact_ref)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents or not path.is_file():
            raise ValueError("registered source bytes are outside the governed workspace")
        if _sha256_file(path) != identity.source_hash:
            raise ValueError("registered source byte hash mismatch")
        if _UPLOAD_MEDIA_BY_SUFFIX.get(path.suffix.lower()) != identity.media_type:
            raise ValueError("registered source media type mismatch")
        return _serve_upload_path(request, ctx, identity_ref, path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="source document not found") from exc
    finally:
        registry.close()



def _serve_receivables_source(request: Request, ctx, identity_ref: str):
    """Use the existing source-content endpoint for pinned tabular originals/IR."""
    from agents.business_answers import native_service
    from contracts.ontology_building import ArtifactPinV1
    from services.runtime_errors import RuntimeContextError

    representation = request.query_params.get('representation', 'original')
    if representation not in {'original', 'canonical'}:
        raise HTTPException(status_code=400, detail='unsupported source representation')
    try:
        with native_service(ctx) as owner:
            source = owner.get_source_by_identity(ctx=ctx, source_identity_ref=identity_ref)
        if representation == 'canonical':
            body = {
                'schema_version': 'tabular_source_content.v1',
                'source_pin': ArtifactPinV1(artifact_kind='receivables_source', artifact_ref=source.artifact_ref, artifact_hash=source.artifact_hash).model_dump(mode='json'),
                'source_identity_ref': source.source_identity_ref,
                'source_hash': source.source_hash,
                'parser_profile_hash': source.parser_profile_hash,
                'canonical_source_ir_pin': source.canonical_source_ir_pin.model_dump(mode='json'),
                'sheet_name': source.sheet_name,
                'header_row': source.header_row,
                'headers': source.headers,
                'rows': source.rows,
                'row_numbers': source.row_numbers,
                'evidence': [item.model_dump(mode='json', by_alias=True) for item in source.evidence],
                'canonical_source_ir': source.canonical_source_ir.model_dump(mode='json', by_alias=True, exclude_none=False),
            }
            return JSONResponse(content=body, headers={
                'ETag': '"' + source.canonical_source_ir_pin.artifact_hash + '"',
                'X-Source-Hash': source.source_hash,
                'X-Canonical-Source-IR-Hash': source.canonical_source_ir_pin.artifact_hash,
                'Cache-Control': 'private, no-store',
            })
        stream = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=source.byte_size).exact_get(byte_owner_ref=source.byte_owner_ref, expected_hash=source.source_hash, expected_size=source.byte_size)
    except RuntimeContextError as exc:
        raise HTTPException(status_code=403, detail='source document access denied') from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail='source document access denied') from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail='source document not found') from exc
    media_type = 'text/csv' if source.original_name.lower().endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ascii_name = re.sub(r'[^\x20-\x7e]', '_', source.original_name).replace('"', '_') or 'source'
    return StreamingResponse(_stream_chunks(stream, _DEFAULT_CHUNK_BYTES), media_type=media_type, headers={
        'ETag': '"' + source.source_hash + '"',
        'X-Source-Hash': source.source_hash,
        'Cache-Control': 'private, no-transform',
        'Content-Disposition': f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(source.original_name)}',
        'Content-Length': str(source.byte_size),
    })

def _serve_prepared_original(request: Request, ctx, identity: str, correlation_ref: str) -> StreamingResponse:
    """Serve a completed preparation's pinned original through its source identity."""
    if _GOVERNED_SOURCE_REF.fullmatch(identity) is None:
        raise HTTPException(status_code=404, detail="source document not found")
    try:
        with _source_read_connection(_eos_path()) as conn:
            source = conn.execute(
                hosted.sql_query("""SELECT source_ref, source_metadata_json FROM knowledge_builder_sources
                   WHERE source_id=? AND organization_id=? AND COALESCE(workspace_id,'')=COALESCE(?,'')""", "SELECT source_ref, source_metadata_json FROM knowledge_builder_sources\n                   WHERE source_id=%s AND organization_id=%s AND COALESCE(workspace_id,'')=COALESCE(%s,'')"),
                (identity.removeprefix("governed-source:"), ctx.organization_id, ctx.workspace_id),
            ).fetchone()
        if source is None or source[0] != correlation_ref:
            raise ValueError("source binding changed")
        metadata = json.loads(source[1] or "{}")
        build_path = ontology_build_store_path(ctx.workspace_root)
        with _source_read_connection(build_path, owner="standalone") as conn:
            operation = conn.execute(
                hosted.sql_query("""SELECT claim_ref, claim_hash FROM workspace_artifact_preparation_operation_index
                   WHERE organization_id=? AND workspace_id=? AND correlation_ref=? AND state='COMPLETE'""", "SELECT claim_ref, claim_hash FROM workspace_artifact_preparation_operation_index\n                   WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s AND state='COMPLETE'"),
                (ctx.organization_id, ctx.workspace_id, correlation_ref),
            ).fetchone()
            if operation is None:
                raise ValueError("preparation incomplete")
            claim_row = conn.execute(
                hosted.sql_query("""SELECT artifact_hash, payload_json FROM ontology_build_artifacts
                   WHERE organization_id=? AND workspace_id=? AND artifact_ref=?""", 'SELECT artifact_hash, payload_json FROM ontology_build_artifacts\n                   WHERE organization_id=%s AND workspace_id=%s AND artifact_ref=%s'),
                (ctx.organization_id, ctx.workspace_id, operation[0]),
            ).fetchone()
            if claim_row is None or claim_row[0] != operation[1]:
                raise ValueError("preparation claim pin mismatch")
            claim = WorkspaceArtifactPreparationClaimV1.model_validate_json(claim_row[1])
            if claim.purpose not in {"DESIGN_SOURCE", "RUNTIME_TARGET"} or (claim.claim_ref, claim.claim_hash, claim.correlation_ref) != (
                operation[0], operation[1], correlation_ref
            ) or (claim.organization_id, claim.workspace_id) != (ctx.organization_id, ctx.workspace_id):
                raise ValueError("preparation claim mismatch")
            pin = claim.artifact_pin
            artifact_row = conn.execute(
                hosted.sql_query("""SELECT artifact_hash, payload_json FROM ontology_build_artifacts
                   WHERE organization_id=? AND workspace_id=? AND artifact_ref=?""", 'SELECT artifact_hash, payload_json FROM ontology_build_artifacts\n                   WHERE organization_id=%s AND workspace_id=%s AND artifact_ref=%s'),
                (ctx.organization_id, ctx.workspace_id, pin.artifact_ref),
            ).fetchone()
            if artifact_row is None or artifact_row[0] != pin.artifact_hash:
                raise ValueError("artifact pin mismatch")
            artifact = WorkspaceArtifactV1.model_validate_json(artifact_row[1])
        if (artifact.artifact_ref, artifact.artifact_hash) != (pin.artifact_ref, pin.artifact_hash):
            raise ValueError("artifact identity mismatch")
        if (artifact.organization_id, artifact.workspace_id) != (ctx.organization_id, ctx.workspace_id):
            raise ValueError("artifact scope mismatch")
        if metadata.get("original_content_hash") != artifact.content_hash or metadata.get("original_path") != artifact.byte_owner_ref:
            raise ValueError("source metadata mismatch")
        media_type = _SUPPORTED_MEDIA_TYPES.get(artifact.media_type)
        if media_type is None:
            raise HTTPException(status_code=415, detail="unsupported source media type")
        stream = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=artifact.byte_size).exact_get(
            byte_owner_ref=artifact.byte_owner_ref,
            expected_hash=artifact.content_hash,
            expected_size=artifact.byte_size,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="source document not found") from exc
    if media_type == "text/html" and request.query_params.get("representation") == "html-projection":
        return _prepared_html_projection(ctx, identity, correlation_ref, pin, build_path, stream, artifact.content_hash)
    name = artifact.original_name
    ascii_name = re.sub(r'[^\x20-\x7e]', '_', name).replace('"', '_') or "source"
    return StreamingResponse(
        _stream_chunks(stream, _DEFAULT_CHUNK_BYTES),
        media_type=media_type,
        headers={
            "ETag": f'"{artifact.content_hash}"',
            "X-Source-Hash": artifact.content_hash,
            "Cache-Control": "private, no-transform",
            "Content-Disposition": f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name)}',
            "Content-Length": str(artifact.byte_size),
        },
    )


def _uploads_root(workspace_root):
    from pathlib import Path as _Path

    root = _Path(workspace_root)
    uploads_root = (root / "wiki" / "uploads").resolve()
    try:
        uploads_root.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("governed upload root escapes workspace root") from exc
    if not uploads_root.is_dir():
        raise ValueError("governed upload artifact not found")
    return uploads_root


def _resolve_upload_by_hash(workspace_root, ref: str):
    """Find the governed workspace upload whose content hash equals `ref`.

    Only regular files under `<workspace_root>/wiki/uploads/` are
    considered; symlink escapes are rejected by the containment check."""

    uploads_root = _uploads_root(workspace_root)
    for candidate in sorted(uploads_root.rglob("*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(uploads_root)
        except ValueError:
            continue
        if _sha256_file(resolved) == ref:
            return resolved
    raise ValueError("governed upload artifact not found")


def _resolve_upload_by_relative_ref(workspace_root, ref: str):
    """Resolve an `uploads/<name>` source identity ref to the servable upload.

    A ref naming a supported binary upload (`.pdf` / `.docx`) resolves to
    itself. A ref naming a normalized representation (`.md` / `.markdown` /
    `.txt` / `.json`) resolves to the original binary upload with the same
    stem in the same directory, since the Source Evidence Viewer renders the
    original document, not the normalized text."""

    uploads_root = _uploads_root(workspace_root)
    candidate = (uploads_root / ref.removeprefix("uploads/")).resolve()
    try:
        candidate.relative_to(uploads_root)
    except ValueError as exc:
        raise ValueError("governed upload reference escapes upload root") from exc
    suffix = candidate.suffix.lower()
    if suffix in _UPLOAD_MEDIA_BY_SUFFIX:
        if not candidate.is_file():
            raise ValueError("governed upload artifact not found")
        return candidate
    if suffix in _NORMALIZED_SUFFIXES:
        for original_suffix in _ORIGINAL_UPLOAD_SUFFIXES:
            original = candidate.with_suffix(original_suffix)
            if original.is_file():
                resolved = original.resolve()
                try:
                    resolved.relative_to(uploads_root)
                except ValueError:
                    continue
                return resolved
        raise ValueError("governed original upload artifact not found")
    raise ValueError("unsupported governed upload reference")


def _resolve_knowledge_upload_ref(
    workspace_root,
    ref: str,
    *,
    expected_workspace_id: str | None = None,
):
    """Resolve a `knowledge-upload:v1:<base64url-json>` identity to its upload.

    The embedded `relativePath` must stay inside the governed
    `<workspace_root>/wiki/uploads/` root; anything else is rejected before
    any disk access."""
    from pathlib import Path as _Path

    relative = decode_governed_upload_ref(
        ref,
        expected_workspace_id=expected_workspace_id,
    )
    uploads_root = _uploads_root(workspace_root)
    candidate = (_Path(workspace_root) / relative).resolve()
    try:
        candidate.relative_to(uploads_root)
    except ValueError as exc:
        raise ValueError("governed upload reference escapes upload root") from exc
    if not candidate.is_file():
        raise ValueError("governed upload artifact not found")
    return candidate


def _serve_upload_path(request: Request, ctx, ref: str, path) -> StreamingResponse:
    """Stream a resolved governed upload with identity-bound headers."""
    media_type = _UPLOAD_MEDIA_BY_SUFFIX.get(path.suffix.lower())
    if media_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported source media type for upload: {path.suffix}",
        )

    # ETag/X-Source-Hash carry the SERVED file's content hash, not the raw
    # ref: HTTP headers are latin-1 only, so a non-ASCII ref (e.g. an
    # `uploads/<chinese-name>.md` identity) would fail header encoding, and
    # the contract requires a strong ETag matching the governed sourceHash —
    # which is the hash of the served bytes, not of the selector string.
    served_hash = _sha256_file(path)
    content_length = path.stat().st_size
    _LOGGER.info(
        "source_document.served sourceIdentityRef=%s organization_id=%s workspace_id=%s content_hash=%s content_length=%d media_type=%s request_id=%s",
        ref,
        str(getattr(ctx, "organization_id", "") or ""),
        str(getattr(ctx, "workspace_id", "") or ""),
        served_hash,
        content_length,
        media_type,
        str(getattr(request.state, "request_id", "")),
    )
    ascii_fallback = re.sub(r"[^\x20-\x7e]", "_", path.name) or "download"
    content_disposition = (
        f'inline; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(path.name)}"
    )
    headers = {
        "ETag": f'"{served_hash}"',
        "X-Source-Hash": served_hash,
        "Cache-Control": "private, no-transform",
        "Content-Disposition": content_disposition,
    }
    return StreamingResponse(
        _stream_chunks(open(path, "rb"), _DEFAULT_CHUNK_BYTES),
        media_type=media_type,
        headers=headers,
        status_code=200,
    )


def _serve_session_upload(request: Request, ctx, ref: str) -> StreamingResponse:
    """Resolve exact bytes only inside an authenticated, existing session."""
    from agents.webapi.session_routes import (
        _resolve_workspace_runtime_session_id,
        _session_belongs_to_workspace,
    )
    from runtime_paths import workspace_session_runs_root_from_hermes_home

    from contracts.knowledge_classification import RuntimeContextDTO as ActorContext
    from services.runtime_context_transition_service import RuntimeAuthorityResolver

    try:
        RuntimeAuthorityResolver.require_current_actor(
            actor_context=ActorContext(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                actor_id=ctx.user_id,
            ),
            authorization_class="WORKSPACE_LOCAL",
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Outside current workspace source scope") from exc

    session, separator, digest = ref.removeprefix("session-upload:").rpartition(
        ":sha256:"
    )
    if not separator or not re.fullmatch(r"[A-Za-z0-9_.:-]+", session) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HTTPException(status_code=404, detail="source document not found")
    internal = _resolve_workspace_runtime_session_id(ctx, session)
    if (
        not internal
        or not _session_belongs_to_workspace(internal, ctx)
    ):
        raise HTTPException(status_code=404, detail="source document not found")
    root = (
        workspace_session_runs_root_from_hermes_home(
            Path(ctx.hermes_home), session
        ).parent
        / "uploads"
    ).resolve()
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() not in _UPLOAD_MEDIA_BY_SUFFIX
            ):
                continue
            if _sha256_file(path) == "sha256:" + digest:
                return _serve_upload_path(request, ctx, ref, path)
    raise HTTPException(status_code=404, detail="source document not found")

def _serve_upload_by_hash(request: Request, ctx, ref: str) -> StreamingResponse:
    """Serve an original workspace upload addressed by its `sha256:` identity."""
    try:
        path = _resolve_upload_by_hash(getattr(ctx, "workspace_root", ""), ref)
    except ValueError as exc:
        _LOGGER.info(
            "source_document.upload_not_found sourceIdentityRef=%s organization_id=%s reason=%s",
            ref,
            str(getattr(ctx, "organization_id", "") or ""),
            str(exc),
        )
        raise HTTPException(status_code=404, detail="source document not found") from exc
    return _serve_upload_path(request, ctx, ref, path)


def _serve_upload_by_relative_ref(request: Request, ctx, ref: str) -> StreamingResponse:
    """Serve a workspace upload addressed by its `uploads/<name>` identity."""
    try:
        path = _resolve_upload_by_relative_ref(getattr(ctx, "workspace_root", ""), ref)
    except ValueError as exc:
        _LOGGER.info(
            "source_document.upload_not_found sourceIdentityRef=%s organization_id=%s reason=%s",
            ref,
            str(getattr(ctx, "organization_id", "") or ""),
            str(exc),
        )
        raise HTTPException(status_code=404, detail="source document not found") from exc
    return _serve_upload_path(request, ctx, ref, path)


def _serve_knowledge_upload_ref(request: Request, ctx, ref: str) -> StreamingResponse:
    """Serve a workspace upload addressed by a `knowledge-upload:v1:` identity."""
    try:
        path = _resolve_knowledge_upload_ref(
            getattr(ctx, "workspace_root", ""),
            ref,
            expected_workspace_id=str(getattr(ctx, "workspace_id", "") or ""),
        )
    except ValueError as exc:
        _LOGGER.info(
            "source_document.upload_not_found sourceIdentityRef=%s organization_id=%s reason=%s",
            ref,
            str(getattr(ctx, "organization_id", "") or ""),
            str(exc),
        )
        raise HTTPException(status_code=404, detail="source document not found") from exc
    return _serve_upload_path(request, ctx, ref, path)


def _authz(request: Request, method: str, path: str, authorization_class=None):
    authz_class = authorization_class or classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


def _build_runtime_context(ctx) -> RuntimeContextDTO:
    """Translate the authenticated request context into a RuntimeContextDTO.

    The artifact service requires both `actor_id` and `actor_role`. We
    propagate the request's user id and assign `actor_role="workspace-member"`
    consistent with the route's TENANT_MEMBER authorization class.
    """

    return RuntimeContextDTO(
        organization_id=str(ctx.organization_id or "").strip(),
        workspace_id=str(ctx.workspace_id or "").strip() or None,
        actor_id=str(ctx.user_id or "").strip() or "anonymous",
        actor_role="workspace-member",
    )


def _build_artifact_service() -> StoreBackedGovernedArtifactService:
    """Construct the artifact service. Reads the same env var overrides the
    service itself uses; deferred to a thin factory so tests can substitute
    a mock service without touching this module."""

    # No explicit db_path; the service resolves it from runtime_paths
    # and the SEMANTIER_EOS_DB_PATH override inside _active_db_path().
    return StoreBackedGovernedArtifactService()


def _resolve_source_identity_ref(raw: str) -> str:
    """Reject absolute / URL-style / empty references before any disk lookup.

    The plan requires that the route only accept a `sourceIdentityRef`, not a
    raw filesystem path. We strip whitespace, reject empty values, and reject
    anything that smells like a path or URL. Path traversal is further blocked
    inside `StoreBackedGovernedArtifactService._resolve_session_ref`."""

    candidate = str(raw or "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="sourceIdentityRef is required")
    if os.path.isabs(candidate):
        raise HTTPException(status_code=400, detail="sourceIdentityRef must not be an absolute path")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://", "file://")):
        raise HTTPException(status_code=400, detail="sourceIdentityRef must be a governed reference, not a URL")
    return candidate


def _resolve_byte_size(stream) -> int:
    """Determine the governed artifact's byte length without reading the body.

    `open_artifact_for_read` returns the result of `Path.open("rb")`, which
    carries a `.name` (and on POSIX also `.fileno()` for `os.fstat`). We use
    `os.fstat` when available, otherwise fall back to the underlying `Path`
    stat — both avoid loading the bytes into memory."""

    underlying = getattr(stream, "name", None)
    if underlying:
        try:
            return int(os.fstat(stream.fileno()).st_size)
        except (OSError, AttributeError):
            try:
                from pathlib import Path
                return int(Path(underlying).stat().st_size)
            except OSError:
                pass
    # Last-resort fallback: stream the body once to count it. This still
    # avoids `read()` of the full body; it only counts chunks.
    total = 0
    for chunk in iter(lambda: stream.read(_DEFAULT_CHUNK_BYTES), b""):
        total += len(chunk)
    stream.seek(0)
    return total


def _stream_chunks(stream, chunk_bytes: int) -> Iterator[bytes]:
    """Yield fixed-size chunks from a binary stream without loading the whole
    body into worker memory. The stream is closed by FastAPI's
    `StreamingResponse` lifecycle after iteration completes."""

    try:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                return
            yield chunk
    finally:
        try:
            stream.close()
        except OSError:
            # Closing a closed stream is not a failure; suppress to keep the
            # response lifecycle clean.
            pass


def _get_source_document_content(
    request: Request,
    sourceIdentityRef: str,
) -> StreamingResponse:
    """Resolve a governed source identity to its original PDF/DOCX bytes.

    The route is part of the C2 contract. It never accepts a raw filesystem
    path; the only selector is `sourceIdentityRef`."""

    method = request.method
    if request.headers.get("range"):
        # Range is out of scope for v1. Reject without reading bytes so the
        # plan's "no body iteration for unsupported requests" rule holds.
        raise HTTPException(status_code=416, detail="Range requests are not supported in v1")

    policy = classify_route(method, _ROUTE_PATH_TEMPLATE)
    if policy is None or policy.value != "authenticated":
        raise HTTPException(status_code=500, detail="Route policy not configured")

    # `require_route_authorization` returns the authenticated RequestContext;
    # it does not populate `request.state.context`.
    ctx = _authz(request, method, _ROUTE_PATH_TEMPLATE)

    ref = _resolve_source_identity_ref(sourceIdentityRef)
    identity = ref
    # Personal source custody is narrower than organization membership. Keep
    # session reads on the same WORKSPACE_LOCAL boundary as expense replay.
    if ref.startswith("session-upload:"):
        return _serve_session_upload(request, ctx, ref)

    # This public producer identity resolves only the installed, hash-verified
    # teaching bundle; it grants no access to tenant source registrations.
    if identity.startswith('source:expense-policy:'):
        from services.workshop_teaching_materials import load_expense_teaching_bundle
        try:
            prepared = load_expense_teaching_bundle(ctx)
            if prepared is not None and prepared['policy'].source_identity_ref == identity:
                return _serve_upload_path(request, ctx, identity,
                    Path(ctx.hermes_home) / 'demo-materials' / '财务报销政策.pdf')
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail='source document not found') from exc

    if ref.startswith('policy-revision:'):
        from services.policy_revision_service import PolicyRevisionService
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        from starlette.responses import Response
        store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
        try:
            source = PolicyRevisionService(ctx=ctx, store=store, policy_owner=None).read(ref)
            if source.kind != 'source':
                raise ValueError('POLICY_SOURCE_REQUIRED')
            return Response(source.data['markdown'].encode(), media_type='text/markdown', headers={
                'ETag': '"' + source.data['source_hash'] + '"',
                'X-Source-Hash': source.data['source_hash'],
                'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff',
                'Content-Disposition': 'inline; filename="policy-v' + str(source.data['version']) + '.md"'})
        except (PermissionError, ValueError):
            raise HTTPException(404, 'source document not found')
        finally:
            store.close()

    # All remaining source families retain their tenant-member authorization.
    ctx = _authz(request, method, _ROUTE_PATH_TEMPLATE, RouteAuthorizationClass.TENANT_MEMBER)
    if ref.startswith(("jd-source:", "jd-basis-source:")):
        from agents.webapi.jd_review_routes import serve_jd_source
        return serve_jd_source(request, ctx, ref)

    if ref.startswith("governed-source:kbs_receivables_"):
        return _serve_receivables_source(request, ctx, ref)

    if ref.startswith('tender-run:'):
        try:
            source = StoreBackedGovernedArtifactService().resolve_tender_run_source(
                ctx=ctx, run_id=ref.removeprefix('tender-run:'),
            )
            return _serve_upload_path(request, ctx, ref, source)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail='source document not found') from exc

    try:
        ref = _resolve_governed_source_ref(ctx, ref)
    except (ValueError, sqlite3.Error) as exc:
        _LOGGER.info(
            "source_document.not_found sourceIdentityRef=%s organization_id=%s reason=%s",
            ref,
            str(ctx.organization_id or ""),
            str(exc),
        )
        raise HTTPException(status_code=404, detail="source document not found") from exc

    if ref.startswith("session-upload:"):
        return _serve_session_upload(request, ctx, ref)

    if ref.startswith("workspace-artifact-prepare:"):
        return _serve_prepared_original(request, ctx, identity, ref)

    # SourceIdentity is opaque: registered producer identities need not use
    # the legacy governed-source prefix. Resolve exact scoped registrations
    # before format-specific adapters, retaining the registered byte/hash gate.
    db_path = _eos_path()
    if hosted.enabled() or db_path.is_file():
        with _source_read_connection(db_path) as conn:
            registered = None
            if conn.execute(hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_builder_source_identities'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='knowledge_builder_source_identities'")).fetchone():
                registered = conn.execute(
                    hosted.sql_query('SELECT 1 FROM knowledge_builder_source_identities WHERE source_identity_ref=? AND organization_id=? AND workspace_id=?', 'SELECT 1 FROM knowledge_builder_source_identities WHERE source_identity_ref=%s AND organization_id=%s AND workspace_id=%s'),
                    (identity, ctx.organization_id, ctx.workspace_id),
                ).fetchone()
        if registered is not None:
            return _serve_registered_original(request, ctx, identity)

    # Registered canonical sources may predate workspace-artifact preparation.
    # Their explicit governed content binding is resolved through SourceIdentity,
    # never by stripping a prefix into an unmanaged upload/hash fallback.
    if _GOVERNED_SOURCE_REF.fullmatch(identity) and ref.startswith("governed:sha256:"):
        return _serve_registered_original(request, ctx, identity)

    # Bare 64-hex content hashes (as persisted on extraction candidates) are
    # normalized to the canonical `sha256:` upload identity form.
    if _UPLOAD_BARE_HASH_REF.match(ref):
        ref = f"sha256:{ref}"

    if _UPLOAD_HASH_REF.match(ref):
        return _serve_upload_by_hash(request, ctx, ref)

    if _UPLOAD_RELATIVE_REF.match(ref):
        return _serve_upload_by_relative_ref(request, ctx, ref)

    if ref.startswith(_KNOWLEDGE_UPLOAD_REF_PREFIX):
        return _serve_knowledge_upload_ref(request, ctx, ref)

    service = _build_artifact_service()
    runtime_context = _build_runtime_context(ctx)

    try:
        artifact = service.resolve_artifact(
            ref,
            organization_id=runtime_context.organization_id,
            workspace_id=runtime_context.workspace_id,
        )
    except ValueError as exc:
        _LOGGER.info(
            "source_document.not_found sourceIdentityRef=%s organization_id=%s reason=%s",
            ref,
            runtime_context.organization_id,
            str(exc),
        )
        raise HTTPException(status_code=404, detail="source document not found") from exc

    media_type = _SUPPORTED_MEDIA_TYPES.get(str(artifact.media_type or ""))
    if media_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported source media type: {artifact.media_type}",
        )

    expected_hash = str(artifact.content_hash or "")
    if not expected_hash:
        raise HTTPException(status_code=500, detail="governed artifact has no content_hash")

    if not service.verify_artifact_hash(ref, expected_hash, runtime_context):
        _LOGGER.warning(
            "source_document.hash_mismatch sourceIdentityRef=%s expected=%s",
            ref,
            expected_hash,
        )
        raise HTTPException(status_code=409, detail="source document hash mismatch")

    try:
        stream = service.open_artifact_for_read(ref, runtime_context)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="source document not found") from exc

    content_length = _resolve_byte_size(stream)

    # Audit log: metadata only. Per the plan, request and response bodies must
    # never enter the structured log; only identifiers, hashes, and request id.
    _LOGGER.info(
        "source_document.served sourceIdentityRef=%s organization_id=%s workspace_id=%s content_hash=%s content_length=%d media_type=%s request_id=%s",
        ref,
        runtime_context.organization_id,
        runtime_context.workspace_id,
        expected_hash,
        content_length,
        media_type,
        str(getattr(request.state, "request_id", "")),
    )

    headers = {
        "ETag": f'"{expected_hash}"',
        "X-Source-Hash": expected_hash,
        "Cache-Control": "private, no-transform",
        "Content-Disposition": f'inline; filename="{ref.split("/")[-1] or "source"}"',
    }

    return StreamingResponse(
        _stream_chunks(stream, _DEFAULT_CHUNK_BYTES),
        media_type=media_type,
        headers=headers,
        status_code=200,
    )


__all__ = ["router", "get_source_document_content"]


def _prepared_html_projection(ctx, identity, correlation_ref, artifact_pin, build_path, stream, source_hash):
    from contracts.canonical_source_ir import CanonicalSourceDocument
    from contracts.ontology_workbench_context import WorkspaceArtifactPreparationAdapterResultV1
    from eos.artifact_hashing import content_hash
    from services.original_html_source import verify_projection
    from fastapi.responses import JSONResponse
    try:
        with _source_read_connection(build_path, owner="standalone") as conn:
            row = conn.execute(
                hosted.sql_query("""SELECT a.payload_json,a.artifact_hash FROM workspace_artifact_preparation_operation_index i
                   JOIN ontology_build_artifacts a ON a.artifact_ref=i.adapter_result_ref
                   AND a.organization_id=i.organization_id AND a.workspace_id=i.workspace_id
                   WHERE i.organization_id=? AND i.workspace_id=? AND i.correlation_ref=?
                   AND a.artifact_hash=i.adapter_result_hash AND i.state='COMPLETE'""", "SELECT a.payload_json,a.artifact_hash FROM workspace_artifact_preparation_operation_index i\n                   JOIN ontology_build_artifacts a ON a.artifact_ref=i.adapter_result_ref\n                   AND a.organization_id=i.organization_id AND a.workspace_id=i.workspace_id\n                   WHERE i.organization_id=%s AND i.workspace_id=%s AND i.correlation_ref=%s\n                   AND a.artifact_hash=i.adapter_result_hash AND i.state='COMPLETE'"),
                (ctx.organization_id, ctx.workspace_id, correlation_ref),
            ).fetchone()
        if row is None:
            raise ValueError("HTML preparation missing")
        result = WorkspaceArtifactPreparationAdapterResultV1.model_validate_json(row[0])
        if result.adapter_result_hash != row[1] or result.artifact_pin != artifact_pin or result.outcome != "READY":
            raise ValueError("HTML preparation mismatch")
        ir_pin = result.canonical_source_ir_pin
        if ir_pin is None:
            raise ValueError("HTML mapping missing")
        path = (ctx.workspace_root / ir_pin.artifact_ref).resolve()
        if not path.is_relative_to(ctx.workspace_root.resolve()):
            raise ValueError("HTML mapping outside workspace")
        document = CanonicalSourceDocument.model_validate_json(path.read_bytes())
        if document.source_identity_ref != identity or "sha256:" + content_hash(document.model_dump(mode="json", by_alias=True)) != ir_pin.artifact_hash:
            raise ValueError("HTML mapping pin mismatch")
        packets = [a['packet'] for a in document.assets if a.get('kind') == 'original_html_projection']
        if len(packets) != 1:
            raise ValueError("HTML mapping unavailable")
        verify_projection(stream.read(), packets[0])
        return JSONResponse({'source_identity_ref': identity, 'source_hash': source_hash,
            'canonical_ir_pin': ir_pin.model_dump(mode='json'), 'packet': packets[0]},
            headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        raise HTTPException(status_code=404, detail="original HTML mapping unavailable") from exc
    finally:
        stream.close()


def _secure_html_response(response):
    if response.headers.get("content-type", "").split(";")[0] == "text/html":
        response.headers.update({
            "Content-Security-Policy": "sandbox; default-src 'none'; script-src 'none'; style-src 'none'; img-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Content-Disposition": "attachment; filename=original.html",
        })
    return response


@router.api_route(_ROUTE_PATH_TEMPLATE, methods=["GET", "HEAD"])
async def get_source_document_content(request: Request, sourceIdentityRef: str):
    from starlette.concurrency import run_in_threadpool
    return _secure_html_response(await run_in_threadpool(_get_source_document_content, request, sourceIdentityRef))
