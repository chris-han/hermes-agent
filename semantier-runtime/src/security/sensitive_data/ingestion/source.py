from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from agents.auth_session import RequestContext
from agents.workspace_uploads import sanitize_upload_session_id
from semantier.sensitive_governance_contracts.hashing import canonical_hash
from semantier.sensitive_governance_contracts.requests import SensitiveGovernanceRequest


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".txt", ".md", ".docx", ".pdf", ".xlsx"})


@dataclass(frozen=True)
class GovernedSourceContent:
    content: str
    source_evidence: dict


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _document_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HERMES_DOCUMENT_CACHE_DIR"),
        os.environ.get("SEMANTIER_HERMES_DOCUMENT_CACHE_DIR"),
    ):
        if raw:
            roots.append(Path(raw).expanduser().resolve())
    roots.append((Path.home() / ".hermes" / "cache" / "documents").resolve())
    return list(dict.fromkeys(roots))


def _session_uploads_root(ctx: RequestContext, session_id: str) -> Path:
    if ctx.workspace_root is None:
        raise ValueError("workspace root required for governed source ingestion")
    safe_session_id = sanitize_upload_session_id(session_id)
    return Path(ctx.workspace_root).resolve() / "sessions" / safe_session_id / "uploads"


def _relative_upload_path(source_ref: str) -> Path:
    raw = source_ref.strip()
    if raw.startswith(("http://", "https://", "file://")):
        raise ValueError("remote source_ref rejected")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if not raw.startswith("uploads/"):
        raise ValueError("source_ref must be an uploads/ ref")
    relative = Path(*Path(raw).parts[1:])
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("invalid uploads source_ref")
    return relative


def _import_document_cache_ref(
    *,
    source: Path,
    uploads_root: Path,
) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("source_ref not found")
    if not any(_inside(resolved, root) for root in _document_cache_roots()):
        raise ValueError("absolute source_ref rejected")
    uploads_root.mkdir(parents=True, exist_ok=True)
    digest = canonical_hash({"document_cache_source": str(resolved)})[:16]
    safe_name = resolved.name.replace("/", "_").replace("\\", "_")
    destination = (uploads_root / f"imported-{digest}-{safe_name}").resolve()
    if not _inside(destination, uploads_root):
        raise ValueError("document cache import escaped uploads root")
    if not destination.exists():
        shutil.copy2(resolved, destination)
    return destination


def resolve_governed_source_content(
    *,
    ctx: RequestContext,
    session_id: str,
    request: SensitiveGovernanceRequest,
) -> GovernedSourceContent:
    if request.source_kind == "pasted_text":
        return GovernedSourceContent(
            content=request.content,
            source_evidence={
                "source_kind": "pasted_text",
                "source_ref": None,
                "ingestion_mode": "inline_authenticated_request",
            },
        )
    source_ref = request.source_ref or ""
    uploads_root = _session_uploads_root(ctx, session_id)
    ref_path = _relative_upload_path(source_ref)
    imported = False
    if ref_path.is_absolute():
        resolved = _import_document_cache_ref(source=ref_path, uploads_root=uploads_root)
        imported = True
    else:
        resolved = (uploads_root / ref_path).resolve()
        if not _inside(resolved, uploads_root):
            raise ValueError("source_ref escaped uploads root")
        if not resolved.is_file():
            raise ValueError("source_ref not found")
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("unsupported source_ref document format")
    if suffix != "." + request.source_kind:
        raise ValueError(f"{request.source_kind} source_kind requires a .{request.source_kind} source_ref")
    data = resolved.read_bytes()
    if not data:
        raise ValueError("source_ref content required")
    governed_upload_ref = "uploads/" + resolved.relative_to(uploads_root).as_posix()
    source_hash = "sha256:" + hashlib.sha256(data).hexdigest()
    parser_version = "utf8_text.v1"
    parsed_spans = []
    if suffix in {".txt", ".md"}:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("source_ref must be utf-8 text") from exc
    else:
        content, parser_version, parsed_spans = _parse_document_upload(
            resolved, governed_upload_ref=governed_upload_ref, source_hash=source_hash,
            byte_length=len(data), session_id=session_id,
        )
    content_hash = canonical_hash({"source_content": content})
    governed_upload_ref = "uploads/" + resolved.relative_to(uploads_root).as_posix()
    ingestion_mode = "document_cache_import" if imported else "session_upload"
    return GovernedSourceContent(
        content=content,
        source_evidence={
            "source_kind": request.source_kind,
            "source_ref": source_ref,
            "ingestion_mode": ingestion_mode,
            "governed_upload_ref": governed_upload_ref,
            "content_hash": content_hash,
            "parse_ref": "sensitive-text-parse:" + canonical_hash({
                "governed_upload_ref": governed_upload_ref,
                "content_hash": content_hash,
                "source_kind": request.source_kind,
                "parser": parser_version,
            })[:24],
            "parser_version": parser_version,
            "source_hash": source_hash,
            "parsed_spans": parsed_spans,
        },
    )


def _parse_document_upload(
    source: Path, *, governed_upload_ref: str, source_hash: str,
    byte_length: int, session_id: str,
) -> tuple[str, str, list[dict]]:
    """Project existing local parser evidence without creating a second cache/store."""
    from plugins.document_extraction.service import _parser_for
    from plugins.document_extraction.storage import media_type_for

    media_type = media_type_for(source)
    parser = _parser_for(source, media_type)
    try:
        if source.suffix.lower() in {".docx", ".pdf"}:
            document = parser.parse_bound(
                source, artifact_ref=governed_upload_ref, source_sha256=source_hash,
                media_type=media_type, byte_length=byte_length,
                execution_scope_ref="session:" + session_id,
            )
        else:
            document = parser.parse(source, allow_ocr=False, language_hints=[])
    except Exception as exc:
        raise ValueError("document parsing failed") from exc
    evidence = {item["id"]: item for item in document.get("evidence", [])}
    pieces = []
    spans = []
    offset = 0
    for element in document.get("elements") or document.get("blocks", []):
        rows = [cell for row in element.get("rows", []) for cell in row.get("cells", [])] if element.get("kind") == "table" else [element]
        for block in rows:
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            refs = block.get("evidence_ids", [])
            locators = [evidence[ref]["locator"] for ref in refs if ref in evidence]
            spans.append({"start": offset, "end": offset + len(text),
                          "evidence_ids": refs, "locator": locators})
            pieces.append(text)
            offset += len(text) + 1
    if not pieces:
        raise ValueError("no parseable text; local OCR is not qualified")
    return "\n".join(pieces), document["parser"]["method"] + ":" + document["parser"]["version"], spans
