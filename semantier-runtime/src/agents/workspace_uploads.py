from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import uuid

from gateway.execution_boundary import DiskIOBoundaryGuard, ExecutionBoundary


ALLOWED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".md", ".txt"})
_FILENAME_SAFE_RE = re.compile(r"[\x00-\x1f\x7f/\\]+")
_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._:-]+")


class WorkspaceUploadError(ValueError):
    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class WorkspaceUploadResult:
    filename: str
    file_path: Path
    relative_path: str
    content_type: str
    size: int


def sanitize_upload_filename(
    filename: str,
    *,
    allowed_extensions: frozenset[str] = ALLOWED_DOCUMENT_EXTENSIONS,
) -> str:
    basename = Path(str(filename or "").strip()).name
    basename = _FILENAME_SAFE_RE.sub("_", basename).strip(" .")
    if not basename:
        raise WorkspaceUploadError("filename required", code="filename_required")

    extension = Path(basename).suffix.lower()
    if extension not in allowed_extensions:
        raise WorkspaceUploadError(
            "unsupported upload extension",
            code="unsupported_extension",
        )
    return basename


def sanitize_upload_session_id(session_id: str) -> str:
    safe = _SESSION_ID_RE.sub("_", str(session_id or "").strip()).strip(" .")
    if not safe or safe in {".", ".."}:
        raise WorkspaceUploadError("session_id required", code="session_id_required")
    if "/" in safe or "\\" in safe:
        raise WorkspaceUploadError("invalid session_id", code="invalid_session_id")
    return safe


def session_uploads_dir(workspace_root: Path, session_id: str) -> Path:
    return (
        Path(workspace_root)
        / "sessions"
        / sanitize_upload_session_id(session_id)
        / "uploads"
    )


def _write_unique_file(
    uploads_dir: Path,
    filename: str,
    content: bytes,
    guard: DiskIOBoundaryGuard | None = None,
) -> Path:
    requested = uploads_dir / filename
    stem = requested.stem
    suffix = requested.suffix
    candidates = [requested]
    candidates.extend(
        uploads_dir / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}" for _ in range(20)
    )
    for candidate in candidates:
        try:
            if guard is not None:
                resolved = guard.resolve_write_path(
                    candidate, purpose="document_import"
                )
                if resolved != candidate:
                    raise WorkspaceUploadError(
                        "invalid upload path",
                        code="invalid_upload_path",
                        status_code=403,
                    )
            with candidate.open("xb") as handle:
                handle.write(content)
            return candidate
        except FileExistsError:
            continue
    raise WorkspaceUploadError(
        "Unable to allocate upload filename",
        code="filename_allocation_failed",
        status_code=409,
    )


def save_workspace_upload(
    *,
    workspace_root: Path,
    session_id: str,
    filename: str,
    content: bytes,
    content_type: str = "",
    allowed_extensions: frozenset[str] = ALLOWED_DOCUMENT_EXTENSIONS,
    boundary: ExecutionBoundary | None = None,
) -> WorkspaceUploadResult:
    safe_filename = sanitize_upload_filename(
        filename,
        allowed_extensions=allowed_extensions,
    )
    if not content:
        raise WorkspaceUploadError("file is empty", code="empty_file")

    safe_session_id = sanitize_upload_session_id(session_id)
    uploads_dir = Path(workspace_root) / "sessions" / safe_session_id / "uploads"
    guard = DiskIOBoundaryGuard(boundary) if boundary is not None else None
    if guard is not None:
        resolved = guard.resolve_write_path(uploads_dir, purpose="document_import")
        if resolved != uploads_dir or resolved != boundary.paths.uploads_root:
            raise WorkspaceUploadError(
                "invalid upload path", code="invalid_upload_path", status_code=403
            )
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = _write_unique_file(uploads_dir, safe_filename, content, guard)

    return WorkspaceUploadResult(
        filename=target.name,
        file_path=target,
        relative_path=f"sessions/{safe_session_id}/uploads/{target.name}",
        content_type=content_type,
        size=len(content),
    )


def _read_workspace_upload(
    *, boundary: ExecutionBoundary, upload_ref: str, upload_hash: str | None,
    max_bytes: int, allowed_extensions: frozenset[str],
) -> bytes:
    """Read only an immediate file in the authorized session upload root."""
    if max_bytes <= 0:
        raise ValueError("INVALID_BYTE_LENGTH")
    root = boundary.paths.hermes_home
    uploads = boundary.paths.uploads_root
    if root is None or uploads is None:
        raise WorkspaceUploadError(
            "upload unavailable", code="NOT_FOUND", status_code=404
        )
    candidate = root / upload_ref
    expected_ref = (uploads / candidate.name).relative_to(root).as_posix()
    if upload_ref != expected_ref or candidate.suffix.lower() not in allowed_extensions:
        raise WorkspaceUploadError(
            "upload unavailable", code="NOT_FOUND", status_code=404
        )
    path = DiskIOBoundaryGuard(boundary).resolve_read_path(
        candidate, purpose="document_input"
    )
    if path != candidate or path.parent != uploads:
        raise WorkspaceUploadError(
            "upload unavailable", code="NOT_FOUND", status_code=404
        )
    try:
        with path.open("rb") as handle:
            content = handle.read(max_bytes + 1)
    except OSError:
        raise WorkspaceUploadError(
            "upload unavailable", code="NOT_FOUND", status_code=404
        ) from None
    if len(content) > max_bytes:
        raise WorkspaceUploadError(
            "upload too large", code="RESOURCE_LIMIT_EXCEEDED", status_code=413
        )
    if upload_hash is not None and "sha256:" + sha256(content).hexdigest() != upload_hash:
        raise WorkspaceUploadError(
            "upload unavailable", code="NOT_FOUND", status_code=404
        )
    return content


def read_temporal_workspace_upload(
    *, boundary: ExecutionBoundary, upload_ref: str, upload_hash: str
) -> bytes:
    if not isinstance(upload_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", upload_hash):
        raise WorkspaceUploadError("upload unavailable", code="NOT_FOUND", status_code=404)
    return _read_workspace_upload(boundary=boundary, upload_ref=upload_ref, upload_hash=upload_hash,
                                  max_bytes=1_048_576, allowed_extensions=frozenset({".ipynb"}))


def read_document_workspace_upload(
    *, boundary: ExecutionBoundary, upload_ref: str, upload_hash: str, max_bytes: int
) -> bytes:
    if not isinstance(upload_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", upload_hash):
        raise WorkspaceUploadError("upload unavailable", code="NOT_FOUND", status_code=404)
    return _read_workspace_upload(boundary=boundary, upload_ref=upload_ref, upload_hash=upload_hash,
                                  max_bytes=max_bytes, allowed_extensions=ALLOWED_DOCUMENT_EXTENSIONS)


def list_document_workspace_uploads(*, boundary: ExecutionBoundary, max_bytes: int, offset: int = 0, limit: int = 50) -> dict:
    """Discover upload identities; importing still rechecks the returned byte hash."""
    root, uploads = boundary.paths.hermes_home, boundary.paths.uploads_root
    if root is None or uploads is None or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("INVALID_UPLOAD_SCOPE")
    path = DiskIOBoundaryGuard(boundary).resolve_read_path(uploads, purpose="document_input")
    if path != uploads:
        raise ValueError("INVALID_UPLOAD_SCOPE")
    files = sorted((item for item in path.iterdir() if item.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS), key=lambda item: item.name) if path.exists() else []
    items = []
    for item in files[offset:offset + limit]:
        ref = item.relative_to(root).as_posix()
        content = _read_workspace_upload(boundary=boundary, upload_ref=ref, upload_hash=None,
                                         max_bytes=max_bytes, allowed_extensions=ALLOWED_DOCUMENT_EXTENSIONS)
        items.append({"upload_ref": ref, "upload_hash": "sha256:" + sha256(content).hexdigest(),
                      "original_name": item.name, "byte_size": len(content)})
    has_more = len(files) > offset + limit
    return {"uploads": items, "has_more": has_more,
            "next_offset": offset + limit if has_more else None}
