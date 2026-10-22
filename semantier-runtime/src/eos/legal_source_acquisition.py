"""Deterministic acquisition helpers for legal source artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener

from eos.legal_corpus_store import LEGAL_SOURCE_QUARANTINED, LegalCorpusStore

LEGAL_ACQUISITION_SOURCE_ACCESS_DENIED = "SOURCE_ACCESS_DENIED"
LEGAL_ACQUISITION_RETRY_PENDING = "RETRY_PENDING"
LEGAL_ACQUISITION_SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
LEGAL_ACQUISITION_PROVENANCE_CONFLICT = "PROVENANCE_CONFLICT"
LEGAL_ACQUISITION_EMPTY_CONTENT = "EMPTY_CONTENT"
LEGAL_ACQUISITION_OVERSIZED_CONTENT = "OVERSIZED_CONTENT"
LEGAL_ACQUISITION_MIME_MISMATCH = "MIME_MISMATCH"
LEGAL_ACQUISITION_PDF_SIGNATURE_MISMATCH = "PDF_SIGNATURE_MISMATCH"
LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION = "REDIRECT_POLICY_VIOLATION"
LEGAL_ACQUISITION_DOMAIN_NOT_ALLOWED = "DOMAIN_NOT_ALLOWED"


@dataclass(frozen=True)
class AcquiredSourceArtifact:
    uri: str
    content: bytes
    mime_type: str
    http_metadata: dict[str, Any]
    redirect_metadata: list[dict[str, Any]]


FetchSource = Callable[[str], AcquiredSourceArtifact]


class OfficialSourceAcquisitionError(ValueError):
    def __init__(self, reason_code: str, result_class: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result_class = result_class


class OfficialSourceAcquisitionAdapter:
    def __init__(
        self,
        fetcher: FetchSource | None = None,
        *,
        official_domain_allowlist: list[str] | None = None,
        max_artifact_bytes: int = 10_000_000,
        max_redirects: int = 5,
        redirect_policy: str = "same_host",
        mime_allowlist: list[str] | None = None,
        timeout_seconds: int = 20,
    ) -> None:
        self._fetcher = fetcher or (lambda uri: _fetch_url(
            uri, timeout_seconds=timeout_seconds, max_artifact_bytes=self._max_artifact_bytes,
            redirect_validator=self._validate_redirect, max_redirects=self._max_redirects,
        ))
        self._official_domain_allowlist = _normalize_hosts(official_domain_allowlist or [])
        self._max_artifact_bytes = max_artifact_bytes
        self._max_redirects = max_redirects
        self._redirect_policy = str(redirect_policy or "same_host").strip()
        self._mime_allowlist = set(
            mime_allowlist
            or [
                "application/pdf",
                "text/html",
                "text/plain",
                "application/xhtml+xml",
            ]
        )

    def acquire(
        self,
        *,
        store: LegalCorpusStore,
        version_id: str | None,
        uri: str,
        storage_ref: str,
        source_id: str | None = None,
        actor_ref: str | None = None,
        organization_id: str | None = None,
        correlation_id: str | None = None,
        baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            self._assert_allowed_host(uri, reason_code=LEGAL_ACQUISITION_DOMAIN_NOT_ALLOWED)
            artifact = self._fetcher(uri)
            self._validate_artifact(uri=uri, artifact=artifact)
        except OfficialSourceAcquisitionError as exc:
            if source_id and actor_ref and organization_id and correlation_id:
                return store.quarantine_acquisition(
                    source_id=source_id,
                    reason_code=exc.reason_code,
                    uri=uri,
                    event_payload={
                        "result_class": exc.result_class,
                        "message": str(exc),
                        "source_uri": uri,
                    },
                    actor_ref=actor_ref,
                    organization_id=organization_id,
                    correlation_id=correlation_id,
                )
            raise
        except HTTPError as exc:
            error = _http_error(exc)
            if source_id and actor_ref and organization_id and correlation_id:
                return store.quarantine_acquisition(
                    source_id=source_id,
                    reason_code=error.reason_code,
                    uri=uri,
                    event_payload={"result_class": error.result_class, "status_code": exc.code},
                    actor_ref=actor_ref,
                    organization_id=organization_id,
                    correlation_id=correlation_id,
                )
            raise error from exc
        except (TimeoutError, URLError, OSError) as exc:
            error = OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_SOURCE_UNAVAILABLE,
                LEGAL_ACQUISITION_SOURCE_UNAVAILABLE,
                f"official source unavailable: {exc}",
            )
            if source_id and actor_ref and organization_id and correlation_id:
                return store.quarantine_acquisition(
                    source_id=source_id,
                    reason_code=error.reason_code,
                    uri=uri,
                    event_payload={"result_class": error.result_class, "message": str(exc)},
                    actor_ref=actor_ref,
                    organization_id=organization_id,
                    correlation_id=correlation_id,
                )
            raise error from exc
        if version_id is None:
            if not (source_id and organization_id and correlation_id and actor_ref and baseline):
                raise ValueError("observation requires governed source, actor, correlation and baseline")
            return store.preserve_acquired_observation(
                source_id=source_id, organization_id=organization_id, correlation_id=correlation_id,
                actor_ref=actor_ref, uri=artifact.uri, content=artifact.content, mime_type=artifact.mime_type,
                http_metadata=artifact.http_metadata, redirect_metadata=artifact.redirect_metadata,
                baseline=baseline,
            )
        return store.preserve_artifact(
            version_id=version_id,
            artifact_kind="official_html" if artifact.mime_type.startswith("text/html") else "official_artifact",
            uri=artifact.uri,
            content=artifact.content,
            mime_type=artifact.mime_type,
            storage_ref=storage_ref,
            http_metadata=artifact.http_metadata,
            redirect_metadata=artifact.redirect_metadata,
        )

    def _validate_redirect(self, initial_uri: str, target: str) -> None:
        self._assert_allowed_host(target, reason_code=LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION)
        if self._redirect_policy not in {"same_host", "allowlist"} or (
            self._redirect_policy == "same_host" and _hostname(initial_uri) != _hostname(target)
        ):
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                "official source redirect violates governed policy",
            )

    def _validate_artifact(self, *, uri: str, artifact: AcquiredSourceArtifact) -> None:
        status_code = int(artifact.http_metadata.get("status_code") or 200)
        if not 200 <= status_code < 300:
            raise _status_error(status_code)
        if len(artifact.redirect_metadata) > self._max_redirects:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                "official source exceeded redirect limit",
            )
        if not artifact.content:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_EMPTY_CONTENT,
                LEGAL_ACQUISITION_SOURCE_UNAVAILABLE,
                "official source returned empty content",
            )
        if len(artifact.content) > self._max_artifact_bytes:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_OVERSIZED_CONTENT,
                LEGAL_ACQUISITION_RETRY_PENDING,
                "official source response exceeded max_artifact_bytes",
            )
        mime_type = str(artifact.mime_type or "application/octet-stream").split(";")[0].strip().lower()
        if mime_type not in self._mime_allowlist:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_MIME_MISMATCH,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                f"official source MIME type not allowed: {mime_type}",
            )
        if mime_type == "application/pdf" and not artifact.content.startswith(b"%PDF-"):
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_PDF_SIGNATURE_MISMATCH,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                "official PDF response does not start with a PDF signature",
            )
        self._assert_allowed_host(artifact.uri, reason_code=LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION)
        for hop in artifact.redirect_metadata:
            self._validate_redirect(uri, str(hop.get("to") or ""))
        initial_host = _hostname(uri)
        final_host = _hostname(artifact.uri)
        if self._redirect_policy == "same_host" and initial_host != final_host:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                "official source redirected outside the original host",
            )
        if self._redirect_policy not in {"same_host", "allowlist"}:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                f"unsupported redirect policy: {self._redirect_policy}",
            )

    def _assert_allowed_host(self, uri: str, *, reason_code: str) -> None:
        host = _hostname(uri)
        if urlparse(uri).scheme.lower() not in {"http", "https"} or not host:
            raise OfficialSourceAcquisitionError(
                reason_code,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                "official source URI must include a host",
            )
        if self._official_domain_allowlist and host not in self._official_domain_allowlist:
            raise OfficialSourceAcquisitionError(
                reason_code,
                LEGAL_ACQUISITION_PROVENANCE_CONFLICT,
                f"official source host is not allowlisted: {host}",
            )


class GovernedManualFallbackAdapter:
    def ingest(
        self,
        *,
        store: LegalCorpusStore,
        source_id: str,
        version_id: str,
        governed_upload_ref: str,
        content: bytes,
        mime_type: str,
        storage_ref: str,
        actor_ref: str,
        organization_id: str,
        correlation_id: str,
        provenance_reviewed: bool,
    ) -> dict[str, Any]:
        if not governed_upload_ref.startswith("uploads/") and not governed_upload_ref.startswith("artifacts/"):
            return store.quarantine_acquisition(
                source_id=source_id,
                reason_code=LEGAL_SOURCE_QUARANTINED,
                uri=governed_upload_ref,
                event_payload={"reason": "manual fallback must use governed upload or artifact refs"},
                actor_ref=actor_ref,
                organization_id=organization_id,
                correlation_id=correlation_id,
            )
        if not provenance_reviewed:
            return store.quarantine_acquisition(
                source_id=source_id,
                reason_code=LEGAL_SOURCE_QUARANTINED,
                uri=governed_upload_ref,
                event_payload={"reason": "manual fallback provenance review required"},
                actor_ref=actor_ref,
                organization_id=organization_id,
                correlation_id=correlation_id,
            )
        return store.preserve_artifact(
            version_id=version_id,
            artifact_kind="manual_fallback",
            uri=governed_upload_ref,
            content=content,
            mime_type=mime_type,
            storage_ref=storage_ref,
            http_metadata={"manual_fallback": True, "provenance_reviewed": True},
            redirect_metadata=[],
        )


def _fetch_url(
    uri: str, timeout_seconds: int = 20, *, max_artifact_bytes: int = 10_000_000,
    redirect_validator: Callable[[str, str], None] | None = None, max_redirects: int = 5,
) -> AcquiredSourceArtifact:
    redirects: list[dict[str, Any]] = []

    class GovernedRedirectHandler(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if len(redirects) >= max_redirects:
                raise OfficialSourceAcquisitionError(
                    LEGAL_ACQUISITION_REDIRECT_POLICY_VIOLATION,
                    LEGAL_ACQUISITION_PROVENANCE_CONFLICT, "official source exceeded redirect limit")
            if redirect_validator is not None:
                redirect_validator(uri, newurl)
            redirects.append({"from": req.full_url, "to": newurl, "status_code": code})
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    request = Request(uri, headers={"User-Agent": "SemantierLegalCorpus/1.0"})
    deadline = monotonic() + timeout_seconds
    with build_opener(GovernedRedirectHandler()).open(request, timeout=timeout_seconds) as response:
        # read1 returns after one underlying read, so slow streaming cannot reset
        # the whole-response deadline forever. Socket reads also have a finite timeout.
        chunks = []
        remaining = max_artifact_bytes + 1
        while remaining:
            if monotonic() >= deadline:
                raise TimeoutError("official source response deadline exceeded")
            chunk = response.read1(min(64 * 1024, remaining))
            if monotonic() >= deadline:
                raise TimeoutError("official source response deadline exceeded")
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_artifact_bytes:
            raise OfficialSourceAcquisitionError(
                LEGAL_ACQUISITION_OVERSIZED_CONTENT, LEGAL_ACQUISITION_RETRY_PENDING,
                "official source response exceeded max_artifact_bytes")
        return AcquiredSourceArtifact(
            uri=str(response.geturl()), content=content,
            mime_type=str(response.headers.get_content_type() or "application/octet-stream"),
            http_metadata={"status_code": int(getattr(response, "status", 200)),
                           "headers": dict(response.headers.items())},
            redirect_metadata=redirects,
        )


def _normalize_hosts(hosts: list[str]) -> set[str]:
    result: set[str] = set()
    for host in hosts:
        value = str(host or "").strip().lower()
        if not value:
            continue
        parsed = urlparse(value if "://" in value else f"https://{value}")
        if parsed.hostname:
            result.add(parsed.hostname.lower())
    return result


def _hostname(uri: str) -> str:
    return str(urlparse(str(uri or "")).hostname or "").lower()


def _status_error(status_code: int) -> OfficialSourceAcquisitionError:
    if status_code == 403:
        return OfficialSourceAcquisitionError(
            LEGAL_ACQUISITION_SOURCE_ACCESS_DENIED,
            LEGAL_ACQUISITION_SOURCE_ACCESS_DENIED,
            "official source access denied",
        )
    if status_code == 429 or status_code >= 500:
        return OfficialSourceAcquisitionError(
            LEGAL_ACQUISITION_RETRY_PENDING,
            LEGAL_ACQUISITION_RETRY_PENDING,
            f"official source returned retryable status: {status_code}",
        )
    return OfficialSourceAcquisitionError(
        LEGAL_ACQUISITION_SOURCE_UNAVAILABLE,
        LEGAL_ACQUISITION_SOURCE_UNAVAILABLE,
        f"official source returned unsupported status: {status_code}",
    )


def _http_error(exc: HTTPError) -> OfficialSourceAcquisitionError:
    return _status_error(int(exc.code))
