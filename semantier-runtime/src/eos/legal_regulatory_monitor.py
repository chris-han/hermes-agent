"""Deterministic orchestration boundary for legal regulatory source monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
import hashlib
import codecs
from datetime import timedelta
from typing import Any

from eos.db import init_db, now_iso
from eos.legal_corpus_store import (
    LegalCorpusStore,
    canonical_scheduled_window,
    _format_utc_z,
    _parse_aware_utc,
)
from eos.artifact_hashing import content_hash
from eos.legal_source_acquisition import FetchSource, OfficialSourceAcquisitionAdapter
from eos.legal_text_normalization import normalize_legal_html, normalize_legal_text, normalize_authoritative_pdf_pages


@dataclass(frozen=True)
class RegulatoryMonitorRun:
    scan_run: dict[str, Any]
    scan_items: list[dict[str, Any]]
    resolution_errors: list[dict[str, Any]]


class LegalRegulatoryMonitor:
    """Finite legal source monitor service.

    The service composes the legal corpus store contracts. It does not certify,
    activate, or redefine legal source authority.
    """

    def __init__(
        self,
        store: LegalCorpusStore | None = None,
        *,
        official_fetcher: FetchSource | None = None,
    ) -> None:
        self._store = store or LegalCorpusStore()
        self._official_fetcher = official_fetcher

    def execute_tick(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        scheduled_at: str,
        trigger: str = "SCHEDULED",
        lease_owner: str = "regulatory-source-monitor",
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Execute a finite refresh in the runtime owner shared by triggers."""
        if not organization_id.strip() or trigger not in {"SCHEDULED", "MANUAL", "SKILL"}:
            raise ValueError("INVALID_TRIGGER_ENVELOPE")
        _parse_aware_utc(scheduled_at, field_name="scheduled_at")
        init_db()
        run = self.create_or_resume_run(
            organization_id=organization_id, workspace_id=workspace_id,
            scheduled_at=scheduled_at, trigger=trigger, limit=limit, offset=offset,
        )
        completed_items = []
        for item in run.scan_items:
            actual_run = self._store.get_scan_run(organization_id=organization_id, scan_run_id=item["scan_run_id"])["scan_run"]
            if actual_run.get("completed_at") or item["status"] in {"SUCCEEDED", "FAILED"}:
                completed_items.append(item)
                continue
            checked_at = now_iso()
            resolution = self._store.resolve_monitor_policy(
                organization_id=organization_id, source_ref=item["source_id"], workspace_id=workspace_id,
                scheduled_at=scheduled_at, require_workspace=trigger != "SCHEDULED")
            if not resolution["ok"]:
                completed_items.append(item)
                continue
            policy = resolution["policy"]
            try:
                claimed = self.claim_source(
                    scan_item_id=item["scan_item_id"], lease_owner=lease_owner,
                    lease_expires_at=_format_utc_z(
                        _parse_aware_utc(checked_at, field_name="checked_at") + timedelta(minutes=5)),
                    as_of=checked_at,
                )
            except ValueError:
                completed_items.append(item)
                continue
            evidence: dict[str, Any] = {"activation_performed": False, "comparator": "legal_source_versions.v1",
                "normalizer": policy["normalization_profile"], "comparison_profile": policy["comparison_profile"],
                "policy_version_ref": policy["policy_version_ref"], "scheduled_window": item["scheduled_window"],
                "scan_item_id": item["scan_item_id"], "attempt_count": claimed["attempt_count"]}
            uri = self._store.latest_source_acquisition_uri(source_id=item["source_id"]) or "unavailable:governed-source-uri"
            observation = None
            status, result_class, error_code, error_detail = "FAILED", "UNKNOWN_REVIEW_REQUIRED", None, None
            try:
                if policy["monitor_policy_version"] != item["monitor_policy_version"]:
                    raise ValueError("POLICY_CHANGED_DURING_RUN")
                if policy["normalization_profile"] != "legal_text.v1" or policy["comparison_profile"] != "stable_anchor.v1":
                    raise ValueError("UNSUPPORTED_COMPARISON_PROFILE")
                if claimed["attempt_count"] > int(policy["retry_policy"].get("max_attempts", 3)):
                    raise ValueError("RETRY_LIMIT_EXHAUSTED")
                observation = self._store.scan_item_observation(organization_id=organization_id,
                    source_id=item["source_id"], scan_item_id=item["scan_item_id"])
                baseline = self._store.pin_refresh_basis(organization_id=organization_id,
                    source_id=item["source_id"], scan_item_id=item["scan_item_id"], lease_owner=lease_owner,
                    as_of=scheduled_at, policy_version_ref=policy["policy_version_ref"])
                evidence.update({"baseline_version_id": baseline["version_id"],
                                 "baseline_artifact_id": baseline["artifact_id"],
                                 "baseline_content_hash": baseline["content_hash"],
                                 "baseline_normalized_hash": baseline["normalized_hash"]})
                uri = baseline["uri"]
                if not observation:
                    adapter = OfficialSourceAcquisitionAdapter(fetcher=self._official_fetcher,
                        official_domain_allowlist=policy["official_domain_allowlist"],
                        max_artifact_bytes=int(policy["max_artifact_bytes"]), redirect_policy=policy["redirect_policy"],
                        mime_allowlist=policy.get("mime_allowlist"))
                    acquired = adapter.acquire(store=self._store, source_id=item["source_id"], version_id=None,
                        uri=uri, storage_ref="artifacts/legal/" + item["scan_item_id"], actor_ref=lease_owner,
                        organization_id=organization_id, correlation_id=item["scan_item_id"], baseline=baseline)
                    if acquired.get("event_type") != "ACQUIRED_OBSERVATION" or not acquired.get("content_hash"):
                        evidence["acquisition_failure"] = acquired
                        reason = str(acquired.get("reason_code") or "ACQUISITION_EVIDENCE_MISSING")
                        result_class = str(acquired.get("result_class") or (
                            reason if reason in {"SOURCE_ACCESS_DENIED", "RETRY_PENDING", "SOURCE_UNAVAILABLE"}
                            else "SOURCE_UNAVAILABLE"))
                        raise ValueError(reason)
                    observation = acquired
                evidence.update({"artifact_id": observation["artifact_id"],
                                 "acquired_content_hash": observation["content_hash"],
                                 "observed_ref": observation["artifact_id"], "observed_version_id": None,
                                 "retrieved_at": observation["retrieved_at"]})
                normalized, normalizer = self.normalize_observation(observation)
                evidence["normalizer"] = normalizer
                evidence["normalized_text"] = normalized
                evidence["normalized_content_hash"] = content_hash({"text": normalized})
                baseline_normalized = None
                if baseline["content_hash"] != observation["content_hash"]:
                    if not baseline["normalized_hash"]:
                        raise ValueError("BASELINE_NORMALIZATION_MISSING")
                    base_artifact = self._store.read_preserved_artifact(artifact_id=baseline["artifact_id"],
                        organization_id=organization_id, source_id=item["source_id"])
                    baseline_normalized, _ = self.normalize_observation(base_artifact)
                comparison = self._store.compare_source_versions(prior_version_id=baseline["version_id"],
                    acquired_observation={**observation, "organization_id": organization_id}, normalized_text=normalized,
                    baseline_normalized_text=baseline_normalized)
                evidence["comparison"] = comparison
                result_class = comparison["comparison_class"]
                if result_class in {"SEMANTICALLY_UNCHANGED", "REPRESENTATION_CHANGED"}:
                    status = "SUCCEEDED"
                    evidence["review_state"] = "NO_AUTHORITY_REVIEW_REQUIRED"
                else:
                    evidence["review_state"] = "PENDING_REVIEW"
                    raise ValueError("LEGAL_VERSION_METADATA_REQUIRED")
            except (ValueError, KeyError) as exc:
                error_code = str(exc).strip("'")
                error_detail = str(exc)
                evidence["error_code"] = error_code
                evidence["review_state"] = "PENDING_REVIEW"
                if result_class in {"SOURCE_UNAVAILABLE", "RETRY_PENDING", "SOURCE_ACCESS_DENIED"}:
                    status = "RETRY_PENDING" if claimed["attempt_count"] < int(policy["retry_policy"].get("max_attempts", 3)) else "FAILED"
                if uri.startswith("unavailable:"):
                    error_code = "ACQUISITION_URI_MISSING"
                    status, result_class = "RETRY_PENDING", "SOURCE_UNAVAILABLE"
            check_status = "UNCHANGED" if status == "SUCCEEDED" and result_class == "SEMANTICALLY_UNCHANGED" else (
                "CHANGE_DETECTED" if "comparison" in evidence else result_class)
            completed_items.append(self._store.complete_scan_item(
                scan_item_id=claimed["scan_item_id"], status=status, result_class=result_class,
                error_code=error_code, error_detail=error_detail, lease_owner=lease_owner,
                expected_attempt=claimed["attempt_count"], refresh_result=evidence, checked_uri=uri,
                check_status=check_status, checked_at=observation["retrieved_at"] if observation else checked_at,
                observed_content=bytes.fromhex(observation["content_hex"]) if observation else None))
        run_ids = list(dict.fromkeys(item["scan_run_id"] for item in completed_items)) or [run.scan_run["scan_run_id"]]
        runs = [self._store.finalize_scan_run(scan_run_id=rid) for rid in run_ids]
        statuses = {entry["status"] for entry in runs}
        aggregate_status = "RUNNING" if "RUNNING" in statuses else (
            "PARTIAL" if run.resolution_errors or len(statuses) > 1 else runs[-1]["status"])
        return {"scan_run": runs[-1], "scan_runs": runs, "aggregate_status": aggregate_status,
                "scan_items": completed_items,
                "resolution_errors": run.resolution_errors, "dispatch_accepted": True,
                "refresh_succeeded": bool(completed_items) and not run.resolution_errors
                    and all(item["status"] == "SUCCEEDED" for item in completed_items),
                "no_work": not completed_items and not run.resolution_errors}

    def create_or_resume_run(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        scheduled_at: str,
        trigger: str = "SCHEDULED",
        limit: int | None = None,
        offset: int = 0,
    ) -> RegulatoryMonitorRun:
        due_sources = self.list_due_sources(
            organization_id=organization_id,
            as_of=scheduled_at,
            limit=limit,
            offset=offset,
        )
        scan_items: list[dict[str, Any]] = []
        resolution_errors: list[dict[str, Any]] = []
        scan_run: dict[str, Any] | None = None
        for due in due_sources:
            resolution = self._store.resolve_monitor_policy(
                organization_id=organization_id,
                source_ref=due["source_id"],
                workspace_id=workspace_id,
                scheduled_at=scheduled_at,
                require_workspace=trigger != "SCHEDULED",
            )
            if not resolution["ok"]:
                resolution_errors.append(
                    {
                        "source_id": due["source_id"],
                        "error_code": resolution["error_code"],
                    }
                )
                continue
            policy = resolution["policy"]
            window = resolution["canonical_scheduled_window"]
            scan_run = self._store.create_or_resume_scan_run(
                organization_id=organization_id,
                workspace_id=workspace_id,
                monitor_policy_version=policy["monitor_policy_version"],
                scheduled_window=window,
                trigger=trigger,
            )
            scan_items.append(
                self._store.create_scan_item(
                    scan_run_id=scan_run["scan_run_id"],
                    organization_id=organization_id,
                    source_id=due["source_id"],
                    monitor_policy_version=policy["monitor_policy_version"],
                    scheduled_window=window,
                )
            )
        if not due_sources:
            for item in self._store.completed_scan_items_at(
                organization_id=organization_id, workspace_id=workspace_id, scheduled_at=scheduled_at,
            ):
                resolution = self._store.resolve_monitor_policy(
                    organization_id=organization_id, source_ref=item["source_id"],
                    workspace_id=workspace_id, scheduled_at=scheduled_at,
                    require_workspace=trigger != "SCHEDULED",
                )
                if (resolution["ok"]
                    and resolution["policy"]["monitor_policy_version"] == item["monitor_policy_version"]
                    and resolution["canonical_scheduled_window"] == item["scheduled_window"]):
                    scan_items.append(item)
                    scan_run = self._store.get_scan_run(
                        organization_id=organization_id, scan_run_id=item["scan_run_id"],
                    )["scan_run"]
        if scan_run is None:
            fallback_window = canonical_scheduled_window(
                scheduled_at=scheduled_at,
                schedule_anchor_utc=scheduled_at,
                interval_seconds=1,
            )
            scan_run = self._store.create_or_resume_scan_run(
                organization_id=organization_id,
                workspace_id=workspace_id,
                monitor_policy_version="NO_RESOLVED_POLICY",
                scheduled_window=fallback_window,
                trigger=trigger,
            )
        return RegulatoryMonitorRun(
            scan_run=scan_run,
            scan_items=scan_items,
            resolution_errors=resolution_errors,
        )

    def list_due_sources(
        self,
        *,
        organization_id: str,
        as_of: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self._store.due_refresh_sources(
            organization_id=organization_id,
            as_of=as_of,
            limit=limit,
            offset=offset,
        )

    def claim_source(
        self,
        *,
        scan_item_id: str,
        lease_owner: str,
        lease_expires_at: str,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        return self._store.claim_scan_item(
            scan_item_id=scan_item_id,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            as_of=as_of,
        )

    def register_refresh_evidence(
        self,
        *,
        source_id: str,
        checked_uri: str,
        status: str,
        result: dict[str, Any],
        observed_content: bytes | None = None,
        checked_at: str | None = None,
    ) -> dict[str, Any]:
        return self._store.record_refresh_check(
            source_id=source_id,
            checked_uri=checked_uri,
            status=status,
            result=result,
            observed_content=observed_content,
            checked_at=checked_at,
        )

    def complete_source(
        self,
        *,
        scan_item_id: str,
        status: str,
        result_class: str,
        error_code: str | None = None,
        error_detail: str | None = None,
        observed_version_id: str | None = None,
        refresh_check_id: str | None = None,
    ) -> dict[str, Any]:
        return self._store.complete_scan_item(
            scan_item_id=scan_item_id,
            status=status,
            result_class=result_class,
            error_code=error_code,
            error_detail=error_detail,
            observed_version_id=observed_version_id,
            refresh_check_id=refresh_check_id,
        )

    def finalize_run(self, *, scan_run_id: str) -> dict[str, Any]:
        return self._store.finalize_scan_run(scan_run_id=scan_run_id)

    def get_status(self, *, organization_id: str, source_id: str, as_of: str | None = None) -> dict[str, Any]:
        return self._store.get_source_status_projection(
            organization_id=organization_id,
            source_id=source_id,
            as_of=as_of or now_iso(),
        )

    def acquire_source(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_id: str,
        version_id: str,
        uri: str,
        storage_ref: str,
        scheduled_at: str,
        actor_ref: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        resolution = self._store.resolve_monitor_policy(
            organization_id=organization_id,
            source_ref=source_id,
            workspace_id=workspace_id,
            scheduled_at=scheduled_at,
        )
        if not resolution["ok"]:
            return {
                "ok": False,
                "error_code": resolution["error_code"],
                "source_id": source_id,
            }
        policy = resolution["policy"]
        adapter = OfficialSourceAcquisitionAdapter(
            fetcher=self._official_fetcher,
            official_domain_allowlist=policy["official_domain_allowlist"],
            max_artifact_bytes=int(policy["max_artifact_bytes"]),
            redirect_policy=policy["redirect_policy"],
            mime_allowlist=policy.get("mime_allowlist"),
        )
        artifact = adapter.acquire(
            store=self._store,
            source_id=source_id,
            version_id=version_id,
            uri=uri,
            storage_ref=storage_ref,
            actor_ref=actor_ref,
            organization_id=organization_id,
            correlation_id=correlation_id,
        )
        return {
            "ok": bool(artifact.get("artifact_id")) and artifact.get("event_type") != "QUARANTINED",
            "artifact": artifact,
            "policy_version_ref": policy["policy_version_ref"],
            "canonical_scheduled_window": resolution["canonical_scheduled_window"],
        }

    def normalize_observation(self, observation: dict[str, Any]) -> tuple[str, str]:
        raw = bytes.fromhex(observation["content_hex"])
        mime = observation["mime_type"].split(";")[0].strip().lower()
        if mime == "application/pdf":
            from plugins.document_extraction.parsers.pymupdf_native import parse_bytes_bound
            try:
                parsed = parse_bytes_bound(raw, artifact_ref=observation["artifact_id"],
                    source_sha256="sha256:" + hashlib.sha256(raw).hexdigest(), media_type=mime,
                    byte_length=len(raw), execution_scope_ref="legal-regulatory-monitor")
                pages = [{"page_number": page["page"], "text": "\n".join(
                    block["text"] for block in parsed["blocks"] if block["page"] == page["page"])}
                    for page in parsed["pages"]]
                text, _ = normalize_authoritative_pdf_pages(pages)
            except Exception as exc:
                raise ValueError("PDF_TEXT_LAYER_EXTRACTION_REQUIRED") from exc
            identity = "pdf_text_layer/" + parsed["parser"]["version"] + "/legal_text.v1"
        elif mime in {"text/html", "application/xhtml+xml", "text/plain"}:
            headers = observation["http_metadata"].get("headers", {})
            header = next((str(v) for k, v in headers.items() if k.lower() == "content-type"), mime)
            message = Message()
            message["content-type"] = header
            charset = message.get_content_charset() or "utf-8"
            try:
                canonical_charset = codecs.lookup(charset).name
                if canonical_charset not in {"utf-8", "utf-8-sig", "ascii", "gb18030", "gbk", "gb2312",
                                              "big5", "iso8859-1", "cp1252", "utf-16", "utf-16-le", "utf-16-be"}:
                    raise LookupError("unsupported source charset")
                decoded = raw.decode(charset, errors="strict")
            except (LookupError, UnicodeError) as exc:
                raise ValueError("UNSUPPORTED_CHARSET_OR_MALFORMED_TEXT") from exc
            if "\x00" in decoded or "\ufffd" in decoded:
                raise ValueError("MALFORMED_TEXT")
            text = normalize_legal_html(decoded) if mime != "text/plain" else normalize_legal_text(decoded)
            profile = "legal_text.v1" if mime == "text/plain" else "legal_html.inline.v2"
            identity = profile + "/" + mime + "/" + charset
        else:
            raise ValueError("UNSUPPORTED_EXTRACTION")
        if not text:
            raise ValueError("EMPTY_NORMALIZED_TEXT")
        return text, identity

    def compare_observation(
        self,
        *,
        prior_version_id: str,
        observed_version_id: str,
        checked_uri: str | None = None,
        checked_at: str | None = None,
    ) -> dict[str, Any]:
        comparison = self._store.compare_source_versions(
            prior_version_id=prior_version_id,
            observed_version_id=observed_version_id,
        )
        if checked_uri:
            status = (
                "UNCHANGED"
                if comparison["comparison_class"] == "SEMANTICALLY_UNCHANGED"
                else "CHANGE_DETECTED"
            )
            if comparison["comparison_class"] == "REPRESENTATION_CHANGED":
                status = "CHANGE_DETECTED"
            refresh_check = self._store.record_refresh_check(
                source_id=comparison["source_id"],
                checked_uri=checked_uri,
                status=status,
                result={
                    "comparison": comparison,
                    "review_state": "PENDING_REVIEW"
                    if comparison["candidate_required"]
                    else "NO_AUTHORITY_REVIEW_REQUIRED",
                    "activation_performed": False,
                },
                checked_at=checked_at,
            )
            comparison["refresh_check_id"] = refresh_check["refresh_check_id"]
        return comparison

    def register_observed_version(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("observed source-version registration must use LegalCorpusStore")

    def create_review_candidate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("candidate creation must use the legal candidate compiler/store")

    def compute_impact(self, *, organization_id: str, candidate_id: str) -> dict[str, Any]:
        return self._store.compute_candidate_impact(
            organization_id=organization_id,
            candidate_id=candidate_id,
        )

    def acknowledge_alert(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("regulatory alert acknowledgement persistence is not implemented in this phase")
