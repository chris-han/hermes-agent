from __future__ import annotations

from storage import hosted_postgres as hosted

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from eos import real_company_setup_store
from eos.db import get_conn, init_db, now_iso


ADMISSION_PROFILE_VERSION = "real_company_rea_admission.v1"
ADMISSION_PROFILE_HASH = hashlib.sha256(ADMISSION_PROFILE_VERSION.encode("utf-8")).hexdigest()


class RealCompanyAdmissionError(RuntimeError):
    pass


class RealCompanyAdmissionAuthorizationError(RealCompanyAdmissionError):
    pass


def admit_rea_claims(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
    member_role: str | None = None,
    can_change_settings: bool = False,
) -> dict[str, Any]:
    init_db()
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="rea_admission",
    )
    if run.get("ok") is False:
        return {
            "ok": False,
            "setup_run": run,
            "admitted_claim_count": 0,
            "existing_claim_count": 0,
            "exception_count": 0,
            "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
        }
    if run.get("status") in real_company_setup_store.TERMINAL_RUN_STATUSES:
        return {
            "ok": True,
            "setup_run": run,
            "admitted_claim_count": 0,
            "existing_claim_count": _count_admitted_claims(
                organization_id=organization_id,
                dataset_version_id=dataset_version_id,
            ),
            "exception_count": len(
                real_company_setup_store.list_admission_exceptions(
                    organization_id=organization_id,
                    dataset_version_id=dataset_version_id,
                )
            ),
            "blocking_reasons": [],
            "admission_profile_hash": ADMISSION_PROFILE_HASH,
        }

    source_rows = real_company_setup_store.list_source_rows(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
    )
    admitted = 0
    existing = 0
    exceptions = 0
    for source_row in source_rows:
        if source_row.get("row_type") != "economic_event":
            real_company_setup_store.record_admission_exception(
                organization_id=organization_id,
                dataset_version_id=dataset_version_id,
                source_row_hash=str(source_row["source_row_hash"]),
                admission_profile_hash=ADMISSION_PROFILE_HASH,
                reason_code="UNSUPPORTED_SOURCE_ROW",
                details={"row_type": source_row.get("row_type")},
            )
            exceptions += 1
            continue
        claim = _claim_from_source_row(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            source_row=source_row,
            requested_by=requested_by,
        )
        if _existing_claim_id(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            source_row_hash=str(source_row["source_row_hash"]),
        ):
            existing += 1
            continue
        _insert_rea_claim(claim)
        admitted += 1

    event_status = "ready" if admitted or existing else "blocked"
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="rea_admission",
        event_type="rea_claims_admitted",
        event_status=event_status,
        details={
            "admitted_claim_count": admitted,
            "existing_claim_count": existing,
            "exception_count": exceptions,
            "admission_profile_hash": ADMISSION_PROFILE_HASH,
        },
    )
    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="completed" if event_status == "ready" else "blocked",
        current_stage="rea_admission",
        blocking_reason=None if event_status == "ready" else "NO_ADMISSIBLE_SOURCE_ROWS",
    )
    return {
        "ok": True,
        "setup_run": run,
        "admitted_claim_count": admitted,
        "existing_claim_count": existing,
        "exception_count": exceptions,
        "blocking_reasons": [] if event_status == "ready" else ["NO_ADMISSIBLE_SOURCE_ROWS"],
        "admission_profile_hash": ADMISSION_PROFILE_HASH,
    }


def legacy_claims_report(
    *,
    organization_id: str,
    dataset_version_id: str | None = None,
) -> dict[str, Any]:
    init_db()
    fingerprints: list[dict[str, str]] = []
    with get_conn() as conn:
        rows = conn.execute("SELECT rea_id, claim_json, claim_hash FROM rea_claims").fetchall()
    for row in rows:
        try:
            claim = json.loads(row["claim_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if str(claim.get("org_id") or "") != organization_id:
            continue
        if str(claim.get("dataset_version_id") or "").strip():
            continue
        fingerprints.append(
            {
                "rea_id": str(row["rea_id"]),
                "claim_hash": str(row["claim_hash"]),
                "fingerprint": hashlib.sha256(str(row["claim_json"]).encode("utf-8")).hexdigest(),
            }
        )
    return {
        "ok": True,
        "organization_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "unversioned_legacy_claim_count": len(fingerprints),
        "fingerprints": fingerprints,
    }


def _claim_from_source_row(
    *,
    organization_id: str,
    dataset_version_id: str,
    source_row: dict[str, Any],
    requested_by: str,
) -> dict[str, Any]:
    row = source_row.get("row") or {}
    metadata = source_row.get("metadata") or {}
    classification = metadata.get("classification") or {}
    amount_key = str(classification.get("amount_field_key") or "")
    amount = _parse_amount(row.get(amount_key))
    if amount is None:
        amount = 0.0
    claim = {
        "org_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "resource_type": _resource_type(row, metadata),
        "amount": amount,
        "currency": _first_labeled_value(row, metadata, ("currency", "币种")) or "CNY",
        "counterparty": _first_labeled_value(row, metadata, ("counterparty", "供应商", "客户", "对方")) or "",
        "occurred_at": _occurred_at(row, metadata),
        "evidence_ref": _evidence_ref(source_row),
        "projectable": True,
        "blocked": False,
        "source_provenance": {
            "import_id": source_row.get("import_id"),
            "file_id": source_row.get("file_id"),
            "source_row_hash": source_row.get("source_row_hash"),
            "source_file_hash": source_row.get("source_file_hash"),
            "parser_profile_hash": source_row.get("parser_profile_hash"),
            "field_mapping_profile_hash": source_row.get("field_mapping_profile_hash"),
            "admission_profile_hash": ADMISSION_PROFILE_HASH,
        },
        "admitted_by": requested_by,
        "admission_profile_hash": ADMISSION_PROFILE_HASH,
    }
    claim_hash = _hash_json(claim)
    return {
        "rea_id": f"real_rea_{claim_hash[:24]}",
        "workflow_id": f"real_company_admission:{dataset_version_id}",
        "claim_json": _json_dumps(claim),
        "claim_hash": claim_hash,
        "created_at": now_iso(),
    }


def _insert_rea_claim(claim: dict[str, Any]) -> None:
    created = datetime.fromisoformat(str(claim["created_at"]))
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
        raise ValueError("created_at must be timezone-aware UTC")
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO rea_claims (rea_id, workflow_id, claim_json, claim_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, """
            INSERT INTO rea_claims (rea_id, workflow_id, claim_json, claim_hash, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """),
            (
                claim["rea_id"],
                claim["workflow_id"],
                claim["claim_json"],
                claim["claim_hash"],
                claim["created_at"],
            ),
        )


def _existing_claim_id(
    *,
    organization_id: str,
    dataset_version_id: str,
    source_row_hash: str,
) -> str | None:
    with get_conn() as conn:
        rows = conn.execute("SELECT rea_id, claim_json FROM rea_claims").fetchall()
    for row in rows:
        try:
            claim = json.loads(row["claim_json"] or "{}")
        except json.JSONDecodeError:
            continue
        provenance = claim.get("source_provenance") or {}
        if (
            str(claim.get("org_id") or "") == organization_id
            and str(claim.get("dataset_version_id") or "") == dataset_version_id
            and str(provenance.get("source_row_hash") or "") == source_row_hash
            and str(provenance.get("admission_profile_hash") or "") == ADMISSION_PROFILE_HASH
        ):
            return str(row["rea_id"])
    return None


def _count_admitted_claims(
    *,
    organization_id: str,
    dataset_version_id: str,
) -> int:
    count = 0
    with get_conn() as conn:
        rows = conn.execute("SELECT claim_json FROM rea_claims").fetchall()
    for row in rows:
        try:
            claim = json.loads(row["claim_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if (
            str(claim.get("org_id") or "") == organization_id
            and str(claim.get("dataset_version_id") or "") == dataset_version_id
        ):
            count += 1
    return count




def _first_labeled_value(
    row: dict[str, Any],
    metadata: dict[str, Any],
    tokens: tuple[str, ...],
) -> str | None:
    for column in metadata.get("columns") or []:
        label = str(column.get("source_column_label") or "").casefold()
        if any(token.casefold() in label for token in tokens):
            value = str(row.get(column.get("field_key")) or "").strip()
            if value:
                return value
    return None


def _resource_type(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    labeled = _first_labeled_value(row, metadata, ("type", "类型", "科目", "摘要", "名称"))
    if labeled:
        return _ascii_slug(labeled) or "source_event"
    return "source_event"


def _occurred_at(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = _first_labeled_value(row, metadata, ("date", "日期", "时间"))
    if value:
        normalized = _normalize_date(value)
        if normalized:
            return normalized
    return "1970-01-01T00:00:00+00:00"


def _normalize_date(value: str) -> str | None:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _evidence_ref(source_row: dict[str, Any]) -> str:
    return ":".join(
        [
            "real_company_source_row",
            str(source_row.get("dataset_version_id") or ""),
            str(source_row.get("source_row_hash") or ""),
        ]
    )


def _parse_amount(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    text = re.sub(r"^[^\d.-]+", "", text)
    text = re.sub(r"[^\d.-]+$", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def _ascii_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _hash_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
