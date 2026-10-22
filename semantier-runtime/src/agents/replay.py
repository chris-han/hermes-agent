from __future__ import annotations

from typing import Any

from storage.tender_detection_report_store import TenderDetectionReportStore


def build_tender_detection_replay_bundle(
    *,
    run_id: str,
    organization_id: str,
    include_audit_details: bool,
) -> dict[str, Any]:
    bundle = TenderDetectionReportStore().build_replay_bundle(
        run_id=run_id,
        include_audit_details=include_audit_details,
    )
    if bundle["run"]["organization_id"] != organization_id:
        raise ValueError("tender detection run not found")
    return bundle
