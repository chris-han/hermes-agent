from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from eos.db import get_conn


SETUP_STATUS_REMOVE_AUTHORITY_STATE_AFTER = "2026-07-31"


@dataclass(frozen=True)
class RealCompanySetupSnapshot:
    organization_id: str | None
    dataset_type: str | None
    active_dataset_version_id: str | None = None
    promoted_import_count: int = 0
    normalized_source_row_count: int = 0
    classified_source_row_count: int = 0
    admitted_rea_claim_count: int = 0
    unversioned_legacy_claim_count: int = 0
    active_coa_projection_ready: bool = False
    projection_result_count: int = 0
    lifecycle_artifact_count: int = 0
    required_lifecycle_artifact_count: int = 3
    lakehouse_ready: bool = False
    lakehouse_stale: bool = False
    failed_setup_run_count: int = 0
    extra_blocking_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RealCompanySetupStatus:
    setup_status: str
    blocking_reasons: tuple[str, ...]
    counts: dict[str, int]
    next_actions: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_status": self.setup_status,
            "blocking_reasons": list(self.blocking_reasons),
            "counts": dict(self.counts),
            "next_actions": list(self.next_actions),
        }


_NEXT_ACTIONS: dict[str, tuple[str, ...]] = {
    "DEMO_ACTIVE": (),
    "REAL_EMPTY": ("upload_source_files",),
    "REAL_IMPORTED": ("reconcile_promoted_imports", "preview_source_rows"),
    "REAL_REA_ADMISSION_REQUIRED": ("admit_rea_claims",),
    "LEGACY_REAL_CLAIMS_UNVERSIONED": ("run_legacy_claims_report",),
    "REAL_COA_REQUIRED": ("bootstrap_coa",),
    "REAL_PROJECTION_REQUIRED": ("materialize_lifecycle",),
    "REAL_LIFECYCLE_MATERIALIZATION_REQUIRED": ("materialize_lifecycle",),
    "REAL_LAKEHOUSE_STALE": ("refresh_lakehouse",),
    "REAL_SETUP_FAILED": ("inspect_setup_events",),
    "REAL_READY": (),
}


def evaluate_real_company_setup_state(
    snapshot: RealCompanySetupSnapshot,
) -> RealCompanySetupStatus:
    dataset_type = str(snapshot.dataset_type or "").upper()
    if dataset_type and dataset_type != "REAL":
        return RealCompanySetupStatus(
            setup_status="DEMO_ACTIVE",
            blocking_reasons=tuple(snapshot.extra_blocking_reasons),
            counts=_counts(snapshot),
            next_actions=_NEXT_ACTIONS["DEMO_ACTIVE"],
        )

    blockers: list[str] = []
    if snapshot.failed_setup_run_count:
        blockers.append("REAL_SETUP_FAILED")
    if not snapshot.active_dataset_version_id and snapshot.promoted_import_count == 0:
        blockers.append("REAL_EMPTY")
    if (
        snapshot.active_dataset_version_id
        and snapshot.classified_source_row_count == 0
        and snapshot.admitted_rea_claim_count == 0
    ):
        blockers.append("REAL_IMPORTED")
    if snapshot.classified_source_row_count > 0 and snapshot.admitted_rea_claim_count == 0:
        blockers.append("REAL_REA_ADMISSION_REQUIRED")
    if snapshot.unversioned_legacy_claim_count > 0:
        blockers.append("LEGACY_REAL_CLAIMS_UNVERSIONED")
    if snapshot.admitted_rea_claim_count > 0 and not snapshot.active_coa_projection_ready:
        blockers.append("REAL_COA_REQUIRED")
    if snapshot.active_coa_projection_ready and snapshot.projection_result_count == 0:
        blockers.append("REAL_PROJECTION_REQUIRED")
    if (
        snapshot.projection_result_count > 0
        and snapshot.lifecycle_artifact_count < snapshot.required_lifecycle_artifact_count
    ):
        blockers.append("REAL_LIFECYCLE_MATERIALIZATION_REQUIRED")
    if snapshot.lakehouse_stale:
        blockers.append("REAL_LAKEHOUSE_STALE")
    if snapshot.lifecycle_artifact_count >= snapshot.required_lifecycle_artifact_count and not snapshot.lakehouse_ready:
        blockers.append("REAL_LAKEHOUSE_STALE")
    blockers.extend(snapshot.extra_blocking_reasons)

    precedence = (
        "REAL_SETUP_FAILED",
        "REAL_LAKEHOUSE_STALE",
        "REAL_EMPTY",
        "REAL_IMPORTED",
        "REAL_REA_ADMISSION_REQUIRED",
        "LEGACY_REAL_CLAIMS_UNVERSIONED",
        "REAL_COA_REQUIRED",
        "REAL_PROJECTION_REQUIRED",
        "REAL_LIFECYCLE_MATERIALIZATION_REQUIRED",
    )
    setup_status = next((status for status in precedence if status in blockers), "REAL_READY")
    return RealCompanySetupStatus(
        setup_status=setup_status,
        blocking_reasons=tuple(dict.fromkeys(blockers)),
        counts=_counts(snapshot),
        next_actions=_NEXT_ACTIONS.get(setup_status, ()),
    )


def collect_real_company_setup_snapshot(
    *,
    organization_id: str | None,
    dataset_type: str | None,
    active_dataset_version_id: str | None = None,
) -> RealCompanySetupSnapshot:
    organization_id = str(organization_id or "").strip() or None
    dataset_type = str(dataset_type or "").strip() or None
    if not organization_id:
        return RealCompanySetupSnapshot(
            organization_id=None,
            dataset_type=dataset_type,
            extra_blocking_reasons=("ACTIVE_ORGANIZATION_REQUIRED",),
        )
    if str(dataset_type or "").upper() != "REAL":
        return RealCompanySetupSnapshot(
            organization_id=organization_id,
            dataset_type=dataset_type,
            active_dataset_version_id=active_dataset_version_id,
        )

    from eos import company_dataset_import_store

    company_dataset_import_store.ensure_schema()
    with get_conn() as conn:
        if not active_dataset_version_id:
            active = company_dataset_import_store.get_active_dataset_version(
                organization_id=organization_id,
            )
            if active:
                active_dataset_version_id = str(active.get("dataset_version_id") or "") or None
        return RealCompanySetupSnapshot(
            organization_id=organization_id,
            dataset_type=dataset_type,
            active_dataset_version_id=active_dataset_version_id,
            promoted_import_count=_promoted_import_count(conn, organization_id),
            normalized_source_row_count=_count_source_rows(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                classified=False,
            ),
            classified_source_row_count=_count_source_rows(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                classified=True,
            ),
            admitted_rea_claim_count=_count_rea_claims(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                versioned=True,
            ),
            unversioned_legacy_claim_count=_count_rea_claims(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                versioned=False,
            ),
            active_coa_projection_ready=_active_coa_projection_ready(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
            ),
            projection_result_count=_count_org_rows(
                conn,
                "journal_voucher_projections",
                "org_id",
                organization_id,
            ),
            lifecycle_artifact_count=_lifecycle_artifact_count(conn, organization_id),
            lakehouse_ready=_latest_marker_ready(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                stage="lakehouse_refresh",
            ),
            lakehouse_stale=_latest_marker_status(
                conn,
                organization_id=organization_id,
                dataset_version_id=active_dataset_version_id,
                stage="lakehouse_refresh",
            )
            == "stale",
            failed_setup_run_count=_failed_setup_run_count(conn, organization_id),
        )


def real_company_setup_status_payload(
    *,
    organization_id: str | None,
    dataset_type: str | None,
    active_dataset_version_id: str | None = None,
) -> dict[str, Any]:
    snapshot = collect_real_company_setup_snapshot(
        organization_id=organization_id,
        dataset_type=dataset_type,
        active_dataset_version_id=active_dataset_version_id,
    )
    status = evaluate_real_company_setup_state(snapshot)
    return {
        **status.to_payload(),
        "snapshot": asdict(snapshot),
    }


def _counts(snapshot: RealCompanySetupSnapshot) -> dict[str, int]:
    return {
        "promoted_import_count": snapshot.promoted_import_count,
        "normalized_source_row_count": snapshot.normalized_source_row_count,
        "classified_source_row_count": snapshot.classified_source_row_count,
        "admitted_rea_claim_count": snapshot.admitted_rea_claim_count,
        "unversioned_legacy_claim_count": snapshot.unversioned_legacy_claim_count,
        "projection_result_count": snapshot.projection_result_count,
        "lifecycle_artifact_count": snapshot.lifecycle_artifact_count,
        "failed_setup_run_count": snapshot.failed_setup_run_count,
    }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=current_schema() AND c.relkind IN ('r','v','m','p') AND c.relname=%s"),
        (table_name,),
    ).fetchone()
    return row is not None


def _promoted_import_count(conn: sqlite3.Connection, organization_id: str) -> int:
    if not _table_exists(conn, "company_dataset_imports"):
        return 0
    row = conn.execute(
        hosted.sql_query("""
        SELECT COUNT(*) AS n FROM company_dataset_imports
        WHERE organization_id = ? AND status = 'promoted'
        """, """
        SELECT COUNT(*) AS n FROM company_dataset_imports
        WHERE organization_id = %s AND status = 'promoted'
        """),
        (organization_id,),
    ).fetchone()
    return int(row["n"] if row else 0)


def _count_source_rows(
    conn: sqlite3.Connection,
    *,
    organization_id: str,
    dataset_version_id: str | None,
    classified: bool,
) -> int:
    if not dataset_version_id or not _table_exists(conn, "real_company_source_rows"):
        return 0
    status_clause = "AND classification_status IS NOT NULL" if classified else ""
    row = conn.execute(
        hosted.sql_query(f"""
        SELECT COUNT(*) AS n FROM real_company_source_rows
        WHERE organization_id = ? AND dataset_version_id = ? {status_clause}
        """, f'\n        SELECT COUNT(*) AS n FROM real_company_source_rows\n        WHERE organization_id = %s AND dataset_version_id = %s {status_clause}\n        '),
        (organization_id, dataset_version_id),
    ).fetchone()
    return int(row["n"] if row else 0)


def _count_rea_claims(
    conn: sqlite3.Connection,
    *,
    organization_id: str,
    dataset_version_id: str | None,
    versioned: bool,
) -> int:
    if not _table_exists(conn, "rea_claims"):
        return 0
    rows = conn.execute("SELECT claim_json FROM rea_claims").fetchall()
    count = 0
    for row in rows:
        try:
            claim = json.loads(row["claim_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if str(claim.get("org_id") or "") != organization_id:
            continue
        claim_dataset_version_id = str(claim.get("dataset_version_id") or "").strip()
        if versioned:
            if dataset_version_id and claim_dataset_version_id == dataset_version_id:
                count += 1
        elif not claim_dataset_version_id:
            count += 1
    return count


def _count_org_rows(
    conn: sqlite3.Connection,
    table_name: str,
    org_column: str,
    organization_id: str,
) -> int:
    if not _table_exists(conn, table_name):
        return 0
    row = conn.execute(
        hosted.sql_query(f"SELECT COUNT(*) AS n FROM {table_name} WHERE {org_column} = ?", f'SELECT COUNT(*) AS n FROM {table_name} WHERE {org_column} = %s'),
        (organization_id,),
    ).fetchone()
    return int(row["n"] if row else 0)


def _lifecycle_artifact_count(conn: sqlite3.Connection, organization_id: str) -> int:
    tables = (
        "general_ledger_views",
        "financial_statement_packages",
        "accounting_archive_packages",
    )
    return sum(
        1
        for table in tables
        if _count_org_rows(conn, table, "org_id", organization_id) > 0
    )


def _latest_marker_status(
    conn: sqlite3.Connection,
    *,
    organization_id: str,
    dataset_version_id: str | None,
    stage: str,
) -> str | None:
    if not dataset_version_id or not _table_exists(conn, "real_company_setup_events"):
        return None
    row = conn.execute(
        hosted.sql_query("""
        SELECT event_status FROM real_company_setup_events
        WHERE organization_id = ? AND dataset_version_id = ? AND stage = ?
        ORDER BY created_at DESC, event_id DESC
        LIMIT 1
        """, """
        SELECT event_status FROM real_company_setup_events
        WHERE organization_id = %s AND dataset_version_id = %s AND stage = %s
        ORDER BY created_at DESC, event_id DESC
        LIMIT 1
        """),
        (organization_id, dataset_version_id, stage),
    ).fetchone()
    return str(row["event_status"]) if row else None


def _latest_marker_ready(
    conn: sqlite3.Connection,
    *,
    organization_id: str,
    dataset_version_id: str | None,
    stage: str,
) -> bool:
    return _latest_marker_status(
        conn,
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage=stage,
    ) == "ready"


def _active_coa_projection_ready(
    conn: sqlite3.Connection,
    *,
    organization_id: str,
    dataset_version_id: str | None,
) -> bool:
    if not dataset_version_id:
        return False
    if _latest_marker_ready(
        conn,
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="coa_projection",
    ):
        return True
    if not (
        _table_exists(conn, "coa_projection_bundles")
        and _table_exists(conn, "projection_bundles")
        and _table_exists(conn, "replay_bindings")
    ):
        return False
    bundle = conn.execute(
        hosted.sql_query("""
        SELECT bundle_version FROM coa_projection_bundles
        WHERE org_id = ? AND status = 'ACTIVE'
        ORDER BY activated_at DESC, created_at DESC
        LIMIT 1
        """, """
        SELECT bundle_version FROM coa_projection_bundles
        WHERE org_id = %s AND status = 'ACTIVE'
        ORDER BY activated_at DESC, created_at DESC
        LIMIT 1
        """),
        (organization_id,),
    ).fetchone()
    projection = conn.execute(
        hosted.sql_query("""
        SELECT bundle_version FROM projection_bundles
        WHERE org_id = ? AND status = 'ACTIVE'
        ORDER BY activated_at DESC, created_at DESC
        LIMIT 1
        """, """
        SELECT bundle_version FROM projection_bundles
        WHERE org_id = %s AND status = 'ACTIVE'
        ORDER BY activated_at DESC, created_at DESC
        LIMIT 1
        """),
        (organization_id,),
    ).fetchone()
    binding = conn.execute(
        hosted.sql_query("""
        SELECT event_id FROM replay_bindings
        WHERE event_id = ? AND projection_bundle_version IS NOT NULL
        """, """
        SELECT event_id FROM replay_bindings
        WHERE event_id = %s AND projection_bundle_version IS NOT NULL
        """),
        (f"real_company_coa:{organization_id}:{dataset_version_id}",),
    ).fetchone()
    return bundle is not None and projection is not None and binding is not None


def _failed_setup_run_count(conn: sqlite3.Connection, organization_id: str) -> int:
    if not _table_exists(conn, "real_company_setup_runs"):
        return 0
    row = conn.execute(
        hosted.sql_query("""
        SELECT COUNT(*) AS n FROM real_company_setup_runs
        WHERE organization_id = ? AND status = 'failed'
        """, """
        SELECT COUNT(*) AS n FROM real_company_setup_runs
        WHERE organization_id = %s AND status = 'failed'
        """),
        (organization_id,),
    ).fetchone()
    return int(row["n"] if row else 0)
