from __future__ import annotations

from typing import Any

from eos import real_company_setup_store
from eos.real_company_lakehouse_refresh import refresh_lakehouse
from eos.real_company_lifecycle_materialization import materialize_lifecycle
from eos.real_company_rea_admission import RealCompanyAdmissionError, admit_rea_claims
from eos.real_company_setup_state import real_company_setup_status_payload
from eos.real_company_source_rows import reconcile_promoted_imports


def run_real_company_setup(
    *,
    organization_id: str,
    dataset_type: str,
    active_dataset_version_id: str | None,
    requested_by: str,
    idempotency_key: str,
    member_role: str | None = None,
    can_change_settings: bool = False,
    db_path: str | None = None,
    sim_json: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    if not organization_id.strip():
        raise ValueError("organization_id required")
    if not requested_by.strip():
        raise ValueError("requested_by required")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key required")

    status = real_company_setup_status_payload(
        organization_id=organization_id,
        dataset_type=dataset_type,
        active_dataset_version_id=active_dataset_version_id,
    )
    if status["setup_status"] in {"DEMO_ACTIVE", "REAL_EMPTY"}:
        return _result(status=status, setup_run=None, stage_results=[])

    dataset_version_id = str(
        active_dataset_version_id
        or status.get("snapshot", {}).get("active_dataset_version_id")
        or ""
    ).strip()
    if not dataset_version_id:
        return _result(status=status, setup_run=None, stage_results=[])

    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="run_setup",
    )
    if run.get("ok") is False:
        return {
            "ok": False,
            "setup_status": status["setup_status"],
            "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
            "next_actions": status.get("next_actions", []),
            "setup_run": run,
            "stage_results": [],
        }
    if run.get("status") in real_company_setup_store.TERMINAL_RUN_STATUSES:
        return _result(status=status, setup_run=run, stage_results=[])

    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="completed",
        current_stage="run_setup",
    )

    stage_results: list[dict[str, Any]] = []
    for _ in range(8):
        status = real_company_setup_status_payload(
            organization_id=organization_id,
            dataset_type=dataset_type,
            active_dataset_version_id=dataset_version_id,
        )
        setup_status = str(status["setup_status"])
        if setup_status in {"REAL_READY", "REAL_EMPTY", "LEGACY_REAL_CLAIMS_UNVERSIONED", "REAL_SETUP_FAILED"}:
            break

        if setup_status == "REAL_IMPORTED":
            stage_results.append(
                {
                    "stage": "reconcile_promoted_imports",
                    "result": reconcile_promoted_imports(
                        organization_id=organization_id,
                        dataset_version_id=dataset_version_id,
                        requested_by=requested_by,
                        idempotency_key=f"{idempotency_key}:reconcile",
                    ),
                }
            )
            continue

        if setup_status == "REAL_REA_ADMISSION_REQUIRED":
            try:
                admission = admit_rea_claims(
                    organization_id=organization_id,
                    dataset_version_id=dataset_version_id,
                    requested_by=requested_by,
                    idempotency_key=f"{idempotency_key}:admit",
                    member_role=member_role,
                    can_change_settings=can_change_settings,
                )
            except RealCompanyAdmissionError as exc:
                stage_results.append(
                    {
                        "stage": "rea_admission",
                        "result": {
                            "ok": False,
                            "blocking_reasons": [exc.__class__.__name__],
                            "error": str(exc),
                        },
                    }
                )
                break
            stage_results.append({"stage": "rea_admission", "result": admission})
            continue

        if setup_status == "REAL_COA_REQUIRED":
            break

        if setup_status in {"REAL_PROJECTION_REQUIRED", "REAL_LIFECYCLE_MATERIALIZATION_REQUIRED"}:
            stage_results.append(
                {
                    "stage": "lifecycle_materialization",
                    "result": materialize_lifecycle(
                        organization_id=organization_id,
                        dataset_version_id=dataset_version_id,
                        requested_by=requested_by,
                        idempotency_key=f"{idempotency_key}:lifecycle",
                    ),
                }
            )
            continue

        if setup_status == "REAL_LAKEHOUSE_STALE":
            stage_results.append(
                {
                    "stage": "lakehouse_refresh",
                    "result": refresh_lakehouse(
                        organization_id=organization_id,
                        dataset_version_id=dataset_version_id,
                        requested_by=requested_by,
                        idempotency_key=f"{idempotency_key}:lakehouse",
                        db_path=db_path,
                        sim_json=sim_json,
                        output_dir=output_dir,
                        replace=True,
                    ),
                }
            )
            continue

        break

    final_status = real_company_setup_status_payload(
        organization_id=organization_id,
        dataset_type=dataset_type,
        active_dataset_version_id=dataset_version_id,
    )
    return _result(status=final_status, setup_run=run, stage_results=stage_results)


def _result(
    *,
    status: dict[str, Any],
    setup_run: dict[str, Any] | None,
    stage_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ok": not bool(status.get("blocking_reasons")) or status.get("setup_status") == "REAL_READY",
        "setup_status": status.get("setup_status"),
        "blocking_reasons": status.get("blocking_reasons", []),
        "next_actions": status.get("next_actions", []),
        "status": status,
        "setup_run": setup_run,
        "stage_results": stage_results,
    }
