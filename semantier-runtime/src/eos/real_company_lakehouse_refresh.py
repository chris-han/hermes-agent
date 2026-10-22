from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from eos import real_company_setup_store


def refresh_lakehouse(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
    db_path: str | None = None,
    sim_json: str | None = None,
    output_dir: str | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    if not organization_id.strip():
        raise ValueError("organization_id required")
    if not dataset_version_id.strip():
        raise ValueError("dataset_version_id required")
    if not requested_by.strip():
        raise ValueError("requested_by required")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key required")

    resolved_db_path = _resolve_db_path(db_path)
    resolved_sim_json = _resolve_sim_json(sim_json)
    resolved_output_dir = _resolve_output_dir(output_dir, resolved_db_path)

    real_company_setup_store.ensure_schema()
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="lakehouse_refresh",
    )
    if run.get("ok") is False:
        return {
            "ok": False,
            "setup_run": run,
            "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
        }
    if run.get("status") in real_company_setup_store.TERMINAL_RUN_STATUSES:
        return {
            "ok": True,
            "setup_run": run,
            "manifest": _read_manifest(resolved_output_dir),
            "blocking_reasons": [],
        }

    try:
        materialize_lakehouse = _load_materialize_lakehouse()
        outputs = materialize_lakehouse(
            argparse.Namespace(
                db_path=str(resolved_db_path),
                sim_json=str(resolved_sim_json),
                output_dir=str(resolved_output_dir),
                replace=replace,
            )
        )
        manifest = _read_manifest(resolved_output_dir)
        real_company_setup_store.append_setup_event(
            setup_run_id=str(run["setup_run_id"]),
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            stage="lakehouse_refresh",
            event_type="lakehouse_refreshed",
            event_status="ready",
            details={
                "manifest_path": str(resolved_output_dir / "lakehouse_manifest.json"),
                "manifest_version": manifest.get("manifest_version"),
                "datasets": sorted((manifest.get("datasets") or {}).keys()),
                "outputs": outputs,
            },
        )
        real_company_setup_store.transition_setup_run(
            setup_run_id=str(run["setup_run_id"]),
            status="completed",
            current_stage="lakehouse_refresh",
        )
        return {
            "ok": True,
            "setup_run": run,
            "manifest": manifest,
            "outputs": outputs,
            "blocking_reasons": [],
        }
    except Exception as exc:
        real_company_setup_store.append_setup_event(
            setup_run_id=str(run["setup_run_id"]),
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            stage="lakehouse_refresh",
            event_type="lakehouse_refresh_failed",
            event_status="failed",
            details={"error": str(exc)},
        )
        real_company_setup_store.transition_setup_run(
            setup_run_id=str(run["setup_run_id"]),
            status="failed",
            current_stage="lakehouse_refresh",
            blocking_reason="REAL_SETUP_FAILED",
        )
        return {
            "ok": False,
            "setup_run": run,
            "blocking_reasons": ["REAL_SETUP_FAILED"],
            "error": str(exc),
        }


def mark_lakehouse_stale(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
    reason: str,
) -> dict[str, Any]:
    real_company_setup_store.ensure_schema()
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="lakehouse_refresh",
    )
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="lakehouse_refresh",
        event_type="lakehouse_stale",
        event_status="stale",
        details={"reason": reason},
    )
    return {"ok": True, "setup_run": run, "blocking_reasons": ["REAL_LAKEHOUSE_STALE"]}


def _resolve_db_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    override = os.environ.get("SEMANTIER_EOS_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    state_dir = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    if state_dir:
        return (Path(state_dir).expanduser() / "eos.db").resolve()
    return (Path(__file__).resolve().parents[2] / ".semantier-home" / "eos.db").resolve()


def _resolve_sim_json(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    state_dir = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    if state_dir:
        return (Path(state_dir).expanduser() / "sim.json").resolve()
    return (Path(__file__).resolve().parents[2] / ".semantier-home" / "sim.json").resolve()


def _resolve_output_dir(value: str | None, db_path: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    override = os.environ.get("SEMANTIER_LAKEHOUSE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (db_path.parent / "lakehouse").resolve()


def _load_materialize_lakehouse():
    script_path = Path(__file__).resolve().parents[2] / "bootstrap" / "bootstrap_materialize_lakehouse.py"
    spec = importlib.util.spec_from_file_location("semantier_bootstrap_materialize_lakehouse", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load lakehouse materializer at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize_lakehouse


def _read_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "lakehouse_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
