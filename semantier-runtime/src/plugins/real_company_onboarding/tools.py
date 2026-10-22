from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def real_company_setup_status(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_setup_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    payload = real_company_setup_status_payload(
        organization_id=resolved["organization_id"],
        dataset_type=resolved["dataset_type"],
        active_dataset_version_id=resolved["active_dataset_version_id"],
    )
    return _envelope(payload)


def real_company_reconcile_promoted_imports(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_source_rows import reconcile_promoted_imports
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = reconcile_promoted_imports(
        organization_id=_text(resolved.get("organization_id")),
        dataset_version_id=_optional_text(resolved.get("active_dataset_version_id")),
        requested_by=_text(args.get("requested_by")),
        idempotency_key=_optional_text(args.get("idempotency_key")),
    )
    status = real_company_setup_status_payload(
        organization_id=_text(resolved.get("organization_id")),
        dataset_type=_text(resolved.get("dataset_type")),
        active_dataset_version_id=_optional_text(resolved.get("active_dataset_version_id"))
        or result.get("dataset_version_id"),
    )
    return _envelope({**status, "result": result}, ok=bool(result.get("ok", True)))


def real_company_legacy_claims_report(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_rea_admission import legacy_claims_report

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = legacy_claims_report(
        organization_id=_text(resolved.get("organization_id")),
        dataset_version_id=_optional_text(resolved.get("active_dataset_version_id")),
    )
    return _envelope(
        {
            "setup_status": "LEGACY_REAL_CLAIMS_REPORT",
            "blocking_reasons": [],
            "result": result,
            "next_actions": ["run_versioned_admission_migration"]
            if result.get("unversioned_legacy_claim_count")
            else [],
        }
    )


def real_company_preview_source_rows(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_source_rows import preview_source_rows

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = preview_source_rows(
        organization_id=_text(resolved.get("organization_id")),
        dataset_version_id=_optional_text(resolved.get("active_dataset_version_id")),
        limit=int(args.get("limit") or 50),
    )
    return _envelope(
        {
            "setup_status": None,
            "blocking_reasons": result.get("blocking_reasons", []),
            "result": result,
            "next_actions": [],
        },
        ok=bool(result.get("ok", True)),
    )


def real_company_admit_rea_claims(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_rea_admission import admit_rea_claims
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_setup_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    dataset_type = _text(resolved.get("dataset_type"))
    if dataset_type != "REAL":
        status = real_company_setup_status_payload(
            organization_id=_text(resolved.get("organization_id")),
            dataset_type=dataset_type,
            active_dataset_version_id=_optional_text(resolved.get("active_dataset_version_id")),
        )
        return _envelope(status)

    try:
        result = admit_rea_claims(
            organization_id=_text(resolved.get("organization_id")),
            dataset_version_id=_text(resolved.get("active_dataset_version_id")),
            requested_by=_text(args.get("requested_by")),
            idempotency_key=_text(args.get("idempotency_key")),
            member_role=_optional_text(args.get("member_role")),
            can_change_settings=bool(args.get("can_change_settings")),
        )
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "setup_status": "REAL_REA_ADMISSION_REQUIRED",
                "blocking_reasons": [exc.__class__.__name__],
                "result": {"error": str(exc)},
                "next_actions": ["admit_rea_claims"],
            }
        )
    status = real_company_setup_status_payload(
        organization_id=_text(resolved.get("organization_id")),
        dataset_type=dataset_type,
        active_dataset_version_id=_text(resolved.get("active_dataset_version_id")),
    )
    return _envelope({**status, "result": result}, ok=bool(result.get("ok", True)))


def real_company_bootstrap_coa(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_coa_bootstrap import bootstrap_default_coa
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = bootstrap_default_coa(
        organization_id=_text(resolved.get("organization_id")),
        dataset_version_id=_text(resolved.get("active_dataset_version_id")),
        requested_by=_text(args.get("requested_by")),
        idempotency_key=_text(args.get("idempotency_key")),
        uploaded_coa_candidate=args.get("uploaded_coa_candidate")
        if isinstance(args.get("uploaded_coa_candidate"), dict)
        else None,
    )
    status = real_company_setup_status_payload(
        organization_id=_text(resolved.get("organization_id")),
        dataset_type=_text(resolved.get("dataset_type")),
        active_dataset_version_id=_text(resolved.get("active_dataset_version_id")),
    )
    return _envelope({**status, "result": result}, ok=bool(result.get("ok", True)))


def real_company_materialize_lifecycle(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_lifecycle_materialization import materialize_lifecycle
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = materialize_lifecycle(
        organization_id=_text(resolved.get("organization_id")),
        dataset_version_id=_text(resolved.get("active_dataset_version_id")),
        requested_by=_text(args.get("requested_by")),
        idempotency_key=_text(args.get("idempotency_key")),
    )
    status = real_company_setup_status_payload(
        organization_id=_text(resolved.get("organization_id")),
        dataset_type=_text(resolved.get("dataset_type")),
        active_dataset_version_id=_text(resolved.get("active_dataset_version_id")),
    )
    return _envelope({**status, "result": result}, ok=bool(result.get("ok", True)))


def real_company_run_setup(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_setup_orchestration import run_real_company_setup

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    result = run_real_company_setup(
        organization_id=_text(resolved.get("organization_id")),
        dataset_type=_text(resolved.get("dataset_type")),
        active_dataset_version_id=_optional_text(resolved.get("active_dataset_version_id")),
        requested_by=_text(args.get("requested_by")),
        idempotency_key=_text(args.get("idempotency_key")),
        member_role=_optional_text(args.get("member_role")),
        can_change_settings=bool(args.get("can_change_settings")),
        db_path=_optional_text(args.get("db_path")),
        sim_json=_optional_text(args.get("sim_json")),
        output_dir=_optional_text(args.get("output_dir")),
    )
    return _envelope(result, ok=bool(result.get("ok", True)))


def _run_stage(args: dict[str, Any], *, stage: str, event_status: str) -> str:
    from eos import real_company_setup_store
    from eos.real_company_setup_state import real_company_setup_status_payload

    resolved = _resolve_real_mutation_context(args)
    if resolved.get("error_code"):
        return _blocked_envelope(str(resolved["error_code"]), resolved)
    organization_id = _text(resolved.get("organization_id"))
    dataset_version_id = _text(resolved.get("active_dataset_version_id"))
    requested_by = _text(args.get("requested_by"))
    idempotency_key = _text(args.get("idempotency_key"))
    if not all([organization_id, dataset_version_id, requested_by, idempotency_key]):
        return _json(
            {
                "ok": False,
                "setup_status": "SETUP_INPUT_REQUIRED",
                "blocking_reasons": ["SETUP_INPUT_REQUIRED"],
                "result": {},
                "next_actions": [],
            }
        )
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage=stage,
    )
    if run.get("ok") is False:
        return _json(
            {
                "ok": False,
                "setup_status": "SETUP_RUN_LOCKED",
                "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
                "result": run,
                "next_actions": [],
            }
        )
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage=stage,
        event_type=f"{stage}_completed" if event_status == "ready" else f"{stage}_recorded",
        event_status=event_status,
        details={"source": "real_company_onboarding"},
    )
    if event_status == "ready":
        real_company_setup_store.transition_setup_run(
            setup_run_id=str(run["setup_run_id"]),
            status="completed",
            current_stage=stage,
        )
    status = real_company_setup_status_payload(
        organization_id=organization_id,
        dataset_type=_dataset_type(args),
        active_dataset_version_id=dataset_version_id,
    )
    return _envelope({**status, "result": {"setup_run": run, "stage": stage}})


def _envelope(payload: dict[str, Any], *, ok: bool = True) -> str:
    return _json(
        {
            "ok": ok,
            "setup_status": payload.get("setup_status"),
            "blocking_reasons": payload.get("blocking_reasons", []),
            "result": payload,
            "next_actions": payload.get("next_actions", []),
        }
    )


def _blocked_envelope(code: str, details: dict[str, Any]) -> str:
    return _json(
        {
            "ok": False,
            "setup_status": code,
            "blocking_reasons": [code],
            "result": {
                "error_code": code,
                "details": details,
                "handoff": "governed_query" if code == "DEMO_ACTIVE" else "resolve_active_workspace_organization",
            },
            "next_actions": ["governed_query"] if code == "DEMO_ACTIVE" else [],
        }
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _dataset_type(args: dict[str, Any]) -> str:
    return resolve_dataset_type(args)


def resolve_dataset_type(args: dict[str, Any]) -> str:
    explicit = _text(args.get("dataset_type")).upper()
    if explicit:
        return explicit

    organization_id = _text(args.get("organization_id"))
    if organization_id:
        try:
            from agents.gateway_identity import _load_organizations

            org = _load_organizations().get(organization_id) or {}
            governed_type = _text(org.get("dataset_type")).upper()
            if governed_type:
                return governed_type
        except Exception:
            pass
    return ""


def _resolve_setup_context(args: dict[str, Any]) -> dict[str, str | None]:
    organization_id = _text(args.get("organization_id"))
    dataset_type = _text(args.get("dataset_type")).upper()
    active_dataset_version_id = _optional_text(args.get("active_dataset_version_id"))

    active_context = _active_workspace_org_context()
    active_org_id = _text(active_context.get("organization_id")) if active_context else ""
    if active_context and active_org_id:
        organization_id = active_org_id
        dataset_type = _text(active_context.get("dataset_type")).upper()
        active_dataset_version_id = _optional_text(active_context.get("active_dataset_version_id"))

    if not dataset_type:
        dataset_type = resolve_dataset_type({"organization_id": organization_id})
    if not organization_id:
        return {
            "organization_id": organization_id,
            "dataset_type": dataset_type,
            "active_dataset_version_id": active_dataset_version_id,
            "error_code": "ACTIVE_ORGANIZATION_REQUIRED",
        }
    if not dataset_type:
        return {
            "organization_id": organization_id,
            "dataset_type": dataset_type,
            "active_dataset_version_id": active_dataset_version_id,
            "error_code": "DATASET_CONTEXT_REQUIRED",
        }
    return {
        "organization_id": organization_id,
        "dataset_type": dataset_type,
        "active_dataset_version_id": active_dataset_version_id,
    }


def _resolve_real_mutation_context(args: dict[str, Any]) -> dict[str, str | None]:
    resolved = _resolve_setup_context(args)
    if resolved.get("error_code"):
        return resolved
    if _text(resolved.get("dataset_type")).upper() != "REAL":
        return {**resolved, "error_code": "DEMO_ACTIVE"}
    if not _optional_text(resolved.get("active_dataset_version_id")):
        return {**resolved, "error_code": "ACTIVE_DATASET_VERSION_REQUIRED"}
    return resolved


def _active_workspace_org_context() -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_session_home()
    if not workspace_id:
        return None
    try:
        from agents import gateway_identity

        user = gateway_identity.get_user_record_by_workspace_id(workspace_id)
        if not isinstance(user, dict):
            return None
        user_id = _text(user.get("user_id"))
        if not user_id:
            return None
        org_context = gateway_identity.get_user_organization_context(user_id)
        return dict(org_context) if isinstance(org_context, dict) else None
    except Exception:
        return None


def _workspace_id_from_session_home() -> str | None:
    raw_home = os.environ.get("HERMES_HOME") or os.environ.get("HERMES_SESSION_HERMES_HOME")
    if not raw_home:
        return None
    try:
        hermes_home = Path(raw_home).expanduser().resolve()
    except OSError:
        return None
    workspace_id = hermes_home.parent.name.strip() if hermes_home.name == ".hermes" else hermes_home.name.strip()
    return workspace_id or None


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
