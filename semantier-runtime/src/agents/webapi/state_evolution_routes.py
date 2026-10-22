"""Authenticated State Evolution routes for core-owned continuous simulations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from agents.auth_session import request_context_from_request
from contracts.state_evolution import evolution_hash
from services.state_evolution_compiler import compile_continuous_evolution
from services.state_evolution_service import (
    authenticated_state_evolution_store,
    invoke_state_evolution_worker,
    project_continuous_result,
    replay_evolution_run,
    seal_evolution_result,
    state_evolution_worker_build_pin,
)

router = APIRouter(tags=["state-evolution"])


def _code_pin() -> dict:
    import contracts.state_evolution as evolution_contract
    import services.state_evolution_compiler as evolution_compiler
    import services.state_evolution_service as evolution_service

    root = Path(__file__).resolve().parents[3]
    paths = [Path(module.__file__).resolve() for module in (evolution_contract, evolution_compiler, evolution_service)]
    inventory = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return {
        "artifact_kind": "state_evolution_compiler",
        "artifact_ref": "semantier:state-evolution-continuous-compiler",
        "artifact_hash": evolution_hash(inventory),
    }


def _create_continuous_run(ctx, body: dict) -> dict:
    session_ref = body.get("sessionRef")
    scenario_id = body.get("scenarioId", "baseline")
    if not isinstance(session_ref, str) or not session_ref.strip():
        raise ValueError("SESSION_REF_REQUIRED")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("SCENARIO_ID_REQUIRED")

    with authenticated_state_evolution_store(ctx, session_ref=session_ref, channel="web") as (store, scope, _):
        worker_pin = state_evolution_worker_build_pin()
        source = invoke_state_evolution_worker(ctx, "describe-continuous-source", worker_pin=worker_pin)
        compiler_pin = _code_pin()
        recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        comparison = None
        comparison_ref = None
        if scenario_id != "baseline":
            baseline_id = body.get("baselineRunId")
            comparison = (replay_evolution_run(store, baseline_id, trusted_scope=scope)
                          if isinstance(baseline_id, str) and baseline_id else
                          _execute_and_seal_continuous(ctx, store, scope, source, "baseline", compiler_pin, worker_pin, recorded_at))
            if comparison["plan"]["scenario"]["id"] != "baseline":
                raise ValueError("CONTINUOUS_BASELINE_SCENARIO_REQUIRED")
            comparison_ref = {
                "artifact_kind": "evolution_run",
                "artifact_ref": "evolution-run:" + comparison["run"]["runId"].split(":", 1)[1],
                "artifact_hash": comparison["run"]["runId"],
            }
        ir, plan = compile_continuous_evolution(
            scope=scope,
            source=source,
            scenario_id=scenario_id,
            compiler=compiler_pin,
            backend=worker_pin,
            producer=worker_pin,
            effective_at=recorded_at,
            comparison_basis_ref=comparison_ref,
        )
        owner_run = invoke_state_evolution_worker(
            ctx,
            "run-continuous",
            worker_pin=worker_pin,
            payload={"scenarioId": scenario_id},
        )
        run = project_continuous_result(
            owner_run,
            plan,
            attempt_id="attempt_" + uuid4().hex,
            recorded_at=recorded_at,
        )
        seal_evolution_result(
            store,
            ir,
            plan,
            run,
            trusted_scope=scope,
            resolved_source_pins=ir["sourcePins"],
            trusted_producer=worker_pin,
        )
        return {"ir": ir, "plan": plan, "run": run,
                "comparisonPlan": comparison["plan"] if comparison else plan,
                "comparisonRun": comparison["run"] if comparison else run}


def _execute_and_seal_continuous(ctx, store, scope, source, scenario_id, compiler_pin, worker_pin, recorded_at):
    ir, plan = compile_continuous_evolution(
        scope=scope, source=source, scenario_id=scenario_id, compiler=compiler_pin,
        backend=worker_pin, producer=worker_pin, effective_at=recorded_at,
    )
    owner_run = invoke_state_evolution_worker(
        ctx, "run-continuous", worker_pin=worker_pin, payload={"scenarioId": scenario_id},
    )
    run = project_continuous_result(
        owner_run, plan, attempt_id="attempt_" + uuid4().hex, recorded_at=recorded_at,
    )
    seal_evolution_result(
        store, ir, plan, run, trusted_scope=scope,
        resolved_source_pins=ir["sourcePins"], trusted_producer=worker_pin,
    )
    return {"ir": ir, "plan": plan, "run": run}


@router.post("/api/state-evolution/continuous/runs")
async def create_continuous_state_evolution_run(request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("INVALID_REQUEST")
        return await run_in_threadpool(_create_continuous_run, ctx, body)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status = 404 if message in {"SESSION_NOT_FOUND", "EVOLUTION_MISSING_ARTIFACT"} else 422 if message.endswith("_REQUIRED") or message.startswith("UNKNOWN_") else 409
        raise HTTPException(status_code=status, detail=message) from exc


@router.get("/api/state-evolution/continuous/runs/{run_id}")
async def replay_continuous_state_evolution_run(run_id: str, request: Request, sessionRef: str):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="authentication required")

    def replay():
        with authenticated_state_evolution_store(ctx, session_ref=sessionRef, channel="web") as (store, scope, _):
            selected = replay_evolution_run(store, run_id, trusted_scope=scope)
            comparison_ref = selected["plan"].get("comparisonBasisRef")
            comparison = selected if comparison_ref is None else replay_evolution_run(
                store, comparison_ref["artifact_hash"], trusted_scope=scope,
            )
            if comparison["plan"]["scenario"]["id"] != "baseline":
                raise ValueError("CONTINUOUS_BASELINE_SCENARIO_REQUIRED")
            return {**selected, "comparisonPlan": comparison["plan"], "comparisonRun": comparison["run"]}

    try:
        return await run_in_threadpool(replay)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status = 404 if message in {"SESSION_NOT_FOUND", "EVOLUTION_MISSING_ARTIFACT"} else 409
        raise HTTPException(status_code=status, detail=message) from exc
