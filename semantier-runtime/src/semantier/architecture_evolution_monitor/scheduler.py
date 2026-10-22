from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

from agents import cron_store
from eos.artifact_hashing import canonical_json, content_hash

from .collectors import collect_architecture_metric_snapshot
from .store import ArchitectureEvolutionMonitorStore


@dataclass(frozen=True)
class MonitorSchedule:
    schedule_ref: str
    organization_id: str
    environment: str
    architecture_profile: str
    cron_expression: str
    repeat: int = 0
    enabled: bool = True

    def to_cron_job(self) -> dict[str, Any]:
        return {
            "id": self.schedule_ref,
            "name": "architecture-evolution-monitor",
            "schedule": self.cron_expression,
            "repeat": self.repeat,
            "enabled": self.enabled,
            "no_agent": True,
            "delivery": "local",
            "script": (
                "python -m semantier.architecture_evolution_monitor.scheduler "
                f"run --schedule-ref {self.schedule_ref}"
            ),
            "metadata": {
                "owner": "architecture-evolution-monitor",
                "organization_id": self.organization_id,
                "environment": self.environment,
                "architecture_profile": self.architecture_profile,
                "resource_ownership": ["architecture_monitor_scheduled_run"],
                "cleanup": "owned scheduled-run records are append-only; revoke by disabling schedule",
            },
        }


def register_monitor_schedule(*, scope_id: str, schedule: MonitorSchedule) -> dict[str, Any]:
    baseline = cron_store.list_jobs(scope_id)
    jobs = [job for job in baseline if job.get("id") != schedule.schedule_ref]
    job = schedule.to_cron_job()
    jobs.append(job)
    cron_store.replace_jobs(scope_id, jobs, expected_jobs=baseline)
    ArchitectureEvolutionMonitorStore().put_schedule(
        scope_id=scope_id,
        schedule=schedule.__dict__,
        cron_job=job,
    )
    return job


def list_monitor_schedules(scope_id: str) -> list[dict[str, Any]]:
    return [
        job
        for job in cron_store.list_jobs(scope_id)
        if (job.get("metadata") or {}).get("owner") == "architecture-evolution-monitor"
    ]


def run_scheduled_evaluation(
    *,
    scope_id: str,
    schedule_ref: str,
    workload: dict[str, Any],
    observed_from: str,
    observed_to: str,
    evaluation_evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schedule = ArchitectureEvolutionMonitorStore().get_schedule(scope_id, schedule_ref)
    if schedule is None:
        raise KeyError(schedule_ref)
    snapshot = collect_architecture_metric_snapshot(
        organization_id=schedule["organization_id"],
        environment=schedule["environment"],
        architecture_profile=schedule["architecture_profile"],
        observed_from=observed_from,
        observed_to=observed_to,
        workload=workload,
        evaluation_evidence_refs=evaluation_evidence_refs,
    )
    from .service import ArchitectureEvolutionMonitorService

    result = ArchitectureEvolutionMonitorService().scheduled_evaluate_snapshot(
        {**snapshot.to_dict(), "schedule_ref": schedule_ref}
    )
    run = {
        "schema_version": "architecture_monitor_scheduled_run.v1",
        "scheduled_run_id": f"scheduled-run:{content_hash([scope_id, schedule_ref, snapshot.content_hash])}",
        "scope_id": scope_id,
        "schedule_ref": schedule_ref,
        "snapshot_ref": snapshot.content_hash,
        "assessment_ref": result["assessment"]["content_hash"],
        "classification_ref": result["classification"]["content_hash"],
        "review_ref": (result.get("review_package") or {}).get("content_hash"),
        "output_hash": f"sha256:{content_hash(result)}",
    }
    ArchitectureEvolutionMonitorStore().put_scheduled_run(run)
    cron_store.save_output(scope_id, schedule_ref, canonical_json(run))
    return {"run": run, "result": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scope-id", default="platform")
    run.add_argument("--schedule-ref", required=True)
    run.add_argument("--observed-from", required=True)
    run.add_argument("--observed-to", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        payload = json.loads(sys.stdin.read() or "{}")
        output = run_scheduled_evaluation(
            scope_id=args.scope_id,
            schedule_ref=args.schedule_ref,
            workload=dict(payload.get("workload") or {}),
            observed_from=args.observed_from,
            observed_to=args.observed_to,
            evaluation_evidence_refs=list(payload.get("evaluation_evidence_refs") or []),
        )
        print(canonical_json(output["run"]))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
