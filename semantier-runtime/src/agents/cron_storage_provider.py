from __future__ import annotations

from pathlib import Path
from contextvars import ContextVar
from typing import Any

from agents import cron_store
from runtime_paths import workspace_runtime_home_path


class SemantierCronStorageProvider:
    def __init__(self):
        self._snapshots = ContextVar("semantier_cron_snapshots", default=None)

    @staticmethod
    def _scope(home: Path) -> str:
        resolved = Path(home).expanduser().resolve()
        try:
            expected = workspace_runtime_home_path(resolved.name).resolve()
        except ValueError as exc:
            raise RuntimeError('Cron storage requires a governed workspace home') from exc
        if resolved != expected:
            raise RuntimeError('Cron storage requires a governed workspace home')
        return f'workspace:{resolved.name}'

    def load_jobs(self, home: Path) -> list[dict[str, Any]]:
        scope = self._scope(home)
        jobs = cron_store.list_jobs(scope)
        snapshots = dict(self._snapshots.get() or {})
        snapshots[scope] = cron_store.jobs_snapshot(jobs)
        self._snapshots.set(snapshots)
        return jobs

    def save_jobs(self, home: Path, jobs: list[dict[str, Any]], *, removed_ids=None, replace=False) -> None:
        scope = self._scope(home)
        snapshots = dict(self._snapshots.get() or {})
        saved = cron_store.save_jobs(scope, jobs, removed_ids=removed_ids, replace=replace,
                                     expected_snapshot=snapshots.get(scope, {}))
        snapshots[scope] = cron_store.jobs_snapshot(saved)
        self._snapshots.set(snapshots)

    def save_output(self, home: Path, job_id: str, output: str) -> str:
        scope = self._scope(home)
        run_at = cron_store.utc_now_iso()
        cron_store.save_output(scope, job_id, output, run_at=run_at)
        return f"sqlite:semantier_cron_outputs/{scope}/{job_id}/{run_at}"

    def list_outputs(self, home: Path, job_id: str) -> list[dict[str, Any]]:
        return cron_store.list_outputs(self._scope(home), job_id, limit=100)

    def delete_outputs(self, home: Path, job_id: str) -> None:
        cron_store.delete_outputs(self._scope(home), job_id)

    def lookup_creation_identity(self, home: Path, key: str):
        return next((job for job in cron_store.list_jobs(self._scope(home), include_retired=True)
                     if (job.get("creation_identity") or {}).get("key") == key), None)

    def creation_identity(self):
        from agents.cron_creation_identity import creation_identity
        return creation_identity()
