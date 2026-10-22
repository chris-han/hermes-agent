from __future__ import annotations

from pathlib import Path
from typing import Any


class StoreBackedSensitiveLabelingReportService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        from storage.sensitive_labeling_report_store import SensitiveLabelingReportStore

        self._store = SensitiveLabelingReportStore(db_path=db_path)

    def create_report(self, *, run_id: str, created_by: str) -> dict[str, Any]:
        return self._store.create_report(run_id=run_id, created_by=created_by)

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        return self._store.get_report(report_id)
