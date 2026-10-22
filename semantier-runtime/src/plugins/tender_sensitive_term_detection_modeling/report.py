from __future__ import annotations

from pathlib import Path
from typing import Any

from services.sensitive_labeling_report_service import StoreBackedSensitiveLabelingReportService


def create_sensitive_labeling_report(**kwargs: Any) -> dict[str, Any]:
    db_path = kwargs.pop("db_path", None)
    service = kwargs.pop("service", None) or StoreBackedSensitiveLabelingReportService(
        db_path=Path(db_path) if db_path is not None else None
    )
    return service.create_report(**kwargs)
