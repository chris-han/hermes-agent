from __future__ import annotations

from typing import Any

from plugins.tender_sensitive_term_detection_modeling.report import create_sensitive_labeling_report as _plugin_create_report


def create_sensitive_labeling_report(
    *,
    run_id: str,
    created_by: str,
    store: Any | None = None,
) -> dict[str, Any]:
    if store is not None:
        return store.create_report(run_id=run_id, created_by=created_by)
    return _plugin_create_report(run_id=run_id, created_by=created_by)
