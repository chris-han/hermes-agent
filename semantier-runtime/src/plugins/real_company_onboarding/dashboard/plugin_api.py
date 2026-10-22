from __future__ import annotations

from typing import Any


def setup_status(
    *,
    organization_id: str,
    dataset_type: str | None = None,
    active_dataset_version_id: str | None = None,
) -> dict[str, Any]:
    from eos.real_company_setup_state import real_company_setup_status_payload
    from plugins.real_company_onboarding.tools import resolve_dataset_type

    return real_company_setup_status_payload(
        organization_id=organization_id,
        dataset_type=resolve_dataset_type(
            {"organization_id": organization_id, "dataset_type": dataset_type}
        ),
        active_dataset_version_id=active_dataset_version_id,
    )
