"""Finite no-agent cron entrypoint for legal regulatory source monitoring."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from eos.db import now_iso
from eos.legal_regulatory_monitor import LegalRegulatoryMonitor


def run_regulatory_monitor_tick(
    *,
    organization_id: str,
    workspace_id: str | None,
    scheduled_at: str,
    trigger: str = "SCHEDULED",
    lease_owner: str = "regulatory-source-monitor-cron",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    return LegalRegulatoryMonitor().execute_tick(
        organization_id=organization_id,
        workspace_id=workspace_id,
        scheduled_at=scheduled_at,
        trigger=trigger,
        lease_owner=lease_owner,
        limit=limit,
        offset=offset,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one finite legal regulatory monitor tick.")
    parser.add_argument("--organization-id", default=os.getenv("SEMANTIER_LEGAL_MONITOR_ORGANIZATION_ID", ""))
    parser.add_argument("--workspace-id", default=os.getenv("SEMANTIER_LEGAL_MONITOR_WORKSPACE_ID", ""))
    parser.add_argument("--scheduled-at", default=os.getenv("SEMANTIER_LEGAL_MONITOR_SCHEDULED_AT", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    organization_id = str(args.organization_id or "").strip()
    if not organization_id:
        raise SystemExit("SEMANTIER_LEGAL_MONITOR_ORGANIZATION_ID or --organization-id is required")
    result = run_regulatory_monitor_tick(
        organization_id=organization_id,
        workspace_id=str(args.workspace_id or "").strip() or None,
        scheduled_at=str(args.scheduled_at or "").strip() or now_iso(),
        limit=args.limit,
        offset=args.offset,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
