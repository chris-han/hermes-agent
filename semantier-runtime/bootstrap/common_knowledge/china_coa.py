"""Real Chinese Chart of Accounts loaded from bootstrap/common_knowledge/t3-doctrine/accounting/coa_china_smb.json.

Source dataset:
    - CHINA_COA: standard CAS / 企业会计准则 account rows
    - CUSTOM_COA_ACCOUNTS: org-local demo extensions used by bootstrap
    - PROJECTION_RULE_ACCOUNTS: subset referenced by projection rules

Tier note:
    - The underlying external accounting structure is modeled as T3 doctrine data.
    - The organization's active runtime COA_v that adopts and extends this structure is T4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DATA_PATH = Path(__file__).parent / "t3-doctrine" / "accounting" / "coa_china_smb.json"


def _load_coa_payload() -> dict[str, Any]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


_PAYLOAD = _load_coa_payload()

CHINA_COA: list[dict[str, Any]] = list(_PAYLOAD["CHINA_COA"])
CUSTOM_COA_ACCOUNTS: list[dict[str, Any]] = list(_PAYLOAD["CUSTOM_COA_ACCOUNTS"])
ALL_COA_ACCOUNTS: list[dict[str, Any]] = CHINA_COA + CUSTOM_COA_ACCOUNTS
PROJECTION_RULE_ACCOUNTS: list[str] = list(_PAYLOAD["PROJECTION_RULE_ACCOUNTS"])
