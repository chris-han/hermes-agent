"""Period close validation — PeriodCloseValidation_t rule engine.

Architecture ref: docs/canonical/architecture.md §22.9.11

Validates that income-statement accounts (revenue, expense, cost) have zero
ending balances after period close.  This check MUST NOT run as a hard failure
during an open period (§TC7).

This module must not:
- query live memory or live LLMs
- mutate REA facts or projection results
- apply closing rules to open periods
"""

from __future__ import annotations

from typing import Any, Dict, List

_PNL_CLASSES = {"revenue", "expense", "cost"}

BALANCE_TOLERANCE = 0.01


def _ending_net_balance(account: Dict[str, Any]) -> float:
    debit = float(account.get("ending_debit") or 0)
    credit = float(account.get("ending_credit") or 0)
    return debit - credit


def run_period_close_validation(
    accounts: List[Dict[str, Any]],
    period_close_state: str,
    pins: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run period-close validation over income-statement accounts.

    Parameters
    ----------
    accounts:
        Account balance rows from TrialBalanceView_t.
    period_close_state:
        ``"open"`` or ``"closed"``.  Closing checks are skipped entirely when
        the period is open (no false failures as per §TC7).
    pins:
        Version pins (traceability only).

    Returns
    -------
    list[dict]
        PeriodCloseValidation_t check result dicts.  Returns an empty list
        when the period is open.
    """
    # Open period: income statement balances are expected to exist — skip checks
    if period_close_state != "closed":
        return []

    results: List[Dict[str, Any]] = []

    for acct in accounts:
        account_class = (acct.get("account_class") or "").lower()
        if account_class not in _PNL_CLASSES:
            continue

        code = acct.get("account_code", "")
        name = acct.get("account_name", "")
        net = abs(_ending_net_balance(acct))

        if net <= BALANCE_TOLERANCE:
            results.append({
                "check_id": f"tb.period_close.pnl_zero.{code}",
                "check_type": "period_close",
                "severity": "info",
                "status": "passed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"ending_balance": net},
                "expected_value": {"ending_balance": 0},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })
        else:
            results.append({
                "check_id": f"tb.period_close.pnl_zero.{code}",
                "check_type": "period_close",
                "severity": "high",
                "status": "failed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"ending_balance": net},
                "expected_value": {"ending_balance": 0, "flag": "incomplete_period_close"},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })

    return results
