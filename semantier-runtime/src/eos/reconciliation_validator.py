"""Reconciliation validation — ReconciliationValidation_t rule engine.

Architecture ref: docs/canonical/architecture.md §22.9.10

Checks logical relationships across accounts:
- Assets = Liabilities + Equity
- ΔRetainedEarnings ≈ NetIncome − Distributions (closed periods only)

Reconciliation failures are stronger signals than direction warnings because
they indicate a broken relationship between derived financial views.

This module must not:
- query live memory or live LLMs
- mutate REA facts or projection results
- block REA fact persistence
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_ASSET_CLASSES = {"asset"}
_LIABILITY_CLASSES = {"liability"}
_EQUITY_CLASSES = {"equity"}
_REVENUE_CLASSES = {"revenue"}
_EXPENSE_CLASSES = {"expense", "cost"}

# Tolerance for floating-point balance comparison (e.g. rounding in projections)
BALANCE_TOLERANCE = 0.01


def _sum_class(accounts: List[Dict[str, Any]], *classes: str) -> float:
    total = 0.0
    for a in accounts:
        cls = (a.get("account_class") or "").lower()
        if cls in classes:
            debit = float(a.get("ending_debit") or 0)
            credit = float(a.get("ending_credit") or 0)
            if cls in _ASSET_CLASSES or cls in _EXPENSE_CLASSES:
                total += debit - credit
            else:
                total += credit - debit
    return total


def run_reconciliation_validation(
    accounts: List[Dict[str, Any]],
    period_close_state: str,
    retained_earnings_account_code: Optional[str],
    distributions_amount: float,
    pins: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run reconciliation validation over the trial balance.

    Parameters
    ----------
    accounts:
        Account balance rows from TrialBalanceView_t.
    period_close_state:
        ``"open"`` or ``"closed"``.  The retained-earnings check only runs
        when the period is ``"closed"``.
    retained_earnings_account_code:
        Account code for the retained earnings account.  Required for the
        ΔRetainedEarnings reconciliation check.
    distributions_amount:
        Total declared distributions (dividends) for the period.
    pins:
        Version pins (traceability only).

    Returns
    -------
    list[dict]
        ReconciliationValidation_t check result dicts.
    """
    results: List[Dict[str, Any]] = []

    # --- Check 1: Assets = Liabilities + Equity ---
    total_assets = _sum_class(accounts, *_ASSET_CLASSES)
    total_liabilities = _sum_class(accounts, *_LIABILITY_CLASSES)
    total_equity = _sum_class(accounts, *_EQUITY_CLASSES)

    balance_diff = abs(total_assets - (total_liabilities + total_equity))
    if balance_diff <= BALANCE_TOLERANCE:
        results.append({
            "check_id": "tb.reconciliation.accounting_equation",
            "check_type": "reconciliation",
            "severity": "info",
            "status": "passed",
            "affected_accounts": [],
            "observed_value": {
                "assets": total_assets,
                "liabilities": total_liabilities,
                "equity": total_equity,
                "diff": balance_diff,
            },
            "expected_value": {"diff": 0},
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })
    else:
        results.append({
            "check_id": "tb.reconciliation.accounting_equation",
            "check_type": "reconciliation",
            "severity": "critical",
            "status": "failed",
            "affected_accounts": [],
            "observed_value": {
                "assets": total_assets,
                "liabilities": total_liabilities,
                "equity": total_equity,
                "diff": balance_diff,
            },
            "expected_value": {"diff": 0},
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    # --- Check 2: ΔRetainedEarnings ≈ NetIncome − Distributions (closed only) ---
    if period_close_state == "closed" and retained_earnings_account_code:
        # Find retained earnings account
        re_acct = next(
            (a for a in accounts if a.get("account_code") == retained_earnings_account_code),
            None,
        )
        if re_acct is not None:
            re_opening = float(re_acct.get("opening_credit") or 0) - float(re_acct.get("opening_debit") or 0)
            re_ending = float(re_acct.get("ending_credit") or 0) - float(re_acct.get("ending_debit") or 0)
            re_delta = re_ending - re_opening

            net_income = _sum_class(accounts, *_REVENUE_CLASSES) - _sum_class(accounts, *_EXPENSE_CLASSES)
            expected_delta = net_income - distributions_amount
            re_diff = abs(re_delta - expected_delta)

            if re_diff <= BALANCE_TOLERANCE:
                results.append({
                    "check_id": "tb.reconciliation.retained_earnings_net_income",
                    "check_type": "reconciliation",
                    "severity": "info",
                    "status": "passed",
                    "affected_accounts": [
                        {"account_code": retained_earnings_account_code, "account_name": re_acct.get("account_name", "")}
                    ],
                    "observed_value": {
                        "retained_earnings_delta": re_delta,
                        "net_income": net_income,
                        "distributions": distributions_amount,
                        "expected_delta": expected_delta,
                        "diff": re_diff,
                    },
                    "expected_value": {"diff": 0},
                    "evidence_refs": [],
                    "user_feedback_ref": None,
                    "justification_ref": None,
                })
            else:
                results.append({
                    "check_id": "tb.reconciliation.retained_earnings_net_income",
                    "check_type": "reconciliation",
                    "severity": "critical",
                    "status": "failed",
                    "affected_accounts": [
                        {"account_code": retained_earnings_account_code, "account_name": re_acct.get("account_name", "")}
                    ],
                    "observed_value": {
                        "retained_earnings_delta": re_delta,
                        "net_income": net_income,
                        "distributions": distributions_amount,
                        "expected_delta": expected_delta,
                        "diff": re_diff,
                    },
                    "expected_value": {"diff": 0},
                    "evidence_refs": [],
                    "user_feedback_ref": None,
                    "justification_ref": None,
                })

    return results
