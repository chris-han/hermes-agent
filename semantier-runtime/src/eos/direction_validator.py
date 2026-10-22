"""Direction validation — DirectionValidation_t rule engine.

Architecture ref: docs/canonical/architecture.md §22.9.8

Evaluates each account's ending balance side against the COA normal_balance
direction.  Accounts flagged with ``contra_account: true`` in the COA entry
are exempt from the direction check.

This module must not:
- query live memory or live LLMs
- mutate REA facts or projection results
- encode semantic authority in SQL
"""

from __future__ import annotations

from typing import Any, Dict, List

# Account classes whose normal balance is DEBIT
_DEBIT_NORMAL_CLASSES = {"asset", "cost", "expense"}

# Account classes whose normal balance is CREDIT
_CREDIT_NORMAL_CLASSES = {"liability", "equity", "revenue"}


def _net_ending_balance(account: Dict[str, Any]) -> tuple[float, str]:
    """Return (net_amount, 'debit'|'credit') for the ending balance of an account."""
    debit = float(account.get("ending_debit") or 0)
    credit = float(account.get("ending_credit") or 0)
    if debit >= credit:
        return debit - credit, "debit"
    return credit - debit, "credit"


def run_direction_validation(
    accounts: List[Dict[str, Any]],
    coa_entries: Dict[str, Dict[str, Any]],
    pins: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run direction validation over all accounts in the trial balance.

    Parameters
    ----------
    accounts:
        List of account balance rows from ``TrialBalanceView_t``.  Each row
        must include at minimum:
        ``account_code``, ``account_name``, ``account_class``,
        ``ending_debit``, ``ending_credit``.
    coa_entries:
        Mapping from ``account_code`` to the full COA entry dict.  Each entry
        should include ``normal_balance`` ("debit"|"credit") and optionally
        ``contra_account`` (bool).
    pins:
        Version pins from the validation run (for traceability only; this
        function does not use them for logic).

    Returns
    -------
    list[dict]
        List of DirectionValidation_t check result dicts.  One entry per
        account that was evaluated.  Accounts with zero balance are skipped.
    """
    results: List[Dict[str, Any]] = []

    for acct in accounts:
        code = acct.get("account_code", "")
        name = acct.get("account_name", "")
        account_class = (acct.get("account_class") or "").lower()

        net_amount, actual_side = _net_ending_balance(acct)

        # Skip zero-balance accounts — no direction to evaluate
        if net_amount == 0:
            continue

        coa_entry = coa_entries.get(code, {})
        contra = bool(coa_entry.get("contra_account", False))

        # Explicit normal_balance in COA takes precedence over class-based rule
        expected_side: str
        coa_normal = (coa_entry.get("normal_balance") or "").lower()
        if coa_normal in ("debit", "credit"):
            expected_side = coa_normal
        elif account_class in _DEBIT_NORMAL_CLASSES:
            expected_side = "debit"
        elif account_class in _CREDIT_NORMAL_CLASSES:
            expected_side = "credit"
        else:
            # Unknown class — cannot evaluate direction
            continue

        # Contra accounts have the opposite normal balance — they are exempt
        if contra:
            results.append({
                "check_id": f"tb.direction.{code}",
                "check_type": "direction",
                "severity": "info",
                "status": "passed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"side": actual_side, "amount": net_amount},
                "expected_value": {"note": "contra_account_exempt"},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })
            continue

        if actual_side == expected_side:
            results.append({
                "check_id": f"tb.direction.{code}",
                "check_type": "direction",
                "severity": "info",
                "status": "passed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"side": actual_side, "amount": net_amount},
                "expected_value": {"side": expected_side},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })
        else:
            results.append({
                "check_id": f"tb.direction.{code}",
                "check_type": "direction",
                "severity": "warning",
                "status": "failed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"side": actual_side, "amount": net_amount},
                "expected_value": {"side": expected_side},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })

    return results
