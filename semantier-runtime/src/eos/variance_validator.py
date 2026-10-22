"""Variance validation — VarianceValidation_t rule engine.

Architecture ref: docs/canonical/architecture.md §22.9.9

Compares account movements across periods and flags material unexplained
changes.  A large variance is suspicious only when material AND unexplained
by evidence refs supplied by the caller.

This module must not:
- query live memory or live LLMs
- mutate REA facts or projection results
- treat cross-projection difference as contradiction (§TC8)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Default materiality threshold: 30% change ratio triggers variance check
DEFAULT_VARIANCE_THRESHOLD = 0.30

# Inventory/COGS mismatch: inventory drops ≥ 30% but COGS change < 10%
INVENTORY_DROP_THRESHOLD = -0.30
COGS_MOVEMENT_THRESHOLD = 0.10

# Long-aging receivable: balance > 90 days is flagged if material
LONG_AGING_DAYS = 90

# Balance-sheet ratio checks — Chinese tax authority fraud indicators
# 预付账款/总资产 > 10%
PREPAID_TO_ASSETS_THRESHOLD = 0.10
# (应收账款 + 合同资产) / 总资产 > 20%
RECEIVABLE_TO_ASSETS_THRESHOLD = 0.20
# 双高联动: 存货/总资产 > 15% 且 应收/总资产 > 5%
DUAL_HIGH_INVENTORY_THRESHOLD = 0.15
DUAL_HIGH_RECEIVABLE_THRESHOLD = 0.05
# 货币资金利息率异常: 利息收入 / 货币资金余额 < 2%
CASH_INTEREST_MIN_RATIO = 0.02

# Tax-burden variance — compare current vs prior period effective tax rate
# Flag when absolute deviation > 30 percentage points
TAX_BURDEN_VARIANCE_THRESHOLD = 0.30

# Invoice behaviour checks (SourceDocumentReview_t layer)
# 发票作废率 > 20%
INVOICE_VOID_RATE_THRESHOLD = 0.20
# 发票开票量环比突增 > 30%
INVOICE_SURGE_THRESHOLD = 0.30


def _period_net_balance(account: Dict[str, Any]) -> float:
    """Return the period net balance (debit - credit, signed)."""
    debit = float(account.get("period_debit") or 0)
    credit = float(account.get("period_credit") or 0)
    return debit - credit


def _ending_net_balance(account: Dict[str, Any]) -> float:
    debit = float(account.get("ending_debit") or 0)
    credit = float(account.get("ending_credit") or 0)
    return debit - credit


def _opening_net_balance(account: Dict[str, Any]) -> float:
    debit = float(account.get("opening_debit") or 0)
    credit = float(account.get("opening_credit") or 0)
    return debit - credit


def _variance_ratio(current: float, prior: float) -> Optional[float]:
    if prior == 0:
        return None  # Cannot compute ratio against zero base
    return (current - prior) / abs(prior)


def _account_by_class(accounts: List[Dict[str, Any]], *classes: str) -> List[Dict[str, Any]]:
    return [a for a in accounts if (a.get("account_class") or "").lower() in classes]


def run_variance_validation(
    accounts: List[Dict[str, Any]],
    prior_period_accounts: Optional[List[Dict[str, Any]]],
    evidence_refs: Optional[Dict[str, List[str]]],
    pins: Dict[str, Any],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Run variance validation over trial balance accounts.

    Parameters
    ----------
    accounts:
        Current-period account balance rows from TrialBalanceView_t.
    prior_period_accounts:
        Prior-period account balance rows for comparison.  If ``None``, all
        period-comparison checks are skipped.
    evidence_refs:
        Mapping from ``account_code`` to a list of evidence ref IDs that
        explain large movements.  Present evidence refs mark a check as
        "explained" and produce a passed result rather than a failure.
    pins:
        Version pins (for traceability; not used for logic).
    variance_threshold:
        Ratio threshold above which a movement is considered material
        (default 0.30 = 30%).

    Returns
    -------
    list[dict]
        VarianceValidation_t check result dicts.
    """
    results: List[Dict[str, Any]] = []
    evidence_refs = evidence_refs or {}

    if prior_period_accounts is None:
        return results  # Cannot run period-comparison without prior data

    prior_by_code = {a["account_code"]: a for a in prior_period_accounts}

    # Build current-period index by account class
    current_by_code = {a["account_code"]: a for a in accounts}

    for acct in accounts:
        code = acct.get("account_code", "")
        name = acct.get("account_name", "")
        account_class = (acct.get("account_class") or "").lower()
        prior = prior_by_code.get(code)

        if prior is None:
            continue  # New account in this period — no prior baseline

        current_ending = _ending_net_balance(acct)
        prior_ending = _ending_net_balance(prior)
        ratio = _variance_ratio(current_ending, prior_ending)

        if ratio is None:
            continue

        is_material = abs(ratio) >= variance_threshold
        acct_evidence = evidence_refs.get(code, [])

        if not is_material:
            continue  # Only evaluate material movements

        # Explained by supplied evidence — pass
        if acct_evidence:
            results.append({
                "check_id": f"tb.variance.{code}",
                "check_type": "variance",
                "severity": "info",
                "status": "passed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"ending": current_ending, "prior_ending": prior_ending, "ratio": ratio},
                "expected_value": {"explanation": "evidence_supplied"},
                "evidence_refs": acct_evidence,
                "user_feedback_ref": None,
                "justification_ref": None,
            })
            continue

        # Inventory/COGS mismatch check
        # Matches both English "inventory" and Chinese "库存" account names
        _is_inventory = "inventory" in name.lower() or "库存" in name
        if account_class in ("asset",) and _is_inventory:
            if ratio <= INVENTORY_DROP_THRESHOLD:
                # Check if COGS moved proportionally
                cogs_accounts = [
                    a for a in accounts
                    if (a.get("account_class") or "").lower() == "cost"
                ]
                cogs_moved = False
                for ca in cogs_accounts:
                    cp = prior_by_code.get(ca["account_code"])
                    if cp:
                        cr = _variance_ratio(_ending_net_balance(ca), _ending_net_balance(cp))
                        if cr is not None and abs(cr) >= COGS_MOVEMENT_THRESHOLD:
                            cogs_moved = True
                            break

                if not cogs_moved:
                    results.append({
                        "check_id": f"tb.variance.inventory_cogs_mismatch.{code}",
                        "check_type": "variance",
                        "severity": "high",
                        "status": "failed",
                        "affected_accounts": [{"account_code": code, "account_name": name}],
                        "observed_value": {
                            "inventory_ratio": ratio,
                            "cogs_moved": False,
                        },
                        "expected_value": {
                            "flag": "inventory_cogs_mismatch",
                            "requires_evidence": [
                                "inventory_movement_events",
                                "cost_recognition_events",
                            ],
                        },
                        "evidence_refs": [],
                        "user_feedback_ref": None,
                        "justification_ref": None,
                    })
                    continue

        # Long-aging receivable check
        aging_days = acct.get("aging_days")
        if aging_days is not None and aging_days > LONG_AGING_DAYS:
            results.append({
                "check_id": f"tb.variance.long_aging_receivable.{code}",
                "check_type": "variance",
                "severity": "warning",
                "status": "failed",
                "affected_accounts": [{"account_code": code, "account_name": name}],
                "observed_value": {"aging_days": aging_days, "balance": current_ending},
                "expected_value": {"flag": "long_outstanding_receivable"},
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })
            continue

        # Generic material variance without evidence
        results.append({
            "check_id": f"tb.variance.unexplained.{code}",
            "check_type": "variance",
            "severity": "warning",
            "status": "failed",
            "affected_accounts": [{"account_code": code, "account_name": name}],
            "observed_value": {"ending": current_ending, "prior_ending": prior_ending, "ratio": ratio},
            "expected_value": {"flag": "material_unexplained_variance", "threshold": variance_threshold},
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    return results


def run_balance_sheet_ratio_checks(
    accounts: List[Dict[str, Any]],
    pins: Dict[str, Any],
    interest_income: float = 0.0,
) -> List[Dict[str, Any]]:
    """Check balance-sheet ratio indicators used by Chinese tax authorities.

    Implements four of the nine fraud-indicator categories:

    1. 预付账款占比 — prepaid accounts / total assets > 10 %
    2. 应收高占比    — (receivables + contract assets) / total assets > 20 %
    3. 双高联动      — inventory > 15 % of assets AND receivables > 5 % simultaneously
    4. 货币资金利息率 — interest income / cash & equivalents < 2 %

    Parameters
    ----------
    accounts:
        Current-period account balance rows from TrialBalanceView_t.
    pins:
        Version pins (for traceability).
    interest_income:
        Period interest income amount.  Used for the cash-interest-rate check.
        Caller should pass the total interest income recognised in the period.

    Returns
    -------
    list[dict]
        VarianceValidation_t-compatible check result dicts.
    """
    results: List[Dict[str, Any]] = []

    def _ending(a: Dict[str, Any]) -> float:
        return _ending_net_balance(a)

    def _name_matches(a: Dict[str, Any], *keywords: str) -> bool:
        n = (a.get("account_name") or "").lower()
        c = (a.get("account_code") or "")
        return any(kw in n for kw in keywords)

    # Aggregate totals
    total_assets = sum(
        _ending(a)
        for a in accounts
        if (a.get("account_class") or "").lower() == "asset"
    )
    if total_assets <= 0:
        return results  # Cannot compute ratios without a positive asset base

    # 1. Prepaid accounts ratio — 预付账款/总资产
    prepaid_accounts = [
        a for a in accounts
        if _name_matches(a, "prepaid", "预付")
        and (a.get("account_class") or "").lower() == "asset"
    ]
    prepaid_total = sum(_ending(a) for a in prepaid_accounts)
    prepaid_ratio = prepaid_total / total_assets

    if prepaid_ratio > PREPAID_TO_ASSETS_THRESHOLD:
        results.append({
            "check_id": "tb.ratio.prepaid_to_assets",
            "check_type": "balance_sheet_ratio",
            "severity": "high",
            "status": "failed",
            "affected_accounts": [
                {"account_code": a.get("account_code"), "account_name": a.get("account_name")}
                for a in prepaid_accounts
            ],
            "observed_value": {
                "prepaid_total": prepaid_total,
                "total_assets": total_assets,
                "ratio": prepaid_ratio,
            },
            "expected_value": {
                "flag": "prepaid_to_assets_high",
                "threshold": PREPAID_TO_ASSETS_THRESHOLD,
                "requires_evidence": [
                    "prepaid_contract_refs",
                    "四流合一_four_stream_evidence",
                ],
            },
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    # 2. Receivables + contract assets ratio — (应收账款+合同资产)/总资产
    receivable_accounts = [
        a for a in accounts
        if _name_matches(a, "receivable", "应收", "contract asset", "合同资产")
        and (a.get("account_class") or "").lower() == "asset"
    ]
    receivable_total = sum(_ending(a) for a in receivable_accounts)
    receivable_ratio = receivable_total / total_assets

    if receivable_ratio > RECEIVABLE_TO_ASSETS_THRESHOLD:
        results.append({
            "check_id": "tb.ratio.receivable_to_assets",
            "check_type": "balance_sheet_ratio",
            "severity": "high",
            "status": "failed",
            "affected_accounts": [
                {"account_code": a.get("account_code"), "account_name": a.get("account_name")}
                for a in receivable_accounts
            ],
            "observed_value": {
                "receivable_total": receivable_total,
                "total_assets": total_assets,
                "ratio": receivable_ratio,
            },
            "expected_value": {
                "flag": "receivable_to_assets_high",
                "threshold": RECEIVABLE_TO_ASSETS_THRESHOLD,
                "requires_evidence": ["sales_contracts", "collection_records"],
            },
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    # 3. Dual-high linkage — 双高联动 (存货高 + 应收高同时出现)
    inventory_accounts = [
        a for a in accounts
        if _name_matches(a, "inventory", "存货", "库存")
        and (a.get("account_class") or "").lower() == "asset"
    ]
    inventory_total = sum(_ending(a) for a in inventory_accounts)
    inventory_ratio = inventory_total / total_assets

    if (
        inventory_ratio > DUAL_HIGH_INVENTORY_THRESHOLD
        and receivable_ratio > DUAL_HIGH_RECEIVABLE_THRESHOLD
    ):
        results.append({
            "check_id": "tb.ratio.dual_high_inventory_receivable",
            "check_type": "balance_sheet_ratio",
            "severity": "high",
            "status": "failed",
            "affected_accounts": [
                {"account_code": a.get("account_code"), "account_name": a.get("account_name")}
                for a in inventory_accounts + receivable_accounts
            ],
            "observed_value": {
                "inventory_ratio": inventory_ratio,
                "receivable_ratio": receivable_ratio,
                "total_assets": total_assets,
            },
            "expected_value": {
                "flag": "dual_high_inventory_receivable",
                "inventory_threshold": DUAL_HIGH_INVENTORY_THRESHOLD,
                "receivable_threshold": DUAL_HIGH_RECEIVABLE_THRESHOLD,
                "requires_evidence": [
                    "inventory_movement_events",
                    "sales_contracts",
                    "delivery_records",
                ],
            },
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    # 4. Cash & equivalents interest-rate check — 货币资金利息率
    cash_accounts = [
        a for a in accounts
        if _name_matches(a, "cash", "货币资金", "bank deposit", "银行存款")
        and (a.get("account_class") or "").lower() == "asset"
    ]
    cash_total = sum(_ending(a) for a in cash_accounts)

    if cash_total > 0 and interest_income >= 0:
        cash_interest_ratio = interest_income / cash_total
        if cash_interest_ratio < CASH_INTEREST_MIN_RATIO:
            results.append({
                "check_id": "tb.ratio.cash_interest_rate_low",
                "check_type": "balance_sheet_ratio",
                "severity": "warning",
                "status": "failed",
                "affected_accounts": [
                    {"account_code": a.get("account_code"), "account_name": a.get("account_name")}
                    for a in cash_accounts
                ],
                "observed_value": {
                    "interest_income": interest_income,
                    "cash_total": cash_total,
                    "cash_interest_ratio": cash_interest_ratio,
                },
                "expected_value": {
                    "flag": "cash_interest_rate_abnormally_low",
                    "min_threshold": CASH_INTEREST_MIN_RATIO,
                    "requires_evidence": [
                        "bank_statements",
                        "interest_income_refs",
                        "cross_account_reconciliation",
                    ],
                },
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })

    return results


def run_tax_burden_check(
    current_tax_amount: float,
    current_revenue: float,
    prior_tax_amount: float,
    prior_revenue: float,
    pins: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check tax burden rate variance between periods.

    Implements the tax-burden variance indicator (税负率变动):
    flags when the effective tax burden rate shifts by more than
    TAX_BURDEN_VARIANCE_THRESHOLD between the current and prior period.

    This check belongs to the TaxFilingPackage_t independent projection layer,
    not to FinancialStatementPackage_t.  It is a pure function and must not
    access live state.

    Parameters
    ----------
    current_tax_amount:
        Total tax paid / accrued in the current period.
    current_revenue:
        Total revenue recognised in the current period.
    prior_tax_amount:
        Total tax paid / accrued in the prior period.
    prior_revenue:
        Total revenue recognised in the prior period.
    pins:
        Version pins (for traceability).

    Returns
    -------
    list[dict]
        VarianceValidation_t-compatible check result dicts.
    """
    results: List[Dict[str, Any]] = []

    if current_revenue <= 0 or prior_revenue <= 0:
        return results  # Cannot compute burden rate without positive revenue

    current_burden = current_tax_amount / current_revenue
    prior_burden = prior_tax_amount / prior_revenue
    deviation = abs(current_burden - prior_burden)

    if deviation > TAX_BURDEN_VARIANCE_THRESHOLD:
        results.append({
            "check_id": "tb.tax.burden_rate_variance",
            "check_type": "tax_burden_variance",
            "severity": "high",
            "status": "failed",
            "affected_accounts": [],
            "observed_value": {
                "current_burden_rate": current_burden,
                "prior_burden_rate": prior_burden,
                "deviation": deviation,
            },
            "expected_value": {
                "flag": "tax_burden_rate_material_shift",
                "max_deviation": TAX_BURDEN_VARIANCE_THRESHOLD,
                "requires_evidence": [
                    "tax_rule_bundle_version",
                    "revenue_recognition_events",
                    "tax_filing_package_ref",
                ],
            },
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    return results


def run_invoice_behavior_checks(
    total_invoices: int,
    voided_invoices: int,
    prior_total_invoices: int,
    pins: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check invoice behaviour anomalies at the SourceDocumentReview_t layer.

    Implements two of the nine tax authority fraud indicators:

    - 发票作废率 > 20 %   (void rate)
    - 发票开票量环比突增 > 30 % (surge rate)

    Parameters
    ----------
    total_invoices:
        Total invoices issued in the current period.
    voided_invoices:
        Number of voided / cancelled invoices in the current period.
    prior_total_invoices:
        Total invoices issued in the prior period (for surge comparison).
    pins:
        Version pins (for traceability).

    Returns
    -------
    list[dict]
        Check result dicts compatible with SourceDocumentReview_t.
    """
    results: List[Dict[str, Any]] = []

    if total_invoices <= 0:
        return results

    void_rate = voided_invoices / total_invoices
    if void_rate > INVOICE_VOID_RATE_THRESHOLD:
        results.append({
            "check_id": "src.invoice.void_rate_high",
            "check_type": "invoice_behavior",
            "severity": "high",
            "status": "failed",
            "affected_accounts": [],
            "observed_value": {
                "total_invoices": total_invoices,
                "voided_invoices": voided_invoices,
                "void_rate": void_rate,
            },
            "expected_value": {
                "flag": "invoice_void_rate_high",
                "threshold": INVOICE_VOID_RATE_THRESHOLD,
                "requires_evidence": [
                    "invoice_cancellation_records",
                    "business_reason_refs",
                ],
            },
            "evidence_refs": [],
            "user_feedback_ref": None,
            "justification_ref": None,
        })

    if prior_total_invoices > 0:
        surge_ratio = (total_invoices - prior_total_invoices) / prior_total_invoices
        if surge_ratio > INVOICE_SURGE_THRESHOLD:
            results.append({
                "check_id": "src.invoice.volume_surge",
                "check_type": "invoice_behavior",
                "severity": "warning",
                "status": "failed",
                "affected_accounts": [],
                "observed_value": {
                    "current_total": total_invoices,
                    "prior_total": prior_total_invoices,
                    "surge_ratio": surge_ratio,
                },
                "expected_value": {
                    "flag": "invoice_volume_surge",
                    "threshold": INVOICE_SURGE_THRESHOLD,
                    "requires_evidence": [
                        "new_customer_contracts",
                        "revenue_recognition_events",
                    ],
                },
                "evidence_refs": [],
                "user_feedback_ref": None,
                "justification_ref": None,
            })

    return results
