from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .. import schemas

AMOUNT_RE = re.compile(r"(?:[$¥￥]|CNY|RMB|USD|EUR)?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})|-?\d+(?:\.\d{2})?)", re.I)
DATE_RE = re.compile(r"(\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)")


def canonical_text(canonical: dict[str, Any]) -> str:
    pieces: list[str] = []
    for block in canonical.get("blocks") or []:
        text = str(block.get("text") or "").strip()
        if text:
            pieces.append(text)
    for table in canonical.get("tables") or []:
        for row in table.get("rows") or []:
            for cell in row.get("cells") or []:
                text = str(cell.get("text") or "").strip()
                if text:
                    pieces.append(text)
    return "\n".join(pieces)


def first_evidence(canonical: dict[str, Any], needle: str) -> list[dict[str, Any]]:
    if not needle:
        return []
    for block in canonical.get("blocks") or []:
        if needle in str(block.get("text") or ""):
            return [{"field": "", "evidence_ids": list(block.get("evidence_ids") or []), "confidence": 0.72}]
    for table in canonical.get("tables") or []:
        for row in table.get("rows") or []:
            for cell in row.get("cells") or []:
                if needle in str(cell.get("text") or ""):
                    return [{"field": "", "evidence_ids": list(cell.get("evidence_ids") or []), "confidence": 0.7}]
    return []


def _decimal(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _find_labeled_amount(text: str, labels: list[str]) -> Decimal | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for label in labels:
        pattern = re.compile(rf"^{re.escape(label)}\s*[:：]?\s*(?:[$¥￥]|CNY|RMB|USD)?\s*(-?\d[\d,]*(?:\.\d{{2}})?)", re.I)
        for line in lines:
            match = pattern.search(line)
            if match:
                return _decimal(match.group(1))
    return None


def _find_labeled_value(text: str, labels: list[str]) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for label in labels:
        pattern = re.compile(rf"^{re.escape(label)}\s*[:：#]?\s*(.+?)\s*$", re.I)
        for line in lines:
            match = pattern.search(line)
            if match:
                return match.group(1).strip()
    return None


def _find_invoice_date(text: str) -> str | None:
    match = re.search(r"(?:Invoice Date|Issue Date|开票日期|日期)\s*[:：]?\s*" + DATE_RE.pattern, text, re.I)
    if match:
        return match.group(1).replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
    match = DATE_RE.search(text)
    return match.group(1).replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-") if match else None


def _currency(text: str, preferred: str | None) -> str:
    if preferred:
        return preferred
    if "¥" in text or "￥" in text or "CNY" in text.upper() or "RMB" in text.upper():
        return "CNY"
    if "$" in text or "USD" in text.upper():
        return "USD"
    if "EUR" in text.upper():
        return "EUR"
    return "CNY" if re.search(r"[\u4e00-\u9fff]", text) else "USD"


def extract_generic(canonical: dict[str, Any], *, preferred_currency: str | None = None, adapter_name: str = "generic_invoice") -> dict[str, Any]:
    text = canonical_text(canonical)
    invoice_number = _find_labeled_value(text, ["Invoice Number", "Invoice No", "Invoice #", "发票号码", "发票号"]) or ""
    invoice_date = _find_invoice_date(text) or ""
    subtotal = _find_labeled_amount(text, ["Subtotal", "Amount excluding tax", "不含税金额", "金额"])
    tax = _find_labeled_amount(text, ["Tax", "VAT", "税额"])
    total = _find_labeled_amount(text, ["Total", "Amount Due", "价税合计", "合计", "总计"])
    supplier = _find_labeled_value(text, ["Supplier", "Vendor", "Seller", "销售方", "销方名称", "销售方名称"]) or ""
    customer = _find_labeled_value(text, ["Customer", "Bill To", "Buyer", "购买方", "购方名称", "购买方名称"]) or ""
    unresolved = []
    for field, value in {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "supplier.name": supplier,
        "total": total,
    }.items():
        if value in (None, ""):
            unresolved.append(field)
    findings = []
    if subtotal is not None and tax is not None and total is not None and subtotal + tax != total:
        findings.append(
            {
                "code": "ARITHMETIC_RECONCILIATION_FAILED",
                "severity": "review",
                "message": "subtotal plus tax does not equal total",
                "expected_total": str(subtotal + tax),
                "reported_total": str(total),
            }
        )
    evidence = []
    for field, value in {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "supplier.name": supplier,
        "customer.name": customer,
        "total": str(total) if total is not None else "",
    }.items():
        for item in first_evidence(canonical, str(value)):
            item["field"] = field
            evidence.append(item)
    fingerprint_base = "|".join([supplier, invoice_number, invoice_date, str(total or "")])
    invoice = {
        "schema_version": schemas.INVOICE_SCHEMA_VERSION,
        "document_type": "vendor_invoice",
        "supplier": {"name": supplier, "taxpayer_id": _find_labeled_value(text, ["Seller Tax ID", "销售方纳税人识别号", "纳税人识别号"])},
        "customer": {"name": customer, "taxpayer_id": _find_labeled_value(text, ["Buyer Tax ID", "购买方纳税人识别号"])} if customer else None,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": None,
        "billing_period": None,
        "currency": _currency(text, preferred_currency),
        "subtotal": str(subtotal) if subtotal is not None else None,
        "tax": str(tax) if tax is not None else None,
        "total": str(total) if total is not None else None,
        "amount_due": str(_find_labeled_amount(text, ["Amount Due", "Balance Due"])) if _find_labeled_amount(text, ["Amount Due", "Balance Due"]) is not None else None,
        "sections": [],
        "line_items": [],
        "evidence": evidence,
        "parser": {"adapter_name": adapter_name, "adapter_version": schemas.ADAPTER_VERSION},
        "validation": {
            "ruleset_version": schemas.VALIDATION_RULESET_VERSION,
            "findings": findings,
            "duplicate_fingerprint": "sha256:" + hashlib.sha256(fingerprint_base.encode("utf-8")).hexdigest(),
        },
    }
    status = "review_required" if unresolved or findings else "admissible"
    return {"status": status, "document": invoice, "findings": findings, "unresolved_fields": unresolved}
