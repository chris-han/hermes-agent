"""
Full-Cycle Accounting Runtime Stores — v8.1

Provides typed insert/query helpers for:
  SourceDocumentReview_t
  JournalVoucherProjection_t
  CashJournalView_t
  BankJournalView_t
  SubsidiaryLedgerView_t
  GeneralLedgerView_t
  FinancialStatementPackage_t
  TaxFilingPackage_t
  AccountingArchivePackage_t

Invariants (§Full-Cycle Accounting Runtime Contracts):
  - All tables are append-only. No UPDATE or DELETE on business records.
  - content_hash required on every insert.
  - SourceDocumentReview_t evidence_status does not constitute REA fact admission.
  - JournalVoucherProjection_t lines_json must not contain 'rea_account_code' at
    the REA event level (account codes in projection lines are fine).
  - FinancialStatementPackage_t requires validation_results_json.
  - AccountingArchivePackage_t requires archive_hash.
"""
from storage import hosted_postgres as hosted

import json
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_hash
from eos.db import get_conn, now_iso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_or_none(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Deserialize JSON fields for caller convenience
    for key in list(d.keys()):
        if key.endswith("_json") and d[key] is not None:
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _serialize_json_fields(record: dict, json_fields: list[str]) -> dict:
    out = dict(record)
    for field in json_fields:
        if field in out and not isinstance(out[field], str):
            out[field] = json.dumps(out[field], ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Source Document Review
# ---------------------------------------------------------------------------

SOURCE_DOCUMENT_REVIEW_JSON_FIELDS = ["missing_fields_json", "extraction_result_json"]

VALID_EVIDENCE_STATUSES = {"PENDING", "ELIGIBLE_FOR_REA", "REJECTED"}
VALID_CHECK_RESULTS = {"PASS", "FAIL", "PENDING"}


def insert_source_document_review(record: dict) -> dict:
    """Insert a SourceDocumentReview_t record.

    Evidence status ELIGIBLE_FOR_REA does NOT constitute REA fact admission;
    it only marks the document as eligible for a separate REA formation step.
    """
    if record.get("evidence_status") not in VALID_EVIDENCE_STATUSES:
        raise ValueError(
            f"evidence_status must be one of {VALID_EVIDENCE_STATUSES}, "
            f"got: {record.get('evidence_status')!r}"
        )
    for check in ("authenticity_check", "legality_check", "completeness_check", "arithmetic_check"):
        if record.get(check) not in VALID_CHECK_RESULTS:
            raise ValueError(
                f"{check} must be one of {VALID_CHECK_RESULTS}, got: {record.get(check)!r}"
            )

    row = dict(record)
    row.setdefault("review_id", f"sdr_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, SOURCE_DOCUMENT_REVIEW_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO source_document_reviews ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return _row_or_none(None) or record | {"review_id": row["review_id"], "content_hash": row["content_hash"]}


def get_source_document_review(review_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM source_document_reviews WHERE review_id = ?", 'SELECT * FROM source_document_reviews WHERE review_id = %s'), (review_id,)
        ).fetchone()
    return _row_or_none(row)


def list_source_document_reviews(org_id: str, period_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM source_document_reviews WHERE org_id = ? AND period_id = ? ORDER BY created_at ASC", 'SELECT * FROM source_document_reviews WHERE org_id = %s AND period_id = %s ORDER BY created_at ASC'),
            (org_id, period_id),
        ).fetchall()
    return [_row_or_none(r) for r in rows]


# ---------------------------------------------------------------------------
# Journal Voucher Projection
# ---------------------------------------------------------------------------

JVP_JSON_FIELDS = ["lines_json"]

VALID_VOUCHER_TYPES = {"purchase", "sale", "payment", "receipt", "adjustment", "closing", "reversal"}
VALID_PROJECTION_STATUSES = {"projection_candidate", "projection_validated", "projection_exception"}


def insert_journal_voucher_projection(record: dict) -> dict:
    """Insert a JournalVoucherProjection_t record.

    lines_json is the projection output (may contain account_code per ledger line).
    REAEvent_t must never store account_code — enforcement is at the REA formation
    layer, not here. This store validates lines_json structure only.
    """
    if record.get("voucher_type") not in VALID_VOUCHER_TYPES:
        raise ValueError(
            f"voucher_type must be one of {VALID_VOUCHER_TYPES}, got: {record.get('voucher_type')!r}"
        )
    if record.get("projection_status") not in VALID_PROJECTION_STATUSES:
        raise ValueError(
            f"projection_status must be one of {VALID_PROJECTION_STATUSES}, "
            f"got: {record.get('projection_status')!r}"
        )

    row = dict(record)
    row.setdefault("jvp_id", f"jvp_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, JVP_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO journal_voucher_projections ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"jvp_id": row["jvp_id"], "content_hash": row["content_hash"]}


def get_journal_voucher_projection(jvp_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM journal_voucher_projections WHERE jvp_id = ?", 'SELECT * FROM journal_voucher_projections WHERE jvp_id = %s'), (jvp_id,)
        ).fetchone()
    return _row_or_none(row)


def list_journal_voucher_projections(org_id: str, period_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM journal_voucher_projections WHERE org_id = ? AND period_id = ? ORDER BY created_at ASC", 'SELECT * FROM journal_voucher_projections WHERE org_id = %s AND period_id = %s ORDER BY created_at ASC'),
            (org_id, period_id),
        ).fetchall()
    return [_row_or_none(r) for r in rows]


# ---------------------------------------------------------------------------
# Ledger View Stores (shared pattern)
# ---------------------------------------------------------------------------

LEDGER_JSON_FIELDS = ["source_jvp_ids_json", "entries_json"]


def _insert_ledger_view(table: str, id_field: str, prefix: str, record: dict) -> dict:
    row = dict(record)
    row.setdefault(id_field, f"{prefix}_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, LEDGER_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {id_field: row[id_field], "content_hash": row["content_hash"]}


def _get_ledger_view(table: str, id_field: str, view_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query(f"SELECT * FROM {table} WHERE {id_field} = ?", f'SELECT * FROM {table} WHERE {id_field} = %s'), (view_id,)
        ).fetchone()
    return _row_or_none(row)


# Cash Journal View

def insert_cash_journal_view(record: dict) -> dict:
    return _insert_ledger_view("cash_journal_views", "view_id", "cjv", record)


def get_cash_journal_view(view_id: str) -> dict | None:
    return _get_ledger_view("cash_journal_views", "view_id", view_id)


# Bank Journal View

def insert_bank_journal_view(record: dict) -> dict:
    return _insert_ledger_view("bank_journal_views", "view_id", "bjv", record)


def get_bank_journal_view(view_id: str) -> dict | None:
    return _get_ledger_view("bank_journal_views", "view_id", view_id)


# Subsidiary Ledger View

def insert_subsidiary_ledger_view(record: dict) -> dict:
    row = dict(record)
    row.setdefault("view_id", f"slv_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    json_fields = ["source_jvp_ids_json", "entries_json"]
    row = _serialize_json_fields(row, json_fields)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO subsidiary_ledger_views ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"view_id": row["view_id"], "content_hash": row["content_hash"]}


def get_subsidiary_ledger_view(view_id: str) -> dict | None:
    return _get_ledger_view("subsidiary_ledger_views", "view_id", view_id)


# General Ledger View

def insert_general_ledger_view(record: dict) -> dict:
    row = dict(record)
    row.setdefault("view_id", f"glv_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    json_fields = ["source_jvp_ids_json", "balances_json"]
    row = _serialize_json_fields(row, json_fields)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO general_ledger_views ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"view_id": row["view_id"], "content_hash": row["content_hash"]}


def get_general_ledger_view(view_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM general_ledger_views WHERE view_id = ?", 'SELECT * FROM general_ledger_views WHERE view_id = %s'), (view_id,)
        ).fetchone()
    return _row_or_none(row)


# ---------------------------------------------------------------------------
# Financial Statement Package
# ---------------------------------------------------------------------------

FSP_JSON_FIELDS = [
    "balance_sheet_json",
    "income_statement_json",
    "cash_flow_statement_json",
    "validation_results_json",
]


def insert_financial_statement_package(record: dict) -> dict:
    """Insert a FinancialStatementPackage_t.

    Invariant: validation_results_json must be present.
    Invariant: must be called only after period is closed (enforced by caller
               via MonthlyAccountingWorkflow_t.close_state == 'closed').
    """
    if not record.get("validation_results_json") and record.get("validation_results_json") != {}:
        raise ValueError("validation_results_json is required for FinancialStatementPackage_t")
    if not record.get("trial_balance_view_id"):
        raise ValueError("trial_balance_view_id is required for FinancialStatementPackage_t")

    row = dict(record)
    row.setdefault("package_id", f"fsp_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, FSP_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO financial_statement_packages ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"package_id": row["package_id"], "content_hash": row["content_hash"]}


def get_financial_statement_package(package_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM financial_statement_packages WHERE package_id = ?", 'SELECT * FROM financial_statement_packages WHERE package_id = %s'), (package_id,)
        ).fetchone()
    return _row_or_none(row)


# ---------------------------------------------------------------------------
# Tax Filing Package
# ---------------------------------------------------------------------------

TAX_FILING_JSON_FIELDS = ["source_refs_json", "filing_form_json", "validation_result_json"]

VALID_APPROVAL_STATES = {"draft", "reviewed", "approved", "submitted", "rejected"}


def insert_tax_filing_package(record: dict) -> dict:
    """Insert a TaxFilingPackage_t.

    Invariant: uses pinned tax_rule_bundle_version; independent of financial statements.
    """
    if not record.get("tax_rule_bundle_version"):
        raise ValueError("tax_rule_bundle_version is required for TaxFilingPackage_t")

    approval_state = record.get("approval_state", "draft")
    if approval_state not in VALID_APPROVAL_STATES:
        raise ValueError(
            f"approval_state must be one of {VALID_APPROVAL_STATES}, got: {approval_state!r}"
        )

    row = dict(record)
    row.setdefault("filing_id", f"tfp_{uuid4().hex}")
    row.setdefault("approval_state", "draft")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, TAX_FILING_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO tax_filing_packages ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"filing_id": row["filing_id"], "content_hash": row["content_hash"]}


def get_tax_filing_package(filing_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM tax_filing_packages WHERE filing_id = ?", 'SELECT * FROM tax_filing_packages WHERE filing_id = %s'), (filing_id,)
        ).fetchone()
    return _row_or_none(row)


# ---------------------------------------------------------------------------
# Accounting Archive Package
# ---------------------------------------------------------------------------

ARCHIVE_JSON_FIELDS = [
    "source_document_review_ids_json",
    "rea_event_ids_json",
    "jvp_ids_json",
    "ledger_view_ids_json",
    "validation_result_ids_json",
    "governance_decision_ids_json",
    "financial_statement_package_ids_json",
    "tax_filing_package_ids_json",
    "external_audit_export_refs_json",
]


def insert_accounting_archive_package(record: dict) -> dict:
    """Insert an AccountingArchivePackage_t.

    Invariant: archive_hash must be present (supports offline verification).
    """
    if not record.get("archive_hash"):
        raise ValueError("archive_hash is required for AccountingArchivePackage_t")

    row = dict(record)
    row.setdefault("archive_id", f"aap_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, ARCHIVE_JSON_FIELDS)

    keys = list(row.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO accounting_archive_packages ({columns}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    return record | {"archive_id": row["archive_id"], "content_hash": row["content_hash"]}


def get_accounting_archive_package(archive_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM accounting_archive_packages WHERE archive_id = ?", 'SELECT * FROM accounting_archive_packages WHERE archive_id = %s'), (archive_id,)
        ).fetchone()
    return _row_or_none(row)
