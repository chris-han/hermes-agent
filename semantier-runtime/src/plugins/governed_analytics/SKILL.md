---
name: governed-analytics
description: "Use the governed_analytics plugin tools for governed organization analytics through authorization-enforced DuckDB surfaces."
version: 1.0.0
author: Semantier Runtime
license: MIT
metadata:
  hermes:
    plugin: governed_analytics
    toolset: governed_analytics
    platform_compatibility: [cli, weixin, feishu, api_server, acp]
    tags: [business, analytics, authorization, governance, organization, dataset, duckdb]
    category: semantier
---

# Business Analytics Plugin

Use the `governed_analytics` toolset for user-facing analytics over governed
organization data. Use `governed_schema` first for unfamiliar or broad
analytics questions, then `governed_query` for read-only SQL. Use
`duckdb_python` when a DuckDB/Python-backed execution surface is needed. These
tools resolve the active workspace user and organization from governed
authorization records, verify active membership and dataset binding, and expose
only organization-scoped views.

When the user asks to generate, produce, create, or prepare a business report
from governed analytics data, answer from governed query results and persist
the finished report with `governed_report_export`. Treat Chinese requests such
as “生成报税报告” as report-generation requests that should produce a Markdown
session artifact, not only inline chat text. After export, briefly summarize
the report and mention that the report was saved.

Use the sibling `governed_data_engineering` toolset when the task is about data
engineering diagnostics rather than business analytics: table profiling,
missingness/duplicate/high-cardinality checks, explicit quality gates,
sample-size feasibility, data-readiness four-question summaries, dataset-card
preparation, or future drift/access-level checks. Do not implement those
diagnostics by ad hoc `governed_query` SQL unless the user specifically asks
for a SQL aggregate over scoped organization data.

`org_rea_claims` contains governed business-event facts, not only accounting
records. It can include sales, purchase, collection, inventory, fulfillment,
tax, risk, and accounting projection inputs depending on the active dataset.
For sales questions, first inspect available flattened columns such as
`event_type`, `counterparty`, `amount_10k`, `occurred_at`, and `claim_json` in
`org_rea_claims`. Optional business descriptors such as localized descriptions,
tags, project ids, or category labels may live inside `claim_json`; confirm their
JSON paths from sample rows before extracting them. Do not say product or sales
data is unavailable just because it is represented as REA claims.

Available scoped views:

- `org_rea_claims`
- `org_journal_voucher_projections`
- `org_general_ledger_views`
- `org_general_ledger_balance_lines`
- `org_financial_statement_packages`
- `org_tax_filing_packages`
- `org_accounting_archive_packages`

Do not use raw external `execute_sql` MCP tools, terminal SQL, general code,
file/search tools, raw `eos.*` tables, or unscoped lakehouse files for
user-facing governed data. Those paths do not enforce active organization,
membership, capability scope, or dataset binding.

If a business analytics tool returns an authorization denial, report the denial
and do not answer from another dataset or from local files.

## Sibling Toolset Routing

- Workflow preflight order for analytics SQL:
  1. Call `governed_schema` first to discover ASCII machine identifiers.
  2. Call `governed_data_engineering.governed_profile_table` when schema
     statistics or column quality evidence is needed before querying.
  3. Write `governed_query`/`duckdb_python` SQL using ASCII identifiers only.

- Use `governed_schema` to inspect authorized `org_*` view structure.
- Use `governed_query` or `duckdb_python` to answer numeric business questions
  over authorized `org_*` views.
- Use `governed_data_engineering.governed_profile_table` for deterministic
  profiling of an explicit table artifact.
- Use `governed_data_engineering.governed_quality_gate` for explicit
  PASS/FAIL data quality rules.
- Use `governed_data_engineering.governed_sample_size_check` and
  `governed_data_engineering.governed_data_feasibility` before research,
  modeling, or dataset-readiness decisions.

File-artifact diagnostics from `governed_data_engineering` are not authority
for organization identity or semantic truth. Promotion from a diagnostic file
artifact into governed organization data remains a Semantier core import and
admission workflow.

## Tool Arguments

```python
governed_schema(view: str | None = None, include_sample_rows: bool = True, sample_limit: int = 1, reload: bool = False) -> dict
governed_query(sql: str, reload: bool = False) -> dict
duckdb_python(sql: str, reload: bool = False) -> dict
governed_report_export(content: str, session_id: str | None = None, run_id: str | None = None, filename_stem: str | None = None) -> dict
```

- `view` (optional): One scoped `org_*` view to inspect. Omit to inspect all governed analytics views.
- `include_sample_rows` (optional): Include sample rows with the schema result.
- `sample_limit` (optional): Maximum sample rows per view, clamped to 0..5.
- `sql` (required): The SQL query to execute against scoped views. SQL identifiers and separators must be ASCII-only.
- `reload` (optional): Force rebuild of scoped views from lakehouse sources
- `content` (required for `governed_report_export`): Complete Markdown report
  body to save under the active session artifact root and attach to the
  Inspector-visible run metadata.
- After `governed_report_export`, show one saved-location line only: make the
  visible file path `/sessions/<session_id>/artifacts/<relative_path>` a
  markdown link to `#artifact=<urlencoded-relative_path>`. Do not print a
  separate Inspector raw-link line.
- `filename_stem` (optional): ASCII filename stem for the saved report. Use a
  concise stable stem such as `tax_report_2026q4` for tax report requests.

`duckdb_python` is a Python API wrapper around the same governed DuckDB
connection. It is not arbitrary Python execution and must not be used to bypass
the governed SQL validator.

## Pitfalls

- **SQL required**: The `sql` argument is mandatory. Calling a query tool
  without it will fail with `SQL_REQUIRED`.
- **Schema discovery**: When working with an unfamiliar view, call
  `governed_schema` before writing aggregate queries. It returns column names,
  display metadata, and optional sample rows without requiring schema-guessing
  SQL.
- **Ledger balances**: For cost/accounting questions over `balances_json`, use
  `org_general_ledger_balance_lines` instead of exporting JSON to Python. It
  flattens each account entry into SQL columns such as `account_code`,
  `debit`, `credit`, `net`, `debit_10k`, `credit_10k`, and `net_10k`.
- **DuckDB JSON extraction**: Use DuckDB JSON functions for JSON columns. Do
  not use Snowflake-style `claim_json:field::TYPE` or PostgreSQL-style
  `claim_json->>'field'` syntax. Use `json_extract_string(claim_json,
  '$.field')` for strings and `TRY_CAST(json_extract(claim_json, '$.field') AS
  DOUBLE)` or `TRY_CAST(json_extract(claim_json, '$.field') AS BOOLEAN)` for
  typed values.
- **Prefer flattened governed columns**: Before extracting from JSON columns,
  check whether `governed_schema` already exposes a flattened column such as
  `event_type`, `amount_10k`, `occurred_at`, `blocked`, `period_id`,
  `account_code`, `debit_10k`, `credit_10k`, or `net_10k`. Use those columns
  directly when available.
- **Machine identifiers**: Use ASCII column identifiers such as
  `amount_10k`, `debit_10k`, `credit_10k`, and `net_10k`. Chinese labels and
  units belong in display metadata, not SQL identifiers.
- **SQL punctuation**: SQL separators must use ASCII punctuation. Do not use
  Chinese full-width commas (`，`) between column names. `debit_10k, credit_10k,
  net_10k` is valid; `"debit_10k，credit_10k，net_10k"` is one invalid quoted
  identifier.
- **Identifier quoting**: If quoting identifiers, quote each column separately:
  `"debit_10k", "credit_10k", "net_10k"`.
- **Rewrite on violation**: If a query fails with `QUERY_SQL_IDENTIFIER_VIOLATION`
  or `QUERY_SQL_SYNTAX_VIOLATION`, immediately rewrite the SQL using ASCII
  identifiers and ASCII separators before retrying.
- **Localized labels mapping**: When user wording uses localized labels (for
  example Chinese names), call `governed_schema` first and map requested fields
  from `column_metadata` or `display_columns` to ASCII machine identifiers.

## Query Pattern

```sql
SELECT event_type, COUNT(*) AS n, SUM(amount_10k) AS amount_10k
FROM org_rea_claims
WHERE blocked = FALSE
GROUP BY event_type
ORDER BY amount_10k DESC;
```

```sql
SELECT
  event_type,
  substr(occurred_at, 1, 4) AS year,
  COUNT(*) AS n,
  SUM(amount_10k) AS amount_10k
FROM org_rea_claims
WHERE blocked = FALSE
  AND event_type IN ('invoice_issued', 'contract_signed', 'cash_collected', 'deposit_received')
GROUP BY event_type, substr(occurred_at, 1, 4)
ORDER BY year, amount_10k DESC;
```

```sql
SELECT period_id, projection_status, COUNT(*) AS n
FROM org_journal_voucher_projections
GROUP BY period_id, projection_status
ORDER BY period_id, projection_status;
```

```sql
SELECT
  period_id,
  account_code,
  ROUND(SUM(debit_10k), 2) AS debit_10k,
  ROUND(SUM(credit_10k), 2) AS credit_10k,
  ROUND(SUM(net_10k), 2) AS net_10k
FROM org_general_ledger_balance_lines
WHERE period_id = '2026-04'
  AND account_code LIKE '5%'
GROUP BY period_id, account_code
ORDER BY account_code;
```

```sql
SELECT
  json_extract_string(claim_json, '$.project_id') AS project_id,
  COUNT(*) AS n,
  ROUND(SUM(amount_10k), 2) AS amount_10k
FROM org_rea_claims
WHERE blocked = FALSE
GROUP BY project_id
ORDER BY amount_10k DESC;
```

```sql
SELECT
  period_id,
  TRY_CAST(json_extract(income_statement_json, '$.revenue') AS DOUBLE) / 10000.0 AS revenue_10k,
  TRY_CAST(json_extract(income_statement_json, '$.cost_of_revenue') AS DOUBLE) / 10000.0 AS cost_of_revenue_10k
FROM org_financial_statement_packages
ORDER BY period_id;
```
