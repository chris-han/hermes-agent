---
name: real-company-onboarding
description: Use governed Semantier core tools to inspect and advance REAL company data onboarding.
metadata:
  hermes:
    plugin: real_company_onboarding
    toolset: real_company_onboarding
    platform_compatibility: [cli, weixin, feishu, api_server, acp]
---

# Real Company Onboarding

Use the registered `real_company_onboarding` tools for REAL organization setup status, reconciliation, source preview, REA admission, COA bootstrap, lifecycle materialization, and one-shot setup. Use `governed_data_engineering.company_refresh_lakehouse` for target-company lakehouse refreshes.

Do not inspect or mutate raw SQLite tables, artifact files, prompt memory, or local workspace state as a workaround for missing setup data. Missing governed context must be reported through the tool result.

Sibling toolsets:

- Use `automate_excel` for deterministic spreadsheet/CSV mechanics before or
  during import validation: required-column checks, duplicate-key checks,
  sheet/header handling, conversion, filtering, and cleanup of explicit
  workspace artifacts.
- Use `governed_data_engineering` for non-authoritative data diagnostics during
  onboarding: table profiling, missingness/duplicate/high-cardinality signals,
  explicit quality gates, sample-size feasibility, and data-readiness summaries.
- Use `real_company_onboarding` only for governed REAL setup status and setup
  stage transitions. It delegates semantic authority and persistence to
  Semantier core.

Expected flow:

1. Use `real_company_setup_status` to determine the authoritative setup state.
2. Use `automate_excel` when an uploaded spreadsheet needs deterministic file
   validation or mechanical normalization.
3. Use `governed_data_engineering` diagnostics to explain data quality and
   readiness. Treat these diagnostics as advisory evidence, not authority.
4. Use `real_company_reconcile_promoted_imports`,
   `real_company_admit_rea_claims`, `real_company_bootstrap_coa`,
   `real_company_materialize_lifecycle`, and `real_company_run_setup` to
   advance governed REAL setup.
5. Use `governed_data_engineering.company_refresh_lakehouse` when a target
   company lakehouse refresh is needed outside the REAL onboarding setup flow.

Primary read path:

- Call `real_company_setup_status` before choosing a setup action.
- Treat `setup_status`, `blocking_reasons`, and `next_actions` as authoritative.
- Do not derive readiness from legacy `authority_state`.

Write paths:

- Use an explicit `idempotency_key`.
- Let core setup run locking serialize competing writes.
- Stop on typed blockers returned by core services.
