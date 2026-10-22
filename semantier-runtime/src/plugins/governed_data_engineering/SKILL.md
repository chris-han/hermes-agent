---
name: governed-data-engineering
description: Deterministic data profiling, quality gates, sample-size feasibility checks, data-feasibility summaries, and governed lakehouse refreshes for Semantier runtime data work.
metadata:
  semantier:
    route: procedural_only
---

# Governed Data Engineering

Use this plugin for deterministic data engineering diagnostics: profiling a
declared table artifact, running explicit quality rules, checking rough sample
size sufficiency, producing a data feasibility summary, and refreshing the
governed lakehouse for an explicit company target.

Registered tools:

- `governed_profile_table`
- `governed_quality_gate`
- `governed_sample_size_check`
- `governed_data_feasibility`
- `company_refresh_lakehouse`

This toolset is a data-engineering diagnostic layer, not an authority source.
It may inspect explicit user/workspace artifacts or future governed dataset
adapters, but it must not infer identity, organization membership, active
authority context, or semantic truth from file contents, prompt memory, or user
self-claims. When the question depends on organization authority or governed
analytics, use Semantier governed context and `governed_analytics` first.

Use `automate_excel` for spreadsheet transformation. Use
`governed_data_engineering` for profiling, quality gates, feasibility checks,
dataset-card preparation, and `company_refresh_lakehouse` when the target
company lakehouse must be refreshed. Use `governed_analytics` for scoped SQL
analytics over authorized `org_*` views.

Do not use terminal, generated Python, generated shell scripts, ad hoc pandas
snippets, unmanaged temporary files, or direct script calls as substitutes for
the registered tool surface. If the tool surface is not loaded, stop and report
that the `governed_data_engineering` plugin must be installed or enabled.
