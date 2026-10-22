from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="real_company_setup_status",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.STATUS_SCHEMA,
        handler=tools.real_company_setup_status,
    )
    ctx.register_tool(
        name="real_company_reconcile_promoted_imports",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.WRITE_SCHEMA,
        handler=tools.real_company_reconcile_promoted_imports,
    )
    ctx.register_tool(
        name="real_company_legacy_claims_report",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.READ_ONLY_DATASET_SCHEMA,
        handler=tools.real_company_legacy_claims_report,
    )
    ctx.register_tool(
        name="real_company_preview_source_rows",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.READ_ONLY_DATASET_SCHEMA,
        handler=tools.real_company_preview_source_rows,
    )
    ctx.register_tool(
        name="real_company_admit_rea_claims",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.WRITE_SCHEMA,
        handler=tools.real_company_admit_rea_claims,
    )
    ctx.register_tool(
        name="real_company_bootstrap_coa",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.WRITE_SCHEMA,
        handler=tools.real_company_bootstrap_coa,
    )
    ctx.register_tool(
        name="real_company_materialize_lifecycle",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.WRITE_SCHEMA,
        handler=tools.real_company_materialize_lifecycle,
    )
    ctx.register_tool(
        name="real_company_run_setup",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.WRITE_SCHEMA,
        handler=tools.real_company_run_setup,
    )
