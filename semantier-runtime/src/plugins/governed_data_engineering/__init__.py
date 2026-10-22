from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="governed_profile_table",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.PROFILE_TABLE_SCHEMA,
        handler=tools.governed_profile_table,
    )
    ctx.register_tool(
        name="governed_quality_gate",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.QUALITY_GATE_SCHEMA,
        handler=tools.governed_quality_gate,
    )
    ctx.register_tool(
        name="governed_sample_size_check",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.SAMPLE_SIZE_CHECK_SCHEMA,
        handler=tools.governed_sample_size_check,
    )
    ctx.register_tool(
        name="governed_data_feasibility",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.DATA_FEASIBILITY_SCHEMA,
        handler=tools.governed_data_feasibility,
    )
    ctx.register_tool(
        name="company_refresh_lakehouse",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.COMPANY_LAKEHOUSE_REFRESH_SCHEMA,
        handler=tools.company_refresh_lakehouse,
    )
