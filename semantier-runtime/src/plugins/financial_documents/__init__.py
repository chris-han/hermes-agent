from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="extract_financial_document",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.EXTRACT_FINANCIAL_DOCUMENT_SCHEMA,
        handler=tools.extract_financial_document,
        emoji="🧾",
    )
    ctx.register_tool(
        name="evaluate_expense_reimbursement",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.EVALUATE_EXPENSE_REIMBURSEMENT_SCHEMA,
        handler=tools.evaluate_expense_reimbursement,
        emoji="🧾",
    )
