from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="compile_sensitive_lexicon",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.COMPILE_SENSITIVE_LEXICON_SCHEMA,
        handler=tools.compile_sensitive_lexicon,
    )
    ctx.register_tool(
        name="review_sensitive_lexicon_candidates",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.REVIEW_SENSITIVE_LEXICON_CANDIDATES_SCHEMA,
        handler=tools.review_sensitive_lexicon_candidates,
    )
    ctx.register_tool(
        name="label_tender_document",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.LABEL_TENDER_DOCUMENT_SCHEMA,
        handler=tools.label_tender_document,
    )
    ctx.register_tool(
        name="generate_sensitive_labeling_report",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GENERATE_SENSITIVE_LABELING_REPORT_SCHEMA,
        handler=tools.generate_sensitive_labeling_report,
    )
    ctx.register_tool(
        name="generate_labeled_docx",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GENERATE_LABELED_DOCX_SCHEMA,
        handler=tools.generate_labeled_docx,
    )
    ctx.register_tool(
        name="verify_sensitive_labeling_replay",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.VERIFY_SENSITIVE_LABELING_REPLAY_SCHEMA,
        handler=tools.verify_sensitive_labeling_replay,
    )
